"""FFT-accelerated GMRES solver for Foldy-Lax voxelized sphere.

Replaces the O(N^3) dense solver in sphere_scattering.py with an
O(N_iter * N log N) iterative solver using FFT convolution.

The sub-cells sit on a regular 3D grid, making the propagator matrix
block-Toeplitz.  The matvec (I - P*T)*w is computed via 3D FFT
circular convolution, and GMRES solves the system iteratively.

Algorithm (mirrors FFTLaxFoldy.wl):
    1. Map sphere sub-cells to (i0, i1, i2) grid indices
    2. Build kernel = -P(r)*T_loc on (2n-1)^3 circular-embedded grid
    3. FFT each of the 81 kernel components
    4. Matvec via: pack -> FFT -> pointwise multiply -> IFFT -> unpack
    5. GMRES solve with incident field as initial guess
    6. Extract composite T-matrix from exciting field
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

if TYPE_CHECKING:
    from numpy.typing import NDArray

from .cell_averaged_pair import averaged_pair_block_9x9
from .effective_contrasts import (
    MaterialContrast,
    ReferenceMedium,
    compute_cube_tmatrix,
)
from .resonance_tmatrix import (
    _build_incident_field_coupled,
    _build_incident_plane_wave_basis,
    _propagator_blocks_9x9,
    _sub_cell_tmatrix_9x9,
)
from .sphere_scattering import SphereDecompositionResult


def _build_grid_index_map(
    radius: float,
    n_sub: int,
    inside: Callable[[NDArray[np.floating]], bool] | None = None,
) -> tuple[NDArray[np.intp], NDArray[np.floating], float]:
    """Build grid index mapping for sphere sub-cells.

    Creates an n_sub x n_sub x n_sub grid, filters to cells inside the
    sphere, and records each cell's (i0, i1, i2) grid index.

    Args:
        radius: Sphere radius (m).
        n_sub: Number of sub-cells per edge of bounding cube.
        inside: Which grid cells belong to the scatterer, decided from the cell centre.  ``None`` (the
            default) keeps the cells whose centre lies inside the sphere of ``radius``.  Any other shape
            inside the bounding cube ``[-radius, radius]^3`` can be given, e.g. a fixed staircase refined
            into sub-voxels, so that the same shape is solved at several resolutions.

    Returns:
        (grid_idx, centres, a_sub) where:
            grid_idx: shape (N, 3), integer grid indices for each cell
            centres: shape (N, 3), cell centre coordinates
            a_sub: sub-cell half-width
    """
    a_sub = radius / n_sub
    dd = 2.0 * a_sub  # sub-cell side length

    # Grid coordinates (matching Mathematica: halfG = (-n/2 + 0.5 + i) * dd)
    coords_1d = np.array([(-n_sub / 2.0 + 0.5 + i) * dd for i in range(n_sub)])

    # Build full grid and filter to sphere
    grid_indices = []
    centres_list = []
    for i0 in range(n_sub):
        for i1 in range(n_sub):
            for i2 in range(n_sub):
                pos = np.array([coords_1d[i0], coords_1d[i1], coords_1d[i2]])
                keep = np.linalg.norm(pos) < radius if inside is None else inside(pos)
                if keep:
                    grid_indices.append([i0, i1, i2])
                    centres_list.append(pos)

    grid_idx = np.array(grid_indices, dtype=np.intp)
    centres = np.array(centres_list, dtype=float)
    return grid_idx, centres, a_sub


def _build_fft_kernel(
    n_sub: int,
    a_sub: float,
    T_loc: NDArray[np.complexfloating],
    omega: float,
    ref: ReferenceMedium,
    *,
    cell_average: bool = True,
    n_gauss: int | None = None,
) -> NDArray[np.complexfloating]:
    """Build FFT kernel for the propagator convolution.

    For all separations (d0, d1, d2) in [-(n-1), +(n-1)]^3, computes
    -P(r)*T_loc (9x9) and stores on a (2n-1)^3 grid with circular
    embedding.  Then FFTs each of the 81 components.

    This is the natural home for the receiver-cell average: the kernel is
    built once per DISTINCT SEPARATION rather than once per pair, so the few
    near-contact separations that need a high-order quadrature are evaluated a
    handful of times whatever the cell count.

    Args:
        n_sub: Sub-cells per edge.
        a_sub: Sub-cell half-width (m).
        T_loc: Local 9x9 T-matrix for each sub-cell.
        omega: Angular frequency (rad/s).
        ref: Background medium.
        cell_average: Use the SINGLE (sinc^1) receiver-cell average, which is
            what pairs with the collocation single-site closure.  See
            ``cell_averaged_pair`` and
            ``scripts/gate_sphere_cell_average_vs_mie.py``.
        n_gauss: Gauss points per axis; ``None`` picks it from the separation.

    Returns:
        kernel_hat: shape (9, 9, nP, nP, nP), complex. FFT of the
            circularly-embedded kernel.
    """
    dd = 2.0 * a_sub
    nP = 2 * n_sub - 1

    kernel = np.zeros((9, 9, nP, nP, nP), dtype=complex)

    span = np.arange(-(n_sub - 1), n_sub)
    offsets = np.stack(np.meshgrid(span, span, span, indexing="ij"), -1).reshape(-1, 3)
    offsets = offsets[np.any(offsets != 0, axis=1)]
    R = offsets.astype(float) * dd
    P_blocks = (
        np.stack([averaged_pair_block_9x9(r, omega, ref, dd, n_gauss=n_gauss) for r in R])
        if cell_average
        else _propagator_blocks_9x9(R, omega, ref)
    )
    # Circular embedding: negative offsets wrap
    i0, i1, i2 = (offsets % nP).T
    kernel[:, :, i0, i1, i2] = -np.einsum("kab,bc->ack", P_blocks, T_loc)

    # FFT each of the 81 (i, j) components
    kernel_hat = np.zeros_like(kernel)
    for i in range(9):
        for j in range(9):
            kernel_hat[i, j] = np.fft.fftn(kernel[i, j])

    return kernel_hat


def _pack(
    w_flat: NDArray[np.complexfloating],
    grid_idx: NDArray[np.intp],
    nP: int,
) -> NDArray[np.complexfloating]:
    """Pack flat 9*nC vector onto (9, nP, nP, nP) grid.

    Uses vectorized fancy indexing for efficiency.

    Args:
        w_flat: Flat vector of shape (9*nC,).
        grid_idx: Grid indices, shape (nC, 3).
        nP: Padded grid size (2*n_sub - 1).

    Returns:
        grids: shape (9, nP, nP, nP), zero-padded.
    """
    nC = len(grid_idx)
    grids = np.zeros((9, nP, nP, nP), dtype=complex)
    # Reshape w_flat to (nC, 9) and scatter to grid
    w_block = w_flat.reshape(nC, 9)
    gi = grid_idx
    for c in range(9):
        grids[c, gi[:, 0], gi[:, 1], gi[:, 2]] = w_block[:, c]
    return grids


def _unpack(
    grids: NDArray[np.complexfloating],
    grid_idx: NDArray[np.intp],
    nC: int,
) -> NDArray[np.complexfloating]:
    """Unpack (9, nP, nP, nP) grid to flat 9*nC vector.

    Args:
        grids: Grid data, shape (9, nP, nP, nP).
        grid_idx: Grid indices, shape (nC, 3).
        nC: Number of active cells.

    Returns:
        w_flat: Flat vector of shape (9*nC,).
    """
    w_block = np.zeros((nC, 9), dtype=complex)
    gi = grid_idx
    for c in range(9):
        w_block[:, c] = grids[c, gi[:, 0], gi[:, 1], gi[:, 2]]
    return w_block.ravel()


def _matvec_fft(
    w_flat: NDArray[np.complexfloating],
    kernel_hat: NDArray[np.complexfloating],
    grid_idx: NDArray[np.intp],
    nP: int,
    nC: int,
) -> NDArray[np.complexfloating]:
    """Compute (I - P*T)*w via FFT convolution.

    The kernel stores -P*T, so w + IFFT(kernel_hat * FFT(w)) = (I - P*T)*w.

    Args:
        w_flat: Input vector, shape (9*nC,).
        kernel_hat: FFT of kernel, shape (9, 9, nP, nP, nP).
        grid_idx: Grid indices, shape (nC, 3).
        nP: Padded grid size.
        nC: Number of active cells.

    Returns:
        Result vector, shape (9*nC,).
    """
    return w_flat + _conv_fft(w_flat, kernel_hat, grid_idx, nP, nC)


def _conv_fft(
    w_flat: NDArray[np.complexfloating],
    kernel_hat: NDArray[np.complexfloating],
    grid_idx: NDArray[np.intp],
    nP: int,
    nC: int,
) -> NDArray[np.complexfloating]:
    """The FFT convolution of the stored kernel with w alone: IFFT(kernel_hat * FFT(w)).

    Args:
        w_flat: Input vector, shape (9*nC,).
        kernel_hat: FFT of kernel, shape (9, 9, nP, nP, nP).
        grid_idx: Grid indices, shape (nC, 3).
        nP: Padded grid size.
        nC: Number of active cells.

    Returns:
        The convolution, shape (9*nC,).
    """
    # Pack input onto grid and FFT
    grids = _pack(w_flat, grid_idx, nP)
    w_hat = np.zeros_like(grids)
    for c in range(9):
        w_hat[c] = np.fft.fftn(grids[c])

    # Pointwise 9x9 multiply in frequency domain
    y_hat = np.zeros_like(w_hat)
    for i in range(9):
        for j in range(9):
            y_hat[i] += kernel_hat[i, j] * w_hat[j]

    # IFFT and unpack
    y_grids = np.zeros_like(y_hat)
    for c in range(9):
        y_grids[c] = np.fft.ifftn(y_hat[c])

    return _unpack(y_grids, grid_idx, nC)


def compute_sphere_foldy_lax_fft(
    omega: float,
    radius: float,
    ref: ReferenceMedium,
    contrast: MaterialContrast,
    n_sub: int,
    k_hat: NDArray | None = None,
    wave_type: str = "S",
    gmres_tol: float = 1e-8,
    gmres_maxiter: int = 200,
    *,
    cell_average: bool = True,
    n_gauss: int | None = None,
    inside: Callable[[NDArray[np.floating]], bool] | None = None,
    contrast_profile: Callable[[NDArray[np.floating]], float] | None = None,
) -> SphereDecompositionResult:
    """Compute sphere T-matrix via FFT-accelerated Foldy-Lax.

    Drop-in replacement for compute_sphere_foldy_lax that uses
    FFT convolution + GMRES instead of dense assembly + direct solve.
    Scales to O(N log N) per iteration instead of O(N^3).

    Args:
        omega: Angular frequency (rad/s).
        radius: Sphere radius (m).
        ref: Background medium.
        contrast: Material contrasts.
        n_sub: Number of sub-cells per edge of bounding cube.
        k_hat: Unit incident direction (default z-hat).
        wave_type: 'S' or 'P'.
        gmres_tol: Relative tolerance for GMRES (default 1e-8).
        gmres_maxiter: Maximum GMRES iterations (default 200).
        cell_average: Use the SINGLE (sinc^1) receiver-cell average.  Default
            True: it is the propagator that matches the collocation single-site
            closure, and it is measured 5.3x closer to exact Mie at ka = 0.1.
        n_gauss: Gauss points per axis; ``None`` picks it from the separation.
        inside: Scatterer shape as a cell-centre test; ``None`` is the sphere of ``radius``.  See
            ``_build_grid_index_map``.
        contrast_profile: A medium that varies from cell to cell: the cell centred at x carries
            ``contrast_profile(x)`` times ``contrast``, and so its own cube T-matrix.  The kernel is then
            the propagator alone and each cell's T is applied before the convolution.  ``None`` (the
            default) gives every cell ``contrast``, with the single T folded into the kernel.

    Returns:
        SphereDecompositionResult with composite T-matrix.
    """
    # Step 1: Grid index mapping
    grid_idx, centres, a_sub = _build_grid_index_map(radius, n_sub, inside)
    nC = len(centres)
    nP = 2 * n_sub - 1

    def cell_tmatrix(c: MaterialContrast) -> NDArray[np.complexfloating]:
        return _sub_cell_tmatrix_9x9(compute_cube_tmatrix(omega, a_sub, ref, c), omega, a_sub)

    dim = 9 * nC
    t_local: NDArray[np.complexfloating] | None = None
    if contrast_profile is None:
        # Every cell carries the same T: fold it into the kernel, which then stores -P T
        T_loc = cell_tmatrix(contrast)
        kernel_hat = _build_fft_kernel(
            n_sub, a_sub, T_loc, omega, ref, cell_average=cell_average, n_gauss=n_gauss
        )

        def matvec(w: NDArray) -> NDArray:
            return _matvec_fft(w, kernel_hat, grid_idx, nP, nC)

    else:
        # Each cell its own T (computed once per distinct contrast factor); the kernel stores -P alone,
        # and (I - P T) w = w + conv(-P, T w)
        cache: dict[float, NDArray[np.complexfloating]] = {}
        t_cells = np.empty((nC, 9, 9), dtype=complex)
        for n_cell, pos in enumerate(centres):
            g = float(contrast_profile(pos))
            if g not in cache:
                cache[g] = cell_tmatrix(
                    MaterialContrast(
                        Dlambda=g * contrast.Dlambda, Dmu=g * contrast.Dmu, Drho=g * contrast.Drho
                    )
                )
            t_cells[n_cell] = cache[g]
        t_local = t_cells
        kernel_hat = _build_fft_kernel(
            n_sub, a_sub, np.eye(9, dtype=complex), omega, ref, cell_average=cell_average, n_gauss=n_gauss
        )

        def matvec(w: NDArray) -> NDArray:
            # the convolution alone: forming tw + conv and subtracting tw again cancelled about eight
            # digits, since in SI units tw = T w is of order 1e8 while the convolution is of order one
            tw = np.einsum("nij,nj->ni", t_cells, w.reshape(nC, 9)).ravel()
            return w + _conv_fft(tw, kernel_hat, grid_idx, nP, nC)

    A_op = LinearOperator((dim, dim), matvec=matvec, dtype=complex)

    # Step 4: Build incident fields (9N x 9 matrices, solved column by column): the phase-free Taylor
    # patterns of the composite T, and the plane-wave basis for the far field.  They share no column.
    psi_inc = _build_incident_field_coupled(centres)
    pw_inc = _build_incident_plane_wave_basis(centres, omega, ref, k_hat=k_hat, wave_type=wave_type)

    def solve(rhs: NDArray, label: str) -> NDArray:
        x0 = rhs.copy()  # Born approximation as initial guess
        solution, info = gmres(A_op, rhs, x0=x0, rtol=gmres_tol, maxiter=gmres_maxiter)
        if info != 0:
            import warnings

            warnings.warn(f"GMRES did not converge for {label} (info={info})", UserWarning, stacklevel=3)
        return solution

    psi_exc = np.zeros((dim, 9), dtype=complex)
    for col in range(9):
        psi_exc[:, col] = solve(psi_inc[:, col], f"column {col}")
    psi_pw = np.zeros((dim, 9), dtype=complex)
    for col in range(9):
        psi_pw[:, col] = solve(pw_inc[:, col], f"plane-wave column {col}")

    # Step 5: Extract composite T-matrix
    T_comp = np.zeros((9, 9), dtype=complex)
    for n in range(nC):
        T_comp += (T_loc if t_local is None else t_local[n]) @ psi_exc[9 * n : 9 * n + 9, :]

    T3x3 = T_comp[:3, :3].copy()

    return SphereDecompositionResult(
        T3x3=T3x3,
        T_comp_9x9=T_comp,
        centres=centres,
        n_sub=n_sub,
        n_cells=nC,
        a_sub=a_sub,
        condition_number=float("nan"),  # not available from iterative solver
        psi_exc=psi_exc,
        omega=omega,
        radius=radius,
        ref=ref,
        contrast=contrast,
        t_local=t_local,
        psi_pw=psi_pw,
    )
