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

from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

if TYPE_CHECKING:
    from numpy.typing import NDArray

from .effective_contrasts import (
    MaterialContrast,
    ReferenceMedium,
    compute_cube_tmatrix,
)
from .resonance_tmatrix import (
    _build_incident_field_coupled,
    _propagator_block_9x9,
    _sub_cell_tmatrix_9x9,
)
from .sphere_scattering import SphereDecompositionResult


def _build_grid_index_map(
    radius: float,
    n_sub: int,
) -> tuple[NDArray[np.intp], NDArray[np.floating], float]:
    """Build grid index mapping for sphere sub-cells.

    Creates an n_sub x n_sub x n_sub grid, filters to cells inside the
    sphere, and records each cell's (i0, i1, i2) grid index.

    Args:
        radius: Sphere radius (m).
        n_sub: Number of sub-cells per edge of bounding cube.

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
                if np.linalg.norm(pos) < radius:
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
) -> NDArray[np.complexfloating]:
    """Build FFT kernel for the propagator convolution.

    For all separations (d0, d1, d2) in [-(n-1), +(n-1)]^3, computes
    -P(r)*T_loc (9x9) and stores on a (2n-1)^3 grid with circular
    embedding.  Then FFTs each of the 81 components.

    Args:
        n_sub: Sub-cells per edge.
        a_sub: Sub-cell half-width (m).
        T_loc: Local 9x9 T-matrix for each sub-cell.
        omega: Angular frequency (rad/s).
        ref: Background medium.

    Returns:
        kernel_hat: shape (9, 9, nP, nP, nP), complex. FFT of the
            circularly-embedded kernel.
    """
    dd = 2.0 * a_sub
    nP = 2 * n_sub - 1

    kernel = np.zeros((9, 9, nP, nP, nP), dtype=complex)

    for d0 in range(-(n_sub - 1), n_sub):
        for d1 in range(-(n_sub - 1), n_sub):
            for d2 in range(-(n_sub - 1), n_sub):
                if d0 == 0 and d1 == 0 and d2 == 0:
                    continue
                r_vec = np.array([d0, d1, d2], dtype=float) * dd
                P_block = _propagator_block_9x9(r_vec, omega, ref)
                block = -(P_block @ T_loc)
                # Circular embedding: negative offsets wrap
                i0 = d0 % nP
                i1 = d1 % nP
                i2 = d2 % nP
                kernel[:, :, i0, i1, i2] = block

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

    conv_result = _unpack(y_grids, grid_idx, nC)

    # (I - P*T)*w = w + conv_result  (since kernel = -P*T)
    return w_flat + conv_result


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

    Returns:
        SphereDecompositionResult with composite T-matrix.
    """
    # Step 1: Grid index mapping
    grid_idx, centres, a_sub = _build_grid_index_map(radius, n_sub)
    nC = len(centres)
    nP = 2 * n_sub - 1

    # Sub-cell Rayleigh T-matrix (same for all cells)
    rayleigh_sub = compute_cube_tmatrix(omega, a_sub, ref, contrast)
    T_loc = _sub_cell_tmatrix_9x9(rayleigh_sub, omega, a_sub)

    # Step 2: Build FFT kernel
    kernel_hat = _build_fft_kernel(n_sub, a_sub, T_loc, omega, ref)

    # Step 3: Build matvec operator
    dim = 9 * nC

    def matvec(w: NDArray) -> NDArray:
        return _matvec_fft(w, kernel_hat, grid_idx, nP, nC)

    A_op = LinearOperator((dim, dim), matvec=matvec, dtype=complex)

    # Step 4: Build incident field (9N x 9 matrix, solve column by column)
    psi_inc = _build_incident_field_coupled(
        centres, omega, ref, k_hat=k_hat, wave_type=wave_type
    )

    # Solve 9 independent RHS columns via GMRES
    psi_exc = np.zeros((dim, 9), dtype=complex)
    for col in range(9):
        rhs = psi_inc[:, col]
        x0 = rhs.copy()  # Born approximation as initial guess
        solution, info = gmres(
            A_op,
            rhs,
            x0=x0,
            rtol=gmres_tol,
            maxiter=gmres_maxiter,
        )
        if info != 0:
            import warnings

            warnings.warn(
                f"GMRES did not converge for column {col} (info={info})",
                UserWarning,
                stacklevel=2,
            )
        psi_exc[:, col] = solution

    # Step 5: Extract composite T-matrix
    T_comp = np.zeros((9, 9), dtype=complex)
    for n in range(nC):
        T_comp += T_loc @ psi_exc[9 * n : 9 * n + 9, :]

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
    )
