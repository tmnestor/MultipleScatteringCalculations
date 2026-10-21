"""GPU-accelerated FFT sphere Foldy-Lax solver.

PyTorch port of sphere_scattering_fft.py. CPU code builds the kernel
and incident field; GPU handles the GMRES matvec via 3D FFT convolution.

Pack/unpack uses fancy indexing (not torch.scatter_ which doesn't
support complex on MPS).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from numpy.typing import NDArray

from .effective_contrasts import (
    MaterialContrast,
    ReferenceMedium,
    compute_cube_tmatrix,
)
from .resonance_tmatrix import (
    _build_incident_field_coupled,
    _build_incident_plane_wave_basis,
    _sub_cell_tmatrix_9x9,
)
from .sphere_scattering import SphereDecompositionResult
from .sphere_scattering_fft import _build_fft_kernel, _build_grid_index_map
from .torch_gmres import get_device, select_dtype, to_numpy, to_torch, torch_gmres

__all__ = ["compute_sphere_foldy_lax_fft_gpu", "point_wave_far_field", "solve_point_plane_wave"]


def _pack_gpu(
    w_flat: torch.Tensor,
    grid_idx: torch.Tensor,
    nP: int,
) -> torch.Tensor:
    """Pack flat (9*nC,) tensor onto (9, nP, nP, nP) grid via fancy indexing.

    Args:
        w_flat: Input vector, shape (9*nC,).
        grid_idx: Grid indices, shape (nC, 3), long tensor on same device.
        nP: Padded grid size (2*n_sub - 1).

    Returns:
        grids: shape (9, nP, nP, nP), zero-padded.
    """
    nC = grid_idx.shape[0]
    device = w_flat.device
    dtype = w_flat.dtype

    grids = torch.zeros(9, nP, nP, nP, device=device, dtype=dtype)
    w_block = w_flat.reshape(nC, 9)

    i0 = grid_idx[:, 0]
    i1 = grid_idx[:, 1]
    i2 = grid_idx[:, 2]

    for c in range(9):
        grids[c, i0, i1, i2] = w_block[:, c]

    return grids


def _unpack_gpu(
    grids: torch.Tensor,
    grid_idx: torch.Tensor,
    nC: int,
) -> torch.Tensor:
    """Unpack (9, nP, nP, nP) grid to flat (9*nC,) via fancy indexing.

    Args:
        grids: Grid data, shape (9, nP, nP, nP).
        grid_idx: Grid indices, shape (nC, 3), long tensor.
        nC: Number of active cells.

    Returns:
        w_flat: Flat vector, shape (9*nC,).
    """
    device = grids.device
    dtype = grids.dtype

    w_block = torch.zeros(nC, 9, device=device, dtype=dtype)

    i0 = grid_idx[:, 0]
    i1 = grid_idx[:, 1]
    i2 = grid_idx[:, 2]

    for c in range(9):
        w_block[:, c] = grids[c, i0, i1, i2]

    return w_block.ravel()


def _point_major(kernel_hat: torch.Tensor) -> torch.Tensor:
    """The FFT kernel (9, 9, nP, nP, nP) rearranged to (nP, nP, nP, 9, 9), contiguous.

    One 9 x 9 matrix per Fourier point, stored together, so the pointwise product is a batched matrix
    multiply over contiguous memory: 2.3 ms against 5.2 ms for the einsum over the (9, 9, ...) layout at
    n_sub = 25 (complex128, 8 threads), with identical results.
    """
    return kernel_hat.permute(2, 3, 4, 0, 1).contiguous()


@dataclass(frozen=True)
class CanonicalKernel:
    """The FFT kernel at the canonical k only, with what applies it at every Fourier point.

    THE CUBE GROUP IN FOURIER SPACE.  The point kernel -P(r) T is equivariant, -P(Q r) T = S (-P(r) T) S^T
    for the 48 signed permutations Q (S = diag(Q, R6(Q)), and S T S^T = T for the cube's T-matrix), and Q
    maps the circular grid to itself, so K^(Q k) = S K^(k) S^T: one 9 x 9 block per orbit of k, the
    canonical k1 >= k2 >= k3 >= 0, determines the kernel (C(n + 2, 3) blocks: 1/42 of the points at
    n = 32). See ``graded_voxel.fft.canonical_transformed_kernel`` for the Galerkin kernel.

    TWO ROUTES TO THE SAME PRODUCT.  On the CPU with ``point_kernel.backend: fortran`` the compiled
    ``kernel_apply.apply_canon_*`` (``graded_voxel.kernel_fortran.apply_canonical_fortran``) does each point
    in one pass: the sources re-indexed with signs, the canonical block applied, the result re-indexed
    back. Elsewhere (a GPU, or the reference backend) the same product is three gathers-and-multiply steps,
    device-agnostic and without a scatter (MPS supports none for complex): the points of each orbit are
    listed with their canonical block, padded to the largest orbit (48) with a zero point, the sources are
    gathered into each point's canonical frame from [w, -w] (the sign folded into the index), multiplied
    by the stored blocks in one batched matmul, and the results gathered back from [z, -z]. On the CPU the
    gathers cost more than reading the full kernel does (measured 1.5 times slower per matvec at n = 32),
    hence the compiled route there.

    Attributes:
        kc: The canonical blocks (K, 9, 9).
        shape: (K, M): canonical blocks, and the points listed per block (the largest orbit).
        idx_in: (K * M * 9,) index into [w, -w, 0] flattened, w the point-major sources (F, 9).
        idx_out: (F * 9,) index into [z, -z] flattened, z the products (K, M, 9).
        compiled: The compiled route (the CPU with the compiled backend).
        khc: For the compiled route, the blocks as ``apply_canonical_fortran`` takes them, (K, col, row).
        tables: For the compiled route, ``graded_voxel.fft.fourier_symmetry_tables`` for 9 components.
    """

    kc: torch.Tensor
    shape: tuple[int, int]
    idx_in: torch.Tensor
    idx_out: torch.Tensor
    compiled: bool
    khc: NDArray[np.complexfloating] | None
    tables: tuple[NDArray, ...] | None


def _canonical_kernel(
    kernel_hat: NDArray[np.complexfloating], n_sub: int, device: torch.device, dtype: torch.dtype
) -> CanonicalKernel:
    """The canonical blocks of ``_build_fft_kernel``'s kernel (9, 9, nP, nP, nP) and their index tables.

    The re-indexing at a point f = Q k_c is ``graded_voxel.fft.fourier_symmetry_tables`` with one test
    function and one source monomial (the constant, which every symmetry fixes): rows and columns are the
    9 components, K^(f)[r, c] = sr[r] sc[c] K^(k_c)[pr[r], pc[c]].
    """
    from .graded_voxel.fft import _orbit_arrays, fourier_symmetry_tables
    from .graded_voxel.kernel import point_kernel_backend

    n_p = 2 * n_sub - 1
    n_f = n_p**3
    keys = _orbit_arrays(n_sub)[3]
    tables = fourier_symmetry_tables(n_sub, 1, 1, 10, 4)
    q_f, k_f, order, pr, sr, pc, sc = tables
    at = np.array([(a * n_p + b) * n_p + d for a, b, d in keys])
    kc = np.ascontiguousarray(kernel_hat.reshape(9, 9, n_f)[:, :, at].transpose(2, 0, 1))
    n_k = len(keys)
    counts = np.bincount(k_f, minlength=n_k)
    m = int(counts.max())
    start = np.concatenate([[0], np.cumsum(counts)[:-1]])
    slot = np.empty(n_f, dtype=np.int64)
    slot[order] = np.arange(n_f) - start[k_f[order]]  # each point's place in its orbit's list
    members = np.full((n_k, m), n_f, dtype=np.int64)  # n_f: the zero point that pads short orbits
    members[k_f, slot] = np.arange(n_f)
    q_m = np.where(members < n_f, q_f[np.minimum(members, n_f - 1)], 0)  # symmetry 0 is the identity
    inv_pc = np.argsort(pc, axis=1)  # the source column that lands on each canonical column
    src = inv_pc[q_m]  # (K, M, 9)
    negative = np.take_along_axis(sc[q_m], src, axis=2) < 0
    # [w, -w, 0]: w at rows 0 .. F - 1, -w at F .. 2F - 1, the zero point at 2F
    rows = np.where(members < n_f, members, 2 * n_f)[:, :, None] + n_f * negative
    idx_in = rows * 9 + src
    idx_out = (k_f[:, None] * m + slot[:, None]) * 9 + pr[q_f] + n_k * m * 9 * (sr[q_f] < 0)
    compiled = device.type == "cpu" and point_kernel_backend() == "fortran"
    return CanonicalKernel(
        kc=to_torch(kc, device, dtype),
        shape=(n_k, m),
        idx_in=torch.from_numpy(idx_in.ravel().astype(np.int64)).to(device),
        idx_out=torch.from_numpy(idx_out.ravel().astype(np.int64)).to(device),
        compiled=compiled,
        khc=np.ascontiguousarray(kc.transpose(0, 2, 1)).astype(_NUMPY_DTYPE[dtype]) if compiled else None,
        tables=tables if compiled else None,
    )


#: the NumPy dtype of each torch complex dtype, for the compiled route
_NUMPY_DTYPE = {torch.complex128: np.complex128, torch.complex64: np.complex64}


def _apply_canonical(ck: CanonicalKernel, w_pm: torch.Tensor) -> torch.Tensor:
    """y[f] = K^(f) w[f] at every Fourier point through the canonical blocks; w_pm, y point-major (F, 9)."""
    if ck.compiled:
        from .graded_voxel.kernel_fortran import apply_canonical_fortran  # local: imports the kernel

        assert ck.khc is not None and ck.tables is not None
        y = apply_canonical_fortran(ck.khc, w_pm.numpy(), ck.tables)  # (F, 9), in w_pm's precision
        return torch.from_numpy(y).to(w_pm.dtype)  # no copy: the dtypes already agree
    zero = torch.zeros(1, 9, device=w_pm.device, dtype=w_pm.dtype)
    s = torch.cat([w_pm, -w_pm, zero]).reshape(-1)[ck.idx_in].reshape(*ck.shape, 9)
    z = torch.matmul(s, ck.kc.transpose(1, 2)).reshape(-1)  # z[k, m, i] = sum_j kc[k, i, j] s[k, m, j]
    return torch.cat([z, -z])[ck.idx_out].reshape(-1, 9)


def _conv_fft_canonical(
    w_flat: torch.Tensor,
    ck: CanonicalKernel,
    grid_idx: torch.Tensor,
    nP: int,
    nC: int,
) -> torch.Tensor:
    """The convolution of the stored kernel with w alone, through its canonical blocks.

    Args:
        w_flat: Input vector, shape (9*nC,).
        ck: The canonical kernel (``_canonical_kernel``).
        grid_idx: Grid indices, shape (nC, 3).
        nP: Padded grid size.
        nC: Number of active cells.

    Returns:
        The convolution, shape (9*nC,).
    """
    # point-major throughout, (nP, nP, nP, 9), so that the product needs no transposed copies
    i0, i1, i2 = grid_idx[:, 0], grid_idx[:, 1], grid_idx[:, 2]
    grids = torch.zeros(nP, nP, nP, 9, device=w_flat.device, dtype=w_flat.dtype)
    grids[i0, i1, i2] = w_flat.reshape(nC, 9)
    w_hat = torch.fft.fftn(grids, dim=(0, 1, 2))
    y_pm = _apply_canonical(ck, w_hat.reshape(-1, 9))
    y_grids = torch.fft.ifftn(y_pm.reshape(nP, nP, nP, 9), dim=(0, 1, 2))
    return y_grids[i0, i1, i2].reshape(-1)


def _matvec_fft_canonical(
    w_flat: torch.Tensor, ck: CanonicalKernel, grid_idx: torch.Tensor, nP: int, nC: int
) -> torch.Tensor:
    """(I - P*T)*w via FFT convolution, the kernel (-P T) applied through its canonical blocks."""
    return w_flat + _conv_fft_canonical(w_flat, ck, grid_idx, nP, nC)


def _column_orbits(qs: Sequence[NDArray]) -> tuple[list[int], list[tuple[int, int, NDArray, float]]]:
    """The orbits of the 9 incident columns e_c under the symmetries ``qs`` (signed permutations).

    THE RIGHT-HAND SIDES BY THE CUBE GROUP.  (I - P T) commutes with U_Q, (U_Q psi)(Q x) = S psi(x), for
    every Q that maps the grid of cells to itself (the centred sphere grid: all 48), since the kernel is
    equivariant and S T S^T = T. An incident column whose field transforms as U_Q b_c = S[p, c] b_p, with
    p the component that S takes c to, therefore has the solution psi_p = S[p, c] U_Q psi_c. The Taylor
    patterns (uniform displacement, linear strain about the centre) transform so under every Q: three
    orbits, displacements, normal strains and shear strains. The plane-wave columns e_c exp(i k.x) do
    under the Q with Q k = k: on a coordinate axis eight of them, and six orbits.

    Returns:
        (reps, derived): the column solved for each orbit, in increasing order, and for every other column
        (c, rep, Q, sign) with psi_c = sign U_Q psi_rep.
    """
    from .graded_voxel.fft import symmetry_reps

    reps: list[int] = []
    derived: list[tuple[int, int, NDArray, float]] = []
    seen: set[int] = set()
    for c in range(9):
        if c in seen:
            continue
        reps.append(c)
        seen.add(c)
        for q in qs:
            s = symmetry_reps(q)[2]
            p = int(np.argmax(np.abs(s[:, c])))
            if p not in seen:
                seen.add(p)
                derived.append((p, c, q, float(s[p, c])))
    return reps, derived


def _cell_permutation(grid_idx: NDArray[np.intp], n_sub: int, q: NDArray) -> NDArray[np.intp]:
    """perm[n]: the cell at Q x_n, for the cells of the centred grid of ``_build_grid_index_map``.

    Raises:
        ValueError: if Q does not map the cells to themselves (a shape without the cube's symmetry).
    """
    half = (n_sub - 1) / 2.0
    image = np.rint((grid_idx - half) @ q.T + half).astype(np.intp)
    lookup = np.full((n_sub,) * 3, -1, dtype=np.intp)
    lookup[grid_idx[:, 0], grid_idx[:, 1], grid_idx[:, 2]] = np.arange(len(grid_idx))
    inside = np.all((image >= 0) & (image < n_sub), axis=1)
    perm = np.full(len(grid_idx), -1, dtype=np.intp)
    perm[inside] = lookup[image[inside, 0], image[inside, 1], image[inside, 2]]
    if np.any(perm < 0):
        raise ValueError(
            "_cell_permutation: a symmetry of the cube maps a cell outside the scatterer, so the incident "
            "columns cannot be generated by symmetry. The shape must be mapped to itself by the cube group "
            "(the centred sphere of _build_grid_index_map is)."
        )
    return perm


def _apply_symmetry(psi: NDArray, grid_idx: NDArray[np.intp], n_sub: int, q: NDArray) -> NDArray:
    """(U_Q psi): the field of every cell n carried to the cell at Q x_n, its components rotated by S."""
    from .graded_voxel.fft import symmetry_reps

    s = symmetry_reps(q)[2]
    out = np.empty((len(grid_idx), 9), dtype=complex)
    out[_cell_permutation(grid_idx, n_sub, q)] = psi.reshape(-1, 9) @ s.T
    return out.ravel()


def _matvec_fft_gpu(
    w_flat: torch.Tensor,
    kernel_pm: torch.Tensor,
    grid_idx: torch.Tensor,
    nP: int,
    nC: int,
) -> torch.Tensor:
    """Compute (I - P*T)*w via FFT convolution on GPU.

    Args:
        w_flat: Input vector, shape (9*nC,).
        kernel_pm: FFT of kernel, point-major (``_point_major``), shape (nP, nP, nP, 9, 9).
        grid_idx: Grid indices, shape (nC, 3).
        nP: Padded grid size.
        nC: Number of active cells.

    Returns:
        Result vector, shape (9*nC,).
    """
    # Pack onto grid and batch-FFT
    grids = _pack_gpu(w_flat, grid_idx, nP)
    w_hat = torch.fft.fftn(grids, dim=(1, 2, 3))

    # Pointwise 9x9 multiply: y_hat[i] = sum_j kernel_hat[i,j] * w_hat[j], one matrix per Fourier point
    y_pm = torch.matmul(kernel_pm, w_hat.permute(1, 2, 3, 0).unsqueeze(-1)).squeeze(-1)
    y_hat = y_pm.permute(3, 0, 1, 2)

    # Batch-IFFT and unpack
    y_grids = torch.fft.ifftn(y_hat, dim=(1, 2, 3))
    conv_result = _unpack_gpu(y_grids, grid_idx, nC)

    # (I - P*T)*w = w + conv_result  (kernel = -P*T)
    return w_flat + conv_result


def compute_sphere_foldy_lax_fft_gpu(
    omega: float,
    radius: float,
    ref: ReferenceMedium,
    contrast: MaterialContrast,
    n_sub: int,
    k_hat: NDArray | None = None,
    wave_type: str = "S",
    gmres_tol: float = 1e-8,
    gmres_maxiter: int = 200,
    initial_guess: str = "born",
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> SphereDecompositionResult:
    """Compute sphere T-matrix via GPU FFT-accelerated Foldy-Lax.

    Drop-in replacement for compute_sphere_foldy_lax_fft that runs
    the GMRES iterations on GPU via PyTorch.

    Args:
        omega: Angular frequency (rad/s).
        radius: Sphere radius (m).
        ref: Background medium.
        contrast: Material contrasts.
        n_sub: Number of sub-cells per edge of bounding cube.
        k_hat: Unit incident direction (default z-hat).
        wave_type: 'S' or 'P'.
        gmres_tol: Relative tolerance for GMRES.
        gmres_maxiter: Maximum GMRES iterations.
        initial_guess: "born" or "zero".
        device: PyTorch device (default: auto-detect).
        dtype: PyTorch complex dtype (default: auto-select).

    Returns:
        SphereDecompositionResult with composite T-matrix.
    """
    if device is None:
        device = get_device()
    if dtype is None:
        dtype = select_dtype(device)

    # Step 1: Build grid and kernel on CPU (reuse existing code)
    grid_idx_np, centres, a_sub = _build_grid_index_map(radius, n_sub)
    nC = len(centres)
    nP = 2 * n_sub - 1

    rayleigh_sub = compute_cube_tmatrix(omega, a_sub, ref, contrast)
    T_loc = _sub_cell_tmatrix_9x9(rayleigh_sub, omega, a_sub)

    kernel_hat_np = _build_fft_kernel(n_sub, a_sub, T_loc, omega, ref)

    # Step 2: Transfer to GPU, the kernel at the canonical k only (1/42 of it at n_sub = 32)
    kernel_c = _canonical_kernel(kernel_hat_np, n_sub, device, dtype)
    del kernel_hat_np
    grid_idx_gpu = torch.from_numpy(grid_idx_np.copy()).long().to(device)

    # Step 3: Build incident field on CPU
    psi_inc = _build_incident_field_coupled(centres)

    # the plane-wave basis for the far field: it shares no column with the Taylor patterns psi_inc
    pw_inc = _build_incident_plane_wave_basis(centres, omega, ref, k_hat=k_hat, wave_type=wave_type)

    # Step 4: Solve one column per orbit of the cube group via GPU GMRES (``_column_orbits``): the Taylor
    # columns under all 48 symmetries (3 solves for 9), the plane-wave columns under those fixing k (6 for 9
    # on an axis); the other columns follow exactly by the cell permutation and the signs
    dim = 9 * nC
    psi_exc = np.zeros((dim, 9), dtype=complex)
    psi_pw = np.zeros((dim, 9), dtype=complex)

    def matvec(w: torch.Tensor) -> torch.Tensor:
        return _matvec_fft_canonical(w, kernel_c, grid_idx_gpu, nP, nC)

    from .graded_voxel.fft import signed_permutations
    from .resonance_tmatrix import _plane_wave_k_vec

    qs = signed_permutations()
    k_vec = _plane_wave_k_vec(omega, ref, k_hat, wave_type)
    fixing_k = [q for q in qs if np.allclose(q @ k_vec, k_vec, rtol=0.0, atol=1e-14 * np.abs(k_vec).max())]
    orbits = {"exc": _column_orbits(qs), "pw": _column_orbits(fixing_k)}
    jobs = [("exc", col, psi_inc[:, col]) for col in orbits["exc"][0]] + [
        ("pw", col, pw_inc[:, col]) for col in orbits["pw"][0]
    ]
    for which, col, rhs_np in jobs:
        b = to_torch(rhs_np, device, dtype)

        if initial_guess == "born":
            x0 = b.clone()
        else:
            x0 = None

        x, n_iter, rel_res = torch_gmres(matvec, b, x0=x0, tol=gmres_tol, maxiter=gmres_maxiter)

        if rel_res > gmres_tol:
            import warnings

            warnings.warn(
                f"GPU GMRES did not converge for column {col}: "
                f"rel_res={rel_res:.2e} after {n_iter} iterations",
                UserWarning,
                stacklevel=2,
            )

        (psi_exc if which == "exc" else psi_pw)[:, col] = to_numpy(x)

    for which, psi in (("exc", psi_exc), ("pw", psi_pw)):
        for col, rep, q, sign in orbits[which][1]:  # psi_col = sign U_Q psi_rep
            psi[:, col] = sign * _apply_symmetry(psi[:, rep], grid_idx_np, n_sub, q)

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
        condition_number=float("nan"),
        psi_exc=psi_exc,
        omega=omega,
        radius=radius,
        ref=ref,
        contrast=contrast,
        psi_pw=psi_pw,
    )


@dataclass(frozen=True)
class PointWaveResult:
    """The point formulation's exciting field for one incident plane wave (``solve_point_plane_wave``).

    Attributes:
        centres: Cell centres (N, 3).
        psi: The exciting field in each cell (N, 9): displacement, then Voigt strain.
        t_cells: Each cell's 9 x 9 T-matrix (N, 9, 9).
        omega: Angular frequency.
        ref: Background medium.
        a_sub: Cell half-width (m).
    """

    centres: NDArray[np.floating]
    psi: NDArray[np.complexfloating]
    t_cells: NDArray[np.complexfloating]
    omega: float
    ref: ReferenceMedium
    a_sub: float


def solve_point_plane_wave(
    omega: float,
    radius: float,
    ref: ReferenceMedium,
    contrast: MaterialContrast,
    n_sub: int,
    k_hat: NDArray,
    pol: NDArray,
    wave_type: str,
    *,
    profile: Callable[[NDArray[np.floating]], float] | None,
    gmres_tol: float,
    device: torch.device,
    dtype: torch.dtype,
    gmres_maxiter: int = 500,
    timings: dict[str, float] | None = None,
) -> PointWaveResult:
    """The point formulation for one incident plane wave: one GMRES solve on the canonical kernel.

    The scattered field of one wave needs only its own right-hand side, the plane wave
    exp(i k.x) [p, eps] at every cell, not the eighteen columns of ``compute_sphere_foldy_lax_fft_gpu``
    (the composite T-matrix and the plane-wave basis). A uniform contrast folds the cube's T into the
    kernel, -P T; a graded one (``profile``: the cell centred at x carries profile(x) times ``contrast``,
    as ``compute_sphere_foldy_lax_fft``'s ``contrast_profile``) applies each cell's T before the
    convolution with -P alone, w + conv(-P, T w). Both kernels are equivariant under the cube group, so
    both are stored at the canonical k only (``CanonicalKernel``).

    Args:
        omega: Angular frequency (rad/s).
        radius: Sphere radius (m).
        ref: Background medium.
        contrast: Material contrast (scaled by ``profile`` when given).
        n_sub: Cells per edge of the bounding cube; cells whose centre is inside the sphere are kept.
        k_hat: Incident direction.
        pol: Incident polarisation.
        wave_type: 'P' or 'S'.
        profile: The contrast profile, or None for a uniform contrast.
        gmres_tol: GMRES relative tolerance.
        device: Torch device.
        dtype: Torch complex dtype.
        gmres_maxiter: GMRES iteration bound.
        timings: When given, filled with kernel_s (cell T-matrices, kernel and its canonical blocks),
            solve_s (GMRES) and iterations.

    Raises:
        RuntimeError: if GMRES does not reach ``gmres_tol`` within ``gmres_maxiter`` iterations.
    """
    from .resonance_tmatrix import _plane_wave_k_vec
    from .sphere_scattering import _plane_wave_strain_voigt

    t0 = time.perf_counter()
    grid_idx_np, centres, a_sub = _build_grid_index_map(radius, n_sub)
    n_c, n_p = len(centres), 2 * n_sub - 1

    def cell_tmatrix(c: MaterialContrast) -> NDArray[np.complexfloating]:
        return _sub_cell_tmatrix_9x9(compute_cube_tmatrix(omega, a_sub, ref, c), omega, a_sub)

    if profile is None:
        t_loc = cell_tmatrix(contrast)
        t_cells = np.broadcast_to(t_loc, (n_c, 9, 9))
        kernel_hat = _build_fft_kernel(n_sub, a_sub, t_loc, omega, ref)
    else:
        from .graded_voxel.site import BatchProfile

        cache: dict[float, NDArray[np.complexfloating]] = {}
        t_list = []
        # a BatchProfile at every centre in one call; a plain profile point by point
        factors = (
            profile.batch(centres) if isinstance(profile, BatchProfile) else [profile(x) for x in centres]
        )
        for g in map(float, factors):
            if g not in cache:
                cache[g] = cell_tmatrix(
                    MaterialContrast(
                        Dlambda=g * contrast.Dlambda, Dmu=g * contrast.Dmu, Drho=g * contrast.Drho
                    )
                )
            t_list.append(cache[g])
        t_cells = np.array(t_list)
        kernel_hat = _build_fft_kernel(n_sub, a_sub, np.eye(9, dtype=complex), omega, ref)
    kernel_c = _canonical_kernel(kernel_hat, n_sub, device, dtype)
    del kernel_hat
    grid_idx = torch.from_numpy(grid_idx_np.copy()).long().to(device)
    t_dev = None if profile is None else to_torch(t_cells, device, dtype)
    t1 = time.perf_counter()

    def matvec(w: torch.Tensor) -> torch.Tensor:
        if t_dev is None:
            return w + _conv_fft_canonical(w, kernel_c, grid_idx, n_p, n_c)
        tw = torch.einsum("nij,nj->ni", t_dev, w.reshape(n_c, 9)).reshape(-1)
        return w + _conv_fft_canonical(tw, kernel_c, grid_idx, n_p, n_c)

    k_vec = _plane_wave_k_vec(omega, ref, k_hat, wave_type)
    pol = np.asarray(pol, dtype=float)
    k_mag = float(np.linalg.norm(k_vec))
    inc = np.concatenate([pol, _plane_wave_strain_voigt(k_vec / k_mag, pol, k_mag)])
    rhs = (np.exp(1j * (centres @ k_vec))[:, None] * inc[None, :]).ravel()
    b = to_torch(rhs, device, dtype)
    x, n_iter, rel_res = torch_gmres(matvec, b, x0=b.clone(), tol=gmres_tol, maxiter=gmres_maxiter)
    if rel_res > gmres_tol:
        raise RuntimeError(
            f"solve_point_plane_wave: GMRES reached a relative residual of {rel_res:.2e} after {n_iter} "
            f"iterations, above gmres_tol = {gmres_tol:g} (n_sub = {n_sub}). Fix: raise gmres_maxiter or "
            "loosen gmres_tol."
        )
    if timings is not None:
        timings.update(kernel_s=t1 - t0, solve_s=time.perf_counter() - t1, iterations=float(n_iter))
    return PointWaveResult(centres, to_numpy(x).reshape(n_c, 9), np.asarray(t_cells), omega, ref, a_sub)


def point_wave_far_field(
    res: PointWaveResult, r_hat_arr: NDArray[np.floating], r_distance: float
) -> tuple[NDArray[np.complexfloating], NDArray[np.complexfloating]]:
    """(u_P, u_S) far fields of ``solve_point_plane_wave``'s solution: ``point_far_field`` of T_n psi_n."""
    from .sphere_scattering import point_far_field

    sources = np.einsum("nij,nj->ni", res.t_cells, res.psi)
    return point_far_field(res.centres, sources, res.omega, res.ref, r_hat_arr, r_distance)
