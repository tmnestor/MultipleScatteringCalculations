"""GPU-accelerated FFT sphere Foldy-Lax solver.

PyTorch port of sphere_scattering_fft.py. CPU code builds the kernel
and incident field; GPU handles the GMRES matvec via 3D FFT convolution.

Pack/unpack uses fancy indexing (not torch.scatter_ which doesn't
support complex on MPS).
"""

from __future__ import annotations

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
    _sub_cell_tmatrix_9x9,
)
from .sphere_scattering import SphereDecompositionResult
from .sphere_scattering_fft import _build_fft_kernel, _build_grid_index_map
from .torch_gmres import get_device, select_dtype, to_numpy, to_torch, torch_gmres

__all__ = ["compute_sphere_foldy_lax_fft_gpu"]


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


def _matvec_fft_gpu(
    w_flat: torch.Tensor,
    kernel_hat: torch.Tensor,
    grid_idx: torch.Tensor,
    nP: int,
    nC: int,
) -> torch.Tensor:
    """Compute (I - P*T)*w via FFT convolution on GPU.

    Args:
        w_flat: Input vector, shape (9*nC,).
        kernel_hat: FFT of kernel, shape (9, 9, nP, nP, nP).
        grid_idx: Grid indices, shape (nC, 3).
        nP: Padded grid size.
        nC: Number of active cells.

    Returns:
        Result vector, shape (9*nC,).
    """
    # Pack onto grid and batch-FFT
    grids = _pack_gpu(w_flat, grid_idx, nP)
    w_hat = torch.fft.fftn(grids, dim=(1, 2, 3))

    # Pointwise 9x9 multiply: y_hat[i] = sum_j kernel_hat[i,j] * w_hat[j]
    y_hat = torch.einsum("ijxyz,jxyz->ixyz", kernel_hat, w_hat)

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

    # Step 2: Transfer to GPU
    kernel_hat_gpu = to_torch(kernel_hat_np, device, dtype)
    grid_idx_gpu = torch.from_numpy(grid_idx_np.copy()).long().to(device)

    # Step 3: Build incident field on CPU
    psi_inc = _build_incident_field_coupled(
        centres, omega, ref, k_hat=k_hat, wave_type=wave_type
    )

    # Step 4: Solve 9 RHS columns via GPU GMRES
    dim = 9 * nC
    psi_exc = np.zeros((dim, 9), dtype=complex)

    def matvec(w: torch.Tensor) -> torch.Tensor:
        return _matvec_fft_gpu(w, kernel_hat_gpu, grid_idx_gpu, nP, nC)

    for col in range(9):
        rhs_np = psi_inc[:, col]
        b = to_torch(rhs_np, device, dtype)

        if initial_guess == "born":
            x0 = b.clone()
        else:
            x0 = None

        x, n_iter, rel_res = torch_gmres(
            matvec, b, x0=x0, tol=gmres_tol, maxiter=gmres_maxiter
        )

        if rel_res > gmres_tol:
            import warnings

            warnings.warn(
                f"GPU GMRES did not converge for column {col}: "
                f"rel_res={rel_res:.2e} after {n_iter} iterations",
                UserWarning,
                stacklevel=2,
            )

        psi_exc[:, col] = to_numpy(x)

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
    )
