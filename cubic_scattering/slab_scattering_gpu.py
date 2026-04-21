"""GPU-accelerated 3D Slab Foldy-Lax solver.

PyTorch port of slab_scattering.py. The key algorithmic improvement is
replacing the O(N_z²) layer-pair loop with z-FFT convolution (the kernel
is Toeplitz in z), giving a full 3D FFT convolution matvec.

CPU code builds the T-matrices, kernels, and incident field.
GPU handles the GMRES iterations via 3D FFT convolution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from .effective_contrasts import ReferenceMedium

from .slab_scattering import (
    SlabGeometry,
    SlabMaterial,
    SlabResult,
    _build_slab_incident_field,
    _build_slab_kernels,
    compute_slab_tmatrices,
)
from .torch_gmres import get_device, select_dtype, to_numpy, to_torch, torch_gmres

__all__ = ["compute_slab_scattering_gpu"]


def _build_slab_kernels_gpu(
    geometry: SlabGeometry,
    omega: float,
    ref: ReferenceMedium,
    device: torch.device,
    dtype: torch.dtype,
    *,
    volume_averaged: bool = False,
    n_orders: int = 2,
) -> torch.Tensor:
    """Build 3D FFT kernel on GPU from CPU kernel.

    The CPU kernel is xy-FFT'd with shape (n_dz, S, S, 9, 9).
    We IFFT in xy to get spatial, then do full 3D FFT on GPU.

    The kernel is Toeplitz in z — index by dz = m - n + (N_z - 1).
    We rearrange to (9, 9, n_dz, S, S) for batched 3D FFT.

    Args:
        geometry: Slab lattice geometry.
        omega: Angular frequency (rad/s).
        ref: Background elastic medium.
        device: Target torch device.
        dtype: Target complex dtype.
        volume_averaged: If True, use volume-averaged inter-voxel propagator
            for nearest-neighbour separations.
        n_orders: Dynamic correction orders for volume-averaged propagator.

    Returns:
        kernel_hat_3d: shape (9, 9, n_dz, S, S), complex, on device.
            3D FFT of the full convolution kernel.
    """
    M, N_z = geometry.M, geometry.N_z
    S = 2 * M - 1
    n_dz = 2 * N_z - 1

    # Build kernel on CPU — returns shape (n_dz, S, S, 9, 9) already xy-FFT'd
    kernel_hat_xy = _build_slab_kernels(
        geometry,
        omega,
        ref,
        volume_averaged=volume_averaged,
        n_orders=n_orders,
    )

    # IFFT in xy to recover spatial domain
    kernel_spatial = np.fft.ifft2(kernel_hat_xy, axes=(1, 2))

    # Rearrange to (9, 9, n_dz, S, S) for batched 3D FFT
    kernel_rearranged = kernel_spatial.transpose(3, 4, 0, 1, 2).copy()

    # Transfer to GPU and do 3D FFT
    kernel_gpu = to_torch(kernel_rearranged, device, dtype)
    kernel_hat_3d = torch.fft.fftn(kernel_gpu, dim=(2, 3, 4))

    return kernel_hat_3d


def _slab_matvec_gpu(
    psi_flat: torch.Tensor,
    T_local: torch.Tensor,
    kernel_hat_3d: torch.Tensor,
    M: int,
    N_z: int,
) -> torch.Tensor:
    """Compute (I - G*T)*psi using full 3D FFT convolution on GPU.

    Args:
        psi_flat: Exciting field, flat tensor of length N_z * M * M * 9.
        T_local: Per-cube T-matrices, shape (N_z, M, M, 9, 9).
        kernel_hat_3d: 3D FFT of kernel, shape (9, 9, n_dz, S, S).
        M: Horizontal grid size.
        N_z: Number of vertical layers.

    Returns:
        Result of (I - G*T)*psi, flat tensor.
    """
    S = 2 * M - 1
    n_dz = 2 * N_z - 1

    psi = psi_flat.reshape(N_z, M, M, 9)

    # T-multiply: tau[l,i,j,:] = T[l,i,j,:,:] @ psi[l,i,j,:]
    tau = torch.einsum("lmnab,lmnb->lmna", T_local, psi)

    # Rearrange to (9, N_z, M, M) and zero-pad to (9, n_dz, S, S)
    tau_perm = tau.permute(3, 0, 1, 2)  # (9, N_z, M, M)

    tau_pad = torch.zeros(9, n_dz, S, S, device=psi_flat.device, dtype=psi_flat.dtype)
    tau_pad[:, :N_z, :M, :M] = tau_perm

    # 3D FFT
    tau_hat = torch.fft.fftn(tau_pad, dim=(1, 2, 3))

    # Pointwise 9x9 multiply in frequency domain
    acc_hat = torch.einsum("ijzxy,jzxy->izxy", kernel_hat_3d, tau_hat)

    # 3D IFFT
    acc_full = torch.fft.ifftn(acc_hat, dim=(1, 2, 3))

    # Extract valid (alias-free) region
    # z: linear convolution of N_z with n_dz kernel, valid at [N_z-1 : 2*N_z-1]
    # xy: circular convolution valid at [M-1 : S]
    acc_valid = acc_full[:, N_z - 1 : 2 * N_z - 1, M - 1 : S, M - 1 : S]

    # Rearrange back to (N_z, M, M, 9)
    acc_out = acc_valid.permute(1, 2, 3, 0)

    return (psi - acc_out).reshape(-1)


def compute_slab_scattering_gpu(
    geometry: SlabGeometry,
    material: SlabMaterial,
    omega: float,
    k_hat: np.ndarray,
    wave_type: str = "P",
    gmres_tol: float = 1e-6,
    max_iter: int = 500,
    initial_guess: str = "born",
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    *,
    volume_averaged: bool = False,
    n_orders: int = 2,
) -> SlabResult:
    """Solve the Foldy-Lax slab scattering problem via GPU GMRES.

    Solves (I - G*T)*psi = psi0 with 3D FFT convolution matvec on GPU.
    Replaces the O(N_z²) layer-pair loop with z-direction FFT.

    Args:
        geometry: Slab lattice geometry.
        material: Per-cube material contrasts.
        omega: Angular frequency (rad/s).
        k_hat: Unit incident propagation direction (z, x, y).
        wave_type: 'P' or 'S'.
        gmres_tol: GMRES relative tolerance.
        max_iter: Maximum GMRES iterations.
        initial_guess: "born" or "zero".
        device: PyTorch device (default: auto-detect).
        dtype: PyTorch complex dtype (default: auto-select).
        volume_averaged: If True, use volume-averaged inter-voxel propagator
            for nearest-neighbour separations.
        n_orders: Dynamic correction orders for volume-averaged propagator.

    Returns:
        SlabResult with exciting and incident fields.
    """
    if device is None:
        device = get_device()
    if dtype is None:
        dtype = select_dtype(device)

    M, N_z = geometry.M, geometry.N_z

    # Step 1: Build T-matrices, kernel, incident field on CPU
    T_local_np = compute_slab_tmatrices(geometry, material, omega)
    psi0_np = _build_slab_incident_field(
        geometry, omega, material.ref, k_hat, wave_type
    )

    # Build 3D FFT kernel on GPU
    kernel_hat_3d = _build_slab_kernels_gpu(
        geometry,
        omega,
        material.ref,
        device,
        dtype,
        volume_averaged=volume_averaged,
        n_orders=n_orders,
    )

    # Transfer T-matrices and incident field to GPU
    T_local_gpu = to_torch(T_local_np, device, dtype)
    b = to_torch(psi0_np.ravel(), device, dtype)

    # Step 2: Set up matvec
    def matvec(psi: torch.Tensor) -> torch.Tensor:
        return _slab_matvec_gpu(psi, T_local_gpu, kernel_hat_3d, M, N_z)

    # Step 3: Initial guess
    if initial_guess == "born":
        x0 = b.clone()
    else:
        x0 = None

    # Step 4: Solve via GPU GMRES
    x, n_iter, rel_res = torch_gmres(matvec, b, x0=x0, tol=gmres_tol, maxiter=max_iter)

    if rel_res > gmres_tol:
        import warnings

        warnings.warn(
            f"GPU GMRES did not converge: rel_res={rel_res:.2e} "
            f"after {n_iter} iterations",
            stacklevel=2,
        )

    # Step 5: Transfer solution back to CPU
    psi_np = to_numpy(x).reshape(N_z, M, M, 9)

    return SlabResult(
        psi=psi_np,
        psi0=psi0_np,
        geometry=geometry,
        material=material,
        omega=omega,
        k_hat=np.asarray(k_hat, dtype=float),
        wave_type=wave_type,
        n_gmres_iter=n_iter,
        gmres_residual=rel_res,
    )
