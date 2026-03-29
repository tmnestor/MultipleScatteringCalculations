"""Shared GPU backend: device helpers and GMRES solver.

Provides PyTorch-based GMRES with Arnoldi + Givens rotations,
entirely on-device (no CPU sync per iteration). Device selection
follows the Kennett reflectivity convention: MPS > CUDA > CPU.

Dtype strategy:
    - MPS: complex64 (fast, exp(ikr) phases are O(1), no underflow)
    - CUDA/CPU: complex128 (safe default)
"""

from collections.abc import Callable

import numpy as np
import torch
from numpy.typing import NDArray

__all__ = [
    "get_device",
    "select_dtype",
    "to_numpy",
    "to_torch",
    "torch_gmres",
]


def get_device() -> torch.device:
    """Select the best available PyTorch device.

    Priority: MPS (Apple Silicon) > CUDA > CPU.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def select_dtype(
    device: torch.device,
    prefer_double: bool = False,
) -> torch.dtype:
    """Select complex dtype appropriate for device.

    Args:
        device: Target torch device.
        prefer_double: Force complex128 even on GPU.

    Returns:
        complex64 for MPS (unless prefer_double), complex128 otherwise.
    """
    if prefer_double:
        return torch.complex128
    if device.type == "mps":
        return torch.complex64
    return torch.complex128


def to_torch(
    arr: NDArray[np.complexfloating],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert NumPy complex array to torch tensor on device."""
    t = torch.from_numpy(np.ascontiguousarray(arr))
    return t.to(device=device, dtype=dtype)


def to_numpy(tensor: torch.Tensor) -> NDArray[np.complexfloating]:
    """Convert torch tensor to NumPy complex128 array."""
    return tensor.detach().cpu().to(torch.complex128).numpy()


def _complex_norm(x: torch.Tensor) -> torch.Tensor:
    """Compute norm of complex tensor (MPS-safe).

    MPS may not support torch.linalg.norm for complex tensors,
    so we compute sqrt(sum(re^2 + im^2)) explicitly.
    """
    return torch.sqrt(torch.sum(x.real**2 + x.imag**2))


def torch_gmres(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    x0: torch.Tensor | None = None,
    tol: float = 1e-6,
    maxiter: int = 500,
) -> tuple[torch.Tensor, int, float]:
    """GMRES solver entirely on-device (Arnoldi + Givens rotations).

    Solves A @ x = b where A is defined implicitly by matvec.

    Args:
        matvec: Function computing A @ x.
        b: Right-hand side vector.
        x0: Initial guess (default: zero).
        tol: Relative residual tolerance.
        maxiter: Maximum number of iterations.

    Returns:
        (x, n_iter, rel_residual) — solution, iteration count,
        final relative residual.
    """
    n = b.shape[0]
    device = b.device
    dtype = b.dtype

    if x0 is None:
        x0 = torch.zeros(n, device=device, dtype=dtype)

    # Initial residual
    r = b - matvec(x0)
    b_norm = _complex_norm(b)
    if b_norm < 1e-30:
        return x0.clone(), 0, 0.0

    r_norm = _complex_norm(r)
    if r_norm / b_norm < tol:
        return x0.clone(), 0, float((r_norm / b_norm).cpu())

    # Arnoldi basis vectors
    V = torch.zeros(maxiter + 1, n, device=device, dtype=dtype)
    V[0] = r / r_norm

    # Upper Hessenberg matrix
    H = torch.zeros(maxiter + 1, maxiter, device=device, dtype=dtype)

    # Givens rotation parameters
    cs = torch.zeros(maxiter, device=device, dtype=dtype)
    sn = torch.zeros(maxiter, device=device, dtype=dtype)

    # RHS for least-squares
    g = torch.zeros(maxiter + 1, device=device, dtype=dtype)
    g[0] = r_norm

    n_iter = 0
    for j in range(maxiter):
        n_iter = j + 1

        # Arnoldi step
        w = matvec(V[j])

        # Modified Gram-Schmidt
        for i in range(j + 1):
            H[i, j] = torch.sum(torch.conj(V[i]) * w)
            w = w - H[i, j] * V[i]

        H[j + 1, j] = _complex_norm(w)

        if H[j + 1, j].abs() < 1e-30:
            # Lucky breakdown
            V[j + 1] = torch.zeros(n, device=device, dtype=dtype)
        else:
            V[j + 1] = w / H[j + 1, j]

        # Apply previous Givens rotations to column j
        for i in range(j):
            temp = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
            H[i + 1, j] = -torch.conj(sn[i]) * H[i, j] + cs[i] * H[i + 1, j]
            H[i, j] = temp

        # Compute new Givens rotation
        a_val = H[j, j]
        b_val = H[j + 1, j]
        denom = torch.sqrt(a_val.abs() ** 2 + b_val.abs() ** 2)
        if denom.abs() < 1e-30:
            cs[j] = torch.ones(1, device=device, dtype=dtype).squeeze()
            sn[j] = torch.zeros(1, device=device, dtype=dtype).squeeze()
        else:
            cs[j] = a_val / denom
            sn[j] = b_val / denom

        # Apply rotation
        H[j, j] = cs[j] * a_val + sn[j] * b_val
        H[j + 1, j] = torch.zeros(1, device=device, dtype=dtype).squeeze()

        # Update g
        temp_g = cs[j] * g[j] + sn[j] * g[j + 1]
        g[j + 1] = -torch.conj(sn[j]) * g[j] + cs[j] * g[j + 1]
        g[j] = temp_g

        # Check convergence
        rel_res = g[j + 1].abs() / b_norm
        if rel_res < tol:
            break

    # Back-substitution
    y = torch.zeros(n_iter, device=device, dtype=dtype)
    for i in range(n_iter - 1, -1, -1):
        y[i] = g[i]
        for k in range(i + 1, n_iter):
            y[i] = y[i] - H[i, k] * y[k]
        y[i] = y[i] / H[i, i]

    # Construct solution
    x = x0.clone()
    for i in range(n_iter):
        x = x + y[i] * V[i]

    # Final relative residual
    r_final = b - matvec(x)
    rel_residual = float((_complex_norm(r_final) / b_norm).cpu())

    return x, n_iter, rel_residual
