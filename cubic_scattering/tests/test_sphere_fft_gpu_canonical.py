"""The point solver's FFT kernel stored at the canonical k only (the cube group in Fourier space).

K^(Q k) = S K^(k) S^T for the 48 signed permutations (S the signed permutation of the 9 components), with
the cube's T-matrix folded in, since S T S^T = T. The product through the canonical kernel must equal the
full point-major product to round-off in both precisions, the matvec must equal the NumPy one, and the
solver must take this path.
"""

from math import comb

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose

from cubic_scattering import sphere_scattering_fft_gpu as G
from cubic_scattering.effective_contrasts import compute_cube_tmatrix
from cubic_scattering.resonance_tmatrix import _sub_cell_tmatrix_9x9
from cubic_scattering.sphere_scattering_fft import _build_fft_kernel, _build_grid_index_map, _matvec_fft
from cubic_scattering.tests.test_sphere_scattering_fft_gpu import CONTRAST, REF
from cubic_scattering.torch_gmres import to_numpy, to_torch

CPU = torch.device("cpu")


def _setup(n_sub: int, radius: float = 10.0):
    grid_idx, centres, a_sub = _build_grid_index_map(radius, n_sub)
    omega = 0.8 * REF.beta / a_sub  # k_S a_sub = 0.8: the kernel far from trivial
    t_loc = _sub_cell_tmatrix_9x9(compute_cube_tmatrix(omega, a_sub, REF, CONTRAST), omega, a_sub)
    return grid_idx, len(centres), _build_fft_kernel(n_sub, a_sub, t_loc, omega, REF)


def test_only_the_canonical_k_are_stored() -> None:
    n_sub = 5
    _, _, kernel_hat = _setup(n_sub)
    ck = G._canonical_kernel(kernel_hat, n_sub, CPU, torch.complex128)
    assert ck.kc.shape == (comb(n_sub + 2, 3), 9, 9)  # 35 of the 729 points


@pytest.mark.parametrize("backend", ["fortran", "python"])  # the compiled route, and the gathers a GPU uses
@pytest.mark.parametrize(("dtype", "tol"), [(torch.complex128, 1e-13), (torch.complex64, 1e-5)])
def test_the_canonical_product_equals_the_full_product(monkeypatch, backend, dtype, tol) -> None:
    from cubic_scattering.graded_voxel import kernel as KN

    monkeypatch.setattr(KN, "point_kernel_backend", lambda: backend)
    n_sub = 5
    _, _, kernel_hat = _setup(n_sub)
    rng = np.random.default_rng(7)
    f = (2 * n_sub - 1) ** 3
    w_pm = to_torch(rng.normal(size=(f, 9)) + 1j * rng.normal(size=(f, 9)), CPU, dtype)
    full = G._point_major(to_torch(kernel_hat, CPU, dtype)).reshape(f, 9, 9)
    want = to_numpy(torch.matmul(full, w_pm.unsqueeze(-1)).squeeze(-1))
    ck = G._canonical_kernel(kernel_hat, n_sub, CPU, dtype)
    assert ck.compiled == (backend == "fortran")
    got_t = G._apply_canonical(ck, w_pm)
    assert got_t.dtype == dtype
    got = to_numpy(got_t)
    assert got.shape == (f, 9)
    assert np.abs(got - want).max() <= tol * np.abs(want).max()


@pytest.mark.parametrize("n_sub", [3, 5])
def test_the_canonical_matvec_equals_the_numpy_matvec(n_sub) -> None:
    grid_idx, nc, kernel_hat = _setup(n_sub)
    rng = np.random.default_rng(42)
    w = rng.standard_normal(9 * nc) + 1j * rng.standard_normal(9 * nc)
    want = _matvec_fft(w, kernel_hat, grid_idx, 2 * n_sub - 1, nc)
    ck = G._canonical_kernel(kernel_hat, n_sub, CPU, torch.complex128)
    gi = torch.from_numpy(grid_idx.copy()).long()
    got = G._matvec_fft_canonical(to_torch(w, CPU, torch.complex128), ck, gi, 2 * n_sub - 1, nc)
    assert_allclose(to_numpy(got), want, rtol=1e-11, atol=1e-11 * np.abs(want).max())


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_the_canonical_matvec_on_mps() -> None:
    n_sub = 5
    grid_idx, nc, kernel_hat = _setup(n_sub)
    rng = np.random.default_rng(42)
    w = rng.standard_normal(9 * nc) + 1j * rng.standard_normal(9 * nc)
    want = _matvec_fft(w, kernel_hat, grid_idx, 2 * n_sub - 1, nc)
    mps = torch.device("mps")
    ck = G._canonical_kernel(kernel_hat, n_sub, mps, torch.complex64)
    gi = torch.from_numpy(grid_idx.copy()).long().to(mps)
    got = G._matvec_fft_canonical(to_torch(w, mps, torch.complex64), ck, gi, 2 * n_sub - 1, nc)
    assert_allclose(to_numpy(got), want, rtol=1e-4, atol=1e-4 * np.abs(want).max())


def test_the_solver_takes_the_canonical_path(monkeypatch) -> None:
    built = []
    real = G._canonical_kernel
    monkeypatch.setattr(G, "_canonical_kernel", lambda *a, **k: built.append(1) or real(*a, **k))
    G.compute_sphere_foldy_lax_fft_gpu(
        omega=150.0,
        radius=0.5,
        ref=REF,
        contrast=CONTRAST,
        n_sub=3,
        wave_type="P",
        device=CPU,
        dtype=torch.complex128,
    )
    assert built == [1]
