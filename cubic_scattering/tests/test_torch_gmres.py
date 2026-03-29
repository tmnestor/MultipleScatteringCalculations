"""Tests for the PyTorch GMRES solver and device helpers."""

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose

from cubic_scattering.torch_gmres import (
    _complex_norm,
    get_device,
    select_dtype,
    to_numpy,
    to_torch,
    torch_gmres,
)

# ── Device/dtype helpers ─────────────────────────────────────────


class TestDeviceHelpers:
    def test_get_device_returns_valid(self):
        device = get_device()
        assert device.type in ("mps", "cuda", "cpu")

    def test_select_dtype_cpu(self):
        dtype = select_dtype(torch.device("cpu"))
        assert dtype == torch.complex128

    def test_select_dtype_cpu_prefer_double(self):
        dtype = select_dtype(torch.device("cpu"), prefer_double=True)
        assert dtype == torch.complex128

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="MPS not available"
    )
    def test_select_dtype_mps(self):
        dtype = select_dtype(torch.device("mps"))
        assert dtype == torch.complex64

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="MPS not available"
    )
    def test_select_dtype_mps_prefer_double(self):
        dtype = select_dtype(torch.device("mps"), prefer_double=True)
        assert dtype == torch.complex128


class TestConversion:
    def test_to_torch_and_back(self):
        arr = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128)
        device = torch.device("cpu")
        t = to_torch(arr, device, torch.complex128)
        assert t.device.type == "cpu"
        assert t.dtype == torch.complex128
        result = to_numpy(t)
        assert_allclose(result, arr)

    def test_to_torch_float32(self):
        arr = np.array([1 + 2j, 3 + 4j], dtype=np.complex128)
        device = torch.device("cpu")
        t = to_torch(arr, device, torch.complex64)
        assert t.dtype == torch.complex64
        result = to_numpy(t)
        assert_allclose(result, arr, rtol=1e-6)


class TestComplexNorm:
    def test_norm_real(self):
        x = torch.tensor([3.0 + 0j, 4.0 + 0j])
        assert_allclose(float(_complex_norm(x)), 5.0, rtol=1e-10)

    def test_norm_complex(self):
        x = torch.tensor([1.0 + 1j, 1.0 + 1j])
        expected = np.sqrt(4.0)  # 2*|1+i|^2 = 4
        assert_allclose(float(_complex_norm(x)), expected, rtol=1e-10)


# ── GMRES solver ─────────────────────────────────────────────


class TestTorchGMRES:
    def test_identity_system(self):
        """I @ x = b should converge immediately."""
        b = torch.tensor([1.0 + 0j, 2.0 + 0j, 3.0 + 0j], dtype=torch.complex128)
        x, n_iter, rel_res = torch_gmres(lambda x: x, b, tol=1e-10)
        assert_allclose(to_numpy(x), to_numpy(b), rtol=1e-9)
        assert n_iter <= 1
        assert rel_res < 1e-10

    def test_diagonal_system(self):
        """Diagonal system should converge quickly."""
        d = torch.tensor([2.0 + 0j, 3.0 + 0j, 5.0 + 0j], dtype=torch.complex128)
        b = torch.tensor([4.0 + 0j, 9.0 + 0j, 25.0 + 0j], dtype=torch.complex128)
        expected = b / d

        def matvec(x):
            return d * x

        x, n_iter, rel_res = torch_gmres(matvec, b, tol=1e-10)
        assert_allclose(to_numpy(x), to_numpy(expected), rtol=1e-8)
        assert rel_res < 1e-10

    def test_dense_spd_system(self):
        """Dense symmetric positive-definite system."""
        rng = np.random.default_rng(42)
        n = 20
        A_np = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        A_np = A_np @ A_np.conj().T + 5 * np.eye(n)  # make SPD
        b_np = rng.standard_normal(n) + 1j * rng.standard_normal(n)

        x_exact = np.linalg.solve(A_np, b_np)

        A_t = torch.from_numpy(A_np)
        b_t = torch.from_numpy(b_np)

        def matvec(x):
            return A_t @ x

        x, n_iter, rel_res = torch_gmres(matvec, b_t, tol=1e-10, maxiter=50)
        assert_allclose(to_numpy(x), x_exact, rtol=1e-6)
        assert rel_res < 1e-8

    def test_tolerance_respected(self):
        """Loose tolerance should converge faster."""
        rng = np.random.default_rng(123)
        n = 30
        A_np = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        A_np = A_np @ A_np.conj().T + 10 * np.eye(n)
        b_np = rng.standard_normal(n) + 1j * rng.standard_normal(n)

        A_t = torch.from_numpy(A_np)
        b_t = torch.from_numpy(b_np)

        def matvec(x):
            return A_t @ x

        _, n_tight, _ = torch_gmres(matvec, b_t, tol=1e-10, maxiter=100)
        _, n_loose, _ = torch_gmres(matvec, b_t, tol=1e-3, maxiter=100)

        assert n_loose <= n_tight

    def test_initial_guess(self):
        """Providing a good initial guess should help."""
        d = torch.tensor([1.0 + 0j, 2.0 + 0j, 3.0 + 0j], dtype=torch.complex128)
        b = torch.tensor([1.0 + 0j, 4.0 + 0j, 9.0 + 0j], dtype=torch.complex128)

        # Exact solution as initial guess
        x0 = b / d

        def matvec(x):
            return d * x

        x, n_iter, rel_res = torch_gmres(matvec, b, x0=x0, tol=1e-10)
        assert n_iter == 0
        assert rel_res < 1e-10

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="MPS not available"
    )
    def test_mps_device(self):
        """GMRES should work on MPS with complex64."""
        device = torch.device("mps")
        d = torch.tensor(
            [2.0 + 0j, 3.0 + 0j, 5.0 + 0j], device=device, dtype=torch.complex64
        )
        b = torch.tensor(
            [4.0 + 0j, 9.0 + 0j, 25.0 + 0j], device=device, dtype=torch.complex64
        )
        expected = b / d

        def matvec(x):
            return d * x

        x, n_iter, rel_res = torch_gmres(matvec, b, tol=1e-5)
        assert x.device.type == "mps"
        assert_allclose(to_numpy(x), to_numpy(expected), rtol=1e-4)
