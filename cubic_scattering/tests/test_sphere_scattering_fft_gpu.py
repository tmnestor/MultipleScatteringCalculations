"""Tests for the GPU-accelerated sphere Foldy-Lax FFT solver."""

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose

from cubic_scattering.effective_contrasts import (
    MaterialContrast,
    ReferenceMedium,
    compute_cube_tmatrix,
)
from cubic_scattering.resonance_tmatrix import _sub_cell_tmatrix_9x9
from cubic_scattering.sphere_scattering_fft import (
    _build_fft_kernel,
    _build_grid_index_map,
    _matvec_fft,
    _pack,
    compute_sphere_foldy_lax_fft,
)
from cubic_scattering.sphere_scattering_fft_gpu import (
    _matvec_fft_gpu,
    _pack_gpu,
    _unpack_gpu,
    compute_sphere_foldy_lax_fft_gpu,
)
from cubic_scattering.torch_gmres import to_numpy, to_torch

REF = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
CONTRAST = MaterialContrast(Dlambda=2e9, Dmu=1e9, Drho=100.0)
OMEGA = 0.05 * REF.beta / 1.0  # ka=0.05 at a=1
RADIUS = 0.5
N_SUB = 3


@pytest.fixture
def sphere_setup():
    """Build grid index map and kernel for tests."""
    grid_idx, centres, a_sub = _build_grid_index_map(RADIUS, N_SUB)
    nC = len(centres)
    nP = 2 * N_SUB - 1
    rayleigh_sub = compute_cube_tmatrix(OMEGA, a_sub, REF, CONTRAST)
    T_loc = _sub_cell_tmatrix_9x9(rayleigh_sub, OMEGA, a_sub)
    kernel_hat = _build_fft_kernel(N_SUB, a_sub, T_loc, OMEGA, REF)
    return grid_idx, centres, a_sub, nC, nP, T_loc, kernel_hat


class TestPackUnpackGPU:
    def test_roundtrip(self, sphere_setup):
        """Pack then unpack should recover original data."""
        grid_idx, _, _, nC, nP, _, _ = sphere_setup
        rng = np.random.default_rng(42)
        w_np = rng.standard_normal(9 * nC) + 1j * rng.standard_normal(9 * nC)

        device = torch.device("cpu")
        dtype = torch.complex128
        w_t = to_torch(w_np, device, dtype)
        gi_t = torch.from_numpy(grid_idx.copy()).long()

        grids = _pack_gpu(w_t, gi_t, nP)
        w_back = _unpack_gpu(grids, gi_t, nC)

        assert_allclose(to_numpy(w_back), w_np, rtol=1e-12)

    def test_matches_cpu(self, sphere_setup):
        """GPU pack/unpack should match CPU version."""
        grid_idx, _, _, nC, nP, _, _ = sphere_setup
        rng = np.random.default_rng(42)
        w_np = rng.standard_normal(9 * nC) + 1j * rng.standard_normal(9 * nC)

        # CPU
        grids_cpu = _pack(w_np, grid_idx, nP)

        # GPU (on CPU device for testing)
        device = torch.device("cpu")
        dtype = torch.complex128
        w_t = to_torch(w_np, device, dtype)
        gi_t = torch.from_numpy(grid_idx.copy()).long()
        grids_gpu = _pack_gpu(w_t, gi_t, nP)

        assert_allclose(to_numpy(grids_gpu), grids_cpu, rtol=1e-12)


class TestMatvecGPU:
    def test_matches_cpu(self, sphere_setup):
        """GPU matvec should match CPU matvec."""
        grid_idx, _, _, nC, nP, _, kernel_hat = sphere_setup
        rng = np.random.default_rng(42)
        w_np = rng.standard_normal(9 * nC) + 1j * rng.standard_normal(9 * nC)

        # CPU reference
        result_cpu = _matvec_fft(w_np, kernel_hat, grid_idx, nP, nC)

        # GPU (on CPU device)
        device = torch.device("cpu")
        dtype = torch.complex128
        w_t = to_torch(w_np, device, dtype)
        k_t = to_torch(kernel_hat, device, dtype)
        gi_t = torch.from_numpy(grid_idx.copy()).long()

        result_gpu = _matvec_fft_gpu(w_t, k_t, gi_t, nP, nC)

        assert_allclose(to_numpy(result_gpu), result_cpu, rtol=1e-10)

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="MPS not available"
    )
    def test_matches_cpu_on_mps(self, sphere_setup):
        """GPU matvec on MPS should match CPU (float32 tolerance)."""
        grid_idx, _, _, nC, nP, _, kernel_hat = sphere_setup
        rng = np.random.default_rng(42)
        w_np = rng.standard_normal(9 * nC) + 1j * rng.standard_normal(9 * nC)

        result_cpu = _matvec_fft(w_np, kernel_hat, grid_idx, nP, nC)

        device = torch.device("mps")
        dtype = torch.complex64
        w_t = to_torch(w_np, device, dtype)
        k_t = to_torch(kernel_hat, device, dtype)
        gi_t = torch.from_numpy(grid_idx.copy()).long().to(device)

        result_gpu = _matvec_fft_gpu(w_t, k_t, gi_t, nP, nC)

        assert_allclose(to_numpy(result_gpu), result_cpu, rtol=1e-4)


class TestSphereGPUSolver:
    def test_matches_cpu_solution(self):
        """GPU solver should match CPU solver for small problem."""
        result_cpu = compute_sphere_foldy_lax_fft(
            omega=OMEGA,
            radius=RADIUS,
            ref=REF,
            contrast=CONTRAST,
            n_sub=N_SUB,
            wave_type="P",
            gmres_tol=1e-8,
            gmres_maxiter=200,
        )

        result_gpu = compute_sphere_foldy_lax_fft_gpu(
            omega=OMEGA,
            radius=RADIUS,
            ref=REF,
            contrast=CONTRAST,
            n_sub=N_SUB,
            wave_type="P",
            gmres_tol=1e-8,
            gmres_maxiter=200,
            device=torch.device("cpu"),
            dtype=torch.complex128,
        )

        # Use atol scaled to max element: off-diag entries are near-zero noise
        scale = np.abs(result_cpu.T_comp_9x9).max()
        assert_allclose(
            result_gpu.T_comp_9x9,
            result_cpu.T_comp_9x9,
            rtol=1e-6,
            atol=scale * 1e-6,
        )
        assert_allclose(result_gpu.T3x3, result_cpu.T3x3, rtol=1e-6, atol=scale * 1e-6)

    def test_zero_contrast(self):
        """Zero contrast should give T=0."""
        zero = MaterialContrast(Dlambda=0.0, Dmu=0.0, Drho=0.0)
        result = compute_sphere_foldy_lax_fft_gpu(
            omega=OMEGA,
            radius=RADIUS,
            ref=REF,
            contrast=zero,
            n_sub=N_SUB,
            wave_type="P",
            device=torch.device("cpu"),
            dtype=torch.complex128,
        )
        assert_allclose(result.T_comp_9x9, 0.0, atol=1e-20)
