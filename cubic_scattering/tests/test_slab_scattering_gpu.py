"""Tests for the GPU-accelerated slab Foldy-Lax solver."""

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose

from cubic_scattering.effective_contrasts import MaterialContrast, ReferenceMedium
from cubic_scattering.slab_scattering import (
    SlabGeometry,
    _build_slab_incident_field,
    _build_slab_kernels,
    _slab_matvec,
    compute_slab_scattering,
    compute_slab_tmatrices,
    uniform_slab_material,
)
from cubic_scattering.slab_scattering_gpu import (
    _build_slab_kernels_gpu,
    _slab_matvec_gpu,
    compute_slab_scattering_gpu,
)
from cubic_scattering.torch_gmres import to_numpy, to_torch

REF = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
CONTRAST = MaterialContrast(Dlambda=2e9, Dmu=1e9, Drho=100.0)
A = 1.0
OMEGA = 0.05 * REF.beta / A  # ka=0.05
K_HAT = np.array([1.0, 0.0, 0.0])


class TestSlabMatvecGPU:
    """Test GPU matvec matches CPU matvec."""

    @pytest.fixture
    def small_slab(self):
        """Small 2x2x2 slab for matvec testing."""
        geom = SlabGeometry(M=2, N_z=2, a=A)
        material = uniform_slab_material(geom, REF, CONTRAST)
        T_local = compute_slab_tmatrices(geom, material, OMEGA)
        kernel_hat = _build_slab_kernels(geom, OMEGA, REF)
        psi0 = _build_slab_incident_field(geom, OMEGA, REF, K_HAT, "P")
        return geom, T_local, kernel_hat, psi0

    def test_matvec_matches_cpu(self, small_slab):
        """GPU matvec should produce same result as CPU matvec."""
        geom, T_local, kernel_hat_xy, psi0 = small_slab
        M, N_z = geom.M, geom.N_z
        psi_flat = psi0.ravel()

        # CPU reference
        result_cpu = _slab_matvec(psi_flat, T_local, kernel_hat_xy, geom)

        # GPU (on CPU device)
        device = torch.device("cpu")
        dtype = torch.complex128

        kernel_hat_3d = _build_slab_kernels_gpu(geom, OMEGA, REF, device, dtype)
        T_gpu = to_torch(T_local, device, dtype)
        psi_gpu = to_torch(psi_flat, device, dtype)

        result_gpu = _slab_matvec_gpu(psi_gpu, T_gpu, kernel_hat_3d, M, N_z)

        assert_allclose(to_numpy(result_gpu), result_cpu, rtol=1e-10)

    def test_matvec_random_input(self, small_slab):
        """GPU matvec with random input matches CPU."""
        geom, T_local, kernel_hat_xy, _ = small_slab
        M, N_z = geom.M, geom.N_z
        rng = np.random.default_rng(42)
        n = N_z * M * M * 9
        psi_flat = rng.standard_normal(n) + 1j * rng.standard_normal(n)

        result_cpu = _slab_matvec(psi_flat, T_local, kernel_hat_xy, geom)

        device = torch.device("cpu")
        dtype = torch.complex128

        kernel_hat_3d = _build_slab_kernels_gpu(geom, OMEGA, REF, device, dtype)
        T_gpu = to_torch(T_local, device, dtype)
        psi_gpu = to_torch(psi_flat, device, dtype)

        result_gpu = _slab_matvec_gpu(psi_gpu, T_gpu, kernel_hat_3d, M, N_z)

        assert_allclose(to_numpy(result_gpu), result_cpu, rtol=1e-10)

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="MPS not available"
    )
    def test_matvec_mps_float32(self, small_slab):
        """GPU matvec on MPS with float32 should match CPU (relaxed tol)."""
        geom, T_local, kernel_hat_xy, psi0 = small_slab
        M, N_z = geom.M, geom.N_z
        psi_flat = psi0.ravel()

        result_cpu = _slab_matvec(psi_flat, T_local, kernel_hat_xy, geom)

        device = torch.device("mps")
        dtype = torch.complex64

        kernel_hat_3d = _build_slab_kernels_gpu(geom, OMEGA, REF, device, dtype)
        T_gpu = to_torch(T_local, device, dtype)
        psi_gpu = to_torch(psi_flat, device, dtype)

        result_gpu = _slab_matvec_gpu(psi_gpu, T_gpu, kernel_hat_3d, M, N_z)

        assert_allclose(to_numpy(result_gpu), result_cpu, rtol=1e-4)


class TestSlabGPUSolver:
    """Test full GPU solver matches CPU solver."""

    def test_matches_cpu_solution(self):
        """GPU solution should match CPU for small slab."""
        geom = SlabGeometry(M=2, N_z=2, a=A)
        material = uniform_slab_material(geom, REF, CONTRAST)

        result_cpu = compute_slab_scattering(
            geom,
            material,
            OMEGA,
            K_HAT,
            "P",
            gmres_tol=1e-8,
            max_iter=500,
        )

        result_gpu = compute_slab_scattering_gpu(
            geom,
            material,
            OMEGA,
            K_HAT,
            "P",
            gmres_tol=1e-8,
            max_iter=500,
            device=torch.device("cpu"),
            dtype=torch.complex128,
        )

        assert_allclose(result_gpu.psi, result_cpu.psi, rtol=1e-6)
        assert result_gpu.gmres_residual < 1e-7

    def test_zero_contrast(self):
        """Zero contrast: psi should equal psi0."""
        geom = SlabGeometry(M=2, N_z=2, a=A)
        zero = MaterialContrast(Dlambda=0.0, Dmu=0.0, Drho=0.0)
        material = uniform_slab_material(geom, REF, zero)

        result = compute_slab_scattering_gpu(
            geom,
            material,
            OMEGA,
            K_HAT,
            "P",
            device=torch.device("cpu"),
            dtype=torch.complex128,
        )

        assert_allclose(result.psi, result.psi0, atol=1e-12)

    def test_born_scaling(self):
        """Born approximation: weak contrast result should scale linearly."""
        geom = SlabGeometry(M=2, N_z=2, a=A)
        c1 = MaterialContrast(Dlambda=1e5, Dmu=5e4, Drho=0.01)
        c2 = MaterialContrast(Dlambda=2e5, Dmu=1e5, Drho=0.02)

        m1 = uniform_slab_material(geom, REF, c1)
        m2 = uniform_slab_material(geom, REF, c2)

        device = torch.device("cpu")
        dtype = torch.complex128

        r1 = compute_slab_scattering_gpu(
            geom,
            m1,
            OMEGA,
            K_HAT,
            "P",
            gmres_tol=1e-10,
            max_iter=500,
            device=device,
            dtype=dtype,
        )
        r2 = compute_slab_scattering_gpu(
            geom,
            m2,
            OMEGA,
            K_HAT,
            "P",
            gmres_tol=1e-10,
            max_iter=500,
            device=device,
            dtype=dtype,
        )

        # In Born approx, scattered field scales linearly
        scat1 = r1.psi - r1.psi0
        scat2 = r2.psi - r2.psi0
        ratio = scat2 / (scat1 + 1e-30)

        # Should be approximately 2.0 (linear scaling)
        mask = np.abs(scat1) > 1e-15
        if np.any(mask):
            assert_allclose(np.abs(ratio[mask]), 2.0, rtol=0.1)
