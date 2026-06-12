"""Tests for the 3D slab Foldy-Lax multiple scattering solver."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from cubic_scattering.effective_contrasts import (
    MaterialContrast,
    ReferenceMedium,
    compute_cube_tmatrix,
)
from cubic_scattering.resonance_tmatrix import (
    _propagator_block_9x9,
    _sub_cell_tmatrix_9x9,
)
from cubic_scattering.slab_scattering import (
    SlabGeometry,
    SlabMaterial,
    _build_slab_incident_field,
    _build_slab_kernels,
    _slab_matvec,
    compute_slab_scattering,
    compute_slab_tmatrices,
    kennett_reference_matrix,
    kennett_reference_rpp,
    random_slab_material,
    slab_reflected_field,
    slab_reflection_matrix,
    slab_rpp_periodic,
    slab_weyl_amplitudes,
    uniform_slab_material,
)
from cubic_scattering.sphere_scattering import _plane_wave_strain_voigt

# ── Shared fixtures ──────────────────────────────────────────────

REF = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
CONTRAST = MaterialContrast(Dlambda=2e9, Dmu=1e9, Drho=100.0)
# Use scaled CONTRAST (not scaled reference!) to ensure non-zero impedance contrast.
# Scaling reference properties uniformly gives zero P-impedance contrast.
WEAK_CONTRAST = MaterialContrast(
    Dlambda=CONTRAST.Dlambda * 1e-4,
    Dmu=CONTRAST.Dmu * 1e-4,
    Drho=CONTRAST.Drho * 1e-4,
)
A = 1.0  # cube half-width
# ka = 0.05 → ω = 0.05 * β / a = 150 rad/s
OMEGA = 0.05 * REF.beta / A
K_HAT = np.array([1.0, 0.0, 0.0])  # vertical downgoing


# ── 1. TestSlabGeometry ──────────────────────────────────────────


class TestSlabGeometry:
    def test_basic_construction(self):
        g = SlabGeometry(M=4, N_z=3, a=1.0)
        assert g.M == 4
        assert g.N_z == 3
        assert g.a == 1.0

    def test_d_property(self):
        g = SlabGeometry(M=2, N_z=1, a=0.5)
        assert g.d == pytest.approx(1.0)

    def test_n_cubes(self):
        g = SlabGeometry(M=3, N_z=2, a=1.0)
        assert g.n_cubes == 3 * 3 * 2

    def test_cube_centre(self):
        g = SlabGeometry(M=3, N_z=2, a=1.0)
        # Layer 0, centre cube (i=1, j=1) → z=1.0, x=0, y=0
        c = g.cube_centre(0, 1, 1)
        assert_allclose(c, [1.0, 0.0, 0.0])
        # Layer 1, corner cube (i=0, j=0) → z=3.0, x=-2, y=-2
        c = g.cube_centre(1, 0, 0)
        assert_allclose(c, [3.0, -2.0, -2.0])

    def test_all_centres_shape(self):
        g = SlabGeometry(M=4, N_z=3, a=0.5)
        centres = g.all_centres()
        assert centres.shape == (3, 4, 4, 3)

    def test_all_centres_matches_cube_centre(self):
        g = SlabGeometry(M=3, N_z=2, a=1.0)
        centres = g.all_centres()
        for lz in range(g.N_z):
            for i in range(g.M):
                for j in range(g.M):
                    assert_allclose(centres[lz, i, j], g.cube_centre(lz, i, j))

    def test_reject_bad_M(self):
        with pytest.raises(ValueError, match="M must be >= 1"):
            SlabGeometry(M=0, N_z=1, a=1.0)

    def test_reject_bad_N_z(self):
        with pytest.raises(ValueError, match="N_z must be >= 1"):
            SlabGeometry(M=1, N_z=0, a=1.0)

    def test_reject_bad_a(self):
        with pytest.raises(ValueError, match="a must be > 0"):
            SlabGeometry(M=1, N_z=1, a=-1.0)


# ── 2. TestSlabMaterial ──────────────────────────────────────────


class TestSlabMaterial:
    def test_construction(self):
        shape = (2, 3, 3)
        mat = SlabMaterial(
            Dlambda=np.zeros(shape),
            Dmu=np.zeros(shape),
            Drho=np.zeros(shape),
            ref=REF,
        )
        assert mat.Dlambda.shape == shape

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="shapes must match"):
            SlabMaterial(
                Dlambda=np.zeros((2, 3, 3)),
                Dmu=np.zeros((2, 3, 4)),
                Drho=np.zeros((2, 3, 3)),
                ref=REF,
            )

    def test_wrong_ndim(self):
        with pytest.raises(ValueError, match="3D"):
            SlabMaterial(
                Dlambda=np.zeros((2, 3)),
                Dmu=np.zeros((2, 3)),
                Drho=np.zeros((2, 3)),
                ref=REF,
            )


# ── 3. TestSlabTMatrices ─────────────────────────────────────────


class TestSlabTMatrices:
    def test_matches_individual_computation(self):
        """Each cube's T-matrix matches compute_cube_tmatrix + _sub_cell_tmatrix_9x9."""
        geom = SlabGeometry(M=2, N_z=2, a=A)
        mat = uniform_slab_material(geom, REF, CONTRAST)
        T_all = compute_slab_tmatrices(geom, mat, OMEGA)

        assert T_all.shape == (2, 2, 2, 9, 9)

        # Compute reference T-matrix
        result = compute_cube_tmatrix(OMEGA, A, REF, CONTRAST)
        T_ref = _sub_cell_tmatrix_9x9(result, OMEGA, A)

        assert_allclose(T_all[0, 0, 0], T_ref, rtol=1e-12)
        assert_allclose(T_all[1, 1, 1], T_ref, rtol=1e-12)

    def test_zero_contrast_gives_zero_T(self):
        geom = SlabGeometry(M=2, N_z=1, a=A)
        zero = MaterialContrast(0.0, 0.0, 0.0)
        mat = uniform_slab_material(geom, REF, zero)
        T_all = compute_slab_tmatrices(geom, mat, OMEGA)
        assert_allclose(T_all, 0.0, atol=1e-30)

    def test_caching_for_binary(self):
        """Binary medium should only compute 2 unique T-matrices."""
        geom = SlabGeometry(M=4, N_z=2, a=A)
        mat = random_slab_material(geom, REF, CONTRAST, phi=0.5, seed=42)
        T_all = compute_slab_tmatrices(geom, mat, OMEGA)
        # All T-matrices should be either zero or the CONTRAST T-matrix
        T_inc = _sub_cell_tmatrix_9x9(
            compute_cube_tmatrix(OMEGA, A, REF, CONTRAST), OMEGA, A
        )
        for lz in range(geom.N_z):
            for i in range(geom.M):
                for j in range(geom.M):
                    T_cube = T_all[lz, i, j]
                    is_zero = np.allclose(T_cube, 0.0, atol=1e-30)
                    is_inc = np.allclose(T_cube, T_inc, rtol=1e-12)
                    assert is_zero or is_inc


# ── 4. TestKernelBuilding ────────────────────────────────────────


class TestKernelBuilding:
    def test_self_term_zero(self):
        """Kernel at (dz=0, dx=0, dy=0) should be zero (self-term excluded)."""
        geom = SlabGeometry(M=3, N_z=2, a=A)
        kernel_hat = _build_slab_kernels(geom, OMEGA, REF)
        # dz=0 → k = N_z-1 = 1
        # Recover spatial kernel via IFFT
        kernel_k1 = np.fft.ifft2(kernel_hat[1], axes=(0, 1))
        # Self-term is at index (M-1, M-1) = (2, 2)
        assert_allclose(kernel_k1[2, 2], 0.0, atol=1e-20)

    def test_spot_check_propagator(self):
        """Kernel at a specific offset matches direct _propagator_block_9x9."""
        geom = SlabGeometry(M=3, N_z=2, a=A)
        d = geom.d
        kernel_hat = _build_slab_kernels(geom, OMEGA, REF)
        # Check dz=d (k=2), dx=1, dy=0 → spatial index (1+2, 0+2) = (3, 2)
        kernel_k2 = np.fft.ifft2(kernel_hat[2], axes=(0, 1))
        G_direct = _propagator_block_9x9(np.array([d, d, 0.0]), OMEGA, REF)
        scale = np.max(np.abs(G_direct))
        assert_allclose(kernel_k2[3, 2], G_direct, atol=scale * 1e-10)

    def test_d4h_symmetry(self):
        """Kernel at (dx, dy) matches reflection/rotation of fundamental domain."""
        geom = SlabGeometry(M=3, N_z=2, a=A)
        kernel_hat = _build_slab_kernels(geom, OMEGA, REF)
        # dz=0 → k=1.  Check (dx=1, dy=0) vs (dx=-1, dy=0)
        kernel_k1 = np.fft.ifft2(kernel_hat[1], axes=(0, 1))
        G_pos = kernel_k1[1 + 2, 0 + 2]  # (dx=1, dy=0)
        G_neg = kernel_k1[-1 + 2, 0 + 2]  # (dx=-1, dy=0)
        from cubic_scattering.lattice_greens import _apply_refl_x

        expected = _apply_refl_x(G_pos)
        scale = np.max(np.abs(G_pos))
        assert_allclose(G_neg, expected, atol=scale * 1e-10)

    def test_kernel_shape(self):
        geom = SlabGeometry(M=4, N_z=3, a=A)
        kernel_hat = _build_slab_kernels(geom, OMEGA, REF)
        assert kernel_hat.shape == (5, 7, 7, 9, 9)


# ── 5. TestSlabMatvec ────────────────────────────────────────────


def _direct_matvec(psi_flat, T_local, geometry, omega, ref):
    """Reference non-FFT matvec for validation on small systems."""
    M, N_z = geometry.M, geometry.N_z
    psi = psi_flat.reshape(N_z, M, M, 9)
    tau = np.einsum("lmnab,lmnb->lmna", T_local, psi)

    centres = geometry.all_centres()
    acc = np.zeros_like(psi)

    for m in range(N_z):
        for i1 in range(M):
            for j1 in range(M):
                for n in range(N_z):
                    for i2 in range(M):
                        for j2 in range(M):
                            if m == n and i1 == i2 and j1 == j2:
                                continue
                            r_vec = centres[m, i1, j1] - centres[n, i2, j2]
                            P = _propagator_block_9x9(r_vec, omega, ref)
                            acc[m, i1, j1] += P @ tau[n, i2, j2]

    return (psi - acc).ravel()


class TestSlabMatvec:
    def test_identity_when_T_zero(self):
        """When T=0, matvec should be identity: (I - G·0)ψ = ψ."""
        geom = SlabGeometry(M=3, N_z=2, a=A)
        T_zero = np.zeros((2, 3, 3, 9, 9), dtype=complex)
        kernel_hat = _build_slab_kernels(geom, OMEGA, REF)
        psi = np.random.default_rng(42).standard_normal(2 * 3 * 3 * 9) + 0j
        result = _slab_matvec(psi, T_zero, kernel_hat, geom)
        assert_allclose(result, psi, atol=1e-12)

    def test_matches_direct_matvec(self):
        """FFT matvec matches direct dense matvec for M=2, N_z=2."""
        geom = SlabGeometry(M=2, N_z=2, a=A)
        mat = uniform_slab_material(geom, REF, CONTRAST)
        T_local = compute_slab_tmatrices(geom, mat, OMEGA)
        kernel_hat = _build_slab_kernels(geom, OMEGA, REF)

        rng = np.random.default_rng(123)
        psi = rng.standard_normal(geom.n_cubes * 9) + 1j * rng.standard_normal(
            geom.n_cubes * 9
        )

        fft_result = _slab_matvec(psi, T_local, kernel_hat, geom)
        direct_result = _direct_matvec(psi, T_local, geom, OMEGA, REF)
        assert_allclose(fft_result, direct_result, rtol=1e-10)


# ── 6. TestIncidentField ─────────────────────────────────────────


class TestIncidentField:
    def test_p_wave_vertical(self):
        """Vertical P-wave: pol=[1,0,0], phase=exp(ikPz)."""
        geom = SlabGeometry(M=2, N_z=3, a=A)
        psi0 = _build_slab_incident_field(geom, OMEGA, REF, K_HAT, "P")
        assert psi0.shape == (3, 2, 2, 9)

        kP = OMEGA / REF.alpha
        centres = geom.all_centres()
        for lz in range(geom.N_z):
            for i in range(geom.M):
                for j in range(geom.M):
                    phase = np.exp(1j * kP * centres[lz, i, j, 0])
                    # Displacement: pol * phase = [phase, 0, 0]
                    assert_allclose(psi0[lz, i, j, 0], phase, rtol=1e-12)
                    assert_allclose(psi0[lz, i, j, 1], 0.0, atol=1e-15)
                    assert_allclose(psi0[lz, i, j, 2], 0.0, atol=1e-15)

    def test_strain_components(self):
        """Strain components match _plane_wave_strain_voigt."""
        geom = SlabGeometry(M=2, N_z=1, a=A)
        psi0 = _build_slab_incident_field(geom, OMEGA, REF, K_HAT, "P")

        kP = OMEGA / REF.alpha
        eps_ref = _plane_wave_strain_voigt(K_HAT, K_HAT, kP)
        phase = np.exp(1j * kP * geom.cube_centre(0, 0, 0)[0])
        assert_allclose(psi0[0, 0, 0, 3:], eps_ref * phase, rtol=1e-12)

    def test_s_wave_perpendicular(self):
        """S-wave polarisation is perpendicular to k_hat."""
        geom = SlabGeometry(M=2, N_z=1, a=A)
        psi0 = _build_slab_incident_field(geom, OMEGA, REF, K_HAT, "S")
        # At any cube, displacement should be perpendicular to k_hat
        u = psi0[0, 0, 0, :3]
        # Remove phase
        kS = OMEGA / REF.beta
        phase = np.exp(1j * kS * geom.cube_centre(0, 0, 0)[0])
        pol = u / phase
        assert abs(np.dot(pol.real, K_HAT)) < 1e-10

    def test_bad_wave_type(self):
        geom = SlabGeometry(M=2, N_z=1, a=A)
        with pytest.raises(ValueError, match="wave_type"):
            _build_slab_incident_field(geom, OMEGA, REF, K_HAT, "X")


# ── 7. TestHomogeneousSlab ───────────────────────────────────────


class TestHomogeneousSlab:
    def test_zero_contrast_gives_psi_equals_psi0(self):
        """Zero contrast → ψ = ψ⁰ (no scattering)."""
        geom = SlabGeometry(M=2, N_z=2, a=A)
        zero_contrast = MaterialContrast(0.0, 0.0, 0.0)
        mat = uniform_slab_material(geom, REF, zero_contrast)
        result = compute_slab_scattering(geom, mat, OMEGA, K_HAT)
        assert_allclose(result.psi, result.psi0, atol=1e-10)

    def test_zero_contrast_zero_reflection(self):
        """Zero contrast → zero reflection."""
        geom = SlabGeometry(M=2, N_z=2, a=A)
        zero_contrast = MaterialContrast(0.0, 0.0, 0.0)
        mat = uniform_slab_material(geom, REF, zero_contrast)
        result = compute_slab_scattering(geom, mat, OMEGA, K_HAT)
        T_local = compute_slab_tmatrices(geom, mat, OMEGA)
        R_PP, R_PS, R_SP = slab_reflected_field(result, T_local)
        assert abs(R_PP) < 1e-20
        assert abs(R_PS) < 1e-20


# ── 8. TestSingleScatterer ──────────────────────────────────────


class TestSingleScatterer:
    def test_single_cube_scattering(self):
        """One cube with contrast in a grid of transparent cubes."""
        geom = SlabGeometry(M=3, N_z=1, a=A)
        shape = (1, 3, 3)
        Dlambda = np.zeros(shape)
        Dmu = np.zeros(shape)
        Drho = np.zeros(shape)
        # Centre cube gets contrast
        Dlambda[0, 1, 1] = CONTRAST.Dlambda
        Dmu[0, 1, 1] = CONTRAST.Dmu
        Drho[0, 1, 1] = CONTRAST.Drho
        mat = SlabMaterial(Dlambda=Dlambda, Dmu=Dmu, Drho=Drho, ref=REF)

        result = compute_slab_scattering(geom, mat, OMEGA, K_HAT)
        # The centre cube should have ψ ≈ ψ⁰ (no back-coupling from
        # transparent neighbours).  Use atol since some components are zero.
        assert_allclose(result.psi[0, 1, 1], result.psi0[0, 1, 1], atol=1e-10)

        # Non-centre cubes receive scattered field
        diff = np.abs(result.psi[0, 0, 0] - result.psi0[0, 0, 0])
        assert np.any(diff > 1e-15)  # some scattering


# ── 9. TestBornApproximation ─────────────────────────────────────


class TestBornApproximation:
    def test_reflection_scales_linearly(self):
        """Weak contrast: doubling contrast approximately doubles reflection."""
        geom = SlabGeometry(M=2, N_z=2, a=A)

        # Single weak contrast
        mat1 = uniform_slab_material(geom, REF, WEAK_CONTRAST)
        res1 = compute_slab_scattering(geom, mat1, OMEGA, K_HAT)
        T1 = compute_slab_tmatrices(geom, mat1, OMEGA)
        R1, _, _ = slab_reflected_field(res1, T1)

        # Double the contrast
        double_contrast = MaterialContrast(
            Dlambda=2 * WEAK_CONTRAST.Dlambda,
            Dmu=2 * WEAK_CONTRAST.Dmu,
            Drho=2 * WEAK_CONTRAST.Drho,
        )
        mat2 = uniform_slab_material(geom, REF, double_contrast)
        res2 = compute_slab_scattering(geom, mat2, OMEGA, K_HAT)
        T2 = compute_slab_tmatrices(geom, mat2, OMEGA)
        R2, _, _ = slab_reflected_field(res2, T2)

        # Ratio should be ~2 for weak contrast (Born regime)
        if abs(R1) > 1e-30:
            ratio = abs(R2) / abs(R1)
            assert_allclose(ratio, 2.0, rtol=0.05)


# ── 10. TestKennettComparison ────────────────────────────────────


class TestKennettComparison:
    def test_laterally_uniform_gives_consistent_phase(self):
        """Laterally uniform slab: dominant (z) component is uniform across layer."""
        geom = SlabGeometry(M=3, N_z=2, a=A)
        mat = uniform_slab_material(geom, REF, WEAK_CONTRAST)
        result = compute_slab_scattering(geom, mat, OMEGA, K_HAT)

        # For normal P-wave incidence on laterally uniform slab, the z-displacement
        # and zz-strain should be uniform across each layer.
        # Transverse components (u_x, u_y) differ due to finite-size edge effects.
        for lz in range(geom.N_z):
            psi_z_ref = result.psi[lz, 0, 0, 0]  # u_z
            psi_zz_ref = result.psi[lz, 0, 0, 3]  # ε_zz
            for i in range(geom.M):
                for j in range(geom.M):
                    assert_allclose(result.psi[lz, i, j, 0], psi_z_ref, rtol=1e-4)
                    assert_allclose(result.psi[lz, i, j, 3], psi_zz_ref, rtol=1e-4)


# ── 11. TestReciprocity ─────────────────────────────────────────


class TestReciprocity:
    def test_symmetric_slab_symmetric_reflection(self):
        """For a symmetric slab at normal incidence, R_PS should be small."""
        geom = SlabGeometry(M=2, N_z=2, a=A)
        mat = uniform_slab_material(geom, REF, WEAK_CONTRAST)
        result = compute_slab_scattering(geom, mat, OMEGA, K_HAT)
        T_local = compute_slab_tmatrices(geom, mat, OMEGA)
        R_PP, R_PS, R_SP = slab_reflected_field(result, T_local)
        # Normal incidence on isotropic medium: no P→S conversion
        assert abs(R_PS) < abs(R_PP) * 0.01 or abs(R_PP) < 1e-20


# ── 12. TestConvergence ──────────────────────────────────────────


class TestConvergence:
    def test_gmres_converges(self):
        """GMRES converges and residual is below tolerance."""
        geom = SlabGeometry(M=2, N_z=2, a=A)
        mat = uniform_slab_material(geom, REF, CONTRAST)
        result = compute_slab_scattering(geom, mat, OMEGA, K_HAT, gmres_tol=1e-8)
        assert result.gmres_residual < 1e-6
        assert result.n_gmres_iter > 0

    def test_weak_contrast_fast_convergence(self):
        """Weak contrast should converge in few iterations."""
        geom = SlabGeometry(M=2, N_z=2, a=A)
        mat = uniform_slab_material(geom, REF, WEAK_CONTRAST)
        result = compute_slab_scattering(geom, mat, OMEGA, K_HAT)
        assert result.gmres_residual < 1e-6
        # Weak contrast → near-identity system → fast convergence
        assert result.n_gmres_iter < 50


# ── 13. TestRandomSlab ───────────────────────────────────────────


class TestRandomSlab:
    def test_seeded_reproducibility(self):
        """Same seed gives identical material."""
        geom = SlabGeometry(M=4, N_z=2, a=A)
        mat1 = random_slab_material(geom, REF, CONTRAST, phi=0.3, seed=99)
        mat2 = random_slab_material(geom, REF, CONTRAST, phi=0.3, seed=99)
        assert_allclose(mat1.Dlambda, mat2.Dlambda)
        assert_allclose(mat1.Dmu, mat2.Dmu)
        assert_allclose(mat1.Drho, mat2.Drho)

    def test_different_seeds_differ(self):
        geom = SlabGeometry(M=4, N_z=2, a=A)
        mat1 = random_slab_material(geom, REF, CONTRAST, phi=0.3, seed=1)
        mat2 = random_slab_material(geom, REF, CONTRAST, phi=0.3, seed=2)
        assert not np.allclose(mat1.Dlambda, mat2.Dlambda)

    def test_depth_dependent_phi(self):
        """Callable phi(layer) produces depth-dependent volume fraction."""
        geom = SlabGeometry(M=8, N_z=3, a=A)
        mat = random_slab_material(
            geom, REF, CONTRAST, phi=lambda lz: 0.1 * (lz + 1), seed=42
        )
        # Layer 0: phi=0.1, Layer 1: phi=0.2, Layer 2: phi=0.3
        # Check that deeper layers have more inclusions (statistically)
        n_inc = [(mat.Dlambda[lz] > 0).sum() for lz in range(3)]
        # With 64 cubes per layer, expected: 6.4, 12.8, 19.2
        assert n_inc[0] < n_inc[2]  # trend should hold

    def test_physical_reflection_magnitude(self):
        """Moderate contrast: |R| should be physically reasonable (< 1)."""
        geom = SlabGeometry(M=2, N_z=2, a=A)
        mat = random_slab_material(geom, REF, CONTRAST, phi=0.5, seed=42)
        result = compute_slab_scattering(geom, mat, OMEGA, K_HAT)
        T_local = compute_slab_tmatrices(geom, mat, OMEGA)
        R_PP, R_PS, R_SP = slab_reflected_field(result, T_local)
        # Reflection amplitude should be physically bounded
        assert abs(R_PP) < 1.0


# ── 14. TestVolumeAveragedPropagator ─────────────────────────────


class TestVolumeAveragedPropagator:
    """Integration tests for the volume-averaged NN propagator in Foldy-Lax."""

    def test_volume_averaged_kernel_shape(self):
        """Volume-averaged kernels have the same shape as point-scatterer kernels."""
        geom = SlabGeometry(M=3, N_z=2, a=A)
        k_pt = _build_slab_kernels(geom, OMEGA, REF)
        k_va = _build_slab_kernels(geom, OMEGA, REF, volume_averaged=True)
        assert k_va.shape == k_pt.shape

    def test_volume_averaged_self_term_zero(self):
        """Self-term (R=0) remains zero with volume_averaged=True."""
        geom = SlabGeometry(M=3, N_z=2, a=A)
        kernel_hat = _build_slab_kernels(geom, OMEGA, REF, volume_averaged=True)
        # dz=0 → k = N_z-1 = 1; self at (M-1, M-1)
        kernel_spatial = np.fft.ifft2(kernel_hat[1], axes=(0, 1))
        assert_allclose(kernel_spatial[2, 2], 0.0, atol=1e-20)

    def test_nn_propagator_differs_from_point(self):
        """At face-adjacent separation, volume-averaged != point-scatterer."""
        geom = SlabGeometry(M=3, N_z=2, a=A)
        k_pt = _build_slab_kernels(geom, OMEGA, REF)
        k_va = _build_slab_kernels(geom, OMEGA, REF, volume_averaged=True)
        # dz=d (k=2), dx=0, dy=0 → spatial index (0+2, 0+2) = (2, 2)
        # Use dz≠0 face-adjacent for non-trivial propagator values
        sp_pt = np.fft.ifft2(k_pt[2], axes=(0, 1))
        sp_va = np.fft.ifft2(k_va[2], axes=(0, 1))
        G_pt = sp_pt[2, 2]
        G_va = sp_va[2, 2]
        # They should differ — volume averaging modifies the NN propagator
        scale = max(np.max(np.abs(G_pt)), np.max(np.abs(G_va)), 1e-30)
        diff = np.max(np.abs(G_pt - G_va)) / scale
        assert diff > 1e-6, f"Expected difference, got relative diff={diff:.2e}"

    def test_far_field_unchanged(self):
        """At dx=2 (beyond NN), both modes give identical results."""
        geom = SlabGeometry(M=4, N_z=2, a=A)
        k_pt = _build_slab_kernels(geom, OMEGA, REF)
        k_va = _build_slab_kernels(geom, OMEGA, REF, volume_averaged=True)
        # dz=d (k=2), dx=2, dy=0 → spatial index (2+3, 0+3) = (5, 3)
        sp_pt = np.fft.ifft2(k_pt[2], axes=(0, 1))
        sp_va = np.fft.ifft2(k_va[2], axes=(0, 1))
        scale = max(np.max(np.abs(sp_pt[5, 3])), 1e-30)
        assert_allclose(sp_va[5, 3], sp_pt[5, 3], atol=scale * 1e-10)

    def test_zero_contrast_identity_volume_averaged(self):
        """(I - G·0)ψ = ψ still works with volume_averaged=True."""
        geom = SlabGeometry(M=2, N_z=2, a=A)
        zero = MaterialContrast(0.0, 0.0, 0.0)
        mat = uniform_slab_material(geom, REF, zero)
        result = compute_slab_scattering(
            geom,
            mat,
            OMEGA,
            K_HAT,
            volume_averaged=True,
        )
        assert_allclose(result.psi, result.psi0, atol=1e-10)

    def test_single_cube_volume_averaged(self):
        """Single scatterer with volume_averaged gives physical result."""
        geom = SlabGeometry(M=3, N_z=1, a=A)
        shape = (1, 3, 3)
        Dlambda = np.zeros(shape)
        Dmu = np.zeros(shape)
        Drho = np.zeros(shape)
        Dlambda[0, 1, 1] = CONTRAST.Dlambda
        Dmu[0, 1, 1] = CONTRAST.Dmu
        Drho[0, 1, 1] = CONTRAST.Drho
        mat = SlabMaterial(Dlambda=Dlambda, Dmu=Dmu, Drho=Drho, ref=REF)

        result = compute_slab_scattering(
            geom,
            mat,
            OMEGA,
            K_HAT,
            volume_averaged=True,
        )
        # Centre cube ψ ≈ ψ⁰ (transparent neighbours, no back-coupling)
        assert_allclose(result.psi[0, 1, 1], result.psi0[0, 1, 1], atol=1e-10)

    def test_born_limit_volume_averaged(self):
        """Weak contrast: doubling contrast ~doubles reflection with volume_averaged."""
        geom = SlabGeometry(M=2, N_z=2, a=A)

        mat1 = uniform_slab_material(geom, REF, WEAK_CONTRAST)
        res1 = compute_slab_scattering(
            geom,
            mat1,
            OMEGA,
            K_HAT,
            volume_averaged=True,
        )
        T1 = compute_slab_tmatrices(geom, mat1, OMEGA)
        R1, _, _ = slab_reflected_field(res1, T1)

        double_contrast = MaterialContrast(
            Dlambda=2 * WEAK_CONTRAST.Dlambda,
            Dmu=2 * WEAK_CONTRAST.Dmu,
            Drho=2 * WEAK_CONTRAST.Drho,
        )
        mat2 = uniform_slab_material(geom, REF, double_contrast)
        res2 = compute_slab_scattering(
            geom,
            mat2,
            OMEGA,
            K_HAT,
            volume_averaged=True,
        )
        T2 = compute_slab_tmatrices(geom, mat2, OMEGA)
        R2, _, _ = slab_reflected_field(res2, T2)

        if abs(R1) > 1e-30:
            ratio = abs(R2) / abs(R1)
            assert_allclose(ratio, 2.0, rtol=0.05)

    def test_volume_averaged_vs_point_convergence(self):
        """Both modes give close results at low ka; volume-averaged kernel is smoother."""
        geom = SlabGeometry(M=2, N_z=2, a=A)
        mat = uniform_slab_material(geom, REF, WEAK_CONTRAST)

        res_pt = compute_slab_scattering(geom, mat, OMEGA, K_HAT)
        res_va = compute_slab_scattering(
            geom,
            mat,
            OMEGA,
            K_HAT,
            volume_averaged=True,
        )

        # At low ka (0.05), both should give similar exciting fields
        assert_allclose(res_va.psi, res_pt.psi, rtol=0.1)


# ── 15. TestPeriodicConvolution ────────────────────────────────


class TestPeriodicConvolution:
    """Tests for the periodic (circular convolution) mode."""

    def test_periodic_kernel_shape(self):
        """Periodic kernel is (n_dz, M, M, 9, 9)."""
        geom = SlabGeometry(M=4, N_z=3, a=A)
        kernel_hat = _build_slab_kernels(geom, OMEGA, REF, periodic=True)
        assert kernel_hat.shape == (5, 4, 4, 9, 9)

    def test_aperiodic_kernel_shape_unchanged(self):
        """Default (aperiodic) kernel shape is unchanged."""
        geom = SlabGeometry(M=4, N_z=3, a=A)
        kernel_hat = _build_slab_kernels(geom, OMEGA, REF)
        assert kernel_hat.shape == (5, 7, 7, 9, 9)

    def test_periodic_matches_direct_circular(self):
        """FFT periodic matvec matches explicit circular convolution with folded kernel."""
        geom = SlabGeometry(M=3, N_z=1, a=A)
        mat = uniform_slab_material(geom, REF, CONTRAST)
        T_local = compute_slab_tmatrices(geom, mat, OMEGA)

        # Build both periodic and aperiodic kernels
        kernel_hat_p = _build_slab_kernels(geom, OMEGA, REF, periodic=True)
        kernel_hat_ap = _build_slab_kernels(geom, OMEGA, REF)

        # Recover spatial folded kernel via IFFT
        M, N_z = geom.M, geom.N_z
        n_dz = 2 * N_z - 1
        kernel_circ = np.fft.ifft2(kernel_hat_p, axes=(1, 2))  # (n_dz, M, M, 9, 9)

        rng = np.random.default_rng(42)
        psi = rng.standard_normal(geom.n_cubes * 9) + 1j * rng.standard_normal(
            geom.n_cubes * 9
        )

        fft_result = _slab_matvec(psi, T_local, kernel_hat_p, geom, periodic=True)

        # Direct circular convolution using folded kernel
        psi_arr = psi.reshape(N_z, M, M, 9)
        tau = np.einsum("lmnab,lmnb->lmna", T_local, psi_arr)
        acc = np.zeros_like(psi_arr)

        for m in range(N_z):
            for i1 in range(M):
                for j1 in range(M):
                    for n in range(N_z):
                        for i2 in range(M):
                            for j2 in range(M):
                                dz_idx = (m - n) + (N_z - 1)
                                dx_idx = (i1 - i2) % M
                                dy_idx = (j1 - j2) % M
                                G = kernel_circ[dz_idx, dx_idx, dy_idx]
                                acc[m, i1, j1] += G @ tau[n, i2, j2]

        direct_result = (psi_arr - acc).ravel()
        assert_allclose(fft_result, direct_result, rtol=1e-10)

        # Also verify the folded kernel is correct: sum of aperiodic spatial kernel
        # folded mod M should equal periodic kernel
        kernel_spatial_ap = np.fft.ifft2(kernel_hat_ap, axes=(1, 2))
        S = 2 * M - 1
        kernel_folded_check = np.zeros((n_dz, M, M, 9, 9), dtype=complex)
        for k in range(n_dz):
            for ix in range(S):
                for iy in range(S):
                    dx_val = ix - (M - 1)
                    dy_val = iy - (M - 1)
                    kernel_folded_check[k, dx_val % M, dy_val % M] += kernel_spatial_ap[
                        k, ix, iy
                    ]
        assert_allclose(kernel_circ, kernel_folded_check, atol=1e-12)

    def test_identity_when_T_zero_periodic(self):
        """T=0 → identity in periodic mode."""
        geom = SlabGeometry(M=3, N_z=2, a=A)
        T_zero = np.zeros((2, 3, 3, 9, 9), dtype=complex)
        kernel_hat = _build_slab_kernels(geom, OMEGA, REF, periodic=True)
        psi = np.random.default_rng(42).standard_normal(2 * 3 * 3 * 9) + 0j
        result = _slab_matvec(psi, T_zero, kernel_hat, geom, periodic=True)
        assert_allclose(result, psi, atol=1e-12)

    def test_periodic_solver_zero_contrast(self):
        """Zero contrast → ψ = ψ₀ in periodic mode."""
        geom = SlabGeometry(M=2, N_z=2, a=A)
        zero = MaterialContrast(0.0, 0.0, 0.0)
        mat = uniform_slab_material(geom, REF, zero)
        result = compute_slab_scattering(geom, mat, OMEGA, K_HAT, periodic=True)
        assert_allclose(result.psi, result.psi0, atol=1e-10)

    def test_periodic_flag_stored_in_result(self):
        """SlabResult.periodic is stored correctly."""
        geom = SlabGeometry(M=2, N_z=1, a=A)
        zero = MaterialContrast(0.0, 0.0, 0.0)
        mat = uniform_slab_material(geom, REF, zero)

        res_ap = compute_slab_scattering(geom, mat, OMEGA, K_HAT)
        assert res_ap.periodic is False

        res_p = compute_slab_scattering(geom, mat, OMEGA, K_HAT, periodic=True)
        assert res_p.periodic is True


# ── TestWeylAmplitudes ───────────────────────────────────────────


class TestWeylAmplitudes:
    """slab_weyl_amplitudes: shared specular extractor for P/SV/SH."""

    def _solve_uniform(self, p, wave_type="P"):
        ref = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
        contrast = MaterialContrast(Dlambda=2.0e9, Dmu=1.0e9, Drho=100.0)
        geom = SlabGeometry(M=8, N_z=1, a=2.0)
        mat = uniform_slab_material(geom, ref, contrast)
        omega = 150.0
        c = ref.alpha if wave_type == "P" else ref.beta
        eta = np.sqrt(1.0 / c**2 - p**2 + 0j)
        k_hat = np.array([float(np.real(eta * c)), p * c, 0.0])
        result = compute_slab_scattering(
            geom, mat, omega, k_hat, wave_type=wave_type, periodic=True
        )
        T_local = compute_slab_tmatrices(geom, mat, omega)
        return result, T_local

    def test_rp_matches_rpp_periodic_normal_incidence(self):
        """At p=0: wrapper delegation is exact AND R_P anchors to Kennett.

        The delegation assertion documents the wiring (slab_rpp_periodic is
        a thin wrapper over the extractor); the Kennett comparison is the
        independent value-level anchor that pins the p=0 physics.
        """
        result, T_local = self._solve_uniform(p=0.0)
        amps = slab_weyl_amplitudes(result, T_local, p=0.0)
        legacy = slab_rpp_periodic(result, T_local, p=0.0)
        np.testing.assert_allclose(amps.R_P, legacy, rtol=1e-12)

        # Independent anchor: exact Kennett R_PP for the uniform layer.
        # H = N_z * d = 1 * (2 * 2.0) = 4.0 m for the _solve_uniform slab.
        ref = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
        contrast = MaterialContrast(Dlambda=2.0e9, Dmu=1.0e9, Drho=100.0)
        geom = result.geometry
        H = geom.N_z * geom.d
        R_K = kennett_reference_rpp(ref, contrast, H=H, omega=result.omega)
        np.testing.assert_allclose(amps.R_P, R_K, rtol=0.05)

    def test_conversions_vanish_at_normal_incidence(self):
        """P incidence at p=0: no SV or SH specular conversion."""
        result, T_local = self._solve_uniform(p=0.0)
        amps = slab_weyl_amplitudes(result, T_local, p=0.0)
        assert abs(amps.R_SV) < 1e-10 * max(abs(amps.R_P), 1e-30)
        assert abs(amps.R_SH) < 1e-10 * max(abs(amps.R_P), 1e-30)


class TestSHIncidence:
    """SH plane-wave incidence (polarisation ŷ)."""

    def test_sh_polarisation_is_y(self):
        ref = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
        geom = SlabGeometry(M=4, N_z=1, a=2.0)
        k_hat = np.array([0.8, 0.6, 0.0])
        psi0 = _build_slab_incident_field(geom, 150.0, ref, k_hat, "SH")
        # Displacement components: only u_y nonzero
        assert np.max(np.abs(psi0[..., 0])) < 1e-14
        assert np.max(np.abs(psi0[..., 1])) < 1e-14
        assert np.max(np.abs(psi0[..., 2])) > 0.99

    def test_unknown_wave_type_fails_fast(self):
        ref = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
        geom = SlabGeometry(M=4, N_z=1, a=2.0)
        with pytest.raises(ValueError, match="wave_type"):
            _build_slab_incident_field(geom, 150.0, ref, np.array([1.0, 0.0, 0.0]), "X")


class TestSlabReflectionMatrix:
    """Full specular matrix vs Kennett for a uniform slab.

    Modified (energy-normalized) convention throughout — the convention
    in which the Kennett matrix is symmetric (TestModifiedConventionSymmetry).
    """

    REF = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
    CONTRAST = MaterialContrast(Dlambda=2.0e9, Dmu=1.0e9, Drho=100.0)
    OMEGA = 150.0

    def _slab_matrix(self, p, M=8, N_z=1, a=2.0, contrast=None):
        contrast = contrast or self.CONTRAST
        geom = SlabGeometry(M=M, N_z=N_z, a=a)
        mat = uniform_slab_material(geom, self.REF, contrast)
        return slab_reflection_matrix(geom, mat, self.OMEGA, p=p)

    def test_normal_incidence_diagonal(self):
        """p=0: R_PP and R_SS match Kennett; conversions vanish; R_SS == R_SH."""
        p = 0.0
        slab = self._slab_matrix(p)
        kref = kennett_reference_matrix(
            self.REF, self.CONTRAST, H=4.0, omega=self.OMEGA, p=p
        )
        R_mod = slab.to_modified()
        np.testing.assert_allclose(R_mod[0, 0], kref.R_PP, rtol=0.05)
        np.testing.assert_allclose(R_mod[1, 1], kref.R_SS, rtol=0.05)
        np.testing.assert_allclose(slab.R_sh, kref.R_SH, rtol=0.05)
        # SV sign convention under reflection: K_SS(0) = -K_SH(0)
        # (same physics, opposite sign bookkeeping)
        np.testing.assert_allclose(slab.R_sh, -R_mod[1, 1], rtol=1e-6)
        assert abs(R_mod[0, 1]) < 0.01 * abs(R_mod[0, 0])
        assert abs(R_mod[1, 0]) < 0.01 * abs(R_mod[1, 1])

    def test_oblique_all_channels(self):
        """Sub-critical oblique p: all five channels vs Kennett."""
        p = 1.0e-4  # sin(theta_P) = 0.5
        slab = self._slab_matrix(p)
        kref = kennett_reference_matrix(
            self.REF, self.CONTRAST, H=4.0, omega=self.OMEGA, p=p
        )
        R_mod = slab.to_modified()
        np.testing.assert_allclose(R_mod[0, 0], kref.R_PP, rtol=0.07)
        np.testing.assert_allclose(R_mod[1, 1], kref.R_SS, rtol=0.07)
        np.testing.assert_allclose(slab.R_sh, kref.R_SH, rtol=0.07)
        # Off-diagonals: nonzero and matching (Kennett symmetric, so either index)
        assert abs(kref.R_PS) > 1e-4
        np.testing.assert_allclose(R_mod[1, 0], kref.R_PS, rtol=0.10)
        np.testing.assert_allclose(R_mod[0, 1], kref.R_SP, rtol=0.10)

    def test_modified_matrix_symmetric(self):
        """Reciprocity: the slab's modified matrix is symmetric like Kennett's."""
        slab = self._slab_matrix(p=1.0e-4)
        R_mod = slab.to_modified()
        np.testing.assert_allclose(R_mod[0, 1], R_mod[1, 0], rtol=0.05)

    def test_born_scaling_conversion_channel(self):
        """Weak contrast: doubling the contrast doubles R_PS (Born linearity)."""
        weak = MaterialContrast(
            Dlambda=self.REF.mu * 1e-4,
            Dmu=self.REF.mu * 1e-4,
            Drho=self.REF.rho * 1e-4,
        )
        weak2 = MaterialContrast(
            Dlambda=2 * weak.Dlambda, Dmu=2 * weak.Dmu, Drho=2 * weak.Drho
        )
        p = 1.0e-4
        r1 = self._slab_matrix(p, contrast=weak).to_modified()[1, 0]
        r2 = self._slab_matrix(p, contrast=weak2).to_modified()[1, 0]
        np.testing.assert_allclose(r2 / r1, 2.0, rtol=0.05)

    # Evanescent-P channels (PP, PS, SP past 1/alpha) are not asserted:
    # the Weyl extractor's evanescent-P branch is untested against Kennett
    # and deviates; propagating channels (SS, SH) are asserted.
    def test_post_critical_smoke(self):
        """p past the P-critical slowness: finite, branch-consistent R_SS."""
        p = 2.5e-4  # > 1/alpha = 2e-4 (P evanescent), < 1/beta (SV propagating)
        slab = self._slab_matrix(p)
        kref = kennett_reference_matrix(
            self.REF, self.CONTRAST, H=4.0, omega=self.OMEGA, p=p
        )
        R_mod = slab.to_modified()
        assert np.isfinite(R_mod).all()
        np.testing.assert_allclose(R_mod[1, 1], kref.R_SS, rtol=0.15)
        np.testing.assert_allclose(slab.R_sh, kref.R_SH, rtol=0.15)
