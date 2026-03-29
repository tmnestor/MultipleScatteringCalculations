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
    random_slab_material,
    slab_reflected_field,
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
