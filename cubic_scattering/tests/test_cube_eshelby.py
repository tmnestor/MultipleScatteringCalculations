"""Tests for cube Eshelby concentration factors.

Validates:
  1. Born linearity — T-matrix components scale linearly with contrast
  2. Known Eshelby amp_theta — matches K₀/(K₀ + alpha_E·DK)
  3. Known Eshelby amp_u — approx 1.0 at static limit
  4. Known Eshelby amp_e_off — matches 1/(1 + beta_E·Dmu/mu₀)
  5. Known Eshelby amp_e_diag — distinct from amp_e_off (cubic anisotropy)
  6. Static factors are real-valued at low ka
  7. Weak contrast -> unity for all 4 channels
  8. Scale-free — concentration factors independent of cube half-width a
  9. Dynamic -> static at low ka
 10. Cubic anisotropy — Dmu_star_diag != Dmu_star_off
 11. Convergence study structure validation
 12. CubeEshelbyResult structure validation
"""

import numpy as np
import pytest

from cubic_scattering import MaterialContrast, ReferenceMedium
from cubic_scattering.cube_eshelby import (
    CubeConvergenceResult,
    CubeEshelbyResult,
    compute_cube_born_tmatrix,
    compute_cube_eshelby,
    compute_cube_eshelby_factors,
    cube_convergence_study,
)

# =====================================================================
# Shared test fixtures
# =====================================================================

REF = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)

CONTRAST = MaterialContrast(
    Dlambda=2.0e9,
    Dmu=1.0e9,
    Drho=100.0,
)

WEAK_CONTRAST = MaterialContrast(
    Dlambda=REF.mu * 1e-4,
    Dmu=REF.mu * 1e-4,
    Drho=REF.rho * 1e-4,
)


def _known_cube_amplification_factors(
    ref: ReferenceMedium, contrast: MaterialContrast
) -> tuple[float, float]:
    """Known static amplification factors for amp_theta and amp_e_off.

    For a sphere these are exact; for a cube, the isotropic part of
    the Eshelby tensor gives the same leading-order structure:

    amp_theta ~ K₀/(K₀ + alpha_E·DK)
    amp_e_off ~ 1/(1 + beta_E·Dmu/mu₀)

    where alpha_E, beta_E are the sphere Eshelby depolarization factors.
    The cube values differ because the Eshelby tensor geometry differs,
    but the structure is analogous.
    """
    K0 = ref.lam + 2.0 * ref.mu / 3.0
    alpha_E = 3.0 * K0 / (3.0 * K0 + 4.0 * ref.mu)
    beta_E = 6.0 * (K0 + 2.0 * ref.mu) / (5.0 * (3.0 * K0 + 4.0 * ref.mu))
    DK = contrast.Dlambda + 2.0 * contrast.Dmu / 3.0
    amp_theta_sphere = K0 / (K0 + alpha_E * DK)
    amp_e_off_sphere = 1.0 / (1.0 + beta_E * contrast.Dmu / ref.mu)
    return amp_theta_sphere, amp_e_off_sphere


# =====================================================================
# Test 1: Born linearity
# =====================================================================


class TestBornLinearity:
    """Verify that Born T-matrix components scale linearly with contrast."""

    def test_born_linearity_effective_contrasts(self) -> None:
        """Born effective contrasts at eps vs 2*eps should match."""
        omega = 0.1 * REF.beta / 1.0  # ka=0.1

        born1 = compute_cube_born_tmatrix(omega, 1.0, REF, CONTRAST, epsilon=1e-5)
        born2 = compute_cube_born_tmatrix(omega, 1.0, REF, CONTRAST, epsilon=2e-5)

        # Effective contrasts should be independent of epsilon
        np.testing.assert_allclose(
            abs(born2.Drho_star), abs(born1.Drho_star), rtol=1e-4, err_msg="Drho_star"
        )
        np.testing.assert_allclose(
            abs(born2.Dlambda_star),
            abs(born1.Dlambda_star),
            rtol=1e-4,
            err_msg="Dlambda_star",
        )
        np.testing.assert_allclose(
            abs(born2.Dmu_star_off),
            abs(born1.Dmu_star_off),
            rtol=1e-4,
            err_msg="Dmu_star_off",
        )
        np.testing.assert_allclose(
            abs(born2.Dmu_star_diag),
            abs(born1.Dmu_star_diag),
            rtol=1e-4,
            err_msg="Dmu_star_diag",
        )


# =====================================================================
# Tests 2-5: Known Eshelby concentration factors
# =====================================================================


class TestKnownEshelbyFactors:
    """Validate cube amplification factors against known values."""

    @pytest.fixture()
    def static_result(self) -> CubeEshelbyResult:
        return compute_cube_eshelby_factors(REF, CONTRAST, ka=0.01)

    def test_eshelby_amp_theta(self, static_result: CubeEshelbyResult) -> None:
        """amp_theta should be close to the sphere monopole analog.

        The cube Eshelby tensor for volumetric strain has the same
        structure as the sphere (isotropic Ja part), so amp_theta
        should be close to K₀/(K₀ + alpha_E·DK).
        """
        amp_theta_sphere, _ = _known_cube_amplification_factors(REF, CONTRAST)
        # Cube differs from sphere, so allow moderate tolerance
        np.testing.assert_allclose(
            static_result.amp_theta.real,
            amp_theta_sphere,
            rtol=0.15,
            err_msg="amp_theta vs sphere analog",
        )

    def test_eshelby_amp_u(self, static_result: CubeEshelbyResult) -> None:
        """amp_u ~ 1.0 at static limit (density has no static correction)."""
        np.testing.assert_allclose(
            static_result.amp_u.real, 1.0, atol=1e-3, err_msg="amp_u vs 1.0"
        )

    def test_eshelby_amp_e_off(self, static_result: CubeEshelbyResult) -> None:
        """amp_e_off should be close to the sphere quadrupole analog."""
        _, amp_e_off_sphere = _known_cube_amplification_factors(REF, CONTRAST)
        np.testing.assert_allclose(
            static_result.amp_e_off.real,
            amp_e_off_sphere,
            rtol=0.15,
            err_msg="amp_e_off vs sphere analog",
        )

    def test_eshelby_amp_e_diag_distinct(
        self, static_result: CubeEshelbyResult
    ) -> None:
        """amp_e_diag should differ from amp_e_off (cubic anisotropy)."""
        # They should both be real and positive, but different
        assert static_result.amp_e_diag.real > 0, "amp_e_diag should be positive"
        assert static_result.amp_e_off.real > 0, "amp_e_off should be positive"
        # The difference captures cubic anisotropy (small but nonzero)
        diff = abs(static_result.amp_e_diag - static_result.amp_e_off)
        # At finite contrast the difference should be measurable
        assert diff > 1e-6, (
            f"amp_e_diag={static_result.amp_e_diag}, "
            f"amp_e_off={static_result.amp_e_off}, diff={diff}"
        )


# =====================================================================
# Test 6: Static factors are real-valued
# =====================================================================


class TestStaticFactorsReal:
    """Static amplification factors should be purely real at low ka."""

    def test_imaginary_part_negligible(self) -> None:
        result = compute_cube_eshelby_factors(REF, CONTRAST, ka=0.01)
        for name, val in [
            ("amp_u", result.amp_u),
            ("amp_theta", result.amp_theta),
            ("amp_e_off", result.amp_e_off),
            ("amp_e_diag", result.amp_e_diag),
        ]:
            assert abs(val.imag) < 1e-4, f"Im({name}) = {val.imag}"


# =====================================================================
# Test 7: Weak contrast -> unity
# =====================================================================


class TestWeakContrastUnity:
    """At weak contrast, all amplification factors -> 1.0."""

    def test_all_factors_near_one(self) -> None:
        result = compute_cube_eshelby_factors(REF, WEAK_CONTRAST, ka=0.01)
        for name, val in [
            ("amp_u", result.amp_u),
            ("amp_theta", result.amp_theta),
            ("amp_e_off", result.amp_e_off),
            ("amp_e_diag", result.amp_e_diag),
        ]:
            np.testing.assert_allclose(
                val.real, 1.0, atol=1e-3, err_msg=f"{name} at weak contrast"
            )

    def test_concentration_ratios_near_one(self) -> None:
        result = compute_cube_eshelby_factors(REF, WEAK_CONTRAST, ka=0.01)
        for name, val in [
            ("E_u", result.E_u),
            ("E_theta", result.E_theta),
            ("E_e_off", result.E_e_off),
            ("E_e_diag", result.E_e_diag),
        ]:
            np.testing.assert_allclose(
                val.real, 1.0, atol=1e-3, err_msg=f"{name} at weak contrast"
            )


# =====================================================================
# Test 8: Scale-free (a-independent)
# =====================================================================


class TestScaleFree:
    """Concentration factors should be independent of cube half-width a."""

    def test_different_a_values(self) -> None:
        r_small = compute_cube_eshelby_factors(REF, CONTRAST, a=0.5, ka=0.01)
        r_large = compute_cube_eshelby_factors(REF, CONTRAST, a=2.0, ka=0.01)

        for name in ["amp_u", "amp_theta", "amp_e_off", "amp_e_diag"]:
            val_s = getattr(r_small, name)
            val_l = getattr(r_large, name)
            np.testing.assert_allclose(
                val_s.real, val_l.real, atol=1e-3, err_msg=f"{name} a-dependence"
            )


# =====================================================================
# Test 9: Dynamic -> static at low ka
# =====================================================================


class TestDynamicApproachesStatic:
    """At low ka, dynamic factors should match static."""

    def test_low_ka_consistency(self) -> None:
        r_static = compute_cube_eshelby_factors(REF, CONTRAST, ka=0.01)
        r_dynamic = compute_cube_eshelby_factors(REF, CONTRAST, ka=0.05)

        for name in ["amp_u", "amp_theta", "amp_e_off", "amp_e_diag"]:
            val_s = getattr(r_static, name)
            val_d = getattr(r_dynamic, name)
            np.testing.assert_allclose(
                val_d.real,
                val_s.real,
                rtol=0.01,
                err_msg=f"{name} dynamic vs static at ka=0.05",
            )


# =====================================================================
# Test 10: Cubic anisotropy
# =====================================================================


class TestCubicAnisotropy:
    """Dmu_star_diag != Dmu_star_off at finite contrast (cube-specific)."""

    def test_anisotropy_nonzero(self) -> None:
        result = compute_cube_eshelby_factors(REF, CONTRAST, ka=0.01)
        assert abs(result.cubic_anisotropy) > 1e-3, (
            f"cubic_anisotropy={result.cubic_anisotropy} too small"
        )

    def test_anisotropy_zero_at_weak_contrast(self) -> None:
        result = compute_cube_eshelby_factors(REF, WEAK_CONTRAST, ka=0.01)
        # At weak contrast, anisotropy should be very small (proportional to contrast)
        assert abs(result.cubic_anisotropy) < 1e2, (
            f"cubic_anisotropy={result.cubic_anisotropy} too large at weak contrast"
        )


# =====================================================================
# Test 11: Convergence study structure
# =====================================================================


class TestConvergenceStudy:
    """Verify cube_convergence_study returns sensible structure."""

    def test_convergence_study_shape(self) -> None:
        ka_vals = np.array([0.05, 0.1, 0.2])

        result = cube_convergence_study(REF, CONTRAST, ka_values=ka_vals)

        assert isinstance(result, CubeConvergenceResult)
        assert len(result.ka_values) == 3
        assert len(result.amp_u) == 3
        assert len(result.amp_theta) == 3
        assert len(result.amp_e_off) == 3
        assert len(result.amp_e_diag) == 3
        assert len(result.E_u) == 3
        assert isinstance(result.E_static, CubeEshelbyResult)
        assert len(result.used_resonance) == 3


# =====================================================================
# Test 12: CubeEshelbyResult structure
# =====================================================================


class TestCubeEshelbyResult:
    """Verify compute_cube_eshelby returns correct structure."""

    def test_result_fields(self) -> None:
        result = compute_cube_eshelby(REF, CONTRAST)

        assert isinstance(result, CubeEshelbyResult)
        # Amplification factors present
        assert result.amp_u is not None
        assert result.amp_theta is not None
        assert result.amp_e_off is not None
        assert result.amp_e_diag is not None
        # Effective contrasts present
        assert result.Drho_star is not None
        assert result.Dlambda_star is not None
        assert result.Dmu_star_off is not None
        assert result.Dmu_star_diag is not None
        # Born contrasts present
        assert result.Drho_star_born is not None
        assert result.Dlambda_star_born is not None
        # Concentration ratios present
        assert result.E_u is not None
        assert result.E_theta is not None
        assert result.E_e_off is not None
        assert result.E_e_diag is not None
        # T-matrix internals
        assert result.T1c is not None
        assert result.T2c is not None
        assert result.T3c is not None
        assert result.Gamma0 is not None
        # Metadata
        assert result.ka > 0
        assert result.ref is REF
        assert result.contrast is CONTRAST

    def test_born_contrasts_match_bare(self) -> None:
        """At weak contrast, Born contrasts ≈ bare contrasts."""
        result = compute_cube_eshelby_factors(REF, WEAK_CONTRAST, ka=0.01)
        np.testing.assert_allclose(
            result.Drho_star_born.real,
            WEAK_CONTRAST.Drho,
            rtol=1e-3,
            err_msg="Born Drho vs bare",
        )
        np.testing.assert_allclose(
            result.Dmu_star_off_born.real,
            WEAK_CONTRAST.Dmu,
            rtol=1e-3,
            err_msg="Born Dmu_off vs bare",
        )
