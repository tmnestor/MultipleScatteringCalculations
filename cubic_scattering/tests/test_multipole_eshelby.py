"""Tests for multipole Eshelby concentration factors.

Validates:
  1. Born linearity
  2. Known Eshelby n=0 (monopole/bulk modulus)
  3. Known Eshelby n=1 (dipole/density)
  4. Known Eshelby n=2 (quadrupole/shear modulus)
  5. Static factors are real-valued
  6. Weak contrast → all E_n → 1.0
  7. Scale-free (E_n independent of radius)
  8. Dynamic → static at low ka
  9. Truncation error monotonically decreasing with n_trunc
 10. n=3 truncation improves on n=2
 11. Convergence study structure
 12. MultipoleEshelbyResult structure
"""

import numpy as np
import pytest

from cubic_scattering import MaterialContrast, ReferenceMedium
from cubic_scattering.multipole_eshelby import (
    ConvergenceResult,
    MultipoleEshelbyResult,
    compute_born_coefficients,
    compute_multipole_eshelby,
    compute_static_eshelby_factors,
    convergence_study,
    far_field_truncation_error,
)
from cubic_scattering.sphere_scattering import compute_elastic_mie

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


def _known_amplification_factors(
    ref: ReferenceMedium, contrast: MaterialContrast
) -> tuple[float, float]:
    """Return known static amplification factors for n=0 (amp_vol) and n=2 (amp_dev).

    E_0 = K₀/(K₀ + alpha_E·ΔK)   where alpha_E = 3K₀/(3K₀+4μ₀)
    E_2 = 1/(1 + beta_E·Δμ/μ₀)   where beta_E = 6(K₀+2μ₀)/(5(3K₀+4μ₀))
    """
    K0 = ref.lam + 2.0 * ref.mu / 3.0
    alpha_E = 3.0 * K0 / (3.0 * K0 + 4.0 * ref.mu)
    beta_E = 6.0 * (K0 + 2.0 * ref.mu) / (5.0 * (3.0 * K0 + 4.0 * ref.mu))
    DK = contrast.Dlambda + 2.0 * contrast.Dmu / 3.0
    amp_vol = K0 / (K0 + alpha_E * DK)
    amp_dev = 1.0 / (1.0 + beta_E * contrast.Dmu / ref.mu)
    return amp_vol, amp_dev


# =====================================================================
# Test 1: Born linearity
# =====================================================================


class TestBornLinearity:
    """Verify that Born coefficients scale linearly with contrast."""

    def test_born_linearity_ratio(self) -> None:
        """a_n(2*eps) / a_n(eps) ≈ 2.0 for all orders."""
        omega = 0.1 * REF.beta / 1.0  # ka=0.1, radius=1
        n_max = 5

        a_eps1 = compute_born_coefficients(
            omega, 1.0, REF, CONTRAST, n_max=n_max, epsilon=1e-6
        )
        a_eps2 = compute_born_coefficients(
            omega, 1.0, REF, CONTRAST, n_max=n_max, epsilon=2e-6
        )

        for n in range(n_max + 1):
            if abs(a_eps1[n]) > 1e-30:
                ratio = abs(a_eps2[n] / a_eps1[n])
                np.testing.assert_allclose(ratio, 1.0, atol=1e-4, err_msg=f"n={n}")


# =====================================================================
# Tests 2-4: Known Eshelby concentration factors
# =====================================================================


class TestKnownEshelbyFactors:
    """Validate E_n against known analytical Eshelby amplification factors."""

    @pytest.fixture()
    def static_factors(self) -> np.ndarray:
        return compute_static_eshelby_factors(REF, CONTRAST, n_max=5)

    def test_eshelby_n0_monopole(self, static_factors: np.ndarray) -> None:
        """E_0 ≈ amp_vol = K₀/(K₀ + alpha_E·ΔK)."""
        amp_vol, _ = _known_amplification_factors(REF, CONTRAST)
        np.testing.assert_allclose(
            static_factors[0].real, amp_vol, atol=1e-4, err_msg="E_0 vs amp_vol"
        )

    def test_eshelby_n1_dipole(self, static_factors: np.ndarray) -> None:
        """E_1 ≈ 1.0 (no static density renormalization)."""
        np.testing.assert_allclose(
            static_factors[1].real, 1.0, atol=1e-4, err_msg="E_1 vs 1.0"
        )

    def test_eshelby_n2_quadrupole(self, static_factors: np.ndarray) -> None:
        """E_2 ≈ amp_dev = 1/(1 + beta_E·Δμ/μ₀)."""
        _, amp_dev = _known_amplification_factors(REF, CONTRAST)
        np.testing.assert_allclose(
            static_factors[2].real, amp_dev, atol=1e-4, err_msg="E_2 vs amp_dev"
        )


# =====================================================================
# Test 5: Static factors are real-valued
# =====================================================================


class TestStaticFactorsReal:
    """Static Eshelby factors should be purely real."""

    def test_imaginary_part_negligible(self) -> None:
        E_n = compute_static_eshelby_factors(REF, CONTRAST, n_max=5)
        for n in range(len(E_n)):
            assert abs(E_n[n].imag) < 1e-6, f"Im(E_{n}) = {E_n[n].imag}"


# =====================================================================
# Test 6: Weak contrast → unity
# =====================================================================


class TestWeakContrastUnity:
    """At weak contrast, all E_n → 1.0."""

    def test_all_factors_near_one(self) -> None:
        # Use ka=0.05 so higher-order coefficients are above noise floor
        E_n = compute_static_eshelby_factors(
            REF, WEAK_CONTRAST, n_max=3, ka_static=0.05
        )
        for n in range(len(E_n)):
            np.testing.assert_allclose(
                E_n[n].real, 1.0, atol=1e-3, err_msg=f"E_{n} at weak contrast"
            )


# =====================================================================
# Test 7: Scale-free (radius-independent)
# =====================================================================


class TestScaleFree:
    """E_n should be independent of the sphere radius."""

    def test_different_radii(self) -> None:
        E_small = compute_static_eshelby_factors(REF, CONTRAST, n_max=4, radius=0.5)
        E_large = compute_static_eshelby_factors(REF, CONTRAST, n_max=4, radius=2.0)
        for n in range(5):
            np.testing.assert_allclose(
                E_small[n].real,
                E_large[n].real,
                atol=1e-3,
                err_msg=f"E_{n} radius dependence",
            )


# =====================================================================
# Test 8: Dynamic → static at low ka
# =====================================================================


class TestDynamicApproachesStatic:
    """At low ka, dynamic E_n should match static E_n."""

    def test_low_ka_consistency(self) -> None:
        result = compute_multipole_eshelby(REF, CONTRAST, n_max=4, ka_dynamic=0.05)
        for n in range(5):
            np.testing.assert_allclose(
                result.E_n_dynamic[n].real,
                result.E_n_static[n].real,
                rtol=0.01,
                err_msg=f"E_{n} dynamic vs static at ka=0.05",
            )


# =====================================================================
# Test 9: Truncation error monotonically decreasing
# =====================================================================


class TestTruncationMonotone:
    """Error should decrease as more multipole orders are included."""

    def test_error_decreasing(self) -> None:
        omega = 0.5 * REF.beta / 1.0  # ka=0.5
        mie = compute_elastic_mie(omega, 1.0, REF, CONTRAST, n_max=10)

        errors = [far_field_truncation_error(mie, n_trunc) for n_trunc in range(2, 8)]
        for i in range(len(errors) - 1):
            assert errors[i] >= errors[i + 1], (
                f"Error not monotone: err(n={i + 2})={errors[i]:.6e} "
                f"< err(n={i + 3})={errors[i + 1]:.6e}"
            )


# =====================================================================
# Test 10: n=3 improves on n=2 at ka=0.3
# =====================================================================


class TestOctupoleImprovement:
    """Including n=3 should reduce far-field error at ka=0.3."""

    def test_n3_beats_n2(self) -> None:
        omega = 0.3 * REF.beta / 1.0  # ka=0.3
        mie = compute_elastic_mie(omega, 1.0, REF, CONTRAST, n_max=10)

        err_n2 = far_field_truncation_error(mie, n_trunc=2)
        err_n3 = far_field_truncation_error(mie, n_trunc=3)

        assert err_n3 < err_n2, (
            f"n=3 truncation ({err_n3:.6e}) not better than n=2 ({err_n2:.6e})"
        )


# =====================================================================
# Integration: convergence_study produces valid results
# =====================================================================


class TestConvergenceStudy:
    """Verify convergence_study returns sensible structure."""

    def test_convergence_study_shape(self) -> None:
        ka_vals = np.array([0.1, 0.3, 0.5])
        n_trunc_vals = np.array([2, 3, 4])

        result = convergence_study(
            REF, CONTRAST, ka_values=ka_vals, n_trunc_values=n_trunc_vals
        )

        assert isinstance(result, ConvergenceResult)
        assert result.errors.shape == (3, 3)
        assert len(result.ka_thresholds) > 0
        # All errors should be non-negative
        assert np.all(result.errors >= 0)


# =====================================================================
# Integration: MultipoleEshelbyResult structure
# =====================================================================


class TestMultipoleEshelbyResult:
    """Verify compute_multipole_eshelby returns correct structure."""

    def test_result_fields(self) -> None:
        result = compute_multipole_eshelby(REF, CONTRAST, n_max=5)

        assert isinstance(result, MultipoleEshelbyResult)
        assert result.n_max == 5
        assert len(result.E_n_static) == 6
        assert len(result.E_n_dynamic) == 6
        assert len(result.a_n_born) == 6
        assert len(result.a_n_full) == 6
        assert result.ka_P > 0
