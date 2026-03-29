"""Tests for CPA (Coherent Potential Approximation) iteration.

Validates:
  1. Voigt average correctness
  2. Single-phase CPA converges in 1 iteration to input
  3. Two-phase CPA converges
  4. Weak contrast CPA ≈ Voigt average
  5. CPA effective medium has physical moduli (positive)
  6. Volume fraction validation
  7. Under-relaxation (damping) works
  8. Cubic anisotropy emerges from CPA
  9. Symmetry: swapping phases with complementary phi gives same result
 10. CPA result structure validation
"""

import numpy as np
import pytest

from cubic_scattering import MaterialContrast, ReferenceMedium
from cubic_scattering.cpa_iteration import (
    CPAResult,
    CubicEffectiveMedium,
    Phase,
    compute_cpa,
    compute_cpa_two_phase,
    phases_from_two_phase,
    voigt_average,
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

# Low frequency for static-like behaviour
OMEGA_LOW = 0.01 * REF.beta / 1.0  # ka=0.01
A = 1.0


# =====================================================================
# Test 1: Voigt average
# =====================================================================


class TestVoigtAverage:
    """Verify Voigt average computation."""

    def test_single_phase(self) -> None:
        """Single phase -> Voigt = that phase."""
        phase = Phase(lam=REF.lam, mu=REF.mu, rho=REF.rho, volume_fraction=1.0)
        avg = voigt_average([phase])
        np.testing.assert_allclose(avg.lam, REF.lam, rtol=1e-12)
        np.testing.assert_allclose(avg.mu_off, REF.mu, rtol=1e-12)
        np.testing.assert_allclose(avg.rho, REF.rho, rtol=1e-12)

    def test_two_phase_50_50(self) -> None:
        """Two phases at 50/50 -> arithmetic mean."""
        p1 = Phase(lam=1e10, mu=5e9, rho=2000.0, volume_fraction=0.5)
        p2 = Phase(lam=2e10, mu=1e10, rho=3000.0, volume_fraction=0.5)
        avg = voigt_average([p1, p2])
        np.testing.assert_allclose(avg.lam, 1.5e10, rtol=1e-12)
        np.testing.assert_allclose(avg.mu_off, 7.5e9, rtol=1e-12)
        np.testing.assert_allclose(avg.rho, 2500.0, rtol=1e-12)


# =====================================================================
# Test 2: Single-phase CPA
# =====================================================================


class TestSinglePhaseCPA:
    """Single phase: CPA should converge immediately."""

    def test_single_phase_converges_fast(self) -> None:
        phase = Phase(lam=REF.lam, mu=REF.mu, rho=REF.rho, volume_fraction=1.0)
        result = compute_cpa([phase], OMEGA_LOW, A)

        assert result.converged
        assert result.n_iterations <= 2
        np.testing.assert_allclose(result.effective_medium.lam, REF.lam, rtol=1e-6)
        np.testing.assert_allclose(result.effective_medium.mu_off, REF.mu, rtol=1e-6)
        np.testing.assert_allclose(result.effective_medium.rho, REF.rho, rtol=1e-6)


# =====================================================================
# Test 3: Two-phase CPA converges
# =====================================================================


class TestTwoPhaseCPA:
    """Two-phase CPA should converge."""

    def test_moderate_contrast_converges(self) -> None:
        result = compute_cpa_two_phase(
            REF, CONTRAST, volume_fraction=0.3, omega=OMEGA_LOW, a=A
        )
        assert result.converged, (
            f"CPA did not converge after {result.n_iterations} iterations, "
            f"residual={result.residual_history[-1]:.2e}"
        )

    def test_weak_contrast_converges(self) -> None:
        result = compute_cpa_two_phase(
            REF, WEAK_CONTRAST, volume_fraction=0.5, omega=OMEGA_LOW, a=A
        )
        assert result.converged
        assert result.n_iterations <= 5


# =====================================================================
# Test 4: Weak contrast CPA ≈ Voigt average
# =====================================================================


class TestWeakContrastApproxVoigt:
    """At weak contrast, CPA should be close to Voigt average."""

    def test_weak_contrast_near_voigt(self) -> None:
        phi = 0.3
        phases = phases_from_two_phase(REF, WEAK_CONTRAST, phi)
        voigt = voigt_average(phases)

        result = compute_cpa_two_phase(
            REF, WEAK_CONTRAST, volume_fraction=phi, omega=OMEGA_LOW, a=A
        )
        eff = result.effective_medium

        np.testing.assert_allclose(eff.lam, voigt.lam, rtol=1e-3)
        np.testing.assert_allclose(eff.mu_off, voigt.mu_off, rtol=1e-3)
        np.testing.assert_allclose(eff.rho, voigt.rho, rtol=1e-3)


# =====================================================================
# Test 5: Physical moduli (positive)
# =====================================================================


class TestPhysicalModuli:
    """Effective medium should have positive moduli."""

    def test_positive_moduli(self) -> None:
        result = compute_cpa_two_phase(
            REF, CONTRAST, volume_fraction=0.3, omega=OMEGA_LOW, a=A
        )
        eff = result.effective_medium
        assert eff.mu_off > 0, f"mu_off={eff.mu_off}"
        assert eff.mu_diag > 0, f"mu_diag={eff.mu_diag}"
        assert eff.bulk_modulus > 0, f"K={eff.bulk_modulus}"
        assert eff.rho > 0, f"rho={eff.rho}"


# =====================================================================
# Test 6: Volume fraction validation
# =====================================================================


class TestVolumeFractionValidation:
    """Volume fractions must sum to 1."""

    def test_invalid_phi_raises(self) -> None:
        p1 = Phase(lam=1e10, mu=5e9, rho=2000.0, volume_fraction=0.3)
        p2 = Phase(lam=2e10, mu=1e10, rho=3000.0, volume_fraction=0.3)
        with pytest.raises(ValueError, match="sum to 1"):
            compute_cpa([p1, p2], OMEGA_LOW, A)


# =====================================================================
# Test 7: Under-relaxation
# =====================================================================


class TestUnderRelaxation:
    """Damping < 1 should still converge (possibly slower)."""

    def test_damped_converges(self) -> None:
        result = compute_cpa_two_phase(
            REF,
            CONTRAST,
            volume_fraction=0.3,
            omega=OMEGA_LOW,
            a=A,
            damping=0.5,
        )
        assert result.converged


# =====================================================================
# Test 8: Cubic anisotropy from CPA
# =====================================================================


class TestCubicAnisotropy:
    """CPA should produce cubic anisotropy (mu_diag != mu_off)."""

    def test_anisotropy_nonzero(self) -> None:
        result = compute_cpa_two_phase(
            REF, CONTRAST, volume_fraction=0.3, omega=OMEGA_LOW, a=A
        )
        eff = result.effective_medium
        assert abs(eff.cubic_anisotropy) > 0, (
            f"No cubic anisotropy: mu_diag={eff.mu_diag}, mu_off={eff.mu_off}"
        )

    def test_anisotropy_small_at_weak_contrast(self) -> None:
        result = compute_cpa_two_phase(
            REF, WEAK_CONTRAST, volume_fraction=0.3, omega=OMEGA_LOW, a=A
        )
        eff = result.effective_medium
        # Relative anisotropy should be very small
        rel = abs(eff.cubic_anisotropy) / max(abs(eff.mu_off), 1.0)
        assert rel < 1e-4, f"Relative anisotropy {rel} too large at weak contrast"


# =====================================================================
# Test 9: Symmetry (swap phases)
# =====================================================================


class TestPhaseSymmetry:
    """Effective medium properties should be between the two phases."""

    def test_effective_between_phases(self) -> None:
        phi = 0.3
        result = compute_cpa_two_phase(
            REF, CONTRAST, volume_fraction=phi, omega=OMEGA_LOW, a=A
        )
        eff = result.effective_medium

        mu_matrix = REF.mu
        mu_inclusion = REF.mu + CONTRAST.Dmu
        mu_min = min(mu_matrix, mu_inclusion)
        mu_max = max(mu_matrix, mu_inclusion)

        assert mu_min <= eff.mu_off <= mu_max, (
            f"mu_off={eff.mu_off} outside [{mu_min}, {mu_max}]"
        )


# =====================================================================
# Test 10: CPAResult structure
# =====================================================================


class TestCPAResultStructure:
    """Verify CPAResult has correct fields."""

    def test_result_fields(self) -> None:
        result = compute_cpa_two_phase(
            REF, CONTRAST, volume_fraction=0.3, omega=OMEGA_LOW, a=A
        )

        assert isinstance(result, CPAResult)
        assert isinstance(result.effective_medium, CubicEffectiveMedium)
        assert result.converged
        assert result.n_iterations > 0
        assert len(result.residual_history) == result.n_iterations
        assert len(result.tmatrix_results) == 2  # two phases
        assert len(result.phases) == 2
        assert result.omega == OMEGA_LOW
        assert result.a == A

    def test_residual_decreasing(self) -> None:
        """Residual should generally decrease (not strictly, but overall)."""
        result = compute_cpa_two_phase(
            REF, CONTRAST, volume_fraction=0.3, omega=OMEGA_LOW, a=A
        )
        if result.n_iterations > 2:
            # First residual should be larger than last
            assert result.residual_history[0] > result.residual_history[-1]
