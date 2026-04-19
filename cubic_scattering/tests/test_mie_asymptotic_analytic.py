"""Cross-validation: analytic ka→0 Mie closed-forms ↔ numerical Mie.

Locks down the reference for downstream point-scatterer / mean-field
near-field comparisons.

The analytic forms in `mie_asymptotic_analytic.py` are transcribed from
`Mathematica/MieAsymptotic.wl` and were validated against textbook
Eshelby in `Mathematica/MieAsymptoticVerify.wl` (V1–V5 all PASS).

These tests confirm the same forms agree numerically with the existing
`compute_elastic_mie` in this package, at small ka and finite contrast.
"""

from __future__ import annotations

import pytest

from cubic_scattering import MaterialContrast, ReferenceMedium
from cubic_scattering.mie_asymptotic_analytic import (
    E_0,
    E_1,
    E_2,
    Dmu_star_from_a2,
    NondimContrast,
    U_r_n0_leading,
    U_r_n2_leading,
    U_theta_n2_leading,
    a_0_analytic,
    a_1_analytic,
    a_2_analytic,
    b_2_analytic,
    beta_E,
)
from cubic_scattering.sphere_scattering import (
    compute_elastic_mie,
    mie_extract_effective_contrasts,
)

# =====================================================================
# Standard test fixtures (match Python convention used elsewhere)
# =====================================================================

REF = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
RADIUS = 10.0
KA_S_TARGET = 0.05  # w = ω·a/β

OMEGA_TEST = KA_S_TARGET * REF.beta / RADIUS  # such that ω·a/β = 0.05


def _physical_to_nondim(contrast: MaterialContrast) -> NondimContrast:
    return NondimContrast.from_physical(
        alpha=REF.alpha,
        beta=REF.beta,
        rho=REF.rho,
        Dlambda=contrast.Dlambda,
        Dmu=contrast.Dmu,
        Drho=contrast.Drho,
    )


def _w_from_omega(omega: float) -> float:
    return omega * RADIUS / REF.beta


# =====================================================================
# Section A: identities that hold purely symbolically
# =====================================================================


class TestEshelbyIdentities:
    """Sanity-check the Python evaluator reproduces the symbolic identities."""

    def test_E0_zero_contrast(self):
        c = NondimContrast(lam0=7 / 9, dlam=0.0, dmu=0.0, drho=0.0)
        assert E_0(c) == pytest.approx(1.0)

    def test_E1_identically_one(self):
        for d in [0.0, 0.1, 0.5, 1.0, 5.0]:
            c = NondimContrast(lam0=7 / 9, dlam=d, dmu=d, drho=d)
            assert E_1(c) == 1.0

    def test_E2_matches_textbook_betaE(self):
        """E₂ = 1 / (1 + β_E·Δμ)."""
        lam0 = 7 / 9
        bE = beta_E(lam0)
        for dmu in [-0.3, -0.05, 0.0, 0.1, 0.5, 1.0]:
            c = NondimContrast(lam0=lam0, dlam=0.0, dmu=dmu, drho=0.0)
            expected = 1.0 / (1.0 + bE * dmu)
            assert E_2(c) == pytest.approx(expected, rel=1e-14)

    def test_betaE_two_forms_agree(self):
        """β_E = 2(8+3λ₀)/(15(λ₀+2)) ≡ 2(4-5ν)/(15(1-ν))."""
        for lam0 in [0.1, 7 / 9, 2.5, 10.0]:
            nu = lam0 / (2.0 * (lam0 + 1.0))
            bE_poisson = 2.0 * (4.0 - 5.0 * nu) / (15.0 * (1.0 - nu))
            assert beta_E(lam0) == pytest.approx(bE_poisson, rel=1e-14)


# =====================================================================
# Section B: analytic vs numerical Mie (the real cross-check)
# =====================================================================


@pytest.fixture(params=[1e-4, 0.01, 0.05, 0.1, 0.3])
def dmu_eps(request):
    return request.param


class TestAnalyticVsNumericalMie:
    """At small ka, our analytic closed-forms should match Python's
    numerical Mie partial-wave coefficients to leading order.

    Conventions to bridge:
      * Python's a_n carries units of length (the radius `a` in m); our
        non-dim analytic uses a=1, so we scale Python by 1/RADIUS.
      * Python applies a (-1)^n post-solve sign convention; for odd n
        we negate the analytic value to match.
    """

    def _ratio(self, num: complex, ana: complex) -> complex:
        """Convert Python's physical-units a_n to non-dim and ratio."""
        return (num / RADIUS) / ana

    def test_a0_matches_numerical(self, dmu_eps):
        contrast = MaterialContrast(
            Dlambda=REF.lam * dmu_eps,
            Dmu=REF.mu * dmu_eps,
            Drho=REF.rho * dmu_eps,
        )
        c = _physical_to_nondim(contrast)
        w = _w_from_omega(OMEGA_TEST)

        mie = compute_elastic_mie(OMEGA_TEST, RADIUS, REF, contrast)
        a0_num_nondim = complex(mie.a_n[0]) / RADIUS  # (-1)^0 = +1
        a0_ana = a_0_analytic(c, w)

        # Relative error at ka_S=0.05 (ka_P~0.03) should be O(ka²) ~ 1e-3
        assert abs(a0_num_nondim - a0_ana) / abs(a0_ana) < 0.01, (
            f"eps={dmu_eps}: nondim a0_num={a0_num_nondim}, a0_ana={a0_ana}"
        )

    def test_a1_matches_numerical(self, dmu_eps):
        contrast = MaterialContrast(
            Dlambda=REF.lam * dmu_eps,
            Dmu=REF.mu * dmu_eps,
            Drho=REF.rho * dmu_eps,
        )
        c = _physical_to_nondim(contrast)
        w = _w_from_omega(OMEGA_TEST)

        mie = compute_elastic_mie(OMEGA_TEST, RADIUS, REF, contrast)
        a1_num_nondim = complex(mie.a_n[1]) / RADIUS
        # Python applies (-1)^1 = -1 post-solve; undo to match analytic
        a1_num_undone = -a1_num_nondim
        a1_ana = a_1_analytic(c, w)

        assert abs(a1_num_undone - a1_ana) / abs(a1_ana) < 0.05

    def test_a2_matches_numerical(self, dmu_eps):
        contrast = MaterialContrast(
            Dlambda=REF.lam * dmu_eps,
            Dmu=REF.mu * dmu_eps,
            Drho=REF.rho * dmu_eps,
        )
        c = _physical_to_nondim(contrast)
        w = _w_from_omega(OMEGA_TEST)

        mie = compute_elastic_mie(OMEGA_TEST, RADIUS, REF, contrast)
        a2_num_nondim = complex(mie.a_n[2]) / RADIUS  # (-1)^2 = +1
        a2_ana = a_2_analytic(c, w)

        assert abs(a2_num_nondim - a2_ana) / abs(a2_ana) < 0.01

    def test_b2_matches_numerical(self, dmu_eps):
        contrast = MaterialContrast(
            Dlambda=REF.lam * dmu_eps,
            Dmu=REF.mu * dmu_eps,
            Drho=REF.rho * dmu_eps,
        )
        c = _physical_to_nondim(contrast)
        w = _w_from_omega(OMEGA_TEST)

        mie = compute_elastic_mie(OMEGA_TEST, RADIUS, REF, contrast)
        b2_num_nondim = complex(mie.b_n[2]) / RADIUS
        b2_ana = b_2_analytic(c, w)

        assert abs(b2_num_nondim - b2_ana) / abs(b2_ana) < 0.01


# =====================================================================
# Section C: extracted Δμ* matches Eshelby formula (the load-bearing test)
# =====================================================================


class TestEffectiveContrastExtraction:
    """Extracted Δμ* from numerical Mie should equal Δμ/(1+β_E·Δμ).

    This is the finite-contrast Eshelby concentration test: as Δμ grows,
    the extracted Δμ* deviates from the bare Δμ by exactly the
    static-Eshelby concentration factor."""

    @pytest.mark.parametrize("dmu_eps", [0.01, 0.05, 0.1, 0.2, 0.5])
    def test_Dmu_star_eshelby_corrected(self, dmu_eps):
        contrast = MaterialContrast(Dlambda=0.0, Dmu=REF.mu * dmu_eps, Drho=0.0)
        c = _physical_to_nondim(contrast)
        bE = beta_E(c.lam0)

        mie = compute_elastic_mie(OMEGA_TEST, RADIUS, REF, contrast)
        mc = mie_extract_effective_contrasts(mie)

        # Bare contrast (what a naive point-scatterer would use)
        Dmu_bare = contrast.Dmu / REF.mu

        # Eshelby-corrected (what Mie actually sees)
        Dmu_eshelby = Dmu_bare / (1.0 + bE * Dmu_bare)

        Dmu_extracted = mc.Dmu_star.real / REF.mu

        # Mie extraction should match Eshelby, NOT bare
        assert abs(Dmu_extracted - Dmu_eshelby) / abs(Dmu_eshelby) < 1e-3, (
            f"eps={dmu_eps}: extracted={Dmu_extracted}, "
            f"eshelby={Dmu_eshelby}, bare={Dmu_bare}"
        )

    @pytest.mark.parametrize("dmu_eps", [0.05, 0.1, 0.2, 0.5, 1.0])
    def test_Dmu_star_from_analytic_a2_matches_extracted(self, dmu_eps):
        """The Δμ*_from_a2 helper round-trips analytic a₂ to give Δμ_eshelby."""
        contrast = MaterialContrast(Dlambda=0.0, Dmu=REF.mu * dmu_eps, Drho=0.0)
        c = _physical_to_nondim(contrast)
        w = _w_from_omega(OMEGA_TEST)

        a2 = a_2_analytic(c, w)
        Dmu_star = complex(Dmu_star_from_a2(a2, c, w)).real

        Dmu_eshelby = c.dmu / (1.0 + beta_E(c.lam0) * c.dmu)
        assert abs(Dmu_star - Dmu_eshelby) < 1e-12


# =====================================================================
# Section D: interior near-field Eshelby concentration
# =====================================================================


class TestInteriorNearFieldEshelby:
    """Interior u_r and u_θ at r=a/2 should reduce to the static Eshelby
    interior strain at ka→0, exact at finite Δμ."""

    @pytest.mark.parametrize("dmu_eps", [0.01, 0.1, 0.5, 1.0])
    def test_n2_interior_concentration_factor(self, dmu_eps):
        """U_r^{(2)}(r) / U_r^{(2)}(r)|_{Δ=0} = 1/(1+β_E·Δμ)."""
        c = NondimContrast(lam0=7 / 9, dlam=0.0, dmu=dmu_eps, drho=0.0)
        c0 = NondimContrast(lam0=7 / 9, dlam=0.0, dmu=0.0, drho=0.0)
        r, w = 0.5, 0.05

        ratio = U_r_n2_leading(c, r, w) / U_r_n2_leading(c0, r, w)
        expected = E_2(c)
        assert abs(complex(ratio).real - expected) < 1e-14

    @pytest.mark.parametrize("dmu_eps", [0.01, 0.1, 0.5, 1.0])
    def test_n0_interior_concentration_factor(self, dmu_eps):
        """U_r^{(0)}(r) / U_r^{(0)}(r)|_{Δ=0} = E_0 (bulk Eshelby)."""
        c = NondimContrast(lam0=7 / 9, dlam=dmu_eps, dmu=dmu_eps, drho=0.0)
        c0 = NondimContrast(lam0=7 / 9, dlam=0.0, dmu=0.0, drho=0.0)
        r, w = 0.5, 0.05

        ratio = U_r_n0_leading(c, r, w) / U_r_n0_leading(c0, r, w)
        expected = E_0(c)
        assert abs(complex(ratio).real - expected) < 1e-14

    def test_ur_to_utheta_ratio_exactly_two(self):
        """U_r^{(2)} / U_θ^{(2)} = 2 exactly (canonical n=2 angular factor)."""
        c = NondimContrast(lam0=7 / 9, dlam=0.1, dmu=0.3, drho=0.05)
        r, w = 0.3, 0.05
        ratio = U_r_n2_leading(c, r, w) / U_theta_n2_leading(c, r, w)
        assert abs(complex(ratio) - 2.0) < 1e-14
