"""Near-field comparison: bare-Born T₀ vs Eshelby-corrected T₀ vs exact Mie.

Setup
-----
For a single spherical inclusion under a P-wave at ka_S → 0, three
single-scatterer models predict the n=2 (shear quadrupole) scattering
amplitude:

  * **Bare T₀** (naive Born point scatterer): the scattering amplitude is
    linear in the bare contrast Δμ. This is what every "point-scatterer
    with Born T-matrix" prediction gives.

  * **Eshelby-corrected T₀** (analytic sphere T-matrix with the static
    Eshelby concentration factor pre-applied):
        Δμ → Δμ_eshelby = Δμ / (1 + β_E·Δμ)
    where β_E = 2(8+3λ₀) / (15(λ₀+2)) = 2(4-5ν) / (15(1-ν)).

  * **Exact Mie** (numerical 4×4 BC solve).

What this test proves
---------------------
At finite Δμ, the discrepancy between the bare-Born and exact Mie
predictions equals exactly (1 + β_E·Δμ) — the inverse of the Eshelby
concentration factor. The Eshelby-corrected T₀ matches Mie to O(ka²)
residual (from subleading partial waves not captured by the leading
ka→0 truncation).

This is the load-bearing finite-contrast result: a point-scatterer
model that uses bare contrasts systematically overestimates scattering
in the shear channel; the Eshelby correction is essential.

References
----------
  * Eshelby (1957) - inclusion theory, sphere concentration factors
  * Pao & Mow (1973) - elastic Mie scattering (the reference solution)
  * Mathematica/MieAsymptotic.wl - symbolic derivation of the closed
    forms used here, validated in MieAsymptoticVerify.wl (V1-V5).
  * cubic_scattering/mie_asymptotic_analytic.py - Python evaluator for
    the same closed forms, cross-checked in
    test_mie_asymptotic_analytic.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from cubic_scattering import MaterialContrast, ReferenceMedium
from cubic_scattering.effective_contrasts import compute_cube_tmatrix
from cubic_scattering.mie_asymptotic_analytic import (
    NondimContrast,
    a_2_analytic,
    beta_E,
)
from cubic_scattering.sphere_scattering import (
    compute_elastic_mie,
    mie_extract_effective_contrasts,
)

# =====================================================================
# Fixtures
# =====================================================================

REF = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
RADIUS = 10.0
KA_S_TARGET = 0.05  # w = ω·a/β
OMEGA_TEST = KA_S_TARGET * REF.beta / RADIUS

# Sweep Δμ from very weak (Born regime) to strong (~50% of background)
DMU_SWEEP = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]

# Equal-volume cube half-width: V_sphere = (4π/3)R³ = (2a)³ ⇒ a = R·(π/6)^(1/3)
A_CUBE_EQUAL_VOLUME = RADIUS * (np.pi / 6.0) ** (1.0 / 3.0)


# =====================================================================
# Two T₀ models (analytic, single-scatterer, leading order in ka)
# =====================================================================


def a_2_T0_bare(c: NondimContrast, w: float) -> complex:
    """Bare-Born T₀ scattering amplitude.

    The leading-order Born scattering from a sphere with shear contrast
    Δμ uses the bare contrast in the Eshelby-amplitude formula. This is
    obtained by linearising a_2_analytic in Δμ — equivalently, dropping
    the (1 + β_E·Δμ) denominator factor.
    """
    return complex(w**2 * 4.0 * c.dmu / (9.0 * (c.lam0 + 2.0) ** 2))


def a_2_T0_eshelby(c: NondimContrast, w: float) -> complex:
    """Eshelby-corrected T₀ scattering amplitude.

    Equivalent to applying the static Eshelby shear concentration factor
    to the contrast before the Born formula:
        Δμ → Δμ / (1 + β_E·Δμ)
    Same as `a_2_analytic`, by construction.
    """
    return a_2_analytic(c, w)


# =====================================================================
# Tests
# =====================================================================


class TestPointScattererVsMie:
    """The discriminating finite-contrast test."""

    @pytest.mark.parametrize("dmu_eps", DMU_SWEEP)
    def test_eshelby_T0_matches_mie(self, dmu_eps):
        """Eshelby-corrected T₀ reproduces Mie at finite Δμ (to O(ka²))."""
        contrast = MaterialContrast(Dlambda=0.0, Dmu=REF.mu * dmu_eps, Drho=0.0)
        c = NondimContrast.from_physical(
            REF.alpha, REF.beta, REF.rho, contrast.Dlambda, contrast.Dmu, contrast.Drho
        )
        w = OMEGA_TEST * RADIUS / REF.beta

        a2_mie = complex(compute_elastic_mie(OMEGA_TEST, RADIUS, REF, contrast).a_n[2])
        a2_mie_nondim = a2_mie / RADIUS  # bridge units

        a2_eshelby = a_2_T0_eshelby(c, w)

        rel_err = abs(a2_mie_nondim - a2_eshelby) / abs(a2_eshelby)
        # Subleading O((ka)²) ~ 2.5e-3 residual; allow a bit more
        assert rel_err < 0.01, (
            f"Δμ/μ₀={dmu_eps}: Eshelby-T₀={a2_eshelby:.4e}, "
            f"Mie={a2_mie_nondim:.4e}, rel_err={rel_err:.4e}"
        )

    @pytest.mark.parametrize("dmu_eps", DMU_SWEEP)
    def test_bare_T0_overestimates_by_eshelby_factor(self, dmu_eps):
        """Bare T₀ / Mie = (1 + β_E·Δμ) — the inverse Eshelby concentration.

        This is THE finite-contrast result. The naive point-scatterer
        prediction is too large by exactly the factor that the Eshelby
        concentration corrects.
        """
        contrast = MaterialContrast(Dlambda=0.0, Dmu=REF.mu * dmu_eps, Drho=0.0)
        c = NondimContrast.from_physical(
            REF.alpha, REF.beta, REF.rho, contrast.Dlambda, contrast.Dmu, contrast.Drho
        )
        w = OMEGA_TEST * RADIUS / REF.beta

        a2_mie = complex(compute_elastic_mie(OMEGA_TEST, RADIUS, REF, contrast).a_n[2])
        a2_mie_nondim = a2_mie / RADIUS
        a2_bare = a_2_T0_bare(c, w)

        ratio = a2_bare / a2_mie_nondim
        expected = 1.0 + beta_E(c.lam0) * c.dmu

        rel_err = abs(complex(ratio).real - expected) / expected
        # Same O(ka²) residual budget
        assert rel_err < 0.01, (
            f"Δμ/μ₀={dmu_eps}: bare/Mie ratio = {complex(ratio).real:.6f}, "
            f"expected (1+β_E·Δμ) = {expected:.6f}"
        )

    @pytest.mark.parametrize("dmu_eps", DMU_SWEEP)
    def test_bare_T0_born_limit_recovers(self, dmu_eps):
        """In the Born limit Δμ→0, bare-T₀ and Eshelby-T₀ converge."""
        contrast = MaterialContrast(Dlambda=0.0, Dmu=REF.mu * dmu_eps, Drho=0.0)
        c = NondimContrast.from_physical(
            REF.alpha, REF.beta, REF.rho, contrast.Dlambda, contrast.Dmu, contrast.Drho
        )
        w = OMEGA_TEST * RADIUS / REF.beta

        a2_bare = a_2_T0_bare(c, w)
        a2_eshelby = a_2_T0_eshelby(c, w)

        # Both agree to (1 + β_E·Δμ) — small for small Δμ
        ratio = a2_bare / a2_eshelby
        bE_dmu = beta_E(c.lam0) * c.dmu

        assert abs(complex(ratio).real - (1 + bE_dmu)) < 1e-12


class TestFiniteContrastDeparture:
    """Quantitative summary: how big is the Eshelby correction at each Δμ?

    Just prints / asserts the discrepancies at increasing contrast so
    the physics is visible from the test output.
    """

    def test_summary_table(self, capsys):
        """Print the bare-T₀ vs Eshelby-T₀ vs Mie comparison across Δμ."""
        c0 = NondimContrast.from_physical(REF.alpha, REF.beta, REF.rho, 0.0, 0.0, 0.0)
        bE = beta_E(c0.lam0)
        w = OMEGA_TEST * RADIUS / REF.beta

        print()
        print("=" * 72)
        print(f"  Finite-contrast Eshelby test  (ka_S={KA_S_TARGET}, β_E={bE:.4f})")
        print("=" * 72)
        print(
            f"  {'Δμ/μ₀':>8}  {'a₂_bare':>12}  {'a₂_Eshelby':>12}  {'a₂_Mie':>12}  "
            f"{'bare/Mie':>10}  {'1+β_E·Δμ':>10}"
        )
        print("-" * 72)

        worst_eshelby_err = 0.0
        for dmu_eps in DMU_SWEEP:
            contrast = MaterialContrast(Dlambda=0.0, Dmu=REF.mu * dmu_eps, Drho=0.0)
            c = NondimContrast.from_physical(
                REF.alpha,
                REF.beta,
                REF.rho,
                contrast.Dlambda,
                contrast.Dmu,
                contrast.Drho,
            )
            a2_mie = complex(
                compute_elastic_mie(OMEGA_TEST, RADIUS, REF, contrast).a_n[2]
            )
            a2_mie_nd = a2_mie / RADIUS
            a2_bare = a_2_T0_bare(c, w)
            a2_eshelby = a_2_T0_eshelby(c, w)

            ratio_bare_mie = (a2_bare / a2_mie_nd).real
            expected_ratio = 1.0 + bE * dmu_eps
            eshelby_err = abs((a2_eshelby - a2_mie_nd) / a2_eshelby).real
            worst_eshelby_err = max(worst_eshelby_err, eshelby_err)

            print(
                f"  {dmu_eps:>8.3f}  {a2_bare.real:>12.4e}  "
                f"{a2_eshelby.real:>12.4e}  {a2_mie_nd.real:>12.4e}  "
                f"{ratio_bare_mie:>10.4f}  {expected_ratio:>10.4f}"
            )

        print("-" * 72)
        print(f"  Worst Eshelby-T₀ vs Mie relative error: {worst_eshelby_err:.4e}")
        print(f"  (subleading O(ka²) residual; ka_S² = {KA_S_TARGET**2:.4e})")
        print("=" * 72)

        # Sanity assertion (the table is the real point)
        assert worst_eshelby_err < 0.01


# =====================================================================
# Cube 27×27 T-matrix vs Sphere Mie  (the real geometric question)
# =====================================================================


class TestCubeTMatrixVsMieSphere:
    """Compare the project's 27×27 cubic T-matrix to sphere Mie.

    The cube T-matrix from `effective_contrasts.compute_cube_tmatrix`
    captures Eshelby-like concentration through the analytic Galerkin
    closure, but for a cube — not a sphere. Two questions:

      1. **Eshelby capture**: does the cube's effective Δμ* track the
         sphere's Eshelby-corrected value (which is what Mie gives)?
      2. **Geometric error**: how big is the cube vs sphere discrepancy
         at the same volume, as a function of contrast?

    The cube has cubic anisotropy: it returns *two* shear concentrations
    (`Dmu_star_off` for the 3 off-diagonal pure-shear modes, and
    `Dmu_star_diag` for the diagonal trace-free shear). Volume-weighted
    average uses (2 × off + 1 × diag) / 3 — three modes total.

    Equal-volume cube: a_cube = R_sphere · (π/6)^(1/3).
    """

    def _cube_dmu_star_avg(self, contrast: MaterialContrast) -> float:
        """Volume-weighted average shear concentration from the cube T-matrix."""
        cube = compute_cube_tmatrix(OMEGA_TEST, A_CUBE_EQUAL_VOLUME, REF, contrast)
        return (2.0 * cube.Dmu_star_off.real + cube.Dmu_star_diag.real) / 3.0 / REF.mu

    def _mie_dmu_star(self, contrast: MaterialContrast) -> float:
        mie = compute_elastic_mie(OMEGA_TEST, RADIUS, REF, contrast)
        return mie_extract_effective_contrasts(mie).Dmu_star.real / REF.mu

    @pytest.mark.parametrize("dmu_eps", DMU_SWEEP)
    def test_cube_tmatrix_tracks_eshelby(self, dmu_eps):
        """Cube T-matrix avg-Δμ* tracks sphere Eshelby Δμ/(1+β_E·Δμ).

        At small ka, both should encode the static Eshelby concentration
        but with the relevant shape factor. The cube's per-mode
        concentrations differ from the sphere's by the geometric factor
        only — captured by the volume-weighted average.
        """
        contrast = MaterialContrast(Dlambda=0.0, Dmu=REF.mu * dmu_eps, Drho=0.0)

        dmu_cube = self._cube_dmu_star_avg(contrast)
        dmu_mie = self._mie_dmu_star(contrast)

        rel_err = abs(dmu_cube - dmu_mie) / abs(dmu_mie)
        # Geometric error budget: O(few %) at strong contrast; bigger
        # than the O(ka²) numerical residual but still small in absolute
        # terms compared to the Eshelby correction itself.
        assert rel_err < 0.01, (
            f"Δμ/μ₀={dmu_eps}: cube_avg={dmu_cube:.6e}, "
            f"sphere_mie={dmu_mie:.6e}, rel_err={rel_err:.4e}"
        )

    @pytest.mark.parametrize("dmu_eps", DMU_SWEEP)
    def test_cube_anisotropy_small_at_low_contrast(self, dmu_eps):
        """Cube off/diag shear split is small in the Born regime, grows with Δμ."""
        contrast = MaterialContrast(Dlambda=0.0, Dmu=REF.mu * dmu_eps, Drho=0.0)
        cube = compute_cube_tmatrix(OMEGA_TEST, A_CUBE_EQUAL_VOLUME, REF, contrast)

        off = cube.Dmu_star_off.real / REF.mu
        diag = cube.Dmu_star_diag.real / REF.mu
        anisotropy = (off - diag) / ((off + diag) / 2.0)

        # Anisotropy is ~ second-order in Δμ for small contrast (cubic
        # symmetry breaks linearly when contrast is non-zero, so we
        # expect anisotropy ≈ const·Δμ at small Δμ).
        assert abs(anisotropy) < 0.15, (
            f"Δμ/μ₀={dmu_eps}: cube anisotropy = {anisotropy:.4f}"
        )

    def test_cube_vs_mie_summary_table(self, capsys):
        """Print the cube-T-matrix vs Mie comparison vs the bare-Born baseline."""
        bE = beta_E(
            NondimContrast.from_physical(REF.alpha, REF.beta, REF.rho, 0, 0, 0).lam0
        )

        print()
        print("=" * 80)
        print(
            f"  Cube 27×27 T-matrix vs Mie sphere  "
            f"(equal-volume; ka_S={KA_S_TARGET}; β_E={bE:.4f})"
        )
        print("=" * 80)
        print(
            f"  {'Δμ/μ₀':>7}  {'bare':>10}  {'sphere-Eshelby':>14}  "
            f"{'Mie sphere':>11}  {'cube avg':>10}  {'cube off':>10}  "
            f"{'cube diag':>10}"
        )
        print("-" * 80)

        worst_cube_err = 0.0
        for dmu_eps in DMU_SWEEP:
            contrast = MaterialContrast(Dlambda=0.0, Dmu=REF.mu * dmu_eps, Drho=0.0)
            dmu_eshelby = dmu_eps / (1.0 + bE * dmu_eps)
            dmu_mie = self._mie_dmu_star(contrast)

            cube = compute_cube_tmatrix(OMEGA_TEST, A_CUBE_EQUAL_VOLUME, REF, contrast)
            cube_off = cube.Dmu_star_off.real / REF.mu
            cube_diag = cube.Dmu_star_diag.real / REF.mu
            cube_avg = (2.0 * cube_off + cube_diag) / 3.0

            err = abs(cube_avg - dmu_mie) / abs(dmu_mie)
            worst_cube_err = max(worst_cube_err, err)

            print(
                f"  {dmu_eps:>7.3f}  {dmu_eps:>10.4e}  {dmu_eshelby:>14.4e}  "
                f"{dmu_mie:>11.4e}  {cube_avg:>10.4e}  {cube_off:>10.4e}  "
                f"{cube_diag:>10.4e}"
            )

        print("-" * 80)
        print(f"  Worst cube-avg vs sphere-Mie relative error: {worst_cube_err:.4e}")
        print(
            f"  (geometric cube↔sphere shape error; for comparison the "
            f"bare/Mie error at Δμ/μ₀=0.5 is {bE * 0.5 * 100:.1f}%)"
        )
        print("=" * 80)

        assert worst_cube_err < 0.01
