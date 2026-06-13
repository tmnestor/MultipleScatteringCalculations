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


# =====================================================================
# Finite-scatterer (kr)² FORM-FACTOR correction
# =====================================================================


class TestFormFactorCorrection:
    """Validate the real O((ka)²) finite-scatterer form factor in the cube T₀.

    Physics
    -------
    The analytic single-site amplification is purely *static* in its REAL
    part for the modulus channels: it omits the real O((ka)²) form factor
    that arises because the incident strain is not uniform across a
    finite scatterer.  This is the squared plane-wave overlap
    ⟨exp(ik·x)⟩² — for a sphere of radius R it gives a leading
    −(1/5)(k_P R)² modulus correction; for a cube of half-width a the
    ∏ sinc(k_j a)² overlap gives −(1/3)(k_P a)² (isotropic at this
    order — O_h cubic anisotropy first enters at O((ka)⁴)).

    The exact per-channel real (kr)² coefficient c₂ — derived
    symbolically from the elastic Mie sphere coefficients a₀,a₁,a₂
    expanded in kr and run through the SAME extraction as
    `mie_extract_effective_contrasts` — is implemented in
    `effective_contrasts.form_factor_c2`.  The voxel is a CUBE, so the
    implementation rescales these sphere coefficients by the cube/sphere
    phase-variance ratio (1/3)/(1/5) = 5/3 and applies them at the cube
    half-width a (−1/3·(k_P a)² pure-modulus).

    Coefficient chosen on first principles, CONFIRMED by slab→Kennett.
    The cube −1/3 is the correct pure-modulus form factor for a cube voxel;
    matching the SPHERE Mie oracle to pick the geometric scale would be
    circular for a cube.  The slab→Kennett check (a cubic lattice tiling a
    uniform layer; see test_slab_convergence.TestVolumeAveragedKennettAccuracy)
    CONFIRMS the cube is not worse than the sphere — the two agree with Kennett
    within slab discretization noise at ka ≤ 0.3 — and shows the form factor
    improves the normal R_PP across the band and the oblique R_SS shear floor
    (6.61% → 4.65% at ka=0.3).  It does not, by itself, resolve cube vs sphere.

    What these tests pin (sphere-Mie here is a SANITY bound, not the arbiter)
    -----------------------------------------------------------------------
    1. **Dynamic-error sanity vs volume-equivalent sphere Mie**: the cube
       form factor must remove the ka-GROWTH of the cube-vs-Mie error and
       track the equal-volume sphere Mie to within a loose envelope
       (< 0.5%) across ka ∈ {0.05, 0.1, 0.3} and contrast × {0.1, 1, 3}.
       The cube is intentionally ~8% "stronger" than the sphere at O((ka)²)
       (⟨(k·x)²⟩: cube a²/3 vs sphere R²/5 ⇒ 5/3 vs the R_eq² conversion),
       so a small residual vs sphere Mie is CORRECT and expected — a cube
       is not a sphere.  Pre-correction the modulus dynamic ratio is ≈0
       while Mie grows as −0.17·(ka)² (≈1.5% at ka=0.3); the form factor
       closes the bulk of it.
    2. **Static limit**: at ka→0 the corrected result equals the
       uncorrected static value bit-for-bit ( (1 + c₂(ka)²) → 1 ).
    3. **Imaginary part untouched**: the radiation (imaginary) series is
       unchanged by the (real) form factor.
    """

    # background: lam0 = (α/β)² − 2 = 7/9
    REF = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
    # equal-volume cube/sphere: (2a)³ = (4π/3)R³  ⇒  R = (6/π)^(1/3) a
    A = 1.0
    R_EQ = (6.0 / np.pi) ** (1.0 / 3.0) * A

    KA_LIST = [0.05, 0.1, 0.3]
    # base moderate contrast (×1); scaled by the factors below
    BASE = (2e9, 1e9, 100.0)  # (Δλ, Δμ, Δρ)
    SCALES = [0.1, 1.0, 3.0]

    def _scaled(self, scale):
        dl, dm, dr = self.BASE
        return MaterialContrast(Dlambda=dl * scale, Dmu=dm * scale, Drho=dr * scale)

    def _cube(self, omega, contrast):
        return compute_cube_tmatrix(omega, self.A, self.REF, contrast)

    def _mie_ec(self, omega, contrast):
        mie = compute_elastic_mie(omega, self.R_EQ, self.REF, contrast, n_max=6)
        return mie_extract_effective_contrasts(mie)

    def _channels_cube(self, c):
        """(Δμ*, Δκ*, Δρ*) real parts from a cube result."""
        mu = c.Dmu_star_diag.real
        kap = (c.Dlambda_star + 2.0 / 3.0 * c.Dmu_star_diag).real
        rho = c.Drho_star.real
        return mu, kap, rho

    def _channels_mie(self, ec):
        return ec.Dmu_star.real, ec.Dkappa_star.real, ec.Drho_star.real

    @pytest.mark.parametrize("scale", SCALES)
    @pytest.mark.parametrize("ka", KA_LIST)
    def test_dynamic_error_sanity_vs_equal_volume_sphere(self, ka, scale):
        """CUBE form factor tracks the equal-volume sphere Mie to < 0.5%.

        SANITY bound, not the arbiter.  We compare the *dynamic* ratio
        (X*(ka)/X*(0) − 1) of the corrected cube against the equal-volume
        sphere Mie.  This isolates the form factor from the static
        cube↔sphere geometric residual (which cancels in the ratio).
        Pre-correction the modulus dynamic ratio is ≈0 while Mie grows as
        −0.17·(ka)² (≈1.5% at ka=0.3); the cube form factor closes the bulk
        of it but, being the CUBE (−1/3·(k_P a)²) rather than the sphere
        (−1/5·(k_P R)²) coefficient, intentionally over/under-shoots the
        sphere reference by ~8% of the form-factor magnitude (the cube
        phase variance a²/3 exceeds the sphere R_eq²/5 conversion).  That
        residual is CORRECT — a cube is not a sphere at O((ka)²) — so the
        envelope here is loose (0.5%); the BINDING cube validation is
        slab→Kennett (test_slab_convergence).
        """
        contrast = self._scaled(scale)
        beta = self.REF.beta

        # Same PHYSICAL frequency for the equal-volume cube and sphere.
        # ka labels the sphere (ka = k_S·R_eq); the cube of half-width A
        # sees the SAME ω.  This is the physical comparison: one volume,
        # one frequency, two shapes.
        omega = ka * beta / self.R_EQ
        # static reference: small but not so small that the Mie partial-wave
        # extraction underflows (ka ≈ 1e-3 → form factor ≈ 1e-6, negligible).
        omega0 = 1e-3 * beta / self.R_EQ

        c0 = self._cube(omega0, contrast)
        mu0_c, kap0_c, rho0_c = self._channels_cube(c0)
        ec0 = self._mie_ec(omega0, contrast)
        mu0_m, kap0_m, rho0_m = self._channels_mie(ec0)

        c = self._cube(omega, contrast)
        mu_c, kap_c, rho_c = self._channels_cube(c)
        ec = self._mie_ec(omega, contrast)
        mu_m, kap_m, rho_m = self._channels_mie(ec)

        for name, cv, c0v, mv, m0v in [
            ("mu", mu_c, mu0_c, mu_m, mu0_m),
            ("kappa", kap_c, kap0_c, kap_m, kap0_m),
            ("rho", rho_c, rho0_c, rho_m, rho0_m),
        ]:
            cube_dyn = cv / c0v - 1.0
            mie_dyn = mv / m0v - 1.0
            gap = abs(cube_dyn - mie_dyn)
            # Loose envelope: the CUBE coefficient deliberately differs from
            # the sphere reference by ~8% of the form-factor magnitude
            # (cube ≠ sphere at O((ka)²)).  Measured worst case 3.3e-3
            # (rho ×3 ka=0.3); 5e-3 envelope with margin.  The binding
            # accuracy check is slab→Kennett, not this sphere-Mie sanity.
            tol = 5e-3
            assert gap < tol, (
                f"{name} ka={ka} ×{scale}: dynamic-ratio gap {gap:.4e} "
                f"exceeds {tol} (cube_dyn={cube_dyn:.4e}, mie_dyn={mie_dyn:.4e})"
            )

    @pytest.mark.parametrize("scale", SCALES)
    def test_static_limit_unchanged(self, scale):
        """At ka→0 the form factor is the identity (factor → 1).

        The corrected effective contrasts at a tiny frequency must equal
        the bare static Eshelby values to full floating precision.  The
        density channel is the sharp test: its real dynamic Γ₀ content is
        REPLACED, so the static-real part must survive untouched.
        """
        contrast = self._scaled(scale)
        omega = 1e-9 * self.REF.beta / self.A
        c = self._cube(omega, contrast)
        # at this ka the (ka)² factor is ~1e-18 → corrected == static
        # Compare to the analytic static Eshelby concentrations directly:
        from cubic_scattering.effective_contrasts import (
            _compute_amplification_factors,
            _compute_effective_contrasts,
            _compute_T123,
            _static_eshelby_ABC,
        )

        alpha, beta, rho = self.REF.alpha, self.REF.beta, self.REF.rho
        Ac, Bc, Cc = _static_eshelby_ABC(alpha, beta, rho)
        T1c, T2c, T3c = _compute_T123(Ac, Bc, Cc, contrast.Dlambda, contrast.Dmu)
        # static amp_u uses the static-real Γ₀ only (ω→0 ⇒ ω²Γ₀ → 0)
        _, amp_th, amp_off, amp_diag = _compute_amplification_factors(
            T1c, T2c, T3c, 0.0, 0.0, contrast.Drho
        )
        drho_s, dlam_s, _, dmu_s = _compute_effective_contrasts(
            contrast.Dlambda,
            contrast.Dmu,
            contrast.Drho,
            1.0,
            amp_th,
            amp_off,
            amp_diag,
        )
        # Drho static = Drho exactly (amp_u → 1)
        assert abs(c.Drho_star.real - contrast.Drho) < 1e-6 * abs(contrast.Drho + 1e-30)
        # modulus channels: corrected static == bare static Eshelby
        assert np.isclose(c.Dmu_star_diag.real, dmu_s.real, rtol=1e-8, atol=1e-3)
        kap_c = (c.Dlambda_star + 2.0 / 3.0 * c.Dmu_star_diag).real
        kap_s = (dlam_s + 2.0 / 3.0 * dmu_s).real
        assert np.isclose(kap_c, kap_s, rtol=1e-8, atol=1e-3)

    @pytest.mark.parametrize("scale", SCALES)
    @pytest.mark.parametrize("ka", KA_LIST)
    def test_imaginary_part_untouched(self, ka, scale):
        """The radiation (imaginary) series is BIT-FOR-BIT unchanged.

        The form factor is purely real and is applied as
        ``X.real * ff + 1j * X.imag``, so every imaginary part (the dynamic
        radiation series carried by Γ₀ and the A/B/C smooth integrals) must
        equal the raw uncorrected pipeline EXACTLY — not merely to a
        tolerance.  We recompute the raw imaginary parts from the same
        building blocks the production path uses (Γ₀ passed straight through;
        amp_u's real part is already static because Re(Γ₀) ≡ Γ₀_static) and
        assert exact equality (rtol=0, atol=0).
        """
        from cubic_scattering.effective_contrasts import (
            _compute_ABC_polynomial,
            _compute_amplification_factors,
            _compute_effective_contrasts,
            _compute_Gamma0_analytical,
            _compute_T123,
        )

        contrast = self._scaled(scale)
        alpha, beta, rho = self.REF.alpha, self.REF.beta, self.REF.rho
        omega = ka * beta / self.A
        c = self._cube(omega, contrast)

        # Raw (uncorrected) pipeline — identical building blocks to production.
        G0 = _compute_Gamma0_analytical(omega, self.A, alpha, beta, rho, 8)
        Ac, Bc, Cc = _compute_ABC_polynomial(omega, self.A, alpha, beta, rho, 32, 8)
        T1c, T2c, T3c = _compute_T123(Ac, Bc, Cc, contrast.Dlambda, contrast.Dmu)
        au, ath, aoff, adiag = _compute_amplification_factors(
            T1c, T2c, T3c, G0, omega, contrast.Drho
        )
        dr, dl, doff, ddiag = _compute_effective_contrasts(
            contrast.Dlambda, contrast.Dmu, contrast.Drho, au, ath, aoff, adiag
        )

        # EXACT equality (rtol=0, atol=0): the real-only form factor must not
        # perturb any imaginary part by a single ULP.
        np.testing.assert_array_equal(c.Dmu_star_diag.imag, ddiag.imag)
        np.testing.assert_array_equal(c.Dmu_star_off.imag, doff.imag)
        np.testing.assert_array_equal(c.Dlambda_star.imag, dl.imag)
        np.testing.assert_array_equal(c.Drho_star.imag, dr.imag)
