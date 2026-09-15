"""Tests for slab voxel-refinement convergence: periodic R_PP and Kennett reference."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from cubic_scattering.effective_contrasts import MaterialContrast, ReferenceMedium
from cubic_scattering.slab_scattering import (
    SlabGeometry,
    build_slab_kernels,
    compute_slab_scattering,
    compute_slab_tmatrices,
    kennett_reference_matrix,
    kennett_reference_rpp,
    slab_reflection_matrix,
    slab_rpp_periodic,
    uniform_slab_material,
)

# ── Shared fixtures ──────────────────────────────────────────────

REF = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
CONTRAST = MaterialContrast(Dlambda=2e9, Dmu=1e9, Drho=100.0)
WEAK_CONTRAST = MaterialContrast(
    Dlambda=CONTRAST.Dlambda * 1e-4,
    Dmu=CONTRAST.Dmu * 1e-4,
    Drho=CONTRAST.Drho * 1e-4,
)
A = 1.0
OMEGA = 0.05 * REF.beta / A  # ka_S = 0.05 → ω = 150 rad/s
K_HAT = np.array([1.0, 0.0, 0.0])


# ── 1. test_rpp_periodic_zero_contrast ───────────────────────────


def test_rpp_periodic_zero_contrast():
    """R_PP = 0 for zero-contrast slab."""
    geom = SlabGeometry(M=4, N_z=1, a=A)
    zero = MaterialContrast(0.0, 0.0, 0.0)
    mat = uniform_slab_material(geom, REF, zero)
    result = compute_slab_scattering(geom, mat, OMEGA, K_HAT)
    T_local = compute_slab_tmatrices(geom, mat, OMEGA)
    R_PP = slab_rpp_periodic(result, T_local)
    assert abs(R_PP) < 1e-20


# ── 2. test_rpp_periodic_born_scaling ────────────────────────────


def test_rpp_periodic_born_scaling():
    """Doubling weak contrast approximately doubles |R_PP|."""
    geom = SlabGeometry(M=4, N_z=1, a=A)

    mat1 = uniform_slab_material(geom, REF, WEAK_CONTRAST)
    res1 = compute_slab_scattering(geom, mat1, OMEGA, K_HAT)
    T1 = compute_slab_tmatrices(geom, mat1, OMEGA)
    R1 = slab_rpp_periodic(res1, T1)

    double = MaterialContrast(
        Dlambda=2 * WEAK_CONTRAST.Dlambda,
        Dmu=2 * WEAK_CONTRAST.Dmu,
        Drho=2 * WEAK_CONTRAST.Drho,
    )
    mat2 = uniform_slab_material(geom, REF, double)
    res2 = compute_slab_scattering(geom, mat2, OMEGA, K_HAT)
    T2 = compute_slab_tmatrices(geom, mat2, OMEGA)
    R2 = slab_rpp_periodic(res2, T2)

    if abs(R1) > 1e-30:
        ratio = abs(R2) / abs(R1)
        assert_allclose(ratio, 2.0, rtol=0.05)


# ── 3. test_rpp_periodic_vs_existing ─────────────────────────────


def test_rpp_periodic_weak_matches_kennett():
    """Weak-contrast multi-layer FL R_PP matches Kennett within 5%."""
    geom = SlabGeometry(M=4, N_z=2, a=A)
    mat = uniform_slab_material(geom, REF, WEAK_CONTRAST)
    result = compute_slab_scattering(geom, mat, OMEGA, K_HAT)
    T_local = compute_slab_tmatrices(geom, mat, OMEGA)

    R_FL = slab_rpp_periodic(result, T_local)
    R_K = kennett_reference_rpp(REF, WEAK_CONTRAST, H=geom.d * geom.N_z, omega=OMEGA)

    rel_err = abs(R_FL - R_K) / abs(R_K)
    assert rel_err < 0.05, f"Relative error {rel_err:.4f} exceeds 5%"


# ── 4. test_kennett_reference_zero_contrast ──────────────────────


def test_kennett_reference_zero_contrast():
    """Kennett R_PP = 0 for identical layers."""
    zero = MaterialContrast(0.0, 0.0, 0.0)
    R = kennett_reference_rpp(REF, zero, H=2.0, omega=OMEGA)
    assert abs(R) < 1e-12


# ── 5. test_kennett_reference_impedance ──────────────────────────


def test_kennett_reference_impedance():
    """Kennett R_PP scales linearly with weak contrast (Born regime)."""
    R1 = kennett_reference_rpp(REF, WEAK_CONTRAST, H=2.0, omega=OMEGA)
    assert abs(R1) > 0, "R_PP should be nonzero for nonzero contrast"

    double = MaterialContrast(
        Dlambda=2 * WEAK_CONTRAST.Dlambda,
        Dmu=2 * WEAK_CONTRAST.Dmu,
        Drho=2 * WEAK_CONTRAST.Drho,
    )
    R2 = kennett_reference_rpp(REF, double, H=2.0, omega=OMEGA)

    ratio = abs(R2) / abs(R1)
    assert_allclose(ratio, 2.0, rtol=0.02)


# ── 6. test_single_layer_weak_matches_kennett ────────────────────


def test_single_layer_weak_matches_kennett():
    """FL R_PP within 15% of Kennett at ka_S=0.05, M=8."""
    a = 1.0
    geom = SlabGeometry(M=8, N_z=1, a=a)
    mat = uniform_slab_material(geom, REF, CONTRAST)
    result = compute_slab_scattering(geom, mat, OMEGA, K_HAT, gmres_tol=1e-8)
    T_local = compute_slab_tmatrices(geom, mat, OMEGA)
    R_FL = slab_rpp_periodic(result, T_local)

    H = geom.d  # single layer thickness
    R_K = kennett_reference_rpp(REF, CONTRAST, H=H, omega=OMEGA)

    rel_err = abs(R_FL - R_K) / abs(R_K)
    assert rel_err < 0.15, f"Relative error {rel_err:.3f} exceeds 15%"


# ── 7. test_convergence_error_decreases ──────────────────────────


def test_convergence_error_decreases():
    """Error monotonically decreases over 3 single-layer refinement levels."""
    a_values = [2.0, 1.0, 0.5]
    M_values = [4, 8, 16]
    errors: list[float] = []

    for a, M in zip(a_values, M_values, strict=True):
        geom = SlabGeometry(M=M, N_z=1, a=a)
        mat = uniform_slab_material(geom, REF, CONTRAST)
        result = compute_slab_scattering(geom, mat, OMEGA, K_HAT, gmres_tol=1e-8)
        T_local = compute_slab_tmatrices(geom, mat, OMEGA)
        R_FL = slab_rpp_periodic(result, T_local)

        H = geom.d
        R_K = kennett_reference_rpp(REF, CONTRAST, H=H, omega=OMEGA)

        rel_err = abs(R_FL - R_K) / abs(R_K)
        errors.append(rel_err)

    for i in range(len(errors) - 1):
        assert errors[i + 1] < errors[i], (
            f"Error did not decrease: a={a_values[i]}→{a_values[i + 1]}, "
            f"err={errors[i]:.4e}→{errors[i + 1]:.4e}"
        )


# ── 8. test_periodic_uniform_slab_closer_to_kennett ──────────


def test_periodic_uniform_slab_closer_to_kennett():
    """Periodic mode gives ≤ aperiodic error vs Kennett at M=4."""
    geom = SlabGeometry(M=4, N_z=1, a=A)
    mat = uniform_slab_material(geom, REF, CONTRAST)
    H = geom.d
    R_K = kennett_reference_rpp(REF, CONTRAST, H=H, omega=OMEGA)

    # Aperiodic (default)
    res_ap = compute_slab_scattering(geom, mat, OMEGA, K_HAT, gmres_tol=1e-8)
    T_local = compute_slab_tmatrices(geom, mat, OMEGA)
    R_ap = slab_rpp_periodic(res_ap, T_local)
    err_ap = abs(R_ap - R_K) / abs(R_K)

    # Periodic
    res_p = compute_slab_scattering(
        geom, mat, OMEGA, K_HAT, gmres_tol=1e-8, periodic=True
    )
    T_local_p = compute_slab_tmatrices(geom, mat, OMEGA)
    R_p = slab_rpp_periodic(res_p, T_local_p)
    err_p = abs(R_p - R_K) / abs(R_K)

    assert err_p <= err_ap + 1e-10, (
        f"Periodic error {err_p:.4e} should be ≤ aperiodic error {err_ap:.4e}"
    )


# ═══════════════════════════════════════════════════════════════
#  End-to-end Kennett validation of the volume-averaged path
# ═══════════════════════════════════════════════════════════════
#
# These tests give compute_slab_scattering(volume_averaged=True) its first
# genuine accuracy check against the exact Kennett reference, now that the
# volume-averaged inter-voxel propagator is physically complete (H
# engineering factor, corner-S C3 symmetry, physical pitch threading, and
# the radiation/imaginary part all fixed on branch feature/propagator-fixes).
#
# A UNIFORM N_z = 3 slab is used so vertical inter-voxel coupling — the
# nearest-neighbour separations the volume-averaged propagator replaces — is
# exercised.  The slab voxels themselves stay sub-resonant (ka ≤ 0.5).
#
# Measured 2026-06-14 (M=4, N_z=3, a=1, H=6, normal P, periodic, gmres 1e-9),
# WITH the finite-scatterer FORM-FACTOR correction in the single-site cube
# T-matrix carried to O((ka)⁴): the CUBE O((ka)² form factor (−1/3·(k_P a)²,
# the real ∏ sinc(k_j a)² squared overlap; effective_contrasts.form_factor_c2)
# PLUS the exact Mie-reconstructed O((ka)⁴) coefficient (form_factor_c4) with
# the cube 77/27 geometric scaling.  The slab specular response is orientation-
# averaged, so the ISOTROPIC c₄ is used (k_hat=None — the cubic O_h anisotropy
# averages out and does not enter the normal-incidence specular R_PP).  This
# slab→Kennett check (a cubic lattice tiling a uniform layer) CONFIRMS the cube
# is not worse than the sphere; it does NOT by itself resolve cube vs sphere
# (the differences are within coarse-mesh discretization noise):
#
#   ka    |R_K|        err pt(VA=F)   err VA n=2   err VA n=3
#   0.10  1.7646e-02   1.2065e-02     7.5810e-03   7.5810e-03
#   0.20  3.3067e-02   1.3993e-02     1.1215e-02   1.1212e-02
#   0.30  4.4331e-02   1.5735e-02     1.5919e-02   1.5917e-02
#   0.50  4.9489e-02   1.7022e-02     2.1426e-02   2.3373e-02
#
# i.e. vol-avg matches Kennett to 0.76% / 1.12% / 1.59% / 2.14%.  The O((ka)²)
# form factor supplied the dominant real (ka)² content; the new O((ka)⁴) term
# improves the band edge (ka=0.5: 2.25% c₂-only → 2.14% c₂+c₄, toward the
# geometric-only 1.94%) while leaving ka ≤ 0.3 undisturbed:
#
#   ka    err VA n=2  (no-FF → sphere-c₂ → cube-c₂ → cube-c₂+c₄)
#   0.10   0.95%  →  0.77%  →  0.76%  →  0.76%
#   0.20   1.71%  →  1.15%  →  1.12%  →  1.12%
#   0.30   2.38%  →  1.58%  →  1.59%  →  1.59%
#   0.50   3.32%  →  1.82%  →  2.25%  →  2.14%
#
# The full Mie c₄ (Poisson-dynamic content, not geometric-only) beats the
# c₂-only baseline at ka=0.5 and is far below the no-FF 3.32%.  The robust
# end-to-end result remains the OBLIQUE R_SS shear-floor improvement (6.61% →
# 4.65%; see test_oblique_vol_avg_matches_kennett).

_KA_KENNETT = (0.1, 0.2, 0.3, 0.5)
# Measured VA-vs-Kennett errors (n=2) above, with margin (cube c₂+c₄ form factor).
# Re-baselined 2026-09-15 onto the EXACT lattice sum (`lattice_ewald=True`) and
# the SINGLE-average contact operator (`contact_average='single'`, the default).
#
# Two separate moves, kept separate here so neither hides the other:
#
#   ka    truncated   exact/double   exact/SINGLE   tol
#   0.1   1.0e-2      5.90e-4        4.19e-4        6.3e-4
#   0.2   1.4e-2      3.39e-4        1.21e-3        1.8e-3
#   0.3   1.8e-2      1.78e-3        2.62e-3        3.9e-3
#   0.5   2.4e-2      8.80e-3        7.81e-3        1.2e-2
#
# Leaving the truncated path (10-40x) is an unambiguous gain. Moving to the
# single average TIGHTENS ka = 0.1 and 0.5 but LOOSENS ka = 0.2 and 0.3, and
# that is NOT hidden: at a coarse mesh the double average benefits from the
# T-matrix's own O((ka)^6) truncation being partly cancelled by the propagator
# error, so it flatters at mid ka. The default was chosen on the REFINEMENT
# ladder, where that cancellation is absent because ka_cell -> 0: the double
# average saturates (1.47e-3 -> 2.91e-3 over n_z = 1..8) while the single
# converges (1.89e-3 -> 5.70e-4), in BOTH channels. See
# `slab_scattering._contact_operator`.
#
# Pinned at ~1.5x the measured error, so drift is caught without being brittle.
_VA_KENNETT_TOL = {0.1: 6.3e-4, 0.2: 1.8e-3, 0.3: 3.9e-3, 0.5: 1.2e-2}


def _rpp_vol_avg(geom, mat, omega, *, volume_averaged, n_orders=2):
    """Periodic normal-incidence specular R_PP, on the EXACT lattice sum.

    ⚠ `lattice_ewald=True` IS LOAD-BEARING, not a tuning flag. Without it the
    lateral sum is TRUNCATED -- it covers only dx, dy in [-(M-1), M-1] and wraps
    -- and at M = 4 roughly 74% of the measured error is that artifact. Every
    number this class asserts is ~1e-3 or smaller, i.e. an order below the
    artifact, so on the truncated path these tests were adjudicating a defect
    rather than the quantity named in their docstrings.

    That was not hypothetical. Three findings documented in this class were
    artifacts of the truncated sum and INVERT on the exact one:
      * "point beats VA at ka >= 0.3"  -- VA now beats point at EVERY ka;
      * the oblique R_PP "documented regression" above the no-FF baseline
        -- it closes (3.37% -> 1.98%, below the 2.03% baseline);
      * "n=3 is worse than n=2 at ka=0.5" -- n=3 is now better.
    """
    kernel_hat = build_slab_kernels(
        geom,
        omega,
        mat.ref,
        periodic=True,
        volume_averaged=volume_averaged,
        n_orders=n_orders,
        lattice_ewald=True,
    )
    res = compute_slab_scattering(
        geom,
        mat,
        omega,
        K_HAT,
        wave_type="P",
        gmres_tol=1e-9,
        volume_averaged=volume_averaged,
        n_orders=n_orders,
        periodic=True,
        kernel_hat=kernel_hat,
    )
    T_local = compute_slab_tmatrices(geom, mat, omega)
    return slab_rpp_periodic(res, T_local, p=0.0)


@pytest.mark.slow
class TestVolumeAveragedKennettAccuracy:
    """End-to-end accuracy of the volume-averaged path vs exact Kennett."""

    M = 4
    N_z = 3
    A = 1.0  # cube half-width; H = N_z * 2a = 6 m

    def _geom_mat(self):
        geom = SlabGeometry(M=self.M, N_z=self.N_z, a=self.A)
        mat = uniform_slab_material(geom, REF, CONTRAST)
        return geom, mat

    @pytest.mark.parametrize("ka", _KA_KENNETT)
    def test_vol_avg_matches_kennett(self, ka):
        """Vol-avg R_PP matches Kennett within the measured tolerance.

        Headline assertion: the volume-averaged path reproduces the exact
        Kennett R_PP for a uniform multi-layer slab across the sub-resonant
        band ka ∈ {0.1, 0.2, 0.3, 0.5} at normal incidence.
        """
        geom, mat = self._geom_mat()
        omega = ka * REF.beta / self.A
        H = geom.d * geom.N_z
        R_K = kennett_reference_rpp(REF, CONTRAST, H=H, omega=omega)
        R_va = _rpp_vol_avg(geom, mat, omega, volume_averaged=True, n_orders=2)
        rel_err = abs(R_va - R_K) / abs(R_K)
        tol = _VA_KENNETT_TOL[ka]
        assert rel_err < tol, (
            f"ka={ka}: vol-avg vs Kennett rel-err {rel_err:.4e} exceeds {tol}"
        )

    @pytest.mark.parametrize("ka", _KA_KENNETT)
    def test_vol_avg_beats_point_at_every_ka(self, ka):
        """Vol-avg beats the point propagator STRICTLY at every ka.

        ⚠ THIS ASSERTION WAS STRENGTHENED ON 2026-09-15, and the reason is a
        correction rather than an improvement in the physics.

        The previous version split at ka = 0.3, asserting only a
        "within-envelope" property above it, because the point mode measured
        marginally BETTER there (0.30: point 1.57% vs VA 1.59%; 0.50: point
        1.70% vs VA 2.14%).  That split was honest about what was measured --
        but what was measured was the TRUNCATED lateral sum, not the
        propagator.  On the exact sum (`lattice_ewald=True`, see `_rpp_vol_avg`)
        the crossover does not exist:

            ka    err VA      err point    ratio
            0.10  5.90e-4     4.29e-3      7.3x
            0.20  3.39e-4     5.17e-3      15.3x
            0.30  1.78e-3     6.70e-3      3.8x
            0.50  8.80e-3     1.24e-2      1.4x

        So volume averaging is strictly more accurate across the whole band,
        and the documented high-ka "coarse-mesh effect" was an artifact of the
        truncation. The margin is kept at 1e-4 -- the claim is strict, not
        envelope-based.
        """
        geom, mat = self._geom_mat()
        omega = ka * REF.beta / self.A
        H = geom.d * geom.N_z
        R_K = kennett_reference_rpp(REF, CONTRAST, H=H, omega=omega)
        R_va = _rpp_vol_avg(geom, mat, omega, volume_averaged=True, n_orders=2)
        R_pt = _rpp_vol_avg(geom, mat, omega, volume_averaged=False, n_orders=2)
        err_va = abs(R_va - R_K) / abs(R_K)
        err_pt = abs(R_pt - R_K) / abs(R_K)
        assert err_va <= err_pt + 1e-4, (
            f"ka={ka}: vol-avg error {err_va:.4e} should STRICTLY beat point "
            f"{err_pt:.4e} — strict at EVERY ka on the exact lattice sum"
        )

    @pytest.mark.parametrize("ka", _KA_KENNETT)
    def test_n_orders_convergence(self, ka):
        """n_orders=3 is at least as accurate as n_orders=2 for ka ≤ 0.3.

        The dynamic ω²ⁿ extension converges: the ω⁶ (n=3) series is not worse
        than the ω⁴ (n=2) series against Kennett through ka ≤ 0.3 (the two are
        identical to ≲1e-4).  Once the single-site CUBE form-factor correction
        supplies the dominant real (ka)² content, the marginal n=3 real
        higher-order term is no longer monotonically helpful at the very top
        of the band (ka=0.5: n=2 2.14%, n=3 2.34%); both remain far below the
        no-form-factor n=2/n=3 errors (3.32% / 3.10%).  We therefore pin
        convergence for ka ≤ 0.3 and bound the n=3 excursion at ka=0.5.
        """
        geom, mat = self._geom_mat()
        omega = ka * REF.beta / self.A
        H = geom.d * geom.N_z
        R_K = kennett_reference_rpp(REF, CONTRAST, H=H, omega=omega)
        R_n2 = _rpp_vol_avg(geom, mat, omega, volume_averaged=True, n_orders=2)
        R_n3 = _rpp_vol_avg(geom, mat, omega, volume_averaged=True, n_orders=3)
        err_n2 = abs(R_n2 - R_K) / abs(R_K)
        err_n3 = abs(R_n3 - R_K) / abs(R_K)
        margin = 1e-4 if ka <= 0.3 else 3e-3
        assert err_n3 <= err_n2 + margin, (
            f"ka={ka}: n_orders=3 error {err_n3:.4e} should be ≤ n_orders=2 "
            f"error {err_n2:.4e} + {margin} (dynamic series convergence)"
        )

    def test_oblique_vol_avg_matches_kennett(self):
        """Oblique (sub-critical p) vol-avg R_PP/R_SS match Kennett.

        Confirms the physical pitch and the complex (reactive + radiation)
        propagator work off normal incidence.  At p=1e-4, ka=0.3, measured
        across the three form-factor prescriptions (2026-06-13):

            channel   vol-avg err  (no-FF → sphere-FF → CUBE-FF)   |R_K|
            R_PP      2.028e-02  →  3.259e-02  →  3.372e-02        3.096e-02
            R_SS      6.607e-02  →  4.782e-02  →  4.649e-02        2.567e-02

        FINDING (R_SS): the form factor IMPROVES the shear floor by ~2 pts
        (6.61% → 4.65%).  The cube and sphere R_SS (4.65% vs 4.78%) agree
        within slab discretization noise on this coarse M=4 mesh, so this does
        NOT by itself decide cube vs sphere — the cube is chosen on physical
        first-principles grounds (correct voxel shape) and Kennett merely
        CONFIRMS it is not worse.  The clear, robust result is that the missing
        real (ka)² form factor drove the R_SS floor and the correction removes
        most of it.

        FINDING (R_PP, oblique) — DOCUMENTED REGRESSION, asserted explicitly:
        the R_PP crossover at this COARSE mesh (M=4, pd≈1 rad per cell at
        p=1e-4) shifts the OTHER way: no-FF 2.03% → cube-FF 3.37%.  Making the
        single-site response more correct exposes a residual that the point
        propagator's coupling error had been partly cancelling (the 2026-06-13
        block-isolation root-cause) — an in-plane-Bloch-phase / per-voxel
        artifact that mesh refinement (pd→0) closes, not a core-T-matrix defect
        (the normal-incidence R_PP improves at every ka ≤ 0.3, _VA_KENNETT_TOL).
        We do NOT loosen a threshold to hide this: the assertions below pin the
        R_PP regression in a tight band (3.3–3.7%) and assert it is ABOVE the
        no-FF baseline, so the regression is visible and any further drift —
        or its closing — is caught.
        """
        geom, mat = self._geom_mat()
        ka = 0.3
        omega = ka * REF.beta / self.A
        H = geom.d * geom.N_z
        p = 1e-4  # sub-critical: p < 1/alpha = 2e-4
        kernel_hat = build_slab_kernels(
            geom, omega, mat.ref, periodic=True, volume_averaged=True,
            n_orders=2, lattice_ewald=True,
        )
        sm = slab_reflection_matrix(
            geom, mat, omega, p=p, volume_averaged=True, n_orders=2,
            kernel_hat=kernel_hat,
        )
        R_mod = sm.to_modified()
        kref = kennett_reference_matrix(REF, CONTRAST, H=H, omega=omega, p=p)
        err_pp = abs(R_mod[0, 0] - kref.R_PP) / abs(kref.R_PP)
        err_ss = abs(R_mod[1, 1] - kref.R_SS) / abs(kref.R_SS)
        # ═══ RE-BASELINED 2026-09-15 ONTO THE EXACT LATTICE SUM ═══════════════
        # Everything above this line was measured on the TRUNCATED spatial path.
        # On the exact sum both channels improve sharply and the R_PP
        # "documented regression" CLOSES:
        #
        #     channel   truncated   exact     note
        #     R_PP      3.37e-2     1.98e-2   now BELOW the 2.03% no-FF baseline
        #     R_SS      4.65e-2     1.29e-2   3.6x better
        #
        # The old test asserted `err_pp > 0.0203` and pinned a 3.3-3.7% band,
        # with the docstring saying: "if this now PASSES below baseline, the
        # artifact closed and the comment is stale". It has, and it was — the
        # artifact was the truncated lateral sum, not a per-voxel Bloch-phase
        # effect. The narrative above is kept as the record of what was believed
        # when it was measured through the truncation.
        assert err_ss < 0.016, (
            f"oblique R_SS vol-avg vs Kennett {err_ss:.4e} > 1.6% "
            f"(exact-lattice-sum baseline 1.29e-2; was 4.65% truncated)"
        )
        assert err_pp < 0.024, (
            f"oblique R_PP vol-avg vs Kennett {err_pp:.4e} > 2.4% "
            f"(exact-lattice-sum baseline 1.98e-2; was 3.37% truncated). The "
            f"old above-baseline regression assertion is GONE: it closed with "
            f"the exact lattice sum."
        )
