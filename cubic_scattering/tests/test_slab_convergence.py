"""Tests for slab voxel-refinement convergence: periodic R_PP and Kennett reference."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from cubic_scattering.effective_contrasts import MaterialContrast, ReferenceMedium
from cubic_scattering.slab_scattering import (
    SlabGeometry,
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
# Measured 2026-06-13 (M=4, N_z=3, a=1, H=6, normal P, periodic, gmres 1e-9),
# WITH the radiation part now included:
#
#   ka    |R_K|        err pt(VA=F)   err VA n=2   err VA n=3
#   0.10  1.7646e-02   1.4206e-02     9.5286e-03   9.5286e-03
#   0.20  3.3067e-02   2.1319e-02     1.7072e-02   1.7067e-02
#   0.30  4.4331e-02   2.7796e-02     2.3766e-02   2.3705e-02
#   0.50  4.9489e-02   3.8357e-02     3.3247e-02   3.1007e-02
#
# i.e. vol-avg matches Kennett to 0.95% / 1.71% / 2.38% / 3.32% and beats
# the point baseline at every ka.  Adding the radiation part barely moves
# the normal-incidence P result (the C/S imaginary fractions are ≲3% at
# ka ≤ 0.5, and the force/density imaginary part is small), so the numbers
# are essentially identical to the pre-radiation diagnostic
# (0.95/1.68/2.31/3.46%) — confirming the radiation fix does not regress the
# real-dominated normal-incidence accuracy.

_KA_KENNETT = (0.1, 0.2, 0.3, 0.5)
# Measured VA-vs-Kennett errors (n=2) above, with margin.
_VA_KENNETT_TOL = {0.1: 0.012, 0.2: 0.020, 0.3: 0.027, 0.5: 0.037}


def _rpp_vol_avg(geom, mat, omega, *, volume_averaged, n_orders=2):
    """Periodic normal-incidence specular R_PP for a given propagator mode."""
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
    def test_vol_avg_at_least_as_accurate_as_point(self, ka):
        """Volume averaging does not HURT vs the point propagator.

        This is the property that justifies the volume-averaging machinery
        for the multi-layer normal-incidence P response: replacing the point
        nearest-neighbour propagator with its volume average must be at least
        as accurate against Kennett (a small tolerance allows for GMRES
        round-off).  Measured margin is ~0.4–0.5% in favour of vol-avg at
        every ka.
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
            f"ka={ka}: vol-avg error {err_va:.4e} should be ≤ point error "
            f"{err_pt:.4e} (volume averaging must not hurt the normal-P response)"
        )

    @pytest.mark.parametrize("ka", _KA_KENNETT)
    def test_n_orders_convergence(self, ka):
        """n_orders=3 is at least as accurate as n_orders=2.

        The dynamic ω²ⁿ extension should converge: the ω⁶ (n=3) series must
        not be worse than the ω⁴ (n=2) series against Kennett.  Measured: the
        two are identical to ≲1e-4 at ka ≤ 0.3 and n=3 improves marginally at
        ka=0.5 (3.10% vs 3.32%).
        """
        geom, mat = self._geom_mat()
        omega = ka * REF.beta / self.A
        H = geom.d * geom.N_z
        R_K = kennett_reference_rpp(REF, CONTRAST, H=H, omega=omega)
        R_n2 = _rpp_vol_avg(geom, mat, omega, volume_averaged=True, n_orders=2)
        R_n3 = _rpp_vol_avg(geom, mat, omega, volume_averaged=True, n_orders=3)
        err_n2 = abs(R_n2 - R_K) / abs(R_K)
        err_n3 = abs(R_n3 - R_K) / abs(R_K)
        assert err_n3 <= err_n2 + 1e-4, (
            f"ka={ka}: n_orders=3 error {err_n3:.4e} should be ≤ n_orders=2 "
            f"error {err_n2:.4e} (dynamic series must converge)"
        )

    def test_oblique_vol_avg_matches_kennett(self):
        """Oblique (sub-critical p) vol-avg R_PP/R_SS match Kennett.

        Confirms the physical pitch and the complex (reactive + radiation)
        propagator work off normal incidence.  At p=1e-4, ka=0.3 (measured
        2026-06-13):

            channel   point err   vol-avg err   |R_K|
            R_PP      1.571e-02    2.028e-02     3.096e-02
            R_SS      5.010e-02    6.607e-02     2.567e-02

        FINDING: unlike the normal-incidence P response, volume averaging does
        NOT beat the point propagator off normal incidence here — the R_PP
        point/vol-avg curves cross near p≈1e-4 and R_SS vol-avg is ~1–1.5%
        worse than point across sub-critical p.  The headline beats-point
        guarantee is therefore pinned only for the normal-incidence P channel
        (where volume averaging of the vertical coupling is the dominant
        correction); off normal we assert only that vol-avg still matches
        Kennett within a loose measured envelope, not that it beats the point
        propagator.
        """
        geom, mat = self._geom_mat()
        ka = 0.3
        omega = ka * REF.beta / self.A
        H = geom.d * geom.N_z
        p = 1e-4  # sub-critical: p < 1/alpha = 2e-4
        sm = slab_reflection_matrix(
            geom, mat, omega, p=p, volume_averaged=True, n_orders=2
        )
        R_mod = sm.to_modified()
        kref = kennett_reference_matrix(REF, CONTRAST, H=H, omega=omega, p=p)
        err_pp = abs(R_mod[0, 0] - kref.R_PP) / abs(kref.R_PP)
        err_ss = abs(R_mod[1, 1] - kref.R_SS) / abs(kref.R_SS)
        # Measured 2.03% / 6.61%; envelopes with margin.
        assert err_pp < 0.03, f"oblique R_PP vol-avg vs Kennett {err_pp:.4e} > 3%"
        assert err_ss < 0.08, f"oblique R_SS vol-avg vs Kennett {err_ss:.4e} > 8%"
