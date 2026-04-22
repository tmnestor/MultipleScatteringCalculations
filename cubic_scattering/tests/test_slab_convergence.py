"""Tests for slab voxel-refinement convergence: periodic R_PP and Kennett reference."""

import numpy as np
from numpy.testing import assert_allclose

from cubic_scattering.effective_contrasts import MaterialContrast, ReferenceMedium
from cubic_scattering.slab_scattering import (
    SlabGeometry,
    compute_slab_scattering,
    compute_slab_tmatrices,
    kennett_reference_rpp,
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
