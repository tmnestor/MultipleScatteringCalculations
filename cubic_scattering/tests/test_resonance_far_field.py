"""
test_resonance_far_field.py
Convergence of resonance (multi-cell) far-field.

The resonance T-matrix (n^3 sub-cells, each 9-DOF Voigt) and the T27
(single-cell, 27-DOF Galerkin) are two *different* approximations to the
exact scattering integral.  As n increases, the resonance far-field converges
to its own (more spatially resolved) limit — not to the T27.  The ~5%
offset at moderate contrast reflects the 9-DOF vs 27-DOF basis truncation.

Tests verify:
  1. Self-convergence: |f(n+1) − f(n)| decreases monotonically
  2. Proximity: resonance stays within ~10% of T27 at moderate contrast
  3. Weak contrast: both methods agree in the Born limit (< 0.1%)
  4. Symmetry, stored fields, SH vanishing
"""

import numpy as np
import pytest

from cubic_scattering import (
    MaterialContrast,
    ReferenceMedium,
    compute_cube_tmatrix_galerkin,
    compute_resonance_tmatrix,
    resonance_far_field,
)
from cubic_scattering.incident_field import cube_overlap_integrals
from cubic_scattering.scattered_field import cube_far_field
from cubic_scattering.tmatrix_assembly import assemble_tmatrix_27

REF = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
CONTRAST = MaterialContrast(Dlambda=2e9, Dmu=1e9, Drho=100.0)
WEAK = MaterialContrast(Dlambda=REF.mu * 1e-4, Dmu=REF.mu * 1e-4, Drho=REF.rho * 1e-4)


def _t27_reference(ka: float, a: float, contrast: MaterialContrast, theta: np.ndarray):
    """T27 far-field for P-wave along axis 2."""
    omega = ka * REF.beta / a
    galerkin = compute_cube_tmatrix_galerkin(omega, a, REF, contrast)
    T27 = assemble_tmatrix_27(galerkin)
    kP = omega / REF.alpha
    k_vec = np.array([0.0, 0.0, kP])
    pol = np.array([0.0, 0.0, 1.0])
    c_inc = cube_overlap_integrals(k_vec, pol, a)
    c_sc = T27 @ c_inc
    f_P, f_SV, f_SH = cube_far_field(
        c_inc, c_sc, theta, REF, galerkin, contrast, omega, a, k_vec, pol
    )
    return f_P, f_SV, f_SH, omega, k_vec, pol


def test_resonance_self_convergence():
    """Successive differences |f(n+1) − f(n)| decrease monotonically."""
    a = 10.0
    ka = 0.05
    omega = ka * REF.beta / a
    theta = np.linspace(0, np.pi, 37)
    kP = omega / REF.alpha
    k_vec = np.array([0.0, 0.0, kP])
    pol = np.array([0.0, 0.0, 1.0])

    results = {}
    for n in [1, 2, 3, 4]:
        res = compute_resonance_tmatrix(
            omega,
            a,
            REF,
            CONTRAST,
            n_sub=n,
            k_hat=np.array([0.0, 0.0, 1.0]),
            wave_type="P",
        )
        f_P, _, _ = resonance_far_field(res, theta, REF, CONTRAST, omega, a, k_vec, pol)
        results[n] = f_P

    diffs = []
    for n in [2, 3, 4]:
        diffs.append(np.max(np.abs(results[n] - results[n - 1])))

    # Successive differences must decrease
    for i in range(len(diffs) - 1):
        assert diffs[i + 1] < diffs[i], (
            f"|f(n={i + 3}) - f(n={i + 2})| = {diffs[i + 1]:.4e} >= "
            f"|f(n={i + 2}) - f(n={i + 1})| = {diffs[i]:.4e}"
        )


def test_resonance_proximity_to_t27():
    """Resonance far-field stays within 10% of T27 at moderate contrast."""
    a = 10.0
    ka = 0.05
    omega = ka * REF.beta / a
    theta = np.linspace(0, np.pi, 37)

    f_P_ref, _, _, _, k_vec, pol = _t27_reference(ka, a, CONTRAST, theta)

    for n in [1, 2, 4]:
        res = compute_resonance_tmatrix(
            omega,
            a,
            REF,
            CONTRAST,
            n_sub=n,
            k_hat=np.array([0.0, 0.0, 1.0]),
            wave_type="P",
        )
        f_P_res, _, _ = resonance_far_field(
            res, theta, REF, CONTRAST, omega, a, k_vec, pol
        )
        err = np.max(np.abs(f_P_res - f_P_ref)) / np.max(np.abs(f_P_ref))
        assert err < 0.10, f"n={n}: resonance vs T27 error {err:.4e} exceeds 10%"


def test_resonance_weak_contrast_agrees_with_t27():
    """In the Born limit (weak contrast), resonance and T27 agree closely."""
    a = 10.0
    ka = 0.05
    omega = ka * REF.beta / a
    theta = np.linspace(0, np.pi, 37)

    f_P_ref, _, _, _, k_vec, pol = _t27_reference(ka, a, WEAK, theta)

    res = compute_resonance_tmatrix(
        omega, a, REF, WEAK, n_sub=2, k_hat=np.array([0.0, 0.0, 1.0]), wave_type="P"
    )
    f_P_res, _, _ = resonance_far_field(res, theta, REF, WEAK, omega, a, k_vec, pol)
    err = np.max(np.abs(f_P_res - f_P_ref)) / np.max(np.abs(f_P_ref))
    assert err < 0.001, f"Weak contrast error {err:.4e} exceeds 0.1%"


def test_resonance_n1_matches_composite():
    """n=1 (single sub-cell) far-field is non-zero and SH vanishes on symmetry axis."""
    a = 10.0
    ka = 0.05
    omega = ka * REF.beta / a
    theta = np.linspace(0, np.pi, 19)
    kP = omega / REF.alpha
    k_vec = np.array([0.0, 0.0, kP])
    pol = np.array([0.0, 0.0, 1.0])

    res = compute_resonance_tmatrix(
        omega, a, REF, CONTRAST, n_sub=1, k_hat=np.array([0.0, 0.0, 1.0]), wave_type="P"
    )
    f_P_res, f_SV_res, f_SH_res = resonance_far_field(
        res, theta, REF, CONTRAST, omega, a, k_vec, pol
    )

    assert np.max(np.abs(f_P_res)) > 0, "Far-field should be non-zero"
    assert np.max(np.abs(f_SH_res)) < 1e-10 * np.max(np.abs(f_P_res)), (
        "SH should vanish for P-wave along symmetry axis"
    )


def test_resonance_far_field_symmetry():
    """f_P(θ) = f_P(−θ): reflection symmetry about propagation direction."""
    a = 10.0
    ka = 0.05
    omega = ka * REF.beta / a
    kP = omega / REF.alpha
    k_vec = np.array([0.0, 0.0, kP])
    pol = np.array([0.0, 0.0, 1.0])

    res = compute_resonance_tmatrix(
        omega, a, REF, CONTRAST, n_sub=2, k_hat=np.array([0.0, 0.0, 1.0]), wave_type="P"
    )

    theta_pos = np.array([0.3, 0.6, 1.0, 1.5])
    theta_neg = -theta_pos

    f_P_pos, _, _ = resonance_far_field(
        res, theta_pos, REF, CONTRAST, omega, a, k_vec, pol
    )
    f_P_neg, _, _ = resonance_far_field(
        res, theta_neg, REF, CONTRAST, omega, a, k_vec, pol
    )

    np.testing.assert_allclose(
        f_P_pos, f_P_neg, atol=1e-15, err_msg="f_P should be symmetric under θ → −θ"
    )


def test_resonance_psi_exc_stored():
    """ResonanceTmatrixResult stores psi_exc, centres, T_loc_9x9."""
    a = 10.0
    ka = 0.05
    omega = ka * REF.beta / a

    res = compute_resonance_tmatrix(omega, a, REF, CONTRAST, n_sub=2)

    N = 2**3
    assert res.psi_exc.shape == (9 * N, 9)
    assert res.centres.shape == (N, 3)
    assert res.T_loc_9x9.shape == (9, 9)


@pytest.mark.slow
def test_resonance_convergence_sv():
    """SV far-field also converges (self-convergence check)."""
    a = 10.0
    ka = 0.05
    omega = ka * REF.beta / a
    theta = np.linspace(0.1, np.pi - 0.1, 19)
    kP = omega / REF.alpha
    k_vec = np.array([0.0, 0.0, kP])
    pol = np.array([0.0, 0.0, 1.0])

    results = {}
    for n in [1, 2, 3]:
        res = compute_resonance_tmatrix(
            omega,
            a,
            REF,
            CONTRAST,
            n_sub=n,
            k_hat=np.array([0.0, 0.0, 1.0]),
            wave_type="P",
        )
        _, f_SV, _ = resonance_far_field(
            res, theta, REF, CONTRAST, omega, a, k_vec, pol
        )
        results[n] = f_SV

    diff_12 = np.max(np.abs(results[2] - results[1]))
    diff_23 = np.max(np.abs(results[3] - results[2]))
    assert diff_23 < diff_12, (
        f"SV not self-converging: |f(3)-f(2)| = {diff_23:.4e} >= |f(2)-f(1)| = {diff_12:.4e}"
    )
