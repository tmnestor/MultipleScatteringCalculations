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
    compute_cube_tmatrix,
    compute_cube_tmatrix_galerkin,
    compute_resonance_tmatrix,
    resonance_far_field,
)
from cubic_scattering.incident_field import cube_overlap_integrals
from cubic_scattering.resonance_tmatrix import _sub_cell_tmatrix_9x9
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
    f_P, f_SV, f_SH = cube_far_field(c_inc, c_sc, theta, REF, galerkin, contrast, omega, a, k_vec, pol)
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
        f_P_res, _, _ = resonance_far_field(res, theta, REF, CONTRAST, omega, a, k_vec, pol)
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
    f_P_res, f_SV_res, f_SH_res = resonance_far_field(res, theta, REF, CONTRAST, omega, a, k_vec, pol)

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

    f_P_pos, _, _ = resonance_far_field(res, theta_pos, REF, CONTRAST, omega, a, k_vec, pol)
    f_P_neg, _, _ = resonance_far_field(res, theta_neg, REF, CONTRAST, omega, a, k_vec, pol)

    np.testing.assert_allclose(f_P_pos, f_P_neg, atol=1e-15, err_msg="f_P should be symmetric under θ → −θ")


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
        _, f_SV, _ = resonance_far_field(res, theta, REF, CONTRAST, omega, a, k_vec, pol)
        results[n] = f_SV

    diff_12 = np.max(np.abs(results[2] - results[1]))
    diff_23 = np.max(np.abs(results[3] - results[2]))
    assert diff_23 < diff_12, (
        f"SV not self-converging: |f(3)-f(2)| = {diff_23:.4e} >= |f(2)-f(1)| = {diff_12:.4e}"
    )


# ================================================================
# The composite is a fixed operator; the far field is driven by the plane wave itself
# ================================================================

W_VOIGT = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5])


def _sinc_factor(q: np.ndarray, a: float) -> float:
    """Continuum Born form factor of the cube [-a, a]^3 over its volume: prod_i sin(q_i a) / (q_i a)."""
    return float(np.prod(np.sinc(q * a / np.pi)))


@pytest.mark.parametrize(("wave", "ka"), [("S", 0.5), ("S", 1.0), ("P", 0.5)])
def test_born_far_field_converges_to_the_cube_form_factor(wave, ka):
    """At weak contrast the far field converges, at second order, to the continuum Born integral.

    The reference is independent of the multi-cell incident field: one cell at the centre carrying the
    Born T (the STATIC cube T at weak contrast, with the actual omega^2 in the density block) times the
    analytic cube form factor prod_i sinc(q_i a), with q = k_in - k_out per channel.  The static T is
    needed because a cell's own T carries a finite-ka interior factor even at weak contrast (0.947 in
    the density channel at k_S a_cell = 0.5), which tends to 1 only as the cells shrink.  A far field
    that counts the incident linear variation twice levels off instead of converging.
    """
    a = 10.0
    omega = ka * REF.beta / a
    k_in = omega / (REF.alpha if wave == "P" else REF.beta)
    k_hat = np.array([0.0, 0.0, 1.0])
    k_vec = k_in * k_hat
    pol = k_hat if wave == "P" else np.array([1.0, 0.0, 0.0])
    theta = np.linspace(0.1, np.pi - 0.1, 9)

    def far(n):
        res = compute_resonance_tmatrix(omega, a, REF, WEAK, n_sub=n, k_hat=k_hat, wave_type=wave)
        return resonance_far_field(res, theta, REF, WEAK, omega, a, k_vec, pol)

    # Scattering plane as in resonance_far_field: perp1 = x-hat for k_hat = z-hat.
    r_hat = np.stack([np.sin(theta), np.zeros_like(theta), np.cos(theta)], axis=1)
    kP, kS = omega / REF.alpha, omega / REF.beta
    ff_P = np.array([_sinc_factor(k_vec - kP * r, a) for r in r_hat])
    ff_S = np.array([_sinc_factor(k_vec - kS * r, a) for r in r_hat])
    born = compute_resonance_tmatrix(omega, a, REF, WEAK, n_sub=1, k_hat=k_hat, wave_type=wave)
    born.T_loc_9x9 = _sub_cell_tmatrix_9x9(compute_cube_tmatrix(1e-6 * omega, a, REF, WEAK), omega, a)
    f1 = resonance_far_field(born, theta, REF, WEAK, omega, a, k_vec, pol)
    ref = (f1[0] * ff_P, f1[1] * ff_S)

    def err(f):
        return max(np.max(np.abs(g - r)) / np.max(np.abs(r)) for g, r in zip(f[:2], ref, strict=True))

    e2, e4 = err(far(2)), err(far(4))
    assert e4 < 0.3 * e2, f"not converging: err(n=2) = {e2:.3e}, err(n=4) = {e4:.3e}"
    assert e4 < 0.03, f"err(n=4) = {e4:.3e} vs the continuum Born far field"


def test_composite_is_independent_of_the_incidence():
    """T_comp is the cube's own 9x9 operator: the same for every incident direction and wave type."""
    a = 10.0
    omega = 0.5 * REF.beta / a
    oblique = np.array([0.3, 0.5, 0.81])
    t_z = compute_resonance_tmatrix(omega, a, REF, CONTRAST, n_sub=3).T_comp_9x9
    t_ob = compute_resonance_tmatrix(
        omega, a, REF, CONTRAST, n_sub=3, k_hat=oblique / np.linalg.norm(oblique), wave_type="P"
    ).T_comp_9x9
    assert np.linalg.norm(t_ob - t_z) / np.linalg.norm(t_z) < 1e-12


def test_composite_has_no_displacement_strain_coupling_by_parity():
    """A centred cube is inversion-symmetric: uniform displacement (odd) cannot drive the stress dipole
    (even), nor a uniform strain the force monopole.  At k_S a = 0.5 an incident phase breaks this."""
    a = 10.0
    omega = 0.5 * REF.beta / a
    t = compute_resonance_tmatrix(omega, a, REF, CONTRAST, n_sub=3).T_comp_9x9
    scale = np.linalg.norm(t)
    assert np.linalg.norm(t[3:, :3]) / scale < 1e-12
    assert np.linalg.norm(t[:3, 3:]) / scale < 1e-12


def test_composite_is_reciprocal_without_a_density_contrast():
    """With no density contrast the sub-cells carry no force, so the summed output is the adjoint of the
    Taylor input and T_comp W^-1 is symmetric (the reciprocity law of the single-cell T_loc).
    """
    a = 10.0
    omega = 0.5 * REF.beta / a
    con = MaterialContrast(Dlambda=2e9, Dmu=1e9, Drho=0.0)
    t = compute_resonance_tmatrix(omega, a, REF, con, n_sub=3).T_comp_9x9
    m = t @ np.linalg.inv(W_VOIGT)
    assert np.linalg.norm(m - m.T) / np.linalg.norm(m) < 1e-10


def test_plane_wave_exciting_field_is_stored():
    """The far field's own exciting field psi_pw is returned alongside psi_exc."""
    a = 10.0
    omega = 0.05 * REF.beta / a
    res = compute_resonance_tmatrix(omega, a, REF, CONTRAST, n_sub=2)
    assert res.psi_pw is not None
    assert res.psi_pw.shape == res.psi_exc.shape
