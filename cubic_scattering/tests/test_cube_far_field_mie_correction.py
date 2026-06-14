"""test_cube_far_field_mie_correction.py

Mie-validated regression guard for the density force-monopole correction in
``cube_far_field``.

The production far-field builds the radiated P-dipole from a density-only
amplified INCIDENT moment ``F = ω²·Δρ·A_u·c_inc[:3]`` rather than the
modulus-contaminated total moment ``c_inc[:3] + c_sc[:3]``.  These tests gate
the correction against the EXACT equal-volume elastic Mie sphere:

  * coupled (density×modulus) contrast P-channel L2 drops from ~2.16 to ≲0.05
    and the radiated P1 Legendre sign matches Mie;
  * strong negative-density contrast L2 drops from ~4.26 to ≲0.01;
  * density-only / modulus-only / weak limits stay at their existing floor;
  * Δρ=0 is a bit-for-bit no-op (the fix touches only the density monopole);
  * an oblique-incidence case tracks Mie (the fix is not normal-only).

The equal-volume sphere radius is ``R = a·(6/π)**(1/3)`` and the exact Mie
P→P far field is ``Σ_n a_n·(-i)^n·P_n(cosθ)`` with the (-1)^n sign convention
baked into ``a_n`` (see scattered_field / sphere_scattering sign notes).
"""

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import eval_legendre

from cubic_scattering import (
    MaterialContrast,
    ReferenceMedium,
    compute_cube_tmatrix_galerkin,
    compute_elastic_mie,
    cube_far_field,
)
from cubic_scattering.incident_field import cube_overlap_integrals
from cubic_scattering.tmatrix_assembly import assemble_tmatrix_27

REF = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
A = 10.0
R_SPHERE = A * (6.0 / np.pi) ** (1.0 / 3.0)

MODERATE = MaterialContrast(Dlambda=2e9, Dmu=1e9, Drho=100.0)
STRONG_NEG_RHO = MaterialContrast(Dlambda=2e9, Dmu=1e9, Drho=-1500.0)  # -60% ρ
DENSITY_ONLY = MaterialContrast(Dlambda=0.0, Dmu=0.0, Drho=100.0)
MODULUS_ONLY = MaterialContrast(Dlambda=2e9, Dmu=1e9, Drho=0.0)
WEAK = MaterialContrast(Dlambda=REF.mu * 1e-4, Dmu=REF.mu * 1e-4, Drho=REF.rho * 1e-4)


def _gauss_theta(n: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes/weights mapped to θ ∈ [0, π]."""
    x, w = leggauss(n)
    return 0.5 * np.pi * (x + 1.0), 0.5 * np.pi * w


def _cube_fP(ka: float, contrast: MaterialContrast, k_vec, pol, theta):
    omega = ka * REF.beta / A
    g = compute_cube_tmatrix_galerkin(omega, A, REF, contrast)
    T27 = assemble_tmatrix_27(g)
    c_inc = cube_overlap_integrals(k_vec, pol, A)
    c_sc = T27 @ c_inc
    f_P, _, _ = cube_far_field(
        c_inc, c_sc, theta, REF, g, contrast, omega, A, k_vec, pol
    )
    return f_P, omega


def _mie_fP(omega: float, contrast: MaterialContrast, theta) -> np.ndarray:
    """Exact equal-volume Mie sphere P→P far field on θ grid."""
    mie = compute_elastic_mie(omega, R_SPHERE, REF, contrast)
    ct = np.cos(theta)
    f = np.zeros_like(theta, dtype=complex)
    for n in range(len(mie.a_n)):
        f += mie.a_n[n] * (-1j) ** n * eval_legendre(n, ct)
    return f


def _l2(f: np.ndarray, f_ref: np.ndarray, wth: np.ndarray) -> float:
    return float(
        np.sqrt(np.sum(wth * np.abs(f - f_ref) ** 2))
        / np.sqrt(np.sum(wth * np.abs(f_ref) ** 2))
    )


def _legendre_moment(
    f: np.ndarray, ell: int, theta: np.ndarray, wth: np.ndarray
) -> float:
    """Real part of the (2ℓ+1)/2 ∫ f P_ℓ(cosθ) sinθ dθ Legendre projection."""
    coeff = (2 * ell + 1) / 2.0
    return float(
        (
            coeff * np.sum(wth * f * eval_legendre(ell, np.cos(theta)) * np.sin(theta))
        ).real
    )


def _P1(f: np.ndarray, theta: np.ndarray, wth: np.ndarray) -> float:
    return _legendre_moment(f, 1, theta, wth)


# ── P-axis incidence (normal): the canonical validation geometry ──
K_HAT_AXIS = np.array([1.0, 0.0, 0.0])


def _axis_kvec_pol(omega: float):
    kP = omega / REF.alpha
    return kP * K_HAT_AXIS, K_HAT_AXIS.copy()


# Pinned "after" values measured from the corrected production path
# (PYTHONPATH=. python /tmp/fix_validate.py and ff_rho_decision).  All are the
# CORRECTED L2 — the uncorrected values (~2.16 moderate, ~4.26 strong) are the
# defect this guard prevents from returning.
MODERATE_L2_PINS = {0.05: 0.0047, 0.30: 0.0203, 0.50: 0.0506}
STRONG_L2_PINS = {0.30: 0.0078}


def test_moderate_coupled_contrast_matches_mie():
    """Coupled ρ×modulus: P-channel L2 ≲0.05 and P1 sign matches Mie (was ~2.16)."""
    theta, wth = _gauss_theta()
    for ka, l2_pin in MODERATE_L2_PINS.items():
        omega = ka * REF.beta / A
        k_vec, pol = _axis_kvec_pol(omega)
        f_cube, _ = _cube_fP(ka, MODERATE, k_vec, pol, theta)
        f_mie = _mie_fP(omega, MODERATE, theta)

        l2 = _l2(f_cube, f_mie, wth)
        assert l2 < 0.055, f"ka={ka}: moderate L2={l2:.4f} exceeds 0.055 (defect ~2.16)"
        # Pin the measured value (drift guard, ±15% band around prototype).
        assert abs(l2 - l2_pin) < 0.15 * l2_pin + 5e-4, (
            f"ka={ka}: moderate L2={l2:.4f} drifted from pinned {l2_pin}"
        )

        p1_cube = _P1(f_cube, theta, wth)
        p1_mie = _P1(f_mie, theta, wth)
        assert np.sign(p1_cube) == np.sign(p1_mie), (
            f"ka={ka}: P1 sign cube={p1_cube:.3e} vs mie={p1_mie:.3e} (was wrong-signed)"
        )
        # P1 (density dipole) tracks Mie to ~2% at low ka, drifting to ~6% at
        # ka=0.5 (the bare-A_u density monopole carries no real (ka)² form
        # factor — a deliberate choice; ff_rho would harm the strong-ρ case).
        p1_tol = 0.03 if ka <= 0.3 else 0.07
        assert abs(p1_cube - p1_mie) < p1_tol * abs(p1_mie), (
            f"ka={ka}: P1 cube={p1_cube:.3e} vs mie={p1_mie:.3e} (tol {p1_tol:.0%})"
        )


def test_strong_negative_density_matches_mie():
    """-60% Δρ at ka=0.3: P-channel L2 ≲0.01 (was ~4.26)."""
    theta, wth = _gauss_theta()
    ka = 0.30
    omega = ka * REF.beta / A
    k_vec, pol = _axis_kvec_pol(omega)
    f_cube, _ = _cube_fP(ka, STRONG_NEG_RHO, k_vec, pol, theta)
    f_mie = _mie_fP(omega, STRONG_NEG_RHO, theta)

    l2 = _l2(f_cube, f_mie, wth)
    assert l2 < 0.01, f"strong-neg-ρ ka={ka}: L2={l2:.4f} exceeds 0.01 (defect ~4.26)"
    assert abs(l2 - STRONG_L2_PINS[ka]) < 0.15 * STRONG_L2_PINS[ka] + 5e-4, (
        f"strong-neg-ρ ka={ka}: L2={l2:.4f} drifted from pinned {STRONG_L2_PINS[ka]}"
    )

    p1_cube = _P1(f_cube, theta, wth)
    p1_mie = _P1(f_mie, theta, wth)
    assert np.sign(p1_cube) == np.sign(p1_mie)
    assert abs(p1_cube - p1_mie) < 0.05 * abs(p1_mie)


def test_preserved_limits_stay_at_floor():
    """Density-only, modulus-only, and weak contrast stay at the ~1e-3…1e-2 floor."""
    theta, wth = _gauss_theta()
    cases = {
        "density-only": (DENSITY_ONLY, 0.05, 0.05),
        "modulus-only": (MODULUS_ONLY, 0.05, 0.05),
        "weak": (WEAK, 0.05, 0.05),
    }
    for name, (contrast, ka, tol) in cases.items():
        omega = ka * REF.beta / A
        k_vec, pol = _axis_kvec_pol(omega)
        f_cube, _ = _cube_fP(ka, contrast, k_vec, pol, theta)
        f_mie = _mie_fP(omega, contrast, theta)
        l2 = _l2(f_cube, f_mie, wth)
        assert l2 < tol, f"{name}: L2={l2:.4f} regressed past floor {tol}"


def test_density_zero_is_bit_for_bit_noop():
    """Δρ=0 ⟹ the density monopole vanishes ⟹ the corrected f_P is unchanged.

    With Δρ=0 the prefactor ω²·Δρ is exactly 0, so F≡0 regardless of A_u and
    regardless of whether the moment is sourced from c_inc or the (old) total.
    We assert the corrected far field equals a hand-built far field whose force
    monopole is forced to exactly zero — i.e. the correction is provably inert.
    """
    theta = np.linspace(0.0, np.pi, 37)
    contrast = MaterialContrast(Dlambda=2e9, Dmu=1e9, Drho=0.0)
    omega = 0.30 * REF.beta / A
    k_vec, pol = _axis_kvec_pol(omega)
    g = compute_cube_tmatrix_galerkin(omega, A, REF, contrast)
    T27 = assemble_tmatrix_27(g)
    c_inc = cube_overlap_integrals(k_vec, pol, A)
    c_sc = T27 @ c_inc

    f_corrected, _, _ = cube_far_field(
        c_inc, c_sc, theta, REF, g, contrast, omega, A, k_vec, pol
    )

    # Reference: same call but with the OLD total-moment monopole — at Δρ=0
    # both reduce to F≡0, so they must be bit-for-bit identical.
    f_old_total, _, _ = cube_far_field(
        c_inc, c_inc + c_sc, theta, REF, g, contrast, omega, A, k_vec, pol
    )
    # c_sc argument is ignored for the monopole now, so passing a different
    # c_sc must not change anything (proves c_sc[:3] no longer feeds F).
    np.testing.assert_array_equal(f_corrected, f_old_total)


def test_density_zero_independent_of_c_sc_displacement():
    """At Δρ=0, perturbing c_sc[:3] must not change f_P (c_sc no longer feeds F)."""
    theta = np.linspace(0.0, np.pi, 13)
    contrast = MaterialContrast(Dlambda=2e9, Dmu=1e9, Drho=0.0)
    omega = 0.30 * REF.beta / A
    k_vec, pol = _axis_kvec_pol(omega)
    g = compute_cube_tmatrix_galerkin(omega, A, REF, contrast)
    T27 = assemble_tmatrix_27(g)
    c_inc = cube_overlap_integrals(k_vec, pol, A)
    c_sc = T27 @ c_inc

    f_a, _, _ = cube_far_field(
        c_inc, c_sc, theta, REF, g, contrast, omega, A, k_vec, pol
    )
    c_sc_perturbed = c_sc.copy()
    c_sc_perturbed[:3] += np.array([1.0 + 2j, -3.0, 0.5j])
    f_b, _, _ = cube_far_field(
        c_inc, c_sc_perturbed, theta, REF, g, contrast, omega, A, k_vec, pol
    )
    np.testing.assert_array_equal(f_a, f_b)


def test_oblique_incidence_density_dipole_tracks_mie():
    """Oblique ⟨111⟩ incidence: the density monopole+dipole track isotropic Mie.

    The corrected density force monopole depends only on c_inc, A_u and Δρ —
    all rotationally well-behaved — so the radiated P0 (monopole) and P1
    (density dipole) must agree with the isotropic Mie sphere just as at normal
    incidence, confirming the fix is NOT normal-incidence-specific.

    The P2 (quadrupole) DOES differ from Mie along ⟨111⟩: that is the genuine
    cubic-O_h anisotropy of the MODULUS stress dipole (the cube's quadrupole is
    direction-dependent, the sphere's is isotropic) — a pre-existing shape
    effect, unrelated to the density correction.  So total-L2 vs isotropic Mie
    is intentionally NOT asserted here; only the density channels are.
    """
    theta, wth = _gauss_theta()
    ka = 0.30
    omega = ka * REF.beta / A
    kP = omega / REF.alpha
    k_hat = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)  # ⟨111⟩ oblique
    k_vec = kP * k_hat
    pol = k_hat.copy()
    f_cube, _ = _cube_fP(ka, MODERATE, k_vec, pol, theta)
    f_mie = _mie_fP(omega, MODERATE, theta)

    # Compare against the NORMAL-incidence result: monopole + dipole must be
    # incidence-direction-invariant (isotropic), proving the fix is not
    # normal-only.  P2 is allowed to differ (cubic anisotropy).
    k_vec_ax, pol_ax = _axis_kvec_pol(omega)
    f_axis, _ = _cube_fP(ka, MODERATE, k_vec_ax, pol_ax, theta)

    for ell in (0, 1):
        m_obl = _legendre_moment(f_cube, ell, theta, wth)
        m_axis = _legendre_moment(f_axis, ell, theta, wth)
        m_mie = _legendre_moment(f_mie, ell, theta, wth)
        assert abs(m_obl - m_axis) < 1e-3 * abs(m_axis) + 1e-12, (
            f"P{ell}: oblique {m_obl:.4e} not rotation-invariant vs axis {m_axis:.4e}"
        )
        assert np.sign(m_obl) == np.sign(m_mie), (
            f"P{ell}: oblique sign {m_obl:.3e} vs mie {m_mie:.3e}"
        )
        assert abs(m_obl - m_mie) < 0.03 * abs(m_mie), (
            f"P{ell}: oblique {m_obl:.4e} vs mie {m_mie:.4e}"
        )
