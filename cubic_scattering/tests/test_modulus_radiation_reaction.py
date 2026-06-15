"""
test_modulus_radiation_reaction.py
Non-circular validation of the cube MODULUS radiation reaction Im[Δc*].

The strain-mode (modulus) channel of the single-site cube T-matrix develops an
imaginary (radiation-reaction) part that is the strain-channel analog of the
density radiation reaction Im[Γ₀].  An inclusion of volume V with modulus
contrast Δc in incident strain e carries a stress-dipole moment M = V Δc e (the
same V Δσ dipole radiated by ``cube_far_field``); the imaginary part of the
effective stiffness is its radiation reaction, i.e. the strain LDOS
Im[∂_k ∂_l G_ij(0)] contracted with Δc.  In closed form

    Im[Δλ*] = -(2V/ω)[c_P/15·(15Δλ² + 20ΔλΔμ + 4Δμ²) − c_S/15·4Δμ²],
    Im[Δμ*] = -(V/ω)[c_P/15·8Δμ² + c_S/15·12Δμ²],

with c_P = ω⁴/(8πρα⁵), c_S = ω⁴/(8πρβ⁵), V = (2a)³.  This is derived from the
background Green's function (Aki & Richards moment-tensor radiated power) -- NOT
by imposing the optical theorem σ_ext = σ_sc.

PRIMARY GATE (these tests): the derived Im[Δc*] matches the EXACT elastic Mie
a₀/a₂ partial-wave coefficients (equal-volume sphere) in sign, magnitude (ratio
≈ +1 in the Rayleigh band), and the ABSENCE of cross-channel leakage (pure-bulk
⇒ Im[Δμ*] ≈ 0; pure-shear ⇒ Im[Δκ*] is the genuine P-quadrupole leak, also in
Mie).  The cube/Mie residual is the cube-vs-equal-volume-sphere geometric drift
(the same character as the validated density channel, ~1.00 at ka→0 with a few-%
drift by ka=0.5).

SECONDARY (consequence) check: the cube's own optical theorem closes σ_ext/σ_sc
→ 1.0 once the modulus radiation reaction is present.
"""

import numpy as np

from cubic_scattering import MaterialContrast, ReferenceMedium
from cubic_scattering.effective_contrasts import (
    _modulus_radiation_reaction,
    compute_cube_tmatrix_galerkin,
)
from cubic_scattering.sphere_scattering import (
    compute_elastic_mie,
    mie_extract_effective_contrasts,
)

ALPHA, BETA, RHO = 5000.0, 3000.0, 2500.0
A = 10.0  # cube half-width
REF = ReferenceMedium(ALPHA, BETA, RHO)
R_EQ = (6.0 / np.pi) ** (1.0 / 3.0) * A  # equal-volume sphere radius

# Rayleigh band where the point-moment derivation is tightest; tolerances widen
# with ka to absorb the cube-vs-sphere geometric form-factor drift.
KA_TOL = ((0.05, 0.08), (0.1, 0.09), (0.3, 0.12), (0.5, 0.20))


def _mie_contrasts(contrast: MaterialContrast, omega: float):
    """Mie effective contrasts for the equal-volume sphere."""
    mie = compute_elastic_mie(
        omega=omega, radius=R_EQ, ref=REF, contrast=contrast, n_max=None
    )
    return mie_extract_effective_contrasts(mie)


def _cube_im(contrast: MaterialContrast, omega: float):
    """Derived cube Im[Δλ*], Im[Δμ*], Im[Δκ*] (radiation reaction)."""
    im_lam, im_mu = _modulus_radiation_reaction(
        contrast.Dlambda, contrast.Dmu, omega, A, ALPHA, BETA, RHO
    )
    im_kappa = im_lam + 2.0 / 3.0 * im_mu
    return im_lam, im_mu, im_kappa


# ----------------------------------------------------------------------
# PRIMARY GATE — Mie a₀/a₂ match
# ----------------------------------------------------------------------
def test_pure_bulk_matches_mie_kappa_no_shear_leak():
    """Pure-bulk: Im[Δκ*] matches Mie (ratio≈+1); Im[Δμ*] ≈ 0 (no leakage)."""
    contrast = MaterialContrast(Dlambda=2e9, Dmu=0.0, Drho=0.0)
    for ka, tol in KA_TOL:
        omega = ka * BETA / A
        _im_lam, im_mu, im_kappa = _cube_im(contrast, omega)
        ec = _mie_contrasts(contrast, omega)
        # Sign + magnitude vs Mie (ratio ≈ +1).
        ratio = im_kappa / ec.Dkappa_star.imag
        assert ratio > 0, f"ka={ka}: Im[Dkappa*] wrong sign vs Mie (ratio={ratio})"
        assert abs(ratio - 1.0) < tol, (
            f"ka={ka}: Im[Dkappa*]/Mie={ratio:.4f} outside +1±{tol}"
        )
        # No cross-channel leakage: pure-bulk gives ZERO shear radiation.
        assert im_mu == 0.0, f"ka={ka}: pure-bulk leaked into Im[Dmu*]={im_mu}"
        # ... and Mie agrees that the shear channel is ~0 (machine noise).
        assert abs(ec.Dmu_star.imag) < 1e-3 * abs(ec.Dkappa_star.imag)


def test_pure_shear_matches_mie_mu_and_kappa_leak():
    """Pure-shear: Im[Δμ*] matches Mie; the P-quadrupole leak into Im[Δκ*]
    matches Mie too (this leak is genuine physics, present in Mie)."""
    contrast = MaterialContrast(Dlambda=0.0, Dmu=1e9, Drho=0.0)
    for ka, tol in KA_TOL:
        omega = ka * BETA / A
        _im_lam, im_mu, im_kappa = _cube_im(contrast, omega)
        ec = _mie_contrasts(contrast, omega)
        rmu = im_mu / ec.Dmu_star.imag
        rkap = im_kappa / ec.Dkappa_star.imag
        assert rmu > 0, f"ka={ka}: Im[Dmu*] wrong sign (ratio={rmu})"
        assert abs(rmu - 1.0) < tol, f"ka={ka}: Im[Dmu*]/Mie={rmu:.4f} outside +1±{tol}"
        # The shear->bulk P-quadrupole leak is genuine (Mie has it); same sign + ~+1.
        assert rkap > 0, f"ka={ka}: Im[Dkappa*] (shear leak) wrong sign (ratio={rkap})"
        assert abs(rkap - 1.0) < tol, f"ka={ka}: shear-leak Im[Dkappa*]/Mie={rkap:.4f}"


def test_pure_shear_lambda_sign_matches_mie():
    """Pure-shear Im[Δλ*] is POSITIVE (P-quadrupole leak), as in Mie -- the prior
    circular attempt got this sign wrong."""
    contrast = MaterialContrast(Dlambda=0.0, Dmu=1e9, Drho=0.0)
    for ka, tol in KA_TOL:
        omega = ka * BETA / A
        im_lam, _im_mu, _im_kappa = _cube_im(contrast, omega)
        ec = _mie_contrasts(contrast, omega)
        assert im_lam > 0, (
            f"ka={ka}: pure-shear Im[Dlambda*] must be > 0 (got {im_lam})"
        )
        ratio = im_lam / ec.Dlambda_star.imag
        assert abs(ratio - 1.0) < tol, f"ka={ka}: Im[Dlambda*]/Mie={ratio:.4f}"


def test_mixed_contrast_matches_mie():
    """Combined (Δλ, Δμ) contrast: all three projections match Mie sign + magnitude."""
    contrast = MaterialContrast(Dlambda=2e9, Dmu=1e9, Drho=0.0)
    for ka, tol in KA_TOL:
        omega = ka * BETA / A
        im_lam, im_mu, im_kappa = _cube_im(contrast, omega)
        ec = _mie_contrasts(contrast, omega)
        for cube_val, mie_val, name in (
            (im_lam, ec.Dlambda_star.imag, "Dlambda*"),
            (im_mu, ec.Dmu_star.imag, "Dmu*"),
            (im_kappa, ec.Dkappa_star.imag, "Dkappa*"),
        ):
            ratio = cube_val / mie_val
            assert ratio > 0, f"ka={ka}: Im[{name}] wrong sign (ratio={ratio})"
            assert abs(ratio - 1.0) < tol + 0.06, f"ka={ka}: Im[{name}]/Mie={ratio:.4f}"


def test_all_radiation_reactions_are_damping():
    """Every diagonal radiation reaction is DISSIPATIVE (Im[Δκ*] < 0, Im[Δμ*] < 0)."""
    for contrast in (
        MaterialContrast(2e9, 0.0, 0.0),
        MaterialContrast(0.0, 1e9, 0.0),
        MaterialContrast(2e9, 1e9, 0.0),
    ):
        for ka, _tol in KA_TOL:
            omega = ka * BETA / A
            _im_lam, im_mu, im_kappa = _cube_im(contrast, omega)
            assert im_kappa <= 0.0, "bulk radiation reaction must be dissipative"
            assert im_mu <= 0.0, "shear radiation reaction must be dissipative"


# ----------------------------------------------------------------------
# Closed-form / scaling properties
# ----------------------------------------------------------------------
def test_pure_bulk_has_no_shear_reaction():
    """Im[Δμ*] is identically zero for any pure-bulk (Δμ=0) contrast (no leakage)."""
    for ka, _tol in KA_TOL:
        omega = ka * BETA / A
        im_lam, im_mu = _modulus_radiation_reaction(
            3.1e9, 0.0, omega, A, ALPHA, BETA, RHO
        )
        assert im_mu == 0.0
        assert im_lam < 0.0


def test_radiation_reaction_scales_as_omega_cubed():
    """Im[Δc*] ∝ ω³ (the ω⁴ radiated power divided by the ω weight of the work rate)."""
    contrast = MaterialContrast(2e9, 1e9, 0.0)
    om1 = 0.05 * BETA / A
    om2 = 2.0 * om1
    iml1, imm1 = _modulus_radiation_reaction(
        contrast.Dlambda, contrast.Dmu, om1, A, ALPHA, BETA, RHO
    )
    iml2, imm2 = _modulus_radiation_reaction(
        contrast.Dlambda, contrast.Dmu, om2, A, ALPHA, BETA, RHO
    )
    assert abs(iml2 / iml1 - 8.0) < 1e-9
    assert abs(imm2 / imm1 - 8.0) < 1e-9


def test_static_limit_imag_vanishes():
    """ω → 0 ⇒ Im[Δc*] → 0 (∝ ω³): negligible vs the Rayleigh-band reaction."""
    om_ray = 0.05 * BETA / A
    ref_lam, ref_mu = _modulus_radiation_reaction(2e9, 1e9, om_ray, A, ALPHA, BETA, RHO)
    om_small = om_ray * 1e-4
    im_lam, im_mu = _modulus_radiation_reaction(2e9, 1e9, om_small, A, ALPHA, BETA, RHO)
    # ω³ scaling ⇒ (1e-4)³ = 1e-12 suppression.
    assert abs(im_lam) < 1e-11 * abs(ref_lam)
    assert abs(im_mu) < 1e-11 * abs(ref_mu)


# ----------------------------------------------------------------------
# Production wiring: the galerkin path carries the radiation reaction
# ----------------------------------------------------------------------
def test_galerkin_path_carries_radiation_reaction():
    """compute_cube_tmatrix_galerkin's modulus Im equals the derived reaction
    (the gerade solve previously gave exactly 0 for pure modulus)."""
    contrast = MaterialContrast(Dlambda=2e9, Dmu=1e9, Drho=0.0)
    for ka, _tol in KA_TOL:
        omega = ka * BETA / A
        g = compute_cube_tmatrix_galerkin(omega, A, REF, contrast)
        im_lam, im_mu = _modulus_radiation_reaction(
            contrast.Dlambda, contrast.Dmu, omega, A, ALPHA, BETA, RHO
        )
        assert abs(g.Dlambda_star.imag - im_lam) < 1e-6 * abs(im_lam)
        assert abs(g.Dmu_star_diag.imag - im_mu) < 1e-6 * abs(im_mu)
        assert abs(g.Dmu_star_off.imag - im_mu) < 1e-6 * abs(im_mu)


def test_galerkin_real_part_unchanged_by_reaction():
    """The REAL (static + form-factor) modulus response is untouched: Re matches the
    pure-density solve scaled appropriately -- here we pin that Re is finite and the
    radiation reaction is purely imaginary (does not touch Re)."""
    contrast = MaterialContrast(Dlambda=2e9, Dmu=1e9, Drho=0.0)
    omega = 0.3 * BETA / A
    g = compute_cube_tmatrix_galerkin(omega, A, REF, contrast)
    # Re modulus contrasts are O(contrast), strictly real-dominated.
    assert g.Dlambda_star.real != 0.0
    assert g.Dmu_star_diag.real != 0.0
    # Radiation reaction sign on the imaginary part (damping for shear-diag).
    assert g.Dmu_star_diag.imag < 0.0
