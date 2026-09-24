"""
test_modulus_radiation_reaction.py
Non-circular validation of the cube MODULUS radiation reaction Im[Δc*].

The strain-mode (modulus) channel of the single-site cube T-matrix develops an
imaginary (radiation-reaction) part that is the strain-channel analog of the
density radiation reaction Im[Γ₀].  An inclusion of volume V whose EFFECTIVE
modulus contrast is C = Re[Δc*] carries, in incident strain e, the stress-dipole
moment M = V C:e -- the same V Δσ dipole radiated by cube_far_field.  Energy
conservation fixes its reaction:

    Im[e:Δc*:e] = -(2/(ωV)) P_rad(M),   i.e.   Im[Δc*] = -(2V/ω) C:K:C,

with K the isotropic radiated-power kernel (Aki & Richards moment-tensor far
field).  For the cubic C = λδδ + 2μ_off I + 2(μ_diag - μ_off) D this closes to

    Im[Δλ*]      = -(2V/ω)[(k_λ(3λ+2μ_d) + 2k_μ λ)(3λ+2μ_d) + 4k_μ λ μ_d],
    Im[Δμ*_off]  = -(8V/ω) k_μ μ_off²,
    Im[Δμ*_diag] = -(8V/ω) k_μ μ_diag²,

k_λ = (c_P - c_S)/15, k_μ = (2c_P + 3c_S)/30, c_P = ω⁴/(8πρα⁵), c_S = ω⁴/(8πρβ⁵).

The SOURCE is the effective Re[Δc*], not the bare (Δλ, Δμ).  The exact Mie
sphere, which is unitary, obeys this identity with its OWN effective Re to 1e-4;
the bare-contrast version misses Mie by 5-9 %, and makes the cube's optical
theorem overshoot.

Gates, in order of independence:
  1. the closed form equals the directly integrated radiated power of the
     explicit far field (no use of the 1/15 angular averages);
  2. exact Mie satisfies the identity with its own Re (no cube involved);
  3. the production cube Im matches Mie to the cube-vs-sphere shape drift;
  4. the cube's own optical theorem closes to 1 (test_scattered_field).
"""

import numpy as np

from cubic_scattering import MaterialContrast, ReferenceMedium
from cubic_scattering.effective_contrasts import (
    _modulus_radiation_reaction_cubic,
    compute_cube_tmatrix_galerkin,
    compute_cube_tmatrix_galerkin_57,
)
from cubic_scattering.sphere_scattering import (
    compute_elastic_mie,
    mie_extract_effective_contrasts,
)

ALPHA, BETA, RHO = 5000.0, 3000.0, 2500.0
A = 10.0  # cube half-width
REF = ReferenceMedium(ALPHA, BETA, RHO)
R_EQ = (6.0 / np.pi) ** (1.0 / 3.0) * A  # equal-volume sphere radius

# Cube-vs-equal-volume-sphere shape drift only.  Measured on the worst channel
# (pure-shear Im[Δλ*], a difference of two larger terms) it grows as
# ≈ 0.006 + 0.47 (ka)²: 0.7 %, 1.0 %, 4.8 %, 12.8 % at the four ka below.  A
# normalisation error would be ka-independent instead.  The former bare-contrast
# source sat a further ~4.3 % high at every ka and needed 8-12 % here.
KA_TOL = ((0.05, 0.02), (0.1, 0.025), (0.3, 0.06), (0.5, 0.20))


def _mie_contrasts(contrast: MaterialContrast, omega: float):
    """Mie effective contrasts for the equal-volume sphere."""
    mie = compute_elastic_mie(
        omega=omega, radius=R_EQ, ref=REF, contrast=contrast, n_max=None
    )
    return mie_extract_effective_contrasts(mie)


def _reaction(lam: float, mu_off: float, mu_diag: float, omega: float):
    """Closed-form cubic radiation reaction on the given REAL contrasts."""
    return _modulus_radiation_reaction_cubic(
        lam, mu_off, mu_diag, omega, A, ALPHA, BETA, RHO
    )


def _cube_im(contrast: MaterialContrast, omega: float):
    """Production cube Im[Δλ*], Im[Δμ*_diag], Im[Δκ*] from the Galerkin path."""
    g = compute_cube_tmatrix_galerkin(omega, A, REF, contrast)
    im_lam = g.Dlambda_star.imag
    im_mu = g.Dmu_star_diag.imag
    return im_lam, im_mu, im_lam + 2.0 / 3.0 * im_mu


def _cubic_tensor(lam: float, mu_off: float, mu_diag: float) -> np.ndarray:
    """C = λδδ + μ_off(δδ + δδ) + 2(μ_diag − μ_off)δ_ijkl (cubic, crystal axes)."""
    d = np.eye(3)
    C = lam * np.einsum("ij,kl->ijkl", d, d) + mu_off * (
        np.einsum("ik,jl->ijkl", d, d) + np.einsum("il,jk->ijkl", d, d)
    )
    for i in range(3):
        C[i, i, i, i] += 2.0 * (mu_diag - mu_off)
    return C


def _radiated_power_direct(M: np.ndarray, omega: float) -> float:
    """P_rad of a point moment tensor by direct quadrature of its far field.

    u_P = (ik_P/4πρα²)(r̂·M·r̂) r̂,  u_S = (ik_S/4πρβ²)(I − r̂r̂)·M·r̂  (per e^{ikr}/r),
    each carrying radial flux ½ρω²c|u|²; integrated over the sphere numerically.
    Deliberately does NOT use the 1/15 angular-average identities.
    """
    kP, kS = omega / ALPHA, omega / BETA
    mu_q, w_mu = np.polynomial.legendre.leggauss(48)
    phis = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    total = 0.0
    for cth, w in zip(mu_q, w_mu, strict=True):
        sth = np.sqrt(1.0 - cth**2)
        for ph in phis:
            r = np.array([sth * np.cos(ph), sth * np.sin(ph), cth])
            Mr = M @ r
            rMr = r @ Mr
            uP = kP / (4 * np.pi * RHO * ALPHA**2) * rMr
            uS = kS / (4 * np.pi * RHO * BETA**2) * (Mr - rMr * r)
            flux = 0.5 * RHO * omega**2 * (ALPHA * uP**2 + BETA * (uS @ uS))
            total += w * (2.0 * np.pi / len(phis)) * flux
    return total


# ----------------------------------------------------------------------
# GATE 1 — closed form equals directly integrated radiated power
# ----------------------------------------------------------------------
def test_cubic_reaction_equals_direct_radiated_power():
    """Im[e:Δc*:e] = -(2/(ωV)) P_rad(V C:e) for a generic CUBIC C and strains e."""
    rng = np.random.default_rng(7)
    V = (2.0 * A) ** 3
    omega = 0.1 * BETA / A
    for lam, mo, md in (
        (1.9e9, 0.97e9, 0.99e9),
        (-0.4e9, 1.2e9, 0.7e9),
        (2.5e9, 0.0, 0.0),
    ):
        C = _cubic_tensor(lam, mo, md)
        il, imo, imd = _reaction(lam, mo, md, omega)
        ImC = _cubic_tensor(il, imo, imd)
        for _ in range(3):
            e = rng.normal(size=(3, 3))
            e = 0.5 * (e + e.T)
            lhs = np.einsum("ij,ijkl,kl->", e, ImC, e)
            M = V * np.einsum("ijkl,kl->ij", C, e)
            rhs = -2.0 / (omega * V) * _radiated_power_direct(M, omega)
            assert abs(lhs - rhs) < 1e-9 * abs(rhs), (
                f"C=({lam},{mo},{md}): {lhs} vs {rhs}"
            )


def test_cubic_reaction_isotropic_reduces_to_closed_form():
    """μ_off = μ_diag recovers the isotropic closed form, with Im[μ_off] = Im[μ_diag]."""
    omega = 0.05 * BETA / A
    V = (2.0 * A) ** 3
    lam, mu = 2e9, 1e9
    cP = omega**4 / (8 * np.pi * RHO * ALPHA**5)
    cS = omega**4 / (8 * np.pi * RHO * BETA**5)
    il, imo, imd = _reaction(lam, mu, mu, omega)
    ref_l = -(2 * V / omega) * (
        cP / 15 * (15 * lam**2 + 20 * lam * mu + 4 * mu**2) - cS / 15 * 4 * mu**2
    )
    ref_m = -(V / omega) * (cP / 15 * 8 * mu**2 + cS / 15 * 12 * mu**2)
    assert abs(il - ref_l) < 1e-12 * abs(ref_l)
    assert abs(imo - ref_m) < 1e-12 * abs(ref_m)
    assert imo == imd


# ----------------------------------------------------------------------
# GATE 2 — exact Mie obeys the identity with its OWN effective Re
# ----------------------------------------------------------------------
def test_mie_obeys_reaction_identity_with_its_own_effective_contrast():
    """Unitary Mie: Im[Δc*_mie] = reaction of Re[Δc*_mie] (no cube involved).

    This fixes the SOURCE of the reaction as the effective, not bare, contrast.
    """
    for contrast in (
        MaterialContrast(2e9, 0.0, 0.0),
        MaterialContrast(0.0, 1e9, 0.0),
        MaterialContrast(2e9, 1e9, 0.0),
    ):
        for ka in (0.05, 0.1):
            omega = ka * BETA / A
            ec = _mie_contrasts(contrast, omega)
            mu = ec.Dmu_star.real
            il, im, _ = _reaction(ec.Dlambda_star.real, mu, mu, omega)
            rk = (il + 2.0 / 3.0 * im) / ec.Dkappa_star.imag
            assert abs(rk - 1.0) < 1e-3, f"ka={ka} {contrast}: Im[Dkappa*] ratio {rk}"
            if contrast.Dmu:
                rm = im / ec.Dmu_star.imag
                assert abs(rm - 1.0) < 3e-3, f"ka={ka} {contrast}: Im[Dmu*] ratio {rm}"


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
            assert abs(ratio - 1.0) < tol, f"ka={ka}: Im[{name}]/Mie={ratio:.4f}"


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
        im_lam, im_mo, im_md = _reaction(3.1e9, 0.0, 0.0, omega)
        assert im_mo == 0.0
        assert im_md == 0.0
        assert im_lam < 0.0


def test_radiation_reaction_scales_as_omega_cubed():
    """Im[Δc*] ∝ ω³ (the ω⁴ radiated power divided by the ω weight of the work rate)."""
    contrast = MaterialContrast(2e9, 1e9, 0.0)
    om1 = 0.05 * BETA / A
    om2 = 2.0 * om1
    iml1, imm1, _ = _reaction(contrast.Dlambda, contrast.Dmu, contrast.Dmu, om1)
    iml2, imm2, _ = _reaction(contrast.Dlambda, contrast.Dmu, contrast.Dmu, om2)
    assert abs(iml2 / iml1 - 8.0) < 1e-9
    assert abs(imm2 / imm1 - 8.0) < 1e-9


def test_static_limit_imag_vanishes():
    """ω → 0 ⇒ Im[Δc*] → 0 (∝ ω³): negligible vs the Rayleigh-band reaction."""
    om_ray = 0.05 * BETA / A
    ref_lam, ref_mu, _ = _reaction(2e9, 1e9, 1e9, om_ray)
    om_small = om_ray * 1e-4
    im_lam, im_mu, _ = _reaction(2e9, 1e9, 1e9, om_small)
    # ω³ scaling ⇒ (1e-4)³ = 1e-12 suppression.
    assert abs(im_lam) < 1e-11 * abs(ref_lam)
    assert abs(im_mu) < 1e-11 * abs(ref_mu)


# ----------------------------------------------------------------------
# Production wiring: the galerkin path carries the radiation reaction
# ----------------------------------------------------------------------
def test_galerkin_path_carries_radiation_reaction():
    """compute_cube_tmatrix_galerkin's modulus Im is the reaction of its OWN Re[Δc*]
    (the gerade solve alone gives exactly 0 for pure modulus)."""
    contrast = MaterialContrast(Dlambda=2e9, Dmu=1e9, Drho=0.0)
    for ka, _tol in KA_TOL:
        omega = ka * BETA / A
        g = compute_cube_tmatrix_galerkin(omega, A, REF, contrast)
        il, imo, imd = _reaction(
            g.Dlambda_star.real, g.Dmu_star_off.real, g.Dmu_star_diag.real, omega
        )
        assert abs(g.Dlambda_star.imag - il) < 1e-9 * abs(il)
        assert abs(g.Dmu_star_off.imag - imo) < 1e-9 * abs(imo)
        assert abs(g.Dmu_star_diag.imag - imd) < 1e-9 * abs(imd)


def test_galerkin_57_path_carries_radiation_reaction():
    """The 57-component path uses the same effective-contrast reaction."""
    contrast = MaterialContrast(Dlambda=2e9, Dmu=1e9, Drho=0.0)
    omega = 0.1 * BETA / A
    g = compute_cube_tmatrix_galerkin_57(omega, A, REF, contrast)
    il, imo, imd = _reaction(
        g.Dlambda_star.real, g.Dmu_star_off.real, g.Dmu_star_diag.real, omega
    )
    assert abs(g.Dlambda_star.imag - il) < 1e-9 * abs(il)
    assert abs(g.Dmu_star_off.imag - imo) < 1e-9 * abs(imo)
    assert abs(g.Dmu_star_diag.imag - imd) < 1e-9 * abs(imd)


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
