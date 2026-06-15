"""
test_gamma0_radiation_reaction.py
Non-circular validation of the density radiation reaction Im[Gamma0].

The single-site cube T-matrix density channel is
    amp_u = 1 / (1 - omega^2 Drho Gamma0).
Its imaginary part is the radiation damping that drives multiple-scattering
attenuation and the density-dipole optical theorem.

The smooth (radiating) part of the isotropic elastodynamic Green tensor is
    G^s_{ij}(x) = delta_ij Phi(r^2) + x_i x_j Psi(r^2).
At the origin only Phi survives, so the diagonal radiation reaction is
    Im[G_11(0)] = omega/(12 pi rho) (1/alpha^3 + 2/beta^3),
the (1/3) P-wave + (2/3) S-wave diagonal projection of the radiating tensor.
This is derived independently from the spectral (Weyl) plane-wave
representation -- it does NOT reference the cube's own scattering cross
section, so the comparison is non-circular.

These tests pin:
  1. Phi(0) = phi_0^diag (``_compute_phi_diagonal``) equals the exact
     Im[G_11(0)] (the radiation reaction; the pure-S value is ~1.354x too
     large, omitting the P contribution and using S weight 1 not 2/3).
  2. Im[Gamma0] (the full cube volume integral) matches the point-dipole
     reaction V*Im[G_11(0)] to leading order in ka.
  3. Re[Gamma0] is the static Eshelby self-interaction (the defect is purely
     imaginary) and the static limit omega->0 gives Im[Gamma0] -> 0.
"""

import numpy as np

from cubic_scattering.effective_contrasts import (
    _compute_Gamma0_analytical,
    _compute_phi_diagonal,
)

ALPHA, BETA, RHO = 5000.0, 3000.0, 2500.0
A = 10.0  # cube half-width


def _exact_imG11_origin(omega: float) -> float:
    """Exact diagonal elastodynamic radiation reaction Im[G_11(0)].

    Im[G_11(0)] = omega/(12 pi rho) (1/alpha^3 + 2/beta^3), the (1/3) P-wave
    + (2/3) S-wave diagonal projection from the spectral representation
    (validated by direct angular averaging of the plane-wave Green tensor).
    """
    return omega / (12.0 * np.pi * RHO) * (1.0 / ALPHA**3 + 2.0 / BETA**3)


def test_phi0_equals_exact_radiation_reaction():
    """Phi(0) = phi_0^diag must equal the exact Im[G_11(0)] (non-circular).

    The diagonal radiating Green tensor at the origin is the (1/3) P + (2/3) S
    projection (``_compute_phi_diagonal``).  The pure-S value
    omega/(4 pi rho beta^3) (S weight 1, no P) is ~1.354x too large.
    """
    for ka in (0.05, 0.1, 0.3, 0.5):
        omega = ka * BETA / A
        phi = _compute_phi_diagonal(omega, ALPHA, BETA, RHO, 8)
        exact = _exact_imG11_origin(omega)
        rel = abs(phi[0].imag - exact) / exact
        assert rel < 1e-12, (
            f"ka={ka}: Im[phi_0^diag]={phi[0].imag:.6e} must equal exact "
            f"radiation reaction {exact:.6e} (rel err {rel:.2e}); the old pure-S "
            "value omega/(4 pi rho beta^3) is ~1.354x too large."
        )
        # phi_0^diag must remain purely imaginary (no real radiation contribution).
        assert abs(phi[0].real) < 1e-30, (
            f"ka={ka}: Re[phi_0^diag]={phi[0].real:.3e} must be exactly 0."
        )


def test_imGamma0_matches_point_dipole_reaction():
    """Im[Gamma0] -> V*Im[G_11(0)] (point-dipole reaction) to leading ka order.

    For a small scatterer the cube volume integral of the smooth Im[G_11]
    approaches V*Im[G_11(0)].  The previous Im[Gamma0] overshot this by
    ~1.354x (ka-independent), the signature of the radiation-reaction defect.
    """
    V = (2.0 * A) ** 3
    for ka, tol in ((0.05, 0.01), (0.1, 0.01), (0.3, 0.03)):
        omega = ka * BETA / A
        g0 = _compute_Gamma0_analytical(omega, A, ALPHA, BETA, RHO, 8)
        pt = V * _exact_imG11_origin(omega)
        rel = abs(g0.imag - pt) / pt
        assert rel < tol, (
            f"ka={ka}: Im[Gamma0]={g0.imag:.6e} vs point-dipole reaction "
            f"V*Im[G_11(0)]={pt:.6e} (rel err {rel:.3f} > {tol}); the old "
            "value overshot by ~1.354x."
        )


def test_reGamma0_is_static_and_imag_vanishes_at_zero_omega():
    """Re[Gamma0] is the static Eshelby self-interaction; Im -> 0 as omega->0.

    The radiation-reaction defect is PURELY imaginary: Re[Gamma0] equals the
    static value at every omega (the smooth Taylor part is pure imaginary), and
    the imaginary part vanishes in the static limit.
    """
    g_lo = _compute_Gamma0_analytical(0.01 * BETA / A, A, ALPHA, BETA, RHO, 8)
    g_hi = _compute_Gamma0_analytical(0.5 * BETA / A, A, ALPHA, BETA, RHO, 8)
    assert abs(g_lo.real - g_hi.real) < 1e-18 * abs(g_hi.real) + 1e-30, (
        f"Re[Gamma0] must be omega-independent (static Eshelby): "
        f"{g_lo.real:.10e} vs {g_hi.real:.10e}."
    )
    g_static = _compute_Gamma0_analytical(0.0, A, ALPHA, BETA, RHO, 8)
    assert abs(g_static.imag) < 1e-30, (
        f"Im[Gamma0] must vanish at omega=0; got {g_static.imag:.3e}."
    )
    assert g_static.real != 0.0, "Re[Gamma0] (static Eshelby) must be nonzero."


def test_density_optical_theorem_closes_to_one():
    """Pure-density cube optical theorem sigma_ext/sigma_sc -> 1.0.

    With only Drho contrast, only the density dipole channel scatters, so the
    optical-theorem ratio directly measures the density radiation reaction.
    The correct (1/3)P + (2/3)S Im[Gamma0] closes it to 1.0; the old pure-S
    value gave ~1.354 (energy non-conservation).
    """
    from cubic_scattering import (
        MaterialContrast,
        ReferenceMedium,
        compute_cube_tmatrix_galerkin,
    )
    from cubic_scattering.scattered_field import optical_theorem_check
    from cubic_scattering.tmatrix_assembly import assemble_tmatrix_27

    ref = ReferenceMedium(ALPHA, BETA, RHO)
    density_only = MaterialContrast(Dlambda=0.0, Dmu=0.0, Drho=100.0)
    for ka in (0.05, 0.1, 0.3, 0.5):
        omega = ka * BETA / A
        g = compute_cube_tmatrix_galerkin(omega, A, ref, density_only)
        T27 = assemble_tmatrix_27(g)
        kP = omega / ALPHA
        k_vec = np.array([0.0, 0.0, kP])
        pol = np.array([0.0, 0.0, 1.0])
        se, ss = optical_theorem_check(T27, ref, g, density_only, omega, A, k_vec, pol)
        ratio = se / ss
        assert abs(ratio - 1.0) < 0.02, (
            f"ka={ka}: density sigma_ext/sigma_sc={ratio:.5f} must close to 1.0 "
            "(energy conservation); the old pure-S Im[Gamma0] gave ~1.354."
        )
