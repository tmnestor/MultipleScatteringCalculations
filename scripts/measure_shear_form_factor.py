"""MEASUREMENT: is the surviving (ka)^2 shear term already in form_factor_c2?

WHAT PROMPTED IT. After the contact-operator fix, the residual shear correction
splits exactly (`measure_k_residual_scaling.py`):

    s_iso - s*  =  a(Dmu)  +  b omega^2,   b = 0.334 (k_P a)^2

with b contrast-independent to 2e-3 and the quadratic form fitting to 5e-10.
A contrast-independent (ka)^2 term is a finite-size FORM FACTOR, and
`effective_contrasts.py` already ships one: `form_factor_c2` supplies c_mu, and
line 1011 applies it as Dmu_star_diag *= ff_mu with ff_mu = 1 + c_mu (k_S a)^2.
Its docstring states the cube pure-modulus coefficient is -1/3 (k_P a)^2.

    measured b = 0.33363, 0.33431      1/3 = 0.33333

Same magnitude to 0.1-0.3%, OPPOSITE SIGN. So the question is not whether the
form factor is present -- it is, and s_iso contains it -- but whether it is
applied with the right magnitude for a voxel EMBEDDED IN A LATTICE.

WHAT IS MEASURED. s_iso and s* are fitted separately against (k_P a)^2:

    s_iso(w) = s_iso(0) (1 + C_iso (k_P a)^2)
    s*(w)    = s*(0)    (1 + C_star (k_P a)^2)

C_iso is what the library applies. C_star is what the lattice actually wants.
Their RATIO is the whole question:

    ratio ~ 1   the form factor is right and the (ka)^2 residual is something
                else entirely;
    ratio ~ 2   the lattice needs the overlap applied twice where the library
                applies it once -- a definite, checkable defect;
    anything else, it is neither, and the coincidence with 1/3 is numerology.

WHY THE RATIO RATHER THAN THE DIFFERENCE. The difference is what
`measure_k_residual_scaling` already reports; it mixes the two coefficients
with the static offset. The ratio isolates the dynamic sector and is
dimensionless, so it transfers between cube sizes and contrasts -- and it is
the quantity a factor-of-two defect would make obvious.

⚠ The analytic cube T-matrix is validated for ka < 0.3 and returns a plausible
number outside it; every point here is printed with its ka and the fit refuses
to pass through any point above the ceiling.

Run:  conda run -n seismic python scripts/measure_shear_form_factor.py
SI units (m, m/s, kg/m3, Pa).
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cubic_scattering.effective_contrasts import (  # noqa: E402
    form_factor_c2,
)
from measure_shear_renormalisation import (  # noqa: E402
    H_PHYS,
    MU0,
    REF,
    _optimum,
)

KA_CEILING = 0.3
A_HALF = H_PHYS / 8.0  # the measurement runs at n_z = 4 over H_PHYS


def ka_of(omega: float) -> float:
    return omega / REF.alpha * A_HALF


def quad_coeff(ka: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Fit y = y0 (1 + C ka^2); return (y0, C, max abs residual)."""
    A = np.vstack([ka**2, np.ones_like(ka)]).T
    slope, icpt = np.linalg.lstsq(A, y, rcond=None)[0]
    resid = float(np.max(np.abs(y - (icpt + slope * ka**2))))
    return float(icpt), float(slope / icpt), resid


def main() -> int:
    print("=" * 80)
    print("IS THE SURVIVING (ka)^2 SHEAR TERM THE FORM FACTOR?")
    print("=" * 80)

    dmu = 0.25e9
    print(f"\n  Dmu/mu0 = {dmu / MU0:.5f},  cube half-width a = {A_HALF} m")

    print(f"\n  {'omega':>7} {'k_P a':>8} {'s_iso':>12} {'s*':>12} {'s_iso - s*':>13}")
    kas, sis, sss = [], [], []
    for om in (15.0, 30.0, 60.0, 120.0, 240.0):
        _, s_iso, s_star, _ = _optimum(dmu, om)
        ka = ka_of(om)
        flag = "  ** above ceiling" if ka >= KA_CEILING else ""
        print(f"  {om:7.0f} {ka:8.4f} {s_iso:12.8f} {s_star:12.8f} {s_iso - s_star:13.6e}{flag}")
        if ka < KA_CEILING:
            kas.append(ka)
            sis.append(s_iso)
            sss.append(s_star)
    ka_a = np.asarray(kas, dtype=float)
    si_a = np.asarray(sis, dtype=float)
    ss_a = np.asarray(sss, dtype=float)

    s0_i, c_iso, r_i = quad_coeff(ka_a, si_a)
    s0_s, c_star, r_s = quad_coeff(ka_a, ss_a)

    print("\n  fits of the form  s = s(0) (1 + C (k_P a)^2)")
    print(f"    s_iso :  s(0) = {s0_i:.8f}   C_iso  = {c_iso:+.6f}   resid {r_i:.2e}")
    print(f"    s*    :  s(0) = {s0_s:.8f}   C_star = {c_star:+.6f}   resid {r_s:.2e}")
    print(f"\n    ratio C_star / C_iso = {c_star / c_iso:.6f}")

    # what the library itself says it applies
    lam0 = (REF.alpha / REF.beta) ** 2 - 2.0
    c_mu_s, _, _ = form_factor_c2(0.0, dmu / MU0, 0.0, lam0)
    cube_over_sphere = 5.0 / 3.0
    # ff_mu is written in (k_S a)^2; convert to (k_P a)^2 for comparison
    ks_over_kp2 = (REF.alpha / REF.beta) ** 2
    c_applied = cube_over_sphere * c_mu_s * ks_over_kp2
    print(f"\n  library: c_mu (sphere, (k_S R)^2 units)      = {c_mu_s:+.8f}")
    print(f"           x 5/3 and into (k_P a)^2 units        = {c_applied:+.6f}")
    print(f"           docstring's cube pure-modulus value   = {-1.0 / 3.0:+.6f}")
    print(f"  measured C_iso (should match the applied one)  = {c_iso:+.6f}")
    print(f"  measured C_star (what the lattice wants)       = {c_star:+.6f}")

    # sanity: does the measured C_iso reproduce what the library applies?
    print(f"\n  C_iso / c_applied = {c_iso / c_applied:.6f}   (1.0 confirms the")
    print("    measurement is reading the library's own form factor)")

    print("\n" + "=" * 80)
    r = c_star / c_iso
    if abs(r - 2.0) < 0.05:
        print("  C_star / C_iso = 2 within 5%.  The lattice wants the overlap")
        print("  applied TWICE where the library applies it once -- a definite")
        print("  and checkable defect, not a coincidence with 1/3.")
    elif abs(r - 1.0) < 0.05:
        print("  C_star / C_iso = 1 within 5%.  The form factor as shipped is")
        print("  what the lattice wants, and the (ka)^2 residual measured")
        print("  earlier is NOT the form factor -- look elsewhere.")
    else:
        print(f"  C_star / C_iso = {r:.4f} -- neither 1 nor 2.  The agreement of")
        print("  the residual with 1/3 is then not explained by the form factor")
        print("  being singly or doubly applied, and should be treated as")
        print("  unexplained rather than as a near-miss.")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
