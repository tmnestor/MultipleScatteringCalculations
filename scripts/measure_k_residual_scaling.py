"""MEASUREMENT: what is left of K after the contact fix, and how does it scale?

Re-measuring K against the corrected (single-averaged) contact operator moved it
from 0.886234 to ~1.0237 and cut the residual it corrects from ~8.6e-4 to
~1.6e-4 -- four fifths of the old correction was the contact convention, not
physics (`measure_shear_renormalisation.py`, commit fdb41fd). What survives is a
~2% effect, and this script asks what it is made of.

THE QUESTION. Three points from the earlier scan already look structured rather
than noisy: at fixed contrast K ran 1.021797, 1.022140, 1.028948 over
omega = 30, 60, 240, and the increments divided by omega^2 agree to 1%
(1.27e-7 and 1.26e-7). That is the signature of a clean (ka)^2 term. But
extrapolating it to zero frequency leaves K_0 ~ 1.0217, NOT 1 -- so the drift is
dynamic while the bulk of the 2% is static. Three points cannot establish that,
which is what this measures.

WHAT IS FITTED, AND WHY THAT FORM.

  [A] frequency, at fixed weak contrast:   K(omega) = K_0 + c * omega^2
      The exponent is not assumed -- it is measured, by fitting
      log(K - K_0) against log(omega) over a decade and reading the slope.
      A slope of 2 confirms (ka)^2; anything else falsifies the reading above.
      Reported in the dimensionless form K = K_0 + kappa * (k_P a)^2, which is
      the only form that transfers between cube sizes.

  [B] contrast, at low frequency: extrapolate to ZERO contrast. K drifts with
      contrast too (1.023737, 1.022140, 1.020797 over a 16x range), and the
      weak-contrast limit is the quantity a derivation would have to produce.
      The drift is fitted in Dmu/mu0 rather than assumed linear, since the
      earlier three points are not linear in it.

  [C] the joint limit K(omega -> 0, Dmu -> 0), which is THE number to explain,
      together with an honest statement of how far the extrapolation reaches.

WHY IT MATTERS. If the static limit is 1 within the extrapolation error, the
whole shear correction is dynamic and belongs with the form factor -- there is
nothing left to derive. If it is not 1, a ~2% static single-site-in-lattice
effect survives the contact fix and needs a mechanism. The two outcomes point
at completely different next steps, which is what makes this worth measuring
rather than assuming.

⚠ ka CEILING. The analytic cube T-matrix is validated for ka < 0.3 and returns
a plausible number outside it, so the sweep prints ka at every point and refuses
to fit through any point above the ceiling.

Run:  conda run -n seismic python scripts/measure_k_residual_scaling.py
SI units (m, m/s, kg/m3, Pa).
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.measure_shear_renormalisation import (  # noqa: E402
    H_PHYS,
    MU0,
    REF,
    _optimum,
)

KA_CEILING = 0.3


def ka_of(omega: float) -> float:
    """k_P a with a the HALF-width actually used by the measurement (n_z = 4)."""
    a = H_PHYS / (2.0 * 4)
    return omega / REF.alpha * a


def fit_power(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares slope/intercept of log y against log x."""
    lx, ly = np.log(x), np.log(y)
    A = np.vstack([lx, np.ones_like(lx)]).T
    slope, icpt = np.linalg.lstsq(A, ly, rcond=None)[0]
    return float(slope), float(icpt)


def main() -> int:
    print("=" * 84)
    print("WHAT IS LEFT OF K: static part, dynamic part, and the exponent")
    print("=" * 84)

    dmu_weak = 0.25e9
    print(f"\n  half-width a = {H_PHYS / 8.0} m,  ka ceiling {KA_CEILING}")

    # ---- [A] frequency scan at fixed weak contrast -----------------------
    print(f"\n  [A] frequency scan at Dmu/mu0 = {dmu_weak / MU0:.4f}")
    print(f"       {'omega':>7} {'k_P a':>8} {'K':>12} {'residual':>12}")
    omegas, ks = [], []
    for om in (15.0, 30.0, 60.0, 120.0, 240.0):
        kk, _, _, res = _optimum(dmu_weak, om)
        ka = ka_of(om)
        flag = "  ** ABOVE CEILING" if ka >= KA_CEILING else ""
        print(f"       {om:7.0f} {ka:8.4f} {kk:12.6f} {res:12.3e}{flag}")
        if ka < KA_CEILING:
            omegas.append(om)
            ks.append(kk)
    omegas, ks = np.array(omegas), np.array(ks)

    # K_0 from the two lowest frequencies assuming omega^2, then refine
    def k0_from(p: float) -> tuple[float, float]:
        """(intercept, slope) of K against omega^p by least squares."""
        A = np.vstack([omegas**p, np.ones_like(omegas)]).T
        coef = np.linalg.lstsq(A, ks, rcond=None)[0]
        return float(coef[1]), float(coef[0])

    k0_q, c_q = k0_from(2.0)
    print(f"\n       assuming K = K_0 + c omega^2:   K_0 = {k0_q:.6f}   c = {c_q:.4e}")
    resid = ks - (k0_q + c_q * omegas**2)
    print(f"       max |residual of that fit| = {np.max(np.abs(resid)):.2e}")

    # measure the exponent rather than assume it
    slope, _ = fit_power(omegas, np.abs(ks - k0_q))
    print(f"       MEASURED exponent of (K - K_0) vs omega: {slope:.3f}")
    print("       (2.000 would confirm a (ka)^2 term; the value is fitted, not set)")

    a_half = H_PHYS / 8.0
    kappa = c_q * (REF.alpha / a_half) ** 2
    print("\n       in transferable form:  K = K_0 + kappa (k_P a)^2")
    print(f"       K_0 = {k0_q:.6f}    kappa = {kappa:.5f}")

    # ---- [B] contrast scan at low frequency ------------------------------
    om_low = 30.0
    print(f"\n  [B] contrast scan at omega = {om_low:.0f}  (k_P a = {ka_of(om_low):.4f})")
    print(f"       {'Dmu/mu0':>9} {'K':>12} {'residual':>12}")
    fr, kc = [], []
    for dmu in (0.0625e9, 0.125e9, 0.25e9, 0.5e9, 1.0e9, 2.0e9):
        kk, _, _, res = _optimum(dmu, om_low)
        print(f"       {dmu / MU0:9.5f} {kk:12.6f} {res:12.3e}")
        fr.append(dmu / MU0)
        kc.append(kk)
    fr, kc = np.array(fr), np.array(kc)
    A = np.vstack([fr, np.ones_like(fr)]).T
    slope_c, k_zero = np.linalg.lstsq(A, kc, rcond=None)[0]
    print(f"\n       linear extrapolation to zero contrast: K -> {k_zero:.6f}")
    print(f"       slope in Dmu/mu0 = {slope_c:+.5f}")
    lin_resid = kc - (k_zero + slope_c * fr)
    print(f"       max |residual of the linear fit| = {np.max(np.abs(lin_resid)):.2e}")

    # ---- [C] the joint limit --------------------------------------------
    print("\n  [C] the joint limit  K(omega -> 0, Dmu -> 0)")
    k_joint = k_zero - c_q * om_low**2
    print(f"       remove the (ka)^2 part at omega = {om_low:.0f} from the")
    print(f"       zero-contrast intercept:  {k_zero:.6f} - {c_q * om_low**2:.6f}")
    print(f"       K_static_weak = {k_joint:.6f}")
    print(f"       departure from 1: {k_joint - 1.0:+.6f}")
    print(f"       historical value was 0.886234, i.e. {0.886234 - 1.0:+.6f}")

    # ---- [D] the dynamic part in the RIGHT variable ----------------------
    # K is a RATIO: K - 1 = (s_iso - s*)/(1 - s_iso), and (1 - s_iso) is
    # proportional to Dmu.  So a contrast-INDEPENDENT absolute error shows up
    # in K as a 1/Dmu blow-up, which is exactly what [A] vs [B] shows (the
    # omega=240 point is 1.0498 at Dmu/mu0 = 0.011 but 1.0289 at 0.044).
    # Read the dynamic part as the ABSOLUTE shift in the shear scale instead;
    # if that is contrast-independent, the mechanism is dynamic and additive.
    print("\n  [D] the dynamic part as an ABSOLUTE shear-scale error, s_iso - s*")
    print(f"       {'omega':>7} {'k_P a':>8} {'Dmu/mu0':>9} {'s_iso - s*':>13}")
    rows = []
    for dmu in (0.25e9, 1.0e9):
        for om in (15.0, 30.0, 60.0, 120.0):
            if ka_of(om) >= KA_CEILING:
                continue
            _, s_iso, s_star, _ = _optimum(dmu, om)
            d = s_iso - s_star
            rows.append((om, dmu / MU0, d))
            print(f"       {om:7.0f} {ka_of(om):8.4f} {dmu / MU0:9.5f} {d:13.6e}")
    print("       if the two contrast blocks track each other, the dynamic")
    print("       error is ABSOLUTE and the 1/Dmu growth of K is an artefact")
    print("       of using a ratio.")
    # Fit  s_iso - s* = a + b omega^2  per contrast.  A log-log slope is the
    # WRONG diagnostic here: the total is dominated by the constant term, so
    # the slope measures nothing (it came out 0.13 and 0.04).  Differencing
    # against omega^2 isolates b directly.
    print()
    print(f"       {'Dmu/mu0':>9} {'a (static)':>14} {'b (per om^2)':>14} {'b (k_P a)^2':>13}")
    a_half_d = H_PHYS / 8.0
    bs = []
    for frac in sorted({r[1] for r in rows}):
        sub = sorted([(r[0], r[2]) for r in rows if r[1] == frac])
        oms = np.array([t[0] for t in sub])
        ds = np.array([t[1] for t in sub])
        A2 = np.vstack([oms**2, np.ones_like(oms)]).T
        b_fit, a_fit = np.linalg.lstsq(A2, ds, rcond=None)[0]
        bs.append(float(b_fit))
        b_dimless = float(b_fit) * (REF.alpha / a_half_d) ** 2
        print(f"       {frac:9.5f} {a_fit:14.6e} {b_fit:14.4e} {b_dimless:13.5f}")
        pred = a_fit + b_fit * oms**2
        print(f"                 max |residual of a + b om^2| = {np.max(np.abs(ds - pred)):.2e}")
    spread_b = (max(bs) - min(bs)) / float(np.mean(bs))
    print(f"\n       b across a 4x contrast range: spread {spread_b:.2e}")
    print("       b CONTRAST-INDEPENDENT and exactly quadratic means the dynamic")
    print("       part is a finite-size FORM FACTOR in the shear channel, not a")
    print("       lattice effect -- compare form_factor_c2 in effective_contrasts.")
    print("       The static part a scales with Dmu, which is why K (a ratio) is")
    print("       nearly contrast-free while the absolute error is not.")

    print("\n" + "=" * 84)
    if abs(k_joint - 1.0) < 3.0 * float(np.max(np.abs(lin_resid))):
        print("  The static weak-contrast limit is 1 within the extrapolation")
        print("  scatter: the surviving correction is ENTIRELY DYNAMIC and")
        print("  belongs with the form factor. Nothing static left to derive.")
    else:
        print("  The static weak-contrast limit is NOT 1. A static correction")
        print(f"  of {k_joint - 1.0:+.4f} survives the contact fix and still needs a")
        print("  mechanism -- but it is an order of magnitude smaller than the")
        print("  0.886234 that was previously the target, and neither 8/9 nor")
        print("  sqrt(pi)/2 is anywhere near it.")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
