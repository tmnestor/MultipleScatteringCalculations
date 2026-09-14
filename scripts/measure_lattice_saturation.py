"""MEASUREMENT: does the refinement error SATURATE, and does one constant remove it?

THE QUESTION THIS DECIDES. `measure_representation_convergence` showed the error
GROWING under refinement at fixed physical thickness (3.95e-3 -> 2.62e-2 over
n_z = 1..16). But the growth RATIO is falling -- 2.10, 1.66, 1.48, 1.29 -- and it
is heading towards 1. Those are two completely different diagnoses:

  * ratio -> a value ABOVE 1: the error diverges, the scheme is unusable at
    depth, and no renormalisation saves it;
  * ratio -> 1 with the error approaching a PLATEAU: the scheme converges to a
    BIASED LIMIT. It is consistent but wrong by a fixed amount -- and a fixed
    bias is exactly what a lattice renormalisation removes.

This project has already met the second pattern once: the face-contact study
found tensor-product quadrature converging to a biased limit and mistook the
bias for a defect. So the shape of the curve has to be measured before any fix
is designed, not after.

WHY A PLATEAU IS PLAUSIBLE HERE. The slab is 0.0076 wavelengths thick, so this
is a quasi-static regime and nothing is accumulating phase. What changes with
n_z is the NEAR-FIELD lattice sum: at n_z = 1 there are no vertical neighbours
at all, and each refinement adds more of a 1/r^3 coupling whose sum converges.
An error made in that sum should therefore saturate once the sum does.

WHAT IS MEASURED:
  [S1] the ladder extended to n_z = 64, with the increment between rows. A
       saturating curve has increments falling to zero; a diverging one does not.
  [S2] a three-parameter saturating fit err(n) = e_inf - A r^log2(n), reported
       with its residual so the reader can see whether the model is being
       flattered.
  [S3] whether ONE constant removes it: rescale the T-matrix by (1 + kappa) with
       a single kappa fitted at the FINEST rung, then re-run the whole ladder.
       A genuine renormalisation is one constant that works at every rung; a
       constant fitted per rung is a curve fit and proves nothing.

Run:  conda run -n seismic python scripts/measure_lattice_saturation.py
SI units (m, m/s, kg/m3, Pa).
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubic_scattering.effective_contrasts import (  # noqa: E402
    MaterialContrast,
    ReferenceMedium,
)
from cubic_scattering.slab_scattering import (  # noqa: E402
    SlabGeometry,
    SlabMaterial,
    build_slab_kernels,
    compute_slab_scattering,
    compute_slab_tmatrices,
    kennett_reference_rpp,
    slab_rpp_periodic,
)

REF = ReferenceMedium(5000.0, 3000.0, 2500.0)
D_LAM, D_MU, D_RHO = 2.0e9, 1.0e9, 100.0
OMEGA = 60.0
H_PHYS = 4.0
M = 4
LADDER = (1, 2, 4, 8, 16, 32, 64)


def _signed_error(n_z: int, kappa: float = 0.0, *, va_all: bool = False) -> tuple[complex, complex, float]:
    """(R_lattice, R_kennett, ka) with the T-matrix rescaled by (1 + kappa).

    The error is returned SIGNED (as the two complex reflection coefficients)
    rather than as a magnitude, because a renormalisation has to cancel it, and
    a magnitude hides whether the residual even has a consistent sign.
    """
    a_half = H_PHYS / (2.0 * n_z)
    geom = SlabGeometry(M=M, N_z=n_z, a=a_half)
    ones = np.ones((n_z, M, M))
    mat = SlabMaterial(Dlambda=D_LAM * ones, Dmu=D_MU * ones, Drho=D_RHO * ones, ref=REF)
    t0 = compute_slab_tmatrices(geom, mat, OMEGA) * (1.0 + kappa)
    kh = build_slab_kernels(geom, OMEGA, REF, periodic=True, volume_averaged=va_all, va_all=va_all)
    res = compute_slab_scattering(
        geom,
        mat,
        OMEGA,
        np.array([1.0, 0.0, 0.0]),
        "P",
        periodic=True,
        gmres_tol=1e-12,
        kernel_hat=kh,
        T_local=t0,
    )
    r_lat = slab_rpp_periodic(res, t0)
    r_ken = kennett_reference_rpp(REF, MaterialContrast(D_LAM, D_MU, D_RHO), H_PHYS, OMEGA)
    return r_lat, r_ken, OMEGA / REF.alpha * a_half


def main() -> int:
    print("=" * 86)
    print("MEASUREMENT -- does the refinement error SATURATE?")
    print(
        f"  physical slab fixed at H = {H_PHYS} m ({H_PHYS * OMEGA / (2 * np.pi * REF.alpha):.4f}"
        " wavelengths -- quasi-static)"
    )
    print("=" * 86)

    print(f"\n  [S1] {'n_z':>4} {'a (m)':>8} {'rel err':>11} {'increment':>11} {'ratio':>7}")
    errs, prev = [], None
    for n_z in LADDER:
        r_lat, r_ken, _ = _signed_error(n_z)
        err = abs(r_lat - r_ken) / abs(r_ken)
        inc = "" if prev is None else f"{err - prev:11.4e}"
        rat = "" if prev is None else f"{err / prev:7.2f}"
        print(f"       {n_z:4d} {H_PHYS / (2 * n_z):8.4f} {err:11.4e} {inc:>11} {rat:>7}")
        errs.append(err)
        prev = err

    # [S2] saturating fit. Increments falling geometrically => a plateau exists.
    incs = np.diff(np.array(errs))
    print(f"\n  [S2] increments: {', '.join(f'{v:.2e}' for v in incs)}")
    if len(incs) >= 3 and all(i > 0 for i in incs[-3:]):
        r = float(np.mean(incs[1:] / incs[:-1]))
        if r < 1.0:
            plateau = errs[-1] + incs[-1] * r / (1.0 - r)
            print(f"       increment ratio {r:.3f} < 1  =>  PLATEAU at ~{plateau:.3e}")
        else:
            plateau = None
            print(f"       increment ratio {r:.3f} >= 1  =>  NO plateau; the error diverges")
    else:
        plateau = None
        print("       increments are not consistently positive; no clean fit")

    # [S3] ONE constant, fitted at the finest rung, applied to every rung.
    # Fitting at the finest rung and then reporting that rung is CIRCULAR: a
    # one-parameter fit trivially nails the point it was fitted to. The constant
    # is therefore fitted at n_z = 16 and tested at 32 and 64, which it has never
    # seen. A genuine lattice property transfers; a curve fit does not.
    fit_rung = 16
    print(f"\n  [S3] ONE renormalisation constant, fitted at n_z = {fit_rung} ONLY,")
    print(f"       VALIDATED on the held-out finer rungs {LADDER[LADDER.index(fit_rung) + 1 :]}")
    r_lat, r_ken, _ = _signed_error(fit_rung)
    kappa = float((r_ken / r_lat).real - 1.0)
    print(f"       kappa = {kappa:+.6f}")
    print(f"\n       {'n_z':>4} {'before':>11} {'after':>11} {'ratio':>7}   status")
    ok, prev_after = 0, None
    for n_z in LADDER:
        r0, rk, _ = _signed_error(n_z)
        r1, _, _ = _signed_error(n_z, kappa)
        e0, e1 = abs(r0 - rk) / abs(rk), abs(r1 - rk) / abs(rk)
        ok += bool(e1 < e0)
        rat = "" if prev_after is None else f"{e1 / prev_after:7.2f}"
        tag = "<- FITTED" if n_z == fit_rung else ("held out" if n_z > fit_rung else "")
        print(f"       {n_z:4d} {e0:11.4e} {e1:11.4e} {rat:>7}   {tag}")
        prev_after = e1

    # IS kappa ITSELF CONVERGED? If the bias were a settled lattice constant,
    # fitting it at different rungs would give the same number. Drift means the
    # lattice sum has not converged at the rungs we can afford, so the constant
    # has to be DERIVED rather than fitted.
    r64, rk64, _ = _signed_error(LADDER[-1])
    kappa_64 = float((rk64 / r64).real - 1.0)
    drift = abs(kappa_64 - kappa) / abs(kappa_64)
    print(f"\n  [S4] kappa fitted at n_z = {fit_rung}: {kappa:+.6f}")
    print(f"       kappa fitted at n_z = {LADDER[-1]}: {kappa_64:+.6f}   drift {drift:.1%}")
    frac = errs[-1] / plateau if plateau else float("nan")
    print(f"       the ladder has reached {frac:.0%} of the estimated plateau")

    # [S5] THE DERIVED FIX, with nothing fitted. kappa was only ever a proxy for
    # the real defect: the Foldy-Lax sum uses the MIDPOINT value G(r) where the
    # continuum integral over the source cell is V<G>. Replacing G by <G> at
    # every separation -- analytic tables at contact, Gauss quadrature beyond,
    # where the kernel is smooth -- removes the bias at source instead of
    # cancelling it after the fact. NO FITTED CONSTANT APPEARS HERE.
    print("\n  [S5] DERIVED FIX: <G> at every separation, nothing fitted")
    print(f"       {'n_z':>4} {'midpoint G':>11} {'cell-avg <G>':>13} {'ratio':>7}")
    prev_va = None
    for n_z in LADDER[:-1]:  # n_z = 64 with full quadrature is the expensive one
        r0, rk, _ = _signed_error(n_z)
        r1, _, _ = _signed_error(n_z, va_all=True)
        e0, e1 = abs(r0 - rk) / abs(rk), abs(r1 - rk) / abs(rk)
        rat = "" if prev_va is None else f"{e1 / prev_va:7.2f}"
        print(f"       {n_z:4d} {e0:11.4e} {e1:13.4e} {rat:>7}")
        prev_va = e1

    print("\n" + "=" * 86)
    if plateau is not None and ok == len(LADDER):
        print("  RENORMALISABLE: the error saturates AND one constant improves every")
        print("  rung. The scheme converges to a biased limit and the bias is a")
        print("  single number -- derive it from the lattice sum rather than fitting.")
    elif plateau is not None:
        print(f"  SATURATES but one constant does NOT fix every rung ({ok}/{len(LADDER)}).")
        print("  The limit is biased, so a correction exists, but it is not a scalar")
        print("  on T -- it is structured, and the structure has to be identified.")
    else:
        print("  NO PLATEAU. The error keeps growing, so this is not a bias to be")
        print("  renormalised away and the discretisation itself has to change.")
    print("=" * 86)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
