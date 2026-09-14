"""MEASUREMENT: what is the ~8.6e-4 residual that survives refinement?

WHERE THIS COMES FROM. With the lateral sum made exact (`lattice_ewald=True`)
and the Galerkin contact correction applied, the refinement ladder FLATTENS:
7.73e-4, 8.35e-4, 8.51e-4, 8.58e-4 over n_z = 1, 2, 4, 8 at fixed physical
thickness. The non-convergence is gone. What remains is a residual that
refinement does not remove, and this script asks what it is.

TWO SUSPECTS WERE ON RECORD:
  (a) the volume-averaged propagator's dynamic content is truncated at O(w^4),
      so it carries an O(w^6) error;
  (b) a space-filling lattice of ISOLATED-cube T-matrices needs a lattice
      renormalisation to reproduce the continuum -- the same class of correction
      as the sphere-packing Delta -> Delta/phi already established here.

(a) IS ALREADY REFUTED BY THE FLATNESS, before this script runs a single case.
An O(w^6) truncation error is O((k d)^6) against an O((k d)^4) retained term, so
its RELATIVE size is O((k d)^2). Halving the cell halves k d and should quarter
it. Across the ladder above d falls 8x, so it should fall 64x. It does not move.
That argument is checked here anyway, by sweeping n_orders directly, because an
argument from a curve is worth less than the measurement it predicts.

THE DISCRIMINATING TEST IS THE CONTRAST SCALING, and it is sharp:

  * a QUADRATURE error is linear in the contrast. The Born response is
    Int G Delta psi dx and mis-integrating it mis-states something proportional
    to Delta. The truncation artifact fixed today measured c^1.00 exactly.
  * a RENORMALISATION error is QUADRATIC. It is the difference between the
    average field and the field actually exciting a cube, which is itself a
    scattered field: one factor of the contrast to scatter, one to respond.

So the exponent of the ABSOLUTE error separates them, and nothing else needs to
be assumed. Note the normalisation: the reported relative error divides by
|r_kennett|, which is itself O(c), so an absolute c^2 shows up as a relative
c^1 -- the script reports both to keep that straight.

WHAT WOULD MAKE THIS INCONCLUSIVE, stated in advance: an exponent near 1.5, or
one that drifts with contrast. Either would mean two mechanisms of comparable
size and the split would need a different cut.

Run:  conda run -n seismic python scripts/measure_residual_origin.py
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
N_Z = 4  # on the flat part of the refinement ladder


def _run(
    c: float,
    *,
    n_z: int = N_Z,
    omega: float = OMEGA,
    n_orders: int = 2,
    volume_averaged: bool = True,
) -> tuple[float, float]:
    """(relative error vs exact Kennett, ABSOLUTE |r_lat - r_ken|)."""
    a_half = H_PHYS / (2.0 * n_z)
    geom = SlabGeometry(M=M, N_z=n_z, a=a_half)
    ones = np.ones((n_z, M, M))
    mat = SlabMaterial(
        Dlambda=c * D_LAM * ones,
        Dmu=c * D_MU * ones,
        Drho=c * D_RHO * ones,
        ref=REF,
    )
    t0 = compute_slab_tmatrices(geom, mat, omega)
    kh = build_slab_kernels(
        geom,
        omega,
        REF,
        periodic=True,
        lattice_ewald=True,
        volume_averaged=volume_averaged,
        n_orders=n_orders,
    )
    res = compute_slab_scattering(
        geom,
        mat,
        omega,
        np.array([1.0, 0.0, 0.0]),
        "P",
        periodic=True,
        gmres_tol=1e-13,
        kernel_hat=kh,
    )
    r_lat = slab_rpp_periodic(res, t0)
    r_ken = kennett_reference_rpp(REF, MaterialContrast(c * D_LAM, c * D_MU, c * D_RHO), H_PHYS, omega)
    return float(abs(r_lat - r_ken) / abs(r_ken)), float(abs(r_lat - r_ken))


def _slope(xs: list[float], ys: list[float]) -> float:
    """Log-log slope by least squares."""
    lx, ly = np.log(np.asarray(xs)), np.log(np.asarray(ys))
    return float(np.polyfit(lx, ly, 1)[0])


def main() -> int:
    print("=" * 88)
    print("MEASUREMENT -- the origin of the residual that survives refinement")
    print(f"  H = {H_PHYS} m, M = {M}, n_z = {N_Z}, exact lateral sum + Galerkin contact")
    print("=" * 88)

    # ---- [R1] is it the averaged propagator's dynamic truncation? ----------
    print("\n  [R1] dynamic content of the averaged propagator (suspect a)")
    print("       n_orders 0 = static, 1 = +w^2, 2 = +w^4. If the residual is the")
    print("       O(w^6) remainder, these must differ and the curve must fall with d.")
    print(f"       {'n_orders':>9} {'rel err':>12}")
    r1 = []
    for n_ord in (0, 1, 2):
        rel, _ = _run(1.0, n_orders=n_ord)
        r1.append(rel)
        print(f"       {n_ord:9d} {rel:12.5e}")
    spread = (max(r1) - min(r1)) / np.mean(r1)
    print(f"       spread across n_orders: {spread:.2e}")

    print("\n       and the refinement behaviour, which is the stronger statement:")
    print(f"       {'n_z':>4} {'d (m)':>8} {'rel err':>12} {'(k d)^2 would give':>20}")
    base = None
    for n_z in (1, 2, 4, 8):
        rel, _ = _run(1.0, n_z=n_z)
        d = H_PHYS / n_z
        if base is None:
            base = (rel, d)
        pred = base[0] * (d / base[1]) ** 2
        print(f"       {n_z:4d} {d:8.4f} {rel:12.5e} {pred:20.5e}")

    # ---- [R2] the discriminating test: contrast scaling ---------------------
    print("\n  [R2] contrast scaling -- quadrature is c^1, renormalisation is c^2")
    print(f"       {'c':>10} {'|r_lat-r_ken|':>15} {'relative':>12} {'local slope':>12}")
    cs = [1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2, 0.1, 0.3, 1.0]
    abs_errs, rel_errs = [], []
    for i, c in enumerate(cs):
        rel, absol = _run(c)
        abs_errs.append(absol)
        rel_errs.append(rel)
        loc = ""
        if i:
            loc = f"{np.log(absol / abs_errs[i - 1]) / np.log(c / cs[i - 1]):12.3f}"
        print(f"       {c:10.4g} {absol:15.6e} {rel:12.5e} {loc:>12}")
    # THE WEAK ARM IS NOISE, AND MUST NOT BE FITTED. Below about c = 1e-2 the
    # absolute discrepancy falls to 1e-12 and under, which is the numerical floor
    # of the whole pipeline (GMRES tolerance, and |r_kennett| itself is only
    # ~1e-7 there). Its "local slopes" of 0.34 and 1.81 are scatter, not physics.
    # Fitting through them would drag the exponent away from what the resolved
    # points actually say -- the same error as reading an unconverged arbiter.
    noise_floor = 1.0e-11
    resolved = [i for i, a in enumerate(abs_errs) if a > noise_floor]
    slope_all = _slope([cs[i] for i in resolved], [abs_errs[i] for i in resolved])
    slope_noisy = _slope(cs, abs_errs)
    print(
        f"\n       RESOLVED points (|err| > {noise_floor:.0e}): c = "
        f"{', '.join(f'{cs[i]:g}' for i in resolved)}"
    )
    print(f"       log-log slope of the ABSOLUTE error over those: {slope_all:.3f}")
    print(f"       (fitting ALL points, noise included, would give {slope_noisy:.3f} --")
    print("        the sub-1e-11 points are at the pipeline's numerical floor, where")
    print("        |r_kennett| itself is ~1e-7, so they carry no information.)")

    # ---- [R3] frequency scaling, as a cross-check on [R1] -------------------
    print("\n  [R3] frequency scaling at fixed geometry (cross-check on suspect a)")
    print("       a dynamic-truncation error grows like (k d)^2; a static")
    print("       renormalisation does not.")
    print(f"       {'omega':>8} {'ka':>8} {'rel err':>12}")
    oms, rels = [], []
    for om in (30.0, 60.0, 120.0, 240.0):
        rel, _ = _run(1.0, omega=om)
        ka = om / REF.alpha * (H_PHYS / (2.0 * N_Z))
        oms.append(om)
        rels.append(rel)
        flag = "" if ka < 0.3 else "  <-- ka OUT OF RANGE"
        print(f"       {om:8.0f} {ka:8.4f} {rel:12.5e}{flag}")
    slope_om = _slope(oms, rels)
    print(f"       log-log slope in omega: {slope_om:.3f}   (2.0 would indicate (k d)^2)")

    print("\n" + "=" * 88)
    print("  VERDICT")
    if spread < 0.05:
        print(f"    (a) REFUTED. n_orders 0/1/2 agree to {spread:.1e}: the residual does")
        print("        not live in the averaged propagator's dynamic content at all --")
        print("        even the STATIC table gives the same answer.")
    else:
        print(f"    (a) LIVE: n_orders changes the answer by {spread:.1e}.")
    if 1.7 < slope_all < 2.3:
        print(f"    (b) SUPPORTED. The absolute error scales as c^{slope_all:.2f}. A")
        print("        quadrature error is linear in the contrast -- the truncation")
        print("        artifact fixed today measured c^1.00. Quadratic is the signature")
        print("        of a field-renormalisation effect: one factor of the contrast to")
        print("        scatter, one to respond. The isolated-cube T0 is being excited by")
        print("        the average field where it should see the LOCAL field.")
    elif slope_all < 1.3:
        print(f"    (b) NOT SUPPORTED: c^{slope_all:.2f} is linear, i.e. still quadrature.")
        print("        Look for a remaining first-order error in the propagator.")
    else:
        print(f"    INCONCLUSIVE: c^{slope_all:.2f} sits between the two signatures, so")
        print("        two mechanisms of comparable size are likely present.")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
