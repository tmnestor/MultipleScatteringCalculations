"""MEASUREMENT: the irreducible core -- ONE plane, which IS a uniform layer.

WHY REDUCE TO ONE PLANE. Four hypotheses have been closed on the stacked ladder
(T-matrix nonlinearity, propagator quadrature in three variants, scalar
self-energy, the 27-component basis) without finding the cause. But the stacked
case is not the minimal one. At n_z = 1 there is no vertical stacking at all: a
single laterally-periodic plane of identical cubes IS a homogeneous layer of
thickness d, exactly. And it ALREADY fails -- 4.30e-3, 3.99e-3, 3.91e-3 as the
layer is thinned, converging to a ~0.39% floor rather than to zero.

Everything the stacked problem has is present here except the stacking. So this
is where to look.

THE EXACT INVARIANT THIS EXPLOITS. For a LATERALLY UNIFORM medium the answer
cannot depend on the size of the periodic supercell. M = 2, 4, 6, 8 all describe
the identical physical layer -- they differ only in how many identical cubes are
called one repeating unit. So:

    ANY dependence on M is a defect in the lateral lattice sum, full stop.

That is a genuine invariant, not a convergence expectation, and it discriminates
sharply:

  * error INDEPENDENT of M  =>  the lateral sum is right, and the 0.39% floor is
    the single cube's T-matrix failing to represent a slab ELEMENT -- the cube
    is not an isolated scatterer here, it is one tile of an infinite sheet, and
    the Weyl sum is supposed to convert one into the other;
  * error DEPENDS on M  =>  the lateral Weyl sum is incomplete or wrong, and
    that is a defect to fix rather than a physics gap to model.

The second would also explain why nothing aimed at the single-site response or
the propagator quadrature has helped: both leave a broken lateral sum untouched.

ALSO SWEPT: the cell size at fixed layer thickness is NOT available here (one
plane fixes d = H), so instead the layer is thinned, which drives ka -> 0. A
representation error that survives ka -> 0 is static, and that narrows it
further.

Run:  conda run -n seismic python scripts/measure_single_plane_core.py
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
CONTRAST = MaterialContrast(2.0e9, 1.0e9, 100.0)
OMEGA = 60.0


def _err(a_half: float, m: int, *, volume_averaged: bool = False) -> float:
    """One plane of cubes vs the exact uniform layer of the same thickness."""
    geom = SlabGeometry(M=m, N_z=1, a=a_half)
    ones = np.ones((1, m, m))
    mat = SlabMaterial(
        Dlambda=CONTRAST.Dlambda * ones,
        Dmu=CONTRAST.Dmu * ones,
        Drho=CONTRAST.Drho * ones,
        ref=REF,
    )
    t0 = compute_slab_tmatrices(geom, mat, OMEGA)
    kh = build_slab_kernels(
        geom, OMEGA, REF, periodic=True, volume_averaged=volume_averaged, va_all=volume_averaged
    )
    res = compute_slab_scattering(
        geom,
        mat,
        OMEGA,
        np.array([1.0, 0.0, 0.0]),
        "P",
        periodic=True,
        gmres_tol=1e-12,
        kernel_hat=kh,
    )
    r_lat = slab_rpp_periodic(res, t0)
    r_ken = kennett_reference_rpp(REF, CONTRAST, geom.d, OMEGA)
    return abs(r_lat - r_ken) / abs(r_ken)


def main() -> int:
    print("=" * 84)
    print("MEASUREMENT -- ONE plane of cubes, which IS a uniform layer")
    print("  a laterally uniform medium CANNOT depend on the periodic supercell size")
    print("=" * 84)

    # ---- the invariant: independence of M ---------------------------------
    print("\n  [P1] supercell invariance -- every M describes the SAME physical layer")
    print(f"       {'a (m)':>8} {'M=2':>11} {'M=4':>11} {'M=6':>11} {'M=8':>11} {'spread':>9}")
    worst = 0.0
    for a in (1.0, 0.25):
        row, vals = [f"       {a:8.4f}"], []
        for m in (2, 4, 6, 8):
            e = _err(a, m)
            vals.append(e)
            row.append(f" {e:11.4e}")
        spread = (max(vals) - min(vals)) / max(np.mean(vals), 1e-300)
        worst = max(worst, spread)
        row.append(f" {spread:9.2e}")
        print("".join(row))

    # ---- does the floor survive ka -> 0? ----------------------------------
    print("\n  [P2] thinning the layer -- does the floor survive ka -> 0?")
    print(f"       {'a (m)':>10} {'ka':>10} {'point G':>11} {'cell-avg':>11}")
    for a in (2.0, 1.0, 0.5, 0.25, 0.125, 0.0625):
        ka = OMEGA / REF.beta * a
        print(f"       {a:10.5f} {ka:10.5f} {_err(a, 4):11.4e} {_err(a, 4, volume_averaged=True):11.4e}")

    # ---- [P3] the law, and where it comes from ---------------------------
    # build_slab_kernels assembles the spatial kernel only over
    # dx, dy in [-(M-1), M-1] -- a finite (2M-1)^2 patch -- and then WRAPS it
    # into M x M for the circular convolution. A true periodic sum needs every
    # lattice image. The neglected tail of the 1/r^3 strain block beyond radius
    # R ~ M d falls as 1/R, so the error should go as A + B/M. Fitted, not
    # assumed, and the intercept A is the floor that survives the truncation.
    print("\n  [P3] the 1/M law: the lateral sum is truncated at M-1 cells")
    ms = (2, 3, 4, 5, 6, 8, 10, 12)
    errs = [_err(0.25, m) for m in ms]
    inv = np.array([1.0 / m for m in ms])
    coef = np.polyfit(inv, np.array(errs), 1)
    print(f"       {'M':>4} {'error':>11} {'A + B/M':>11} {'resid':>10}")
    for m, e in zip(ms, errs, strict=True):
        pred = coef[1] + coef[0] / m
        print(f"       {m:4d} {e:11.4e} {pred:11.4e} {abs(e - pred) / e:10.2e}")
    print(f"\n       fit: error = {coef[1]:.4e} + {coef[0]:.4e}/M")
    print(
        f"       => truncation tail at M = 4 is {coef[0] / 4:.3e}, "
        f"{coef[0] / 4 / errs[2]:.0%} of the measured error there"
    )
    print(f"       => the floor surviving M -> infinity is {coef[1]:.3e}")

    print("\n" + "=" * 84)
    if worst > 1e-6:
        print(f"  THE LATERAL SUM IS BROKEN: the answer depends on M by {worst:.1e},")
        print("  and it cannot. Every M is the same physical layer. This is a defect")
        print("  to fix, not physics to model -- and it would sit underneath every")
        print("  hypothesis tested so far, none of which touched the lateral sum.")
    else:
        print("  THE LATERAL SUM IS SOUND: the answer is independent of the supercell")
        print("  to machine precision, as an invariant demands. So the floor is the")
        print("  single cube's T-matrix failing to represent a TILE OF AN INFINITE")
        print("  SHEET -- the cube here is not an isolated scatterer, and the Weyl")
        print("  sum is what has to convert one into the other.")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
