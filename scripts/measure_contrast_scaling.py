"""MEASUREMENT: do the floor and the ordering signal scale DIFFERENTLY in contrast?

`measure_constructive_geometry` found the dressing gain [T2]/[T1] falling from
3.7 at omega = 60 to ~1.1 at omega = 400, and flagged two readings: either the
two-potential formulation is incomplete, or the DRESSED arm is simply
floor-limited and the floor is what rises with frequency. Those must be
separated before the first reading is allowed to stand.

THE DISCRIMINATOR IS THE POWER OF THE CONTRAST, and it is a genuine prediction
rather than a fit. The two errors are built from different numbers of
T-matrices:

    the FLOOR [T3] is a NEIGHBOUR-INTERACTION error -- the touching-face
        propagator between two adjacent cubes, so it carries T twice and should
        scale as contrast^2;
    the ORDERING error [T2] is |DeltaG0 . T| -- the reverberation acting once on
        one scatterer, so it carries T once and should scale as contrast^1.

If that is right, three things follow, and all three are checked below:

  * the separation [T2]/[T3] goes as 1/contrast, so it IMPROVES as the contrast
    WEAKENS -- the exact opposite of the "scatter harder" instinct that has
    driven this test from the start, and if true it means the 5x bar was always
    reachable by turning the contrast DOWN;
  * the gain [T2]/[T1] recovers at weak contrast even at omega = 400, which
    would show the gain decay is a floor effect and NOT a defect in the
    formulation;
  * the fitted powers come out near 2 and 1. They are FITTED AND PRINTED, not
    assumed. If the floor comes out linear too, the reasoning above is wrong and
    the separation cannot be bought this way.

Weak contrast eventually runs into machine noise, not physics, so the absolute
errors are printed and the run stops being meaningful once they approach the
GMRES tolerance of 1e-12 times the field scale. Watch the absolute columns.

Run:  conda run -n seismic python scripts/measure_contrast_scaling.py
SI units (m, m/s, kg/m3, Pa) -- the slab machinery's own convention.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import gate_thesis_formulation_periodic as G  # noqa: E402

# 1.0 is the validated moderate contrast (dlambda 2 GPa, dmu 1 GPa, drho 100).
CONTRASTS = (1.0, 0.5, 0.25, 0.125)
# omega 60 is the gate's own configuration; 400 is where the gain had decayed to
# ~1.1 and the two readings have to be told apart.
OMEGAS = (60.0, 400.0)


def _fit_power(xs: list, ys: list) -> float:
    """Slope of log(y) against log(x) -- the exponent, by least squares."""
    lx, ly = np.log(np.array(xs)), np.log(np.array(ys))
    return float(np.polyfit(lx, ly, 1)[0])


def main() -> int:
    print("=" * 92)
    print("MEASUREMENT -- how the floor and the ordering signal scale with CONTRAST")
    print("  prediction: floor ~ contrast^2 (two T's), ordering ~ contrast^1 (one T)")
    print("  if so, the separation improves as the contrast WEAKENS")
    print("=" * 92)

    verdicts = []
    for omega in OMEGAS:
        ka = omega / G.A0 * G.A_HALF
        print(f"\n  omega = {omega:6.1f}   ka = {ka:.4f}")
        print(
            f"    {'contrast':>9} {'[T1]abs':>11} {'[T2]abs':>11} {'[T3]abs':>11} "
            f"{'T2/T3':>7} {'T1/T3':>7} {'GAIN':>6}"
        )
        cs, f3, f2 = [], [], []
        for c in CONTRASTS:
            t1, s1 = G._run(dressed=True, uniform=False, omega=omega, contrast=c)
            t2, _ = G._run(dressed=False, uniform=False, omega=omega, contrast=c)
            t3, s3 = G._run(dressed=True, uniform=True, omega=omega, contrast=c)
            a1, a2, a3 = t1 * s1, t2 * s1, t3 * s3
            cs.append(c)
            f3.append(a3)
            f2.append(a2)
            print(
                f"    {c:9.4f} {a1:11.4e} {a2:11.4e} {a3:11.4e} "
                f"{a2 / a3:7.2f} {a1 / a3:7.2f} {a2 / a1:6.2f}"
            )
        p3, p2 = _fit_power(cs, f3), _fit_power(cs, f2)
        print(f"    fitted powers:  floor [T3] ~ c^{p3:.2f}   ordering [T2] ~ c^{p2:.2f}")
        verdicts.append((omega, p3, p2, f2[-1] / f3[-1], cs[-1]))

    print("\n" + "=" * 92)
    for omega, p3, p2, sep_weak, c_weak in verdicts:
        print(
            f"  omega {omega:6.1f}:  floor c^{p3:.2f}, ordering c^{p2:.2f}, "
            f"separation at c={c_weak} is {sep_weak:.2f}x"
        )
    if all(v[1] - v[2] > 0.4 for v in verdicts):
        print("\n  THE POWERS DIFFER as predicted. The floor falls faster than the")
        print("  signal, so weakening the contrast BUYS SEPARATION -- and the gate")
        print("  has been pushed in the wrong direction all along. The 5x bar is")
        print("  reachable without touching the discretisation at all.")
    else:
        print("\n  THE POWERS DO NOT SEPARATE. The floor and the ordering effect")
        print("  scale alike, so no contrast buys separation and the reasoning")
        print("  above is refuted. The floor itself has to go.")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
