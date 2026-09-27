#!/usr/bin/env python3
"""MEASUREMENT: choose the periodic thesis gate's kernel by the discretisation floor ALONE.

WHY.  ``gate_thesis_formulation_periodic`` was inconclusive: dress-after sat
3.28x above the discretisation floor [T3], short of the 5x bar set before it
ran.  The gate was written on 14 September with the periodic kernel's
TRUNCATED lateral sum; the exact Ewald lattice sum (46f3402, 15 September) and
the source-cell average at every separation (``va_all``, whose reach is a
documented convergence parameter) came later.  Both refine the discretisation,
so both should lower [T3] without touching the ordering effect [T2].

THE RULE, fixed before any result: the kernel is chosen from [T3] ALONE -- the
uniform-background control, which cannot see the layering and so cannot be
tuned toward a pass -- as the first setting in the sequence below whose floor
changes by less than 10% from the previous one.  Only then are [T1] and [T2]
reported, and the gate's 5x bar is not touched.

Run:  conda run -n seismic python scripts/measure_thesis_gate_floor.py
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import gate_thesis_formulation_periodic as gate  # noqa: E402

SEQUENCE: list[tuple[str, dict]] = [
    ("truncated sum (current)", dict(volume_averaged=True, periodic=True)),
    ("Ewald sum", dict(volume_averaged=True, periodic=True, lattice_ewald=True)),
    *[
        (
            f"Ewald + cell average, reach {r}",
            dict(volume_averaged=True, periodic=True, lattice_ewald=True, va_all=True, va_all_reach=r),
        )
        for r in (1, 2, 3, 4, 6)
    ],
]


def main() -> int:
    """Floor per setting, the choice by the 10% rule, then all three arms at the choice.

    Returns:
        0.
    """
    t_start = time.perf_counter()
    print("=" * 78)
    print("THESIS-GATE KERNEL, CHOSEN BY THE DISCRETISATION FLOOR [T3] ALONE")
    print("=" * 78)
    floors = []
    for name, kw in SEQUENCE:
        t3, s3 = gate._run(dressed=True, uniform=True, kernel_kw=kw)
        floors.append(t3 * s3)
        print(
            f"  [{time.perf_counter() - t_start:6.0f} s] {name:32s} floor [T3] = {floors[-1]:.4e}",
            flush=True,
        )
    choice = len(SEQUENCE) - 1
    for i in range(1, len(floors)):
        if abs(floors[i] - floors[i - 1]) < 0.10 * floors[i - 1]:
            choice = i
            break
    name, kw = SEQUENCE[choice]
    print(f"\n  chosen (first setting within 10% of the previous floor): {name}")
    print(f"  kernel_kw = {kw}")
    t1, s1 = gate._run(dressed=True, uniform=False, kernel_kw=kw)
    t2, _ = gate._run(dressed=False, uniform=False, kernel_kw=kw)
    a1, a2, a3 = t1 * s1, t2 * s1, floors[choice]
    print(f"\n  [T1] thesis ordering  {a1:.4e}   [T2] dress-after {a2:.4e}   [T3] floor {a3:.4e}")
    print(
        f"  separation [T2]/[T3] = {a2 / a3:.2f}x (gate needs 5x);  "
        f"thesis [T1]/[T3] = {a1 / a3:.2f}x (needs < 2x)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
