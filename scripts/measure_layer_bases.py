#!/usr/bin/env python3
"""The layer: which error belongs to the wavefield basis, and which to the medium basis.

A layer whose contrast varies smoothly with depth, discretised into n cells.  Each cell carries the field
to Legendre degree p (the wavefield basis) and the contrast projected to Legendre degree r (the medium
basis).  The scheme is ``crosscheck_graded_contrast.graded_scattered``; the exact solution is its
30-digit reference.  Measured:

  [1] the order of the full solve for every (p, r) with p, r in {0, 1, 2}, against min(2p + 2, 2r + 2);
  [2] the Born term T1 and the second-order term T2 separately (central differences in the contrast,
      steps d and 2d, Richardson-combined), and the relative error of each;
  [3] the projection defect of the profile, D_q = sum_cells int (s - P_q s)^2 / int s^2, for q = 0, 1, 2:
      which degree's defect, if any, the error of T2 follows;
  [4] the long-wave limit, in which T2 error = D_min(p,r) exactly: the departure falls as (k D)^2;
  [5] the defects against exact values from an independent symbolic calculation.

Run:  conda run -n seismic python -u scripts/measure_layer_bases.py [omega] [n ...]
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crosscheck_graded_contrast as cg  # noqa: E402
from crosscheck_second_moment_voxel import CONTRAST, D_LAYER, gauss  # noqa: E402

PROFILE = "smooth"
REF_DEFECT = Path(__file__).resolve().parent.parent / "Mathematica" / "ContinuumLimit_layer_defect.json"
D_STEP = 1e-2


def scattered(omega: float, n: int, p: int, r: int, scale: float) -> np.ndarray:
    cg.CONTRAST = tuple(scale * c for c in CONTRAST)
    try:
        return cg.graded_scattered(omega, n, p, r, cg.PROFILES[PROFILE][0])
    finally:
        cg.CONTRAST = CONTRAST


def exact(omega: float, scale: float) -> np.ndarray:
    return cg.exact_graded(omega, PROFILE, tuple(scale * c for c in CONTRAST))


def born_terms(f) -> tuple[np.ndarray, np.ndarray]:
    """(T1, T2) of f(scale) by central differences at D_STEP and 2 D_STEP, Richardson-combined."""
    d = D_STEP
    v = {s: f(s) for s in (d, -d, 2 * d, -2 * d)}
    t1 = (8 * (v[d] - v[-d]) - (v[2 * d] - v[-2 * d])) / (12 * d)
    t2 = (16 * (v[d] + v[-d]) - (v[2 * d] + v[-2 * d])) / (24 * d * d)
    return t1, t2


def defect(n: int, q: int) -> float:
    """sum over the n cells of int (s - P_q s)^2, over int s^2, for the layer's profile."""
    h = D_LAYER / (2 * n)
    s, w = gauss(-h, h)
    prof = cg.PROFILES[PROFILE][0]
    num = den = 0.0
    for zc in (np.arange(n) + 0.5) * 2 * h:
        vals = prof(zc + s)
        fit = sum(
            (2 * c + 1) / (2 * h) * np.sum(w * vals * cg.leg(c, s / h)) * cg.leg(c, s / h)
            for c in range(q + 1)
        )
        num += float(np.sum(w * (vals - fit) ** 2))
        den += float(np.sum(w * vals**2))
    return num / den


def wolfram_real(text: str) -> float:
    """The real part of a number in Wolfram InputForm: mantissa`precision*^exponent, perhaps + 0.*I."""
    m = re.match(r"\s*(-?[0-9.]+)(?:`[0-9.]*)?(?:\*\^(-?[0-9]+))?", text)
    if m is None:
        raise ValueError(f"wolfram_real: cannot read {text!r}")
    return float(m.group(1)) * 10.0 ** int(m.group(2) or 0)


def rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).max() / np.abs(b).max())


def main() -> int:
    omega = float(sys.argv[1]) if len(sys.argv) > 1 else 300.0
    ns = [int(a) for a in sys.argv[2:]] or [2, 4, 8, 16]
    print(f"layer, profile '{PROFILE}', omega = {omega}, k d at n = 1: {omega / 5000.0 * D_LAYER:.3f}")
    ex_full = exact(omega, 1.0)
    ex1, ex2 = born_terms(lambda sc: exact(omega, sc))
    print(f"|T2| / |T1| at full contrast: {np.abs(ex2).max() / np.abs(ex1).max():.3e}")
    print("defects   n: " + "  ".join(f"{n:>9d}" for n in ns))
    for q in (0, 1, 2):
        print(f"   D_{q}     : " + "  ".join(f"{defect(n, q):9.2e}" for n in ns))
    print("(p, r)  predicted order | full-solve error per n, apparent orders")
    print("        then the errors of T1 and of T2, and T2 error / D_min(p,r)")
    for p in (0, 1, 2):
        for r in (0, 1, 2):
            full, e1, e2 = [], [], []
            for n in ns:
                full.append(rel(scattered(omega, n, p, r, 1.0), ex_full))
                t1, t2 = born_terms(lambda sc, n=n, p=p, r=r: scattered(omega, n, p, r, sc))
                e1.append(rel(t1, ex1))
                e2.append(rel(t2, ex2))
            orders = [np.log(full[i] / full[i + 1]) / np.log(ns[i + 1] / ns[i]) for i in range(len(ns) - 1)]
            q = min(p, r)
            ratio = [e2[i] / defect(n, q) for i, n in enumerate(ns)]
            print(
                f"({p}, {r})  {min(2 * p + 2, 2 * r + 2)} | "
                + " ".join(f"{v:.2e}" for v in full)
                + " | orders "
                + " ".join(f"{v:.2f}" for v in orders)
            )
            print("          T1 " + " ".join(f"{v:.2e}" for v in e1))
            print(
                "          T2 "
                + " ".join(f"{v:.2e}" for v in e2)
                + "   T2/D "
                + " ".join(f"{v:.3f}" for v in ratio)
            )
    # [4] the long-wave limit: the ratio T2 error / D tends to one as (k D)^2
    print("long-wave limit, T2 error / D_min(p,r) - 1 for (p, r, n) = (0,0,8), (1,1,8), (2,2,4):")
    gaps = []
    for om in (600.0, 300.0, 100.0, 30.0):
        e1x, e2x = born_terms(lambda sc, om=om: exact(om, sc))
        row = []
        for p, r, n in ((0, 0, 8), (1, 1, 8), (2, 2, 4)):
            _, t2 = born_terms(lambda sc, om=om, p=p, r=r, n=n: scattered(om, n, p, r, sc))
            row.append(rel(t2, e2x) / defect(n, min(p, r)) - 1.0)
        gaps.append(row)
        print(f"   k D = {om / 5000.0 * D_LAYER:.3f}: " + "  ".join(f"{v:+.2e}" for v in row))
    # between k D = 0.12 and 0.04 the gap should fall ninefold
    falls = [gaps[1][i] / gaps[2][i] for i in range(3)]
    ok = all(6.0 < f < 12.0 for f in falls)
    print(
        f"   fall from k D = 0.12 to 0.04: {' '.join(f'{f:.1f}' for f in falls)} (9 for (k D)^2): "
        f"{'PASS' if ok else 'FAIL'}"
    )
    # [5] the defects against their exact values (Mathematica/ContinuumLimit_LayerDefect.wl)
    table = json.loads(REF_DEFECT.read_text())
    worst = max(
        abs(defect(n, q) / wolfram_real(table["defect"][q][i]) - 1.0)
        for q in (0, 1, 2)
        for i, n in enumerate(table["n"])
        if n <= 16
    )
    ok5 = worst < 1e-8
    verdict = "PASS" if ok5 else "FAIL"
    print(f"defects against their exact values, n = 2 to 16, q = 0, 1, 2: {worst:.1e}: {verdict}")
    return 0 if ok and ok5 else 1


if __name__ == "__main__":
    sys.exit(main())
