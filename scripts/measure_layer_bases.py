#!/usr/bin/env python3
"""The layer: which error belongs to the wavefield basis, and which to the medium basis.

A layer whose contrast varies smoothly with depth, discretised into n cells.  Each cell carries the field
to Legendre degree p (the wavefield basis) and the contrast projected to Legendre degree r (the medium
basis).  The scheme is ``crosscheck_graded_contrast.graded_scattered``; the exact solution is its
30-digit reference.  Measured:

  [1] the order of the full solve for every (p, r) with p, r in {0, 1, 2}, against min(2p + 2, 2r + 2);
  [2] the Born term T1 and the second-order term T2 separately (central differences in the contrast,
      steps d and 2d, Richardson-combined), and the relative error of each;
  [3] the relative projection error of the profile, E_q = sum_cells int (s - P_q s)^2 / int s^2, for
      q = 0, 1, 2: which degree's projection error, if any, the error of T2 follows;
  [4] the long-wave limit, in which T2 error = E_min(p,r) exactly: the departure falls as (k D)^2;
  [5] the projection errors against exact values from an independent symbolic calculation;
  [6] whose error it is: T2 split into the part in which the second scattering is in the cell of the
      first (the second-order term of the single-site T-matrix) and the part between different cells,
      for the scheme and for the exact layer on the same cells, and the error of each part;
  [7] the first two terms of the exact layer in closed form (the uniform layer: the formulas of the
      paper; the smooth profile: symbolic integrals) against the difference formulas used above.

Run:  conda run -n seismic python -u scripts/measure_layer_bases.py [omega] [n ...]
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crosscheck_graded_contrast as cg  # noqa: E402
from crosscheck_second_moment_voxel import (  # noqa: E402
    ALPHA,
    CONTRAST,
    D_LAYER,
    M_P,
    Z_OBS_R,
    Z_OBS_T,
    Z_SRC,
    g_inc,
    gauss,
    kernel,
)

PROFILE = "smooth"
REF_BORN = Path(__file__).resolve().parent.parent / "Mathematica" / "ContinuumLimit_born_terms.json"
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


def projection_error(n: int, q: int) -> float:
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


def scheme_split(omega: float, n: int, p: int, r: int) -> tuple[np.ndarray, np.ndarray]:
    """The scheme's T2 at the two observers in two parts: (same cell, different cells).

    With (mass - k_self - k_off) x = rhs, the term of second order in the contrast is
    readout mass^-1 (k_self + k_off) mass^-1 rhs: the field scattered once, scattered again in the same
    cell (k_self, the second-order term of the single-site T-matrix) or in another (k_off).
    """
    mass, k_self, k_off, rhs, readout = cg.graded_system(omega, n, p, r, cg.PROFILES[PROFILE][0])
    x0 = np.linalg.solve(mass, rhs)
    return readout @ np.linalg.solve(mass, k_self @ x0), readout @ np.linalg.solve(mass, k_off @ x0)


def exact_split(omega: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """The exact T2 of the layer, split by the same n cells: (same cell, different cells).

    T2 = int K(z_obs - z) dq f(z) w1(z) dz, with w1(z) = int K(z - z') dq f(z') w0(z') dz' the state
    scattered once (the local term of the strain-strain entry included); w1 is split by whether z' lies
    in the cell of z.
    """
    d_lam, d_mu, d_rho = CONTRAST
    k = omega / ALPHA
    h = D_LAYER / (2 * n)
    dq = np.diag([omega**2 * d_rho, d_lam + 2 * d_mu])
    prof = cg.PROFILES[PROFILE][0]
    centres = (np.arange(n) + 0.5) * 2 * h
    s, ws = gauss(-h, h)

    def source(z: np.ndarray) -> np.ndarray:
        """dq f(z) w0(z), shape z.shape + (2,)."""
        g = np.array([g_inc(k, v) for v in z])
        return prof(z)[:, None] * (np.stack([g, 1j * k * g], -1) @ dq.T)

    parts = [np.zeros(2, dtype=complex), np.zeros(2, dtype=complex)]
    for zc in centres:
        z = zc + s
        w_same = np.zeros((len(s), 2), dtype=complex)
        w_other = np.zeros((len(s), 2), dtype=complex)
        for m, zm in enumerate(z):
            for lo, hi in ((zc - h, zm), (zm, zc + h)):
                t, wt = gauss(lo, hi)
                w_same[m] += np.einsum("n,nij,nj->i", wt, kernel(k, zm - t), source(t))
            w_same[m, 1] -= source(np.array([zm]))[0, 1] / M_P
            for zo in centres:
                if zo != zc:
                    t = zo + s
                    w_other[m] += np.einsum("n,nij,nj->i", ws, kernel(k, zm - t), source(t))
        for o, z_obs in enumerate((Z_OBS_R, Z_OBS_T)):
            row = np.einsum("nj,jk->nk", kernel(k, z_obs - z)[:, 0, :], dq) * (ws * prof(z))[:, None]
            parts[0][o] += np.sum(row * w_same)
            parts[1][o] += np.sum(row * w_other)
    return parts[0], parts[1]


def born_closed(omega: float) -> tuple[np.ndarray, np.ndarray]:
    """(T1, T2) of the UNIFORM layer at the two observers, in closed form.

    With a = w^2 drho, b = dM, c = i / (2 M k):
        T1(R) = c^2 e^{-ik(zo+zs)} (a + b k^2) (e^{2ikD} - 1) / (2ik),
        T1(T) = c^2 e^{ik(zo-zs)} (a - b k^2) D,
    and T2 as in the paper (Mathematica/ContinuumLimit_BornTerms.wl derives both).
    """
    k = omega / ALPHA
    a, b = omega**2 * CONTRAST[2], CONTRAST[0] + 2 * CONTRAST[1]
    c, d = 1j / (2 * M_P * k), D_LAYER
    e = np.exp(2j * k * d)
    pr = c * c * np.exp(-1j * k * (Z_OBS_R + Z_SRC))
    pt = c * c * np.exp(1j * k * (Z_OBS_T - Z_SRC))
    t1 = np.array([pr * (a + b * k * k) * (e - 1) / (2j * k), pt * (a - b * k * k) * d])
    t2r = -1j * (a * a + b * b * k**4) + e * (a * a * (1j + 2 * k * d) + b * b * k**4 * (1j - 2 * k * d))
    t2t = (
        a * a * (1 - e + 2 * k * d * (1j + k * d))
        - 2 * a * b * k * k * (e - 1 + 2 * k * d * (k * d - 1j))
        + b * b * k**4 * (1 - e + 2 * k * d * (k * d - 3j))
    )
    return t1, np.array([pr / (4 * k**3 * M_P) * t2r, 1j * pt / (8 * k**3 * M_P) * t2t])


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
    print("projection errors   n: " + "  ".join(f"{n:>9d}" for n in ns))
    for q in (0, 1, 2):
        print(f"   E_{q}     : " + "  ".join(f"{projection_error(n, q):9.2e}" for n in ns))
    print("(p, r)  predicted order | full-solve error per n, apparent orders")
    print("        then the errors of T1 and of T2, and T2 error / E_min(p,r)")
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
            ratio = [e2[i] / projection_error(n, q) for i, n in enumerate(ns)]
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
                + "   T2/E "
                + " ".join(f"{v:.3f}" for v in ratio)
            )
    # [4] the long-wave limit: the ratio T2 error / E tends to one as (k D)^2
    print("long-wave limit, T2 error / E_min(p,r) - 1 for (p, r, n) = (0,0,8), (1,1,8), (2,2,4):")
    gaps = []
    for om in (600.0, 300.0, 100.0, 30.0):
        e1x, e2x = born_terms(lambda sc, om=om: exact(om, sc))
        row = []
        for p, r, n in ((0, 0, 8), (1, 1, 8), (2, 2, 4)):
            _, t2 = born_terms(lambda sc, om=om, p=p, r=r, n=n: scattered(om, n, p, r, sc))
            row.append(rel(t2, e2x) / projection_error(n, min(p, r)) - 1.0)
        gaps.append(row)
        print(f"   k D = {om / 5000.0 * D_LAYER:.3f}: " + "  ".join(f"{v:+.2e}" for v in row))
    # between k D = 0.12 and 0.04 the gap should fall ninefold
    falls = [gaps[1][i] / gaps[2][i] for i in range(3)]
    ok = all(6.0 < f < 12.0 for f in falls)
    print(
        f"   fall from k D = 0.12 to 0.04: {' '.join(f'{f:.1f}' for f in falls)} (9 for (k D)^2): "
        f"{'PASS' if ok else 'FAIL'}"
    )
    # [5] the projection errors against their exact values (Mathematica/ContinuumLimit_LayerDefect.wl)
    table = json.loads(REF_DEFECT.read_text())
    worst = max(
        abs(projection_error(n, q) / wolfram_real(table["defect"][q][i]) - 1.0)
        for q in (0, 1, 2)
        for i, n in enumerate(table["n"])
        if n <= 16
    )
    ok5 = worst < 1e-8
    verdict = "PASS" if ok5 else "FAIL"
    print(f"projection errors against their exact values, n = 2 to 16, q = 0, 1, 2: {worst:.1e}: {verdict}")
    # [6] whose error: the same-cell part of T2 (the single site) and the part between cells
    print("the error of T2 in two parts, each over |T2| E_min(p,r): same cell | between cells")
    worst_sum, same_all, other_all = 0.0, [], []
    for n in ns[1:]:
        ex_same, ex_other = exact_split(omega, n)
        worst_sum = max(worst_sum, rel(ex_same + ex_other, ex2))
        for p in (0, 1, 2):
            for r in (0, 1, 2):
                sc_same, sc_other = scheme_split(omega, n, p, r)
                scale = np.abs(ex2).max() * projection_error(n, min(p, r))
                e_same = float(np.abs(sc_same - ex_same).max() / scale)
                e_other = float(np.abs(sc_other - ex_other).max() / scale)
                same_all.append(e_same)
                other_all.append(e_other)
                print(f"   n {n:2d} ({p}, {r}): {e_same:.4f} | {e_other:.2e}")
    print(f"   exact parts sum to the exact T2 to {worst_sum:.1e}")
    ok6 = worst_sum < 1e-6 and min(same_all) > 0.95 and max(same_all) < 1.05 and max(other_all) < 0.05
    print(
        f"   same cell {min(same_all):.3f} to {max(same_all):.3f}, between cells at most "
        f"{max(other_all):.1e}: {'PASS' if ok6 else 'FAIL'}"
    )
    # [7] the closed forms of T1 and T2 against the symbolic values and against the difference formulas
    born = json.loads(REF_BORN.read_text())

    def table(profile: str, name: str) -> np.ndarray:
        return np.array([complex(wolfram_real(x[0]), wolfram_real(x[1])) for x in born[profile][name]])

    ok7 = True
    if omega == born["omega"]:
        c1, c2 = born_closed(omega)
        closed = max(rel(c1, table("const", "T1")), rel(c2, table("const", "T2")))
        smooth = max(rel(ex1, table("smooth", "T1")), rel(ex2, table("smooth", "T2")))
        ok7 = closed < 1e-13 and smooth < 1e-10
        print(
            f"closed forms of T1, T2: uniform layer against the symbolic values {closed:.1e}; smooth, "
            f"differences against the symbolic integrals {smooth:.1e}: {'PASS' if ok7 else 'FAIL'}"
        )
    return 0 if ok and ok5 and ok6 and ok7 else 1


if __name__ == "__main__":
    sys.exit(main())
