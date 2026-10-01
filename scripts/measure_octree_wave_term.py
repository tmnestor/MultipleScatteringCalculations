#!/usr/bin/env python3
"""The wave term of a leaf's error: the Born term of a polynomial cell against its closed form.

A cell of half-width h whose field is held to Legendre degree p represents the incident plane wave by its
projection, and radiates through the projection of the outgoing wave.  At first order in the contrast,
with the contrast uniform in the cell, the cell's far field is therefore its exact Born far field times

    1 + E_p = sum_{|a| <= p} prod_i (2 a_i + 1) j_{a_i}(k_in_i h) j_{a_i}(k_out_i h)
              / prod_i j_0((k_in - k_out)_i h),

(``octree.born_wave_factor``), with k_out the P or the S wavevector towards the observer.  For p = 0 the
leading term is E_0 = -(k_in . k_out) d^2 / 12, d = 2h: it depends on the scattering angle.

THE TEST.  A homogeneous cube of half-side A is represented exactly by every grid of cubes, so its Born
far field divided by (1 + E_p) must be the same for every degree and every grid:
  [1] uniform grids n = 1, 2 with p = 0, 1, 2: the Born term (central difference in the contrast) over
      (1 + E_p), against that of the reference (p = 2, n = 2), P and S far field, nine angles;
  [2] the same without the factor, to show the size of what the factor removes;
  [3] a tree of mixed leaf sizes and mixed degrees: the Born term against the sum over leaves of each
      leaf's own exact Born far field times its factor.

Run:  conda run -n seismic python -u scripts/measure_octree_wave_term.py [k_S A]
"""

import itertools
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cubic_scattering.effective_contrasts import MaterialContrast, ReferenceMedium  # noqa: E402
from cubic_scattering.graded_voxel.octree import (  # noqa: E402
    born_wave_factor,
    octree_far_field,
    refine_leaves,
    solve_graded_octree,
)

REF = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
CONTRAST = (2.0e9, 1.0e9, 100.0)
HALF_SIDE = 10.0
K_HAT = np.array([1.0, 0.0, 0.0])
THETA = np.linspace(0.2, np.pi - 0.2, 9)
EPS = 1e-3
R_FAR = 5e8 * HALF_SIDE


def cube_leaves(n: int) -> tuple[np.ndarray, np.ndarray]:
    """The n^3 leaves of the cube [-A, A]^3."""
    h = HALF_SIDE / n
    ticks = (np.arange(n) + 0.5) * 2 * h - HALF_SIDE
    return np.array(list(itertools.product(ticks, repeat=3))), np.full(n**3, h)


def directions() -> np.ndarray:
    out = np.zeros((len(THETA), 3))
    out[:, 0], out[:, 1] = np.cos(THETA), np.sin(THETA)
    return out


def born(omega: float, centres: np.ndarray, hs: np.ndarray, p) -> tuple[np.ndarray, np.ndarray]:
    """The term of first order in the contrast of the far field (u_P, u_S), by a central difference."""
    out = []
    for sign in (1.0, -1.0):
        con = MaterialContrast(*(sign * EPS * c for c in CONTRAST))
        res = solve_graded_octree(omega, REF, con, centres, hs, lambda _x: 1.0, K_HAT, K_HAT, "P", p=p, r=0)
        out.append(octree_far_field(res, directions(), R_FAR))
    return (out[0][0] - out[1][0]) / (2 * EPS), (out[0][1] - out[1][1]) / (2 * EPS)


def factors(omega: float, h: float, p: int) -> tuple[np.ndarray, np.ndarray]:
    """1 + E_p towards each observer, for the P and for the S far field (incident P along K_HAT)."""
    k_in = omega / REF.alpha * K_HAT
    f_p = np.array([born_wave_factor(k_in, omega / REF.alpha * d, h, p) for d in directions()])
    f_s = np.array([born_wave_factor(k_in, omega / REF.beta * d, h, p) for d in directions()])
    return f_p[:, None], f_s[:, None]  # one factor per observer, for the three components


def rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).max() / np.abs(b).max())


def main() -> int:
    ka = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    omega = ka * REF.beta / HALF_SIDE
    t0 = time.perf_counter()
    print(f"homogeneous cube, half-side A: k_S A = {ka}, k_P A = {ka * REF.beta / REF.alpha:.2f}")
    runs = {}
    for n, p in ((2, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)):
        c, hs = cube_leaves(n)
        runs[(n, p)] = (born(omega, c, hs, p), factors(omega, HALF_SIDE / n, p))
        print(f"  solved n = {n}, p = {p}  [{time.perf_counter() - t0:5.0f} s]", flush=True)
    (ref_p, ref_s), (fr_p, fr_s) = runs[(2, 2)]
    exact_p, exact_s = ref_p / fr_p, ref_s / fr_s
    print("[1], [2]  Born far field against the reference: with the factor | without   (P ; S)")
    worst = 0.0
    for (n, p), ((up, us), (fp, fs)) in sorted(runs.items()):
        if (n, p) == (2, 2):
            continue
        with_p, with_s = rel(up / fp, exact_p), rel(us / fs, exact_s)
        worst = max(worst, with_p, with_s)
        print(
            f"   n = {n}, p = {p} (k_S h = {ka / n:.2f}): {with_p:.1e} ; {with_s:.1e} | "
            f"{rel(up, exact_p):.1e} ; {rel(us, exact_s):.1e}"
        )
    ok1 = worst < 1e-5
    print(f"   worst with the factor {worst:.1e}: {'PASS' if ok1 else 'FAIL'}")

    # [3] a tree: one of the eight n = 2 leaves refined, degrees 0 in the small leaves and 1 in the large
    c2, h2 = cube_leaves(2)
    flag = np.zeros(8, dtype=bool)
    flag[0] = True
    ct, ht = refine_leaves(c2, h2, flag)
    p_leaf = np.where(ht < h2[0], 0, 1)
    up, us = born(omega, ct, ht, p_leaf)
    pred_p, pred_s = np.zeros_like(up), np.zeros_like(us)
    k_in = omega / REF.alpha * K_HAT
    dirs = directions()
    for c, h, pl in zip(ct, ht, p_leaf, strict=True):
        # the leaf's exact Born far field: the whole cube's, rescaled in volume, form factor and phase
        for out, pred, speed in ((exact_p, pred_p, REF.alpha), (exact_s, pred_s, REF.beta)):
            for m, d in enumerate(dirs):
                q = k_in - omega / speed * d
                share = (h / HALF_SIDE) ** 3 * np.prod(
                    np.sinc(q * h / np.pi) / np.sinc(q * HALF_SIDE / np.pi)
                )
                share *= np.exp(1j * q @ c)
                pred[m] += out[m] * share * born_wave_factor(k_in, omega / speed * d, float(h), int(pl))
    tree_p, tree_s = rel(up, pred_p), rel(us, pred_s)
    ok3 = max(tree_p, tree_s) < 1e-5
    print(
        f"[3] tree of {len(ht)} leaves, mixed sizes and degrees: Born against the sum of leaf predictions "
        f"{tree_p:.1e} ; {tree_s:.1e}  (without factors {rel(up, exact_p):.1e} ; {rel(us, exact_s):.1e}): "
        f"{'PASS' if ok3 else 'FAIL'}"
    )
    return 0 if ok1 and ok3 else 1


if __name__ == "__main__":
    sys.exit(main())
