#!/usr/bin/env python3
"""An octree of constant cells grown by the first-order error itself, which needs no solve.

WHY. ``measure_octree_product_indicator.py`` found that, for constant cells on a body not small against
the wavelength, the far-field error of every tree is its first-order error to about 5%. The first-order
error of a tree is known before anything is solved: each leaf holds the projected incident wave, and its
Born far field is one pass over the leaf. So the trees can be compared, and a tree can be grown, with no
solve at all. This script grows one and then solves it, to see whether the error it was grown to reduce
is the error it has.

THE RULE. Start from the grid of 2 cells across. The first-order error of the tree is the vector
    R = sum over leaves of (the leaf's Born far field) - (the exact Born far field)
over a set of directions. Refining a leaf replaces its Born far field by the sum of its eight children's,
which changes R by a known vector. At each step the one refinement that most reduces |R| is made, until
the tree has the number of leaves asked for.

THE GUARD AGAINST FITTING. A tree has hundreds of refinements to choose among, and with them it could
cancel R at a handful of directions without being a better grid. So the tree is grown on one set of
directions (40 spread over the whole sphere, out of the plane of the test as well as in it) and judged on
another (the nine of the other experiments, which the growth never sees). The exact Born far field at
the growth directions is a central difference of the exact solution in the contrast, as elsewhere.

WHAT IS MEASURED, at each number of leaves asked for: the tree's leaves by size; its first-order error on
the nine test directions, with no solve; then, solved, its far-field error against the exact solution on
the same nine directions. Beside them, from ``measure_octree_product_indicator_*.json`` where stored, the
trees refined by the medium and by the product at the nearest numbers of leaves.

Results are written to ``scripts/measure_octree_first_order_greedy_ka<ka>_core<c>.json``.

Run small first:
    conda run -n seismic python -u scripts/measure_octree_first_order_greedy.py 4 0.75 --leaves=300
    ... <k_S a> <core_frac> [--leaves=900,1700] [--hmin=0.625] [--dry]
--dry grows the trees and prints their no-solve first-order errors without solving.
"""

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
from gate_sphere_cell_average_vs_mie import CONTRAST, REF, obs_points  # noqa: E402
from measure_graded_voxel_resolution import exact_field, radial  # noqa: E402
from pilot_graded_sphere_vs_exact import RADIUS, THETA  # noqa: E402

from cubic_scattering.graded_voxel.octree import (  # noqa: E402
    born_octree,
    octree_far_field,
    solve_graded_octree,
    uniform_leaves,
)

K_HAT = np.array([1.0, 0.0, 0.0])
EPS = 1e-3
DENSE_LIMIT = 20000
SIGNS = np.array(list(itertools.product((-1.0, 1.0), repeat=3)))
N_GROWTH = 40


def sphere_directions(n: int) -> np.ndarray:
    """n directions spread over the sphere (a Fibonacci spiral)."""
    i = np.arange(n) + 0.5
    polar = np.arccos(1.0 - 2.0 * i / n)
    azimuth = np.pi * (1.0 + 5.0**0.5) * i
    return np.stack([np.cos(polar), np.sin(polar) * np.cos(azimuth), np.sin(polar) * np.sin(azimuth)], 1)


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    dry = "--dry" in sys.argv
    ka, core_frac = float(args[0]), float(args[1])
    targets = sorted(int(v) for v in opts.get("leaves", "900,1700").split(",") if v)
    h_min = float(opts.get("hmin", RADIUS / 16))
    core = RADIUS * core_frac
    omega = ka * REF.beta / RADIUS
    rf = 5e8 * RADIUS
    test_pts = obs_points(rf, THETA)
    grow_dirs = sphere_directions(N_GROWTH)
    all_pts = np.concatenate([rf * grow_dirs, test_pts])
    all_dirs = all_pts / rf
    n_grow = len(grow_dirs)

    def prof(pos: np.ndarray) -> float:
        return radial("s5", core, float(np.linalg.norm(pos)))

    def exact_born_at(points: np.ndarray) -> np.ndarray:
        return (
            exact_field("s5", core, omega, EPS, points) - exact_field("s5", core, omega, -EPS, points)
        ) / (2 * EPS)

    born_exact = exact_born_at(all_pts)  # (directions, 3)
    exact = exact_field("s5", core, omega, 1.0, test_pts)
    peak = float(np.abs(exact).max())

    cache: dict = {}

    def born_of(c: np.ndarray, h: float) -> np.ndarray:
        """One cell's Born far field at every direction, (directions, 3); zero if it carries no contrast."""
        key = (round(float(c[0]), 9), round(float(c[1]), 9), round(float(c[2]), 9), round(h, 9))
        if key not in cache:
            if np.linalg.norm(c) - np.sqrt(3.0) * h >= RADIUS:
                cache[key] = np.zeros((len(all_dirs), 3), dtype=complex)
            else:
                res = born_octree(
                    omega, REF, CONTRAST, c[None, :], np.array([h]), prof, K_HAT, K_HAT, "P", p=0, r=0
                )
                cache[key] = sum(octree_far_field(res, all_dirs, rf))
        return cache[key]

    def carries(c: np.ndarray, h: float) -> bool:
        return bool(np.abs(born_of(c, h)).max() > 0.0)

    centres0, hs0 = uniform_leaves(RADIUS, 2)
    leaves = [(c, float(h)) for c, h in zip(centres0, hs0, strict=True) if carries(c, float(h))]
    total = sum(born_of(c, h) for c, h in leaves)

    def test_error(vec: np.ndarray) -> float:
        return float(np.abs(vec[n_grow:] - born_exact[n_grow:]).max() / peak)

    stored = ROOT / f"scripts/measure_octree_product_indicator_ka{ka:g}_core{core_frac:g}_p0.json"
    others = [t for t in json.loads(stored.read_text())["trees"] if "error" in t] if stored.exists() else []
    out_path = ROOT / f"scripts/measure_octree_first_order_greedy_ka{ka:g}_core{core_frac:g}.json"
    result: dict = {
        "ka_s": ka,
        "core_frac": core_frac,
        "h_min": h_min,
        "growth_directions": N_GROWTH,
        "trees": [],
    }
    print(
        f"an octree of constant cells grown by its first-order error: k_S a = {ka}, core = {core_frac} a, "
        f"{N_GROWTH} growth directions, judged on the nine test directions",
        flush=True,
    )
    block_cache: dict = {}
    t_start = time.perf_counter()
    for target in targets:
        while len(leaves) < target:
            residual = (total - born_exact)[:n_grow]
            best, best_norm, best_kids = None, np.inf, None
            for i, (c, h) in enumerate(leaves):
                if h <= h_min * (1.0 + 1e-9):
                    continue
                kids = [(c + 0.5 * h * s, 0.5 * h) for s in SIGNS]
                delta = sum(born_of(kc, kh) for kc, kh in kids) - born_of(c, h)
                norm = float(np.linalg.norm(residual + delta[:n_grow]))
                if norm < best_norm:
                    best, best_norm, best_kids = i, norm, kids
            if best is None:
                break
            c, h = leaves.pop(best)
            total = total - born_of(c, h)
            for kc, kh in best_kids:
                if carries(kc, kh):
                    leaves.append((kc, kh))
                    total = total + born_of(kc, kh)
        centres = np.array([c for c, _ in leaves])
        hs = np.array([h for _, h in leaves])
        sizes = {f"{float(h):g}": int((hs == h).sum()) for h in np.unique(hs)}
        row = {
            "target": target,
            "cells": len(leaves),
            "sizes": sizes,
            "dT1_test": test_error(total),
            "dT1_growth": float(np.abs((total - born_exact)[:n_grow]).max() / peak),
            "grow_seconds": time.perf_counter() - t_start,
        }
        if not dry and len(leaves) * 9 <= DENSE_LIMIT:
            t0 = time.perf_counter()
            res = solve_graded_octree(
                omega,
                REF,
                CONTRAST,
                centres,
                hs,
                prof,
                K_HAT,
                K_HAT,
                "P",
                p=0,
                r=0,
                block_cache=block_cache,
            )
            err = sum(octree_far_field(res, test_pts / rf, rf)) - exact
            d_t1 = total[n_grow:] - born_exact[n_grow:]
            row.update(
                error=float(np.abs(err).max() / peak),
                rest=float(np.abs(err - d_t1).max() / peak),
                solve_seconds=time.perf_counter() - t0,
            )
        result["trees"].append(row)
        print(
            f"  grown to {row['cells']:5d} leaves {sizes}: first-order error on the test directions "
            f"{row['dT1_test']:.3e} (on the growth directions {row['dT1_growth']:.3e})"
            + (f"; solved: error {row['error']:.3e}, rest {row['rest']:.3e}" if "error" in row else ""),
            flush=True,
        )
        near = sorted(others, key=lambda t: abs(t["cells"] - row["cells"]))[:3]
        for t in sorted(near, key=lambda t: (t["rule"], t["cells"])):
            print(
                f"      beside it, refined by the {t['rule']:7s}: {t['cells']:5d} leaves, error {t['error']:.3e}, "
                f"first-order {t['dT1']:.3e}"
            )
        out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"  wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
