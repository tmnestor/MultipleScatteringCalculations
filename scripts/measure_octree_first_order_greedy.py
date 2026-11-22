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

A SECOND RULE, BECAUSE THE FIRST CAN STALL (``--rule=own``). The rule above (``--rule=residual``, the
default) takes the single step that most reduces |R|. On the uniform sphere with a sharp surface at
k_S a = 4 it stalls: 24 large leaves in the interior, each two radians of shear phase across, are never
divided, and the error stays at 0.05 however many leaves are added at the surface. The errors of those
leaves largely cancel one another in R, so dividing any ONE of them makes |R| larger, although dividing
all of them makes it much smaller. A rule of single best steps cannot see that. The second rule divides
the leaf whose own first-order error is largest, |D| with
    D = (the sum of its eight children's Born far fields) - (its own),
whatever that does to |R| at that step. It cannot stall in this way, since a leaf with a large error of
its own is always divided; what it gives up is the use of cancellation between leaves.

A THIRD RULE, WHICH JOINS THE TWO (``--rule=hybrid``). Run on three bodies, neither rule is the better
everywhere: the first uses cancellation between leaves and is the better where it does not stall; the
second cannot stall but its error does not fall steadily, since it divides leaves whose errors were
cancelling. The sign that the first rule is about to stall is a leaf whose own error exceeds the error of
the whole tree: it can only be so large if other leaves are cancelling it. So the third rule takes the
single best step, as the first does, unless some leaf has |D| > factor x |R| (``--factor``, default 1), in
which case it divides the leaf with the largest |D|, as the second does. ``--nouniform`` skips the uniform
grids, which are the slow part of a run that only compares rules.

THE GUARD AGAINST FITTING. A tree has hundreds of refinements to choose among, and with them it could
cancel R at a handful of directions without being a better grid. So the tree is grown on one set of
directions (40 spread over the whole sphere, out of the plane of the test as well as in it) and judged on
another (the nine of the other experiments, which the growth never sees). The exact Born far field at
the growth directions is a central difference of the exact solution in the contrast, as elsewhere.

WHAT IS MEASURED, at each number of leaves asked for: the tree's leaves by size; its first-order error on
the nine test directions, with no solve; then, solved, its far-field error against the exact solution on
the same nine directions. Beside them, from ``measure_octree_product_indicator_*.json`` where stored, the
trees refined by the medium and by the product at the nearest numbers of leaves.

THE SECOND TERM, ALSO WITH NO SOLVE. Where the body is small against the wavelength the first-order error
is not the whole error (at k_S a = 0.5 it predicts a quarter low). The next term of the error is of
second order in the contrast,
    dT2 = (the scheme's second Born term on the tree) - (the exact second Born term),
and the scheme's second Born term is one product of the assembled matrix with the first-order state
(``born_series_octree``): the matrix must be assembled, but nothing is solved. The exact one is the second
difference of the exact solution in the contrast, at steps d and 2d combined to remove the fourth-order
term. So three numbers are stored for each tree: the first-order prediction |dT1|, the two-term
prediction |dT1 + dT2| (a sum of complex vectors over the directions, not of magnitudes), and the solved
error. What the two-term prediction still leaves out is of third order in the contrast.

A SECOND BODY, WHERE A TREE HAS SOMETHING TO SAVE (``--shape=halo``). On the sphere with a thin graded
shell the contrast is at full strength almost everywhere, so at a high frequency every cell must resolve
the wave and a uniform grid is already the right grid. ``halo`` (defined in
``measure_graded_voxel_resolution.SHAPES``) is a body of the other kind, still radial so that the exact
solution is known: full contrast inside 0.2 a, falling over 0.2 a < r < 0.4 a to a tenth, a plateau of a
tenth out to 0.8 a, and a fall to zero at the surface (run it with core_frac = 0.2). A cell's first-order
error scales with the contrast it holds, so the weak plateau should tolerate cells about three times
larger than the strong centre does; a uniform grid cannot give it them.

A THIRD BODY: THE UNIFORM SPHERE WITH A SHARP SURFACE (``--shape=sharp``, core_frac = 1). The classical
test body, with the classical exact solution. Its medium is constant inside and zero outside, so a cell
wholly inside loses nothing of the medium and a cell cut by the surface loses a great deal: it is the body
on which a tree has most to save, because the medium's part of the error lives on the surface alone. A
constant cell holds the mean of the contrast over the cell, which for a cut cell is the fraction of its
volume inside the sphere. That fraction is computed here exactly in one direction and by a midpoint rule
of 64 x 64 points in the other two; the package's own six-point Gauss rule is not used for it, since it
cannot integrate a step and would add an error of the quadrature to the error of the scheme. The profile
handed to the package is then piecewise constant, each leaf at its own fraction.

UNIFORM GRIDS, WITH NO SOLVE. For comparison the first-order error of the uniform grids of 4, 8 and 16
cells across (and 32 when --hmin allows) is computed on the same test directions, from the same per-cell
Born far fields.

Results are written to ``scripts/measure_octree_first_order_greedy_ka<ka>_core<c>.json`` (with
``_<shape>`` appended for a shape other than s5).

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
    born_series_octree,
    octree_far_field,
    solve_graded_octree,
    uniform_leaves,
)

K_HAT = np.array([1.0, 0.0, 0.0])
EPS = 1e-3
D_STEP = 1e-2
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
    shape = opts.get("shape", "s5")
    rule = opts.get("rule", "residual")
    factor = float(opts.get("factor", 1.0))
    if rule not in ("residual", "own", "hybrid"):
        raise SystemExit(
            f"measure_octree_first_order_greedy: --rule={rule} is not a rule. Allowed: residual (the step "
            "that most reduces the first-order error of the tree) or own (the leaf whose own first-order "
            "error is largest) or hybrid (the first unless a leaf's own error exceeds factor times the "
            "tree's). Fix: pass --rule=residual, --rule=own or --rule=hybrid."
        )
    core = RADIUS * core_frac
    sharp = shape == "sharp"
    if sharp and core_frac != 1.0:
        raise SystemExit(
            f"measure_octree_first_order_greedy: --shape=sharp is the uniform sphere, whose core is the whole "
            f"sphere, but core_frac = {core_frac}. Fix: pass 1 as the second argument, e.g. "
            "'measure_octree_first_order_greedy.py 2 1 --shape=sharp'."
        )
    ref_shape = "s5" if sharp else shape  # with core = radius the exact solution has no shell to integrate
    omega = ka * REF.beta / RADIUS
    rf = 5e8 * RADIUS
    test_pts = obs_points(rf, THETA)
    grow_dirs = sphere_directions(N_GROWTH)
    all_pts = np.concatenate([rf * grow_dirs, test_pts])
    all_dirs = all_pts / rf
    n_grow = len(grow_dirs)

    fractions: dict = {}
    scope: dict = {}
    mid = (2.0 * np.arange(64) + 1.0) / 64.0 - 1.0

    def fraction(c: np.ndarray, h: float) -> float:
        """The fraction of the cell's volume inside the sphere: exact along z, midpoint rule in x and y."""
        key = (round(float(c[0]), 9), round(float(c[1]), 9), round(float(c[2]), 9), round(h, 9))
        if key not in fractions:
            x, y = np.meshgrid(c[0] + h * mid, c[1] + h * mid, indexing="ij")
            half = np.sqrt(np.clip(RADIUS**2 - x**2 - y**2, 0.0, None))
            length = np.clip(np.minimum(c[2] + h, half) - np.maximum(c[2] - h, -half), 0.0, None)
            fractions[key] = float(length.mean() / (2.0 * h))
        return fractions[key]

    def set_scope(cs: np.ndarray, hs_: np.ndarray) -> None:
        """Name the cells the piecewise-constant profile of the sharp sphere is to be read on."""
        if sharp:
            scope["c"], scope["h"] = cs, hs_
            scope["f"] = np.array([fraction(c, float(h)) for c, h in zip(cs, hs_, strict=True)])

    def prof(pos: np.ndarray) -> float:
        if not sharp:
            return radial(shape, core, float(np.linalg.norm(pos)))
        inside = np.all(np.abs(pos - scope["c"]) <= scope["h"][:, None], axis=1)
        return float(scope["f"][np.argmax(inside)])

    def exact_born_at(points: np.ndarray) -> np.ndarray:
        return (
            exact_field(ref_shape, core, omega, EPS, points)
            - exact_field(ref_shape, core, omega, -EPS, points)
        ) / (2 * EPS)

    born_exact = exact_born_at(all_pts)  # (directions, 3)
    exact = exact_field(ref_shape, core, omega, 1.0, test_pts)

    def even_part(d: float) -> np.ndarray:
        return (
            exact_field(ref_shape, core, omega, d, test_pts)
            + exact_field(ref_shape, core, omega, -d, test_pts)
        ) / (2 * d * d)

    second_exact = (4.0 * even_part(D_STEP) - even_part(2 * D_STEP)) / 3.0  # the exact second Born term
    peak = float(np.abs(exact).max())

    cache: dict = {}

    def born_of(c: np.ndarray, h: float) -> np.ndarray:
        """One cell's Born far field at every direction, (directions, 3); zero if it carries no contrast."""
        key = (round(float(c[0]), 9), round(float(c[1]), 9), round(float(c[2]), 9), round(h, 9))
        if key not in cache:
            if np.linalg.norm(c) - np.sqrt(3.0) * h >= RADIUS:
                cache[key] = np.zeros((len(all_dirs), 3), dtype=complex)
            else:
                set_scope(c[None, :], np.array([h]))
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
    others = (
        [t for t in json.loads(stored.read_text())["trees"] if "error" in t]
        if stored.exists() and shape == "s5"
        else []
    )
    tag = ("" if shape == "s5" else f"_{shape}") + (
        ""
        if rule == "residual"
        else f"_{rule}" + (f"{factor:g}" if rule == "hybrid" and factor != 1.0 else "")
    )
    out_path = ROOT / f"scripts/measure_octree_first_order_greedy_ka{ka:g}_core{core_frac:g}{tag}.json"
    result: dict = {
        "ka_s": ka,
        "core_frac": core_frac,
        "h_min": h_min,
        "shape": shape,
        "rule": rule,
        "hybrid_factor": factor if rule == "hybrid" else None,
        "growth_directions": N_GROWTH,
        "uniform": [],
        "trees": [],
    }
    print(
        f"an octree of constant cells grown by its first-order error (rule: {rule}): k_S a = {ka}, "
        f"core = {core_frac} a, shape {shape}, "
        f"{N_GROWTH} growth directions, judged on the nine test directions",
        flush=True,
    )
    n_uniform = 4
    while RADIUS / n_uniform >= h_min * (1.0 - 1e-9) and "--nouniform" not in sys.argv:
        cu, hu = uniform_leaves(RADIUS, n_uniform)
        kept = [(c, float(h)) for c, h in zip(cu, hu, strict=True) if carries(c, float(h))]
        row_u = {
            "n": n_uniform,
            "cells": len(kept),
            "dT1_test": test_error(sum(born_of(c, h) for c, h in kept)),
        }
        result["uniform"].append(row_u)
        print(
            f"  uniform grid of {n_uniform:2d} across, {row_u['cells']:5d} cells with contrast: first-order "
            f"error on the test directions {row_u['dT1_test']:.3e}",
            flush=True,
        )
        n_uniform *= 2
    block_cache: dict = {}
    t_start = time.perf_counter()
    for target in targets:
        while len(leaves) < target:
            residual = (total - born_exact)[:n_grow]
            best, best_norm, best_kids = None, np.inf, None
            own_best, own_norm, own_kids = None, 0.0, None
            for i, (c, h) in enumerate(leaves):
                if h <= h_min * (1.0 + 1e-9):
                    continue
                kids = [(c + 0.5 * h * s, 0.5 * h) for s in SIGNS]
                delta = sum(born_of(kc, kh) for kc, kh in kids) - born_of(c, h)
                own = float(np.linalg.norm(delta[:n_grow]))
                if own > own_norm:
                    own_best, own_norm, own_kids = i, own, kids
                norm = -own if rule == "own" else float(np.linalg.norm(residual + delta[:n_grow]))
                if norm < best_norm:
                    best, best_norm, best_kids = i, norm, kids
            if best is None:
                break
            if rule == "hybrid" and own_norm > factor * float(np.linalg.norm(residual)):
                best, best_kids = own_best, own_kids
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
            set_scope(centres, hs)
            series = born_series_octree(
                omega,
                REF,
                CONTRAST,
                centres,
                hs,
                prof,
                K_HAT,
                K_HAT,
                "P",
                2,
                p=0,
                r=0,
                block_cache=block_cache,
            )
            d_t2 = sum(octree_far_field(series[1], test_pts / rf, rf)) - second_exact
            first_again = sum(octree_far_field(series[0], test_pts / rf, rf))
            second_seconds = time.perf_counter() - t0
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
                dT2_test=float(np.abs(d_t2).max() / peak),
                two_term_test=float(np.abs(d_t1 + d_t2).max() / peak),
                beyond_two_terms=float(np.abs(err - d_t1 - d_t2).max() / peak),
                first_order_check=float(np.abs(first_again - total[n_grow:]).max() / peak),
                second_seconds=second_seconds,
                solve_seconds=time.perf_counter() - t0,
            )
        result["trees"].append(row)
        print(
            f"  grown to {row['cells']:5d} leaves {sizes}: first-order error on the test directions "
            f"{row['dT1_test']:.3e} (on the growth directions {row['dT1_growth']:.3e})"
            + (
                f"; two terms, no solve: {row['two_term_test']:.3e} (second term alone {row['dT2_test']:.3e}); "
                f"solved: error {row['error']:.3e}, beyond two terms {row['beyond_two_terms']:.3e}"
                if "error" in row
                else ""
            ),
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
