#!/usr/bin/env python3
"""Pilot: a constant (Haar) medium in every leaf, the field degree chosen leaf by leaf.

The body, the exact solution and the two indicators are those of ``pilot_octree_two_term_refinement.py``.
Here the contrast degree r is the same in every leaf (0: Haar cells) and the FIELD degree is a choice
made for each leaf.  A leaf whose indicators sum to more than tol is

    raised    in field degree, when its wave indicator (its own first-order error) is the larger of the
              two and its degree is below p_max: a large leaf in a smooth part of the medium needs a
              richer basis for the wavefield, not more cells;
    split     into its eight children otherwise: the medium indicator (its share of the projection
              error) falls only by refinement when the contrast degree is fixed.

COMPARED, at contrast degree r: uniform grids with field degree 0 and 1; trees with the field degree fixed
at 0 and at 1 (the two-term rule, splitting only); trees with the degree chosen per leaf.  For each: cells,
unknowns, the predicted error (no reference) and the true error against the exact solution.

Run small first:
    conda run -n seismic python -u scripts/pilot_octree_hp_refinement.py 1.0 --dry
    ... <k_S a> [--dry] [--r=0] [--pmax=1] [--uniform=4,6,8] [--tol=1e-3,3e-4,1e-4] [--hmin=0.3125]
        [--core=0.1] [--width=0.3] [--summary=path]
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pilot_octree_two_term_refinement as base  # noqa: E402
from cubic_scattering.graded_voxel.octree import (  # noqa: E402
    _descendants,
    adapt_leaves,
    born_octree,
    leaf_energies,
    octree_far_field,
    refine_leaves,
    solve_graded_octree,
    uniform_leaves,
)
from gate_sphere_cell_average_vs_mie import CONTRAST, REF, obs_points  # noqa: E402
from pilot_graded_sphere_vs_exact import RADIUS, THETA  # noqa: E402

K_HAT = base.K_HAT
N_FUN = (1, 4, 10)
DENSE_LIMIT = 20000


class Body:
    """The frequency, the observers and the per-leaf first-order errors, for contrast degree r."""

    def __init__(self, ka: float, r: int) -> None:
        self.omega = ka * REF.beta / RADIUS
        self.r = r
        self.rf = 5e8 * RADIUS
        self.pts = obs_points(self.rf, THETA)
        self.cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

    def far(self, centres: np.ndarray, hs: np.ndarray, p, r: int) -> np.ndarray:
        res = born_octree(self.omega, REF, CONTRAST, centres, hs, base.prof, K_HAT, K_HAT, "P", p=p, r=r)
        return sum(octree_far_field(res, self.pts / self.rf, self.rf))

    def leaf_born_error(self, c: np.ndarray, h: float, p: int) -> tuple[np.ndarray, np.ndarray]:
        """One leaf's first-order far field (field degree p, contrast degree r) minus that of its eight
        children with quadratic cells; and the latter."""
        key = (round(float(c[0]), 9), round(float(c[1]), 9), round(float(c[2]), 9), round(float(h), 9))
        if key not in self.cache:
            kids, hk = _descendants(c, h, 1)
            self.cache[key] = {"fine": self.far(kids, np.full(len(kids), hk), 2, 2)}
        entry = self.cache[key]
        if p not in entry:
            entry[p] = self.far(c[None, :], np.array([h]), p, self.r) - entry["fine"]
        return entry[p], entry["fine"]

    def indicators(self, centres: np.ndarray, hs: np.ndarray, p_leaf: np.ndarray, nu: float) -> dict:
        defect, _, norm = leaf_energies(base.prof_vec, centres, hs, self.r)
        pairs = [
            self.leaf_born_error(c, float(h), int(p)) for c, h, p in zip(centres, hs, p_leaf, strict=True)
        ]
        d_t1 = np.array([e for e, _ in pairs])
        peak = float(np.abs(sum(f for _, f in pairs)).max())
        e_r = float(defect.sum() / norm.sum())
        first = float(np.abs(d_t1.sum(axis=0)).max() / peak)
        return {
            "medium": nu * defect / norm.sum(),
            "wave": np.abs(d_t1).reshape(len(hs), -1).max(axis=1) / peak,
            "norm": norm,
            "E": e_r,
            "first_order": first,
            "predicted": first + nu * e_r,
        }

    def adapt(
        self, tol: float, h_min: float, nu: float, p_start: int, p_max: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Leaves and their field degrees: raise the degree where the wave indicator leads, else split."""
        centres, hs = uniform_leaves(RADIUS, 2)
        p_leaf = np.full(len(hs), p_start)
        while True:
            ind = self.indicators(centres, hs, p_leaf, nu)
            over = ind["medium"] + ind["wave"] > tol
            raise_p = over & (ind["wave"] > ind["medium"]) & (p_leaf < p_max)
            split = over & ~raise_p & (hs > h_min * (1.0 + 1e-9))
            if not raise_p.any() and not split.any():
                keep = ind["norm"] > 0.0
                return centres[keep], hs[keep], p_leaf[keep]
            p_leaf = np.where(raise_p, p_leaf + 1, p_leaf)
            if split.any():
                kept_p = p_leaf[~split]
                child_p = np.repeat(p_leaf[split], 8)
                centres, hs = refine_leaves(centres, hs, split)
                p_leaf = np.concatenate([kept_p, child_p])


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    dry = "--dry" in sys.argv
    base.CORE = RADIUS * float(opts.get("core", 0.1))
    base.WIDTH = RADIUS * float(opts.get("width", 0.3))
    ka = float(args[0])
    r, p_max = int(opts.get("r", 0)), int(opts.get("pmax", 1))
    uniform_ns = [int(v) for v in opts.get("uniform", "4,6,8").split(",") if v]
    tols = [float(v) for v in opts.get("tol", "1e-3,3e-4,1e-4").split(",") if v]
    h_min = float(opts.get("hmin", RADIUS / 32))
    body = Body(ka, r)
    t0 = time.perf_counter()
    print(
        f"halo and compact feature: k_S a = {ka}, core {base.CORE / RADIUS} a, feature to "
        f"{base.WIDTH / RADIUS} a; contrast degree r = {r}, field degree up to {p_max}",
        flush=True,
    )
    cache: dict = {}
    # the nonlinear fraction, from one solve on a coarse tree that resolves the medium: no reference
    c2, h2 = adapt_leaves(base.prof_vec, *uniform_leaves(RADIUS, 2), r, 1e-2, h_min)
    coarse = solve_graded_octree(
        body.omega, REF, CONTRAST, c2, h2, base.prof, K_HAT, K_HAT, "P", p=p_max, r=r, block_cache=cache
    )
    u2 = sum(octree_far_field(coarse, body.pts / body.rf, body.rf))
    nu = float(np.abs(u2 - body.far(c2, h2, p_max, r)).max() / np.abs(u2).max())
    print(f"   nonlinear fraction from a coarse tree ({len(h2)} leaves): nu = {nu:.4f}", flush=True)
    exact = peak = None
    if not dry:
        exact = base.mres.exact_field(base.SHAPE, base.CORE, body.omega, 1.0, body.pts)
        peak = float(np.abs(exact).max())

    grids = []
    for p in range(p_max + 1):
        for n in uniform_ns:
            c, h = uniform_leaves(RADIUS, n)
            grids.append((f"uniform n {n}, field {p}", c, h, np.full(len(h), p)))
    for p in range(p_max + 1):
        grids += [(f"tree, field {p}, tol {t:g}", *body.adapt(t, h_min, nu, p, p)) for t in tols]
    grids += [(f"tree, field chosen, tol {t:g}", *body.adapt(t, h_min, nu, 0, p_max)) for t in tols]
    print("   grid: cells | unknowns | E_r | first-order | predicted | true error | field degrees | sizes")
    rows = []
    for name, centres, hs, p_leaf in grids:
        ind = body.indicators(centres, hs, p_leaf, nu)
        keep = ind["norm"] > 0
        centres, hs, p_leaf = centres[keep], hs[keep], p_leaf[keep]
        unknowns = int(sum(N_FUN[int(p)] * 9 for p in p_leaf))
        degrees = {int(p): int((p_leaf == p).sum()) for p in np.unique(p_leaf)}
        sizes = {float(h): int((hs == h).sum()) for h in np.unique(hs)}
        row = {
            "grid": name,
            "cells": len(hs),
            "unknowns": unknowns,
            "E": ind["E"],
            "first_order": ind["first_order"],
            "predicted": ind["predicted"],
            "degrees": degrees,
            "sizes": sizes,
        }
        if not dry and unknowns <= DENSE_LIMIT:
            res = solve_graded_octree(
                body.omega,
                REF,
                CONTRAST,
                centres,
                hs,
                base.prof,
                K_HAT,
                K_HAT,
                "P",
                p=p_leaf,
                r=r,
                block_cache=cache,
            )
            u = sum(octree_far_field(res, body.pts / body.rf, body.rf))
            row["error"] = float(np.abs(u - exact).max() / peak)
        rows.append(row)
        true = f"{row['error']:.3e}" if "error" in row else "   -     "
        print(
            f"   {name:30s}: {row['cells']:5d} | {unknowns:6d} | {row['E']:.3e} | "
            f"{row['first_order']:.3e} | "
            f"{row['predicted']:.3e} | {true} | {degrees} | {sizes}   [{time.perf_counter() - t0:.0f} s]",
            flush=True,
        )
    if "summary" in opts:
        path = Path(opts["summary"])
        path.parent.mkdir(parents=True, exist_ok=True)
        out = {"ka_s": ka, "r": r, "p_max": p_max, "nu": nu, "core": base.CORE, "width": base.WIDTH}
        path.write_text(json.dumps({**out, "rows": rows}, indent=2) + "\n")
        print(f"   wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
