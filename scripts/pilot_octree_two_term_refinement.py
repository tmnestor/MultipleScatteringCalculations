#!/usr/bin/env python3
"""Pilot: refinement by the two terms of the error, on a sphere with a compact feature.

THE BODY.  A sphere of radius a whose contrast is a weak smooth halo over the whole radius plus a strong
compact feature at the centre:

    f(r) = HALO s5((a - r) / (a - core)) + (1 - HALO) s5((W - r) / (W - core)),   s5 the quintic step,

f = 1 for r < core, the second term zero for r > W.  With core = 0.2 a, W = 0.5 a and HALO = 0.05 the
feature fills an eighth of the volume and carries a share of the scattering comparable with the halo's.
Its exact solution is the radial reference of ``measure_graded_voxel_resolution``.

THE RULE.  Before any solve each leaf has two indicators (``measure_octree_two_term.py``):
  medium   nu x (the leaf's projection error) / (the profile's total energy), with nu the nonlinear
           fraction of the response, estimated from one solve on a coarse tree that resolves the medium
           (the medium-only rule at a loose tolerance);
  wave     the leaf's first-order error: the far field of its projected incident wave
           (``octree.born_octree``) against that of its eight children with quadratic cells,
           over the peak of the Born far field.
A leaf is refined while their sum exceeds tol, down to half-width h_min.  The predicted error of a tree is
|sum over leaves of the first-order errors| / peak + nu x E, with no reference to the exact solution.

COMPARED: uniform grids; the medium-only rule (``octree.adapt_leaves``); the two-term rule.  For each:
cells, the predicted error and the true error against the exact solution.

Run small first:
    conda run -n seismic python -u scripts/pilot_octree_two_term_refinement.py 1.0 0 --dry
    ... <k_S a> <p> [--dry] [--uniform=4,6,8] [--tol=3e-4,1e-4] [--mtol=1e-3,3e-4] [--hmin=0.625]
        [--summary=path] [--core=0.2] [--width=0.5]
--core and --width set the feature: the contrast falls from 1 to the halo between core x a and width x a.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import measure_graded_voxel_resolution as mres  # noqa: E402
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

K_HAT = np.array([1.0, 0.0, 0.0])
CORE, WIDTH, HALO = 0.2 * RADIUS, 0.5 * RADIUS, 0.05
DENSE_LIMIT = 20000
SHAPE = "halo_feature"


def s5(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x**3 * (10.0 - 15.0 * x + 6.0 * x**2)


def profile_r(r: np.ndarray) -> np.ndarray:
    """The contrast factor at radius r (vectorised)."""
    return HALO * s5((RADIUS - r) / (RADIUS - CORE)) + (1.0 - HALO) * s5((WIDTH - r) / (WIDTH - CORE))


# the radial reference takes the profile as a function of x = (a - r) / (a - core)
mres.SHAPES[SHAPE] = lambda x: float(profile_r(np.asarray(RADIUS - x * (RADIUS - CORE))))


def prof_vec(pos: np.ndarray) -> np.ndarray:
    return profile_r(np.linalg.norm(pos, axis=-1))


def prof(pos: np.ndarray) -> float:
    return float(profile_r(np.linalg.norm(pos)))


class Body:
    """The frequency, the observers and the per-leaf first-order errors."""

    def __init__(self, ka: float, p: int) -> None:
        self.omega = ka * REF.beta / RADIUS
        self.p = p
        self.rf = 5e8 * RADIUS
        self.pts = obs_points(self.rf, THETA)
        self.cache: dict[tuple, np.ndarray] = {}

    def far(self, centres: np.ndarray, hs: np.ndarray, p: int) -> np.ndarray:
        res = born_octree(self.omega, REF, CONTRAST, centres, hs, prof, K_HAT, K_HAT, "P", p=p, r=p)
        return sum(octree_far_field(res, self.pts / self.rf, self.rf))

    def leaf_born_error(self, c: np.ndarray, h: float) -> np.ndarray:
        """One leaf's first-order far field minus that of its eight children (quadratic cells)."""
        key = (round(float(c[0]), 9), round(float(c[1]), 9), round(float(c[2]), 9), round(float(h), 9))
        if key not in self.cache:
            kids, hk = _descendants(c, h, 1)
            fine = self.far(kids, np.full(len(kids), hk), 2)
            self.cache[key] = (self.far(c[None, :], np.array([h]), self.p) - fine, fine)
        return self.cache[key]

    def indicators(self, centres: np.ndarray, hs: np.ndarray, nu: float) -> dict:
        """Per leaf: the medium and the wave indicator; and the predicted error of the tree."""
        defect, _, norm = leaf_energies(prof_vec, centres, hs, self.p)
        pairs = [self.leaf_born_error(c, float(h)) for c, h in zip(centres, hs, strict=True)]
        d_t1 = np.array([e for e, _ in pairs])
        born = sum(f for _, f in pairs)
        peak = float(np.abs(born).max())
        medium = nu * defect / norm.sum()
        wave = np.abs(d_t1).reshape(len(hs), -1).max(axis=1) / peak
        e_p = float(defect.sum() / norm.sum())
        first = float(np.abs(d_t1.sum(axis=0)).max() / peak)
        return {
            "medium": medium,
            "wave": wave,
            "norm": norm,
            "E": e_p,
            "first_order": first,
            "predicted": first + nu * e_p,
        }

    def adapt(self, tol: float, h_min: float, nu: float) -> tuple[np.ndarray, np.ndarray]:
        centres, hs = uniform_leaves(RADIUS, 2)
        while True:
            ind = self.indicators(centres, hs, nu)
            flag = (ind["medium"] + ind["wave"] > tol) & (hs > h_min * (1.0 + 1e-9))
            if not flag.any():
                keep = ind["norm"] > 0.0
                return centres[keep], hs[keep]
            centres, hs = refine_leaves(centres, hs, flag)


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    dry = "--dry" in sys.argv
    global CORE, WIDTH  # the feature's extent, read by profile_r at every call
    CORE = RADIUS * float(opts.get("core", CORE / RADIUS))
    WIDTH = RADIUS * float(opts.get("width", WIDTH / RADIUS))
    ka, p = float(args[0]), int(args[1])
    uniform_ns = [int(v) for v in opts.get("uniform", "4,6,8").split(",") if v]
    tols = [float(v) for v in opts.get("tol", "3e-4,1e-4").split(",") if v]
    mtols = [float(v) for v in opts.get("mtol", "1e-3,3e-4").split(",") if v]
    h_min = float(opts.get("hmin", RADIUS / 16))
    body = Body(ka, p)
    t0 = time.perf_counter()
    print(
        f"halo and compact feature: k_S a = {ka}, core {CORE / RADIUS} a, feature to {WIDTH / RADIUS} a, "
        f"halo {HALO}; degree p = r = {p}",
        flush=True,
    )
    cache: dict = {}  # coupling blocks, shared by every solve at this frequency
    # the nonlinear fraction, from one solve on a coarse tree that resolves the medium: no reference
    c2, h2 = adapt_leaves(prof_vec, *uniform_leaves(RADIUS, 2), p, 1e-2, h_min)
    coarse = solve_graded_octree(
        body.omega, REF, CONTRAST, c2, h2, prof, K_HAT, K_HAT, "P", p=p, r=p, block_cache=cache
    )
    u2 = sum(octree_far_field(coarse, body.pts / body.rf, body.rf))
    b2 = body.far(c2, h2, p)
    nu = float(np.abs(u2 - b2).max() / np.abs(u2).max())
    print(f"   nonlinear fraction from a coarse tree ({len(h2)} leaves): nu = {nu:.4f}", flush=True)
    exact = peak = None
    if not dry:
        exact = mres.exact_field(SHAPE, CORE, body.omega, 1.0, body.pts)
        eps = 1e-3
        ex_born = (
            mres.exact_field(SHAPE, CORE, body.omega, eps, body.pts)
            - mres.exact_field(SHAPE, CORE, body.omega, -eps, body.pts)
        ) / (2 * eps)
        peak = float(np.abs(exact).max())
        print(
            f"   exact nonlinear fraction {np.abs(exact - ex_born).max() / peak:.4f}   "
            f"[{time.perf_counter() - t0:.0f} s]",
            flush=True,
        )

    grids = [(f"uniform n {n}", *uniform_leaves(RADIUS, n)) for n in uniform_ns]
    base = uniform_leaves(RADIUS, 2)
    grids += [(f"medium tol {t:g}", *adapt_leaves(prof_vec, *base, p, t, h_min)) for t in mtols]
    grids += [(f"two-term tol {t:g}", *body.adapt(t, h_min, nu)) for t in tols]
    print("   grid: cells | E_p | first-order (no solve) | predicted | true error | leaves by half-width")
    rows = []
    for name, centres, hs in grids:
        ind = body.indicators(centres, hs, nu)
        keep = ind["norm"] > 0
        centres, hs = centres[keep], hs[keep]
        sizes = {float(h): int((hs == h).sum()) for h in np.unique(hs)}
        row = {
            "grid": name,
            "cells": len(hs),
            "E": ind["E"],
            "first_order": ind["first_order"],
            "predicted": ind["predicted"],
            "sizes": sizes,
        }
        unknowns = len(hs) * (1, 4, 10)[p] * 9
        if not dry and unknowns <= DENSE_LIMIT:
            res = solve_graded_octree(
                body.omega, REF, CONTRAST, centres, hs, prof, K_HAT, K_HAT, "P", p=p, r=p, block_cache=cache
            )
            u = sum(octree_far_field(res, body.pts / body.rf, body.rf))
            row["error"] = float(np.abs(u - exact).max() / peak)
        rows.append(row)
        true = f"{row['error']:.3e}" if "error" in row else "   -     "
        print(
            f"   {name:18s}: {row['cells']:5d} | {row['E']:.3e} | {row['first_order']:.3e} | "
            f"{row['predicted']:.3e} | {true} | {sizes}   [{time.perf_counter() - t0:.0f} s]",
            flush=True,
        )
    if "summary" in opts:
        path = Path(opts["summary"])
        path.parent.mkdir(parents=True, exist_ok=True)
        out = {"ka_s": ka, "p": p, "nu": nu, "core": CORE, "width": WIDTH, "halo": HALO, "rows": rows}
        path.write_text(json.dumps(out, indent=2) + "\n")
        print(f"   wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
