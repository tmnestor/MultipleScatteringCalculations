#!/usr/bin/env python3
"""Pilot: an adaptive octree against uniform grids, on a sphere with a thin graded shell.

The body is the graded sphere of ``measure_graded_voxel_resolution.py`` with the core radius raised, so
that the contrast is uniform over most of the sphere and varies only across a thin shell.  Its exact
solution is the radial reference, whatever the core radius.

Arms, all with cells of the same polynomial degree p (field and contrast):
  uniform  the uniform grid of n^3 cells, solved with the FFT solver (p >= 1) or the octree solver;
  octree   a tree grown from the n = 2 grid by refining every leaf whose detail energy exceeds a fraction
           tol of the profile's total energy (``octree.adapt_leaves``), solved densely.
For each: the number of cells that carry contrast, the projection defect D of the profile on those cells
(``octree.leaf_energies``), and the far-field error against the exact solution (max over nine angles,
relative to the peak, at 5e8 radii).

Run small first:
    conda run -n seismic python -u scripts/pilot_graded_voxel_octree.py --dry 0.5 0.75 1
    ... [--dry] <k_S a> <core_frac> <p> [--uniform=4,6,8] [--tol=1e-3,3e-4] [--hmin=0.625] [--summary=path]
        [--field=F] [--ufield=F]
--dry prints the trees and their defects without solving.
--field=F gives the leaves of a tree that are larger than its smallest a field of degree F, the contrast
staying at degree p everywhere: a large cell needs a richer basis for the wavefield, not for the medium.
--ufield=F gives every cell of the uniform grids a field of degree F, again with contrast degree p.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cubic_scattering.graded_voxel.farfield import graded_far_field  # noqa: E402
from cubic_scattering.graded_voxel.fft import solve_graded_sphere_fft  # noqa: E402
from cubic_scattering.graded_voxel.octree import (  # noqa: E402
    adapt_leaves,
    leaf_energies,
    octree_far_field,
    solve_graded_octree,
    uniform_leaves,
)
from gate_sphere_cell_average_vs_mie import CONTRAST, REF, obs_points  # noqa: E402
from measure_graded_voxel_resolution import exact_field, radial  # noqa: E402
from pilot_graded_sphere_vs_exact import RADIUS, THETA  # noqa: E402

K_HAT = np.array([1.0, 0.0, 0.0])
DENSE_LIMIT = 20000  # unknowns the dense octree solve is allowed


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    dry = "--dry" in sys.argv
    ka, core, p = float(args[0]), RADIUS * float(args[1]), int(args[2])
    uniform_ns = [int(v) for v in opts.get("uniform", "4,6,8").split(",")]
    tols = [float(v) for v in opts.get("tol", "1e-3,3e-4,1e-4").split(",")]
    h_min = float(opts.get("hmin", RADIUS / 16))
    field = int(opts["field"]) if "field" in opts else None
    ufield = int(opts["ufield"]) if "ufield" in opts else p
    omega = ka * REF.beta / RADIUS
    rf = 5e8 * RADIUS
    pts = obs_points(rf, THETA)

    def prof_vec(pos: np.ndarray) -> np.ndarray:
        x = np.clip((RADIUS - np.linalg.norm(pos, axis=-1)) / (RADIUS - core), 0.0, 1.0)
        return x**3 * (10.0 - 15.0 * x + 6.0 * x**2)

    def prof(pos: np.ndarray) -> float:
        return radial("s5", core, float(np.linalg.norm(pos)))

    exact = None if dry else exact_field("s5", core, omega, 1.0, pts)
    peak = None if dry else float(np.abs(exact).max())
    out: dict = {"ka_s": ka, "core_frac": core / RADIUS, "p": p, "field": field, "ufield": ufield}
    out.update({"uniform": [], "octree": []})
    print(f"thin-shell sphere: k_S a = {ka}, core = {core / RADIUS} a, degree p = r = {p}", flush=True)

    def describe(centres: np.ndarray, hs: np.ndarray) -> tuple[int, float]:
        defect, _, norm = leaf_energies(prof_vec, centres, hs, p)
        return int((norm > 0).sum()), float(defect.sum() / norm.sum())

    for n in uniform_ns:
        centres, hs = uniform_leaves(RADIUS, n)
        cells, d = describe(centres, hs)
        row = {"n": n, "cells": cells, "unknowns": cells * (1, 4, 10)[ufield] * 9, "defect": d}
        if not dry:
            t0 = time.perf_counter()
            if ufield >= 1:
                res = solve_graded_sphere_fft(
                    omega, RADIUS, REF, CONTRAST, n, prof, K_HAT, K_HAT, "P", p=ufield, r=p
                )
                up, us = graded_far_field(res, pts / rf, rf, K_HAT, K_HAT, "P")
            else:
                ores = solve_graded_octree(
                    omega, REF, CONTRAST, centres, hs, prof, K_HAT, K_HAT, "P", p=0, r=0
                )
                up, us = octree_far_field(ores, pts / rf, rf)
            row["error"] = float(np.abs(up + us - exact).max() / peak)
            row["seconds"] = time.perf_counter() - t0
        out["uniform"].append(row)
        print(
            f"  uniform n {n:3d}: " + "  ".join(f"{k} {v:.4g}" for k, v in row.items() if k != "n"),
            flush=True,
        )

    base_c, base_h = uniform_leaves(RADIUS, 2)
    for tol in tols:
        centres, hs = adapt_leaves(prof_vec, base_c, base_h, p, tol, h_min)
        cells, d = describe(centres, hs)
        sizes = {float(h): int((hs == h).sum()) for h in np.unique(hs)}
        p_leaf = np.full(len(hs), p) if field is None else np.where(hs > hs.min(), field, p)
        unknowns = int(sum((1, 4, 10)[int(v)] * 9 for v in p_leaf))
        row = {"tol": tol, "cells": cells, "unknowns": unknowns, "defect": d, "sizes": sizes}
        if not dry:
            if row["unknowns"] > DENSE_LIMIT:
                print(
                    f"  octree tol {tol:g}: {cells} cells, {row['unknowns']} unknowns: "
                    "over the dense limit, skipped"
                )
                continue
            t0 = time.perf_counter()
            ores = solve_graded_octree(
                omega, REF, CONTRAST, centres, hs, prof, K_HAT, K_HAT, "P", p=p_leaf, r=p
            )
            up, us = octree_far_field(ores, pts / rf, rf)
            row["error"] = float(np.abs(up + us - exact).max() / peak)
            row["seconds"] = time.perf_counter() - t0
        out["octree"].append(row)
        shown = "  ".join(f"{k} {v:.4g}" for k, v in row.items() if k not in ("tol", "sizes"))
        print(f"  octree tol {tol:g}: {shown}  leaves by half-width {sizes}", flush=True)
    if "summary" in opts:
        path = Path(opts["summary"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2) + "\\n")
        print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
