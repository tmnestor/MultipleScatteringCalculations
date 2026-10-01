#!/usr/bin/env python3
"""The error of an octree of polynomial cells in two terms, each known before the solve.

The body is the thin-shell sphere of ``pilot_graded_voxel_octree.py`` (uniform core, graded shell), whose
exact solution is the radial reference at any contrast.  The error of the far field is separated by its
order in the contrast:

  first order   dT1 = (the scheme's Born term) - (the exact Born term).  The scheme's Born term needs no
                solve (``octree.born_octree``: each leaf holds the projected incident wave), and the exact
                one is a central difference of the exact solution in the contrast.  It carries the wave
                term of every leaf (``octree.born_wave_factor``) and the first-order effect of the medium's
                projection;
  the rest      error - dT1: the terms of second and higher order, which for linear cells follow the
                relative projection error E of the medium with one constant on uniform grids and trees.

Measured, for uniform grids and adaptive trees, at field and contrast degree p (0: Haar cells; 1: linear):
the error, dT1, the rest, the projection error E_p, and the ratios error / E_p and rest / E_p.  If the
rest follows E_p with one constant while error / E_p does not, the two terms together predict the error.

Run small first:
    conda run -n seismic python -u scripts/measure_octree_two_term.py 0.5 0.75 0 --uniform=4,6 --tol=1e-3
    ... <k_S a> <core_frac> <p> [--uniform=4,6,8] [--tol=1e-3,3e-4] [--hmin=0.625] [--summary=path]
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cubic_scattering.graded_voxel.octree import (  # noqa: E402
    adapt_leaves,
    born_octree,
    leaf_energies,
    octree_far_field,
    solve_graded_octree,
    uniform_leaves,
)
from gate_sphere_cell_average_vs_mie import CONTRAST, REF, obs_points  # noqa: E402
from measure_graded_voxel_resolution import exact_field, radial  # noqa: E402
from pilot_graded_sphere_vs_exact import RADIUS, THETA  # noqa: E402

K_HAT = np.array([1.0, 0.0, 0.0])
EPS = 1e-3


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    ka, core, p = float(args[0]), RADIUS * float(args[1]), int(args[2])
    uniform_ns = [int(v) for v in opts.get("uniform", "4,6,8").split(",") if v]
    tols = [float(v) for v in opts.get("tol", "1e-3,3e-4").split(",") if v]
    h_min = float(opts.get("hmin", RADIUS / 16))
    omega = ka * REF.beta / RADIUS
    rf = 5e8 * RADIUS
    pts = obs_points(rf, THETA)

    def prof_vec(pos: np.ndarray) -> np.ndarray:
        x = np.clip((RADIUS - np.linalg.norm(pos, axis=-1)) / (RADIUS - core), 0.0, 1.0)
        return x**3 * (10.0 - 15.0 * x + 6.0 * x**2)

    def prof(pos: np.ndarray) -> float:
        return radial("s5", core, float(np.linalg.norm(pos)))

    exact = exact_field("s5", core, omega, 1.0, pts)
    exact_born = (exact_field("s5", core, omega, EPS, pts) - exact_field("s5", core, omega, -EPS, pts)) / (
        2 * EPS
    )
    peak = float(np.abs(exact).max())
    print(
        f"thin-shell sphere: k_S a = {ka}, core = {core / RADIUS} a, degree p = r = {p};  "
        f"|exact - exact Born| / peak = {np.abs(exact - exact_born).max() / peak:.3e}",
        flush=True,
    )
    print("   grid: cells | error | dT1 (no solve) | rest | E_p | error / E_p | rest / E_p")
    grids = [(f"uniform n {n}", *uniform_leaves(RADIUS, n)) for n in uniform_ns]
    base_c, base_h = uniform_leaves(RADIUS, 2)
    grids += [(f"tree tol {tol:g}", *adapt_leaves(prof_vec, base_c, base_h, p, tol, h_min)) for tol in tols]
    rows = []
    for name, centres, hs in grids:
        t0 = time.perf_counter()
        defect, _, norm = leaf_energies(prof_vec, centres, hs, p)
        keep = norm > 0
        centres, hs = centres[keep], hs[keep]
        e_p = float(defect.sum() / norm.sum())
        born = sum(
            octree_far_field(
                born_octree(omega, REF, CONTRAST, centres, hs, prof, K_HAT, K_HAT, "P", p=p, r=p),
                pts / rf,
                rf,
            )
        )
        d_t1 = born - exact_born
        res = solve_graded_octree(omega, REF, CONTRAST, centres, hs, prof, K_HAT, K_HAT, "P", p=p, r=p)
        err = sum(octree_far_field(res, pts / rf, rf)) - exact
        row = {
            "grid": name,
            "cells": len(hs),
            "error": float(np.abs(err).max() / peak),
            "dT1": float(np.abs(d_t1).max() / peak),
            "rest": float(np.abs(err - d_t1).max() / peak),
            "E": e_p,
            "seconds": time.perf_counter() - t0,
        }
        rows.append(row)
        print(
            f"   {name:16s}: {row['cells']:5d} | {row['error']:.3e} | {row['dT1']:.3e} | "
            f"{row['rest']:.3e} | "
            f"{e_p:.3e} | {row['error'] / e_p:.4f} | {row['rest'] / e_p:.4f}   [{row['seconds']:.0f} s]",
            flush=True,
        )
    full = np.array([r["error"] / r["E"] for r in rows])
    rest = np.array([r["rest"] / r["E"] for r in rows])
    print(
        f"   spread (max / min): error / E_p {full.max() / full.min():.2f}, rest / E_p "
        f"{rest.max() / rest.min():.2f}"
    )
    if "summary" in opts:
        path = Path(opts["summary"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"ka_s": ka, "core_frac": core / RADIUS, "p": p, "rows": rows}, indent=2) + "\n"
        )
        print(f"   wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
