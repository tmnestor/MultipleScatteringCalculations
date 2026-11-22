#!/usr/bin/env python3
"""An adaptive tree refined by what a cell loses of (medium x wave), against one refined by the medium
alone: the test of the refinement rule that the frequency dependence of the error law implies.

THE QUESTION. The error of the nonlinear part of the response was first found proportional to the
relative projection error of the medium, so an adaptive tree was refined where the medium's detail
coefficients are large (``octree.adapt_leaves``). But a cell holds the product of the medium f and the
wave w = exp(i k.x), and what it loses is the detail of that product: for constant cells its slope,
grad(f w) = w (grad f + i k f), whose energy is |grad f|^2 + k^2 f^2. Where the medium is uniform its own
detail is zero and the product's is not. The rule this implies is: refine where the detail of f w is
large. Does a tree built that way reach a given error with fewer cells, once the body is not small
against the wavelength?

THE TWO RULES. Both start from the grid of 2 cells across and refine, repeatedly, every leaf whose
indicator exceeds a fraction ``tol`` of the total, down to a smallest half-width.

  medium    indicator = the energy of the leaf's own detail coefficients of f (what one refinement of
            the leaf recovers of f), over int f^2.
  product   indicator = the same for f cos(k x) and f sin(k x), the real and imaginary parts of f w,
            summed over two waves along x with the P and the S wavenumber (the incident wave and the
            shorter outgoing wave), over the total energy of those four functions. For constant cells
            its leading term is (h^2 / 3) int_leaf (|grad f|^2 + k^2 f^2) dV; here it is computed
            exactly, by projection, so that it serves degree-one cells as well.

WHAT IS MEASURED, for each tree: the number of leaves that carry contrast and of unknowns; the error of
the far field against the exact solution (largest over nine angles, relative to the peak); the
first-order error, computed without a solve; the rest; the relative projection error of the medium on
the tree, E_medium; and that of the product, E_product. Uniform grids solved by the FFT solver, read from
``measure_two_term_fft_*.json`` where stored, give the reference curve.

The body is the sphere with a uniform core and a graded shell, whose exact solution is the radial
reference. The trees are solved densely, so they stop near 20 000 unknowns.

Results are written after every tree to
``scripts/measure_octree_product_indicator_ka<ka>_core<c>_p<p>.json``.

Run small first:
    conda run -n seismic python -u scripts/measure_octree_product_indicator.py 2 0.75 0 --tol=1e-2,3e-3
    ... <k_S a> <core_frac> <p> [--tol=1e-2,3e-3,1e-3,3e-4,1e-4] [--hmin=0.625] [--dry]
--dry builds the trees and prints their sizes and projection errors without solving.
"""

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
    leaf_energies,
    octree_far_field,
    refine_leaves,
    solve_graded_octree,
    uniform_leaves,
)

K_HAT = np.array([1.0, 0.0, 0.0])
EPS = 1e-3
DENSE_LIMIT = 20000  # unknowns the dense tree solve is allowed
UNKNOWNS_PER_CELL = {0: 9, 1: 36, 2: 90}


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    dry = "--dry" in sys.argv
    ka, core_frac, p = float(args[0]), float(args[1]), int(args[2])
    tols = [float(v) for v in opts.get("tol", "1e-2,3e-3,1e-3,3e-4,1e-4").split(",") if v]
    h_min = float(opts.get("hmin", RADIUS / 16))
    core = RADIUS * core_frac
    omega = ka * REF.beta / RADIUS
    k_p, k_s = omega / REF.alpha, omega / REF.beta
    rf = 5e8 * RADIUS
    pts = obs_points(rf, THETA)

    def prof_vec(pos: np.ndarray) -> np.ndarray:
        x = np.clip((RADIUS - np.linalg.norm(pos, axis=-1)) / (RADIUS - core), 0.0, 1.0)
        return x**3 * (10.0 - 15.0 * x + 6.0 * x**2)

    def prof(pos: np.ndarray) -> float:
        return radial("s5", core, float(np.linalg.norm(pos)))

    # the four real functions whose detail is the detail of f w for the two waves
    products = [
        (lambda pos, _k=k, _fn=fn: prof_vec(pos) * _fn(_k * pos[..., 0]))
        for k in (k_p, k_s)
        for fn in (np.cos, np.sin)
    ]

    def energies(rule: str, centres: np.ndarray, hs: np.ndarray, levels: int) -> tuple:
        """(defect, detail, norm) per leaf, of the medium or summed over the four product functions."""
        funcs = [prof_vec] if rule == "medium" else products
        parts = [leaf_energies(fn, centres, hs, p, levels=levels) for fn in funcs]
        return tuple(sum(part[i] for part in parts) for i in range(3))

    def adapt(rule: str, tol: float) -> tuple[np.ndarray, np.ndarray]:
        centres, hs = uniform_leaves(RADIUS, 2)
        while True:
            _, detail, norm = energies(rule, centres, hs, 2)
            flag = (detail > tol * norm.sum()) & (hs > h_min * (1.0 + 1e-9))
            if not flag.any():
                keep = leaf_energies(prof_vec, centres, hs, p, levels=2)[2] > 0.0
                return centres[keep], hs[keep]
            centres, hs = refine_leaves(centres, hs, flag)

    def projection_errors(centres: np.ndarray, hs: np.ndarray) -> tuple[float, float]:
        out = []
        for rule in ("medium", "product"):
            defect, _, norm = energies(rule, centres, hs, 2)
            out.append(float(defect.sum() / norm.sum()))
        return out[0], out[1]

    tag = f"ka{ka:g}_core{core_frac:g}_p{p}"
    out_path = ROOT / f"scripts/measure_octree_product_indicator_{tag}.json"
    result: dict = {"ka_s": ka, "core_frac": core_frac, "p": p, "h_min": h_min, "trees": [], "uniform": []}
    print(
        f"refinement by the product against refinement by the medium: k_S a = {ka}, core = {core_frac} a, "
        f"degree p = r = {p}, smallest half-width {h_min:g} m",
        flush=True,
    )

    # the uniform grids already measured by the FFT solver, with their two projection errors
    stored = ROOT / f"scripts/measure_two_term_fft_ka{ka:g}_core{core_frac:g}_p{p}.json"
    if stored.exists():
        for row in json.loads(stored.read_text())["rows"]:
            if row["n"] > 16:
                continue
            centres, hs = uniform_leaves(RADIUS, row["n"])
            e_med, e_prod = projection_errors(centres, hs)
            result["uniform"].append(
                {
                    "n": row["n"],
                    "cells": row["cells"],
                    "error": row["error"],
                    "dT1": row["dT1"],
                    "rest": row["rest"],
                    "E_medium": e_med,
                    "E_product": e_prod,
                }
            )
            print(
                f"  uniform n {row['n']:3d}: cells {row['cells']:6d}  error {row['error']:.3e}  E_medium {e_med:.3e}  "
                f"E_product {e_prod:.3e}",
                flush=True,
            )

    if not dry:
        exact = exact_field("s5", core, omega, 1.0, pts)
        exact_born = (
            exact_field("s5", core, omega, EPS, pts) - exact_field("s5", core, omega, -EPS, pts)
        ) / (2 * EPS)
        peak = float(np.abs(exact).max())
        result["nonlinear_fraction"] = float(np.abs(exact - exact_born).max() / peak)
    cache: dict = {}
    seen: set = set()
    for rule in ("medium", "product"):
        for tol in tols:
            centres, hs = adapt(rule, tol)
            sizes = {f"{float(h):g}": int((hs == h).sum()) for h in np.unique(hs)}
            key = tuple(sorted(sizes.items()))
            e_med, e_prod = projection_errors(centres, hs)
            row = {
                "rule": rule,
                "tol": tol,
                "cells": len(hs),
                "unknowns": len(hs) * UNKNOWNS_PER_CELL[p],
                "E_medium": e_med,
                "E_product": e_prod,
                "sizes": sizes,
            }
            note = ""
            if (rule, key) in seen:
                note = "the same tree as the previous tolerance, not solved again"
            elif row["unknowns"] > DENSE_LIMIT:
                note = "over the dense limit, not solved"
            elif not dry:
                seen.add((rule, key))
                t0 = time.perf_counter()
                res = solve_graded_octree(
                    omega, REF, CONTRAST, centres, hs, prof, K_HAT, K_HAT, "P", p=p, r=p, block_cache=cache
                )
                err = sum(octree_far_field(res, pts / rf, rf)) - exact
                born = sum(
                    octree_far_field(
                        born_octree(omega, REF, CONTRAST, centres, hs, prof, K_HAT, K_HAT, "P", p=p, r=p),
                        pts / rf,
                        rf,
                    )
                )
                d_t1 = born - exact_born
                row.update(
                    error=float(np.abs(err).max() / peak),
                    dT1=float(np.abs(d_t1).max() / peak),
                    rest=float(np.abs(err - d_t1).max() / peak),
                    seconds=time.perf_counter() - t0,
                )
            result["trees"].append(row)
            shown = "  ".join(
                f"{k} {row[k]:.3e}" for k in ("error", "dT1", "rest", "E_medium", "E_product") if k in row
            )
            print(
                f"  {rule:7s} tol {tol:7.0e}: cells {row['cells']:5d}  {shown}  leaves by half-width {sizes}"
                + (f"  [{row['seconds']:.0f} s]" if "seconds" in row else "")
                + (f"  ({note})" if note else ""),
                flush=True,
            )
            out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"  wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
