#!/usr/bin/env python3
"""The two-term error law of a three-dimensional body, measured with the matrix-free FFT solver.

``measure_octree_two_term.py`` measured the law with the dense solver of ``graded_voxel.octree``, which
stops near 20 000 unknowns (1256 constant cells, 408 cells of degree one). This script measures the same
quantities on the same uniform grids with ``graded_voxel.fft.solve_graded_sphere_fft``, a second solver
that shares the coupling blocks but not the assembly or the solve, and carries them to grids the dense
solver cannot reach.

The body is the thin-shell sphere (uniform core, graded shell), whose exact solution is the radial
reference at any contrast. The error of the far field is separated by its order in the contrast:

  first order   dT1 = (the scheme's Born term) - (the exact Born term). The scheme's Born term needs no
                solve (``octree.born_octree``); the exact one is a central difference of the exact
                solution in the contrast.
  the rest      error - dT1, to be compared with the relative projection error E of the medium.

For each grid the script records the error, dT1, the rest, E, the two ratios, the unknowns, the number of
matrix-vector products and the times of the kernel stage and of the solve. Where the dense solver's
figures are stored (``LatexPDFs/PolynomialVoxelError/figures/data_two_term_core<c>_p<p>.json``) the
relative difference from them is printed beside each row.

Results are written to ``scripts/measure_two_term_fft_ka<ka>_core<c>_p<p>.json`` after every grid, so a
run that is stopped keeps what it has measured.

Run small first:
    conda run -n seismic python -u scripts/measure_two_term_fft.py 0.5 0.75 1 --uniform=4,6,8
    ... <k_S a> <core_frac> <p> [--uniform=4,6,8,12,16] [--scale=1.0] [--tol=1e-11]

``--scale`` multiplies the contrast (the exact reference is solved at the same scaled contrast).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gate_sphere_cell_average_vs_mie import CONTRAST, REF, obs_points  # noqa: E402
from measure_graded_voxel_resolution import exact_field, radial  # noqa: E402
from pilot_graded_sphere_vs_exact import RADIUS, THETA  # noqa: E402

from cubic_scattering.effective_contrasts import MaterialContrast  # noqa: E402
from cubic_scattering.graded_voxel.farfield import graded_far_field  # noqa: E402
from cubic_scattering.graded_voxel.fft import solve_graded_sphere_fft  # noqa: E402
from cubic_scattering.graded_voxel.octree import (  # noqa: E402
    born_octree,
    leaf_energies,
    octree_far_field,
    uniform_leaves,
)

ROOT = Path(__file__).resolve().parent.parent
K_HAT = np.array([1.0, 0.0, 0.0])
EPS = 1e-3


def dense_reference(core_frac: float, p: int) -> dict[int, dict]:
    """The dense solver's stored rows on uniform grids, keyed by the number of cells."""
    path = ROOT / f"LatexPDFs/PolynomialVoxelError/figures/data_two_term_core{core_frac:g}_p{p}.json"
    if not path.exists():
        return {}
    rows = json.loads(path.read_text())["rows"]
    return {row["cells"]: row for row in rows if row["grid"].startswith("uniform")}


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    ka, core_frac, p = float(args[0]), float(args[1]), int(args[2])
    core = RADIUS * core_frac
    uniform_ns = [int(v) for v in opts.get("uniform", "4,6,8").split(",") if v]
    scale = float(opts.get("scale", "1.0"))
    tol = float(opts.get("tol", "1e-11"))
    omega = ka * REF.beta / RADIUS
    rf = 5e8 * RADIUS
    pts = obs_points(rf, THETA)
    contrast = MaterialContrast(CONTRAST.Dlambda * scale, CONTRAST.Dmu * scale, CONTRAST.Drho * scale)

    def prof_vec(pos: np.ndarray) -> np.ndarray:
        x = np.clip((RADIUS - np.linalg.norm(pos, axis=-1)) / (RADIUS - core), 0.0, 1.0)
        return x**3 * (10.0 - 15.0 * x + 6.0 * x**2)

    def prof(pos: np.ndarray) -> float:
        return radial("s5", core, float(np.linalg.norm(pos)))

    exact = exact_field("s5", core, omega, scale, pts)
    exact_born = (
        scale
        * (exact_field("s5", core, omega, EPS, pts) - exact_field("s5", core, omega, -EPS, pts))
        / (2 * EPS)
    )
    peak = float(np.abs(exact).max())
    nonlinear = float(np.abs(exact - exact_born).max() / peak)
    print(
        f"thin-shell sphere: k_S a = {ka}, core = {core_frac} a, degree p = r = {p}, contrast x {scale};  "
        f"nonlinear fraction |exact - exact Born| / peak = {nonlinear:.3e}",
        flush=True,
    )
    print(
        "   n: cells unknowns | error | dT1 (no solve) | rest | E_p | error / E_p | rest / E_p | "
        "products | kernel s, solve s | against the dense solver"
    )
    dense = dense_reference(core_frac, p) if scale == 1.0 and ka == 0.5 else {}
    suffix = "" if scale == 1.0 else f"_scale{scale:g}"
    out = ROOT / f"scripts/measure_two_term_fft_ka{ka:g}_core{core_frac:g}_p{p}{suffix}.json"
    rows = []
    for n in uniform_ns:
        centres, hs = uniform_leaves(RADIUS, n)
        defect, _, norm = leaf_energies(prof_vec, centres, hs, p)
        keep = norm > 0
        e_p = float(defect.sum() / norm.sum())
        born = sum(
            octree_far_field(
                born_octree(
                    omega, REF, contrast, centres[keep], hs[keep], prof, K_HAT, K_HAT, "P", p=p, r=p
                ),
                pts / rf,
                rf,
            )
        )
        d_t1 = born - exact_born
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        res = solve_graded_sphere_fft(
            omega,
            RADIUS,
            REF,
            contrast,
            n,
            prof,
            K_HAT,
            K_HAT,
            "P",
            p=p,
            r=p,
            gmres_tol=tol,
            timings=timings,
        )
        err = sum(graded_far_field(res, pts / rf, rf, K_HAT, K_HAT, "P")) - exact
        cells = int(keep.sum())
        row = {
            "n": n,
            "cells": cells,
            "unknowns": int(res.psi.shape[0] * (1 if p == 0 else res.psi.shape[1]) * 9),
            "error": float(np.abs(err).max() / peak),
            "dT1": float(np.abs(d_t1).max() / peak),
            "rest": float(np.abs(err - d_t1).max() / peak),
            "E": e_p,
            "products": int(timings.get("products", 0)),
            "kernel_s": timings.get("kernel_s", float("nan")),
            "solve_s": timings.get("solve_s", float("nan")),
            "seconds": time.perf_counter() - t0,
        }
        note = ""
        if cells in dense:
            d = dense[cells]
            row["dense_error"] = d["error"]
            row["dense_rest"] = d["rest"]
            note = f"error {abs(row['error'] / d['error'] - 1):.1e}, rest {abs(row['rest'] / d['rest'] - 1):.1e}"
        rows.append(row)
        print(
            f"   {n:3d}: {cells:6d} {row['unknowns']:8d} | {row['error']:.3e} | {row['dT1']:.3e} | "
            f"{row['rest']:.3e} | {e_p:.3e} | {row['error'] / e_p:.4f} | {row['rest'] / e_p:.4f} | "
            f"{row['products']:4d} | {row['kernel_s']:.0f}, {row['solve_s']:.0f} | {note}",
            flush=True,
        )
        out.write_text(
            json.dumps(
                {
                    "ka_s": ka,
                    "core_frac": core_frac,
                    "p": p,
                    "contrast_scale": scale,
                    "nonlinear_fraction": nonlinear,
                    "gmres_tol": tol,
                    "rows": rows,
                },
                indent=2,
            )
            + "\n"
        )
    rest = np.array([r["rest"] / r["E"] for r in rows])
    full = np.array([r["error"] / r["E"] for r in rows])
    print(
        f"   spread (max / min): error / E_p {full.max() / full.min():.2f}, rest / E_p "
        f"{rest.max() / rest.min():.2f}"
    )
    print(f"   wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
