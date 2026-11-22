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

``--shape=halo`` (constant cells only, p = 0) and ``--shape=sharp`` run the two further radial bodies of
``measure_octree_first_order_greedy.py``: the strong centre in a weak surround, and the classical sharp
uniform sphere (core_frac 1). On the sharp sphere each cell holds the fraction of its volume inside the
sphere (exact along z, a 64 x 64 midpoint rule across), so the first-order term has no staircase error,
and the relative projection error is the sum over the cells of volume x fraction x (1 - fraction) over
the volume of the sphere. It falls as the first power of the cell and not the second, so this is where
the law says the second term must overtake the first; the uniform grids here go far beyond the dense
solver and show whether the solved error does what the law says. The file name gains ``_<shape>``.

The sharp sphere also runs with degree-one cells (p = 1). The medium each cell holds is then the L2
projection of the indicator of the sphere onto the polynomials of degree one on the cell, from the exact
means of the indicator and of the indicator times each coordinate (exact along z, a 256 x 256 midpoint
rule across), and the relative projection error is what that projection leaves: the sum over the cells
of volume x (phi - phi^2 - 3 |m|^2) over the volume of the sphere, phi the fraction inside and m the
means of indicator times local coordinate. A linear function cannot follow a jump either, so this too
falls only as the first power of the cell.
"""

import json
import sys
import time
from collections.abc import Callable
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
    shape = opts.get("shape", "s5")
    sharp = shape == "sharp"
    if (shape == "halo" and p != 0) or (sharp and p not in (0, 1)):
        raise SystemExit("--shape=halo is for constant cells (p = 0); --shape=sharp for p = 0 or 1")
    ref_shape = "s5" if sharp else shape  # the sharp sphere is the shell body with its core at the surface
    omega = ka * REF.beta / RADIUS
    rf = 5e8 * RADIUS
    pts = obs_points(rf, THETA)
    contrast = MaterialContrast(CONTRAST.Dlambda * scale, CONTRAST.Dmu * scale, CONTRAST.Drho * scale)

    def prof_vec(pos: np.ndarray) -> np.ndarray:
        x = np.clip((RADIUS - np.linalg.norm(pos, axis=-1)) / (RADIUS - core), 0.0, 1.0)
        return x**3 * (10.0 - 15.0 * x + 6.0 * x**2)

    def prof(pos: np.ndarray) -> float:
        return radial(shape, core, float(np.linalg.norm(pos)))

    mid = (np.arange(64) + 0.5) / 32.0 - 1.0
    gauss_x, gauss_w = np.polynomial.legendre.leggauss(6)
    gauss_xi = np.stack(np.meshgrid(gauss_x, gauss_x, gauss_x, indexing="ij"), -1).reshape(-1, 3)
    gauss_ww = np.einsum("i,j,k->ijk", gauss_w, gauss_w, gauss_w).ravel() / 8.0

    def fraction(c: np.ndarray, h: float) -> float:
        """The fraction of the cell's volume inside the sphere: exact along z, midpoint rule in x and y."""
        x, y = np.meshgrid(c[0] + h * mid, c[1] + h * mid, indexing="ij")
        half = np.sqrt(np.clip(RADIUS**2 - x**2 - y**2, 0.0, None))
        length = np.clip(np.minimum(c[2] + h, half) - np.maximum(c[2] - h, -half), 0.0, None)
        return float(length.mean() / (2.0 * h))

    fine = (np.arange(256) + 0.5) / 128.0 - 1.0

    def linear_moments(c: np.ndarray, h: float) -> tuple[float, np.ndarray]:
        """The means over the cell of the indicator of the sphere and of the indicator times each local
        coordinate xi_i = (x_i - c_i) / h: exact along z, a 256 x 256 midpoint rule in x and y. A cell
        wholly inside or wholly outside is recognised by its farthest and nearest points."""
        if np.linalg.norm(np.abs(c) + h) <= RADIUS:
            return 1.0, np.zeros(3)
        if np.linalg.norm(np.clip(np.abs(c) - h, 0.0, None)) >= RADIUS:
            return 0.0, np.zeros(3)
        x, y = np.meshgrid(fine, fine, indexing="ij")  # local coordinates
        half = np.sqrt(np.clip(RADIUS**2 - (c[0] + h * x) ** 2 - (c[1] + h * y) ** 2, 0.0, None))
        top = np.clip((np.minimum(c[2] + h, half) - c[2]) / h, -1.0, 1.0)
        bottom = np.clip((np.maximum(c[2] - h, -half) - c[2]) / h, -1.0, 1.0)
        length = np.clip(top - bottom, 0.0, None) / 2.0  # the fraction of each column inside
        first_z = np.where(top > bottom, (top**2 - bottom**2) / 4.0, 0.0)
        return float(length.mean()), np.array(
            [float((x * length).mean()), float((y * length).mean()), float(first_z.mean())]
        )

    def sharp_profile(h: float) -> tuple[Callable[[np.ndarray], float], dict]:
        """The sharp sphere as the cells of half-width h hold it, and the moments it was built from.

        For constant cells the value at any point is the fraction of its cell inside the sphere. For
        degree-one cells it is the L2 projection of the indicator onto the polynomials of degree one on
        its cell, phi + 3 (m . xi) with phi the fraction and m the means of indicator times xi, so that
        the solver's Gauss rule, which is exact for a polynomial, projects the true discontinuous medium
        and not a sampled one.
        """
        known: dict = {}

        def value(pos: np.ndarray) -> float:
            pos = np.asarray(pos, dtype=float)
            idx = tuple(np.floor((pos + RADIUS) / (2.0 * h)).astype(int))
            c = -RADIUS + (np.array(idx) + 0.5) * 2.0 * h
            if idx not in known:
                known[idx] = (fraction(c, h), np.zeros(3)) if p == 0 else linear_moments(c, h)
            phi, m = known[idx]
            return phi if p == 0 else float(phi + 3.0 * m @ ((pos - c) / h))

        return value, known

    def other_energies(centres: np.ndarray, h: float, profile: Callable) -> tuple[np.ndarray, float]:
        """For the halo and the sharp sphere: each cell's integral of (f - its projection)^2, and that of
        f^2. On the sharp sphere the projection is to the degree of the cells (0 or 1)."""
        if sharp:
            for c in centres:
                profile(c)
            got = [known_n[tuple(np.floor((c + RADIUS) / (2.0 * h)).astype(int))] for c in centres]
            phi = np.array([g[0] for g in got])
            slope = np.array([3.0 * g[1] @ g[1] for g in got])
            lost = (2.0 * h) ** 3 * np.clip(phi * (1.0 - phi) - slope, 0.0, None)
            return lost, 4.0 * np.pi * RADIUS**3 / 3.0
        lost = np.empty(len(centres))
        for i, c in enumerate(centres):
            f = np.array([profile(c + h * xi) for xi in gauss_xi])
            lost[i] = (2.0 * h) ** 3 * max(float(gauss_ww @ f**2 - (gauss_ww @ f) ** 2), 0.0)
        rr = (np.arange(200000) + 0.5) * RADIUS / 200000
        ff = np.array([radial(shape, core, x) for x in rr])
        return lost, float((4.0 * np.pi * rr**2 * ff**2).sum() * RADIUS / 200000)

    exact = exact_field(ref_shape, core, omega, scale, pts)
    exact_born = (
        scale
        * (exact_field(ref_shape, core, omega, EPS, pts) - exact_field(ref_shape, core, omega, -EPS, pts))
        / (2 * EPS)
    )
    peak = float(np.abs(exact).max())
    nonlinear = float(np.abs(exact - exact_born).max() / peak)
    print(
        f"radial body {shape}: k_S a = {ka}, core = {core_frac} a, degree p = r = {p}, contrast x {scale};  "
        f"nonlinear fraction |exact - exact Born| / peak = {nonlinear:.3e}",
        flush=True,
    )
    print(
        "   n: cells unknowns | error | dT1 (no solve) | rest | E_p | error / E_p | rest / E_p | "
        "products | kernel s, solve s | against the dense solver"
    )
    dense = dense_reference(core_frac, p) if scale == 1.0 and ka == 0.5 else {}
    suffix = ("" if shape == "s5" else f"_{shape}") + ("" if scale == 1.0 else f"_scale{scale:g}")
    out = ROOT / f"scripts/measure_two_term_fft_ka{ka:g}_core{core_frac:g}_p{p}{suffix}.json"
    rows = []
    for n in uniform_ns:
        centres, hs = uniform_leaves(RADIUS, n)
        prof_n, known_n = sharp_profile(float(hs[0])) if sharp else (prof, {})
        if shape == "s5":
            defect, _, norm = leaf_energies(prof_vec, centres, hs, p)
            keep = norm > 0
            e_p = float(defect.sum() / norm.sum())
        else:
            lost, whole = other_energies(centres, float(hs[0]), prof_n)
            # the cells that reach inside the sphere: a positive fraction, or a nearest point within it
            nearest = np.linalg.norm(np.clip(np.abs(centres) - float(hs[0]), 0.0, None), axis=1)
            keep = nearest < RADIUS
            if sharp:
                keep = np.array(
                    [
                        known_n[tuple(np.floor((c + RADIUS) / (2.0 * hs[0])).astype(int))][0] > 0.0
                        for c in centres
                    ]
                )
            e_p = float(lost.sum() / whole)
        born = sum(
            octree_far_field(
                born_octree(
                    omega, REF, contrast, centres[keep], hs[keep], prof_n, K_HAT, K_HAT, "P", p=p, r=p
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
            prof_n,
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
                    "shape": shape,
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
