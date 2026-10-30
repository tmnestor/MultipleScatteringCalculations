#!/usr/bin/env python3
"""Equal degree on the graded sphere: the collocation cell against the mean-only Galerkin voxel, like for like.

Both cells hold a uniform field, 9 unknowns a cell, and converge at second order. The Error paper's sphere
section compared them with the package's cube T-matrix as the collocation cell's single site; this measures them
with the uniform-strain closure itself (``point_cell_self``), and separates the two things the collocation cell
of that section changed at once: the single site, and the representation of the contrast.

Arms, each on the n_sub grid of the graded sphere (``pilot_graded_sphere_vs_exact``), at k_S a = 0.5 and 1:
  exact_centre    collocation, exact closure; cells whose centre is inside the sphere, contrast at the centre
                  (the headline arm of compare_point_galerkin.py).
  exact_mean      collocation, exact closure; every cell that overlaps the sphere, the cell mean of the contrast
                  by the six-point Gauss rule per axis of the Galerkin projection: the mean-only voxel's cells and
                  contrast, so only the formulation differs.
  package_mean    as exact_mean with the package T-matrix: reproduces the sphere section's cell-mean collocation
                  (3.4e-3, 1.9e-3 and 8.2e-4 at n_sub = 6, 8 and 12, k_S a = 0.5), the check on the construction.
  galerkin_mean   the mean-only Galerkin voxel, p = r = 0 (graded_voxel.fft).
The error is that of the sphere section: max |far field - exact| / peak over nine angles at 5e8 radii.

Run:  python -u scripts/measure_equal_degree_sphere.py --ka=0.5 6 8 12 16 24 32
Rows are appended to scripts/data/equal_degree_sphere.jsonl.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crosscheck_graded_sphere import graded_mie_result  # noqa: E402
from gate_sphere_cell_average_vs_mie import CONTRAST, REF, obs_points  # noqa: E402
from pilot_graded_sphere_vs_exact import CORE, RADIUS, THETA, _profile_values  # noqa: E402
from pilot_graded_sphere_vs_exact import profile as point_profile  # noqa: E402

from cubic_scattering.graded_voxel.farfield import graded_far_field  # noqa: E402
from cubic_scattering.graded_voxel.fft import solve_graded_sphere_fft  # noqa: E402
from cubic_scattering.graded_voxel.site import BatchProfile  # noqa: E402
from cubic_scattering.sphere_scattering import mie_scattered_displacement  # noqa: E402
from cubic_scattering.sphere_scattering_fft_gpu import (  # noqa: E402
    point_wave_far_field,
    solve_point_plane_wave,
)

OUT = Path(__file__).parent / "data" / "equal_degree_sphere.jsonl"
ARMS = ("exact_centre", "exact_mean", "package_mean", "galerkin_mean")
K_HAT = np.array([1.0, 0.0, 0.0])
POL = np.array([1.0, 0.0, 0.0])
R_MULT = 5.0e8
N_GAUSS = 6  # points per axis of the Galerkin projection, so both cells see the same cell means


def overlaps(h: float):
    """Whether the cell of half-width h centred at c overlaps the sphere: its nearest point lies inside."""

    def inside(c: np.ndarray) -> bool:
        return float(np.linalg.norm(np.maximum(np.abs(c) - h, 0.0))) < RADIUS

    return inside


def cell_mean_profile(h: float) -> BatchProfile:
    """The cell mean of the contrast factor over the cell of half-width h at each centre, by Gauss quadrature."""
    x, w = np.polynomial.legendre.leggauss(N_GAUSS)
    offs = h * np.stack(np.meshgrid(x, x, x, indexing="ij"), -1).reshape(-1, 3)
    wts = (np.einsum("i,j,k->ijk", w, w, w) / 8.0).ravel()

    def means(centres: np.ndarray) -> np.ndarray:
        pts = centres[:, None, :] + offs[None, :, :]
        vals = _profile_values(pts.reshape(-1, 3)).reshape(len(centres), -1)
        return vals @ wts

    return BatchProfile(means)


def one(ka_s: float, arm: str, n: int, exact: np.ndarray, pts: np.ndarray, r_far: float) -> dict:
    omega = ka_s * REF.beta / RADIUS
    h = RADIUS / n
    t0 = time.perf_counter()
    if arm == "galerkin_mean":
        res = solve_graded_sphere_fft(
            omega, RADIUS, REF, CONTRAST, n, point_profile, K_HAT, POL, "P", p=0, r=0
        )
        u_p, u_s = graded_far_field(res, pts / r_far, r_far, K_HAT, POL, "P")
        cells = len(res.centres)
    else:
        mean = arm != "exact_centre"
        res = solve_point_plane_wave(
            omega,
            RADIUS,
            REF,
            CONTRAST,
            n,
            K_HAT,
            POL,
            "P",
            profile=cell_mean_profile(h) if mean else point_profile,
            single_site="package" if arm == "package_mean" else "exact",
            gmres_tol=1e-10,
            device=torch.device("cpu"),
            dtype=torch.complex128,
            inside=overlaps(h) if mean else None,
        )
        u_p, u_s = point_wave_far_field(res, pts / r_far, r_far)
        cells = len(res.centres)
    peak = float(np.max(np.abs(exact)))
    return {
        "ka_s": ka_s,
        "arm": arm,
        "n": n,
        "cells": cells,
        "error": float(np.max(np.abs(u_p + u_s - exact))) / peak,
        "wall_s": time.perf_counter() - t0,
    }


def main() -> int:
    ka_s, ladder, arms = None, [], list(ARMS)
    for a in sys.argv[1:]:
        if a.startswith("--ka="):
            ka_s = float(a.split("=", 1)[1])
        elif a.startswith("--arms="):
            arms = a.split("=", 1)[1].split(",")
        else:
            ladder.append(int(a))
    if ka_s is None or not ladder or not set(arms) <= set(ARMS):
        print(__doc__)
        return 2
    omega = ka_s * REF.beta / RADIUS
    r_far = R_MULT * RADIUS
    pts = obs_points(r_far, THETA)
    n_max = max(8, int(np.ceil(ka_s + 4 * ka_s ** (1 / 3) + 6)))
    exact = mie_scattered_displacement(graded_mie_result(omega, RADIUS, CORE, REF, CONTRAST, n_max), pts)
    print(f"graded sphere, k_S a = {ka_s}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        for n in ladder:
            row = one(ka_s, arm, n, exact, pts, r_far)
            with OUT.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(
                f"  {arm:14s} n {n:3d}: {row['cells']:7d} cells, error {row['error']:.3e}, {row['wall_s']:.1f} s"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
