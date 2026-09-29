#!/usr/bin/env python3
"""Pilot: a voxelised sphere whose contrast falls smoothly to zero, against its exact scattering.

The sharp sphere of the continuum-limit paper (section 10) is dominated by the staircase error of its
shape, so its raw error does not converge under refinement. This body has no staircase: a homogeneous core
r < a/2 carries the validated contrast, and across the shell a/2 < r < a the contrast falls to zero as the
C2 smoothstep s(x) = 10x^3 - 15x^4 + 6x^5, x = (a - r)/(a - a/2), vanishing like (a - r)^3 at the surface.
Each voxel carries the profile's value at its centre (``contrast_profile`` of the FFT solver); the exact
far field is ``graded_mie_result`` of ``scripts/crosscheck_graded_sphere.py`` (validated against the
90-digit notebook ``Mathematica/ContinuumLimit_GradedSphere.wl``). The comparison is that of the
sharp-sphere pilot: max |far field - exact| / peak over nine angles at 5e4 radii
(``gate_sphere_cell_average_vs_mie``).
The prediction: second order in the voxel size, with no plateau.

With --summary=<file> the errors are written as JSON (the data of the paper's convergence figure).

Run small first:
    conda run -n seismic python -u scripts/pilot_graded_sphere_vs_exact.py [--ka=0.5] [--summary=f] 4 6 8
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crosscheck_graded_sphere import graded_mie_result, smoothstep  # noqa: E402
from cubic_scattering.sphere_scattering import foldy_lax_far_field, mie_scattered_displacement  # noqa: E402
from cubic_scattering.sphere_scattering_fft import compute_sphere_foldy_lax_fft  # noqa: E402
from gate_sphere_cell_average_vs_mie import CONTRAST, R_MULT, REF, obs_points  # noqa: E402

RADIUS = 10.0
CORE = RADIUS / 2
K_HAT = np.array([1.0, 0.0, 0.0])
POL = np.array([1.0, 0.0, 0.0])
THETA = np.linspace(0.2, np.pi - 0.2, 9)


def profile(pos: np.ndarray) -> float:
    """The contrast factor at a point: 1 in the core, the smoothstep across the shell, 0 outside."""
    r = float(np.linalg.norm(pos))
    if r <= CORE:
        return 1.0
    if r >= RADIUS:
        return 0.0
    return smoothstep((RADIUS - r) / (RADIUS - CORE))


def main() -> int:
    ka_s = 0.5
    summary = None
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    for a in sys.argv[1:]:
        if a.startswith("--ka="):
            ka_s = float(a.split("=", 1)[1])
        elif a.startswith("--summary="):
            summary = Path(a.split("=", 1)[1])
    omega = ka_s * REF.beta / RADIUS
    ladder = [int(a) for a in args] or [4]
    r_far = R_MULT * RADIUS
    pts = obs_points(r_far, THETA)
    n_max = max(8, int(np.ceil(ka_s + 4 * ka_s ** (1 / 3) + 6)))
    exact = mie_scattered_displacement(graded_mie_result(omega, RADIUS, CORE, REF, CONTRAST, n_max), pts)
    peak = float(np.max(np.abs(exact)))
    print(f"graded sphere, k_S a = {ka_s}, radius {RADIUS} m, core {CORE} m, omega = {omega:.2f} rad/s")
    print("  n_sub  cells  k_S h      error vs exact   seconds", flush=True)
    rows = []
    for n in ladder:
        t0 = time.perf_counter()
        fl = compute_sphere_foldy_lax_fft(
            omega, RADIUS, REF, CONTRAST, n_sub=n, k_hat=K_HAT, wave_type="P", contrast_profile=profile
        )
        u_p, u_s = foldy_lax_far_field(fl, pts / r_far, r_far, K_HAT, POL, wave_type="P")
        err = float(np.max(np.abs(u_p + u_s - exact))) / peak
        dt = time.perf_counter() - t0
        rows.append((n, err, fl.n_cells))
        print(
            f"  {n:5d}  {fl.n_cells:5d}  {omega / REF.beta * fl.a_sub:.4f}   {err:.4e}       {dt:7.1f}",
            flush=True,
        )
    for (n1, e1, _), (n2, e2, _) in zip(rows, rows[1:], strict=False):
        print(f"  apparent order {n1} -> {n2}: {np.log(e1 / e2) / np.log(n2 / n1):.2f}")
    if summary is not None:
        data = {
            "ka_s": ka_s,
            "radius": RADIUS,
            "core": CORE,
            "n_sub": [r[0] for r in rows],
            "cells": [r[2] for r in rows],
            "error_vs_exact": [r[1] for r in rows],
        }
        summary.write_text(json.dumps(data, indent=2) + "\n")
        print(f"  wrote {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
