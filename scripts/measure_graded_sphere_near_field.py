#!/usr/bin/env python3
"""The graded sphere read at a FINITE distance: the voxel scheme's near field against the exact one.

``pilot_graded_voxel_sphere.py`` compares the scheme's asymptotic far field with the exact displacement
and therefore has to observe from 5e8 radii. That is a property of its readout, not of the scheme. Here
the solved cells are radiated through the propagator itself (``graded_voxel.farfield.graded_field``), so
the scattered displacement is evaluated where the observer actually is, and compared with the exact
partial-wave displacement at the same points.

Observers: nine scattering angles (the pilot's) on spheres of radius 1.25 a, 2 a and 10 a.
Error: max |u_scheme - u_exact| over the nine points and three components, over the largest exact
component on that sphere.
Arms: g0 (mean-only voxel, p = r = 0) and g1 (first-moment voxel, p = r = 1), both by FFT + GMRES.
Predicted: the orders of the far field, 2 and 4, at every distance.

The readout integrates the propagator over each cell with 6 Gauss nodes a side. Also reported: the same
solution read with the asymptotic formula at the same distance, to show what the asymptotic readout would
have cost there, and the change from 4 to 6 nodes a side (the size of the readout's own quadrature error).

Run small first:
    conda run -n seismic python -u scripts/measure_graded_sphere_near_field.py --ka=0.5 --arms=g0 4 6
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crosscheck_graded_sphere import graded_mie_result  # noqa: E402
from cubic_scattering.graded_voxel.farfield import graded_far_field, graded_field  # noqa: E402
from cubic_scattering.graded_voxel.fft import solve_graded_sphere_fft  # noqa: E402
from cubic_scattering.sphere_scattering import mie_scattered_displacement  # noqa: E402
from gate_sphere_cell_average_vs_mie import CONTRAST, REF, obs_points  # noqa: E402
from pilot_graded_sphere_vs_exact import CORE, RADIUS, THETA, profile  # noqa: E402

K_HAT = np.array([1.0, 0.0, 0.0])
POL = np.array([1.0, 0.0, 0.0])
DISTANCES = (1.25, 2.0, 10.0)  # in radii
DEGREE = {"g0": 0, "g1": 1}


def main() -> int:
    ka_s, arms, summary, ladder = 0.5, ["g0", "g1"], None, []
    for a in sys.argv[1:]:
        if a.startswith("--ka="):
            ka_s = float(a.split("=", 1)[1])
        elif a.startswith("--arms="):
            arms = a.split("=", 1)[1].split(",")
        elif a.startswith("--summary="):
            summary = Path(a.split("=", 1)[1])
        else:
            ladder.append(int(a))
    omega = ka_s * REF.beta / RADIUS
    n_max = max(8, int(np.ceil(ka_s + 4 * ka_s ** (1 / 3) + 6)))
    mie = graded_mie_result(omega, RADIUS, CORE, REF, CONTRAST, n_max)
    pts = {m: obs_points(m * RADIUS, THETA) for m in DISTANCES}
    exact = {m: mie_scattered_displacement(mie, pts[m]) for m in DISTANCES}
    peak = {m: float(np.max(np.abs(exact[m]))) for m in DISTANCES}
    print(f"graded sphere, k_S a = {ka_s}, observers at {DISTANCES} radii, arms {arms}", flush=True)
    out: dict = {"ka_s": ka_s, "distances_radii": list(DISTANCES), "arms": {}}
    for arm in arms:
        p = DEGREE[arm]
        rows = []
        for n in ladder or [4]:
            t0 = time.perf_counter()
            res = solve_graded_sphere_fft(
                omega, RADIUS, REF, CONTRAST, n, profile, K_HAT, POL, "P", p=p, r=p
            )
            row = {"n_sub": n, "error": [], "asymptotic_readout_error": [], "gauss_4_to_6": []}
            for m in DISTANCES:
                u, _ = graded_field(res, pts[m])
                u6, _ = graded_field(res, pts[m], n_gauss=6)
                up, us = graded_far_field(res, pts[m] / (m * RADIUS), m * RADIUS, K_HAT, POL, "P")
                row["error"].append(float(np.max(np.abs(u6 - exact[m]))) / peak[m])
                row["asymptotic_readout_error"].append(float(np.max(np.abs(up + us - exact[m]))) / peak[m])
                row["gauss_4_to_6"].append(float(np.max(np.abs(u6 - u))) / peak[m])
            rows.append(row)
            print(
                f"  {arm}  n_sub {n:3d}  error "
                + "  ".join(f"{e:.3e}" for e in row["error"])
                + "   asymptotic readout "
                + "  ".join(f"{e:.1e}" for e in row["asymptotic_readout_error"])
                + "   Gauss 4->6 "
                + "  ".join(f"{e:.1e}" for e in row["gauss_4_to_6"])
                + f"   {time.perf_counter() - t0:7.1f} s",
                flush=True,
            )
        for r1, r2 in zip(rows, rows[1:], strict=False):
            orders = [
                np.log(e1 / e2) / np.log(r2["n_sub"] / r1["n_sub"])
                for e1, e2 in zip(r1["error"], r2["error"], strict=True)
            ]
            print(
                f"  {arm}  apparent order {r1['n_sub']} -> {r2['n_sub']}: "
                + "  ".join(f"{o:.2f}" for o in orders)
            )
        out["arms"][arm] = rows
    if summary is not None:
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
