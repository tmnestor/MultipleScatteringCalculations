#!/usr/bin/env python3
"""Measure: how large a graded-sphere voxel model runs on this machine, and how accurate it is.

For the thesis's Chapter 7 (plane-wave scattering by the graded sphere, voxels against the exact solution,
a record section above the sphere). Each resolution n_sub (cells across the diameter) runs in a process of
its own, so that its peak memory is its own: the FFT solve (``graded_voxel.fft.solve_graded_sphere_fft``,
GMRES to 1e-11) of a P wave travelling down (+z, axis 0) onto the graded sphere at k_S a = --ka, degree
p = r = --p; then the field read out (``graded_field``, 6 Gauss nodes a side) on a line of receivers on the
plane z = -GAP a above the sphere, x = 0 ... X_MAX a, against the exact scattered displacement there
(``crosscheck_graded_sphere``). Recorded: cells, unknowns, the solve's and the readout's seconds, the peak
memory, and the relative L2 error over the line. The ladder stops at the first size over --max-gb or
--max-hours.

The body, medium and contrast are those of ``measure_graded_sphere_planes.py``.

Writes scripts/data/graded_sphere_size.jsonl.

Run:  python -u scripts/measure_graded_sphere_size.py --ka=4 --p=1 8 12 16 24 32 40 48
"""

import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT = Path(__file__).parent / "data" / "graded_sphere_size.jsonl"
GAP = 1.25  # the receivers' plane, z = -GAP a
X_MAX = 2.5  # the line's extent, in radii
N_REC = 11
UNKNOWNS = {0: 9, 1: 36, 2: 90}


def one(n_sub: int, ka_s: float, p: int) -> dict:
    """One resolution, in this process."""
    from crosscheck_graded_sphere import graded_mie_result
    from cubic_scattering.graded_voxel.farfield import graded_field
    from cubic_scattering.graded_voxel.fft import solve_graded_sphere_fft
    from cubic_scattering.sphere_scattering import mie_scattered_displacement
    from gate_sphere_cell_average_vs_mie import CONTRAST, REF
    from pilot_graded_sphere_vs_exact import CORE, RADIUS, profile

    omega = ka_s * REF.beta / RADIUS
    rec = np.array([[-GAP * RADIUS, x * RADIUS, 0.0] for x in np.linspace(0.0, X_MAX, N_REC)])
    n_max = max(8, int(np.ceil(ka_s + 4 * ka_s ** (1 / 3) + 6)))
    exact = mie_scattered_displacement(graded_mie_result(omega, RADIUS, CORE, REF, CONTRAST, n_max), rec)
    t0 = time.perf_counter()
    res = solve_graded_sphere_fft(
        omega,
        RADIUS,
        REF,
        CONTRAST,
        n_sub,
        profile,
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        "P",
        p=p,
        r=p,
    )
    t_solve = time.perf_counter() - t0
    t0 = time.perf_counter()
    u, _ = graded_field(res, rec, n_gauss=6)
    t_read = time.perf_counter() - t0
    return {
        "n_sub": n_sub,
        "ka_s": ka_s,
        "p": p,
        "cells": len(res.centres),
        "unknowns": len(res.centres) * UNKNOWNS[p],
        "solve_s": t_solve,
        "readout_s_per_receiver": t_read / len(rec),
        "peak_GB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9,
        "error_line": float(np.linalg.norm(u - exact) / np.linalg.norm(exact)),
    }


def main() -> int:
    if len(sys.argv) == 5 and sys.argv[1] == "--child":
        print(json.dumps(one(int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]))), flush=True)
        return 0
    ka_s, p, max_gb, max_hours, ladder = 4.0, 1, 12.0, 3.0, []
    for a in sys.argv[1:]:
        if a.startswith("--ka="):
            ka_s = float(a.split("=", 1)[1])
        elif a.startswith("--p="):
            p = int(a.split("=", 1)[1])
        elif a.startswith("--max-gb="):
            max_gb = float(a.split("=", 1)[1])
        elif a.startswith("--max-hours="):
            max_hours = float(a.split("=", 1)[1])
        else:
            ladder.append(int(a))
    print(f"graded sphere size ladder: k_S a = {ka_s}, p = r = {p}", flush=True)
    for n in ladder:
        try:
            got = subprocess.run(
                [sys.executable, "-u", __file__, "--child", str(n), str(ka_s), str(p)],
                capture_output=True,
                text=True,
                timeout=max_hours * 3600,
            )
        except subprocess.TimeoutExpired:
            print(f"  n_sub {n}: over {max_hours} h; ladder stopped", flush=True)
            break
        if got.returncode != 0:
            print(f"  n_sub {n}: failed\n{got.stderr[-2000:]}", flush=True)
            break
        row = json.loads(got.stdout.strip().splitlines()[-1])
        with OUT.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(
            f"  n_sub {n:3d}: {row['cells']:7d} cells, {row['unknowns']:8d} unknowns,"
            f" solve {row['solve_s']:8.1f} s,"
            f" readout {row['readout_s_per_receiver']:.2f} s/receiver, peak {row['peak_GB']:.1f} GB,"
            f" error on the line {row['error_line']:.2e}",
            flush=True,
        )
        if row["peak_GB"] > max_gb:
            print(f"  over {max_gb} GB; ladder stopped", flush=True)
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
