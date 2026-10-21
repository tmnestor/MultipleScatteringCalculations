#!/usr/bin/env python3
"""The point formulation against the Galerkin voxel: error against wall time, on two spheres.

Arms, each by its fastest route and for the one incident wave the scattered field needs:
  point     the uniform-field collocation voxel (9 unknowns a cell, ``solve_point_plane_wave``): the cube's
            closed-form T-matrix, the receiver-averaged propagator, the kernel at the canonical k only;
            cells whose centre lies inside the sphere, each at the contrast at its centre.
  galerkin  the graded first-moment Galerkin voxel, p = r = 1 (36 unknowns a cell,
            ``graded_voxel.fft.solve_graded_sphere_fft``), every cell overlapping the sphere carrying the L2
            projection of the contrast; the kernel at the canonical k only.
Bodies (radius a = 10 m, the validated contrast of ``gate_sphere_cell_average_vs_mie``):
  graded    contrast 1 in the core r < a/2, falling to zero across the shell as the C2 smoothstep, so no
            staircase: the exact field is ``crosscheck_graded_sphere.graded_mie_result``.
  sharp     the homogeneous sphere, contrast 1 inside and 0 outside: exact elastic Mie. Its error carries
            the staircase of the voxelised surface.
A P wave travels along +x. The error is max |far field - exact| / peak over nine angles at 5e8 radii,
the measure of the pilots (``pilot_graded_voxel_sphere``). The timings are those the solvers report:
setup (cell T-matrices or contrast projection, blocks, kernel and its canonical blocks), GMRES, readout,
and the wall time of the whole, with the run's peak memory; each run is a child process of its own, both
arms in complex128 on the CPU with GMRES to --tol (1e-10 by default). Rows are appended to
scripts/data/point_vs_galerkin.jsonl; a run already there (same body, k_S a, arm, n and tol) is skipped.

Run:  python -u scripts/compare_point_galerkin.py --body=graded --ka=0.5 --arm=point 8 12 16 24 32 40 48
      python -u scripts/compare_point_galerkin.py --body=graded --ka=0.5 --arm=galerkin 8 12 16 24 32
      python -u scripts/compare_point_galerkin.py --report     (the table of every row)
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

OUT = Path(__file__).parent / "data" / "point_vs_galerkin.jsonl"
RADIUS = 10.0
CORE = RADIUS / 2
K_HAT = np.array([1.0, 0.0, 0.0])
POL = np.array([1.0, 0.0, 0.0])
THETA = np.linspace(0.2, np.pi - 0.2, 9)
R_MULT = 5.0e8  # the asymptotic far fields against the exact displacement: near-field residue ~1e-8


def graded_profile(pos: np.ndarray) -> float:
    """1 in the core, the C2 smoothstep across the shell, 0 outside."""
    from crosscheck_graded_sphere import smoothstep

    r = float(np.linalg.norm(pos))
    if r <= CORE:
        return 1.0
    if r >= RADIUS:
        return 0.0
    return smoothstep((RADIUS - r) / (RADIUS - CORE))


def sharp_profile(pos: np.ndarray) -> float:
    """The homogeneous sphere: 1 inside, 0 outside."""
    return 1.0 if float(np.linalg.norm(pos)) < RADIUS else 0.0


def one(body: str, ka_s: float, arm: str, n: int, tol: float) -> dict:
    import torch
    from crosscheck_graded_sphere import graded_mie_result
    from cubic_scattering.graded_voxel.farfield import graded_far_field
    from cubic_scattering.graded_voxel.fft import solve_graded_sphere_fft
    from cubic_scattering.sphere_scattering import compute_elastic_mie, mie_scattered_displacement
    from cubic_scattering.sphere_scattering_fft_gpu import point_wave_far_field, solve_point_plane_wave
    from gate_sphere_cell_average_vs_mie import CONTRAST, REF, obs_points

    omega = ka_s * REF.beta / RADIUS
    r_far = R_MULT * RADIUS
    pts = obs_points(r_far, THETA)
    if body == "graded":
        n_max = max(8, int(np.ceil(ka_s + 4 * ka_s ** (1 / 3) + 6)))
        exact = mie_scattered_displacement(
            graded_mie_result(omega, RADIUS, CORE, REF, CONTRAST, n_max), pts
        )
    else:
        exact = mie_scattered_displacement(compute_elastic_mie(omega, RADIUS, REF, CONTRAST), pts)
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    if arm == "point":
        res = solve_point_plane_wave(
            omega,
            RADIUS,
            REF,
            CONTRAST,
            n,
            K_HAT,
            POL,
            "P",
            profile=graded_profile if body == "graded" else None,
            gmres_tol=tol,
            device=torch.device("cpu"),
            dtype=torch.complex128,
            timings=timings,
        )
        t1 = time.perf_counter()
        u_p, u_s = point_wave_far_field(res, pts / r_far, r_far)
        cells, unknowns = len(res.centres), 9 * len(res.centres)
    else:
        profile = graded_profile if body == "graded" else sharp_profile
        res = solve_graded_sphere_fft(
            omega,
            RADIUS,
            REF,
            CONTRAST,
            n,
            profile,
            K_HAT,
            POL,
            "P",
            p=1,
            r=1,
            gmres_tol=tol,
            timings=timings,
        )
        t1 = time.perf_counter()
        u_p, u_s = graded_far_field(res, pts / r_far, r_far, K_HAT, POL, "P")
        cells, unknowns = len(res.centres), 36 * len(res.centres)
    t2 = time.perf_counter()
    peak = float(np.max(np.abs(exact)))
    return {
        "body": body,
        "ka_s": ka_s,
        "arm": arm,
        "n": n,
        "tol": tol,
        "cells": cells,
        "unknowns": unknowns,
        "error": float(np.max(np.abs(u_p + u_s - exact))) / peak,
        "setup_s": timings["kernel_s"],
        "gmres_s": timings["solve_s"],
        "products": timings.get("products", timings.get("iterations")),
        "readout_s": t2 - t1,
        "wall_s": t2 - t0,
        "peak_GB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9,  # bytes on macOS
    }


def load() -> list[dict]:
    if not OUT.exists():
        return []
    return [json.loads(line) for line in OUT.read_text().splitlines() if line.strip()]


def report() -> None:
    rows = sorted(load(), key=lambda r: (r["body"], r["ka_s"], r["arm"], r["n"]))
    key = None
    for r in rows:
        if (r["body"], r["ka_s"]) != key:
            key = (r["body"], r["ka_s"])
            print(f"\n{r['body']} sphere, k_S a = {r['ka_s']}")
            print("  arm        n   unknowns     error      setup    GMRES  readout     wall   peak")
        print(
            f"  {r['arm']:9s} {r['n']:3d} {r['unknowns']:10d}  {r['error']:.3e}  {r['setup_s']:7.1f}"
            f"  {r['gmres_s']:7.1f}  {r['readout_s']:7.2f}  {r['wall_s']:7.1f}  {r['peak_GB']:5.1f}"
        )


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "--child":
        body, ka, arm, n, tol = argv[1:]
        print(json.dumps(one(body, float(ka), arm, int(n), float(tol))), flush=True)
        return 0
    if argv == ["--report"]:
        report()
        return 0
    body, ka_s, arm, tol, max_hours, ladder = None, None, None, 1e-10, 3.0, []
    for a in argv:
        if a.startswith("--body="):
            body = a.split("=", 1)[1]
        elif a.startswith("--ka="):
            ka_s = float(a.split("=", 1)[1])
        elif a.startswith("--arm="):
            arm = a.split("=", 1)[1]
        elif a.startswith("--tol="):
            tol = float(a.split("=", 1)[1])
        elif a.startswith("--max-hours="):
            max_hours = float(a.split("=", 1)[1])
        else:
            ladder.append(int(a))
    if body not in ("graded", "sharp") or arm not in ("point", "galerkin") or ka_s is None or not ladder:
        print(__doc__)
        print("give --body=graded|sharp, --ka=<k_S a>, --arm=point|galerkin and the grid sizes")
        return 2
    done = {(r["body"], r["ka_s"], r["arm"], r["n"], r["tol"]) for r in load()}
    print(f"{body} sphere, k_S a = {ka_s}, {arm}, GMRES to {tol:g}", flush=True)
    for n in ladder:
        if (body, ka_s, arm, n, tol) in done:
            print(f"  n {n:3d}: already recorded", flush=True)
            continue
        cmd = [sys.executable, "-u", __file__, "--child", body, str(ka_s), arm, str(n), str(tol)]
        try:
            got = subprocess.run(cmd, capture_output=True, text=True, timeout=max_hours * 3600)
        except subprocess.TimeoutExpired:
            print(f"  n {n:3d}: over {max_hours} h; ladder stopped", flush=True)
            break
        if got.returncode != 0:
            print(f"  n {n:3d}: failed\n{got.stderr[-3000:]}", flush=True)
            return 1
        row = json.loads(got.stdout.strip().splitlines()[-1])
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with OUT.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(
            f"  n {n:3d}: {row['unknowns']:9d} unknowns, error {row['error']:.3e},"
            f" setup {row['setup_s']:.1f} s,"
            f" GMRES {row['gmres_s']:.1f} s ({row['products']:.0f}), readout {row['readout_s']:.2f} s,"
            f" wall {row['wall_s']:.1f} s, peak {row['peak_GB']:.1f} GB",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
