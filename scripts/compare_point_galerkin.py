#!/usr/bin/env python3
"""The point formulation against the Galerkin voxel: error against wall time, on two spheres.

Arms, each by its fastest route and for the one incident wave the scattered field needs:
  point     the uniform-field collocation voxel (9 unknowns a cell, ``solve_point_plane_wave``): the cube's
            closed-form T-matrix, the receiver-averaged propagator, the kernel at the canonical k only;
            cells whose centre lies inside the sphere, each at the contrast at its centre.
  galerkin  the graded first-moment Galerkin voxel, p = r = 1 (36 unknowns a cell,
            ``graded_voxel.fft.solve_graded_sphere_fft``), every cell overlapping the sphere carrying the L2
            projection of the contrast; the kernel at the canonical k only.
  hierarchy the gradient hierarchy at q = 3 (60 unknowns a cell, fourth order like the Galerkin voxel): the
            field a Taylor polynomial about each centre, the equation and its derivatives imposed there, the
            contrast projected to degree 1 in each voxel, the FFT product of
            ``measure_graded_sphere_gradient_hierarchy_fft.solve_fft``. The graded sphere only. It is the
            point formulation at the Galerkin voxel's order: the check that separates the formulation from
            the order.
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
      python -u scripts/compare_point_galerkin.py --body=graded --ka=0.5 --arm=hierarchy 4 6 8 10 12
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

from pilot_graded_sphere_vs_exact import profile as pilot_profile  # noqa: E402

from cubic_scattering.graded_voxel.blocks import _NEAR_STORE  # noqa: E402
from cubic_scattering.graded_voxel.site import BatchProfile  # noqa: E402

OUT = Path(__file__).parent / "data" / "point_vs_galerkin.jsonl"
RADIUS = 10.0
CORE = RADIUS / 2
K_HAT = np.array([1.0, 0.0, 0.0])
POL = np.array([1.0, 0.0, 0.0])
THETA = np.linspace(0.2, np.pi - 0.2, 9)
R_MULT = 5.0e8  # the asymptotic far fields against the exact displacement: near-field residue ~1e-8


def _sharp_values(pts: np.ndarray) -> np.ndarray:
    """The homogeneous sphere at points (M, 3): 1 inside, 0 outside."""
    return (np.linalg.norm(pts, axis=1) < RADIUS).astype(float)


#: the two bodies' profiles, each evaluated at many points in one call (``graded_voxel.site.BatchProfile``)
graded_profile = pilot_profile
sharp_profile = BatchProfile(_sharp_values)


ARMS = ("point", "galerkin", "hierarchy")


def one(body: str, ka_s: float, arm: str, n: int, tol: float) -> dict:
    if arm not in ARMS:
        raise ValueError(
            f"arm {arm!r} is not one of {ARMS}: give --arm=point, --arm=galerkin or --arm=hierarchy"
        )
    if arm == "hierarchy" and body != "graded":
        raise ValueError(
            f"the hierarchy arm is defined for the graded sphere only (asked: {body!r}): it projects the contrast "
            "on polynomials in each voxel, which a step across the sharp sphere's surface does not have; "
            "run it with --body=graded"
        )
    import torch
    from crosscheck_graded_sphere import graded_mie_result
    from gate_sphere_cell_average_vs_mie import CONTRAST, REF, obs_points

    from cubic_scattering.graded_voxel.farfield import graded_far_field
    from cubic_scattering.graded_voxel.fft import solve_graded_sphere_fft
    from cubic_scattering.sphere_scattering import (
        compute_elastic_mie,
        mie_scattered_displacement,
    )
    from cubic_scattering.sphere_scattering_fft_gpu import (
        point_wave_far_field,
        solve_point_plane_wave,
    )

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
    if arm == "hierarchy":
        import measure_graded_sphere_gradient_hierarchy as gs
        import measure_graded_sphere_gradient_hierarchy_fft as hfft
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
    elif arm == "hierarchy":
        # the gradient hierarchy at q = 3 (fourth order), contrast to degree 1 in each voxel, FFT product;
        # the same body, medium and observation set (tests/test_compare_point_galerkin.py)
        # (its modules are imported before the clock starts, as the other arms' are). Setup is everything
        # solve_fft does but GMRES: its own tables_s counts the block tables only, not the operator and the
        # right-hand side, so the wall time would otherwise exceed setup + GMRES + readout
        ts = time.perf_counter()
        side, centres, coefs, asm, sol, info = hfft.solve_fft(n, 3, omega, CONTRAST, tol=tol)
        timings = {
            "kernel_s": time.perf_counter() - ts - info["solve_s"],
            "solve_s": info["solve_s"],
            "iterations": info["iterations"],
        }
        t1 = time.perf_counter()
        u_p = gs.far_field(side, centres, coefs, asm, sol, omega, CONTRAST, pts)
        u_s = np.zeros_like(u_p)  # the hierarchy's readout is the whole displacement
        cells, unknowns = len(centres), 3 * asm.nu * len(centres)
    else:
        profile = graded_profile if body == "graded" else sharp_profile
        gres = solve_graded_sphere_fft(
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
        u_p, u_s = graded_far_field(gres, pts / r_far, r_far, K_HAT, POL, "P")
        cells, unknowns = len(gres.centres), 36 * len(gres.centres)
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
        # what makes timings comparable: the threads used, and whether the one on-disk store (the Galerkin
        # near-block moments) was there; nothing else the three arms read is kept between runs
        "threads": torch.get_num_threads(),
        "near_store": _NEAR_STORE.exists(),
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
        c_body, c_ka, c_arm, c_n, c_tol = argv[1:]
        print(
            json.dumps(one(c_body, float(c_ka), c_arm, int(c_n), float(c_tol))),
            flush=True,
        )
        return 0
    if argv == ["--report"]:
        report()
        return 0
    body: str | None = None
    ka_s: float | None = None
    arm: str | None = None
    tol, max_hours = 1e-10, 3.0
    ladder: list[int] = []
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
    if body not in ("graded", "sharp") or arm not in ARMS or ka_s is None or not ladder:
        print(__doc__)
        print("give --body=graded|sharp, --ka=<k_S a>, --arm=point|galerkin|hierarchy and the grid sizes")
        return 2
    if arm == "hierarchy" and body != "graded":
        print("the hierarchy arm is defined for the graded sphere only: run it with --body=graded")
        return 2
    done = {(r["body"], r["ka_s"], r["arm"], r["n"], r["tol"]) for r in load()}
    print(f"{body} sphere, k_S a = {ka_s}, {arm}, GMRES to {tol:g}", flush=True)
    for n in ladder:
        if (body, ka_s, arm, n, tol) in done:
            print(f"  n {n:3d}: already recorded", flush=True)
            continue
        cmd = [
            sys.executable,
            "-u",
            __file__,
            "--child",
            body,
            str(ka_s),
            arm,
            str(n),
            str(tol),
        ]
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
