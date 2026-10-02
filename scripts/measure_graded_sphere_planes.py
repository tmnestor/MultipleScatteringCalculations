#!/usr/bin/env python3
"""The graded sphere by voxels, read on planes above and below it, against the exact displacement.

The counterpart of ``measure_graded_sphere_march.py``. The impedance march delivers the reflected field on
the plane z = -a above the sphere and the transmitted field on z = +a below it; here the voxel solution is
read on the same two planes with the finite-distance readout (``graded_voxel.farfield.graded_field``) and
compared with the exact partial-wave displacement.

The planes are at z = -GAP a and z = +GAP a with GAP = 1.25, not at the tangent planes z = -a and +a. On
the tangent planes the observers lie on the faces of the cells at the poles, where the Gauss rule of the
readout cannot integrate the singular propagator (measured: its 4-to-6-node change is 2 to 10 percent
there, larger than the scheme's error). A quarter of a radius away it is below 1e-5 from 8 cells across.
The march's amplitudes are carried across the same homogeneous gap exactly, by each order's own phase.

Observers: a square patch of side PATCH radii on each plane, centred over the sphere, on a grid of
N_SIDE x N_SIDE points. The incident wave travels along the axis through the centre of the patch, so the
exact field is symmetric about that axis and the voxel field has the symmetry of the square; only the
points with 0 < y <= x are evaluated, each weighted by the number of its images (4 on the diagonal, 8 off
it). The norms below are those of the whole patch.
Error, for each plane:  || u_scheme - u_exact ||_2 / || u_exact ||_2  over the patch, all three components:
the scattered displacement, so relative to what the sphere scatters. This is the L2 measure the march's
error over its plane-wave orders corresponds to.
Arms: g0 (p = r = 0), g1 (p = r = 1), g2 (p = r = 2), by FFT + GMRES.
The readout uses 6 Gauss nodes a side; the change from 4 to 6 is reported as its own quadrature error,
its size.

Run small first:
    python -u scripts/measure_graded_sphere_planes.py --ka=0.5 --arms=g0 4 6
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crosscheck_graded_sphere import graded_mie_result  # noqa: E402
from cubic_scattering.graded_voxel.farfield import graded_field  # noqa: E402
from cubic_scattering.graded_voxel.fft import solve_graded_sphere_fft  # noqa: E402
from cubic_scattering.sphere_scattering import mie_scattered_displacement  # noqa: E402
from gate_sphere_cell_average_vs_mie import CONTRAST, REF  # noqa: E402
from pilot_graded_sphere_vs_exact import CORE, RADIUS, profile  # noqa: E402

K_HAT = np.array([1.0, 0.0, 0.0])  # +z: axis 0 is depth
POL = np.array([1.0, 0.0, 0.0])
GAP = 1.25  # the planes are at z = -GAP a and +GAP a
PATCH = 5.0  # side of the patch of observers, in radii: one period of the march's array
N_SIDE = 12
DEGREE = {"g0": 0, "g1": 1, "g2": 2}
UNKNOWNS = {0: 9, 1: 36, 2: 90}


def plane(z: float) -> tuple[np.ndarray, np.ndarray]:
    """Observers on the plane at depth z, and their weights: one wedge of the N_SIDE x N_SIDE grid of
    cell centres over the patch (0 < y <= x), with the number of symmetric images of each point."""
    s = (np.arange(N_SIDE // 2) + 0.5) * (PATCH * RADIUS / N_SIDE)
    pts, wts = [], []
    for i, x in enumerate(s):
        for j, y in enumerate(s[: i + 1]):
            pts.append([z, x, y])
            wts.append(4.0 if i == j else 8.0)
    return np.array(pts), np.array(wts)


def norm(u: np.ndarray, w: np.ndarray) -> float:
    """The L2 norm over the whole patch of a field given on the wedge."""
    return float(np.sqrt(np.sum(w[:, None] * np.abs(u) ** 2)))


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
    planes = {"R": plane(-GAP * RADIUS), "T": plane(GAP * RADIUS)}
    pts = {k: v[0] for k, v in planes.items()}
    wts = {k: v[1] for k, v in planes.items()}
    exact = {k: mie_scattered_displacement(mie, p) for k, p in pts.items()}
    print(
        f"graded sphere by voxels on the planes z = -{GAP} a (R) and z = +{GAP} a (T), k_S a = {ka_s}",
        flush=True,
    )
    out: dict = {"ka_s": ka_s, "patch_radii": PATCH, "n_side": N_SIDE, "arms": {}}
    for arm in arms:
        p = DEGREE[arm]
        rows = []
        for n in ladder or [4]:
            t0 = time.perf_counter()
            res = solve_graded_sphere_fft(
                omega, RADIUS, REF, CONTRAST, n, profile, K_HAT, POL, "P", p=p, r=p
            )
            t_solve = time.perf_counter() - t0
            row = {"n_sub": n, "unknowns": len(res.centres) * UNKNOWNS[p], "solve_seconds": t_solve}
            for k in ("R", "T"):
                u6, _ = graded_field(res, pts[k], n_gauss=6)
                u4, _ = graded_field(res, pts[k], n_gauss=4)
                size = norm(exact[k], wts[k])
                row[f"error_{k}"] = norm(u6 - exact[k], wts[k]) / size
                row[f"gauss_{k}"] = norm(u6 - u4, wts[k]) / size
            rows.append(row)
            print(
                f"  {arm}  n_sub {n:3d}  unknowns {row['unknowns']:7d}   R {row['error_R']:.3e}   "
                f"T {row['error_T']:.3e}   Gauss 4->6 {row['gauss_R']:.1e} {row['gauss_T']:.1e}   "
                f"solve {t_solve:7.1f} s",
                flush=True,
            )
        for r1, r2 in zip(rows, rows[1:], strict=False):
            step = np.log(r2["n_sub"] / r1["n_sub"])
            print(
                f"  {arm}  apparent order {r1['n_sub']} -> {r2['n_sub']}: "
                f"R {np.log(r1['error_R'] / r2['error_R']) / step:.2f}  "
                f"T {np.log(r1['error_T'] / r2['error_T']) / step:.2f}"
            )
        out["arms"][arm] = rows
    if summary is not None:
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
