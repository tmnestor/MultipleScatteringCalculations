#!/usr/bin/env python3
"""The gradient hierarchy on a lattice of voxels: one cube subdivided into n^3 voxels.

The body is the homogeneous cube of ``measure_cube_gradient_hierarchy.py`` (half-width 10 m, the paper's
contrast, P wave along a cube axis), now cut into n^3 voxels, each carrying the hierarchy of degree q
(``gradient_voxel_lattice.py``: self block from the cube moments, coupling between voxels from ordinary
Gauss integrals). The reference is the same cube by Galerkin voxels of degree 2 on 3^3 cells, whose own
spread is below 2e-5.

Checks:
  [0] the derivatives of the Green's tensor used for the coupling, against the package's closed forms
      (G, its first and its second derivatives) at a point;
  [1] n = 1 is the single site: the lattice code reproduces ``measure_cube_gradient_hierarchy.py``;
  [2] the far-field error for n = 1, 2, 3 and q = 1, 2, 3.

What to expect. With one voxel the error is the isolated cube's static floor (2.7e-3, 2.4e-3, 3e-4 for
q = 1, 2, 3). Subdividing puts voxel faces inside the body, where the true field is smooth, so the
interior voxels are in the regime of the layer; the edges and corners of the BODY remain, and limit the
order of any polynomial scheme on this body. The run measures how much each degree gains.

Run:  python -u scripts/measure_lattice_gradient_hierarchy.py [--ka=0.2] [--n=1,2,3]
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import gradient_voxel_lattice as lat  # noqa: E402
import measure_ball_gradient_hierarchy as hier  # noqa: E402
import measure_cube_gradient_hierarchy as cube  # noqa: E402
from cubic_scattering.resonance_tmatrix import elastodynamic_greens_deriv  # noqa: E402
from gate_sphere_cell_average_vs_mie import obs_points  # noqa: E402


def check_green(omega: float) -> bool:
    x = np.array([3.1, -1.7, 2.3])
    g, gd, gdd = elastodynamic_greens_deriv(x, omega, hier.REF)
    worst = 0.0
    for i in range(3):
        for n in range(3):
            worst = max(
                worst, abs(lat.green_derivative(i, n, (), omega, x[None])[0] - g[i, n]) / abs(g).max()
            )
            for k in range(3):
                got = lat.green_derivative(i, n, (k,), omega, x[None])[0]
                worst = max(worst, abs(got - gd[i, n, k]) / abs(gd).max())
                for m in range(3):
                    got = lat.green_derivative(i, n, (k, m), omega, x[None])[0]
                    worst = max(worst, abs(got - gdd[i, n, k, m]) / abs(gdd).max())
    # third and fourth derivatives: central differences of the second
    eps = 1e-3
    for k in range(3):
        dx = np.zeros(3)
        dx[k] = eps
        fd = (
            lat.green_derivative(0, 1, (0, 2), omega, (x + dx)[None])[0]
            - lat.green_derivative(0, 1, (0, 2), omega, (x - dx)[None])[0]
        ) / (2 * eps)
        got = lat.green_derivative(0, 1, (0, 2, k), omega, x[None])[0]
        worst = max(worst, abs(got - fd) / abs(got))
        fd = (
            lat.green_derivative(1, 1, (0, 2, 2), omega, (x + dx)[None])[0]
            - lat.green_derivative(1, 1, (0, 2, 2), omega, (x - dx)[None])[0]
        ) / (2 * eps)
        got = lat.green_derivative(1, 1, (0, 2, 2, k), omega, x[None])[0]
        worst = max(worst, abs(got - fd) / abs(got))
    ok = worst < 1e-6
    print(f"[0] Green's tensor derivatives to fourth order: worst {worst:.1e}   {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    ka, ns = 0.2, [1, 2, 3]
    for a in sys.argv[1:]:
        if a.startswith("--ka="):
            ka = float(a.split("=", 1)[1])
        elif a.startswith("--n="):
            ns = [int(v) for v in a.split("=", 1)[1].split(",")]
    omega = ka * hier.REF.beta / cube.HALF
    obs = obs_points(cube.R_FAR, cube.THETA)
    contrast = hier.CONTRAST
    ok = check_green(omega)

    t0 = time.perf_counter()
    ref = cube.reference_far_field(omega, contrast, 2, 3, obs)
    peak = float(np.max(np.abs(ref)))
    print(
        f"reference: Galerkin degree 2 on 3^3 cells, k_S h = {ka}   {time.perf_counter() - t0:.0f} s",
        flush=True,
    )

    for q in (1, 2, 3):
        single, _ = cube.hierarchy_far_field(omega, contrast, q, obs)
        for n in ns:
            t0 = time.perf_counter()
            side = 2.0 * cube.HALF / n
            c1 = (np.arange(n) - 0.5 * (n - 1)) * side
            centres = np.stack(np.meshgrid(c1, c1, c1, indexing="ij"), -1).reshape(-1, 3)
            idx, sol = lat.solve_lattice(centres, side, omega, contrast, q, cube.K_HAT, cube.POL)
            got = lat.lattice_far_field(centres, side, omega, contrast, idx, sol, obs)
            err = float(np.max(np.abs(got - ref)) / peak)
            note = ""
            if n == 1:
                same = float(np.max(np.abs(got - single)) / peak)
                good = same < 1e-9
                ok = ok and good
                note = f"   [1] equals the single site to {same:.1e} {'PASS' if good else 'FAIL'}"
            print(
                f"  q = {q}  n = {n} ({n**3:3d} voxels, {3 * len(idx) * n**3:5d} unknowns):  "
                f"error {err:.2e}"
                f"   {time.perf_counter() - t0:6.1f} s{note}",
                flush=True,
            )
    print("ALL CHECKS PASS" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
