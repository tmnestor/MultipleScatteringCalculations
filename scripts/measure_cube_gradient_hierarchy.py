#!/usr/bin/env python3
"""The gradient hierarchy of the single site for ONE CUBE, against a converged reference.

The assembly is that of ``measure_ball_gradient_hierarchy.py`` (tested there against the exact sphere);
only the scalar moments change, from the ball's to the cube's. The cube's moments are the ones of the
hierarchy, E[m; D; W] over the cube, here evaluated by the independent ball-plus-remainder route of
``crosscheck_cube_moments_ball_shell.py``, which agrees with every stored closed form.

There is no exact solution for a cube. The reference is the same cube solved by Galerkin voxels on a
refined grid (the cube is the one body the grid represents exactly), at two degrees and several grids,
so that the reference's own convergence can be seen beside the errors it is used to measure.

What is measured: the far field of one homogeneous cube of half-width h under a P plane wave along a
cube axis, at nine scattering angles, 5e8 half-widths away; error = largest component of the difference
over the largest component of the reference.

What is NOT expected to be as clean as the ball: the internal field of an isolated cube has
concentrations at its edges and corners, which no Taylor polynomial about the centre represents. The
hierarchy therefore converges in k h, as for the ball, only down to a floor set by that static
misrepresentation, and the floor falls with the degree only as fast as the polynomial can follow the
corners. Both are reported as measured.

Run small first:  python -u scripts/measure_cube_gradient_hierarchy.py --ref=2,3
"""

import functools
import json
import sys
import time
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import crosscheck_cube_moments_ball_shell as cm  # noqa: E402
import measure_ball_gradient_hierarchy as hier  # noqa: E402
from cubic_scattering.graded_voxel.farfield import graded_far_field  # noqa: E402
from cubic_scattering.graded_voxel.solver import solve_graded_sphere  # noqa: E402
from gate_sphere_cell_average_vs_mie import obs_points  # noqa: E402

HALF = 10.0  # half-width of the cube
THETA = np.linspace(0.2, np.pi - 0.2, 9)
R_FAR = 5.0e8 * HALF
K_HAT = np.array([1.0, 0.0, 0.0])
POL = np.array([1.0, 0.0, 0.0])


#: the unit-cube moments already computed, written by ``build_cube_unit_moments.py``: the self block of the
#: hierarchy needs a few thousand of them, each a pure number, and computing them took 13 to 15 s a process
STORE = Path(__file__).resolve().parent / "cube_unit_moments.json"


@functools.cache
def _stored() -> dict[tuple, float]:
    if not STORE.is_file():
        return {}
    data = json.loads(STORE.read_text())
    return {(int(e["m"]), tuple(e["d"]), tuple(e["w"])): float(e["value"]) for e in data["moments"]}


@functools.cache
def cube_unit(m: int, ds: tuple[int, ...], w: tuple[int, ...]) -> float:
    """E[m; ds; w] over the cube [-1/2, 1/2]^3: from the store when it holds it, else computed."""
    key = (m, tuple(ds), tuple(w))
    # the cube is symmetric under each reflection x_k -> -x_k, which multiplies the integrand by (-1) to the
    # number of times k occurs in ds and w: an odd count makes the moment vanish exactly
    if any((ds.count(ax) + w.count(ax)) % 2 for ax in range(3)):
        return 0.0
    stored = _stored()
    if key in stored:
        return stored[key]
    return cm.moment(m, list(ds), list(w))


def cube_scalar_moment(side: float, m: int, ds: tuple[int, ...], w: tuple[int, ...]) -> float:
    """The same for a cube of this side: homogeneous of degree m - D + W + 3."""
    if m % 2 == 0 and len(ds) > m:
        return 0.0
    return side ** (m - len(ds) + len(w) + 3) * cube_unit(m, tuple(sorted(ds)), tuple(sorted(w)))


def cube_quadrature(side: float, n: int = 12) -> tuple[np.ndarray, np.ndarray]:
    x, w = leggauss(n)
    x, w = 0.5 * side * x, 0.5 * side * w
    pts = np.stack(np.meshgrid(x, x, x, indexing="ij"), -1).reshape(-1, 3)
    wts = np.einsum("i,j,k->ijk", w, w, w).ravel()
    return pts, wts


def hierarchy_far_field(omega: float, contrast, q: int, obs: np.ndarray) -> tuple[np.ndarray, float]:
    """One cube by the hierarchy of degree q: far field and the condition number of its system."""
    hier.scalar_moment = cube_scalar_moment
    hier.ball_quadrature = lambda radius, n=12: cube_quadrature(radius, n)
    side = 2.0 * HALF
    idx, sol, cond = hier.solve_site(side, omega, contrast, q, K_HAT, POL)
    return hier.far_field(side, omega, contrast, idx, sol, obs), cond


def reference_far_field(omega: float, contrast, p: int, n_sub: int, obs: np.ndarray) -> np.ndarray:
    """The same cube by Galerkin voxels of degree p on n_sub^3 cells, every cell kept."""
    res = solve_graded_sphere(
        omega,
        HALF,
        hier.REF,
        contrast,
        n_sub,
        lambda _pos: 1.0,
        K_HAT,
        POL,
        "P",
        p=p,
        r=p,
        inside=lambda _c: True,
    )
    u_p, u_s = graded_far_field(res, obs / R_FAR, R_FAR, K_HAT, POL, "P")
    return u_p + u_s


def main() -> int:
    ref_grids, kas = [2, 3, 4], [0.4, 0.2, 0.1]
    for a in sys.argv[1:]:
        if a.startswith("--ref="):
            ref_grids = [int(v) for v in a.split("=", 1)[1].split(",")]
        elif a.startswith("--ka="):
            kas = [float(v) for v in a.split("=", 1)[1].split(",")]
    obs = obs_points(R_FAR, THETA)
    contrast = hier.CONTRAST
    for ka in kas:
        omega = ka * hier.REF.beta / HALF
        print(f"\none cube, k_S h = {ka}", flush=True)
        refs = {}
        for p in (1, 2):
            for n in ref_grids:
                t0 = time.perf_counter()
                refs[(p, n)] = reference_far_field(omega, contrast, p, n, obs)
                print(
                    f"  reference: Galerkin degree {p}, {n}^3 cells   {time.perf_counter() - t0:6.1f} s",
                    flush=True,
                )
        best = refs[(2, ref_grids[-1])]
        peak = float(np.max(np.abs(best)))
        print("  the reference's own convergence (difference from the finest degree-2 grid):")
        for (p, n), u in refs.items():
            if (p, n) != (2, ref_grids[-1]):
                print(f"    degree {p}, {n}^3: {np.max(np.abs(u - best)) / peak:.2e}")
        for q in (1, 2, 3):
            got, cond = hierarchy_far_field(omega, contrast, q, obs)
            err = np.max(np.abs(got - best)) / peak
            print(f"  hierarchy q = {q}: error {err:.2e}   (condition {cond:.1e})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
