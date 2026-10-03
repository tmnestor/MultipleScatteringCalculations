#!/usr/bin/env python3
"""The 3-D voxel scheme's terms of second and third order, and its whole nonlinear part, against the exact
ones, angle by angle and wave by wave, over the relative projection error E.

Scheme: ``octree.born_series_octree`` on a uniform tree (each term one product with the matrix) and
``octree.solve_graded_octree`` for the full response. Exact: the graded sphere at contrasts +-d and
+-2d (radius 10 m, core a/2, smoothstep; stiffness contrast only, as in ``measure_t2_law_3d.py``):

    T2 = (16 (u(d) + u(-d)) - (u(2d) + u(-2d))) / (24 d^2),
    T3 = ((u(2d) - u(-2d)) - 2 (u(d) - u(-d))) / (12 d^3),
    T1 likewise, and the nonlinear part u(1) - T1.

Printed for each angle, P (radial) and S (transverse) parts: (T - T_scheme) / (T E) for T2, T3 and the
whole nonlinear part.

Run:  conda run -n seismic python -u scripts/measure_t3_law_3d.py <k_S a> <p> <n ...>   e.g. 0.125 0 4 6 8
"""

import sys
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(W / "scripts"))
sys.path.insert(0, str(W))
from crosscheck_graded_sphere import graded_mie_result  # noqa: E402
from cubic_scattering import MaterialContrast  # noqa: E402
from cubic_scattering.graded_voxel.octree import (  # noqa: E402
    born_series_octree,
    leaf_energies,
    octree_far_field,
    solve_graded_octree,
    uniform_leaves,
)
from cubic_scattering.sphere_scattering import mie_scattered_displacement  # noqa: E402
from gate_sphere_cell_average_vs_mie import CONTRAST as GATE  # noqa: E402
from gate_sphere_cell_average_vs_mie import REF, obs_points  # noqa: E402
from pilot_graded_sphere_vs_exact import RADIUS, THETA, profile, smoothstep  # noqa: E402

K = np.array([1.0, 0.0, 0.0])
CONTRAST = MaterialContrast(GATE.Dlambda, GATE.Dmu, 0.0)  # stiffness only
D = 1e-2


def prof_vec(pos: np.ndarray) -> np.ndarray:
    r = np.linalg.norm(pos, axis=-1)
    x = np.clip((RADIUS - r) / (RADIUS - RADIUS / 2), 0.0, 1.0)
    return np.vectorize(smoothstep)(x)


def split(u: np.ndarray, dirs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    up = np.einsum("ni,ni->n", u, dirs)[:, None] * dirs
    return up, u - up


def series(f):
    v = {s: f(s) for s in (D, -D, 2 * D, -2 * D)}
    o1, o2 = v[D] - v[-D], v[2 * D] - v[-2 * D]
    e1, e2 = v[D] + v[-D], v[2 * D] + v[-2 * D]
    return (8 * o1 - o2) / (12 * D), (16 * e1 - e2) / (24 * D * D), (o2 - 2 * o1) / (12 * D**3)


def ratio(t_ex: np.ndarray, t_sc: np.ndarray, e: float) -> np.ndarray:
    """Re[(T - T_scheme) . conj(T) / |T|^2] / E, per angle."""
    num = np.einsum("ni,ni->n", t_ex - t_sc, np.conj(t_ex))
    return (num / np.einsum("ni,ni->n", t_ex, np.conj(t_ex))).real / e


def main() -> int:
    ka, p = float(sys.argv[1]), int(sys.argv[2])
    ns = [int(a) for a in sys.argv[3:]]
    omega = ka * REF.beta / RADIUS
    rf = 5e8 * RADIUS
    pts = obs_points(rf, THETA)
    dirs = pts / rf

    def exact(scale: float) -> np.ndarray:
        c = MaterialContrast(scale * CONTRAST.Dlambda, scale * CONTRAST.Dmu, 0.0)
        return mie_scattered_displacement(graded_mie_result(omega, RADIUS, RADIUS / 2, REF, c, 12), pts)

    ex = series(exact)
    u_ex = exact(1.0)
    print(f"3-D scheme, k_S a = {ka}, p = r = {p}, stiffness contrast, graded sphere core a/2", flush=True)
    t2, t3 = split(ex[1], dirs), split(ex[2], dirs)
    print(
        "  exact T3/T2 at 0.2 rad: P "
        + f"{(t3[0][0] @ np.conj(t2[0][0]) / (t2[0][0] @ np.conj(t2[0][0]))).real:+.4f}, S "
        + f"{(t3[1][0] @ np.conj(t2[1][0]) / (t2[1][0] @ np.conj(t2[1][0]))).real:+.4f}",
        flush=True,
    )
    for n in ns:
        cs, hs = uniform_leaves(RADIUS, n)
        defect, _, norm = leaf_energies(prof_vec, cs, hs, p)
        e_proj = float(defect.sum() / norm.sum())
        cache: dict = {}
        terms = born_series_octree(
            omega, REF, CONTRAST, cs, hs, profile, K, K, "P", 3, p=p, r=p, block_cache=cache
        )
        t1s, t2s, t3s = (sum(octree_far_field(t, dirs, rf)) for t in terms)
        solved = solve_graded_octree(
            omega, REF, CONTRAST, cs, hs, profile, K, K, "P", p=p, r=p, block_cache=cache
        )
        u_s = sum(octree_far_field(solved, dirs, rf))
        print(f"\n n = {n}: {len(hs)} leaves, E = {e_proj:.4e}", flush=True)
        print("  angle    T2: P      S       T3: P      S       nonlinear: P      S")
        r2 = [ratio(a, b, e_proj) for a, b in zip(split(ex[1], dirs), split(t2s, dirs), strict=True)]
        r3 = [ratio(a, b, e_proj) for a, b in zip(split(ex[2], dirs), split(t3s, dirs), strict=True)]
        rn = [
            ratio(a, b, e_proj)
            for a, b in zip(split(u_ex - ex[0], dirs), split(u_s - t1s, dirs), strict=True)
        ]
        for j, th in enumerate(THETA):
            print(
                f"  {th:5.2f}   {r2[0][j]:7.4f} {r2[1][j]:7.4f}   {r3[0][j]:7.4f} {r3[1][j]:7.4f}"
                f"   {rn[0][j]:7.4f} {rn[1][j]:7.4f}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
