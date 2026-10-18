#!/usr/bin/env python3
"""GATE: a full 3-D Foldy-Lax solve with the in-plane sweeps (thesis Algorithm 5.2) inside every product.

solve_foldy_lax_3d is run twice on the same problem: with the closed-form same-depth table (the
reference, itself checked against a dense LU and a Neumann series in test_sweep_solver_3d), and with
apply_g0_3d(in_plane=...) carrying every same-depth pair by the left-right and in-out sweeps instead.
The problem is that test's: 2 x 3 x 2 voxels at pitch 0.5 km in one medium, random local T-matrices
scaled so that the spectral radius of G0 T0 is near 0.9, where the solve needs its iterations and a
wrong operator would show.

Reported per transverse rule: the operator's error (G0 on random sources), the solution's error, and the
GMRES products of each route. The solution error should follow the operator error, amplified by at most
the condition of (I - G0 T0).

MEASURED: spectral radius of G0 T0 0.882, cond(I - G0 T0) 29.2; the table route takes 70 products,
0.02 s.

    kr*pitch  n/axis    nodes        G0   solution   products      sec
          20     256    51468  1.64e-04   1.78e-04         70     92.6
          20     512   205892  1.12e-06   1.34e-06         70    352.9
          20    1024   823592  7.59e-07   7.18e-07         70   1248.8

The solution's error follows the operator's almost one for one, far below the bound of 29, and GMRES
takes the same 70 products by either route: the sweeps change neither the answer, to the quadrature's
accuracy, nor the convergence. They cost four to five orders of magnitude more time than the table.

Run:  PYTHONPATH=. python -u scripts/gate_solve_in_plane_sweeps_3d.py
"""

import argparse
import time

import numpy as np
from cubic_scattering import directional_sweeps as ds
from cubic_scattering.effective_contrasts import ReferenceMedium
from cubic_scattering.sweep_solver import solve_foldy_lax_3d

REF = ReferenceMedium(5.0, 3.0, 2.5)
OMEGA = 2 * np.pi * 6.0 * (1 + 0.03j)
PITCH = 0.5
CHUNK = 8192
RULES = "20:256,20:512,20:1024"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rules", default=RULES, help="comma-separated kr*pitch:n_axis pairs")
    parser.add_argument(
        "--scale", type=float, default=0.33, help="T0 scale (0.33: spectral radius near 0.9)"
    )
    args = parser.parse_args()
    rules = [(float(a), int(b)) for a, b in (r.split(":") for r in args.rules.split(","))]

    grid = ds.SweepGrid3D(n_z=2, n_x=3, n_y=2, pitch=PITCH)
    cache = ds.build_g0_cache_3d(grid, REF, OMEGA)
    rng = np.random.default_rng(20260914)
    n = (grid.n_z, grid.n_x, grid.n_y)
    t0 = args.scale * (rng.normal(size=(*n, 9, 9)) + 1j * rng.normal(size=(*n, 9, 9)))
    psi = rng.normal(size=(*n, 9)) + 1j * rng.normal(size=(*n, 9))
    src = rng.normal(size=(*n, 9)) + 1j * rng.normal(size=(*n, 9))

    size = int(np.prod(n)) * 9
    dense = np.zeros((size, size), dtype=complex)
    for c in range(size):
        e = np.zeros(size, dtype=complex)
        e[c] = 1.0
        dense[:, c] = ds.apply_g0_3d(np.einsum("zxyab,zxyb->zxya", t0, e.reshape(*n, 9)), cache).ravel()
    radius = float(np.abs(np.linalg.eigvals(dense)).max())
    cond = float(np.linalg.cond(np.eye(size) - dense))

    t1 = time.time()
    table = solve_foldy_lax_3d(cache, t0, psi, tol=1e-12)
    t_table = time.time() - t1
    g0_table = ds.apply_g0_3d(src, cache)
    print(
        f"lattice 2 x 3 x 2, pitch {PITCH} km; spectral radius of G0 T0 {radius:.3f}, "
        f"cond(I - G0 T0) {cond:.1f}"
    )
    print(f"table route: {table.n_matvec} products, {t_table:.2f} s")
    print(
        f"\n  {'kr*pitch':>8} {'n/axis':>7} {'nodes':>8} {'G0':>9} {'solution':>9} "
        f"{'products':>9} {'sec':>7}"
    )
    for mult, n_axis in rules:
        sweeps = ds.InPlaneSweeps(
            OMEGA, (REF,) * grid.n_z, ds.make_transverse_nodes(mult / PITCH, n_axis), CHUNK
        )
        g0_swept = ds.apply_g0_3d(src, cache, in_plane=sweeps)
        e_op = np.abs(g0_swept - g0_table).max() / np.abs(g0_table).max()
        t1 = time.time()
        res = solve_foldy_lax_3d(cache, t0, psi, tol=1e-12, in_plane=sweeps)
        dt = time.time() - t1
        e_sol = np.abs(res.psi - table.psi).max() / np.abs(table.psi).max()
        print(
            f"  {mult:>8.0f} {n_axis:>7} {sweeps.nodes.kt.size:>8} {e_op:>9.2e} {e_sol:>9.2e} "
            f"{res.n_matvec:>9} {dt:>7.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
