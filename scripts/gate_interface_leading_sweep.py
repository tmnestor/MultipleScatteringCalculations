#!/usr/bin/env python3
"""GATE: the true sweep with whole windows (``LeadingSweep``) on the matrix-free operator (route A).

On boxes straddling the interface, at omega h / beta_A = 1/2:
  (1) ``LeadingSweep`` on ``MatrixFreeOperator``, its leading systems solved to ``--tight``, inverts the
      matrix-free operator: the plane-by-plane solve from the bottom, the solved planes entering through the
      true coupling (on the ``--check`` box);
  (2) it agrees with ``LeadingSweep`` on the stored kernels (``LateralOperator``) applied to the same vector,
      the two operators agreeing to ~1e-11 by construction (same box);
  (3) the cost on the ``--measure`` box: one matrix-free product, and one application of ``LeadingSweep`` at
      inner tolerance 1e-2 with its leading solves' inner products (each a full matrix-free product);
  (4) with ``--fgmres``: the outer products, inner products and time of flexible GMRES to 1e-10 on the
      matrix-free operator with no preconditioner, block-Jacobi, the two-way march (``MatrixFreeDownUp``, one
      pass) and ``LeadingSweep``, inner tolerance 1e-2, at a uniform 50 % and at a contrast varying from cell to
      cell on [0, 50] %.

MEASURED (seismic env, finufft from conda-forge):
  (1) 4 x 2 x 2, to 1e-10: |result - x| / |x| = 8.1e-11, 38 inner products, 62 s.
  (2) matrix-free against stored kernels, same vector: 6.4e-12.
  (3) 4 x 3 x 3: one matrix-free product 1.5 s; one application at 1e-2: 16 inner and 3 coupling products
      over 4 leading solves, 30 s.
  (4) 4 x 3 x 3, outer products (inner products), time:
          contrast   none        block-Jacobi   march          LeadingSweep
          50 %       24 (0) 38 s  15 (0) 24 s   8 (224) 141 s  6 (94) 204 s
          varying    22 (0) 39 s  12 (0) 22 s   6 (120) 65 s   6 (75) 165 s
  The fewest outer products, as on the stored kernels, but not the least time: each of the leading solves'
  inner products is a full matrix-free product, where the march's act on a single plane. Making the
  leading solves cheap is route B: the plane-averaged factorisation on the plane-wave modes.

Run:  PYTHONPATH=. python -u scripts/gate_interface_leading_sweep.py [--check 4x2] [--measure 4x3] [--tight 1e-10]
      [--fgmres]
"""

import argparse
import time

import numpy as np
from cubic_scattering import interface_downup as idu
from cubic_scattering import interface_solver as iso
from cubic_scattering import interface_spectral as isp
from cubic_scattering.effective_contrasts import MaterialContrast, ReferenceMedium
from cubic_scattering.graded_voxel.basis import source_expansion

A = ReferenceMedium(5.0, 3.0, 2.5)
B = ReferenceMedium(6.5, 3.6, 2.8)
H = 0.05
OMEGA = A.beta / (2 * H)
PRECONDITIONERS = ("none", "jacobi", "downup", "leading")


def contrasts(grid, cells: int, pattern: str) -> list:
    if pattern == "varying":
        f = np.random.default_rng(11).uniform(0.0, 0.5, cells)
    else:
        f = np.full(cells, 0.5)
    return [
        MaterialContrast(fm * m.lam, fm * m.mu, fm * m.rho)
        for fm, m in zip(f, map(grid.medium, range(cells)), strict=True)
    ]


def build(spec: str):
    n_layers, n = map(int, spec.split("x"))
    layers = tuple(range(-n_layers // 2, n_layers // 2))
    cells = [(iz, ix, iy) for iz in layers for ix in range(n) for iy in range(n)]
    grid = iso.InterfaceGrid(np.array(cells), H, A, B)
    t0 = time.time()
    c = iso.InterfaceCouplings(grid, window=isp.Window(0.25), radius=16.0, depth_radius=4.0, split=True)
    lat = iso.LateralOperator(c, OMEGA, p=1, r=0)
    mf = iso.MatrixFreeOperator(c, OMEGA, p=1, r=0)
    print(f"{n_layers} x {n} x {n}: set-up {time.time() - t0:.0f} s", flush=True)
    return grid, c, lat, mf, len(cells)


def expansion(grid, op, n_cells: int, pattern: str) -> np.ndarray:
    delta = iso.cell_contrasts(grid, OMEGA, contrasts(grid, n_cells, pattern), 0)
    return np.array([source_expansion(d, op.n_field)[: op.n_source, : op.na] for d in delta])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", default="4x2", help="box for the exactness checks, LxN")
    parser.add_argument("--measure", default="4x3", help="box for the cost, LxN")
    parser.add_argument(
        "--tight", type=float, default=1e-10, help="inner tolerance of the exactness checks"
    )
    parser.add_argument(
        "--fgmres", action="store_true", help="also the FGMRES comparison on the measure box"
    )
    args = parser.parse_args()

    grid, _, lat, mf, n_cells = build(args.check)
    e = expansion(grid, mf, n_cells, "varying")
    x = np.random.default_rng(6).standard_normal((n_cells, mf.na, 9)) + 0j
    src = np.einsum("ncbij,nbj->nci", e, x).reshape(n_cells, -1)
    ax = np.einsum("ab,nbi->nai", mf.m9, x) - mf._convolve(src)
    t0 = time.time()
    sweep = idu.LeadingSweep(mf, e, inner_tol=args.tight)
    got = sweep(ax.ravel()).reshape(x.shape)
    err = np.abs(got - x).max() / np.abs(x).max()
    print(
        f"  (1) matrix-free LeadingSweep to {args.tight:g} on A x: |result - x| / |x| = {err:.1e} "
        f"({sweep.inner_products} inner products, {time.time() - t0:.0f} s)",
        flush=True,
    )
    v = np.random.default_rng(3).standard_normal(n_cells * mf.na * 9) + 0j
    a_mf = idu.LeadingSweep(mf, e, inner_tol=args.tight)(v)
    a_lat = idu.LeadingSweep(lat, e, inner_tol=args.tight)(v)
    print(
        f"  (2) matrix-free against stored kernels, same vector: {np.abs(a_mf - a_lat).max() / np.abs(a_lat).max():.1e}",
        flush=True,
    )

    grid, c, _, mf, n_cells = build(args.measure)
    e = expansion(grid, mf, n_cells, "varying")
    src = np.random.default_rng(4).standard_normal((n_cells, mf.n_source * 9)) + 0j
    t0 = time.time()
    mf._convolve(src)
    t_product = time.time() - t0
    v = np.random.default_rng(3).standard_normal(n_cells * mf.na * 9) + 0j
    sweep = idu.LeadingSweep(mf, e, inner_tol=1e-2)
    t0 = time.time()
    sweep(v)
    print(
        f"  (3) one matrix-free product {t_product:.1f} s; one LeadingSweep application at 1e-2: "
        f"{sweep.inner_products} inner and {sweep.outer_products} coupling products over {sweep.inner_solves} "
        f"leading solves, {time.time() - t0:.0f} s",
        flush=True,
    )

    if args.fgmres:
        kx, ky = 0.3 * OMEGA / A.alpha, 0.1 * OMEGA / A.alpha
        for pattern in ("50%", "varying"):
            ctr = contrasts(grid, n_cells, pattern)
            row = []
            for pre in PRECONDITIONERS:
                t0 = time.time()
                _, st = idu.solve_interface_fgmres(
                    c,
                    OMEGA,
                    ctr,
                    kx,
                    ky,
                    "P",
                    p=1,
                    r=0,
                    operator=mf,
                    preconditioner=pre,
                    tol=1e-10,
                    inner_tol=1e-2,
                    max_iter=3000,
                )
                row.append(f"{pre} {st['outer']} ({st['inner']}) {time.time() - t0:.0f} s")
            print(f"  (4) {pattern:>8}: " + " | ".join(row), flush=True)


if __name__ == "__main__":
    main()
