"""Measure: GMRES products of the interface solve against contrast, box size and frequency (scaling plan S6).

Plan: docs/2026-10-15-interface-solver-scaling-plan.md, S6.

``solve_interface_gmres`` with its block-Jacobi preconditioner (each cell's (M - K_mm E_m)^-1, K_mm the self block with
its own image), relative tolerance 1e-10.  The products are counted, not timed: the stored-kernel operator
(``LateralOperator``) is used, whose product is the matrix-free one's to ~1e-11 (scripts/gate_interface_matrix_free.py)
and costs hundredths of a second at these sizes.

Boxes straddling the interface: 2 x 6 x 6, 2 x 12 x 12, 4 x 12 x 12 and 2 x 24 x 24 cells (layers split evenly between
the sides); omega h / beta_A = 1/6 and 1/2; every cell with a contrast of 10, 30 or 50 % of its own medium's lambda, mu
and rho; a P wave with lateral slowness (0.3, 0.1) / alpha_A.  Set-up: split, C-infinity window from Q0 / 4, radius
16 h, depth radius 4 h.  Media A (5, 3, 2.5) over B (6.5, 3.6, 2.8), h = 0.05 km, p = 1.

GMRES restarts every 100 products and stops after 2000 (reported as not converged).  Resumable: the rows already in
the output file are skipped, a grid's set-up only when one of its rows is missing.  Writes
scripts/data/interface_gmres_iterations.jsonl.

Run:  PYTHONPATH=. python scripts/measure_interface_gmres_iterations.py [grid indices, default all]
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

from cubic_scattering import interface_solver as iso
from cubic_scattering import interface_spectral as isp
from cubic_scattering.effective_contrasts import MaterialContrast, ReferenceMedium

A = ReferenceMedium(5.0, 3.0, 2.5)
B = ReferenceMedium(6.5, 3.6, 2.8)
H = 0.05
OUT = Path(__file__).parent / "data" / "interface_gmres_iterations.jsonl"
#: (layers, cells per side along x and y)
GRIDS = [((-1, 0), 6), ((-1, 0), 12), ((-2, -1, 0, 1), 12), ((-1, 0), 24)]
FREQS = {"1/6": A.beta / (6 * H), "1/2": A.beta / (2 * H)}
CONTRASTS = (0.1, 0.3, 0.5)


def main() -> None:
    done = set()
    if OUT.exists():
        done = {
            (r["layers"], r["n"], r["freq"], r["contrast"])
            for r in map(json.loads, OUT.read_text().splitlines())
        }
    picked = [int(a) for a in sys.argv[1:]] or range(len(GRIDS))
    for layers, n in (GRIDS[i] for i in picked):
        todo = [
            (f, c)
            for f in FREQS
            for c in CONTRASTS
            if (len(layers), n, f, c) not in done
        ]
        if not todo:
            continue
        cells = [(iz, ix, iy) for iz in layers for ix in range(n) for iy in range(n)]
        grid = iso.InterfaceGrid(np.array(cells), H, A, B)
        t0 = time.time()
        c = iso.InterfaceCouplings(
            grid, window=isp.Window(0.25), radius=16.0, depth_radius=4.0, split=True
        )
        t_setup = time.time() - t0
        print(
            f"{len(cells)} cells, {len(layers)} layers: set-up {t_setup:.0f} s",
            flush=True,
        )
        for f, omega in FREQS.items():
            if not any(ff == f for ff, _ in todo):
                continue
            t0 = time.time()
            op = iso.LateralOperator(c, omega, p=1, r=0)
            t_op = time.time() - t0
            kx, ky = 0.3 * omega / A.alpha, 0.1 * omega / A.alpha
            for contrast in CONTRASTS:
                if (f, contrast) not in todo:
                    continue
                ctr = [
                    MaterialContrast(
                        contrast * m.lam, contrast * m.mu, contrast * m.rho
                    )
                    for m in (grid.medium(i) for i in range(len(cells)))
                ]
                t0 = time.time()
                try:
                    _, n_it = iso.solve_interface_gmres(
                        c,
                        omega,
                        ctr,
                        kx,
                        ky,
                        "P",
                        p=1,
                        operator=op,
                        tol=1e-10,
                        restart=100,
                        max_cycles=20,
                    )
                    converged = True
                except RuntimeError:
                    n_it, converged = 2000, False
                row = {
                    "layers": len(layers),
                    "n": n,
                    "cells": len(cells),
                    "freq": f,
                    "contrast": contrast,
                    "products": n_it,
                    "converged": converged,
                    "solve_s": time.time() - t0,
                    "operator_s": t_op,
                    "setup_s": t_setup,
                }
                with OUT.open("a") as fh:
                    fh.write(json.dumps(row) + "\n")
                print(
                    f"  omega h / beta_A = {f}, contrast {contrast:.0%}: {n_it} products"
                    f"{'' if converged else ' (not converged)'}, {row['solve_s']:.1f} s",
                    flush=True,
                )


if __name__ == "__main__":
    main()
