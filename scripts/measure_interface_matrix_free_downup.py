"""Measure: matrix-free FGMRES, the two-way march against block-Jacobi and no preconditioner, by box shape.

Plan: docs/2026-10-15-interface-solver-scaling-plan.md, S6b (the box shapes) and S6c.

``interface_downup.solve_interface_fgmres`` on ``MatrixFreeOperator`` (``cache=True``), relative tolerance
1e-10, at omega h / beta_A = 1/2, every cell at 30 or 50 % of its own medium's lambda, mu and rho, a P wave
with lateral slowness (0.3, 0.1) / alpha_A. Preconditioners: none; block-Jacobi; the march on the recursions
(``MatrixFreeDownUp``), each plane's own reflection lagged, inner GMRES to 1e-2, with nodes of its own
(``interface_downup.COARSE``, or ``COARSE_BOUNDED`` where marked), one or three passes. Set-up as S6: split,
C-infinity window from Q0 / 4, radius 16 h, depth radius 4 h; media A (5, 3, 2.5) over B (6.5, 3.6, 2.8),
h = 0.05 km, p = 1.

These are the boxes whose matrix-free rows were measured before the fix of the cell moments' signed zero
(``cubic_scattering.special``), which made the whole-space march between planes three or more apart wrong:
every box with four or more planes on a side (boxes 0 to 3); box 4 (4 x 24 x 24, two planes a side) was
not affected and is rerun so that every time is from one machine. Recorded per solve: outer products, inner
products, plane solves, seconds (the iteration), the preconditioner's build and the solution's difference
from block-Jacobi's; per box, the set-up and one product. Resumable (rows already in the output are
skipped). Writes scripts/data/interface_matrix_free_downup.jsonl.

Run:  PYTHONPATH=. python scripts/measure_interface_matrix_free_downup.py [box indices, default 0 1 2]
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from cubic_scattering import interface_downup as idu
from cubic_scattering import interface_solver as iso
from cubic_scattering import interface_spectral as isp
from cubic_scattering.effective_contrasts import MaterialContrast, ReferenceMedium

A = ReferenceMedium(5.0, 3.0, 2.5)
B = ReferenceMedium(6.5, 3.6, 2.8)
H = 0.05
OMEGA = A.beta / (2 * H)
OUT = Path(__file__).parent / "data" / "interface_matrix_free_downup.jsonl"
CONTRASTS = (0.3, 0.5)
INNER_TOL = 1e-2
#: (label, preconditioner, passes, coarse nodes)
JACOBI = ("block-Jacobi", "jacobi", 1, None)
NONE = ("none", "none", 1, None)
ONE = ("march, one pass", "downup", 1, "COARSE")
ONE_BOUNDED = ("march, one pass (bounded)", "downup", 1, "COARSE_BOUNDED")
THREE = ("march, three passes", "downup", 3, "COARSE")
THREE_BOUNDED = ("march, three passes (bounded)", "downup", 3, "COARSE_BOUNDED")
#: (layers, cells per side along x and y, runs); Jacobi first, so that the others compare with it
BOXES = [
    (8, 6, [JACOBI, ONE, THREE]),
    (16, 6, [JACOBI, ONE, THREE]),
    (32, 6, [JACOBI, NONE, ONE, THREE_BOUNDED]),
    (8, 12, [JACOBI, ONE, THREE]),
    (4, 24, [JACOBI, NONE, ONE_BOUNDED, THREE_BOUNDED]),
]


def main() -> None:
    done = set()
    if OUT.exists():
        rows = map(json.loads, OUT.read_text().splitlines())
        done = {(r["layers"], r["n"], r["run"], r["contrast"]) for r in rows}
    picked = [int(a) for a in sys.argv[1:]] or [0, 1, 2]
    for n_layers, n, runs in (BOXES[i] for i in picked):
        todo = [(run, c) for run in runs for c in CONTRASTS if (n_layers, n, run[0], c) not in done]
        if not todo:
            continue
        layers = range(-n_layers // 2, n_layers // 2)
        cells = [(iz, ix, iy) for iz in layers for ix in range(n) for iy in range(n)]
        grid = iso.InterfaceGrid(np.array(cells), H, A, B)
        t0 = time.time()
        c = iso.InterfaceCouplings(grid, window=isp.Window(0.25), radius=16.0, depth_radius=4.0, split=True)
        t_setup = time.time() - t0
        t0 = time.time()
        op = iso.MatrixFreeOperator(c, OMEGA, p=1, r=0, cache=True)
        t_operator = time.time() - t0
        src = np.random.default_rng(0).standard_normal((len(cells), op.n_source * 9)) + 0j
        t0 = time.time()
        op._convolve(src)
        t_product = time.time() - t0
        print(
            f"{n_layers} x {n} x {n}: set-up {t_setup:.0f} s, operator {t_operator:.0f} s,"
            f" one product {t_product:.1f} s",
            flush=True,
        )
        kx, ky = 0.3 * OMEGA / A.alpha, 0.1 * OMEGA / A.alpha
        jacobi: dict[float, np.ndarray] = {}
        for (label, pre, passes, coarse), contrast in todo:
            ctr = [
                MaterialContrast(contrast * m.lam, contrast * m.mu, contrast * m.rho)
                for m in (grid.medium(i) for i in range(len(cells)))
            ]
            t0 = time.time()
            res, st = idu.solve_interface_fgmres(
                c,
                OMEGA,
                ctr,
                kx,
                ky,
                "P",
                operator=op,
                preconditioner=pre,
                self_reflection="lagged",
                coarse=None if coarse is None else getattr(idu, coarse),
                tol=1e-10,
                inner_tol=INNER_TOL,
                passes=passes,
                max_iter=2000,
            )
            wall = time.time() - t0
            row = {
                "layers": n_layers,
                "n": n,
                "cells": len(cells),
                "freq": "1/2",
                "contrast": contrast,
                "run": label,
                "preconditioner": pre,
                "passes": passes,
                "coarse": coarse,
                "inner_tol": INNER_TOL,
                **{k: st[k] for k in ("outer", "inner", "planes", "seconds")},
                "build_s": wall - st["seconds"],
                "setup_s": t_setup,
                "operator_s": t_operator,
                "product_s": t_product,
            }
            if pre == "jacobi":
                jacobi[contrast] = res.psi
            elif contrast in jacobi:
                a = jacobi[contrast]
                row["solution_diff"] = float(np.abs(res.psi - a).max() / np.abs(a).max())
            with OUT.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(
                f"  {contrast:.0%}, {label}: {st['outer']} outer, {st.get('inner')} inner,"
                f" {st['seconds']:.0f} s (build {row['build_s']:.0f} s);"
                f" from block-Jacobi {row.get('solution_diff', float('nan')):.1e}",
                flush=True,
            )


if __name__ == "__main__":
    main()
