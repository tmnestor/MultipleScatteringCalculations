"""Measure: the two-way march (DownUp) as FGMRES's preconditioner against block-Jacobi (scaling plan S6b).

Plan: docs/2026-10-15-interface-solver-scaling-plan.md, S6b.

``interface_downup.solve_interface_fgmres`` on the stored kernels (``LateralOperator(parts=True)``), relative
tolerance 1e-10, with the DownUp preconditioner (one pass, the planes' inner GMRES to ``INNER_TOL``) and with
block-Jacobi (the baseline of S6), on boxes straddling the interface of 2, 4, 6 and 8 layers (split evenly between
the sides) of n x n cells; omega h / beta_A = 1/6 and 1/2; every cell at 10, 30 or 50 % of its own medium's
lambda, mu and rho; a P wave with lateral slowness (0.3, 0.1) / alpha_A.  Set-up as S6: split, C-infinity window
from Q0 / 4, radius 16 h, depth radius 4 h; media A (5, 3, 2.5) over B (6.5, 3.6, 2.8), h = 0.05 km, p = 1.

Recorded per solve: outer products (with the whole operator), inner products (with one plane's own operator),
plane solves, seconds, and the solutions' difference between the two preconditioners.  Resumable (rows already in
the output are skipped).  Writes scripts/data/interface_downup.jsonl.

Run:  PYTHONPATH=. python scripts/measure_interface_downup.py [box indices, default all]
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
OUT = Path(__file__).parent / "data" / "interface_downup.jsonl"
#: (layers, cells per side along x and y)
BOXES = [
    ((-1, 0), 6),
    ((-2, -1, 0, 1), 6),
    ((-3, -2, -1, 0, 1, 2), 6),
    ((-4, -3, -2, -1, 0, 1, 2, 3), 6),
    ((-2, -1, 0, 1), 12),
]
FREQS = {"1/6": A.beta / (6 * H), "1/2": A.beta / (2 * H)}
CONTRASTS = (0.1, 0.3, 0.5)
INNER_TOL = 1e-2


def main() -> None:
    done = set()
    if OUT.exists():
        done = {
            (r["layers"], r["n"], r["freq"], r["contrast"])
            for r in map(json.loads, OUT.read_text().splitlines())
        }
    picked = [int(a) for a in sys.argv[1:]] or range(len(BOXES))
    for layers, n in (BOXES[i] for i in picked):
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
            op = iso.LateralOperator(c, omega, p=1, r=0, parts=True)
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
                row = {
                    "layers": len(layers),
                    "n": n,
                    "cells": len(cells),
                    "freq": f,
                    "contrast": contrast,
                    "inner_tol": INNER_TOL,
                }
                sols = {}
                for pre in ("jacobi", "downup"):
                    try:
                        res, st = idu.solve_interface_fgmres(
                            c,
                            omega,
                            ctr,
                            kx,
                            ky,
                            "P",
                            operator=op,
                            preconditioner=pre,
                            tol=1e-10,
                            inner_tol=INNER_TOL,
                            max_iter=2000,
                        )
                        sols[pre] = res.psi
                        row[pre] = {
                            k: st[k] for k in ("outer", "inner", "planes", "seconds")
                        }
                    except RuntimeError:
                        row[pre] = {"outer": None}
                if len(sols) == 2:
                    a, b = sols["jacobi"], sols["downup"]
                    row["solution_diff"] = float(np.abs(a - b).max() / np.abs(a).max())
                with OUT.open("a") as fh:
                    fh.write(json.dumps(row) + "\n")
                j, d = row["jacobi"], row["downup"]
                print(
                    f"  {f}, {contrast:.0%}: jacobi {j['outer']} outer {j.get('seconds', 0):.1f} s;"
                    f" downup {d['outer']} outer, {d.get('inner')} inner, {d.get('planes')} plane solves,"
                    f" {d.get('seconds', 0):.1f} s; solutions {row.get('solution_diff', float('nan')):.1e}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
