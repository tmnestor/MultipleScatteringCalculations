"""Measure: Algorithm 5.1 in its Dirichlet-to-Neumann form against the other preconditioners (scaling plan S6c).

Plan: docs/2026-10-15-interface-solver-scaling-plan.md, S6c.

``interface_downup.solve_interface_fgmres`` on the stored kernels (``LateralOperator(parts=True)``), relative
tolerance 1e-10, with no preconditioner, block-Jacobi, the two-way march (one pass), Algorithm 5.1 for the
plane-averaged contrast and its Dirichlet-to-Neumann form with the true contrast (``DtNSweep``, the pivots solved
to 1e-2).  Boxes straddling the interface of L x n x n cells; omega h / beta_A = 1/2; three contrasts: every cell
at 30 % or at 50 % of its own medium's lambda, mu and rho, and a contrast varying from cell to cell, uniform on
[0, 50] % (seeded; the cell fit allows up to 52 %).  P wave with lateral slowness (0.3, 0.1) / alpha_A; set-up as S6.

Recorded per solve: outer and inner products and seconds.  Resumable.  Writes scripts/data/interface_dtn.jsonl.

Run:  PYTHONPATH=. python scripts/measure_interface_dtn.py LxN [LxN ...]   (e.g. 2x6 8x6 8x12)
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
OUT = Path(__file__).parent / "data" / "interface_dtn.jsonl"
OMEGA = A.beta / (2 * H)
PATTERNS = ("30%", "50%", "random 0-50%")
PRECONDITIONERS = ("none", "jacobi", "downup", "averaged", "dtn")


def contrast_factors(pattern: str, n_cells: int) -> np.ndarray:
    if pattern == "random 0-50%":
        return np.random.default_rng(11).uniform(0.0, 0.5, n_cells)
    return np.full(n_cells, float(pattern.rstrip("%")) / 100)


def main() -> None:
    done = set()
    if OUT.exists():
        done = {
            (r["layers"], r["n"], r["pattern"])
            for r in map(json.loads, OUT.read_text().splitlines())
        }
    for spec in sys.argv[1:]:
        n_layers, n = map(int, spec.split("x"))
        todo = [p for p in PATTERNS if (n_layers, n, p) not in done]
        if not todo:
            continue
        layers = tuple(range(-n_layers // 2, n_layers // 2))
        cells = [(iz, ix, iy) for iz in layers for ix in range(n) for iy in range(n)]
        grid = iso.InterfaceGrid(np.array(cells), H, A, B)
        t0 = time.time()
        c = iso.InterfaceCouplings(
            grid, window=isp.Window(0.25), radius=16.0, depth_radius=4.0, split=True
        )
        print(f"{spec}: set-up {time.time() - t0:.0f} s", flush=True)
        op = iso.LateralOperator(c, OMEGA, p=1, r=0, parts=True)
        kx, ky = 0.3 * OMEGA / A.alpha, 0.1 * OMEGA / A.alpha
        for pattern in todo:
            f = contrast_factors(pattern, len(cells))
            ctr = [
                MaterialContrast(fm * m.lam, fm * m.mu, fm * m.rho)
                for fm, m in zip(
                    f, (grid.medium(i) for i in range(len(cells))), strict=True
                )
            ]
            row = {"layers": n_layers, "n": n, "freq": "1/2", "pattern": pattern}
            for pre in PRECONDITIONERS:
                t0 = time.time()
                try:
                    _, st = idu.solve_interface_fgmres(
                        c,
                        OMEGA,
                        ctr,
                        kx,
                        ky,
                        "P",
                        operator=op,
                        preconditioner=pre,
                        tol=1e-10,
                        inner_tol=1e-2,
                        max_iter=3000,
                    )
                    row[pre] = {
                        "outer": st["outer"],
                        "inner": st["inner"],
                        "total_s": time.time() - t0,
                    }
                except RuntimeError:
                    row[pre] = {"outer": None}
            with OUT.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(
                f"  {pattern}: "
                + ", ".join(f"{p} {row[p]['outer']}" for p in PRECONDITIONERS),
                flush=True,
            )


if __name__ == "__main__":
    main()
