"""Measure: four preconditioners of the interface solve on the stored kernels (scaling plan S6b, S6c).

Plan: docs/2026-10-15-interface-solver-scaling-plan.md, S6b and S6c.

``interface_downup.solve_interface_fgmres`` on the stored kernels (``LateralOperator(parts=True)``), relative
tolerance 1e-10, with no preconditioner (the benchmark), block-Jacobi, the two-way march (``DownUp``, one pass,
inner tolerance 1e-2) and Algorithm 5.1 for the plane-averaged contrast (``AveragedSweep``).  Boxes straddling
the interface of L x n x n cells (layers split evenly between the sides); omega h / beta_A = 1/6 and 1/2; every
cell at 10, 30 or 50 % of its own medium's lambda, mu and rho; a P wave with lateral slowness (0.3, 0.1) /
alpha_A.  Set-up as S6: split, C-infinity window from Q0 / 4, radius 16 h, depth radius 4 h; media A (5, 3, 2.5)
over B (6.5, 3.6, 2.8), h = 0.05 km, p = 1.

Recorded per solve: outer products, inner products (DownUp), seconds, and the preconditioner's build.
Resumable (rows already in the output are skipped).  Writes scripts/data/interface_preconditioners.jsonl.

Run:  PYTHONPATH=. python scripts/measure_interface_preconditioners.py LxN [LxN ...]   (e.g. 4x6 8x6 8x12)
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
OUT = Path(__file__).parent / "data" / "interface_preconditioners.jsonl"
FREQS = {"1/6": A.beta / (6 * H), "1/2": A.beta / (2 * H)}
CONTRASTS = (0.1, 0.3, 0.5)
PRECONDITIONERS = ("none", "jacobi", "downup", "averaged")


def main() -> None:
    done = set()
    if OUT.exists():
        done = {
            (r["layers"], r["n"], r["freq"], r["contrast"])
            for r in map(json.loads, OUT.read_text().splitlines())
        }
    for spec in sys.argv[1:]:
        n_layers, n = map(int, spec.split("x"))
        todo = [
            (f, c) for f in FREQS for c in CONTRASTS if (n_layers, n, f, c) not in done
        ]
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
                row = {"layers": n_layers, "n": n, "freq": f, "contrast": contrast}
                for pre in PRECONDITIONERS:
                    t0 = time.time()
                    try:
                        _, st = idu.solve_interface_fgmres(
                            c,
                            omega,
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
                            "solve_s": st["seconds"],
                            "total_s": time.time() - t0,
                        }
                    except RuntimeError:
                        row[pre] = {"outer": None}
                with OUT.open("a") as fh:
                    fh.write(json.dumps(row) + "\n")
                print(
                    f"  {f}, {contrast:.0%}: "
                    + ", ".join(f"{p} {row[p]['outer']}" for p in PRECONDITIONERS),
                    flush=True,
                )


if __name__ == "__main__":
    main()
