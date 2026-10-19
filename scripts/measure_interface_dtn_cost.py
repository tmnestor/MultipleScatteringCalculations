"""Measure: where the Dirichlet-to-Neumann form's time goes, against the march, on the stored kernels.

Plan: docs/2026-10-15-interface-solver-scaling-plan.md, S6c.

The experiment of ``measure_interface_dtn.py`` (FGMRES to 1e-10 at omega h / beta_A = 1/2 on
``LateralOperator(parts=True)``, every cell at 50 % or at a fraction uniform on [0, 50] % with seed 11), the
solve's wall time split into the preconditioner's build (for DtN the plane-averaged block LDU at every
wavenumber: frequency and contrast only, shared by every incident wave) and the iteration, and the iteration
into the outer products and the preconditioner's applications; for the march (``DownUp``) and for
``DtNSweep`` (window 0) at inner tolerances 1e-2 (the published one), 1e-1 and 3e-1. Set-up as S6: split,
C-infinity window from Q0 / 4, radius 16 h, depth radius 4 h; media A (5, 3, 2.5) over B (6.5, 3.6, 2.8),
h = 0.05 km, p = 1. Each cost is recorded with what it is shared by: the couplings (``InterfaceCouplings``,
corrections as A0 + omega^2 A1) by every frequency; the operator by every contrast and source at one
frequency; the preconditioner's build by every source at one frequency and contrast; the iteration by
nothing. Resumable (rows already in the output are skipped). Writes scripts/data/interface_dtn_cost.jsonl.

Run:  PYTHONPATH=. python scripts/measure_interface_dtn_cost.py 4x6 8x6 16x6
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
OUT = Path(__file__).parent / "data" / "interface_dtn_cost.jsonl"
OMEGA = A.beta / (2 * H)
PATTERNS = ("50%", "random 0-50%")
#: (label, preconditioner, inner tolerance)
RUNS = [
    ("march", "downup", 1e-2),
    ("dtn 1e-2", "dtn", 1e-2),
    ("dtn 1e-1", "dtn", 1e-1),
    ("dtn 3e-1", "dtn", 3e-1),
]


def contrast_factors(pattern: str, n_cells: int) -> np.ndarray:
    """As ``measure_interface_dtn.py``."""
    if pattern == "random 0-50%":
        return np.random.default_rng(11).uniform(0.0, 0.5, n_cells)
    return np.full(n_cells, float(pattern.rstrip("%")) / 100)


def main() -> None:
    done = set()
    if OUT.exists():
        rows = map(json.loads, OUT.read_text().splitlines())
        done = {(r["layers"], r["n"], r["pattern"], r["run"]) for r in rows}
    for spec in sys.argv[1:]:
        n_layers, n = map(int, spec.split("x"))
        todo = [(p, run) for p in PATTERNS for run in RUNS if (n_layers, n, p, run[0]) not in done]
        if not todo:
            continue
        layers = tuple(range(-n_layers // 2, n_layers // 2))
        cells = [(iz, ix, iy) for iz in layers for ix in range(n) for iy in range(n)]
        grid = iso.InterfaceGrid(np.array(cells), H, A, B)
        t0 = time.time()
        c = iso.InterfaceCouplings(grid, window=isp.Window(0.25), radius=16.0, depth_radius=4.0, split=True)
        t_couplings = time.time() - t0
        t0 = time.time()
        op = iso.LateralOperator(c, OMEGA, p=1, r=0, parts=True)
        t_operator = time.time() - t0
        src = np.random.default_rng(0).standard_normal((len(cells), op.n_source * 9)) + 0j
        op._convolve(src)
        t0 = time.time()
        for _ in range(5):
            op._convolve(src)
        t_product = (time.time() - t0) / 5
        print(
            f"{spec}: couplings {t_couplings:.0f} s (no frequency),"
            f" operator {t_operator:.1f} s (per frequency), one product {1e3 * t_product:.1f} ms",
            flush=True,
        )
        kx, ky = 0.3 * OMEGA / A.alpha, 0.1 * OMEGA / A.alpha
        for pattern, (label, pre, inner_tol) in todo:
            f = contrast_factors(pattern, len(cells))
            ctr = [
                MaterialContrast(fm * m.lam, fm * m.mu, fm * m.rho)
                for fm, m in zip(f, (grid.medium(i) for i in range(len(cells))), strict=True)
            ]
            t0 = time.time()
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
                inner_tol=inner_tol,
                max_iter=3000,
            )
            wall = time.time() - t0
            products_s = st["outer"] * t_product
            row = {
                "layers": n_layers,
                "n": n,
                "freq": "1/2",
                "pattern": pattern,
                "run": label,
                "inner_tol": inner_tol,
                "outer": st["outer"],
                "inner": st["inner"],
                "planes": st["planes"],
                "total_s": wall,
                "build_s": wall - st["seconds"],
                "iteration_s": st["seconds"],
                "products_s": products_s,
                "apply_s_each": (st["seconds"] - products_s) / st["outer"],
                "product_s": t_product,
                "couplings_s": t_couplings,
                "operator_s": t_operator,
            }
            with OUT.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(
                f"  {pattern}, {label}: {st['outer']} outer, {st['inner']} inner; total {wall:.1f} s ="
                f" build {row['build_s']:.1f} + products {products_s:.1f}"
                f" + applications {st['seconds'] - products_s:.1f}"
                f" ({1e3 * row['apply_s_each']:.0f} ms each)",
                flush=True,
            )


if __name__ == "__main__":
    main()
