"""Measure: approximate Dirichlet-to-Neumann preconditioners against the march and block-Jacobi.

Plan: docs/2026-10-15-interface-solver-scaling-plan.md, S6c.

The experiment of ``measure_interface_dtn.py``: FGMRES to 1e-10 at omega h / beta_A = 1/2 on the stored
kernels (``LateralOperator(parts=True)``), every cell at 50 % of its medium's lambda, mu and rho or at a
fraction uniform on [0, 50] % (seed 11), a P wave with lateral slowness (0.3, 0.1) / alpha_A. The
Dirichlet-to-Neumann form (``DtNSweep``, window 0) approximated in three ways, each cheaper than the
published form (pivots to 1e-2):

- the pivots' inner solve: to 1e-1 or 3e-1, or a fixed k = 0, 1, 2, 3 iterations (k = 0: S_z^-1 alone, the
  averaged sweep);
- the depth coupling of the averaged factors: only between planes at most b = 1, 2, 4 apart, a block-banded
  factorisation, O(L b^2) to build and O(L b) to apply against O(L^3) and O(L^2);
- both: b = 2 with k = 1.

Against the march (``DownUp``, one pass, inner 1e-2) and block-Jacobi. Each cost is recorded with what
shares it: the couplings (``InterfaceCouplings``, corrections as A0 + omega^2 A1) every frequency; the
operator every contrast and source at one frequency; the preconditioner's build every source at one
frequency and contrast; the iteration nothing. The iteration is split into the outer products (their count
times one stored product) and the preconditioner's applications. Since a product with the matrix-free
operator costs seconds where a stored one costs a fraction of a second, each solve is also projected onto
it: outer products at the matrix-free product's measured time (``interface_matrix_free_downup.jsonl``, the
same box), plus the applications and the build as measured here. The projection assumes the factors could be
had at the same cost there; for the march the measured matrix-free solves are in that file.

Set-up as S6: split, C-infinity window from Q0 / 4, radius 16 h, depth radius 4 h; media A (5, 3, 2.5) over
B (6.5, 3.6, 2.8), h = 0.05 km, p = 1. Resumable (rows already in the output are skipped). Writes
scripts/data/interface_dtn_approximations.jsonl.

Run:  PYTHONPATH=. python scripts/measure_interface_dtn_approximations.py 8x6 16x6 8x12
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
DATA = Path(__file__).parent / "data"
OUT = DATA / "interface_dtn_approximations.jsonl"
MATRIX_FREE = DATA / "interface_matrix_free_downup.jsonl"
OMEGA = A.beta / (2 * H)
PATTERNS = ("50%", "random 0-50%")
#: (label, preconditioner, DtNSweep keywords); the DtN rows are built here and passed as ``precondition``
RUNS = [
    ("block-Jacobi", "jacobi", None),
    ("march", "downup", None),
    ("DtN 1e-2", "dtn", {"inner_tol": 1e-2}),
    ("DtN 1e-1", "dtn", {"inner_tol": 1e-1}),
    ("DtN 3e-1", "dtn", {"inner_tol": 3e-1}),
    ("DtN k=0", "dtn", {"inner_iterations": 0}),
    ("DtN k=1", "dtn", {"inner_iterations": 1}),
    ("DtN k=2", "dtn", {"inner_iterations": 2}),
    ("DtN k=3", "dtn", {"inner_iterations": 3}),
    ("DtN b=1", "dtn", {"inner_tol": 1e-2, "depth_band": 1}),
    ("DtN b=2", "dtn", {"inner_tol": 1e-2, "depth_band": 2}),
    ("DtN b=4", "dtn", {"inner_tol": 1e-2, "depth_band": 4}),
    ("DtN b=2, k=1", "dtn", {"inner_iterations": 1, "depth_band": 2}),
]


def contrast_factors(pattern: str, n_cells: int) -> np.ndarray:
    """As ``measure_interface_dtn.py``."""
    if pattern == "random 0-50%":
        return np.random.default_rng(11).uniform(0.0, 0.5, n_cells)
    return np.full(n_cells, float(pattern.rstrip("%")) / 100)


def matrix_free_product_s(n_layers: int, n: int) -> float | None:
    """The matrix-free product's measured time on this box, if measured."""
    if not MATRIX_FREE.exists():
        return None
    for r in map(json.loads, MATRIX_FREE.read_text().splitlines()):
        if (r["layers"], r["n"]) == (n_layers, n):
            return r["product_s"]
    return None


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
        t_mf = matrix_free_product_s(n_layers, n)
        print(
            f"{spec}: couplings {t_couplings:.0f} s (no frequency), operator {t_operator:.1f} s (per"
            f" frequency), one product {1e3 * t_product:.1f} ms (matrix-free {t_mf} s)",
            flush=True,
        )
        kx, ky = 0.3 * OMEGA / A.alpha, 0.1 * OMEGA / A.alpha
        for pattern, (label, pre, kwargs) in todo:
            f = contrast_factors(pattern, len(cells))
            ctr = [
                MaterialContrast(fm * m.lam, fm * m.mu, fm * m.rho)
                for fm, m in zip(f, (grid.medium(i) for i in range(len(cells))), strict=True)
            ]
            t0 = time.time()
            built = None
            if kwargs is not None:
                e = idu.source_expansions(op, iso.cell_contrasts(grid, OMEGA, ctr, 0))
                built = idu.DtNSweep(op, e, **kwargs)
            t_built = time.time() - t0
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
                precondition=built,
                tol=1e-10,
                inner_tol=1e-2,
                max_iter=3000,
            )
            wall = time.time() - t0
            build_s = t_built + wall - st["seconds"]
            products_s = st["outer"] * t_product
            apply_s = st["seconds"] - products_s
            row = {
                "layers": n_layers,
                "n": n,
                "freq": "1/2",
                "pattern": pattern,
                "run": label,
                "dtn": kwargs,
                "outer": st["outer"],
                "inner": st["inner"],
                "planes": st["planes"],
                "total_s": build_s + st["seconds"],
                "build_s": build_s,
                "iteration_s": st["seconds"],
                "products_s": products_s,
                "apply_s": apply_s,
                "apply_s_each": apply_s / st["outer"],
                "product_s": t_product,
                "couplings_s": t_couplings,
                "operator_s": t_operator,
                "matrix_free_product_s": t_mf,
                "projected_matrix_free_s": None if t_mf is None else st["outer"] * t_mf + apply_s + build_s,
            }
            with OUT.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            proj = row["projected_matrix_free_s"]
            print(
                f"  {pattern}, {label}: {st['outer']} outer, {st['inner']} inner; build {build_s:.1f} s,"
                f" products {products_s:.1f} s, applications {apply_s:.1f} s"
                f" ({1e3 * row['apply_s_each']:.0f} ms each); matrix-free projected"
                f" {float('nan') if proj is None else proj:.0f} s",
                flush=True,
            )


if __name__ == "__main__":
    main()
