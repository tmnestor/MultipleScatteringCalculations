"""Measure: one preconditioner for a band of frequencies, and the neighbouring frequency's solution as x0.

Plan: docs/2026-10-15-interface-solver-scaling-plan.md, S6c.

A time-domain record of length T up to f_max takes ~T f_max frequencies, Delta f = 1 / T apart, and every
source at each. FGMRES accepts an inexact preconditioner, so a preconditioner built at omega_0 may serve the
frequencies near it, its build then paid once per band; and the solution at a neighbouring frequency is a
starting guess. Here, on the stored kernels (``LateralOperator(parts=True)``), the Dirichlet-to-Neumann form
(``DtNSweep``, window 0), the sweep on the true system from the same factorisation (``TrueSweep`` by
``DtNSweep.sharing``: a window of two planes with one inner iteration, and the whole window to 1e-2) and the
march (``DownUp``, one pass), inner solves to 1e-2, are built at omega_0 h / beta_A = 1/2 and used at omega
= omega_0 (1 - delta): each solve to 1e-10 with the preconditioner built at omega ("fresh"), with the one
from omega_0 ("reused"), and with the solution at omega_0 as x0 ("warm"). Every cell at 50 % of its medium's
lambda, mu and rho, or at a fraction uniform on [0, 50] % (seed 11); a P wave with lateral slowness (0.3,
0.1) / alpha_A. Set-up as S6: split, C-infinity window from Q0 / 4, radius 16 h, depth radius 4 h; media A
(5, 3, 2.5) over B (6.5, 3.6, 2.8), h = 0.05 km, p = 1. Recorded per solve: outer and inner products, the
preconditioner's build and the iteration's seconds, and the solution's difference from the fresh DtN
solve's; per frequency, the operator's build; the process's peak memory. Resumable (rows already in the
output are skipped). Writes scripts/data/interface_frequency_reuse.jsonl.

Run:  PYTHONPATH=. python scripts/measure_interface_frequency_reuse.py 8x6 16x6
"""

import json
import resource
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
OUT = Path(__file__).parent / "data" / "interface_frequency_reuse.jsonl"
OMEGA0 = A.beta / (2 * H)
DELTAS = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
PATTERNS = ("50%", "random 0-50%")
INNER_TOL = 1e-2
#: (label, kind, built at omega_0, x0 from omega_0)
RUNS = [
    ("dtn fresh", "dtn", False, False),
    ("dtn reused", "dtn", True, False),
    ("dtn reused, warm", "dtn", True, True),
    ("march fresh", "downup", False, False),
    ("march reused", "downup", True, False),
    ("march fresh, warm", "downup", False, True),
    ("march reused, warm", "downup", True, True),
    ("true w=2 k=1 fresh", "true w=2 k=1", False, False),
    ("true w=2 k=1 reused", "true w=2 k=1", True, False),
    ("true all fresh", "true all", False, False),
    ("true all reused", "true all", True, False),
]
#: the sweep on the true system, from the DtN form's averaged factorisation (``DtNSweep.sharing``): the two
#: forms that led the approximations study (``measure_interface_dtn_approximations.py``)
TRUE = {
    "true w=2 k=1": {"window": 2, "inner_iterations": 1},
    "true all": {"window": None, "inner_tol": INNER_TOL},
}


def contrast_factors(pattern: str, n_cells: int) -> np.ndarray:
    """As ``measure_interface_dtn.py``."""
    if pattern == "random 0-50%":
        return np.random.default_rng(11).uniform(0.0, 0.5, n_cells)
    return np.full(n_cells, float(pattern.rstrip("%")) / 100)


def peak_gb() -> float:
    """The process's peak resident memory (macOS reports bytes)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def solve(c, omega, ctr, op, pre_name, precondition, x0):
    kx, ky = 0.3 * omega / A.alpha, 0.1 * omega / A.alpha
    t0 = time.time()
    res, st = idu.solve_interface_fgmres(
        c,
        omega,
        ctr,
        kx,
        ky,
        "P",
        operator=op,
        preconditioner=pre_name,
        precondition=precondition,
        x0=x0,
        tol=1e-10,
        inner_tol=INNER_TOL,
        max_iter=3000,
    )
    return res, st, time.time() - t0


def main() -> None:
    done = set()
    if OUT.exists():
        rows = map(json.loads, OUT.read_text().splitlines())
        done = {(r["layers"], r["n"], r["pattern"], r["delta"], r["run"]) for r in rows}
    for spec in sys.argv[1:]:
        n_layers, n = map(int, spec.split("x"))
        todo = {
            (p, d): [run for run in RUNS if (n_layers, n, p, d, run[0]) not in done]
            for p in PATTERNS
            for d in DELTAS
        }
        if not any(todo.values()):
            continue
        layers = tuple(range(-n_layers // 2, n_layers // 2))
        cells = [(iz, ix, iy) for iz in layers for ix in range(n) for iy in range(n)]
        grid = iso.InterfaceGrid(np.array(cells), H, A, B)
        t0 = time.time()
        c = iso.InterfaceCouplings(grid, window=isp.Window(0.25), radius=16.0, depth_radius=4.0, split=True)
        print(f"{spec}: couplings {time.time() - t0:.0f} s (no frequency)", flush=True)
        t0 = time.time()
        op0 = iso.LateralOperator(c, OMEGA0, p=1, r=0, parts=True)
        t_op0 = time.time() - t0
        # per pattern: its contrast, the preconditioners built at omega_0 and the solution there (x0 of the
        # warm solves); kept for every frequency, so that each frequency's operator serves both patterns
        at0 = {}
        for pattern in PATTERNS:
            if not any(todo[pattern, d] for d in DELTAS):
                continue
            f = contrast_factors(pattern, len(cells))
            ctr = [
                MaterialContrast(fm * m.lam, fm * m.mu, fm * m.rho)
                for fm, m in zip(f, (grid.medium(i) for i in range(len(cells))), strict=True)
            ]
            e0 = idu.source_expansions(op0, iso.cell_contrasts(grid, OMEGA0, ctr, 0))
            t0 = time.time()
            built0 = {"dtn": idu.DtNSweep(op0, e0, inner_tol=INNER_TOL)}
            t_build0 = {"dtn": time.time() - t0}
            for kind, settings in TRUE.items():  # the same factorisation, shared
                built0[kind] = built0["dtn"].sharing(idu.TrueSweep, **settings)
                t_build0[kind] = t_build0["dtn"]
            t0 = time.time()
            built0["downup"] = idu.DownUp(op0, e0, inner_tol=INNER_TOL)
            t_build0["downup"] = time.time() - t0
            psi0, _, _ = solve(c, OMEGA0, ctr, op0, "dtn", built0["dtn"], None)
            at0[pattern] = (ctr, built0, t_build0, psi0.psi)
            print(
                f"  {pattern}: at omega_0 operator {t_op0:.1f} s, DtN build {t_build0['dtn']:.1f} s,"
                f" march build {t_build0['downup']:.1f} s",
                flush=True,
            )
        for delta in DELTAS:
            if not any(todo[p, delta] for p in at0):
                continue
            omega = OMEGA0 * (1.0 - delta)
            t0 = time.time()
            op = op0 if delta == 0 else iso.LateralOperator(c, omega, p=1, r=0, parts=True)
            t_op = 0.0 if delta == 0 else time.time() - t0
            for pattern, (ctr, built0, t_build0, psi0) in at0.items():
                runs = todo[pattern, delta]
                if not runs:
                    continue
                # the DtN factorisation at omega, and the reference solution with it
                e = idu.source_expansions(op, iso.cell_contrasts(grid, omega, ctr, 0))
                t0 = time.time()
                dtn = idu.DtNSweep(op, e, inner_tol=INNER_TOL)
                t_fresh = time.time() - t0
                ref_res, ref_st, ref_wall = solve(c, omega, ctr, op, "dtn", dtn.sharing(), None)
                reference = ref_res.psi
                for label, kind, reuse, warm in runs:
                    if label == "dtn fresh":  # the reference solve itself
                        res, st, wall, built_s = ref_res, ref_st, ref_wall, t_fresh
                    else:
                        if reuse:
                            pre = built0[kind]
                        elif kind in TRUE:
                            pre = dtn.sharing(idu.TrueSweep, **TRUE[kind])
                        else:
                            pre = None  # the march, built in the solve
                        res, st, wall = solve(
                            c,
                            omega,
                            ctr,
                            op,
                            "downup" if kind == "downup" else "dtn",
                            pre,
                            psi0 if warm else None,
                        )
                        built_s = t_fresh if (kind in TRUE and not reuse) else 0.0
                    row = {
                        "layers": n_layers,
                        "n": n,
                        "freq": "1/2",
                        "pattern": pattern,
                        "delta": delta,
                        "run": label,
                        "inner_tol": INNER_TOL,
                        "outer": st["outer"],
                        "inner": st["inner"],
                        "planes": st["planes"],
                        "total_s": built_s + wall,
                        "build_s": built_s + wall - st["seconds"],
                        "iteration_s": st["seconds"],
                        "operator_s": t_op,
                        "build0_s": t_build0[kind] if reuse else None,
                        "peak_GB": peak_gb(),
                    }
                    if reference is not None and label != "dtn fresh":
                        row["solution_diff"] = float(
                            np.abs(res.psi - reference).max() / np.abs(reference).max()
                        )
                    with OUT.open("a") as fh:
                        fh.write(json.dumps(row) + "\n")
                    print(
                        f"    delta {delta:.3f}, {label}: {st['outer']} outer, {st['inner']} inner,"
                        f" build {row['build_s']:.1f} s + iteration {st['seconds']:.1f} s;"
                        f" from fresh DtN {row.get('solution_diff', float('nan')):.1e};"
                        f" peak {row['peak_GB']:.1f} GB",
                        flush=True,
                    )
                del dtn
            del op


if __name__ == "__main__":
    main()
