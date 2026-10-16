"""Measure: memory and time of the matrix-free coupling product against the box size and the number of layers.

Plan: docs/2026-10-15-interface-solver-scaling-plan.md, S5 (the rest of its gate) and S7.

For boxes of n x n cells in L layers (L / 2 on each side of the interface), each in a fresh process so that its peak
memory is its own: the set-up (``InterfaceCouplings(split=True, matrix_free=True)``, corrections within small radii,
since their stencils are local and do not set the scaling), then ``MatrixFreeOperator`` with the plane-wave factors
recomputed in each product (``cache=False``) and, where they fit, kept (``cache=True``): the operator's build for
one frequency, one product, the peak memory of the process, and the node counts.  The stored-kernel operator's
memory, L^2 (2n - 1)^2 blocks of (na 9) x (n_source 9) complex, is given for comparison.

Media A (5, 3, 2.5) over B (6.5, 3.6, 2.8), h = 0.05 km, omega h / beta_A = 1/6, p = 1, r = 0.

Resumable: the rows already in the output file are skipped.  Writes scripts/data/interface_matrix_free_scaling.jsonl.

Run:  PYTHONPATH=. python scripts/measure_interface_matrix_free_scaling.py
"""

import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "data" / "interface_matrix_free_scaling.jsonl"
#: (n, L, cache): cached factors only where they fit in memory
CONFIGS = [
    (n, layers, cache)
    for n in (12, 24, 36)
    for layers in (2, 4, 8)
    for cache in (False, True)
]
CACHE_LIMIT = (
    12 * 12 * 8
)  # cached factors measured up to this many cells per layer x layers


def one(n: int, n_layers: int, cache: bool) -> dict:
    from cubic_scattering import interface_solver as iso
    from cubic_scattering import interface_spectral as isp
    from cubic_scattering.effective_contrasts import ReferenceMedium

    a, b, h = ReferenceMedium(5.0, 3.0, 2.5), ReferenceMedium(6.5, 3.6, 2.8), 0.05
    omega = a.beta / (6 * h)
    layers = range(-n_layers // 2, n_layers // 2)
    cells = [(iz, ix, iy) for iz in layers for ix in range(n) for iy in range(n)]
    grid = iso.InterfaceGrid(np.array(cells), h, a, b)
    t0 = time.time()
    c = iso.InterfaceCouplings(
        grid,
        window=isp.Window(0.25),
        radius=2.0,
        depth_radius=2.0,
        split=True,
        matrix_free=True,
    )
    t_setup = time.time() - t0
    t0 = time.time()
    op = iso.MatrixFreeOperator(c, omega, p=1, r=0, cache=cache)
    t_op = time.time() - t0
    src = np.ones((len(cells), op.n_source * 9), dtype=complex)
    t0 = time.time()
    op._convolve(src)
    t_prod = time.time() - t0
    nodes = sum(
        len(ch[0])
        for prod in (op.low, op.high)
        for side in (prod.a, prod.b)
        if side
        for ch in side.chunks
    )
    march = sum(len(ch[0]) for _, _, m in op.sides if m for ch in m.chunks)
    stored = n_layers**2 * (2 * n - 1) ** 2 * (op.na * 9) * (op.n_source * 9) * 16
    return {
        "n": n,
        "L": n_layers,
        "cells": len(cells),
        "cache": cache,
        "setup_s": t_setup,
        "operator_s": t_op,
        "product_s": t_prod,
        "peak_GB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
        "nodes_D": nodes,
        "nodes_march": march,
        "stored_kernels_GB": stored / 1e9,
    }


def main() -> None:
    if len(sys.argv) == 4:  # a child: one configuration
        print(
            json.dumps(one(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3] == "1")),
            flush=True,
        )
        return
    done = set()
    if OUT.exists():
        done = {
            (r["n"], r["L"], r["cache"])
            for r in map(json.loads, OUT.read_text().splitlines())
        }
    for n, n_layers, cache in CONFIGS:
        if (n, n_layers, cache) in done or (cache and n * n * n_layers > CACHE_LIMIT):
            continue
        res = subprocess.run(
            [sys.executable, __file__, str(n), str(n_layers), "1" if cache else "0"],
            check=False,  # a failed child is reported and skipped
            capture_output=True,
            text=True,
        )
        rows = [ln for ln in res.stdout.splitlines() if ln.startswith("{")]
        if not rows:
            print(
                f"n={n} L={n_layers} cache={cache}: failed (exit {res.returncode}) {res.stderr[-300:]}",
                flush=True,
            )
            continue
        with OUT.open("a") as f:
            f.write(rows[-1] + "\n")
        r = json.loads(rows[-1])
        print(
            f"n={n:3d} L={n_layers:2d} cache={cache!s:5}: set-up {r['setup_s']:6.0f} s, operator {r['operator_s']:6.0f} s,"
            f" product {r['product_s']:7.1f} s, peak {r['peak_GB']:5.1f} GB; nodes {r['nodes_D']} + {r['nodes_march']};"
            f" stored kernels would be {r['stored_kernels_GB']:.1f} GB",
            flush=True,
        )


if __name__ == "__main__":
    main()
