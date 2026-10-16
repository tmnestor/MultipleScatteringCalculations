"""Profile: where the time of the interface set-up goes, per class, against lateral offset and depth pair.

Plan: docs/2026-10-15-interface-solver-scaling-plan.md, S1.

The box of scripts/gate_interface_gmres.py, 2 x 12 x 12 cells (iz = -1, 0), with its set-up (C-infinity window from
Q0 / 4, radius 16 h): the same jobs as ``InterfaceCouplings``, each timed in its worker and split into
  K0      image_block, order 0 (the closed-form static block)
  Y0<Q0   _static_integral at top level (the plane-wave static integral below Q0 of A0)
  K1      image_block, order 1 (the closed-form omega^2 block)
  FP Y1   fp_low_integral (the finite part of the omega^2 integral below Q0 of A1)
  window  window_parts (the ramp integrals)
  lateral lateral_harmonics (the S_n(q) table of the lateral offset)
Each piece is timed exclusively (a timed function called inside another counts toward the outer one).  The
first job of each worker also pays the process's one-off caches; those are reported separately.

Writes scripts/data/profile_interface_setup.json (one row per class).

Run:  PYTHONPATH=. python scripts/profile_interface_setup.py [n]      (n = 12 by default)
"""

import functools
import itertools
import json
import multiprocessing
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from cubic_scattering import interface_solver as iso
from cubic_scattering import interface_spectral as isp
from cubic_scattering.effective_contrasts import ReferenceMedium

A = ReferenceMedium(5.0, 3.0, 2.5)
B = ReferenceMedium(6.5, 3.6, 2.8)
H = 0.05
RADIUS = 16.0
PIECES = ("K0", "Y0<Q0", "K1", "FP Y1", "window", "lateral")
OUT = Path(__file__).parent / "data" / "profile_interface_setup.json"

_acc: dict = defaultdict(float)
_depth = [0]


def _timed(name_of, fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if _depth[0]:
            return fn(*args, **kwargs)
        _depth[0] += 1
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            _acc[name_of(args, kwargs)] += time.perf_counter() - t0
            _depth[0] -= 1

    return wrapper


def _instrument() -> None:
    if getattr(isp, "_profiled", False):
        return
    isp.image_block = _timed(
        lambda a, k: "K1" if k.get("order", 0) == 1 else "K0", isp.image_block
    )
    isp._static_integral = _timed(lambda a, k: "Y0<Q0", isp._static_integral)
    isp.fp_low_integral = _timed(lambda a, k: "FP Y1", isp.fp_low_integral)
    isp.window_parts = _timed(lambda a, k: "window", isp.window_parts)
    isp.lateral_harmonics = _timed(lambda a, k: "lateral", isp.lateral_harmonics)
    isp._profiled = True


def _job(args):
    _instrument()
    _acc.clear()
    t0, c0 = time.perf_counter(), time.process_time()
    iso._parts_job(args)
    return (
        os.getpid(),
        time.perf_counter() - t0,
        time.process_time() - c0,
        {p: _acc.get(p, 0.0) for p in PIECES},
    )


def jobs_of(n: int) -> list:
    """The jobs ``InterfaceCouplings`` runs for the 2 x n x n box: keys (zr, zs, a, b) and arguments."""
    cells = [(iz, ix, iy) for iz in (-1, 0) for ix in range(n) for iy in range(n)]
    g = iso.InterfaceGrid(np.array(cells), H, A, B)
    u = g.units
    keys = set()
    for m, k in itertools.product(range(len(u)), repeat=2):
        (a, b), _ = iso._canonical(int(u[m, 1] - u[k, 1]), int(u[m, 2] - u[k, 2]))
        keys.add((int(u[m, 0]), int(u[k, 0]), a, b))
    return [
        (
            key,
            (
                (key[0], key[2], key[3]),
                (key[1], 0, 0),
                g.h,
                g.above,
                g.below,
                10,  # n_source
                4,  # n_test
                10.0,  # q0h
                160.0,  # q1h
                isp.Window(0.25),
                False,  # omega4
                bool(np.hypot(key[2], key[3]) <= RADIUS),
            ),
        )
        for key in sorted(keys)
    ]


def _table(title: str, groups: dict) -> None:
    print(f"\n{title}")
    print(f"  {'':>14} {'classes':>7} {'wall s':>8} " + " ".join(f"{p:>8}" for p in PIECES))
    for label, rows in groups.items():
        tot = {p: sum(r["pieces"][p] for r in rows) for p in PIECES}
        print(
            f"  {label:>14} {len(rows):>7} {sum(r['wall'] for r in rows):>8.0f} "
            + " ".join(f"{tot[p]:>8.0f}" for p in PIECES)
        )


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    jobs = jobs_of(n)
    workers = os.cpu_count() or 1
    print(f"2 x {n} x {n}: {len(jobs)} classes, {workers} workers", flush=True)
    t0 = time.perf_counter()
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        res = list(ex.map(_job, [a for _, a in jobs]))
    wall = time.perf_counter() - t0
    seen: set = set()
    rows = []
    for (key, args), (pid, w, cpu, pieces) in zip(jobs, res, strict=True):
        rows.append(
            {
                "key": list(key),
                "reach": float(np.hypot(key[2], key[3])),
                "local": args[-1],
                "first_in_worker": pid not in seen,
                "wall": w,
                "cpu": cpu,
                "pieces": pieces,
            }
        )
        seen.add(pid)
    OUT.write_text(json.dumps({"n": n, "wall": wall, "workers": workers, "rows": rows}, indent=1))

    warm = [r for r in rows if not r["first_in_worker"]]
    first = [r for r in rows if r["first_in_worker"]]
    print(f"set-up wall {wall:.0f} s; sum over jobs {sum(r['wall'] for r in rows):.0f} s "
          f"(cpu {sum(r['cpu'] for r in rows):.0f} s); first job per worker {sum(r['wall'] for r in first):.0f} s")
    _table("by radius (all jobs)", {
        "within 16 h": [r for r in rows if r["local"]],
        "beyond 16 h": [r for r in rows if not r["local"]],
    })
    pair = {(-1, -1): "A <- A (refl)", (1, 1): "B <- B (refl)", (-1, 1): "A <- B (trans)", (1, -1): "B <- A (trans)"}
    _table("by depth pair (warm jobs)", {
        lbl: [r for r in warm if tuple(r["key"][:2]) == zz] for zz, lbl in pair.items()
    })
    bins = [(-1, 4), (4, 8), (8, 12), (12, 16), (16, 24), (24, 32)]
    _table("by lateral reach |d|/h (warm jobs)", {
        f"({max(lo, 0)}, {hi}]": [r for r in warm if lo < r["reach"] <= hi] for lo, hi in bins
    })
    _table("first job of each worker", {"first": first})


if __name__ == "__main__":
    main()
