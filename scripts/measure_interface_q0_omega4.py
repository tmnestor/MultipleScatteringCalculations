"""Measure: how far the omega^4 split lets the cutoff Q0 of the interface blocks come down.

Plan: docs/2026-10-10-stratified-reference-legendre-cells-3d.md (step 2: the per-frequency cost).

The block is K = D(omega; Q0) + A0 + omega^2 A1 [+ omega^4 A2] + E: D the plane-wave integral below Q0 at each
frequency (the per-frequency cost, its radial nodes ~ Q0), E the dropped remainder, ~ omega^4 Q0^-6 with the
omega^2 split and ~ omega^6 Q0^-8 with the omega^4 split.  For each pair and frequency: the block's error
against the omega^4 split at Q0 h = 20, relative to its largest entry, and the time of D (``truncated_blocks``),
for Q0 h = 3 ... 10.

Pairs: the self-image pair (-1, 0, 0) <- (-1, 0, 0) (touching: the slowest decay of E), the pair touching across
the interface (1, 0, 0) <- (-1, 0, 0), and a lateral neighbour (-1, 2, 0) <- (-1, 0, 0).  Media A (5, 3, 2.5)
over B (6.5, 3.6, 2.8), h = 0.05 km, omega h / beta_A = 1/6 and 1/2.  Cached in
scripts/data/interface_q0_omega4.json (the frequency-independent parts are the cost: minutes per pair and Q0).

Measured 15 October 2026: D costs 0.03-0.06 s per pair whatever Q0 h (3 ... 10): its per-frequency work is
fixed (interface solves, frames, the lateral table), not panels in Q0, so lowering Q0 saves no time.  The
omega^4 split buys accuracy instead: worst of the three pairs at omega h / beta = 1/6, Q0 h = 10: omega^2 3.5e-10,
omega^4 4.8e-11 (the rounding floor, reached already at Q0 h = 5); at 1/2, Q0 h = 10: 2.5e-8 against 6.2e-11.

Run:  PYTHONPATH=. python scripts/measure_interface_q0_omega4.py
"""

import itertools
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from cubic_scattering import interface_spectral as isp
from cubic_scattering.effective_contrasts import ReferenceMedium

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "scripts" / "data" / "interface_q0_omega4.json"
A = ReferenceMedium(5.0, 3.0, 2.5)
B = ReferenceMedium(6.5, 3.6, 2.8)
H = 0.05
PAIRS = [((-1, 0, 0), (-1, 0, 0)), ((1, 0, 0), (-1, 0, 0)), ((-1, 2, 0), (-1, 0, 0))]
FREQS = [1 / 6, 1 / 2]  # omega h / beta_A
Q0S = [3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
Q_REF = 20.0


def _parts(args):
    rec, src, q0h, omega4 = args
    return isp.frequency_independent_parts(
        rec, src, H, A, B, 10, 4, q0h, 160.0, omega4=omega4
    )


def main() -> None:
    jobs = [
        (r, s, q, o4) for (r, s), q, o4 in itertools.product(PAIRS, Q0S, (False, True))
    ]
    jobs += [(r, s, Q_REF, True) for r, s in PAIRS]
    with ProcessPoolExecutor(
        os.cpu_count() or 1, mp_context=get_context("spawn")
    ) as ex:
        parts = dict(zip(jobs, ex.map(_parts, jobs), strict=True))
    out = []
    for (rec, src), f in itertools.product(PAIRS, FREQS):
        omega = f * A.beta / H
        off = [(rec[1] - src[1], rec[2] - src[2])]
        ref = isp.interface_block(
            rec,
            src,
            H,
            omega,
            A,
            B,
            10,
            4,
            Q_REF,
            160.0,
            parts[rec, src, Q_REF, True],
            isp.lateral_harmonics(off, H, Q_REF, 10, 4),
        )
        scale = np.abs(ref).max()
        for q0h in Q0S:
            table = isp.lateral_harmonics(off, H, q0h, 10, 4)
            t0 = time.time()
            isp.truncated_blocks([(rec, src)], H, omega, A, B, 10, 4, q0h, table)
            t_d = time.time() - t0
            row = {"pair": [rec, src], "omega_h_beta": f, "q0h": q0h, "t_D": t_d}
            for o4 in (False, True):
                k = isp.interface_block(
                    rec,
                    src,
                    H,
                    omega,
                    A,
                    B,
                    10,
                    4,
                    q0h,
                    160.0,
                    parts[rec, src, q0h, o4],
                    table,
                )
                row["omega4" if o4 else "omega2"] = float(np.abs(k - ref).max() / scale)
            out.append(row)
            print(
                f"{rec} <- {src}  wh/b {f:.3f}  Q0h {q0h:4.0f}: omega^2 split {row['omega2']:.1e}, "
                f"omega^4 split {row['omega4']:.1e}; D {t_d:.2f} s",
                flush=True,
            )
    CACHE.write_text(json.dumps({"reference_q0h": Q_REF, "rows": out}, indent=1))
    print(f"written {CACHE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
