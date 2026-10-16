"""Measure: how the windowed plane-wave part D of an interface block decays with lateral offset.

Plan: docs/2026-10-15-interface-solver-scaling-plan.md, S4 (the risk: wrap-around).

S4 would sample D's spectrum on the FFT grid of a padded lateral box of period P instead of integrating it per offset.
Sampling the spectrum periodises D in space, so each kernel entry gains the images D(Delta + m P), m != 0. The
padding needed for a tolerance is set by how fast D decays laterally: like ~1/r if its propagating part dominates
(the reflected and transmitted waves along the interface), or faster than any power if it does not.

D for the touching cells (the slowest in depth: both at the interface), reflected (both in A) and transmitted,
at lateral offsets 0 ... 128 h along x, against the touching static block K0 (the normalisation of S2 and D1);
Q0 h = 10, the C-infinity window from Q0 / 4, omega h / beta_A = 1/6 and 1/2.

Writes scripts/data/interface_d_decay.json.

Run:  PYTHONPATH=. python scripts/measure_interface_d_decay.py
"""

import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from cubic_scattering import interface_spectral as isp
from cubic_scattering.effective_contrasts import ReferenceMedium

A = ReferenceMedium(5.0, 3.0, 2.5)
B = ReferenceMedium(6.5, 3.6, 2.8)
H = 0.05
Q0H = 10.0
WINDOW = isp.Window(0.25)
N_SOURCE, N_TEST = 10, 4
OFFSETS = (0, 4, 8, 16, 32, 64, 128)  # in h, along x
FREQS = {"1/6": A.beta / (6 * H), "1/2": A.beta / (2 * H)}
PAIRS = {
    "reflected": -1,
    "transmitted": 1,
}  # the receiver's z; the source is at (-1, 0, 0)
OUT = Path(__file__).parent / "data" / "interface_d_decay.json"


def _d(job):
    fam, freq, dx = job
    rec, src = (PAIRS[fam], dx, 0), (-1, 0, 0)
    blk = isp.truncated_blocks(
        [(rec, src)], H, FREQS[freq], A, B, N_SOURCE, N_TEST, Q0H, window=WINDOW
    )[0]
    return float(np.abs(blk).max())


def main() -> None:
    k0 = {
        fam: float(
            np.abs(
                isp.image_block((z, 0, 0), (-1, 0, 0), H, A, B, N_SOURCE, N_TEST)
            ).max()
        )
        for fam, z in PAIRS.items()
    }
    # the farthest first: a table costs ~ reach^2
    jobs = sorted(
        [(f, w, dx) for f in PAIRS for w in FREQS for dx in OFFSETS],
        key=lambda j: -j[2],
    )
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 1, mp_context=ctx) as ex:
        got = dict(zip(jobs, ex.map(_d, jobs), strict=True))
    rows = []
    for fam in PAIRS:
        for freq in FREQS:
            print(
                f"\n{fam}, omega h / beta_A = {freq}: max |D| relative to max |K0| of the touching pair"
            )
            rel = [got[fam, freq, dx] / k0[fam] for dx in OFFSETS]
            for dx, r in zip(OFFSETS, rel, strict=True):
                rows.append(
                    {"family": fam, "omega_h_over_beta": freq, "offset": dx, "D": r}
                )
                print(f"  {dx:>4} h  {r:.2e}")
            far = [(dx, r) for dx, r in zip(OFFSETS, rel, strict=True) if dx >= 16]
            slope = np.polyfit(
                np.log([d for d, _ in far]), np.log([r for _, r in far]), 1
            )[0]
            print(f"  log-log slope beyond 16 h: {slope:.2f}")
    OUT.write_text(
        json.dumps(
            {"q0h": Q0H, "window_start": WINDOW.start, "k0": k0, "rows": rows}, indent=1
        )
    )


if __name__ == "__main__":
    main()
