"""Measure: the depth locality of the frequency-independent corrections A0, omega^2 A1 of the interface split.

Plan: docs/2026-10-15-interface-solver-scaling-plan.md, S2.

Beyond a depth radius the set-up could drop A0 and A1, as it does beyond the lateral radius, and keep the windowed
plane-wave part D alone; the error is then A0 + omega^2 A1.  With the window chi, A0 = int chi Y0 over all q, and chi
rises from start * Q0, so the expected decay is ~ e^{-start Q0 g}, g the gap between the two cells and the interface
in total (g = d_r + d_s; 0 for two touching cells): at Q0 h = 10, start 0.25, e^{-2.5 g / h}.

Pairs at lateral offset 0 (the slowest laterally), cell centres z = -1, -3, -5, -7 (in h; A) and +1 ... +7 (B):
reflected in A (both cells in A), reflected in B, transmitted (receiver in B, source in A).  The largest entry of
|A0|, omega^2 |A1| and |A0 + omega^2 A1|, relative to the largest entry of the touching static block K0 of the same
family (the self-image for reflection, the cells across the interface for transmission), at Q0 h = 10, the
C-infinity window from Q0 / 4, omega h / beta_A = 1/6, n_source 10, n_test 4.

The depth radius at a tolerance is the least total gap g beyond which every pair is below it.

Writes scripts/data/interface_depth_locality.json.

Run:  PYTHONPATH=. python scripts/measure_interface_depth_locality.py
"""

import itertools
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
OMEGA = A.beta / (6 * H)
Q0H = 10.0
WINDOW = isp.Window(0.25)
N_SOURCE, N_TEST = 10, 4
DEPTHS = (1, 3, 5, 7)
TOLS = (1e-6, 1e-8, 1e-10)
OUT = Path(__file__).parent / "data" / "interface_depth_locality.json"

FAMILIES = {
    "reflected A": [(-zr, -zs) for zr, zs in itertools.product(DEPTHS, repeat=2)],
    "reflected B": [(zr, zs) for zr, zs in itertools.product(DEPTHS, repeat=2)],
    "transmitted": [(zr, -zs) for zr, zs in itertools.product(DEPTHS, repeat=2)],
}


def gap(zr: int, zs: int) -> int:
    """The two cells' distances from the interface, in h, summed: |z| - 1 each."""
    return abs(zr) - 1 + abs(zs) - 1


def _parts(pair):
    zr, zs = pair
    return isp.frequency_independent_parts(
        (zr, 0, 0), (zs, 0, 0), H, A, B, N_SOURCE, N_TEST, Q0H, window=WINDOW
    )


def _k0_max(pair) -> float:
    """The largest entry of the touching static block K0 of a family (a source in B mirrored, as the split)."""
    zr, zs = pair
    if zs > 0:
        return float(
            np.abs(
                isp.image_block((-zr, 0, 0), (-zs, 0, 0), H, B, A, N_SOURCE, N_TEST)
            ).max()
        )
    return float(
        np.abs(isp.image_block((zr, 0, 0), (zs, 0, 0), H, A, B, N_SOURCE, N_TEST)).max()
    )


def main() -> None:
    pairs = sorted({p for ps in FAMILIES.values() for p in ps})
    touching = {"reflected A": (-1, -1), "reflected B": (1, 1), "transmitted": (1, -1)}
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 1, mp_context=ctx) as ex:
        parts = dict(zip(pairs, ex.map(_parts, pairs), strict=True))
        norm = dict(zip(touching, ex.map(_k0_max, touching.values()), strict=True))
    rows = []
    for fam, ps in FAMILIES.items():
        print(f"\n{fam}: relative to max |K0| of {touching[fam]} = {norm[fam]:.3e}")
        print(
            f"  {'z_r':>4} {'z_s':>4} {'gap/h':>5} {'|A0|':>10} {'w2|A1|':>10} {'|A0+w2A1|':>10}"
        )
        for zr, zs in sorted(ps, key=lambda p: (gap(*p), p)):
            a0, a1 = parts[zr, zs]
            r0 = float(np.abs(a0).max()) / norm[fam]
            r1 = float(np.abs(OMEGA**2 * a1).max()) / norm[fam]
            rt = float(np.abs(a0 + OMEGA**2 * a1).max()) / norm[fam]
            rows.append(
                {
                    "family": fam,
                    "zr": zr,
                    "zs": zs,
                    "gap": gap(zr, zs),
                    "A0": r0,
                    "w2A1": r1,
                    "total": rt,
                }
            )
            print(
                f"  {zr:>4} {zs:>4} {gap(zr, zs):>5} {r0:>10.1e} {r1:>10.1e} {rt:>10.1e}"
            )
    print(
        "\ndepth radius: the least total gap g/h beyond which every pair is below the tolerance"
    )
    radii = {}
    for tol in TOLS:
        worst = {
            g: max(r["total"] for r in rows if r["gap"] == g)
            for g in sorted({r["gap"] for r in rows})
        }
        above = [g for g, v in worst.items() if v >= tol]
        radii[tol] = (
            None
            if above and max(above) == max(worst)
            else (max(above) + 2 if above else 0)
        )
        print(
            f"  {tol:.0e}: "
            + (
                f"g >= {radii[tol]} h"
                if radii[tol] is not None
                else "not reached within the depths measured"
            )
        )
    OUT.write_text(
        json.dumps(
            {
                "q0h": Q0H,
                "window_start": WINDOW.start,
                "omega_h_over_beta": OMEGA * H / A.beta,
                "norm": norm,
                "rows": rows,
                "radius": {str(k): v for k, v in radii.items()},
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
