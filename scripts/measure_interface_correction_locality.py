"""Measure: how fast the frequency-independent corrections A0, A1 of the interface blocks decay with the
cells' lateral offset, with the sharp cutoff at Q0 and with smooth windows.

Paper: LatexPDFs/InterfaceCouplingIntegrals, Section "The next term: the order omega^2".
Plan: docs/2026-10-10-stratified-reference-legendre-cells-3d.md, step 1.

K(omega) = D(omega) + A0 + omega^2 A1 (``cubic_scattering.interface_spectral``).  A0 and A1 are high-pass parts
of the plane-wave integral.  With the sharp cutoff theta(q - Q0) they ring in space (Gibbs) and decay slowly
with the lateral offset; with a smooth window chi (0 below start Q0, 1 above Q0) they should decay fast, so
that beyond some radius only D, a low-pass integral shared by every pair, is needed.  The block itself is
the same either way: the windowed parts are the sharp ones plus the ramp integrals int chi Y0, int chi Y1,
which D gives back.

Cells touching the interface, in a row along x: reflected, receiver (-1, d, 0) and source (-1, 0, 0);
transmitted, receiver (1, d, 0) and source (-1, 0, 0); d in units of h (a cell is 2 h wide).  Full blocks
(4 field functions x 10 source monomials x 9 x 9).  Reported, relative to max |K0| of the touching pair
(d = 0), the scale of the blocks near the interface:
    A0      max |A0|
    w2A1    omega^2 max |A1| at omega h / beta = 1/6, the top of the band Q0 h = 10 serves
    K0      max |K0(d)| (closed-form static block), for comparison
and, per window, the radius beyond which the corrections stay below each tolerance.

Results are cached in scripts/data/interface_correction_locality.json; delete it to recompute (about
15 minutes on 4 cores, most of it the sharp parts at the largest offsets).

Run:  PYTHONPATH=. python scripts/measure_interface_correction_locality.py [--workers N]
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from cubic_scattering import interface_spectral as isp
from cubic_scattering.effective_contrasts import ReferenceMedium
from cubic_scattering.interface_image import image_block

A = ReferenceMedium(5.0, 3.0, 2.5)
B = ReferenceMedium(6.5, 3.6, 2.8)
H = 0.05
Q0H, Q1H = 10.0, 160.0
OMEGA = 10.0  # omega h / beta = 1/6
OFFSETS = (0, 2, 4, 6, 8, 12, 16, 24, 32)
WINDOWS = {
    "sharp": None,
    "cinf 0.5": isp.Window(0.5, "cinf"),
    "cinf 0.25": isp.Window(0.25, "cinf"),
    "poly3 0.5": isp.Window(0.5, "poly", 3),
}
TOLS = (1e-6, 1e-8, 1e-10)
CACHE = Path(__file__).resolve().parent / "data" / "interface_correction_locality.json"


def _pair(kind: str, d: int) -> tuple:
    return ((-1, d, 0) if kind == "R" else (1, d, 0)), (-1, 0, 0)


def measure(job: tuple) -> dict:
    """Max |A0|, max |A1| for every window, and max |K0|, of one pair."""
    kind, d = job
    rec, src = _pair(kind, d)
    t0 = time.time()
    a0, a1 = isp.frequency_independent_parts(rec, src, H, A, B, q0h=Q0H, q1h=Q1H)
    out = {"K0": float(np.abs(image_block(rec, src, H, A, B)).max())}
    for name, w in WINDOWS.items():
        r0, r1 = (
            (0.0, 0.0) if w is None else isp.window_parts(rec, src, H, A, B, w, q0h=Q0H)
        )
        out[name] = [float(np.abs(a0 + r0).max()), float(np.abs(a1 + r1).max())]
    out["seconds"] = time.time() - t0
    print(f"  {kind} d = {d:2d}: {out['seconds']:.0f} s", flush=True)
    return out


def radius(rows: dict, tol: float) -> str:
    """The smallest measured offset from which on every larger one is below tol ('>max' if none)."""
    ok = [d for d in OFFSETS if all(rows[e] < tol for e in OFFSETS if e >= d)]
    return f"{min(ok)} h" if ok else f"> {OFFSETS[-1]} h"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    data = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    jobs = [(k, d) for k in "RT" for d in OFFSETS if f"{k}{d}" not in data]
    if jobs:
        print(f"computing {len(jobs)} pairs on {args.workers} workers", flush=True)
        # largest offsets first: they dominate the wall time
        jobs.sort(key=lambda j: -j[1])
        with ProcessPoolExecutor(args.workers) as ex:
            for job, res in zip(jobs, ex.map(measure, jobs), strict=True):
                data[f"{job[0]}{job[1]}"] = res
        CACHE.write_text(json.dumps(data, indent=1, sort_keys=True))
    for kind in "RT":
        scale = data[f"{kind}0"]["K0"]
        print(
            f"\n{'reflected' if kind == 'R' else 'transmitted'}: relative to max |K0(0)| = {scale:.3e};"
            f" omega h / beta = {OMEGA * H / A.beta:.3f}, Q0 h = {Q0H:g}"
        )
        print(f"{'d/h':>4} {'K0':>8}  " + "  ".join(f"{n:>19}" for n in WINDOWS))
        print(
            f"{'':>4} {'':>8}  " + "  ".join(f"{'A0':>9} {'w2A1':>9}" for _ in WINDOWS)
        )
        corr = {n: {} for n in WINDOWS}
        for d in OFFSETS:
            r = data[f"{kind}{d}"]
            cells = []
            for n in WINDOWS:
                a0, a1 = r[n][0] / scale, OMEGA**2 * r[n][1] / scale
                corr[n][d] = a0 + a1
                cells.append(f"{a0:9.1e} {a1:9.1e}")
            print(f"{d:4d} {r['K0'] / scale:8.1e}  " + "  ".join(cells))
        print("radius beyond which |A0| + w2|A1| < tol:")
        for n in WINDOWS:
            print(
                f"  {n:>10}: "
                + ", ".join(f"{t:.0e}: {radius(corr[n], t)}" for t in TOLS)
            )


if __name__ == "__main__":
    main()
