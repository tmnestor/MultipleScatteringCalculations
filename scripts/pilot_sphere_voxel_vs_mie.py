#!/usr/bin/env python3
"""Pilot: a voxelised sphere against exact elastic Mie, over a refinement ladder.

The continuum-limit paper establishes the voxel scheme's discretisation error on a layer, where cubes tile
the scatterer exactly. A sphere adds the staircase (shape) error of approximating a curved surface by
cubes. This pilot measures, before any design, how large the sphere's total error is and at what order it
converges, using the validated far-field comparison of ``scripts/gate_sphere_cell_average_vs_mie.py``
(``pattern_error``: max |far field - Mie| / peak over nine angles at 5e4 radii, raw and volume-corrected).

Run small first:  conda run -n seismic python -u scripts/pilot_sphere_voxel_vs_mie.py [--fft] 4 6 8
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import gate_sphere_cell_average_vs_mie as gate  # noqa: E402
from cubic_scattering.sphere_scattering_fft import compute_sphere_foldy_lax_fft  # noqa: E402
from gate_sphere_cell_average_vs_mie import REF, pattern_error  # noqa: E402

KA_S = 0.5  # sphere k_S a
RADIUS = 10.0  # m
OMEGA = KA_S * REF.beta / RADIUS


def main() -> int:
    args = sys.argv[1:]
    if "--fft" in args:
        # the FFT-accelerated solver is a drop-in replacement for the dense one inside pattern_error
        gate.compute_sphere_foldy_lax = compute_sphere_foldy_lax_fft
        args = [a for a in args if a != "--fft"]
    ladder = [int(a) for a in args] or [4]
    print(
        "solver:",
        "FFT + GMRES" if gate.compute_sphere_foldy_lax is compute_sphere_foldy_lax_fft else "dense",
    )
    print(f"sphere k_S a = {KA_S}, radius {RADIUS} m, omega = {OMEGA:.2f} rad/s; the validated contrast")
    print("  n_sub  cells  k_S h(cell)   raw error   vol-corrected   seconds", flush=True)
    rows = []
    for n in ladder:
        t0 = time.perf_counter()
        raw, cor, cells = pattern_error(OMEGA, RADIUS, n, cell_average=True)
        dt = time.perf_counter() - t0
        k_h = OMEGA / REF.beta * RADIUS / n  # half-width of a sub-cell is a / n_sub
        rows.append((n, raw, cor))
        print(f"  {n:5d}  {cells:5d}  {k_h:10.3f}   {raw:.3e}   {cor:.3e}       {dt:7.1f}", flush=True)
    if len(rows) >= 2:
        for (n1, r1, c1), (n2, r2, c2) in zip(rows, rows[1:], strict=False):
            p_raw = np.log(r1 / r2) / np.log(n2 / n1)
            p_cor = np.log(c1 / c2) / np.log(n2 / n1)
            print(f"  apparent order {n1}->{n2}: raw {p_raw:.2f}, volume-corrected {p_cor:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
