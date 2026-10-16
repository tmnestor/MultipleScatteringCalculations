"""Measure: the cost of each frequency with D split (S4) and unsplit, on one machine, in one process.

Plan: docs/2026-10-15-interface-solver-scaling-plan.md, S4.

The kernels of one frequency (``LateralOperator``) are the interface blocks (D and the corrections) plus Paper 2's
whole-space blocks.  The first frequency in a process also pays the whole-space blocks' frequency-independent
coefficients (``far_series_coefficients``, cached per process), so a sweep's cost is that of a further frequency.
This times, for the box of scripts/gate_interface_gmres.py (2 x 12 x 12 cells, radius 16 h), the set-up and two
frequencies (omega h / beta_A = 1/6, then 10 % higher), unsplit and split, and the interface blocks alone.

Run:  PYTHONPATH=. python scripts/measure_interface_split_cost.py
"""

import time

import numpy as np

from cubic_scattering import interface_solver as iso
from cubic_scattering import interface_spectral as isp
from cubic_scattering.effective_contrasts import ReferenceMedium

A = ReferenceMedium(5.0, 3.0, 2.5)
B = ReferenceMedium(6.5, 3.6, 2.8)
H = 0.05
OMEGAS = (A.beta / (6 * H), 1.1 * A.beta / (6 * H))


def main() -> None:
    cells = [(iz, ix, iy) for iz in (-1, 0) for ix in range(12) for iy in range(12)]
    grid = iso.InterfaceGrid(np.array(cells), H, A, B)
    for split in (False, True):
        t0 = time.time()
        c = iso.InterfaceCouplings(
            grid, window=isp.Window(0.25), radius=16.0, split=split
        )
        line = f"{'split' if split else 'unsplit'}: set-up {time.time() - t0:.0f} s"
        for i, omega in enumerate(OMEGAS):
            t0 = time.time()
            c.interface_canonical(omega)
            t_d = time.time() - t0
            t0 = time.time()
            iso.LateralOperator(c, omega, p=1, r=0)
            t_k = time.time() - t0
            line += f"; frequency {i + 1}: interface blocks {t_d:.0f} s, all kernels {t_k:.0f} s"
        print(line, flush=True)


if __name__ == "__main__":
    main()
