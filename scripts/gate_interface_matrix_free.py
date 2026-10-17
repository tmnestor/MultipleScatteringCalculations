"""Gate: the coupling product with no kernel stored per pair of depth layers, against the stored-kernel product.

Plan: docs/2026-10-15-interface-solver-scaling-plan.md, S5.

``MatrixFreeOperator`` applies D through the plane-wave modes (the low band by NUFFTs over its polar quadrature, the
high band at the periodic box's wavenumbers), the corrections as stencils and Paper 2's whole-space blocks by a 3-D
FFT on each side; ``LateralOperator`` stores the transformed blocks of every pair of depth layers.  Both from the
same set-up (split, C-infinity window from Q0 / 4, radius 16 h, depth radius 4 h), the same random source vector:
the largest difference of the products relative to the largest entry, and the time of each operator's set-up for
the frequency and of one product.

Boxes 2 x 5 x 5 (50 cells), 6 x 4 x 4 (96, three layers each side: the whole-space march) and 2 x 12 x 12 (288)
straddling the interface; media A (5, 3, 2.5) over B
(6.5, 3.6, 2.8), h = 0.05 km, omega h / beta_A = 1/6, p = 1.  PASS below 1e-10 (the high band's mode cancellation
leaves ~1e-11 of a block at 1/6; D1 is 1e-8).

Measured 16 October 2026 (four workers): 50 cells, products 7.3e-13 apart, stored kernels 54 s + 0.01 s per
product, matrix-free 30 s + 11.6 s; 288 cells, 2.1e-12, stored 53 s + 0.04 s, matrix-free 38 s + 15.9 s; peak memory
of the process (both operators) 2.6 and 5.7 GB.  With the whole-space march (planes two or more apart; stored kernels
for planes 0 and 1 apart): 50 cells 7.3e-13 again; 96 cells in six layers 3.6e-11, stored 149 s + 0.03 s, matrix-free
179 s + 39.4 s, peak 12.8 GB (the cached factors).  With planes up to 2 apart stored (the march from gap 3, to
q h = 8) and NUFFT eps 1e-11: 50 cells 8.3e-13; 96 cells 7.0e-13, matrix-free 22 s + 24.7 s, peak 4.2 GB.
With the lattice nodes by FFT (17 October): 50 cells 6.3e-13, matrix-free 11 s + 4.0 s; 96 cells 6.3e-13,
21 s + 9.3 s, peak 2.7 GB.

Run:  PYTHONPATH=. python scripts/gate_interface_matrix_free.py [box indices, default all]
"""

import resource
import sys
import time

import numpy as np

from cubic_scattering import interface_solver as iso
from cubic_scattering import interface_spectral as isp
from cubic_scattering.effective_contrasts import ReferenceMedium

A = ReferenceMedium(5.0, 3.0, 2.5)
B = ReferenceMedium(6.5, 3.6, 2.8)
H = 0.05
OMEGA = A.beta / (6 * H)
TOL = 1e-10
#: (layers, cells per side): two layers (no march), six layers (the whole-space march between planes two or more
#: apart), and the 288-cell box.
BOXES = [((-1, 0), 5), (tuple(range(-3, 3)), 4), ((-1, 0), 12)]


def main() -> None:
    ok = True
    picked = [int(a) for a in sys.argv[1:]] or range(len(BOXES))
    for layers, n in (BOXES[i] for i in picked):
        cells = [(iz, ix, iy) for iz in layers for ix in range(n) for iy in range(n)]
        grid = iso.InterfaceGrid(np.array(cells), H, A, B)
        t0 = time.time()
        c = iso.InterfaceCouplings(
            grid, window=isp.Window(0.25), radius=16.0, depth_radius=4.0, split=True
        )
        t_setup = time.time() - t0
        t0 = time.time()
        lat = iso.LateralOperator(c, OMEGA, p=1, r=0)
        t_lat = time.time() - t0
        t0 = time.time()
        mf = iso.MatrixFreeOperator(c, OMEGA, p=1, r=0)
        t_mf = time.time() - t0
        rng = np.random.default_rng(7)
        src = rng.standard_normal(
            (len(cells), lat.n_source * 9)
        ) + 1j * rng.standard_normal((len(cells), lat.n_source * 9))
        t0 = time.time()
        ref = lat._convolve(src)
        p_lat = time.time() - t0
        t0 = time.time()
        got = mf._convolve(src)
        p_mf = time.time() - t0
        err = np.abs(got - ref).max() / np.abs(ref).max()
        good = err < TOL
        ok &= good
        print(
            f"{len(cells)} cells, {len(layers)} layers (set-up {t_setup:.0f} s): products differ by {err:.1e}  {'PASS' if good else 'FAIL'};"
            f" stored kernels {t_lat:.0f} s + {p_lat:.2f} s per product, matrix-free {t_mf:.0f} s + {p_mf:.2f} s;"
            f" peak memory {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB",
            flush=True,
        )
    print("ALL PASS" if ok else "SOME FAILED")


if __name__ == "__main__":
    main()
