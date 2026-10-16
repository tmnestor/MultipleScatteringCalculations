"""Gate: the coupling product with no kernel stored per pair of depth layers, against the stored-kernel product.

Plan: docs/2026-10-15-interface-solver-scaling-plan.md, S5.

``MatrixFreeOperator`` applies D through the plane-wave modes (the low band by NUFFTs over its polar quadrature, the
high band at the periodic box's wavenumbers), the corrections as stencils and Paper 2's whole-space blocks by a 3-D
FFT on each side; ``LateralOperator`` stores the transformed blocks of every pair of depth layers.  Both from the
same set-up (split, C-infinity window from Q0 / 4, radius 16 h, depth radius 4 h), the same random source vector:
the largest difference of the products relative to the largest entry, and the time of each operator's set-up for
the frequency and of one product.

Boxes 2 x 5 x 5 (50 cells) and 2 x 12 x 12 (288) straddling the interface; media A (5, 3, 2.5) over B
(6.5, 3.6, 2.8), h = 0.05 km, omega h / beta_A = 1/6, p = 1.  PASS below 1e-10 (the high band's mode cancellation
leaves ~1e-11 of a block at 1/6; D1 is 1e-8).

Measured 16 October 2026 (four workers): 50 cells, products 7.3e-13 apart, stored kernels 54 s + 0.01 s per
product, matrix-free 30 s + 11.6 s; 288 cells, 2.1e-12, stored 53 s + 0.04 s, matrix-free 38 s + 15.9 s; peak memory
of the process (both operators) 2.6 and 5.7 GB.

Run:  PYTHONPATH=. python scripts/gate_interface_matrix_free.py
"""

import resource
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


def main() -> None:
    ok = True
    for n in (5, 12):
        cells = [(iz, ix, iy) for iz in (-1, 0) for ix in range(n) for iy in range(n)]
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
            f"{2 * n * n} cells (set-up {t_setup:.0f} s): products differ by {err:.1e}  {'PASS' if good else 'FAIL'};"
            f" stored kernels {t_lat:.0f} s + {p_lat:.2f} s per product, matrix-free {t_mf:.0f} s + {p_mf:.2f} s;"
            f" peak memory {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB",
            flush=True,
        )
    print("ALL PASS" if ok else "SOME FAILED")


if __name__ == "__main__":
    main()
