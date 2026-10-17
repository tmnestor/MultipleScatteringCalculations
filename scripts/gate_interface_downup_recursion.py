"""Gate: the two-way march on the recursions applies the same preconditioner as on the stored kernels (S6b).

Plan: docs/2026-10-15-interface-solver-scaling-plan.md, S6b.

``interface_downup.MatrixFreeDownUp`` (``MatrixFreeOperator``'s pieces: the march, the transmitted and reflected
mode amplitudes, the stored kernels of touching planes, the correction stencils) against ``DownUp`` (the stored
kernels of every pair of planes, ``LateralOperator(parts=True)``), from the same set-up, applied to a random
vector with the planes' inner GMRES to 1e-13.  Boxes 8 x 2 x 2 (four planes each side: the march acts between
planes three or more apart) and 4 x 3 x 3; omega h / beta_A = 1/6 and 1/2; contrast 30 %.  Set-up as S6: split,
C-infinity window from Q0 / 4, radius 16 h, depth radius 4 h; media A (5, 3, 2.5) over B (6.5, 3.6, 2.8),
h = 0.05 km, p = 1.  PASS below 1e-9 relative (the matrix-free product is ~2e-11 from the stored one, and the
march's high band ~1e-9 from Paper 2's blocks at 1/2).

Measured 17 October 2026 (8 x 2 x 2): 1.5e-12 (1/6), 1.9e-11 (1/2) PASS.  Time: stored 0.1 s, recursion 620 and
830 s, ~3.5 s per plane product: each plane's own D goes through every wavenumber of the band (185k for the high
band's padded lattice alone), which is the cost to remove.

Run:  PYTHONPATH=. python scripts/gate_interface_downup_recursion.py [box indices, default all]
"""

import sys
import time

import numpy as np

from cubic_scattering import interface_downup as idu
from cubic_scattering import interface_solver as iso
from cubic_scattering import interface_spectral as isp
from cubic_scattering.effective_contrasts import MaterialContrast, ReferenceMedium
from cubic_scattering.graded_voxel.basis import source_expansion

A = ReferenceMedium(5.0, 3.0, 2.5)
B = ReferenceMedium(6.5, 3.6, 2.8)
H = 0.05
TOL = 1e-9
BOXES = [(tuple(range(-4, 4)), 2), ((-2, -1, 0, 1), 3)]
FREQS = {"1/6": A.beta / (6 * H), "1/2": A.beta / (2 * H)}


def main() -> None:
    ok = True
    picked = [int(a) for a in sys.argv[1:]] or range(len(BOXES))
    for layers, n in (BOXES[i] for i in picked):
        cells = [(iz, ix, iy) for iz in layers for ix in range(n) for iy in range(n)]
        grid = iso.InterfaceGrid(np.array(cells), H, A, B)
        c = iso.InterfaceCouplings(
            grid, window=isp.Window(0.25), radius=16.0, depth_radius=4.0, split=True
        )
        ctr = [
            MaterialContrast(0.3 * m.lam, 0.3 * m.mu, 0.3 * m.rho)
            for m in (grid.medium(i) for i in range(len(cells)))
        ]
        for f, omega in FREQS.items():
            lat = iso.LateralOperator(c, omega, p=1, r=0, parts=True)
            mf = iso.MatrixFreeOperator(c, omega, p=1, r=0)
            delta = iso.cell_contrasts(grid, omega, ctr, 0)
            e = np.array(
                [
                    source_expansion(d, lat.n_field)[: lat.n_source, : lat.na]
                    for d in delta
                ]
            )
            v = np.random.default_rng(3).standard_normal(len(cells) * lat.na * 9) + 0j
            t0 = time.time()
            want = idu.DownUp(lat, e, inner_tol=1e-13)(v)
            t_s = time.time() - t0
            du = idu.MatrixFreeDownUp(mf, e, inner_tol=1e-13)
            t0 = time.time()
            got = du(v)
            t_m = time.time() - t0
            err = np.abs(got - want).max() / np.abs(want).max()
            good = err < TOL
            ok &= good
            print(
                f"{len(cells)} cells, {len(layers)} layers, {f}: applications differ by {err:.1e}"
                f"  {'PASS' if good else 'FAIL'}; stored {t_s:.1f} s, recursion {t_m:.1f} s"
                f" ({du.inner_products} plane products)",
                flush=True,
            )
    print("ALL PASS" if ok else "SOME FAILED")


if __name__ == "__main__":
    main()
