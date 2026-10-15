"""Gate: the iterative interface solve (lateral FFT products, GMRES) against the dense solve, and its cost.

Plan: docs/2026-10-10-stratified-reference-legendre-cells-3d.md (step 2: larger scatterers).

1. AGAINST THE DENSE SOLVE.  A box of 2 x 5 x 5 cells straddling the interface (iz = -1, 0), each with a contrast
   of 10% of its medium; the locality-bounded set-up (C-infinity window from Q0 / 4, radius 16 h); GMRES with the
   lateral FFT products against the dense solve from the same blocks.
2. A LARGER BOX.  2 x 12 x 12 cells (288), GMRES alone: the set-up, the kernels of one frequency, the iterations
   and the solve.
Media A (5, 3, 2.5) over B (6.5, 3.6, 2.8), h = 0.05 km, omega h / beta_A = 1/6, p = 1, a P wave with lateral
slowness (0.3, 0.1) / alpha_A.

Run:  PYTHONPATH=. python scripts/gate_interface_gmres.py
"""

import time

import numpy as np

from cubic_scattering import interface_solver as iso
from cubic_scattering import interface_spectral as isp
from cubic_scattering.effective_contrasts import MaterialContrast, ReferenceMedium

A = ReferenceMedium(5.0, 3.0, 2.5)
B = ReferenceMedium(6.5, 3.6, 2.8)
H = 0.05
OMEGA = A.beta / (6 * H)
KX, KY = 0.3 * OMEGA / A.alpha, 0.1 * OMEGA / A.alpha


def box(n: int) -> iso.InterfaceGrid:
    cells = [(iz, ix, iy) for iz in (-1, 0) for ix in range(n) for iy in range(n)]
    return iso.InterfaceGrid(np.array(cells), H, A, B)


def contrasts(grid: iso.InterfaceGrid) -> list:
    return [
        MaterialContrast(0.1 * m.lam, 0.1 * m.mu, 0.1 * m.rho)
        for m in (grid.medium(i) for i in range(len(grid.idx)))
    ]


def main() -> None:
    ok = True
    for n, dense in ((5, True), (12, False)):
        g = box(n)
        t0 = time.time()
        c = iso.InterfaceCouplings(g, window=isp.Window(0.25), radius=16.0)
        t_setup = time.time() - t0
        kept = sum(p is not None for p, _ in c.classes.values())
        t0 = time.time()
        op = iso.LateralOperator(c, OMEGA, p=1, r=0)
        t_kernels = time.time() - t0
        mem = sum(k.nbytes for k in op.kernels.values()) / 1e6
        t0 = time.time()
        res, n_it = iso.solve_interface_gmres(
            c, OMEGA, contrasts(g), KX, KY, "P", p=1, operator=op
        )
        t_solve = time.time() - t0
        line = (
            f"{2 * n * n} cells: set-up {t_setup:.0f} s ({kept}/{len(c.classes)} classes with corrections);"
            f" kernels {t_kernels:.1f} s ({mem:.0f} MB); GMRES {n_it} products, {t_solve:.1f} s"
        )
        if dense:
            ref = iso.solve_interface(c, OMEGA, contrasts(g), KX, KY, "P", p=1)
            err = np.abs(res.psi - ref.psi).max() / np.abs(ref.psi).max()
            good = err < 1e-9
            ok &= good
            line += f"; vs dense {err:.1e}  {'PASS' if good else 'FAIL'}"
        print(line, flush=True)
    print("ALL PASS" if ok else "SOME FAILED")


if __name__ == "__main__":
    main()
