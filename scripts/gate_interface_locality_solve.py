"""Gate: the locality-bounded set-up of the interface couplings, blocks and solve.

Plan: docs/2026-10-10-stratified-reference-legendre-cells-3d.md (step 2: the set-up scales with the radius).
Paper: LatexPDFs/InterfaceCouplingIntegrals, Table tab:locality.

With a smooth window chi the frequency-independent corrections A0 = K0 - int (1 - chi) Y0 and A1 = int chi Y1
decay faster than any power of the lateral offset d (tab:locality), so beyond a radius the block is the windowed
plane-wave part D alone and the set-up need not compute them (``InterfaceCouplings(radius=...)``).

THE ROW.  Two rows of 14 cells along x, one on each side of the interface (iz = -1, 0; ix = 0 ... 13: lateral
offsets to 26 h), each with a contrast of 10% of its medium; media A (5, 3, 2.5) over B (6.5, 3.6, 2.8), h = 0.05
km, omega h / beta_A = 1/6, Q0 h = 10, the C-infinity window from Q0 / 4, a P wave with lateral slowness
(0.3, 0.1) / alpha_A.  The reference is every class with its corrections; the radius 12 h, 16 h, 24 h drops those
beyond it.  Measured: the largest change of any block relative to the largest block, and of the solution.
Expected from tab:locality: about 1e-6, 1e-8, 1e-10.

Run:  PYTHONPATH=. python scripts/gate_interface_locality_solve.py
"""

import copy
import time

import numpy as np

from cubic_scattering import interface_solver as iso
from cubic_scattering import interface_spectral as isp
from cubic_scattering.effective_contrasts import MaterialContrast, ReferenceMedium

A = ReferenceMedium(5.0, 3.0, 2.5)
B = ReferenceMedium(6.5, 3.6, 2.8)
H = 0.05
OMEGA = A.beta / (6 * H)
WINDOW = isp.Window(0.25)
CELLS = [(iz, ix, 0) for iz in (-1, 0) for ix in range(14)]
KX, KY = 0.3 * OMEGA / A.alpha, 0.1 * OMEGA / A.alpha
EXPECT = {12.0: 3e-6, 16.0: 3e-8, 24.0: 3e-10}


def main() -> None:
    grid = iso.InterfaceGrid(np.array(CELLS), H, A, B)
    ctr = [
        MaterialContrast(0.1 * m.lam, 0.1 * m.mu, 0.1 * m.rho)
        for m in (grid.medium(i) for i in range(len(CELLS)))
    ]
    t0 = time.time()
    full = iso.InterfaceCouplings(grid, window=WINDOW)
    t_full = time.time() - t0
    blocks = full.blocks(OMEGA)
    scale = max(np.abs(k).max() for k in blocks.values())
    ref = iso.solve_interface(full, OMEGA, ctr, KX, KY, "P", p=1, blocks=blocks)
    print(
        f"full set-up: {len(full.classes)} classes with corrections, {t_full:.0f} s",
        flush=True,
    )
    ok = True
    for radius, expect in EXPECT.items():
        # the radius-bounded set-up computes exactly the full one's parts within the radius: drop the rest
        cut = copy.copy(full)
        cut.radius = radius
        cut.classes = {
            k: (p if np.hypot(k[2], k[3]) <= radius else None, t)
            for k, (p, t) in full.classes.items()
        }
        kept = sum(p is not None for p, _ in cut.classes.values())
        b2 = cut.blocks(OMEGA)
        err_b = max(np.abs(b2[k] - blocks[k]).max() for k in blocks) / scale
        res = iso.solve_interface(cut, OMEGA, ctr, KX, KY, "P", p=1, blocks=b2)
        err_s = np.abs(res.psi - ref.psi).max() / np.abs(ref.psi).max()
        good = err_b < expect
        ok &= good
        print(
            f"radius {radius:4.0f} h: {kept}/{len(full.classes)} classes with corrections; blocks {err_b:.1e},"
            f" solution {err_s:.1e}  {'PASS' if good else 'FAIL'}",
            flush=True,
        )
    # the radius-bounded set-up itself: the same parts, fewer of them, in less time
    t0 = time.time()
    real = iso.InterfaceCouplings(grid, window=WINDOW, radius=16.0)
    t_cut = time.time() - t0
    same = all(
        (p is None) == (real.classes[k][0] is None)
        and (
            p is None
            or all(
                np.array_equal(x, y) for x, y in zip(p, real.classes[k][0], strict=True)
            )
        )
        for k, (p, _) in {
            k: (v[0] if np.hypot(k[2], k[3]) <= 16.0 else None, v[1])
            for k, v in full.classes.items()
        }.items()
    )
    ok &= same
    print(
        f"radius 16 h set-up: {t_cut:.0f} s (full {t_full:.0f} s); parts identical to the full set-up's: {same}"
    )
    print("ALL PASS" if ok else "SOME FAILED")


if __name__ == "__main__":
    main()
