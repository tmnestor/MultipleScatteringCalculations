"""Gate: the depth-bounded set-up of the interface couplings, blocks and solve.

Plan: docs/2026-10-15-interface-solver-scaling-plan.md, S2 and D1 (block tolerance 1e-8).

The corrections A0, A1 decay with the total gap g = d_r + d_s of the two cells from the interface, by ~1e-4 per 2 h
(scripts/measure_interface_depth_locality.py), so beyond a depth radius the block is the windowed plane-wave part D
alone and the set-up need not compute them (``InterfaceCouplings(depth_radius=...)``).

THE STACK.  Six layers, iz = -3 ... 2 (three on each side of the interface), two cells each along x (lateral
offsets 0 and 2 h), each with a contrast of 10% of its medium; media A (5, 3, 2.5) over B (6.5, 3.6, 2.8),
h = 0.05 km, omega h / beta_A = 1/6, Q0 h = 10, the C-infinity window from Q0 / 4, a P wave with lateral slowness
(0.3, 0.1) / alpha_A.  The reference is every class with its corrections; the depth radius 4 h (D1, 1e-8) and
6 h (1e-10) drops those at and beyond it.  Measured: the largest change of any block relative to the largest
block of the system (as scripts/gate_interface_locality_solve.py), and of the solution.  Then the depth-bounded
set-up itself: its parts are the full set-up's, and its time.

Run:  PYTHONPATH=. python scripts/gate_interface_depth_radius.py
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
CELLS = [(iz, ix, 0) for iz in range(-3, 3) for ix in range(2)]
KX, KY = 0.3 * OMEGA / A.alpha, 0.1 * OMEGA / A.alpha
EXPECT = {4.0: 1e-8, 6.0: 1e-10}


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
    for depth, expect in EXPECT.items():
        # the depth-bounded set-up computes exactly the full one's parts within the radius: drop the rest
        cut = copy.copy(full)
        cut.depth_radius = depth
        cut.classes = {
            k: (p if cut.corrected(k) else None, t)
            for k, (p, t) in full.classes.items()
        }
        kept = sum(p is not None for p, _ in cut.classes.values())
        b2 = cut.blocks(OMEGA)
        err_b = max(np.abs(b2[k] - blocks[k]).max() for k in blocks) / scale
        res = iso.solve_interface(cut, OMEGA, ctr, KX, KY, "P", p=1, blocks=b2)
        err_s = np.abs(res.psi - ref.psi).max() / np.abs(ref.psi).max()
        good = err_b < expect and err_s < expect
        ok &= good
        print(
            f"depth radius {depth:3.0f} h: {kept}/{len(full.classes)} classes with corrections; blocks {err_b:.1e},"
            f" solution {err_s:.1e}  {'PASS' if good else 'FAIL'}",
            flush=True,
        )
    t0 = time.time()
    real = iso.InterfaceCouplings(grid, window=WINDOW, depth_radius=4.0)
    t_cut = time.time() - t0
    same = all(
        (real.classes[k][0] is None) == (not real.corrected(k))
        and (
            real.classes[k][0] is None
            or all(
                np.array_equal(x, y) for x, y in zip(p, real.classes[k][0], strict=True)
            )
        )
        for k, (p, _) in full.classes.items()
    )
    ok &= same
    print(
        f"depth radius 4 h set-up: {t_cut:.0f} s (full {t_full:.0f} s); parts identical to the full set-up's: {same}"
    )
    print("ALL PASS" if ok else "SOME FAILED")


if __name__ == "__main__":
    main()
