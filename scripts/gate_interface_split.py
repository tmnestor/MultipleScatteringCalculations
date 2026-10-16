"""Gate: D split into a low band (per offset) and a high band (sampled on the FFT grid), against the unsplit D.

Plan: docs/2026-10-15-interface-solver-scaling-plan.md, S4 (revised: the split), D1 (block tolerance 1e-8).

D decays only as ~h/r laterally (scripts/measure_interface_d_decay.py), so its spectrum cannot be sampled on a
periodic grid as a whole.  ``InterfaceCouplings(split=True)``: D_low, below the window's ramp, holds the branch
points and the tail and stays a radial integral per offset (tables over q h < 2.5 only); D_high is smooth in k and is
sampled on the grid of a periodic box (``interface_spectral.high_band_blocks``).

1. BLOCKS.  A box of 2 x 4 x 4 cells straddling the interface (iz = -1, 0), every class, sources in A and in B:
   the split against the unsplit D (same window and corrections), relative to the largest entry of the touching
   static block K0, at omega h / beta_A = 1/6 and 1/2.  PASS below D1 = 1e-8.
2. SOLVE.  The same box, 10% contrast, GMRES with the split blocks against GMRES with the unsplit ones.
3. COST.  The box of scripts/gate_interface_gmres.py, 2 x 12 x 12 (radius 16 h): the split set-up and the kernels
   of one frequency, against S1's 1443 s and 183 s.

Media A (5, 3, 2.5) over B (6.5, 3.6, 2.8), h = 0.05 km, Q0 h = 10, the C-infinity window from Q0 / 4, p = 1, a P
wave with lateral slowness (0.3, 0.1) / alpha_A.

Measured 16 October 2026 (four workers): 1. 40 classes; blocks 1.3e-10 of K0 at 1/6, 2.9e-9 at 1/2 (the band is
narrower there, [1, 2.5] against [0.35, 2.5]); set-up 517 s split, 458 s unsplit.  2. GMRES 7.4e-12.  3. 288 cells:
set-up 931 s (S1: 1443 s), kernels of one frequency 321 s (S1: 183 s), 9 products.

Run:  PYTHONPATH=. python scripts/gate_interface_split.py
"""

import time

import numpy as np

from cubic_scattering import interface_solver as iso
from cubic_scattering import interface_spectral as isp
from cubic_scattering.effective_contrasts import MaterialContrast, ReferenceMedium

A = ReferenceMedium(5.0, 3.0, 2.5)
B = ReferenceMedium(6.5, 3.6, 2.8)
H = 0.05
WINDOW = isp.Window(0.25)
FREQS = {"1/6": A.beta / (6 * H), "1/2": A.beta / (2 * H)}
TOL = 1e-8


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
    g = box(4)
    t0 = time.time()
    full = iso.InterfaceCouplings(g, window=WINDOW)
    t_full = time.time() - t0
    t0 = time.time()
    split = iso.InterfaceCouplings(g, window=WINDOW, split=True)
    t_split = time.time() - t0
    print(
        f"1. 2 x 4 x 4: {len(full.classes)} classes; set-up {t_full:.0f} s unsplit, {t_split:.0f} s split",
        flush=True,
    )
    k0 = max(
        float(np.abs(isp.image_block((zr, 0, 0), (-1, 0, 0), H, A, B, 10, 4)).max())
        for zr in (-1, 1)
    )
    for lab, omega in FREQS.items():
        t0 = time.time()
        ref = full.interface_canonical(omega)
        t_ref = time.time() - t0
        t0 = time.time()
        got = split.interface_canonical(omega)
        t_got = time.time() - t0
        err = max(np.abs(got[k] - ref[k]).max() for k in ref) / k0
        good = err < TOL
        ok &= good
        print(
            f"   omega h / beta_A = {lab}: blocks {err:.1e} of K0 ({t_ref:.0f} s unsplit, {t_got:.0f} s split)"
            f"  {'PASS' if good else 'FAIL'}",
            flush=True,
        )
    omega = FREQS["1/6"]
    kx, ky = 0.3 * omega / A.alpha, 0.1 * omega / A.alpha
    a, _ = iso.solve_interface_gmres(full, omega, contrasts(g), kx, ky, "P", p=1)
    b, _ = iso.solve_interface_gmres(split, omega, contrasts(g), kx, ky, "P", p=1)
    err = np.abs(a.psi - b.psi).max() / np.abs(a.psi).max()
    good = err < TOL
    ok &= good
    print(
        f"2. GMRES, split against unsplit: {err:.1e}  {'PASS' if good else 'FAIL'}",
        flush=True,
    )

    g = box(12)
    t0 = time.time()
    c = iso.InterfaceCouplings(g, window=WINDOW, radius=16.0, split=True)
    t_setup = time.time() - t0
    t0 = time.time()
    op = iso.LateralOperator(c, omega, p=1, r=0)
    t_kernels = time.time() - t0
    _, n_it = iso.solve_interface_gmres(
        c, omega, contrasts(g), kx, ky, "P", p=1, operator=op
    )
    print(
        f"3. 2 x 12 x 12, split: set-up {t_setup:.0f} s (S1: 1443 s); kernels of one frequency {t_kernels:.0f} s"
        f" (S1: 183 s); GMRES {n_it} products",
        flush=True,
    )
    print("ALL PASS" if ok else "SOME FAILED")


if __name__ == "__main__":
    main()
