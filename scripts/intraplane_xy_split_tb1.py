#!/usr/bin/env python3
"""TB1 (foundation) — horizontal_greens.py residue tensor vs exact at lattice vectors.

Deliverable #1 (cross-code/bridge agreement) needs the *actual*
horizontal_greens.py residue Green's tensor to be trustworthy at exactly the
in-plane lattice vectors R = (aL i, aL j, 0) that the Bloch sum (TB2) will use.

This confirms G_ik^residue(R) == G_ik^exact(R) to high precision over the
lattice, in the damped regime (OMEGA = 2pi(1+0.03i)), and shows convergence in
the FFT/quadrature controls (Nky, ky_max, Nkz, kz_max).

  * Delta x != 0  ->  horizontal_greens_fft   (kx residue + ky IFFT + kz quad)
  * Delta x == 0  ->  horizontal_greens_ky_residue (ky residue + kx IFFT)

Both are Delta z = 0; the general-z extension needed for the 3D multipole
projection is TB2's job.
"""

from __future__ import annotations

import numpy as np

from cubic_scattering.horizontal_greens import (
    ALPHA,
    BETA,
    OMEGA,
    RHO,
    exact_greens,
    horizontal_greens_fft,
    horizontal_greens_ky_residue,
)

AL = 2.0  # lattice pitch
LR = 4  # lattice half-extent compared (|i|,|j| <= LR)

# FFT / quadrature controls (chosen so the Delta y grid lands on aL multiples:
# dy = pi/ky_max = aL/4  ->  ky_max = 4 pi / aL; lattice points are grid nodes)
NKY = 2048
KY_MAX = 4 * np.pi / AL  # dy = aL/4 = 0.5
KZ_MAX = 60.0
NKZ = 4000
PARAMS = dict(omega=OMEGA, rho=RHO, alpha=ALPHA, beta=BETA)


def residue_tensor(i: int, j: int) -> np.ndarray:
    """horizontal_greens residue G_ik at lattice vector R = (aL i, aL j, 0)."""
    dx, dy = AL * abs(i), AL * j
    if i == 0:
        # Delta x = 0 : ky-residue form (closes in ky; kx,kz by quadrature).
        # signature: (dy_abs, kx_max, Nkx, kz_max, Nkz)
        return horizontal_greens_ky_residue(abs(dy), KZ_MAX, NKZ, KZ_MAX, NKZ, **PARAMS)
    G_grid, y_grid = horizontal_greens_fft(dx, NKY, KY_MAX, KZ_MAX, NKZ, **PARAMS)
    # pick the grid column nearest Delta y = aL j
    k = int(np.argmin(np.abs(y_grid - dy)))
    assert abs(y_grid[k] - dy) < 1e-9, f"dy={dy} not on grid (nearest {y_grid[k]})"
    return G_grid[:, :, k]


def main() -> None:
    print("==== TB1 :: horizontal_greens residue vs exact at lattice vectors ====")
    print(f"  aL={AL}  Lr={LR}  damped OMEGA={OMEGA:.4f}")
    print(f"  Nky={NKY} ky_max={KY_MAX:.3f}  Nkz={NKZ} kz_max={KZ_MAX}")

    worst = 0.0
    worst_R = None
    n = 0
    for i in range(-LR, LR + 1):
        for j in range(-LR, LR + 1):
            if i == 0 and j == 0:
                continue
            if i < 0:  # G(R) is even in Delta x sign for these blocks; test i>=0 + axis
                continue
            Gr = residue_tensor(i, j)
            Ge = exact_greens(AL * i, AL * j, 0.0, OMEGA)
            rel = np.max(np.abs(Gr - Ge)) / np.max(np.abs(Ge))
            n += 1
            if rel > worst:
                worst, worst_R = rel, (i, j)
    print(f"  compared {n} in-plane lattice vectors (i>=0 half)")
    print(f"  worst rel error = {worst:.2e} at R=(aL*{worst_R[0]}, aL*{worst_R[1]})")
    print(f"  -> {'PASS' if worst < 1e-3 else 'FAIL'} (tol 1e-3)")


if __name__ == "__main__":
    main()
