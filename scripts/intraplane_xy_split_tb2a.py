#!/usr/bin/env python3
"""TB2a — general-z extension of the x/y-split residue tensor, vs exact.

TB2's multipole projection needs the lattice field on a 3D sphere (Delta z != 0),
but horizontal_greens.py computes only Delta z = 0.  The k_x-residue
representation extends to general Delta z by carrying the e^{i k_z dz} phase in
the k_z quadrature:

  G_ik(dx, dy, dz) = 1/(2pi)^2 int int  G^_ik(ky,kz; dx)
                       e^{i(k_y dy + k_z dz)} dk_y dk_z
                     = 1/(2pi) int_kz e^{i k_z dz}
                       [ 1/(2pi) int_ky G^_ik e^{i k_y dy} dk_y ] dk_z
                              ^ ky IFFT (per kz)              ^ kz quadrature

The uniform-grid + damping scheme stays valid (the same reason it beat
Gauss-Legendre for the 1/kxc ring): for dx > 0 the evanescent factor
e^{-Im(kxc) dx} forces absolute convergence regardless of dz.

This validates the general-z tensor against exact_greens at off-plane points,
and shows convergence in (Nkz, kz_max).
"""

from __future__ import annotations

import numpy as np

from cubic_scattering.horizontal_greens import (
    ALPHA,
    BETA,
    OMEGA,
    RHO,
    exact_greens,
    fft_grid_1d,
    post_kx_residue_kernel_vec,
)

AL = 2.0
NKY = 2048
KY_MAX = 4 * np.pi / AL  # dy grid step = aL/4 = 0.5
PARAMS = dict(omega=OMEGA, rho=RHO, alpha=ALPHA, beta=BETA)


def residue_tensor_genz(
    dx_abs: float, dz: float, kz_max: float, nkz: int
) -> tuple[np.ndarray, np.ndarray]:
    """G_ik(dx, dy_grid, dz) for all dy on the FFT grid, general dz.

    Returns (G[3,3,Nky], y_grid[Nky]).
    """
    ky_arr, y_grid, dky, _ = fft_grid_1d(NKY, KY_MAX)
    kz_arr = np.linspace(-kz_max, kz_max, nkz)
    dkz = kz_arr[1] - kz_arr[0]
    scale_ky = dky * NKY / (2 * np.pi)
    scale_kz = dkz / (2 * np.pi)

    G = np.zeros((3, 3, NKY), dtype=complex)
    for kz in kz_arr:
        kernel = post_kx_residue_kernel_vec(ky_arr, kz, dx_abs, **PARAMS)
        phase_z = np.exp(1j * kz * dz)
        for i in range(3):
            for j in range(3):
                G[i, j, :] += (
                    np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(kernel[i, j, :])))
                    * scale_ky
                    * scale_kz
                    * phase_z
                )
    return G, y_grid


def worst_vs_exact(
    dx: float, dz: float, kz_max: float, nkz: int, dy_max: float
) -> float:
    """Worst rel error of the general-z residue tensor vs exact over |dy| <= dy_max."""
    G, y_grid = residue_tensor_genz(dx, dz, kz_max, nkz)
    sel = np.nonzero(np.abs(y_grid) <= dy_max + 1e-9)[0]
    worst = 0.0
    for k in sel:
        Ge = exact_greens(dx, y_grid[k], dz, OMEGA)
        rel = np.max(np.abs(G[:, :, k] - Ge)) / np.max(np.abs(Ge))
        worst = max(worst, rel)
    return worst


def main() -> None:
    print("==== TB2a :: general-z x/y-split residue tensor vs exact ====")
    print(f"  aL={AL}  damped OMEGA={OMEGA:.4f}  Nky={NKY} ky_max={KY_MAX:.3f}")

    # convergence in kz controls at a representative off-plane point set
    print("\n  convergence (dx=2.0, dz=0.5, |dy|<=4):")
    prev = None
    for kz_max, nkz in ((40.0, 3000), (60.0, 4000), (80.0, 6000)):
        w = worst_vs_exact(2.0, 0.5, kz_max, nkz, 4.0)
        tag = "" if prev is None else f"  (d={abs(w - prev):.1e})"
        print(f"    kz_max={kz_max:5.0f} Nkz={nkz:5d} -> worst rel = {w:.2e}{tag}")
        prev = w

    # off-plane grid of (dx, dz)
    kz_max, nkz = 80.0, 6000
    print(f"\n  off-plane sweep (kz_max={kz_max}, Nkz={nkz}, |dy|<=4):")
    worst_all = 0.0
    for dx in (1.5, 2.0, 4.0):
        for dz in (0.2, 0.5, 1.0):
            w = worst_vs_exact(dx, dz, kz_max, nkz, 4.0)
            worst_all = max(worst_all, w)
            print(f"    dx={dx:4.1f} dz={dz:4.1f} -> worst rel = {w:.2e}")
    print(
        f"\n  overall worst rel error = {worst_all:.2e} -> "
        f"{'PASS' if worst_all < 1e-3 else 'FAIL'} (tol 1e-3)"
    )


if __name__ == "__main__":
    main()
