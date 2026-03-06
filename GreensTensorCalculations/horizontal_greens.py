#!/usr/bin/env python3
"""
Horizontal Green's tensor for inter-cube coupling on a 2D grid.

All scatterers are at the same depth (Δz = 0), so the Green's tensor
for horizontal coupling is G_ik(Δx, Δy, 0).

Method:
  1. kx residue  → right-going (Δx > 0) or left-going (Δx < 0) waves
  2. ky by 1D IFFT → gives all Δy values on the grid simultaneously
  3. kz by direct quadrature (Δz = 0 means exp(ikz·0) = 1, no phase)

The post-kx-residue kernel at Δz = 0 is:

  Ĝ_ik(ky, kz; Δx) = (i/2ρ) × [ δ_ik eT/(β² kxT)
                                   + kL_i kL_k eL/(ω² kxL)
                                   − kT_i kT_k eT/(ω² kxT) ]

where:
  kxT = √(kS² − ky² − kz²),  kxL = √(kP² − ky² − kz²)
  eT = exp(i kxT |Δx|),  eL = exp(i kxL |Δx|)
  kL = (kxL, ky, kz),  kT = (kxT, ky, kz)

The remaining integral:
  G_ik(Δx, Δy, 0) = 1/(2π)² ∫∫ Ĝ_ik(ky, kz; Δx) × exp(i ky Δy) dky dkz
                    = 1/(2π) ∫ [1/(2π) ∫ Ĝ_ik exp(i ky Δy) dky] dkz
                                 ↑ 1D IFFT                     ↑ kz quadrature
"""

from time import time

import numpy as np

# ═══════════════════════════════════════════════════════════════
#  Physics
# ═══════════════════════════════════════════════════════════════
RHO = 3.0
ALPHA = 5.0
BETA = 3.0
ETA = 0.03
OMEGA = 2 * np.pi * (1 + 1j * ETA)
MU = RHO * BETA**2
KP = OMEGA / ALPHA
KS = OMEGA / BETA


def exact_greens(x, y, z, omega=OMEGA, rho=RHO, alpha=ALPHA, beta=BETA):
    """Ben-Menahem & Singh exact spatial Green's tensor."""
    r = np.sqrt(x**2 + y**2 + z**2 + 0j)
    gam = np.array([x, y, z], dtype=complex) / r
    kP = omega / alpha
    kS = omega / beta
    phiP = np.exp(1j * kP * r)
    phiS = np.exp(1j * kS * r)
    fac = 1.0 / (4 * np.pi * rho)
    f = fac * (
        phiS / (beta**2 * r)
        + (phiP - phiS) / (omega**2 * r**3)
        + 1j * (phiS / beta - phiP / alpha) / (omega * r**2)
    )
    g = fac * (
        phiP / (alpha**2 * r)
        - phiS / (beta**2 * r)
        + 3 * (phiS - phiP) / (omega**2 * r**3)
        - 3j * (phiS / beta - phiP / alpha) / (omega * r**2)
    )
    G = np.zeros((3, 3), dtype=complex)
    for i in range(3):
        for j in range(3):
            G[i, j] = f * (1 if i == j else 0) + g * gam[i] * gam[j]
    return G


# ═══════════════════════════════════════════════════════════════
#  FFT grid (1D, for the ky direction)
# ═══════════════════════════════════════════════════════════════
def fft_grid_1d(N, kmax):
    """
    1D FFT-compatible grid.
    k_n = (n - N/2) × dk,   dk = 2 kmax / N
    x_m = (m - N/2) × dx,   dx = π / kmax
    """
    dk = 2.0 * kmax / N
    dx = np.pi / kmax
    k = (np.arange(N) - N // 2) * dk
    x = (np.arange(N) - N // 2) * dx
    return k, x, dk, dx


# ═══════════════════════════════════════════════════════════════
#  Vectorised post-kx-residue kernel over ky array
# ═══════════════════════════════════════════════════════════════
def post_kx_residue_kernel_vec(
    ky_arr, kz, dx_abs, omega=OMEGA, rho=RHO, alpha=ALPHA, beta=BETA
):
    """
    Compute the post-kx-residue kernel for a 1D array of ky values
    and a single kz value.

    Returns: G[i, j, ky_idx]  — shape (3, 3, Nky) complex array.
    """
    kP = omega / alpha
    kS = omega / beta
    Nky = len(ky_arr)

    # Horizontal kx-wavenumbers at P and S poles
    kx_L = np.sqrt(kP**2 - ky_arr**2 - kz**2 + 0j)
    kx_T = np.sqrt(kS**2 - ky_arr**2 - kz**2 + 0j)

    # Enforce Im >= 0 (UHP for right-going waves)
    kx_L = np.where(np.imag(kx_L) < 0, -kx_L, kx_L)
    kx_T = np.where(np.imag(kx_T) < 0, -kx_T, kx_T)

    eL = np.exp(1j * kx_L * dx_abs)
    eT = np.exp(1j * kx_T * dx_abs)

    # Wave vector components at the poles
    # kL = (kx_L, ky, kz),  kT = (kx_T, ky, kz)
    kL_vec = [kx_L, ky_arr, np.full(Nky, kz, dtype=complex)]
    kT_vec = [kx_T, ky_arr, np.full(Nky, kz, dtype=complex)]

    # Build kernel: shape (3, 3, Nky)
    G = np.zeros((3, 3, Nky), dtype=complex)
    for i in range(3):
        for j in range(3):
            dij = 1.0 if i == j else 0.0
            G[i, j, :] = (1j / (2 * rho)) * (
                dij * eT / (beta**2 * kx_T)
                + kL_vec[i] * kL_vec[j] * eL / (omega**2 * kx_L)
                - kT_vec[i] * kT_vec[j] * eT / (omega**2 * kx_T)
            )
    return G


# ═══════════════════════════════════════════════════════════════
#  Horizontal Green's tensor: kx residue + ky IFFT + kz quadrature
# ═══════════════════════════════════════════════════════════════
def horizontal_greens_fft(
    dx_abs, Nky, ky_max, kz_max, Nkz, omega=OMEGA, rho=RHO, alpha=ALPHA, beta=BETA
):
    """
    Compute G_ik(Δx, Δy_m, 0) for all Δy values on the 1D FFT grid.

    Parameters:
      dx_abs : float — |Δx| (horizontal x-separation, must be > 0)
      Nky    : int   — number of ky grid points (FFT size)
      ky_max : float — ky truncation
      kz_max : float — kz truncation (for quadrature)
      Nkz    : int   — number of kz quadrature points

    Returns:
      G      : (3, 3, Nky) complex — Green's tensor at all Δy grid points
      y_grid : (Nky,) float — the Δy values
    """
    # ky grid (FFT)
    ky_arr, y_grid, dky, dy = fft_grid_1d(Nky, ky_max)

    # kz grid (quadrature, Δz=0 so no phase factor)
    kz_arr = np.linspace(-kz_max, kz_max, Nkz)
    dkz = kz_arr[1] - kz_arr[0]

    # FFT scaling for 1D IFFT:
    #   G(Δy) = (1/2π) ∫ Ĝ(ky) exp(iky Δy) dky
    #         ≈ (dky/2π) Σ Ĝ(ky_n) exp(iky_n Δy_m)
    #         = (dky Nky / 2π) × IFFT[ifftshift(Ĝ)]  (then fftshift)
    #
    # But we also have the 1/(2π) from the kz integral:
    #   Total = (1/(2π)²) ∫∫ ... dky dkz
    #         = (dky dkz)/(2π)² × Σ_kz [Nky × IFFT[ifftshift(Ĝ(:, kz))]]  (fftshifted)

    scale_ky = dky * Nky / (2 * np.pi)  # for the ky IFFT
    scale_kz = dkz / (2 * np.pi)  # for the kz trapezoidal sum

    G_total = np.zeros((3, 3, Nky), dtype=complex)

    for kz_idx, kz in enumerate(kz_arr):
        # Post-kx-residue kernel for all ky at this kz
        # Shape: (3, 3, Nky)
        kernel = post_kx_residue_kernel_vec(ky_arr, kz, dx_abs, omega, rho, alpha, beta)

        # 1D IFFT over ky for each (i, j) component
        for i in range(3):
            for j in range(3):
                # ifftshift → ifft → fftshift (1D along ky axis)
                G_total[i, j, :] += (
                    np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(kernel[i, j, :])))
                    * scale_ky
                    * scale_kz
                )

    return G_total, y_grid


# ═══════════════════════════════════════════════════════════════
#  Vectorised post-ky-residue kernel over kx array (for Δx=0 case)
# ═══════════════════════════════════════════════════════════════
def post_ky_residue_kernel_vec(
    kx_arr, kz, dy_abs, omega=OMEGA, rho=RHO, alpha=ALPHA, beta=BETA
):
    """
    Post-ky-residue kernel for Δy > 0, vectorised over a 1D array of kx values
    and a single kz value.

    After applying the residue theorem to the ky integral (closing in UHP
    for Δy > 0, picking up the pole at ky = +ky_α):

      ky_α = √(k_α² - kx² - kz²),  Im(ky_α) > 0

    The kernel has identical structure to the kx and kz residue cases:

      Ĝ_ik = (i/2ρ) × [ δ_ik eT/(β² kyT)
                         + kL_i kL_k eL/(ω² kyL)
                         − kT_i kT_k eT/(ω² kyT) ]

    where:
      kyL = √(kP² - kx² - kz²),  kyT = √(kS² - kx² - kz²)
      eL = exp(i kyL |Δy|),  eT = exp(i kyT |Δy|)
      kL = (kx, kyL, kz),  kT = (kx, kyT, kz)

    Returns: G[i, j, kx_idx]  — shape (3, 3, Nkx) complex array.
    """
    kP = omega / alpha
    kS = omega / beta
    Nkx = len(kx_arr)

    # ky-wavenumbers at P and S poles
    ky_L = np.sqrt(kP**2 - kx_arr**2 - kz**2 + 0j)
    ky_T = np.sqrt(kS**2 - kx_arr**2 - kz**2 + 0j)

    # Enforce Im >= 0 (UHP for right-going-in-y waves)
    ky_L = np.where(np.imag(ky_L) < 0, -ky_L, ky_L)
    ky_T = np.where(np.imag(ky_T) < 0, -ky_T, ky_T)

    eL = np.exp(1j * ky_L * dy_abs)
    eT = np.exp(1j * ky_T * dy_abs)

    # Wave vector components at the poles: kL = (kx, kyL, kz), kT = (kx, kyT, kz)
    kL_vec = [kx_arr.astype(complex), ky_L, np.full(Nkx, kz, dtype=complex)]
    kT_vec = [kx_arr.astype(complex), ky_T, np.full(Nkx, kz, dtype=complex)]

    # Build kernel: shape (3, 3, Nkx)
    G = np.zeros((3, 3, Nkx), dtype=complex)
    for i in range(3):
        for j in range(3):
            dij = 1.0 if i == j else 0.0
            G[i, j, :] = (1j / (2 * rho)) * (
                dij * eT / (beta**2 * ky_T)
                + kL_vec[i] * kL_vec[j] * eL / (omega**2 * ky_L)
                - kT_vec[i] * kT_vec[j] * eT / (omega**2 * ky_T)
            )
    return G


# ═══════════════════════════════════════════════════════════════
#  Horizontal Green's tensor for Δx=0: ky residue + kx quad + kz quad
# ═══════════════════════════════════════════════════════════════
def horizontal_greens_ky_residue(
    dy_abs, kx_max, Nkx, kz_max, Nkz, omega=OMEGA, rho=RHO, alpha=ALPHA, beta=BETA
):
    """
    Compute G_ik(0, Δy, 0) for a single Δy value via ky residue decomposition.

    When Δx = 0, the kx residue approach fails (Δx=0 means no directional
    separation to close the contour). Instead, we apply residue calculus to ky.

    Method:
      1. ky residue → into-page (Δy > 0) or out-of-page (Δy < 0) waves
      2. kx by direct quadrature (Δx=0 means exp(ikx·0) = 1, no phase)
      3. kz by direct quadrature (Δz=0 means exp(ikz·0) = 1, no phase)

    Both remaining integrals have no oscillatory phase — pure quadrature
    converging via evanescent decay.

    Parameters:
      dy_abs : float — |Δy| (must be > 0)
      kx_max : float — kx truncation for quadrature
      Nkx    : int   — number of kx quadrature points
      kz_max : float — kz truncation for quadrature
      Nkz    : int   — number of kz quadrature points

    Returns:
      G : (3, 3) complex — Green's tensor at (0, Δy, 0)
    """
    # kx grid (quadrature, Δx=0 so no phase)
    kx_arr = np.linspace(-kx_max, kx_max, Nkx)
    dkx = kx_arr[1] - kx_arr[0]

    # kz grid (quadrature, Δz=0 so no phase)
    kz_arr = np.linspace(-kz_max, kz_max, Nkz)
    dkz = kz_arr[1] - kz_arr[0]

    scale = dkx * dkz / (2 * np.pi) ** 2

    G_total = np.zeros((3, 3), dtype=complex)

    for kz in kz_arr:
        # Vectorised kernel over all kx at this kz
        kernel = post_ky_residue_kernel_vec(kx_arr, kz, dy_abs, omega, rho, alpha, beta)
        # Sum over kx (trapezoidal, no phase factor)
        # kernel shape: (3, 3, Nkx)
        G_total += np.sum(kernel, axis=2) * scale

    return G_total


# ═══════════════════════════════════════════════════════════════
#  Direct 2D quadrature (for comparison)
# ═══════════════════════════════════════════════════════════════
def horizontal_greens_direct(
    dx_abs, dy, kmax, nk, omega=OMEGA, rho=RHO, alpha=ALPHA, beta=BETA
):
    """
    Direct 2D (ky, kz) quadrature at a single point. Slow but reliable.
    """
    k1d = np.linspace(-kmax, kmax, nk)
    dk = k1d[1] - k1d[0]

    kP = omega / alpha
    kS = omega / beta

    G = np.zeros((3, 3), dtype=complex)
    for ky in k1d:
        for kz in k1d:
            kx_L = np.sqrt(kP**2 - ky**2 - kz**2 + 0j)
            kx_T = np.sqrt(kS**2 - ky**2 - kz**2 + 0j)
            if np.imag(kx_L) < 0:
                kx_L = -kx_L
            if np.imag(kx_T) < 0:
                kx_T = -kx_T

            eL = np.exp(1j * kx_L * dx_abs)
            eT = np.exp(1j * kx_T * dx_abs)
            kL = np.array([kx_L, ky, kz])
            kT = np.array([kx_T, ky, kz])

            phase = np.exp(1j * ky * dy)  # kz phase = 1 (Δz=0)

            for i in range(3):
                for j in range(3):
                    dij = 1.0 if i == j else 0.0
                    kern = (1j / (2 * rho)) * (
                        dij * eT / (beta**2 * kx_T)
                        + kL[i] * kL[j] * eL / (omega**2 * kx_L)
                        - kT[i] * kT[j] * eT / (omega**2 * kx_T)
                    )
                    G[i, j] += kern * phase * dk**2

    G /= (2 * np.pi) ** 2
    return G


# ═══════════════════════════════════════════════════════════════
#  Tests
# ═══════════════════════════════════════════════════════════════
def main():
    print("=" * 72)
    print("Horizontal Green's tensor: kx residue + ky IFFT + kz quadrature")
    print("=" * 72)
    print(f"  ρ={RHO}, α={ALPHA}, β={BETA}")
    print(f"  ω = {OMEGA:.6f}")
    print(f"  |kP|={abs(KP):.4f},  |kS|={abs(KS):.4f}")
    print()

    # ─── TEST 1: FFT vs direct quadrature at a few Δy values ───
    print("TEST 1: FFT vs direct quadrature (verify FFT implementation)")
    print("─" * 72)

    dx_abs = 1.0
    Nky = 512
    ky_max = 15
    kz_max = 15
    Nkz = 512

    print(f"  Δx={dx_abs}, Nky={Nky}, ky_max={ky_max}, kz_max={kz_max}, Nkz={Nkz}")
    print()

    t0 = time()
    G_fft, y_grid = horizontal_greens_fft(dx_abs, Nky, ky_max, kz_max, Nkz)
    dt_fft = time() - t0
    print(f"  FFT computation: {dt_fft:.1f}s")

    # Compare at a few Δy values
    test_dy = [0.0, 0.3, 0.5, 1.0]
    print(
        f"\n  {'Δy':>8s}  {'grid Δy':>10s}  {'FFT vs Direct':>12s}  {'FFT vs Exact':>12s}"
    )
    print("  " + "─" * 50)

    for dy in test_dy:
        iy = np.argmin(np.abs(y_grid - dy))
        yv = y_grid[iy]
        G_fft_pt = G_fft[:, :, iy]

        # Direct quadrature
        G_dir = horizontal_greens_direct(dx_abs, yv, kmax=ky_max, nk=256)

        # Exact
        G_ex = exact_greens(dx_abs, yv, 0.0)

        err_dir = np.linalg.norm(G_fft_pt - G_dir) / np.linalg.norm(G_dir)
        err_ex = np.linalg.norm(G_fft_pt - G_ex) / np.linalg.norm(G_ex)
        print(f"  {dy:8.3f}  {yv:10.6f}  {err_dir:12.4e}  {err_ex:12.4e}")

    # ─── TEST 2: FFT vs exact at all grid points ───
    print()
    print("TEST 2: Convergence study (FFT vs exact)")
    print("─" * 72)

    dx_abs = 0.8
    test_dy_vals = [0.0, 0.2, 0.5, 1.0]

    print(f"  Δx = {dx_abs},  Δz = 0")
    print(f"\n  {'Nky':>5s}  {'ky_max':>6s}  {'Nkz':>5s}  {'kz_max':>6s}  ", end="")
    for dy in test_dy_vals:
        print(f"  Δy={dy:4.1f}", end="")
    print(f"  {'time':>8s}")
    print("  " + "─" * 80)

    for Nky in [256, 512, 1024]:
        for ky_max in [10, 15]:
            for Nkz in [256, 512]:
                kz_max = ky_max  # same truncation in both directions

                t0 = time()
                G_fft, y_grid = horizontal_greens_fft(dx_abs, Nky, ky_max, kz_max, Nkz)
                dt = time() - t0

                errs = []
                for dy in test_dy_vals:
                    iy = np.argmin(np.abs(y_grid - dy))
                    yv = y_grid[iy]
                    G_ex = exact_greens(dx_abs, yv, 0.0)
                    err = np.linalg.norm(G_fft[:, :, iy] - G_ex) / np.linalg.norm(G_ex)
                    errs.append(err)

                print(f"  {Nky:5d}  {ky_max:6.0f}  {Nkz:5d}  {kz_max:6.0f}  ", end="")
                for e in errs:
                    print(f"  {e:8.2e}", end="")
                print(f"  {dt:7.1f}s")

    # ─── TEST 3: Component detail at best parameters ───
    print()
    print("TEST 3: Component detail at optimal parameters")
    print("─" * 72)

    dx_abs = 0.8
    Nky = 1024
    ky_max = 15
    Nkz = 512
    kz_max = 15

    G_fft, y_grid = horizontal_greens_fft(dx_abs, Nky, ky_max, kz_max, Nkz)

    for dy_target in [0.0, 0.5]:
        iy = np.argmin(np.abs(y_grid - dy_target))
        yv = y_grid[iy]
        G_ff = G_fft[:, :, iy]
        G_ex = exact_greens(dx_abs, yv, 0.0)
        frob_err = np.linalg.norm(G_ff - G_ex) / np.linalg.norm(G_ex)

        print(f"\n  Point: ({dx_abs}, {yv:.6f}, 0)  Frob err = {frob_err:.4e}")
        for i in range(3):
            for j in range(3):
                s, e = G_ff[i, j], G_ex[i, j]
                if abs(e) > 1e-20:
                    ce = abs(s - e) / abs(e)
                    print(f"    G[{i},{j}]: FFT={s:.8e}  exact={e:.8e}  err={ce:.2e}")
                else:
                    print(f"    G[{i},{j}]: FFT={s:.8e}  (exact≈0)")

    # ─── TEST 4: ky residue for Δx=0 case ───
    print()
    print("TEST 4: ky residue (Δx=0 case) vs exact")
    print("─" * 72)

    test_dy_ky = [0.3, 0.5, 0.8, 1.0, 1.5]
    kx_max = 15
    kz_max = 15

    print(f"  Δx = 0, Δz = 0,  kx_max={kx_max}, kz_max={kz_max}")
    print(
        f"\n  {'Δy':>6s}  {'Nkx':>5s}  {'Nkz':>5s}  {'Frob error':>12s}  {'time':>8s}"
    )
    print("  " + "─" * 50)

    for dy in test_dy_ky:
        for Nkx in [256, 512]:
            Nkz = Nkx
            t0 = time()
            G_ky = horizontal_greens_ky_residue(dy, kx_max, Nkx, kz_max, Nkz)
            dt = time() - t0
            G_ex = exact_greens(0.0, dy, 0.0)
            err = np.linalg.norm(G_ky - G_ex) / np.linalg.norm(G_ex)
            print(f"  {dy:6.2f}  {Nkx:5d}  {Nkz:5d}  {err:12.4e}  {dt:7.1f}s")

    # ─── TEST 5: Component detail for ky residue ───
    print()
    print("TEST 5: Component detail for ky residue (Δx=0)")
    print("─" * 72)

    for dy_target in [0.5, 1.0]:
        G_ky = horizontal_greens_ky_residue(
            dy_target, kx_max=15, Nkx=512, kz_max=15, Nkz=512
        )
        G_ex = exact_greens(0.0, dy_target, 0.0)
        frob_err = np.linalg.norm(G_ky - G_ex) / np.linalg.norm(G_ex)

        print(f"\n  Point: (0, {dy_target}, 0)  Frob err = {frob_err:.4e}")
        for i in range(3):
            for j in range(3):
                s, e = G_ky[i, j], G_ex[i, j]
                if abs(e) > 1e-20:
                    ce = abs(s - e) / abs(e)
                    print(
                        f"    G[{i},{j}]: ky_res={s:.8e}  exact={e:.8e}  err={ce:.2e}"
                    )
                else:
                    print(f"    G[{i},{j}]: ky_res={s:.8e}  (exact≈0)")


if __name__ == "__main__":
    main()
