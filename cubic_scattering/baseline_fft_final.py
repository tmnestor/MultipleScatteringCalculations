"""
2D FFT verification: spectral → spatial Green's tensor via kz residues.

Post-residue kernel (z > z', UHP closure):
  Ĝ_ij(kx,ky;z) = i/(2ρ) × [δ_ij eˢ/(β²kzS) + kᴾᵢkᴾⱼ eᴾ/(ω²kzP)
                               − kˢᵢkˢⱼ eˢ/(ω²kzS)]

Spatial Green's tensor:
  G_ij(x,y,z) = 1/(4π²) ∫∫ Ĝ_ij(kx,ky;z) exp(i[kx·x + ky·y]) dkx dky

This is a standard 2D inverse Fourier transform → use numpy.fft.ifft2.

Grid conventions (FFT-compatible):
  k-space:  kx_n = (n − N/2) × dk,  dk = 2 kmax / N,  n = 0…N−1
            kx ∈ [−kmax, kmax − dk)   (N points, excludes +kmax)
  x-space:  x_m = (m − N/2) × dx,  dx = π / kmax
            x ∈ [−N dx/2, (N/2−1) dx)

Accuracy controlled by:
  1. kmax (truncation)  — dominant for near-field (small z)
  2. dk   (sampling)    — must resolve spectral features near k ≈ ω/c
  3. N = 2 kmax / dk    — ties them together

Near-field test: kS × z < 0.1  →  z ≲ 0.048
"""

from time import time

import numpy as np

# ── Physical parameters ──
rho = 3.0
alpha = 5.0
beta = 3.0
omega_r = 2 * np.pi
eta = 0.03
omega = omega_r * (1 + 1j * eta)
kP_mag = abs(omega / alpha)  # ≈ 1.257
kS_mag = abs(omega / beta)  # ≈ 2.094


# ═══════════════════════════════════════════════════════════════
#  Exact spatial Green's tensor (Ben-Menahem & Singh)
# ═══════════════════════════════════════════════════════════════
def exact_greens(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2 + 0j)
    gam = np.array([x, y, z], dtype=complex) / r
    kP, kS = omega / alpha, omega / beta
    phiP, phiS = np.exp(1j * kP * r), np.exp(1j * kS * r)
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
#  Build kernel on FFT-compatible grid
# ═══════════════════════════════════════════════════════════════
def fft_grid(N, kmax):
    """Return k-grid and x-grid in physical (centered) order."""
    dk = 2.0 * kmax / N
    dx = np.pi / kmax
    k = (np.arange(N) - N // 2) * dk  # physical order
    x = (np.arange(N) - N // 2) * dx
    return k, x, dk, dx


def build_kernel(KX, KY, z):
    """Post-residue kernel Ĝ_ij on a 2D grid. Returns (3,3,N,N)."""
    kH2 = KX**2 + KY**2
    kzS = np.sqrt((omega / beta) ** 2 - kH2)
    kzP = np.sqrt((omega / alpha) ** 2 - kH2)
    kzS = np.where(np.imag(kzS) < 0, -kzS, kzS)
    kzP = np.where(np.imag(kzP) < 0, -kzP, kzP)

    eS = np.exp(1j * kzS * z)
    eP = np.exp(1j * kzP * z)
    kS = [KX, KY, kzS]
    kP = [KX, KY, kzP]

    N = KX.shape[0]
    Ghat = np.zeros((3, 3, N, N), dtype=complex)
    for i in range(3):
        for j in range(3):
            dij = 1.0 if i == j else 0.0
            Ghat[i, j] = (1j / (2 * rho)) * (
                dij * eS / (beta**2 * kzS)
                + kP[i] * kP[j] * eP / (omega**2 * kzP)
                - kS[i] * kS[j] * eS / (omega**2 * kzS)
            )
    return Ghat


# ═══════════════════════════════════════════════════════════════
#  Method 1: FFT  — all spatial points at once
# ═══════════════════════════════════════════════════════════════
def greens_fft(N, kmax, z):
    """
    Returns G_ij(x_m, y_n, z) on the full spatial grid via 2D IFFT.
    Output shape: (3, 3, N, N) in physical (centered) spatial order.
    """
    k, x, dk, dx = fft_grid(N, kmax)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    Ghat = build_kernel(KX, KY, z)

    scale = dk**2 * N**2 / (4 * np.pi**2)
    G = np.zeros_like(Ghat)
    for i in range(3):
        for j in range(3):
            G[i, j] = (
                np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(Ghat[i, j]))) * scale
            )
    return G, x


# ═══════════════════════════════════════════════════════════════
#  Method 2: Direct sum — single spatial point, SAME grid as FFT
# ═══════════════════════════════════════════════════════════════
def greens_direct(x, y, z, N, kmax):
    """
    Direct trapezoidal summation on the SAME grid used by the FFT.
    """
    k, _, dk, _ = fft_grid(N, kmax)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    Ghat = build_kernel(KX, KY, z)

    phase = np.exp(1j * (KX * x + KY * y))
    G = np.zeros((3, 3), dtype=complex)
    for i in range(3):
        for j in range(3):
            G[i, j] = np.sum(Ghat[i, j] * phase) * dk**2 / (4 * np.pi**2)
    return G


# ═══════════════════════════════════════════════════════════════
#  Single-component FFT (memory-efficient for large N)
# ═══════════════════════════════════════════════════════════════
def greens_fft_component(i, j, N, kmax, z):
    """Compute one G_ij component via FFT. Returns (N,N) array."""
    k, _, dk, _ = fft_grid(N, kmax)
    KX, KY = np.meshgrid(k, k, indexing="ij")

    kH2 = KX**2 + KY**2
    kzS = np.sqrt((omega / beta) ** 2 - kH2)
    kzP = np.sqrt((omega / alpha) ** 2 - kH2)
    kzS = np.where(np.imag(kzS) < 0, -kzS, kzS)
    kzP = np.where(np.imag(kzP) < 0, -kzP, kzP)

    eS = np.exp(1j * kzS * z)
    eP = np.exp(1j * kzP * z)
    kSv = [KX, KY, kzS]
    kPv = [KX, KY, kzP]

    dij = 1.0 if i == j else 0.0
    kernel = (1j / (2 * rho)) * (
        dij * eS / (beta**2 * kzS)
        + kPv[i] * kPv[j] * eP / (omega**2 * kzP)
        - kSv[i] * kSv[j] * eS / (omega**2 * kzS)
    )

    scale = dk**2 * N**2 / (4 * np.pi**2)
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kernel))) * scale


# ═══════════════════════════════════════════════════════════════
#  TESTS
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 72)
    print("2D FFT: spectral → spatial Green's tensor")
    print("=" * 72)
    print(f"ρ={rho}, α={alpha}, β={beta}")
    print(f"ω = 2π(1+{eta}i) = {omega:.6f}")
    print(f"|kP| = {kP_mag:.4f},  |kS| = {kS_mag:.4f}")
    print()

    # ── TEST 1: FFT == Direct on same grid (sanity) ──
    print("─" * 72)
    print("TEST 1: FFT vs Direct on identical grid (should be machine precision)")
    print("─" * 72)

    for z_val, N, kmax in [(1.0, 256, 25), (0.02, 512, 100)]:
        _, x_grid, _, _ = fft_grid(N, kmax)
        ix = np.argmin(np.abs(x_grid - 0.3))
        iy = np.argmin(np.abs(x_grid - 0.4))
        xv, yv = x_grid[ix], x_grid[iy]

        G_fft_full, _ = greens_fft(N, kmax, z_val)
        G_fft_pt = G_fft_full[:, :, ix, iy]
        G_dir_pt = greens_direct(xv, yv, z_val, N, kmax)

        diff = np.linalg.norm(G_fft_pt - G_dir_pt) / np.linalg.norm(G_dir_pt)
        print(
            f"  z={z_val}, N={N}, kmax={kmax}: ||FFT−Direct||/||Direct|| = {diff:.2e}"
        )
        del G_fft_full

    # ── TEST 2: Near-field convergence (kS·z < 0.1) ──
    print()
    print("─" * 72)
    print("TEST 2: Near-field convergence — G[0,0] component")
    print("  Error vs kmax (truncation) and N (sampling)")
    print("─" * 72)

    for z_val in [0.01, 0.02, 0.04]:
        kSz = kS_mag * z_val
        print(f"\n  z = {z_val:.2f},  kS·z = {kSz:.4f},  1/z = {1 / z_val:.0f}")

        for N, kmax in [
            (512, 50),
            (512, 100),
            (512, 200),
            (1024, 100),
            (1024, 200),
            (1024, 400),
            (2048, 200),
            (2048, 400),
        ]:
            _, x_grid, dk, dx = fft_grid(N, kmax)

            t0 = time()
            G00 = greens_fft_component(0, 0, N, kmax, z_val)
            dt = time() - t0

            # Test at on-axis and one offset point
            errs = []
            for xt, yt in [(0.0, 0.0), (z_val, z_val / 2)]:
                ix = np.argmin(np.abs(x_grid - xt))
                iy = np.argmin(np.abs(x_grid - yt))
                xv, yv = x_grid[ix], x_grid[iy]
                G_ex = exact_greens(xv, yv, z_val)
                errs.append(abs(G00[ix, iy] - G_ex[0, 0]) / abs(G_ex[0, 0]))

            del G00
            print(
                f"    N={N:4d}  kmax={kmax:4d}  dk={dk:.4f}  "
                f"err_axis={errs[0]:.2e}  err_off={errs[1]:.2e}  [{dt:.2f}s]"
            )

    # ── TEST 3: Far-field convergence (kS·z > 1) ──
    print()
    print("─" * 72)
    print("TEST 3: Far-field convergence (z=1.0, kS·z≈2.09)")
    print("─" * 72)

    z_val = 1.0
    for N, kmax in [(256, 15), (256, 25), (512, 25), (1024, 25), (2048, 25)]:
        _, x_grid, dk, dx = fft_grid(N, kmax)

        t0 = time()
        G00 = greens_fft_component(0, 0, N, kmax, z_val)
        dt = time() - t0

        errs = []
        for xt, yt in [(0.0, 0.0), (0.3, 0.4), (1.0, 0.5)]:
            ix = np.argmin(np.abs(x_grid - xt))
            iy = np.argmin(np.abs(x_grid - yt))
            G_ex = exact_greens(x_grid[ix], x_grid[iy], z_val)
            errs.append(abs(G00[ix, iy] - G_ex[0, 0]) / abs(G_ex[0, 0]))

        del G00
        print(
            f"  N={N:4d}  kmax={kmax:3d}  dk={dk:.4f}  "
            f"max_err={max(errs):.2e}  [{dt:.2f}s]"
        )

    # ── TEST 4: Full tensor near-field ──
    print()
    print("─" * 72)
    print("TEST 4: Full 3×3 tensor — near-field (z=0.02)")
    print("─" * 72)

    z_val = 0.02
    for N, kmax in [(1024, 200), (2048, 200), (2048, 400)]:
        _, x_grid, dk, dx = fft_grid(N, kmax)
        ix = np.argmin(np.abs(x_grid - 0.02))
        iy = np.argmin(np.abs(x_grid - 0.01))
        xv, yv = x_grid[ix], x_grid[iy]

        t0 = time()
        G_ff = np.zeros((3, 3), dtype=complex)
        for i in range(3):
            for j in range(i, 3):
                comp = greens_fft_component(i, j, N, kmax, z_val)
                G_ff[i, j] = comp[ix, iy]
                G_ff[j, i] = comp[ix, iy]
                del comp
        dt = time() - t0

        G_ex = exact_greens(xv, yv, z_val)
        err = np.linalg.norm(G_ff - G_ex) / np.linalg.norm(G_ex)
        print(
            f"  N={N:4d}  kmax={kmax:4d}  pt=({xv:.5f},{yv:.5f})  "
            f"Frobenius={err:.2e}  [{dt:.1f}s]"
        )

    # Best-case component detail
    print("\n  Component detail (best case above):")
    for i in range(3):
        for j in range(3):
            s, e = G_ff[i, j], G_ex[i, j]
            if abs(e) > 1e-15:
                ce = abs(s - e) / abs(e)
                print(f"    G[{i},{j}]: FFT={s:.8e}  exact={e:.8e}  err={ce:.2e}")

    # ── TEST 5: Full tensor far-field ──
    print()
    print("─" * 72)
    print("TEST 5: Full 3×3 tensor — far-field (z=1.0)")
    print("─" * 72)

    z_val = 1.0
    N, kmax = 2048, 25
    _, x_grid, dk, dx = fft_grid(N, kmax)

    for xt, yt in [(0.0, 0.0), (0.3, 0.4), (1.0, 0.5)]:
        ix = np.argmin(np.abs(x_grid - xt))
        iy = np.argmin(np.abs(x_grid - yt))
        xv, yv = x_grid[ix], x_grid[iy]

        G_ff = np.zeros((3, 3), dtype=complex)
        for i in range(3):
            for j in range(i, 3):
                comp = greens_fft_component(i, j, N, kmax, z_val)
                G_ff[i, j] = comp[ix, iy]
                G_ff[j, i] = comp[ix, iy]
                del comp

        G_ex = exact_greens(xv, yv, z_val)
        err = np.linalg.norm(G_ff - G_ex) / np.linalg.norm(G_ex)
        print(f"  pt=({xv:.4f},{yv:.4f})  Frobenius={err:.2e}")

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print("""
The 2D FFT computes the EXACT SAME trapezoidal quadrature as a direct sum,
but gives ALL spatial grid points in O(N² log N) time.

Accuracy is controlled by two parameters:
  • kmax  (truncation):  For near-field kS·z < 0.1, need kmax ≫ 1/z.
                         Rule of thumb: kmax ≈ 10/z for ~1% accuracy.
  • dk    (sampling):    Must resolve the spectral peak near kH ≈ ω/c.
                         Rule of thumb: dk ≲ 0.2 for these parameters.
  • N = 2·kmax/dk       ties the two.

Performance at kS·z < 0.1 (z=0.02):
  kmax=200, N=2048:  ~0.8% Frobenius error    (0.6s per component)
  kmax=400, N=2048:  see above                 (0.6s per component)

Performance at kS·z ≈ 2 (z=1.0):
  kmax=25,  N=2048:  ~2×10⁻⁷ error            (0.6s per component)

The FFT approach is ideal when you need G at many spatial points for
a given z (e.g., building a spatial map). For a single point, the
direct sum is equivalent.
""")
    print("Done.")
