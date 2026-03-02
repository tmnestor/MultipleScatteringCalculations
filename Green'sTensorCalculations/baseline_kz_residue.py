"""
BASELINE: Spectral ↔ Spatial Green's tensor via kz residues.

Starting point:
  Equation of motion (freq domain, isotropic):
    [ρω² δ_ij - μK² δ_ij - (λ+μ) k_i k_j] G̃_jn(k) = -δ_in

  Spectral Green's tensor (Christoffel inverse, Sherman-Morrison):
    G̃_ij(k) = -δ_ij / [ρ(ω² - β²K²)]
               - (α²-β²) k_i k_j / [ρ(ω²-β²K²)(ω²-α²K²)]

  where K² = kx² + ky² + kz², k = (kx, ky, kz).

Spatial Green's tensor via inverse Fourier transform:
  G_ij(x) = ∫∫∫ G̃_ij(k) exp(ik·x) d³k / (2π)³

Strategy:
  1. Evaluate kz integral via residue theorem (z > 0, close in UHP)
  2. Numerically integrate the 2D (kx, ky) integral

kz poles of G̃:
  - S-wave: kz = ±kzS, where kzS = √[(ω/β)² - kH²],  kH² = kx²+ky²
  - P-wave: kz = ±kzP, where kzP = √[(ω/α)² - kH²]

  With Im(ω) > 0, Im(kzS) > 0 and Im(kzP) > 0, so +kzS, +kzP are in UHP.

Residues (derived below):
  At kz = kzS:
    R^S_ij = [δ_ij/(β²kzS) - k^S_i k^S_j/(ω²kzS)] × exp(ikzS·z) / (2ρ)

  At kz = kzP:
    R^P_ij = k^P_i k^P_j/(ω²kzP) × exp(ikzP·z) / (2ρ)

  where k^S = (kx, ky, kzS), k^P = (kx, ky, kzP).

Post-residue (kz integral = 2πi × sum of UHP residues, divided by 2π):
  Ĝ_ij(kx,ky; z) = i/(2ρ) × [δ_ij eS/(β²kzS)
                                + k^P_i k^P_j eP/(ω²kzP)
                                - k^S_i k^S_j eS/(ω²kzS)]

  where eS = exp(ikzS·z), eP = exp(ikzP·z).

Full spatial Green's tensor:
  G_ij(x,y,z) = 1/(4π²) ∫∫ Ĝ_ij(kx,ky; z) exp(i[kx·x + ky·y]) dkx dky

Exact spatial formula (Ben-Menahem & Singh):
  G_ij = f·δ_ij + g·γ_i·γ_j
  f = (1/4πρ)[φS/(β²r) + (φP-φS)/(ω²r³) + i(φS/β - φP/α)/(ωr²)]
  g = (1/4πρ)[φP/(α²r) - φS/(β²r) + 3(φS-φP)/(ω²r³) - 3i(φS/β - φP/α)/(ωr²)]
"""

from time import time

import numpy as np

# ── Parameters ──
rho = 3.0
alpha = 5.0
beta = 3.0
omega_r = 2 * np.pi
eta = 0.03
omega = omega_r * (1 + 1j * eta)


# ═══════════════════════════════════════════════════════════════
#  Exact spatial Green's tensor
# ═══════════════════════════════════════════════════════════════
def exact_greens(x, y, z, omega, rho, alpha, beta):
    """Ben-Menahem & Singh formula."""
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
#  Post-residue kernel (kz integral evaluated analytically)
# ═══════════════════════════════════════════════════════════════
def post_residue_kernel(kx, ky, z, omega, rho, alpha, beta):
    """
    Ĝ_ij(kx,ky; z) after evaluating kz integral via residues for z > 0.
    Returns 3×3 complex matrix.

    Coordinates: index 0 = x, 1 = y, 2 = z
    k^S = (kx, ky, kzS),  k^P = (kx, ky, kzP)
    """
    kH2 = kx**2 + ky**2
    kzS = np.sqrt((omega / beta) ** 2 - kH2)
    kzP = np.sqrt((omega / alpha) ** 2 - kH2)

    # Ensure Im > 0 (upper half plane)
    if np.imag(kzS) < 0:
        kzS = -kzS
    if np.imag(kzP) < 0:
        kzP = -kzP

    eS = np.exp(1j * kzS * z)
    eP = np.exp(1j * kzP * z)

    # Wave vectors at the poles
    kS_vec = np.array([kx, ky, kzS])
    kP_vec = np.array([kx, ky, kzP])

    G = np.zeros((3, 3), dtype=complex)
    for i in range(3):
        for j in range(3):
            dij = 1.0 if i == j else 0.0
            G[i, j] = (1j / (2 * rho)) * (
                dij * eS / (beta**2 * kzS)
                + kP_vec[i] * kP_vec[j] * eP / (omega**2 * kzP)
                - kS_vec[i] * kS_vec[j] * eS / (omega**2 * kzS)
            )
    return G


# ═══════════════════════════════════════════════════════════════
#  Numerical kz integral (for verification of residue formula)
# ═══════════════════════════════════════════════════════════════
def numerical_kz_integral(kx, ky, z, omega, rho, alpha, beta, kz_max=80, nkz=8192):
    """
    Numerically integrate G̃_ij(kx,ky,kz) exp(ikz·z) dkz/(2π)
    over a 1D kz grid. No residue theorem.
    """
    kz_1d = np.linspace(-kz_max, kz_max, nkz)
    dkz = kz_1d[1] - kz_1d[0]

    kH2 = kx**2 + ky**2
    K2 = kH2 + kz_1d**2

    denomS = rho * (omega**2 - beta**2 * K2)
    denomP = omega**2 - alpha**2 * K2
    coeff = (alpha**2 - beta**2) / (denomS * denomP)

    phase = np.exp(1j * kz_1d * z)

    kvec_all = np.array([np.full_like(kz_1d, kx), np.full_like(kz_1d, ky), kz_1d])

    G = np.zeros((3, 3), dtype=complex)
    for i in range(3):
        for j in range(3):
            dij = 1.0 if i == j else 0.0
            integrand = (-dij / denomS - coeff * kvec_all[i] * kvec_all[j]) * phase
            G[i, j] = np.sum(integrand) * dkz / (2 * np.pi)
    return G


# ═══════════════════════════════════════════════════════════════
#  2D (kx,ky) integral of post-residue kernel
# ═══════════════════════════════════════════════════════════════
def spectral_2d_integral(x, y, z, omega, rho, alpha, beta, kmax=25.0, nk=512):
    """
    G_ij(x,y,z) = 1/(4π²) ∫∫ Ĝ_ij(kx,ky;z) exp(i[kx·x+ky·y]) dkx dky

    Vectorized over (kx,ky) grid.
    """
    k1d = np.linspace(-kmax, kmax, nk)
    dk = k1d[1] - k1d[0]
    KX, KY = np.meshgrid(k1d, k1d, indexing="ij")

    kH2 = KX**2 + KY**2
    kzS = np.sqrt((omega / beta) ** 2 - kH2)
    kzP = np.sqrt((omega / alpha) ** 2 - kH2)

    # Ensure Im > 0
    kzS = np.where(np.imag(kzS) < 0, -kzS, kzS)
    kzP = np.where(np.imag(kzP) < 0, -kzP, kzP)

    eS = np.exp(1j * kzS * z)
    eP = np.exp(1j * kzP * z)
    phase = np.exp(1j * (KX * x + KY * y))

    kS_vec = [KX, KY, kzS]
    kP_vec = [KX, KY, kzP]

    G = np.zeros((3, 3), dtype=complex)
    for i in range(3):
        for j in range(3):
            dij = 1.0 if i == j else 0.0
            kernel = (
                dij * eS / (beta**2 * kzS)
                + kP_vec[i] * kP_vec[j] * eP / (omega**2 * kzP)
                - kS_vec[i] * kS_vec[j] * eS / (omega**2 * kzS)
            )
            integrand = kernel * phase
            G[i, j] = np.sum(integrand) * dk**2

    G *= 1j / (2 * rho * (2 * np.pi) ** 2)  # = i/(8π²ρ)
    return G


# ═══════════════════════════════════════════════════════════════
#  TESTS
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 72)
    print("BASELINE: kz residue → 2D integral → exact spatial Green's tensor")
    print("=" * 72)
    print(f"rho={rho}, alpha={alpha}, beta={beta}")
    print(f"omega = 2π(1+{eta}i) = {omega:.6f}")
    print()

    # ── STEP 1: Verify kz residue against numerical kz integral ──
    print("─" * 72)
    print("STEP 1: Verify kz residue formula vs numerical kz integral")
    print("        at individual (kx, ky) points")
    print("─" * 72)

    z_val = 1.0
    kx_ky_tests = [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.5),
        (1.5, 0.7),
        (3.0, 2.0),  # near S-wave pole
    ]

    for kx_t, ky_t in kx_ky_tests:
        G_residue = post_residue_kernel(kx_t, ky_t, z_val, omega, rho, alpha, beta)
        G_numerical = numerical_kz_integral(kx_t, ky_t, z_val, omega, rho, alpha, beta)

        err = np.linalg.norm(G_residue - G_numerical) / np.linalg.norm(G_numerical)
        print(
            f"  kx={kx_t:5.1f}, ky={ky_t:5.1f}: "
            f"||residue-numerical||/||numerical|| = {err:.2e}"
        )

        if err > 0.01:
            print("  *** MISMATCH — showing components:")
            for i in range(3):
                for j in range(i, 3):
                    r, n = G_residue[i, j], G_numerical[i, j]
                    if abs(n) > 1e-20:
                        ce = abs(r - n) / abs(n)
                        print(
                            f"      [{i},{j}]: res={r:.6e}  num={n:.6e}  err={ce:.2e}"
                        )

    # Also test z = 0.5 and z = 2.0
    for z_t in [0.5, 2.0]:
        kx_t, ky_t = 1.5, 0.7
        G_residue = post_residue_kernel(kx_t, ky_t, z_t, omega, rho, alpha, beta)
        G_numerical = numerical_kz_integral(kx_t, ky_t, z_t, omega, rho, alpha, beta)
        err = np.linalg.norm(G_residue - G_numerical) / np.linalg.norm(G_numerical)
        print(
            f"  kx={kx_t:5.1f}, ky={ky_t:5.1f}, z={z_t:.1f}: "
            f"||residue-numerical||/||numerical|| = {err:.2e}"
        )

    # ── STEP 2: 2D integral vs exact spatial formula ──
    print()
    print("─" * 72)
    print("STEP 2: 2D (kx,ky) integral of post-residue kernel")
    print("        vs exact spatial Green's tensor")
    print("─" * 72)

    test_points = [
        # (x, y, z) — all with z > 0
        (0.0, 0.0, 1.0),  # on-axis z
        (0.5, 0.0, 1.0),  # slight offset
        (0.0, 0.5, 1.0),
        (0.3, 0.4, 0.8),  # general point
        (0.8, 0.6, 1.5),  # further away
        (0.0, 0.0, 0.5),  # closer
    ]

    nk = 1024
    kmax = 25.0

    for x, y, z in test_points:
        t0 = time()
        G_spec = spectral_2d_integral(
            x, y, z, omega, rho, alpha, beta, kmax=kmax, nk=nk
        )
        dt = time() - t0
        G_exact = exact_greens(x, y, z, omega, rho, alpha, beta)

        err = np.linalg.norm(G_spec - G_exact) / np.linalg.norm(G_exact)
        print(f"  ({x:.1f}, {y:.1f}, {z:.1f}):  err = {err:.4e}  [{dt:.1f}s]")

    # ── STEP 3: Convergence study ──
    print()
    print("─" * 72)
    print("STEP 3: Convergence study at (0.3, 0.4, 0.8)")
    print("─" * 72)

    x, y, z = 0.3, 0.4, 0.8
    G_exact = exact_greens(x, y, z, omega, rho, alpha, beta)

    for nk in [128, 256, 512, 1024]:
        for kmax in [15.0, 20.0, 25.0, 30.0]:
            t0 = time()
            G_spec = spectral_2d_integral(
                x, y, z, omega, rho, alpha, beta, kmax=kmax, nk=nk
            )
            dt = time() - t0
            err = np.linalg.norm(G_spec - G_exact) / np.linalg.norm(G_exact)
            if dt < 60:
                print(f"  nk={nk:4d}  kmax={kmax:5.1f}  err={err:.4e}  [{dt:.1f}s]")
            else:
                print(
                    f"  nk={nk:4d}  kmax={kmax:5.1f}  err={err:.4e}  [{dt:.1f}s] (slow)"
                )

    # ── STEP 4: Show component-level detail at best point ──
    print()
    print("─" * 72)
    print("STEP 4: Component-level detail at (0.3, 0.4, 0.8), nk=1024, kmax=25")
    print("─" * 72)

    G_spec = spectral_2d_integral(x, y, z, omega, rho, alpha, beta, kmax=25.0, nk=1024)
    G_exact = exact_greens(x, y, z, omega, rho, alpha, beta)

    print(
        f"  Frobenius error = {np.linalg.norm(G_spec - G_exact) / np.linalg.norm(G_exact):.4e}"
    )
    print()
    for i in range(3):
        for j in range(3):
            s, e = G_spec[i, j], G_exact[i, j]
            if abs(e) > 1e-15:
                ce = abs(s - e) / abs(e)
                print(f"  G[{i},{j}]:  spectral = {s:.8e}")
                print(f"          exact    = {e:.8e}  err = {ce:.2e}")
            else:
                print(f"  G[{i},{j}]:  spectral = {s:.8e}  (exact ≈ 0)")

    print("\nDone.")
