#!/usr/bin/env python3
"""
CLI for 2D FFT verification of the elastodynamic Green's tensor.

Computes G_ij(x,y,z) via 2D IFFT of the post-residue spectral kernel
and compares with the exact Ben-Menahem & Singh spatial formula.
Provides conservative analytical error bounds for truncation and aliasing.

See README.md for full documentation.
"""

import argparse
from time import time

import numpy as np


# ═══════════════════════════════════════════════════════════════
#  Physics
# ═══════════════════════════════════════════════════════════════
def exact_greens(x, y, z, omega, rho, alpha, beta):
    """
    Ben-Menahem & Singh spatial Green's tensor.

    G_ij = f δ_ij + g γ_i γ_j

    f = (1/4πρ)[φS/(β²r) + (φP−φS)/(ω²r³) + i(φS/β − φP/α)/(ωr²)]
    g = (1/4πρ)[φP/(α²r) − φS/(β²r) + 3(φS−φP)/(ω²r³) − 3i(φS/β − φP/α)/(ωr²)]
    """
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
#  Grid
# ═══════════════════════════════════════════════════════════════
def fft_grid(N, kmax):
    """
    FFT-compatible grid.

    k-space:  k_n = (n − N/2) × dk,  dk = 2 kmax / N
    x-space:  x_m = (m − N/2) × dx,  dx = π / kmax

    Returns: k, x, dk, dx (all 1D arrays / scalars)
    """
    dk = 2.0 * kmax / N
    dx = np.pi / kmax
    k = (np.arange(N) - N // 2) * dk
    x = (np.arange(N) - N // 2) * dx
    return k, x, dk, dx


# ═══════════════════════════════════════════════════════════════
#  FFT computation — one component at a time
# ═══════════════════════════════════════════════════════════════
def greens_fft_component(i, j, N, kmax, z, omega, rho, alpha, beta):
    """
    Compute G_ij on the full spatial grid via 2D IFFT.

    Post-residue kernel (z > 0, UHP closure):
      Ĝ_ij = i/(2ρ) × [δ_ij eˢ/(β²kzS) + kᴾᵢkᴾⱼ eᴾ/(ω²kzP)
                         − kˢᵢkˢⱼ eˢ/(ω²kzS)]

    Returns (N, N) complex array in physical (centered) spatial order.
    """
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
#  Error Bounds
# ═══════════════════════════════════════════════════════════════
def truncation_error_bound(kmax, dz, omega, rho, alpha, beta):
    """
    Conservative upper bound on the truncation error from cutting off
    the spectral integral at |kH| = kmax.

    In the evanescent regime (kH > |ω/β|), the post-residue kernel
    decays as:

      |Ĝ_ij(kH; Δz)| ≤ (1/2ρ) × [1/(β²κ) + 2kH/(|ω|²κ)] × exp(−κ Δz)

    where κ = √(kH² − Re(ω/β)²).  For kH >> |ω/β|, κ ≈ kH and the
    kH/|ω|² term dominates.

    The truncation error (worst-case over all i,j) is bounded by the
    radial integral of the kernel magnitude over kH > kmax:

      |ΔG_ij| ≤ 1/(2π) ∫_{kmax}^∞ |Ĝ_ij|_max × kH dkH

    We evaluate this numerically via Gauss-Laguerre quadrature for
    tightness, and also provide the analytical asymptotic bound.

    Returns: (numerical_bound, analytical_bound, on_axis_G_magnitude)
    """
    om2 = abs(omega) ** 2
    kS_r = abs(omega) / beta
    kP_r = abs(omega) / alpha

    # ── Numerical bound: integrate |kernel|×kH from kmax to ∞ ──
    # Use substitution kH = kmax + t, Gauss-Laguerre for exp decay
    from numpy.polynomial.laguerre import laggauss

    n_quad = 64
    nodes, weights = laggauss(n_quad)

    # The integrand decays roughly as exp(-kH*dz), so substitute
    # kH = kmax + t/dz  →  dkH = dt/dz,  exp(-kH*dz) ~ exp(-kmax*dz)*exp(-t)
    def kernel_magnitude_radial(kH):
        """Max over (i,j) of |Ĝ_ij| × kH  (the radial integrand)."""
        kH2 = kH**2
        # Vertical wavenumbers (evanescent)
        kzS2 = (omega / beta) ** 2 - kH2
        kzP2 = (omega / alpha) ** 2 - kH2
        kzS = np.sqrt(kzS2 + 0j)
        kzP = np.sqrt(kzP2 + 0j)
        if np.imag(kzS) < 0:
            kzS = -kzS
        if np.imag(kzP) < 0:
            kzP = -kzP

        eS = abs(np.exp(1j * kzS * dz))
        eP = abs(np.exp(1j * kzP * dz))

        akzS = abs(kzS)
        akzP = abs(kzP)

        # Bound: max component magnitude × (1/2ρ)
        # Isotropic:  δ eS/(β²|kzS|)
        # Directional: kH²·eP/(|ω|²·|kzP|) + kH²·eS/(|ω|²·|kzS|)  [worst case]
        mag = (1.0 / (2 * rho)) * (
            eS / (beta**2 * akzS) + kH2 * eP / (om2 * akzP) + kH2 * eS / (om2 * akzS)
        )
        return mag * kH  # radial factor

    # Quadrature: ∫_0^∞ f(kmax + t/dz) exp(-t)/dz dt
    integral = 0.0
    for node, weight in zip(nodes, weights, strict=True):
        kH = kmax + node / dz
        integral += kernel_magnitude_radial(kH) * weight / dz * np.exp(node)
        # Note: laggauss weights include exp(-t), so we multiply by exp(t)
        # to undo it, then apply our own decay via kernel_magnitude_radial

    # Correct: Gauss-Laguerre computes ∫_0^∞ f(t) exp(-t) dt = Σ w_i f(t_i)
    # We want ∫_{kmax}^∞ g(kH) dkH = ∫_0^∞ g(kmax+t/dz)/dz dt
    # = ∫_0^∞ [g(kmax+t/dz)/dz × exp(t)] exp(-t) dt
    integral_num = 0.0
    for node, weight in zip(nodes, weights, strict=True):
        kH = kmax + node / dz
        val = kernel_magnitude_radial(kH) / dz * np.exp(node)
        integral_num += weight * val

    numerical_bound = integral_num / (2 * np.pi)

    # ── Analytical asymptotic bound ──
    # For kH >> |ω/β|: |kernel_max| ≈ kH/(ρ|ω|²) × exp(-kH·Δz)
    # ∫_{kmax}^∞ kH²/(ρ|ω|²) × exp(-kH·Δz) dkH
    # = exp(-p)/(ρ|ω|² Δz³) × (p² + 2p + 2),  p = kmax·Δz
    p = kmax * dz
    analytical_bound = np.exp(-p) * (p**2 + 2 * p + 2) / (2 * np.pi * rho * om2 * dz**3)

    # ── On-axis Green's function magnitude (for relative error) ──
    G_onaxis = exact_greens(0, 0, dz, omega, rho, alpha, beta)
    G_mag = np.linalg.norm(G_onaxis)

    return numerical_bound, analytical_bound, G_mag


def aliasing_error_bound(N, kmax, dz, omega, rho, alpha, beta, x=0, y=0):
    """
    Bound on the aliasing error from the finite wavenumber spacing dk.

    The FFT computes the periodized Green's function with spatial period
    L = N·π/kmax.  The aliasing error at (x,y,z) is:

      E_alias = Σ_{(m,n)≠(0,0)} G(x+mL, y+nL, z)

    We evaluate the 4 nearest images (m,n) = (±1,0), (0,±1) exactly,
    and bound the rest by geometric series.

    Returns: (alias_bound, L, nearest_image_dist)
    """
    L = N * np.pi / kmax

    # Nearest 4 images
    image_sum = 0.0
    for dm, dn in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        xi = x + dm * L
        yi = y + dn * L
        G_img = exact_greens(xi, yi, dz, omega, rho, alpha, beta)
        image_sum += np.linalg.norm(G_img)

    # Next-nearest 4 images (diagonal)
    for dm, dn in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        xi = x + dm * L
        yi = y + dn * L
        G_img = exact_greens(xi, yi, dz, omega, rho, alpha, beta)
        image_sum += np.linalg.norm(G_img)

    # Bound remaining images: each ring n has ~8n images at distance ~nL.
    # Far-field: |G| ~ exp(-Im(kS)·r)/(4πρβ²r), so ring n contributes:
    # ~8n × exp(-Im(kS)·nL)/(4πρβ²·nL) = 2·exp(-Im(kS)·nL)/(πρβ²L)
    im_kS = abs(np.imag(omega / beta))
    remainder = 0.0
    for n in range(3, 20):
        r_ring = n * L
        ring_contrib = (
            8 * n * np.exp(-im_kS * r_ring) / (4 * np.pi * rho * beta**2 * r_ring)
        )
        remainder += ring_contrib
        if ring_contrib / (image_sum + 1e-30) < 1e-10:
            break

    alias_bound = image_sum + remainder
    return alias_bound, L, L  # alias_bound, period, nearest_image_distance


def compute_error_bounds(N, kmax, dz, omega, rho, alpha, beta):
    """
    Compute and print all error bounds.

    Returns dict with all computed quantities.
    """
    trunc_num, trunc_ana, G_mag = truncation_error_bound(
        kmax, dz, omega, rho, alpha, beta
    )
    alias_abs, L, _ = aliasing_error_bound(N, kmax, dz, omega, rho, alpha, beta)

    p = kmax * dz
    dk = 2 * kmax / N
    dx = np.pi / kmax
    kS_r = abs(omega) / beta
    im_kS = abs(np.imag(omega / beta))

    rel_trunc_num = trunc_num / G_mag
    rel_trunc_ana = trunc_ana / G_mag
    rel_alias = alias_abs / G_mag
    rel_total = rel_trunc_num + rel_alias

    return {
        "N": N,
        "kmax": kmax,
        "dz": dz,
        "dk": dk,
        "dx": dx,
        "L": L,
        "p": p,  # kmax * dz
        "kS_dz": kS_r * dz,
        "G_mag": G_mag,
        "trunc_num": trunc_num,
        "trunc_ana": trunc_ana,
        "alias_abs": alias_abs,
        "rel_trunc_num": rel_trunc_num,
        "rel_trunc_ana": rel_trunc_ana,
        "rel_alias": rel_alias,
        "rel_total": rel_total,
        "im_kS_L": im_kS * L,
    }


def print_error_bounds(bounds):
    """Pretty-print error bound results."""
    b = bounds
    print(
        f"  Grid:      N={b['N']},  kmax={b['kmax']:.1f},  dk={b['dk']:.4f},  dx={b['dx']:.6f}"
    )
    print(f"  Spatial period:  L = {b['L']:.4f}")
    print(
        f"  Propagation:     Δz = {b['dz']},  kmax·Δz = {b['p']:.2f},  kS·Δz = {b['kS_dz']:.4f}"
    )
    print(f"  |G_onaxis| = {b['G_mag']:.4e}")
    print()
    print("  TRUNCATION (kH > kmax omitted):")
    print(f"    Numerical bound:   |ΔG|/|G| ≤ {b['rel_trunc_num']:.4e}")
    print(f"    Analytical bound:  |ΔG|/|G| ≤ {b['rel_trunc_ana']:.4e}")
    print(f"    (kmax·Δz = {b['p']:.2f} — need ≳15 for <10⁻⁴)")
    print()
    print("  ALIASING (periodic images):")
    print(f"    |E_alias|/|G| ≤ {b['rel_alias']:.4e}")
    print(f"    (Im(kS)·L = {b['im_kS_L']:.2f} — need ≳10 for <10⁻⁴)")
    print()
    print("  TOTAL CONSERVATIVE BOUND:")
    print(f"    |ΔG|/|G| ≤ {b['rel_total']:.4e}")


# ═══════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════
def nearest_grid_index(x_grid, target):
    return np.argmin(np.abs(x_grid - target))


def estimate_memory_gb(N):
    return N * N * 16 / 1e9


def parse_points(s):
    pts = []
    for pair in s.split(";"):
        parts = pair.strip().split(",")
        pts.append((float(parts[0]), float(parts[1])))
    return pts


def parse_component(s):
    parts = s.strip().split(",")
    return int(parts[0]), int(parts[1])


# ═══════════════════════════════════════════════════════════════
#  Actions
# ═══════════════════════════════════════════════════════════════
def run_bounds(args, omega):
    """Compute and display error bounds only (no FFT)."""
    print("Computing error bounds...")
    print()
    bounds = compute_error_bounds(
        args.N, args.kmax, args.z, omega, args.rho, args.alpha, args.beta
    )
    print_error_bounds(bounds)

    if args.sweep:
        print()
        print("─" * 72)
        print("Parameter sweep: finding optimal (N, kmax) for target error")
        print("─" * 72)
        print()
        print(
            f"  {'N':>5s}  {'kmax':>6s}  {'dk':>8s}  {'kmax·Δz':>8s}  "
            f"{'trunc':>10s}  {'alias':>10s}  {'total':>10s}  {'mem/comp':>8s}"
        )
        print("─" * 72)

        dz = args.z
        for N in [512, 1024, 2048, 4096, 8192]:
            mem = estimate_memory_gb(N)
            if mem > 8.0:
                continue
            for kmax in sorted(
                set(
                    [
                        max(25, int(5 / dz)),
                        max(25, int(10 / dz)),
                        max(25, int(15 / dz)),
                        max(50, int(20 / dz)),
                    ]
                )
            ):
                dk = 2.0 * kmax / N
                if dk > 2.0:
                    continue
                b = compute_error_bounds(
                    N, kmax, dz, omega, args.rho, args.alpha, args.beta
                )
                print(
                    f"  {N:5d}  {kmax:6.0f}  {dk:8.4f}  {b['p']:8.2f}  "
                    f"{b['rel_trunc_num']:10.2e}  {b['rel_alias']:10.2e}  "
                    f"{b['rel_total']:10.2e}  {mem:7.2f}GB"
                )


def run_default(args, omega):
    """Compute all 6 unique components, compare at test points."""
    N, kmax, z = args.N, args.kmax, args.z
    rho_v, alpha_v, beta_v = args.rho, args.alpha, args.beta
    _, x_grid, dk, dx = fft_grid(N, kmax)

    # Error bounds first
    print("─" * 72)
    print("ERROR BOUNDS")
    print("─" * 72)
    bounds = compute_error_bounds(N, kmax, z, omega, rho_v, alpha_v, beta_v)
    print_error_bounds(bounds)
    print()

    # Test points
    if args.points:
        test_pts = parse_points(args.points)
    else:
        test_pts = [(0.0, 0.0), (0.3, 0.4), (1.0, 0.5), (z, z / 2), (0.5, 0.8)]

    print("─" * 72)
    print("FFT COMPUTATION")
    print("─" * 72)
    print("Computing full 3×3 tensor (6 unique components)...")
    print(f"  Memory per component: {estimate_memory_gb(N):.2f} GB")
    print()

    G_fft_pts = {pt: np.zeros((3, 3), dtype=complex) for pt in test_pts}
    indices = {}
    for pt in test_pts:
        ix = nearest_grid_index(x_grid, pt[0])
        iy = nearest_grid_index(x_grid, pt[1])
        indices[pt] = (ix, iy)

    total_t0 = time()
    for i in range(3):
        for j in range(i, 3):
            t0 = time()
            comp = greens_fft_component(i, j, N, kmax, z, omega, rho_v, alpha_v, beta_v)
            dt = time() - t0
            print(f"  G[{i},{j}] computed in {dt:.2f}s")
            for pt in test_pts:
                ix, iy = indices[pt]
                G_fft_pts[pt][i, j] = comp[ix, iy]
                G_fft_pts[pt][j, i] = comp[ix, iy]
            del comp
    total_dt = time() - total_t0
    print(f"\nTotal FFT time: {total_dt:.1f}s")

    print(f"\n{'Point':>24s}  {'grid (x,y)':>24s}  {'Frob err':>10s}  {'bound':>10s}")
    print("─" * 72)
    for pt in test_pts:
        ix, iy = indices[pt]
        xv, yv = x_grid[ix], x_grid[iy]
        G_ex = exact_greens(xv, yv, z, omega, rho_v, alpha_v, beta_v)
        G_ff = G_fft_pts[pt]
        err = np.linalg.norm(G_ff - G_ex) / np.linalg.norm(G_ex)
        print(
            f"  ({pt[0]:6.3f},{pt[1]:6.3f})  →  ({xv:10.6f},{yv:10.6f})  "
            f"{err:.4e}  {bounds['rel_total']:.4e}"
        )

    # Component detail for first point
    pt = test_pts[0]
    ix, iy = indices[pt]
    xv, yv = x_grid[ix], x_grid[iy]
    G_ff = G_fft_pts[pt]
    G_ex = exact_greens(xv, yv, z, omega, rho_v, alpha_v, beta_v)
    print(f"\nComponent detail at ({xv:.6f}, {yv:.6f}, {z}):")
    for i in range(3):
        for j in range(3):
            s, e = G_ff[i, j], G_ex[i, j]
            if abs(e) > 1e-20:
                ce = abs(s - e) / abs(e)
                print(f"  G[{i},{j}]: FFT={s:.8e}  exact={e:.8e}  err={ce:.2e}")
            else:
                print(f"  G[{i},{j}]: FFT={s:.8e}  (exact≈0)")


def run_single_component(args, omega):
    """Compute a single G[i,j] component."""
    ci, cj = parse_component(args.component)
    N, kmax, z = args.N, args.kmax, args.z
    rho_v, alpha_v, beta_v = args.rho, args.alpha, args.beta
    _, x_grid, dk, dx = fft_grid(N, kmax)

    # Error bounds
    bounds = compute_error_bounds(N, kmax, z, omega, rho_v, alpha_v, beta_v)
    print(
        f"Bound: |ΔG|/|G| ≤ {bounds['rel_total']:.4e}  "
        f"(trunc={bounds['rel_trunc_num']:.2e}, alias={bounds['rel_alias']:.2e})"
    )
    print()

    print(f"Computing G[{ci},{cj}]...")
    print(f"  Grid: {N}×{N},  kmax={kmax},  dk={dk:.4f},  dx={dx:.6f}")
    print(f"  Memory: {estimate_memory_gb(N):.2f} GB")

    t0 = time()
    comp = greens_fft_component(ci, cj, N, kmax, z, omega, rho_v, alpha_v, beta_v)
    dt = time() - t0
    print(f"  Done in {dt:.2f}s")

    if args.points:
        test_pts = parse_points(args.points)
    else:
        test_pts = [(0.0, 0.0), (0.3, 0.4), (1.0, 0.5)]

    print(
        f"\n{'Point':>20s}  {'grid':>22s}  {'FFT':>26s}  {'exact':>26s}  {'err':>10s}"
    )
    print("─" * 110)
    for pt in test_pts:
        ix = nearest_grid_index(x_grid, pt[0])
        iy = nearest_grid_index(x_grid, pt[1])
        xv, yv = x_grid[ix], x_grid[iy]
        G_ex = exact_greens(xv, yv, z, omega, rho_v, alpha_v, beta_v)
        fft_val = comp[ix, iy]
        ex_val = G_ex[ci, cj]
        err = (
            abs(fft_val - ex_val) / abs(ex_val) if abs(ex_val) > 1e-20 else float("nan")
        )
        print(
            f"  ({pt[0]:6.3f},{pt[1]:6.3f})  ({xv:9.5f},{yv:9.5f})  "
            f"{fft_val:.8e}  {ex_val:.8e}  {err:.2e}"
        )

    if args.save:
        np.savez(
            args.save,
            component=comp,
            x_grid=x_grid,
            z=z,
            N=N,
            kmax=kmax,
            i=ci,
            j=cj,
            omega_r=omega.real,
            omega_i=omega.imag,
            rho=rho_v,
            alpha=alpha_v,
            beta=beta_v,
        )
        print(f"\nSaved to {args.save}")
    del comp


def run_sweep(args, omega):
    """Convergence sweep with error bounds."""
    z = args.z
    rho_v, alpha_v, beta_v = args.rho, args.alpha, args.beta
    kS_mag = abs(omega / beta_v)
    print(f"Convergence sweep at z={z},  kS·z={kS_mag * z:.4f}")
    print()

    print(
        f"  {'N':>5s}  {'kmax':>6s}  {'dk':>8s}  {'kmax·Δz':>8s}  "
        f"{'bound':>10s}  {'actual':>10s}  {'time':>8s}  {'mem':>7s}"
    )
    print("─" * 80)

    dz = z
    _, x_grid_ref, _, _ = fft_grid(2048, 25)

    sweep_params = []
    kmax_candidates = sorted(
        set(
            [
                max(25, int(5 / dz)),
                max(25, int(10 / dz)),
                max(50, int(15 / dz)),
                max(50, int(20 / dz)),
            ]
        )
    )
    for N in [256, 512, 1024, 2048, 4096]:
        mem = estimate_memory_gb(N)
        if mem > 4.0:
            continue
        for kmax in kmax_candidates:
            dk = 2.0 * kmax / N
            if dk > 2.0:
                continue
            sweep_params.append((N, kmax))

    if args.N >= 4096:
        sweep_params.append((args.N, args.kmax))

    for N, kmax in sweep_params:
        _, x_grid, dk, dx = fft_grid(N, kmax)
        mem = estimate_memory_gb(N)

        # Bounds
        b = compute_error_bounds(N, kmax, dz, omega, rho_v, alpha_v, beta_v)

        # Actual FFT error (G[0,0] on axis)
        t0 = time()
        G00 = greens_fft_component(0, 0, N, kmax, dz, omega, rho_v, alpha_v, beta_v)
        dt = time() - t0

        ix0 = N // 2
        G_ex = exact_greens(x_grid[ix0], x_grid[ix0], dz, omega, rho_v, alpha_v, beta_v)
        actual_err = abs(G00[ix0, ix0] - G_ex[0, 0]) / abs(G_ex[0, 0])
        del G00

        print(
            f"  {N:5d}  {kmax:6.0f}  {dk:8.4f}  {b['p']:8.2f}  "
            f"{b['rel_total']:10.2e}  {actual_err:10.2e}  "
            f"{dt:7.2f}s  {mem:.2f}GB"
        )


def run_save(args, omega):
    """Save spatial map to .npz."""
    ci, cj = parse_component(args.component) if args.component else (0, 0)
    N, kmax, z = args.N, args.kmax, args.z
    rho_v, alpha_v, beta_v = args.rho, args.alpha, args.beta
    _, x_grid, dk, dx = fft_grid(N, kmax)

    bounds = compute_error_bounds(N, kmax, z, omega, rho_v, alpha_v, beta_v)
    print(f"Bound: |ΔG|/|G| ≤ {bounds['rel_total']:.4e}")
    print(f"Computing G[{ci},{cj}] on {N}×{N} grid...")
    print(f"  Memory: {estimate_memory_gb(N):.2f} GB")

    t0 = time()
    comp = greens_fft_component(ci, cj, N, kmax, z, omega, rho_v, alpha_v, beta_v)
    dt = time() - t0
    print(f"  Done in {dt:.2f}s")

    np.savez(
        args.save,
        component=comp,
        x_grid=x_grid,
        z=z,
        i=ci,
        j=cj,
        N=N,
        kmax=kmax,
        dk=dk,
        dx=dx,
        omega_r=omega.real,
        omega_i=omega.imag,
        rho=rho_v,
        alpha=alpha_v,
        beta=beta_v,
        rel_error_bound=bounds["rel_total"],
    )
    size_mb = comp.nbytes / 1e6
    print(f"  Saved {args.save} ({size_mb:.0f} MB)")
    del comp


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(
        description="2D FFT verification of the elastodynamic Green's tensor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --z 1.0                                # Quick default test
  %(prog)s --z 0.02 --N 4096 --kmax 400           # Near-field, large grid
  %(prog)s --z 0.02 --bounds                       # Error bounds only (fast)
  %(prog)s --z 0.02 --bounds --sweep               # Sweep to find optimal params
  %(prog)s --z 1.0 --sweep                         # Convergence sweep with FFT
  %(prog)s --z 1.0 --component 0,0 --save G00.npz  # Save to file
  %(prog)s --z 0.8 --points "0.3,0.4;1.0,0.5"     # Test specific points
        """,
    )

    p.add_argument(
        "--z", type=float, required=True, help="Vertical separation (must be > 0)"
    )
    p.add_argument("--N", type=int, default=2048, help="Grid size (default: 2048)")
    p.add_argument(
        "--kmax", type=float, default=None, help="Wavenumber truncation (default: auto)"
    )
    p.add_argument("--rho", type=float, default=3.0, help="Density (default: 3.0)")
    p.add_argument(
        "--alpha", type=float, default=5.0, help="P-wave speed (default: 5.0)"
    )
    p.add_argument(
        "--beta", type=float, default=3.0, help="S-wave speed (default: 3.0)"
    )
    p.add_argument("--omega", type=float, default=None, help="Real ω (default: 2π)")
    p.add_argument(
        "--eta", type=float, default=0.03, help="Im(ω)/Re(ω) (default: 0.03)"
    )
    p.add_argument("--sweep", action="store_true", help="Convergence sweep")
    p.add_argument("--bounds", action="store_true", help="Error bounds only (no FFT)")
    p.add_argument("--component", type=str, default=None, help="Single component i,j")
    p.add_argument("--points", type=str, default=None, help='Test points "x1,y1;x2,y2"')
    p.add_argument("--save", type=str, default=None, help="Save to .npz file")

    args = p.parse_args()

    if args.z <= 0:
        p.error("z must be > 0")

    omega_r = args.omega if args.omega else 2 * np.pi
    omega = omega_r * (1 + 1j * args.eta)

    if args.kmax is None:
        args.kmax = max(25.0, 15.0 / args.z)
        args.kmax = float(int(args.kmax / 5 + 0.5) * 5)

    kS_mag = abs(omega / args.beta)

    print("=" * 72)
    print("Elastodynamic Green's Tensor — 2D FFT")
    print("=" * 72)
    print(f"  Material:  ρ={args.rho}, α={args.alpha}, β={args.beta}")
    print(f"  Frequency: ω = {omega_r:.4f}(1+{args.eta}i) = {omega:.6f}")
    print(f"  |kP|={abs(omega / args.alpha):.4f},  |kS|={kS_mag:.4f}")
    print(f"  z = {args.z},  kS·z = {kS_mag * args.z:.4f}")
    print(f"  N = {args.N},  kmax = {args.kmax}")
    dk = 2.0 * args.kmax / args.N
    dx = np.pi / args.kmax
    print(f"  dk = {dk:.4f},  dx = {dx:.6f}")
    print(f"  Memory per component: {estimate_memory_gb(args.N):.2f} GB")
    print()

    if args.bounds:
        run_bounds(args, omega)
    elif args.sweep:
        run_sweep(args, omega)
    elif args.save and not args.component:
        args.component = "0,0"
        run_save(args, omega)
    elif args.save:
        run_save(args, omega)
    elif args.component:
        run_single_component(args, omega)
    else:
        run_default(args, omega)


if __name__ == "__main__":
    main()
