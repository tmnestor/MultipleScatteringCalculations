#!/usr/bin/env python3
"""
Lattice Green's tensor for multiple scattering of cubic scatterers.

Computes the elastodynamic Green's tensor G_ij(R_m - R_n) between all
pairs of scatterer centres on a 2D square lattice at z=0, with lattice
spacing d.

Three layers:
  1. Spatial evaluation  — exact Ben-Menahem & Singh formula at each separation
  2. Spectral evaluation — 2D IFFT of the post-kz-residue kernel (z=0)
  3. Block-Toeplitz mat-vec — FFT-accelerated matrix-vector product

The spectral kernel at z=0 (eS=eP=1):

  G_ij(kx,ky) = (i/2rho) * [delta_ij/(beta^2 kzS)
                             + kP_i kP_j/(omega^2 kzP)
                             - kS_i kS_j/(omega^2 kzS)]

where kzP = sqrt(kP^2 - kx^2 - ky^2), kzS = sqrt(kS^2 - kx^2 - ky^2),
kP = (kx,ky,kzP), kS = (kx,ky,kzS), Im(kz) >= 0.
"""

from __future__ import annotations

from time import time
from typing import TYPE_CHECKING, Annotated, Callable

import numpy as np
import typer
from scipy.special import jv

from .horizontal_greens import exact_greens, exact_propagator_9x9

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ═══════════════════════════════════════════════════════════════
#  Spectral kernel at z=0
# ═══════════════════════════════════════════════════════════════
def spectral_kernel_z0(
    KX: NDArray[np.complexfloating],
    KY: NDArray[np.complexfloating],
    omega: complex,
    rho: float,
    alpha: float,
    beta: float,
) -> NDArray[np.complexfloating]:
    """
    Post-kz-residue spectral kernel at z=0 for all 9 tensor components.

    At z=0, exp(i kzP z) = exp(i kzS z) = 1, so the kernel simplifies.

    Args:
        KX: 2D array of kx values.
        KY: 2D array of ky values.
        omega: complex angular frequency.
        rho: density.
        alpha: P-wave speed.
        beta: S-wave speed.

    Returns:
        G: shape (*KX.shape, 3, 3) complex array — spectral kernel.
    """
    kP = omega / alpha
    kS = omega / beta

    kH2 = KX**2 + KY**2
    kzP = np.sqrt(kP**2 - kH2 + 0j)
    kzS = np.sqrt(kS**2 - kH2 + 0j)

    # Enforce Im(kz) >= 0 (radiation condition)
    kzP = np.where(np.imag(kzP) < 0, -kzP, kzP)
    kzS = np.where(np.imag(kzS) < 0, -kzS, kzS)

    # Wave vector components at the poles
    kPv = [KX, KY, kzP]
    kSv = [KX, KY, kzS]

    shape = KX.shape
    G = np.zeros((*shape, 3, 3), dtype=complex)
    prefac = 1j / (2 * rho)
    om2 = omega**2

    for i in range(3):
        for j in range(3):
            dij = 1.0 if i == j else 0.0
            G[..., i, j] = prefac * (
                dij / (beta**2 * kzS)
                + kPv[i] * kPv[j] / (om2 * kzP)
                - kSv[i] * kSv[j] / (om2 * kzS)
            )

    return G


# ═══════════════════════════════════════════════════════════════
#  Screened-Coulomb subtraction for convergence acceleration
# ═══════════════════════════════════════════════════════════════
#
# At z=0 the spectral kernel decays as c/kH (1/kH for each diagonal
# component). This slow algebraic decay causes large truncation and
# aliasing errors in the 2D IFFT.
#
# We use a "screened Coulomb" subtraction: replace c/kH with
#   c / √(kH² + kc²)
# where kc is a real screening parameter (typically kc ≈ |kS|).
#
# Properties:
#   - Matches c/kH for kH >> kc (correct asymptotic)
#   - Finite at kH = 0 (no divergence)
#   - Known 2D inverse FT: c·exp(-kc·r) / (2πr)  [Sommerfeld integral]
#   - The spatial transform decays EXPONENTIALLY, so:
#     (a) the residual kernel decays faster than 1/kH
#     (b) the subtracted function's aliases are exponentially suppressed
#
# The residual kernel G - G_sub decays as O(1/kH³) for large kH
# (the 1/kH parts cancel) and is finite at kH = 0 (since G_sub is
# finite there too). This dramatically improves FFT convergence.
# ═══════════════════════════════════════════════════════════════


def _screened_coulomb_coefficients(
    omega: complex, rho: float, alpha: float, beta: float
) -> tuple[complex, complex]:
    """
    Leading-order asymptotic coefficients for diagonal kernel components.

    For kH >> |kS|:
      G_xx = G_yy ~ c_H / kH,  G_zz ~ c_V / kH

    Returns:
        c_H: horizontal coefficient 1/(2μ)
        c_V: vertical coefficient (α²+β²)/(4ρα²β²)
    """
    mu = rho * beta**2
    c_H = 1.0 / (2.0 * mu)
    c_V = (alpha**2 + beta**2) / (4.0 * rho * alpha**2 * beta**2)
    return c_H, c_V


def screened_kernel_z0(
    KX: NDArray[np.complexfloating],
    KY: NDArray[np.complexfloating],
    omega: complex,
    rho: float,
    alpha: float,
    beta: float,
    kc: float,
) -> NDArray[np.complexfloating]:
    """
    Screened-Coulomb subtraction kernel: c_ii / √(kH² + kc²).

    Matches the diagonal components' 1/kH asymptotic for kH >> kc,
    and is finite at kH = 0. Off-diagonal terms are zero (they
    decay as 1/kH³ and don't need correction).

    Args:
        KX, KY: 2D wavenumber arrays.
        omega: complex angular frequency.
        rho, alpha, beta: medium parameters.
        kc: screening parameter (real, positive).

    Returns:
        G: shape (*KX.shape, 3, 3) complex array.
    """
    c_H, c_V = _screened_coulomb_coefficients(omega, rho, alpha, beta)

    kH2 = KX**2 + KY**2
    inv_screened = 1.0 / np.sqrt(kH2 + kc**2 + 0j)

    shape = KX.shape
    G = np.zeros((*shape, 3, 3), dtype=complex)
    G[..., 0, 0] = c_H * inv_screened
    G[..., 1, 1] = c_H * inv_screened
    G[..., 2, 2] = c_V * inv_screened

    return G


def screened_spatial_z0(
    x: float,
    y: float,
    omega: complex,
    rho: float,
    alpha: float,
    beta: float,
    kc: float,
) -> np.ndarray:
    """
    Closed-form 2D inverse FT of the screened-Coulomb kernel.

    FT⁻¹[c / √(kH² + kc²)] = c · exp(-kc·r) / (2πr)

    (Sommerfeld integral / Weber-Schafheitlin formula)

    Args:
        x, y: spatial coordinates.
        omega: complex angular frequency.
        rho, alpha, beta: medium parameters.
        kc: screening parameter (real, positive).

    Returns:
        G: (3, 3) complex array.
    """
    c_H, c_V = _screened_coulomb_coefficients(omega, rho, alpha, beta)

    r = np.sqrt(x**2 + y**2)
    G = np.zeros((3, 3), dtype=complex)

    if r < 1e-30:
        return G

    screening = np.exp(-kc * r) / (2.0 * np.pi * r)
    G[0, 0] = c_H * screening
    G[1, 1] = c_H * screening
    G[2, 2] = c_V * screening

    return G


# ═══════════════════════════════════════════════════════════════
#  FCC Hankel transform helpers
# ═══════════════════════════════════════════════════════════════


def _radial_kernels_z0(
    kH: NDArray[np.complexfloating],
    omega: complex,
    rho: float,
    alpha: float,
    beta: float,
) -> tuple[
    NDArray[np.complexfloating],
    NDArray[np.complexfloating],
    NDArray[np.complexfloating],
]:
    """
    Radial kernel functions A, B, C for the angular-harmonic decomposition.

    After taking the residue in kz and decomposing the 2D spectral kernel
    into angular harmonics (m=0, m=2), three scalar radial functions emerge:

        A(kH) = i / (2 rho beta^2 kzS)              -- isotropic S-wave
        B(kH) = i kH^2 / (2 rho omega^2) (1/kzP - 1/kzS) -- directional coupling
        C(kH) = i (kzP - kzS) / (2 rho omega^2)     -- vertical coupling

    Args:
        kH: horizontal wavenumber magnitudes (real, positive).
        omega: complex angular frequency.
        rho: density.
        alpha: P-wave speed.
        beta: S-wave speed.

    Returns:
        (A, B, C): arrays of same shape as kH.
    """
    kP = omega / alpha
    kS = omega / beta
    kH2 = kH**2 + 0j

    kzP = np.sqrt(kP**2 - kH2)
    kzS = np.sqrt(kS**2 - kH2)
    kzP = np.where(np.imag(kzP) < 0, -kzP, kzP)
    kzS = np.where(np.imag(kzS) < 0, -kzS, kzS)

    om2 = omega**2
    A = 1j / (2 * rho * beta**2 * kzS)
    B = 1j * kH2 / (2 * rho * om2) * (1.0 / kzP - 1.0 / kzS)
    C = 1j * (kzP - kzS) / (2 * rho * om2)

    return A, B, C


def _clenshaw_curtis_nodes_weights(
    N: int, a: float, b: float
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Clenshaw-Curtis quadrature nodes and weights on [a, b].

    Uses the Type II (closed) rule with FFT-based weight computation.

    Args:
        N: number of nodes (must be >= 2).
        a: left endpoint.
        b: right endpoint.

    Returns:
        (nodes, weights): arrays of length N on [a, b].
    """
    n = N - 1  # polynomial degree
    theta = np.pi * np.arange(N) / n
    nodes_ref = np.cos(theta)  # in [-1, 1], from +1 to -1

    # Weights via Trefethen's clencurt algorithm (Spectral Methods in MATLAB)
    weights_ref = np.zeros(N)
    ii = np.arange(1, n)  # interior indices
    v = np.ones(n - 1)

    if n % 2 == 0:
        weights_ref[0] = weights_ref[n] = 1.0 / (n * n - 1)
        for k in range(1, n // 2):
            v -= 2 * np.cos(2 * k * theta[ii]) / (4 * k * k - 1)
        v -= np.cos(n * theta[ii]) / (n * n - 1)
    else:
        weights_ref[0] = weights_ref[n] = 1.0 / (n * n)
        for k in range(1, (n + 1) // 2):
            v -= 2 * np.cos(2 * k * theta[ii]) / (4 * k * k - 1)

    weights_ref[ii] = 2 * v / n

    # Map to [a, b]
    half_len = (b - a) / 2.0
    mid = (a + b) / 2.0
    nodes = mid + half_len * nodes_ref
    weights = weights_ref * half_len

    return nodes, weights


def _hankel_transform_cc(
    r_vals: NDArray[np.floating],
    omega: complex,
    rho: float,
    alpha: float,
    beta: float,
    N_per_seg: int = 96,
    K_max: float | None = None,
    kc_factor: float = 1.0,
) -> tuple[
    NDArray[np.complexfloating],
    NDArray[np.complexfloating],
    NDArray[np.complexfloating],
]:
    """
    Compute Hankel transforms H0, H2, V0 via Filon-Clenshaw-Curtis quadrature.

    Reduces the 2D inverse FT to three 1D integrals by exploiting the angular
    harmonic decomposition (m=0, m=2). Asymptotic tails are subtracted for
    convergence, with the analytical spatial transforms added back.

    Args:
        r_vals: radial distances at which to evaluate.
        omega: complex angular frequency.
        rho: density.
        alpha, beta: P-wave and S-wave speeds.
        N_per_seg: CC nodes per integration segment.
        K_max: upper integration limit (default: 4*|kS|).
        kc_factor: screening parameter factor.

    Returns:
        (H0, H2, V0): arrays of shape (len(r_vals),).
    """
    kP = omega / alpha
    kS = omega / beta
    kP_re = abs(kP.real) if np.iscomplex(kP) else abs(kP)
    kS_re = abs(kS.real) if np.iscomplex(kS) else abs(kS)

    if K_max is None:
        K_max = 20.0 * abs(kS)

    # Segment boundaries bracketing branch points, plus a tail segment
    inner_bounds = sorted({0.0, kP_re, kS_re, 2.0 * kS_re})
    inner_bounds = [b for b in inner_bounds if b < K_max]
    inner_bounds.append(min(4.0 * kS_re, K_max))
    if inner_bounds[-1] < K_max:
        inner_bounds.append(K_max)

    # Build composite CC quadrature
    all_nodes = []
    all_weights = []
    for i in range(len(inner_bounds) - 1):
        a, b = inner_bounds[i], inner_bounds[i + 1]
        if b - a < 1e-14:
            continue
        nodes, weights = _clenshaw_curtis_nodes_weights(N_per_seg, a, b)
        # Avoid duplicate endpoints between segments
        if i > 0:
            nodes = nodes[:-1]  # drop last (= next segment's first)
            weights = weights[:-1]
        all_nodes.append(nodes)
        all_weights.append(weights)

    kH = np.concatenate(all_nodes)
    w = np.concatenate(all_weights)

    # Evaluate radial kernels
    A, B, C = _radial_kernels_z0(kH, omega, rho, alpha, beta)

    # Asymptotic subtraction: two-level scheme for O(1/kH^5) residual.
    #
    # Level 1: subtract c/sqrt(kH^2+kc^2) to cancel O(1/kH) asymptotic.
    # Level 2: subtract c3/(kH^2+kc^2)^{3/2} to cancel O(1/kH^3) residual.
    #
    # Leading-order coefficients:
    #   (A+B/2) ~ c_H0/kH,   B/2 ~ c_B2/kH,   (A+C) ~ c_V/kH
    c_H, c_V = _screened_coulomb_coefficients(omega, rho, alpha, beta)
    kP2 = (omega / alpha) ** 2
    kS2 = (omega / beta) ** 2
    c_B2 = (kP2 - kS2) / (8 * rho * omega**2)
    c_H0 = c_H + c_B2
    kc = kc_factor * abs(kS)

    kH2_kc2 = kH**2 + kc**2 + 0j
    inv_sc1 = 1.0 / np.sqrt(kH2_kc2)  # 1/sqrt(kH^2+kc^2)
    inv_sc3 = inv_sc1**3  # 1/(kH^2+kc^2)^{3/2}

    # Level-1 subtraction
    sub1_H0 = c_H0 * kH * inv_sc1
    sub1_H2 = c_B2 * kH * inv_sc1
    sub1_V0 = c_V * kH * inv_sc1

    # Level-1 residual
    res1_H0 = (A + B / 2) * kH - sub1_H0
    res1_H2 = (B / 2) * kH - sub1_H2
    res1_V0 = (A + C) * kH - sub1_V0

    # Compute O(1/kH^3) coefficient numerically from a large-kH sample
    kH_probe = np.array([K_max * 10])
    Ap, Bp, Cp = _radial_kernels_z0(kH_probe, omega, rho, alpha, beta)
    inv_p1 = 1.0 / np.sqrt(kH_probe**2 + kc**2 + 0j)
    c3_H0 = (
        (Ap[0] + Bp[0] / 2) * kH_probe[0] - c_H0 * kH_probe[0] * inv_p1[0]
    ) * kH_probe[0] ** 2
    c3_H2 = ((Bp[0] / 2) * kH_probe[0] - c_B2 * kH_probe[0] * inv_p1[0]) * kH_probe[
        0
    ] ** 2
    c3_V0 = ((Ap[0] + Cp[0]) * kH_probe[0] - c_V * kH_probe[0] * inv_p1[0]) * kH_probe[
        0
    ] ** 2

    # Level-2 subtraction: c3*kH/(kH^2+kc^2)^{3/2} matches c3/kH^2 asymptotic
    # (the residual integrand from level-1 is O(1/kH^2), need kH factor!)
    sub2_H0 = c3_H0 * kH * inv_sc3
    sub2_H2 = c3_H2 * kH * inv_sc3
    sub2_V0 = c3_V0 * kH * inv_sc3

    # Final residual: O(1/kH^5) decay
    integrand_H0 = res1_H0 - sub2_H0
    integrand_H2 = res1_H2 - sub2_H2
    integrand_V0 = res1_V0 - sub2_V0

    # Bessel evaluation: shape (N_r, N_total)
    r_col = r_vals[:, np.newaxis]
    kH_row = kH[np.newaxis, :]
    kr = kH_row * r_col
    J0_vals = jv(0, kr)
    J2_vals = jv(2, kr)

    # Weighted sums
    w_row = w[np.newaxis, :]
    H0_res = np.sum(w_row * integrand_H0[np.newaxis, :] * J0_vals, axis=1) / (2 * np.pi)
    H2_res = np.sum(w_row * integrand_H2[np.newaxis, :] * J2_vals, axis=1) / (2 * np.pi)
    V0_res = np.sum(w_row * integrand_V0[np.newaxis, :] * J0_vals, axis=1) / (2 * np.pi)

    # Analytical add-backs for level-1 and level-2 subtractions.
    #
    # Order-0 Hankel transforms:
    #   int kH/sqrt(kH^2+kc^2) J_0(kH r) dkH = exp(-kc r)/r
    #   int kH/(kH^2+kc^2)^{3/2} J_0(kH r) dkH = exp(-kc r)/kc
    # Order-2 Hankel transforms:
    #   int kH/sqrt(kH^2+kc^2) J_2(kH r) dkH
    #     = [2 - (2+kc r) exp(-kc r)] / (kc r^2)
    #   int kH/(kH^2+kc^2)^{3/2} J_2(kH r) dkH
    #     = [2(1-exp(-kc r)) - kc r exp(-kc r)] / (kc^2 r^2) - exp(-kc r)/kc
    #     (derived from order-2 recursion on the order-0 result)
    r = r_vals
    exp_kc_r = np.exp(-kc * r)
    kc_r = kc * r

    # Level-1 add-backs
    ab1_J0 = exp_kc_r / (2 * np.pi * r)
    ab1_J2 = (2 - (2 + kc_r) * exp_kc_r) / (2 * np.pi * kc * r**2)

    # Level-2 add-backs
    # Order-2 uses recursion and d/dkc differentiation:
    #   int J_1(kH r)/(kH^2+kc^2)^{3/2} dkH = [1-(1+kc r)exp(-kc r)]/(kc^3 r)
    #   int kH/(kH^2+kc^2)^{3/2} J_2(kH r) dkH
    #     = (2/r)*j1_integral - exp(-kc r)/kc
    j1_integral = (1 - (1 + kc_r) * exp_kc_r) / (kc**3 * r)
    ab2_J0 = exp_kc_r / (2 * np.pi * kc)
    ab2_J2 = ((2 / r) * j1_integral - exp_kc_r / kc) / (2 * np.pi)

    H0 = H0_res + c_H0 * ab1_J0 + c3_H0 * ab2_J0
    H2 = H2_res + c_B2 * ab1_J2 + c3_H2 * ab2_J2
    V0 = V0_res + c_V * ab1_J0 + c3_V0 * ab2_J0

    return H0, H2, V0


# ═══════════════════════════════════════════════════════════════
#  D4h symmetry helpers
# ═══════════════════════════════════════════════════════════════
#
# Legacy 3×3 sign tables (x=0, y=1, z=2 convention — used by exact_greens)
_REFL_X_SIGN = np.array(
    [[1, -1, -1], [-1, 1, 1], [-1, 1, 1]],
    dtype=float,
)
_REFL_Y_SIGN = np.array(
    [[1, -1, 1], [-1, 1, -1], [1, -1, 1]],
    dtype=float,
)
_PERM_90 = np.array(
    [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
)

# ── 9×9 propagator D4h matrices ──────────────────────────────
#
# Seismological index convention: z=0, x=1, y=2
# VOIGT_PAIRS: (0,0)=zz, (1,1)=xx, (2,2)=yy, (1,2)=xy, (0,2)=zy, (0,1)=zx
#
# The 9×9 propagator transforms under point operation R as:
#   P'_9x9 = R_9x9 @ P_9x9 @ R_9x9^T
# where R_9x9 = block_diag(R_3x3, V_R_6x6):
#   R_3x3: Cartesian transformation (seismological ordering)
#   V_R_6x6: Voigt transformation for same operation

# --- x-reflection: flips index 1 (x in z,x,y) ---
# Voigt pairs with one x-index: (1,2)=xy and (0,1)=zx → flip
_REFL_X_9x9 = np.zeros((9, 9))
_REFL_X_9x9[:3, :3] = np.diag([1.0, -1.0, 1.0])
_REFL_X_9x9[3:, 3:] = np.diag([1.0, 1.0, 1.0, -1.0, 1.0, -1.0])

# --- y-reflection: flips index 2 (y in z,x,y) ---
# Voigt pairs with one y-index: (1,2)=xy and (0,2)=zy → flip
_REFL_Y_9x9 = np.zeros((9, 9))
_REFL_Y_9x9[:3, :3] = np.diag([1.0, 1.0, -1.0])
_REFL_Y_9x9[3:, 3:] = np.diag([1.0, 1.0, 1.0, -1.0, -1.0, 1.0])

# --- x↔y swap: swaps indices 1↔2 (x↔y in z,x,y) ---
# Voigt under x↔y: zz→zz, xx↔yy, xy→xy, zy↔zx
_PERM_90_9x9 = np.zeros((9, 9))
_PERM_90_9x9[:3, :3] = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
)
_perm6 = np.zeros((6, 6))
_perm6[0, 0] = 1.0  # zz→zz
_perm6[1, 2] = 1.0  # new xx = old yy
_perm6[2, 1] = 1.0  # new yy = old xx
_perm6[3, 3] = 1.0  # xy→xy
_perm6[4, 5] = 1.0  # new zy = old zx
_perm6[5, 4] = 1.0  # new zx = old zy
_PERM_90_9x9[3:, 3:] = _perm6


def _apply_refl_x(G: NDArray) -> NDArray:
    """G(-n1, n2) from G(n1, n2) via x-reflection. Works for 3×3 or 9×9."""
    if G.shape[0] == 9:
        return _REFL_X_9x9 @ G @ _REFL_X_9x9.T
    return G * _REFL_X_SIGN


def _apply_refl_y(G: NDArray) -> NDArray:
    """G(n1, -n2) from G(n1, n2) via y-reflection. Works for 3×3 or 9×9."""
    if G.shape[0] == 9:
        return _REFL_Y_9x9 @ G @ _REFL_Y_9x9.T
    return G * _REFL_Y_SIGN


def _apply_rot90(G: NDArray) -> NDArray:
    """G(n2, n1) from G(n1, n2) via 90-degree rotation (swap x,y). 3×3 or 9×9."""
    if G.shape[0] == 9:
        return _PERM_90_9x9 @ G @ _PERM_90_9x9.T
    return _PERM_90 @ G @ _PERM_90.T


# ═══════════════════════════════════════════════════════════════
#  LatticeGreens class
# ═══════════════════════════════════════════════════════════════
class LatticeGreens:
    """Green's tensor interaction matrix for a 2D square lattice of scatterers."""

    def __init__(
        self,
        d: float,
        M: int,
        omega: float,
        rho: float,
        alpha: float,
        beta: float,
        eta: float = 0.03,
    ) -> None:
        """
        Args:
            d: lattice spacing (centre-to-centre distance).
            M: lattice dimension (M x M scatterers).
            omega: real angular frequency.
            rho: density of reference medium.
            alpha: P-wave speed.
            beta: S-wave speed.
            eta: attenuation ratio Im(omega)/Re(omega).
        """
        self.d = d
        self.M = M
        self.omega_c = omega * (1 + 1j * eta)
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.eta = eta

        # Derived wavenumbers
        self.kP = self.omega_c / alpha
        self.kS = self.omega_c / beta

        # Storage for computed arrays
        self._G_spatial: NDArray[np.complexfloating] | None = None
        self._G_spectral: NDArray[np.complexfloating] | None = None
        # FFT of the circulant embedding (precomputed for matvec)
        self._G_hat: NDArray[np.complexfloating] | None = None

    # ───────────────────────────────────────────────────────
    #  1. Spatial evaluation (exact formula)
    # ───────────────────────────────────────────────────────
    def compute_spatial(self, block_size: int = 9) -> NDArray[np.complexfloating]:
        """Direct spatial evaluation at all lattice separations.

        Uses the exact Green's tensor formula, exploiting D4h
        symmetry to reduce unique evaluations by ~8x.

        Args:
            block_size: 3 for displacement-only G, 9 for full
                propagator [[G, C], [H, S]]. Default 9.

        Returns:
            G_arr of shape (2M-1, 2M-1, B, B) complex where B=block_size.
            G_arr[n1 + M-1, n2 + M-1] = propagator at (n1*d, n2*d, 0).
            The (0,0) entry is zero (self-term excluded).
        """
        from .effective_contrasts import ReferenceMedium

        M = self.M
        S = 2 * M - 1
        B = block_size
        G_arr = np.zeros((S, S, B, B), dtype=complex)

        ref = ReferenceMedium(alpha=self.alpha, beta=self.beta, rho=self.rho)

        for n1 in range(M):
            for n2 in range(n1 + 1):
                if n1 == 0 and n2 == 0:
                    continue

                x = n1 * self.d
                y = n2 * self.d
                if B == 9:
                    G0 = exact_propagator_9x9(x, y, 0.0, self.omega_c, ref)
                else:
                    G0 = exact_greens(
                        x,
                        y,
                        0.0,
                        self.omega_c,
                        rho=self.rho,
                        alpha=self.alpha,
                        beta=self.beta,
                    )

                pairs = self._d4h_orbit(n1, n2)
                for sn1, sn2, G_sym in pairs:
                    i1 = sn1 + M - 1
                    i2 = sn2 + M - 1
                    G_arr[i1, i2, :, :] = G_sym(G0)

        self._G_spatial = G_arr
        return G_arr

    def _d4h_orbit(
        self, n1: int, n2: int
    ) -> list[tuple[int, int, Callable[[np.ndarray], np.ndarray]]]:
        """
        Generate all D4h-related (n1, n2) pairs and their
        tensor transformation functions.

        For n1 != n2, there are 8 images.
        For n1 == n2 (diagonal), there are 4 images.
        """
        identity = lambda G: G.copy()  # noqa: E731
        refl_x = _apply_refl_x
        refl_y = _apply_refl_y
        refl_xy = lambda G: _apply_refl_x(_apply_refl_y(G))  # noqa: E731
        rot90 = _apply_rot90
        rot90_rx = lambda G: _apply_refl_x(_apply_rot90(G))  # noqa: E731
        rot90_ry = lambda G: _apply_refl_y(_apply_rot90(G))  # noqa: E731
        rot90_rxy = lambda G: _apply_refl_x(_apply_refl_y(_apply_rot90(G)))  # noqa: E731

        if n1 == n2:
            # Diagonal: (n,n), (-n,n), (n,-n), (-n,-n) — only 4 images
            return [
                (n1, n2, identity),
                (-n1, n2, refl_x),
                (n1, -n2, refl_y),
                (-n1, -n2, refl_xy),
            ]
        # Off-diagonal: 8 images
        return [
            (n1, n2, identity),
            (-n1, n2, refl_x),
            (n1, -n2, refl_y),
            (-n1, -n2, refl_xy),
            (n2, n1, rot90),
            (-n2, n1, rot90_rx),
            (n2, -n1, rot90_ry),
            (-n2, -n1, rot90_rxy),
        ]

    # ───────────────────────────────────────────────────────
    #  2. Spectral evaluation (2D IFFT)
    # ───────────────────────────────────────────────────────
    def compute_spectral(
        self,
        N_fft: int | None = None,
        p: int = 1,
        alias_nepers: float = 10.0,
        subtract: bool = True,
        kc_factor: float = 1.0,
        block_size: int = 3,
        **kwargs,
    ) -> NDArray[np.complexfloating]:
        """
        Spectral evaluation via 2D IFFT with screened-Coulomb subtraction.

        At z=0, the spectral kernel decays as c/kH (no exponential damping),
        causing large truncation and aliasing errors. We accelerate
        convergence using screened-Coulomb subtraction:

          1. Subtract c/√(kH² + kc²) from diagonal kernel components
             (matches c/kH at large kH, finite at kH=0)
          2. FFT the fast-decaying residual
          3. Add back c·exp(-kc·r)/(2πr) at each lattice point

        The residual decays as O(1/kH³), and the subtracted function's
        spatial transform decays exponentially, doubly improving convergence.

        Args:
            N_fft: FFT grid size. If None, auto-chosen for aliasing suppression.
            p: oversampling factor (lattice points at every p-th grid point).
            alias_nepers: target aliasing suppression in nepers.
            subtract: if True, apply screened-Coulomb subtraction.
            kc_factor: screening parameter kc = kc_factor * |kS|.

        Returns:
            G_arr of shape (2M-1, 2M-1, B, B) complex where B=block_size.
        """
        if block_size == 9:
            return self._compute_spectral_9x9(**kwargs)

        M = self.M
        S = 2 * M - 1
        d = self.d

        dx = d / p
        kmax = np.pi / dx
        kc = kc_factor * abs(self.kS)

        # Auto-size FFT grid for aliasing suppression
        min_N_spatial = S * p
        im_kS = abs(np.imag(self.kS))
        if im_kS > 0:
            L_alias = alias_nepers / im_kS
            min_N_alias = int(np.ceil(L_alias / dx))
        else:
            min_N_alias = min_N_spatial * 4
        min_N = max(min_N_spatial, min_N_alias)

        if N_fft is None:
            N_fft = 1
            while N_fft < min_N:
                N_fft *= 2
        elif N_fft < min_N_spatial:
            raise ValueError(
                f"N_fft={N_fft} too small; need >= {min_N_spatial} for M={M}, p={p}"
            )

        dk = 2.0 * kmax / N_fft

        # Build k-space grid
        k = (np.arange(N_fft) - N_fft // 2) * dk
        KX, KY = np.meshgrid(k, k, indexing="ij")

        # Full spectral kernel
        kernel = spectral_kernel_z0(
            KX, KY, self.omega_c, self.rho, self.alpha, self.beta
        )

        # Subtract screened-Coulomb kernel from diagonal components
        if subtract:
            kernel_sub = screened_kernel_z0(
                KX, KY, self.omega_c, self.rho, self.alpha, self.beta, kc
            )
            kernel_residual = kernel - kernel_sub
        else:
            kernel_residual = kernel

        # 2D IFFT of residual
        scale = dk**2 * N_fft**2 / (4 * np.pi**2)
        G_res_full = np.zeros((N_fft, N_fft, 3, 3), dtype=complex)
        for i in range(3):
            for j in range(3):
                G_res_full[:, :, i, j] = (
                    np.fft.fftshift(
                        np.fft.ifft2(np.fft.ifftshift(kernel_residual[:, :, i, j]))
                    )
                    * scale
                )

        # Sample at lattice points and add back spatial screened-Coulomb
        G_arr = np.zeros((S, S, 3, 3), dtype=complex)
        centre = N_fft // 2
        for n1 in range(-(M - 1), M):
            for n2 in range(-(M - 1), M):
                ix = centre + n1 * p
                iy = centre + n2 * p
                i1 = n1 + M - 1
                i2 = n2 + M - 1
                G_arr[i1, i2, :, :] = G_res_full[ix, iy, :, :]

                if subtract and not (n1 == 0 and n2 == 0):
                    G_arr[i1, i2, :, :] += screened_spatial_z0(
                        n1 * d,
                        n2 * d,
                        self.omega_c,
                        self.rho,
                        self.alpha,
                        self.beta,
                        kc,
                    )

        # Zero out self-term
        G_arr[M - 1, M - 1, :, :] = 0.0

        self._G_spectral = G_arr
        return G_arr

    def _compute_spectral_9x9(
        self,
        Nky: int = 512,
        Nkz: int = 512,
        target_nepers: float = 14.0,
    ) -> NDArray[np.complexfloating]:
        """9x9 lattice Green's tensor via horizontal residues.

        Uses kx-residue decomposition with ky-IFFT and kz-quadrature.
        The Δx=0 column is obtained via D4h rotation of the (n, 0)
        values. Adaptive kz_max and ky oversampling per separation
        ensure convergence for all blocks including C, H, S.

        Args:
            Nky: number of ky grid points for 1D IFFT.
            Nkz: number of kz quadrature points per separation.
            target_nepers: target attenuation at ky/kz truncation
                boundary. Controls adaptive kz_max and ky oversampling.

        Returns:
            G_arr of shape (2M-1, 2M-1, 9, 9) complex.
        """
        from .horizontal_greens import fft_grid_1d, post_kx_residue_kernel_9x9_vec

        M = self.M
        S = 2 * M - 1
        d = self.d
        kS_abs = abs(self.kS)
        G_arr = np.zeros((S, S, 9, 9), dtype=complex)

        for n1 in range(1, M):
            dx_abs = n1 * d

            # Adaptive truncation per separation
            kz_max = max(4.0 * kS_abs, target_nepers / dx_abs)
            p_ky = max(1, int(np.ceil(target_nepers / (dx_abs * np.pi / d))))
            ky_max = np.pi * p_ky / d
            Nky_eff = max(Nky, S * p_ky * 2)
            if Nky_eff % 2:
                Nky_eff += 1

            # ky grid (lattice-commensurate with oversampling p_ky)
            ky_arr, _y_grid, dky, _dy = fft_grid_1d(Nky_eff, ky_max)
            kz_arr = np.linspace(-kz_max, kz_max, Nkz)
            dkz = kz_arr[1] - kz_arr[0]
            scale_ky = dky * Nky_eff / (2.0 * np.pi)
            scale_kz = dkz / (2.0 * np.pi)

            P_fft = np.zeros((9, 9, Nky_eff), dtype=complex)
            for kz in kz_arr:
                kernel = post_kx_residue_kernel_9x9_vec(
                    ky_arr,
                    kz,
                    dx_abs,
                    self.omega_c,
                    self.rho,
                    self.alpha,
                    self.beta,
                )
                for a in range(9):
                    for b in range(9):
                        P_fft[a, b, :] += (
                            np.fft.fftshift(
                                np.fft.ifft(np.fft.ifftshift(kernel[a, b, :]))
                            )
                            * scale_ky
                            * scale_kz
                        )

            # Extract first-octant points and fill via D4h symmetry
            iy0 = Nky_eff // 2
            for n2 in range(n1 + 1):
                iy = iy0 + n2 * p_ky
                G0 = P_fft[:, :, iy]
                pairs = self._d4h_orbit(n1, n2)
                for sn1, sn2, G_sym in pairs:
                    G_arr[sn1 + M - 1, sn2 + M - 1, :, :] = G_sym(G0)

        self._G_spectral = G_arr
        return G_arr

    # ───────────────────────────────────────────────────────
    #  2b. Hybrid spatial/spectral evaluation
    # ───────────────────────────────────────────────────────
    def compute_hybrid(
        self,
        r_cut: float | None = None,
        N_fft: int | None = None,
        p: int = 1,
        alias_nepers: float = 10.0,
        subtract: bool = True,
        kc_factor: float = 1.0,
        block_size: int = 3,
        **kwargs,
    ) -> NDArray[np.complexfloating]:
        """
        Hybrid spatial/spectral evaluation for optimal accuracy.

        Uses the exact spatial formula (Ben-Menahem & Singh) for
        near-field separations (|R| <= r_cut) and the spectral 2D IFFT
        for far-field separations. This combines machine-precision
        near-field accuracy with efficient spectral far-field evaluation.

        Args:
            r_cut: cutoff radius for spatial evaluation. Separations with
                |R| <= r_cut use exact spatial, the rest use spectral.
                Default: 3*d (covers nearest, next-nearest, and beyond).
            N_fft: FFT grid size (passed to compute_spectral).
            p: oversampling factor (passed to compute_spectral).
            alias_nepers: aliasing suppression (passed to compute_spectral).
            subtract: screened-Coulomb subtraction (passed to compute_spectral).
            kc_factor: screening parameter (passed to compute_spectral).

        Returns:
            G_arr of shape (2M-1, 2M-1, B, B) complex where B=block_size.
        """
        if block_size == 9:
            return self._compute_hybrid_9x9(r_cut=r_cut, **kwargs)

        M = self.M
        S = 2 * M - 1
        d = self.d
        if r_cut is None:
            r_cut = 3.0 * d

        # Spectral evaluation for all separations
        G_arr = self.compute_spectral(
            N_fft=N_fft,
            p=p,
            alias_nepers=alias_nepers,
            subtract=subtract,
            kc_factor=kc_factor,
        )

        # Overwrite near-field with exact spatial values
        n_spatial = 0
        for n1 in range(-(M - 1), M):
            for n2 in range(-(M - 1), M):
                if n1 == 0 and n2 == 0:
                    continue
                r = np.sqrt((n1 * d) ** 2 + (n2 * d) ** 2)
                if r <= r_cut:
                    i1 = n1 + M - 1
                    i2 = n2 + M - 1
                    G_arr[i1, i2, :, :] = exact_greens(
                        n1 * d,
                        n2 * d,
                        0.0,
                        self.omega_c,
                        rho=self.rho,
                        alpha=self.alpha,
                        beta=self.beta,
                    )
                    n_spatial += 1

        self._G_spectral = G_arr
        return G_arr

    def _compute_hybrid_9x9(
        self,
        r_cut: float | None = None,
        **spectral_kwargs,
    ) -> NDArray[np.complexfloating]:
        """Hybrid spatial/spectral 9x9 evaluation.

        Near-field (|R| <= r_cut) uses exact_propagator_9x9,
        far-field uses _compute_spectral_9x9 (horizontal residues).
        """
        from .effective_contrasts import ReferenceMedium

        M = self.M
        d = self.d
        if r_cut is None:
            r_cut = 3.0 * d

        G_arr = self._compute_spectral_9x9(**spectral_kwargs)

        ref = ReferenceMedium(alpha=self.alpha, beta=self.beta, rho=self.rho)
        for n1 in range(-(M - 1), M):
            for n2 in range(-(M - 1), M):
                if n1 == 0 and n2 == 0:
                    continue
                r = np.sqrt((n1 * d) ** 2 + (n2 * d) ** 2)
                if r <= r_cut:
                    i1 = n1 + M - 1
                    i2 = n2 + M - 1
                    G_arr[i1, i2, :, :] = exact_propagator_9x9(
                        n1 * d, n2 * d, 0.0, self.omega_c, ref
                    )

        self._G_spectral = G_arr
        return G_arr

    # ───────────────────────────────────────────────────────
    #  2c. FCC Hankel transform evaluation
    # ───────────────────────────────────────────────────────
    def compute_fcc(
        self,
        N_per_seg: int = 128,
        K_max: float | None = None,
        kc_factor: float = 1.0,
        block_size: int = 3,
        **kwargs,
    ) -> NDArray[np.complexfloating]:
        """
        FCC Hankel transform evaluation of the lattice Green's tensor.

        Reduces the 2D inverse FT to three 1D Hankel integrals (H0, H2, V0)
        by decomposing the spectral kernel into angular harmonics m=0 and m=2.
        Uses Filon-Clenshaw-Curtis quadrature with asymptotic subtraction.
        For block_size=9, delegates to _compute_spectral_9x9 (no FCC Hankel
        decomposition for derivative blocks).

        Args:
            N_per_seg: Clenshaw-Curtis nodes per integration segment.
            K_max: upper integration limit (default: 4*|kS|).
            kc_factor: screening parameter kc = kc_factor * |kS|.
            block_size: 3 for displacement-only, 9 for full propagator.

        Returns:
            G_arr of shape (2M-1, 2M-1, B, B) complex where B=block_size.
        """
        if block_size == 9:
            return self._compute_spectral_9x9(**kwargs)
        M = self.M
        S = 2 * M - 1
        d = self.d
        G_arr = np.zeros((S, S, 3, 3), dtype=complex)

        # Collect unique (r, theta) from first-octant separations
        unique_r_map: dict[tuple[int, int], tuple[float, float]] = {}
        for n1 in range(M):
            for n2 in range(n1 + 1):
                if n1 == 0 and n2 == 0:
                    continue
                x, y = n1 * d, n2 * d
                r = np.sqrt(x**2 + y**2)
                unique_r_map[(n1, n2)] = (r, np.arctan2(y, x))

        # Extract unique distances and evaluate Hankel transforms
        r_vals = np.array([v[0] for v in unique_r_map.values()])
        H0, H2, V0 = _hankel_transform_cc(
            r_vals,
            self.omega_c,
            self.rho,
            self.alpha,
            self.beta,
            N_per_seg=N_per_seg,
            K_max=K_max,
            kc_factor=kc_factor,
        )

        # Assemble G at first-octant points and propagate via D4h symmetry
        for idx, ((n1, n2), (_r, theta)) in enumerate(unique_r_map.items()):
            cos2t = np.cos(2 * theta)
            sin2t = np.sin(2 * theta)

            G0 = np.zeros((3, 3), dtype=complex)
            G0[0, 0] = H0[idx] - cos2t * H2[idx]
            G0[1, 1] = H0[idx] + cos2t * H2[idx]
            G0[0, 1] = -sin2t * H2[idx]
            G0[1, 0] = -sin2t * H2[idx]
            G0[2, 2] = V0[idx]

            pairs = self._d4h_orbit(n1, n2)
            for sn1, sn2, G_sym in pairs:
                i1 = sn1 + M - 1
                i2 = sn2 + M - 1
                G_arr[i1, i2, :, :] = G_sym(G0)

        self._G_spectral = G_arr
        return G_arr

    # ───────────────────────────────────────────────────────
    #  3. FFT-accelerated block-Toeplitz mat-vec
    # ───────────────────────────────────────────────────────
    def _precompute_circulant_fft(self) -> None:
        """Precompute the 2D FFT of the circulant embedding.

        Stores self._G_hat of shape (2M-1, 2M-1, B, B) where B is
        the block size (3 or 9) determined from the stored array.
        """
        if self._G_spatial is not None:
            G_arr = self._G_spatial
        elif self._G_spectral is not None:
            G_arr = self._G_spectral
        else:
            G_arr = self.compute_spatial()

        S = 2 * self.M - 1
        B = G_arr.shape[2]  # 3 or 9
        self._G_hat = np.zeros((S, S, B, B), dtype=complex)
        for i in range(B):
            for j in range(B):
                self._G_hat[:, :, i, j] = np.fft.fft2(G_arr[:, :, i, j])

    def matvec(self, u: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
        """FFT-accelerated block-Toeplitz matrix-vector product.

        Computes v_m = sum_{n != m} P(R_m - R_n) . u_n

        Args:
            u: (M, M, B) complex — field at each scatterer (B=3 or 9).

        Returns:
            v: (M, M, B) complex — scattered field contribution.
        """
        M = self.M
        S = 2 * M - 1

        if self._G_hat is None:
            self._precompute_circulant_fft()

        assert self._G_hat is not None
        B = self._G_hat.shape[2]

        u_pad = np.zeros((S, S, B), dtype=complex)
        u_pad[:M, :M, :] = u

        u_hat = np.zeros((S, S, B), dtype=complex)
        for k in range(B):
            u_hat[:, :, k] = np.fft.fft2(u_pad[:, :, k])

        v_hat = np.zeros((S, S, B), dtype=complex)
        for i in range(B):
            for j in range(B):
                v_hat[:, :, i] += self._G_hat[:, :, i, j] * u_hat[:, :, j]

        v_full = np.zeros((S, S, B), dtype=complex)
        for i in range(B):
            v_full[:, :, i] = np.fft.ifft2(v_hat[:, :, i])

        v = v_full[M - 1 : S, M - 1 : S, :]
        return v

    # ───────────────────────────────────────────────────────
    #  4. Verification
    # ───────────────────────────────────────────────────────
    def verify(self, n_test: int = 10) -> float:
        """
        Compare spectral vs spatial at random lattice separations.

        Args:
            n_test: number of random separations to check.

        Returns:
            max_relative_error: worst-case relative Frobenius error.
        """
        if self._G_spatial is None:
            self.compute_spatial()
        if self._G_spectral is None:
            self.compute_spectral()
        assert self._G_spatial is not None
        assert self._G_spectral is not None

        M = self.M
        rng = np.random.default_rng(42)
        max_err: float = 0.0
        errors: list[float] = []

        # Test all non-zero separations if small enough, else random sample
        if (2 * M - 1) ** 2 <= n_test + 1:
            test_pairs: list[tuple[int, int]] = [
                (n1, n2)
                for n1 in range(-(M - 1), M)
                for n2 in range(-(M - 1), M)
                if not (n1 == 0 and n2 == 0)
            ]
        else:
            test_pairs = []
            while len(test_pairs) < n_test:
                n1 = int(rng.integers(-(M - 1), M))
                n2 = int(rng.integers(-(M - 1), M))
                if n1 == 0 and n2 == 0:
                    continue
                test_pairs.append((n1, n2))

        for n1, n2 in test_pairs:
            i1 = n1 + M - 1
            i2 = n2 + M - 1
            Gs = self._G_spatial[i1, i2, :, :]
            Gf = self._G_spectral[i1, i2, :, :]
            norm_s = float(np.linalg.norm(Gs))
            if norm_s > 1e-30:
                err = float(np.linalg.norm(Gf - Gs)) / norm_s
            else:
                err = float(np.linalg.norm(Gf - Gs))
            errors.append(err)
            if err > max_err:
                max_err = err

        print(f"  Spectral vs spatial verification ({len(test_pairs)} points):")
        print(f"    max relative error: {max_err:.4e}")
        print(f"    mean relative error: {np.mean(errors):.4e}")

        return max_err


# ═══════════════════════════════════════════════════════════════
#  Direct summation mat-vec (for verification of FFT mat-vec)
# ═══════════════════════════════════════════════════════════════
def _matvec_direct(
    G_arr: NDArray[np.complexfloating],
    u: NDArray[np.complexfloating],
    M: int,
) -> NDArray[np.complexfloating]:
    """
    Direct (non-FFT) matrix-vector product for verification.

    v[m1, m2, i] = sum_{n1,n2 != m1,m2} G[m1-n1, m2-n2, i, j] * u[n1, n2, j]
    """
    v = np.zeros_like(u)
    for m1 in range(M):
        for m2 in range(M):
            for n1 in range(M):
                for n2 in range(M):
                    if m1 == n1 and m2 == n2:
                        continue
                    dn1 = m1 - n1
                    dn2 = m2 - n2
                    i1 = dn1 + M - 1
                    i2 = dn2 + M - 1
                    v[m1, m2, :] += G_arr[i1, i2, :, :] @ u[n1, n2, :]
    return v


# ═══════════════════════════════════════════════════════════════
#  Typer CLI
# ═══════════════════════════════════════════════════════════════
app = typer.Typer(help="Lattice Green's tensor verification.")


@app.command()
def verify_cmd(
    d: Annotated[float, typer.Option(help="Lattice spacing.")] = 0.1,
    m: Annotated[int, typer.Option("--M", "-M", help="Lattice dimension (MxM).")] = 8,
    omega: Annotated[float, typer.Option(help="Angular frequency.")] = 2 * np.pi,
    rho: Annotated[float, typer.Option(help="Density.")] = 3.0,
    alpha: Annotated[float, typer.Option(help="P-wave speed.")] = 5.0,
    beta: Annotated[float, typer.Option(help="S-wave speed.")] = 3.0,
    eta: Annotated[float, typer.Option(help="Attenuation ratio.")] = 0.03,
    method: Annotated[
        str,
        typer.Option(help="Inversion method: spatial|spectral|hybrid|fcc|all."),
    ] = "all",
) -> None:
    """Run verification tests for the lattice Green's tensor."""
    M = m

    print("=" * 72)
    print("Lattice Green's Tensor — Verification")
    print("=" * 72)
    print(f"  d={d}, M={M}, omega={omega:.4f}, eta={eta}")
    print(f"  rho={rho}, alpha={alpha}, beta={beta}")
    omega_c = omega * (1 + 1j * eta)
    print(f"  omega_c = {omega_c:.6f}")
    print(f"  |kP|={abs(omega_c / alpha):.4f},  |kS|={abs(omega_c / beta):.4f}")
    print()

    lg = LatticeGreens(d, M, omega, rho, alpha, beta, eta)

    # ─── Test 1: Spatial evaluation ───
    print("TEST 1: Spatial evaluation (exact formula)")
    print("-" * 72)
    t0 = time()
    G_spatial = lg.compute_spatial()
    dt = time() - t0
    print(f"  Computed in {dt:.2f}s")
    print(f"  Shape: {G_spatial.shape}")

    # Check symmetry: G_ij(n1,n2) = G_ji(n1,n2) (reciprocity)
    max_asym = 0.0
    for n1 in range(-(M - 1), M):
        for n2 in range(-(M - 1), M):
            if n1 == 0 and n2 == 0:
                continue
            i1, i2 = n1 + M - 1, n2 + M - 1
            G = G_spatial[i1, i2, :, :]
            asym = float(np.linalg.norm(G - G.T) / np.linalg.norm(G))
            max_asym = max(max_asym, asym)
    print(f"  Max reciprocity asymmetry |G - G^T|/|G|: {max_asym:.4e}")

    # Show a few values
    print("  Sample values:")
    for n1, n2 in [(1, 0), (0, 1), (1, 1), (2, 0)]:
        i1, i2 = n1 + M - 1, n2 + M - 1
        G = G_spatial[i1, i2, :, :]
        print(f"    G({n1},{n2}): |G|={np.linalg.norm(G):.6e}")
    print()

    if method == "spatial":
        print("=" * 72)
        print("All tests complete.")
        print("=" * 72)
        return

    # ─── Test 2: Spectral accuracy (screened-Coulomb subtraction) ───
    if method in ("spectral", "all"):
        print("TEST 2: Spectral accuracy (screened-Coulomb subtraction)")
        print("-" * 72)
        test_seps = [(1, 0), (0, 1), (1, 1), (2, 1), (3, 0)]
        lg._G_spectral = None
        t0 = time()
        G_spec = lg.compute_spectral(subtract=True, kc_factor=1.0)
        dt = time() - t0
        print(f"  Spectral computed in {dt:.1f}s")
        print("  Per-separation errors:")
        worst_spec = 0.0
        for n1, n2 in test_seps:
            i1, i2 = n1 + M - 1, n2 + M - 1
            Gs = G_spatial[i1, i2, :, :]
            Gf = G_spec[i1, i2, :, :]
            err = float(np.linalg.norm(Gf - Gs) / np.linalg.norm(Gs))
            worst_spec = max(worst_spec, err)
            print(f"    ({n1},{n2}): {err:.4e}")
        print(f"  Worst spectral error: {worst_spec:.4e}")
        print()

    # ─── Test 3: Hybrid spatial/spectral accuracy ───
    if method in ("hybrid", "all"):
        print("TEST 3: Hybrid spatial/spectral accuracy")
        print("-" * 72)
        for r_cut_mult in [1.5, 2.5, 3.5, 5.0]:
            r_cut = r_cut_mult * d
            lg._G_spectral = None
            t0 = time()
            G_hyb = lg.compute_hybrid(r_cut=r_cut, subtract=True, kc_factor=1.0)
            dt = time() - t0
            worst_hyb = 0.0
            for n1 in range(-(M - 1), M):
                for n2 in range(-(M - 1), M):
                    if n1 == 0 and n2 == 0:
                        continue
                    i1, i2 = n1 + M - 1, n2 + M - 1
                    Gs = G_spatial[i1, i2, :, :]
                    Gf = G_hyb[i1, i2, :, :]
                    norm_s = float(np.linalg.norm(Gs))
                    if norm_s > 1e-30:
                        err = float(np.linalg.norm(Gf - Gs)) / norm_s
                    else:
                        err = float(np.linalg.norm(Gf - Gs))
                    worst_hyb = max(worst_hyb, err)
            n_near = sum(
                1
                for n1 in range(-(M - 1), M)
                for n2 in range(-(M - 1), M)
                if not (n1 == 0 and n2 == 0)
                and np.sqrt((n1 * d) ** 2 + (n2 * d) ** 2) <= r_cut
            )
            print(
                f"  r_cut={r_cut_mult:.1f}d: {n_near:3d} spatial, "
                f"worst err={worst_hyb:.4e}, time={dt:.1f}s"
            )
        print()

    if method == "all":
        # Use hybrid with r_cut=3d for subsequent tests
        lg._G_spectral = None
        lg._G_hat = None
        lg.compute_hybrid(r_cut=3.0 * d, subtract=True, kc_factor=1.0)

        # ─── Test 4: Mat-vec correctness (using hybrid G) ───
        print("TEST 4: FFT mat-vec vs direct summation (hybrid G)")
        print("-" * 72)
        rng = np.random.default_rng(123)
        u = rng.standard_normal((M, M, 3)) + 1j * rng.standard_normal((M, M, 3))

        t0 = time()
        v_fft = lg.matvec(u)
        dt_fft = time() - t0

        t0 = time()
        v_dir = _matvec_direct(G_spatial, u, M)
        dt_dir = time() - t0

        err_mv = np.linalg.norm(v_fft - v_dir) / np.linalg.norm(v_dir)
        print(f"  FFT mat-vec time:    {dt_fft:.4f}s")
        print(f"  Direct mat-vec time: {dt_dir:.4f}s")
        print(f"  Relative error:      {err_mv:.4e}")
        print(f"  Speedup:             {dt_dir / dt_fft:.1f}x")
        print()

        # ─── Test 5: D4h symmetry check ───
        print("TEST 5: D4h symmetry relations")
        print("-" * 72)
        max_refl_err = 0.0
        max_rot_err = 0.0
        for n1 in range(1, min(M, 4)):
            for n2 in range(n1):
                i1_p, i2_p = n1 + M - 1, n2 + M - 1
                i1_m, i2_m = -n1 + M - 1, n2 + M - 1
                i1_r, i2_r = n2 + M - 1, n1 + M - 1

                G_pos = G_spatial[i1_p, i2_p, :, :]
                G_neg = G_spatial[i1_m, i2_m, :, :]
                G_rot = G_spatial[i1_r, i2_r, :, :]

                G_neg_pred = _apply_refl_x(G_pos)
                err_refl = float(
                    np.linalg.norm(G_neg - G_neg_pred) / np.linalg.norm(G_neg)
                )
                max_refl_err = max(max_refl_err, err_refl)

                G_rot_pred = _apply_rot90(G_pos)
                err_rot = float(
                    np.linalg.norm(G_rot - G_rot_pred) / np.linalg.norm(G_rot)
                )
                max_rot_err = max(max_rot_err, err_rot)

        print(f"  Max x-reflection error: {max_refl_err:.4e}")
        print(f"  Max 90-rotation error:  {max_rot_err:.4e}")
        print()

        # ─── Test 6: Decay check ───
        print("TEST 6: Decay with distance")
        print("-" * 72)
        for n in [1, 2, 4, 6]:
            if n >= M:
                break
            i1 = n + M - 1
            i2 = M - 1
            G = G_spatial[i1, i2, :, :]
            print(f"  |G({n},0)| = {np.linalg.norm(G):.6e}  (r = {n * d:.3f})")
        print()

    # ─── Test 7: FCC Hankel transform accuracy & convergence ───
    if method in ("fcc", "all"):
        print("TEST 7: FCC Hankel transform accuracy")
        print("-" * 72)
        lg._G_spectral = None
        t0 = time()
        G_fcc = lg.compute_fcc(kc_factor=1.0)
        dt = time() - t0
        print(f"  FCC computed in {dt:.3f}s")

        print("  Per-separation errors (FCC vs spatial):")
        worst_fcc = 0.0
        for n1, n2 in [(1, 0), (0, 1), (1, 1), (2, 1), (3, 0), (5, 3), (7, 0)]:
            if abs(n1) >= M or abs(n2) >= M:
                continue
            i1, i2 = n1 + M - 1, n2 + M - 1
            Gs = G_spatial[i1, i2, :, :]
            Gf = G_fcc[i1, i2, :, :]
            norm_s = float(np.linalg.norm(Gs))
            if norm_s > 1e-30:
                err = float(np.linalg.norm(Gf - Gs)) / norm_s
            else:
                err = float(np.linalg.norm(Gf - Gs))
            worst_fcc = max(worst_fcc, err)
            print(f"    ({n1},{n2}): {err:.4e}")

        worst_all = 0.0
        for n1 in range(-(M - 1), M):
            for n2 in range(-(M - 1), M):
                if n1 == 0 and n2 == 0:
                    continue
                i1, i2 = n1 + M - 1, n2 + M - 1
                Gs = G_spatial[i1, i2, :, :]
                Gf = G_fcc[i1, i2, :, :]
                norm_s = float(np.linalg.norm(Gs))
                if norm_s > 1e-30:
                    err = float(np.linalg.norm(Gf - Gs)) / norm_s
                else:
                    err = float(np.linalg.norm(Gf - Gs))
                worst_all = max(worst_all, err)
        print(f"  Worst FCC error (all separations): {worst_all:.4e}")
        print()

        print("  Convergence study (N_per_seg vs worst error):")
        for N_seg in [32, 64, 96, 128, 192]:
            lg._G_spectral = None
            G_test = lg.compute_fcc(N_per_seg=N_seg, kc_factor=1.0)
            worst = 0.0
            for n1 in range(-(M - 1), M):
                for n2 in range(-(M - 1), M):
                    if n1 == 0 and n2 == 0:
                        continue
                    i1, i2 = n1 + M - 1, n2 + M - 1
                    Gs = G_spatial[i1, i2, :, :]
                    Gf = G_test[i1, i2, :, :]
                    norm_s = float(np.linalg.norm(Gs))
                    if norm_s > 1e-30:
                        err = float(np.linalg.norm(Gf - Gs)) / norm_s
                    else:
                        err = float(np.linalg.norm(Gf - Gs))
                    worst = max(worst, err)
            print(f"    N_per_seg={N_seg:4d}: worst err = {worst:.4e}")
        print()

    print("=" * 72)
    print("All tests complete.")
    print("=" * 72)


if __name__ == "__main__":
    app()
