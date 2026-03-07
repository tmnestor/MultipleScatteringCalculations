"""
effective_contrasts.py
Compute the self-consistent cubic T-matrix effective contrasts.

Two computation methods are used:

  Γ₀ (Green's tensor integral):
    3D Gauss-Legendre quadrature of the FULL G_{ij}. The integral
    ∫G_{ij} d³x converges (G ~ 1/r) and must include the static
    contribution to get the correct real part.

  A^c, B^c, C^c (second-derivative integrals):
    Polynomial (Taylor expansion) method.  The full integral
    ∫G_{ij,kl} d³x has a DIVERGENT 1/r³ Eshelby singularity.
    The polynomial approach extracts only the smooth (convergent)
    part by expanding the Green's function in a Taylor series of r²
    and integrating each polynomial term analytically via cube moments.

Reference: TMatrix_Derivation.pdf (Part II, Sections 11-16).

Coordinate system: generic (1,2,3) Cartesian — the coordinate
relabelling to (z,x,y) is handled in voigt_tmatrix.py.

All computation is pure NumPy (no SymPy dependency).
"""

from dataclasses import dataclass
from math import factorial
from typing import Tuple

import numpy as np
from numpy.polynomial.legendre import leggauss

# Default Taylor expansion order (number of phi/psi terms) and quadrature points
N_TAYLOR = 8  # number of Taylor terms for phi and psi
N_GAUSS = 32  # GL quadrature points per dimension


# ================================================================
# Data classes
# ================================================================


@dataclass
class ReferenceMedium:
    """Isotropic elastic reference medium."""

    alpha: float  # P-wave velocity (m/s)
    beta: float  # S-wave velocity (m/s)
    rho: float  # density (kg/m^3)

    @property
    def lam(self) -> float:
        """Lamé parameter λ = ρ(α² - 2β²)."""
        return self.rho * (self.alpha**2 - 2 * self.beta**2)

    @property
    def mu(self) -> float:
        """Shear modulus μ = ρβ²."""
        return self.rho * self.beta**2


@dataclass
class MaterialContrast:
    """Isotropic material contrast relative to reference."""

    Dlambda: float  # Δλ (Pa)
    Dmu: float  # Δμ (Pa)
    Drho: float  # Δρ (kg/m^3)


@dataclass
class CubeTMatrixResult:
    """Complete result of the cubic T-matrix computation."""

    # Green's tensor integrals
    Gamma0: complex
    Ac: complex
    Bc: complex
    Cc: complex

    # T-matrix coupling coefficients
    T1c: complex
    T2c: complex
    T3c: complex

    # Four amplification factors
    amp_u: complex
    amp_theta: complex
    amp_e_off: complex
    amp_e_diag: complex

    # Five effective contrasts
    Drho_star: complex
    Dlambda_star: complex
    Dmu_star_off: complex
    Dmu_star_diag: complex

    @property
    def cubic_anisotropy(self) -> complex:
        """Δμ*_diag − Δμ*_off : measure of cubic anisotropy."""
        return self.Dmu_star_diag - self.Dmu_star_off


# ================================================================
# Γ₀ computation: numerical quadrature of the FULL Green's tensor
# ================================================================


def _green_f_g_vec(
    r: np.ndarray, omega: float, alpha: float, beta: float, rho: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate f(r) and g(r) at an array of r values.

    G_{ij}(x) = f(r) δ_{ij} + g(r) n_i n_j
    """
    C_nf = (1.0 - beta**2 / alpha**2) / (8.0 * np.pi * rho * beta**2)
    X = np.exp(1j * omega * r / alpha) / (4.0 * np.pi * rho * alpha**2)
    V = np.exp(1j * omega * r / beta) / (4.0 * np.pi * rho * beta**2)

    rinv = 1.0 / r
    f_val = (V - C_nf) * rinv
    g_val = (3.0 * C_nf + X - V) * rinv
    return f_val, g_val


def _setup_cube_quadrature(a: float, n_gauss: int = N_GAUSS):
    """Set up 3D GL quadrature: returns (x1, x2, x3, r, wt, mask)."""
    nodes, weights = leggauss(n_gauss)
    x = a * nodes
    w = a * weights

    X1, X2, X3 = np.meshgrid(x, x, x, indexing="ij")
    W1, W2, W3 = np.meshgrid(w, w, w, indexing="ij")

    x1 = X1.ravel()
    x2 = X2.ravel()
    x3 = X3.ravel()
    wt = (W1 * W2 * W3).ravel()

    r = np.sqrt(x1**2 + x2**2 + x3**2)
    mask = r > 1e-200
    return x1, x2, x3, r, wt, mask


def _compute_Gamma0_numerical(
    omega: float,
    a: float,
    alpha: float,
    beta: float,
    rho: float,
    n_gauss: int = N_GAUSS,
) -> complex:
    """
    Compute Γ₀^cube = ∫_cube G_{11}(x) d³x using vectorised GL quadrature.

    The integral converges (G ~ 1/r) and correctly captures both real
    (static) and imaginary (frequency-dependent) contributions.
    """
    x1, x2, x3, r, wt, mask = _setup_cube_quadrature(a, n_gauss)

    rm = r[mask]
    f_val, g_val = _green_f_g_vec(rm, omega, alpha, beta, rho)

    # G_{11} = f(r) + g(r) · n_1² = f + g · x1²/r²
    n1sq = x1[mask] ** 2 / (rm * rm)
    G11 = f_val + g_val * n1sq

    return np.sum(wt[mask] * G11)


# ================================================================
# A^c, B^c, C^c computation: polynomial Taylor expansion method
# ================================================================


def _compute_taylor_coefficients(
    omega: float, alpha: float, beta: float, rho: float, n_taylor: int = N_TAYLOR
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Taylor coefficients φ_n and ψ_n.

    The smooth (polynomial) part of the Green's tensor is:
      G^s_{ij}(x) = δ_{ij} Φ(r²) + x_i x_j Ψ(r²)

    where Φ(u) = Σ φ_n u^n and Ψ(u) = Σ ψ_n u^n.

    These come from the odd-power terms of the Taylor expansion of
    f·r and g·r, which are the only terms that produce integrable
    (polynomial) integrands over the cube.

    φ_n = (iω/β)^{2n+1} / ((2n+1)! · 4πρβ²)

    ψ_n = (iω)^{2n+3} · [1/α^{2n+5} - 1/β^{2n+5}] / ((2n+3)! · 4πρ)
    """
    phi = np.zeros(n_taylor, dtype=complex)
    psi = np.zeros(n_taylor, dtype=complex)

    ik_beta = 1j * omega / beta
    ik_alpha = 1j * omega / alpha
    coeff_f = 1.0 / (4.0 * np.pi * rho * beta**2)
    coeff_g = 1.0 / (4.0 * np.pi * rho)

    for n in range(n_taylor):
        m = 2 * n + 1  # odd power index for f·r
        phi[n] = ik_beta**m / factorial(m) * coeff_f

        m2 = 2 * n + 3  # odd power index for g·r (shifted by 2)
        psi[n] = (
            (1j * omega) ** m2
            * (1.0 / alpha ** (m2 + 2) - 1.0 / beta ** (m2 + 2))
            / (factorial(m2) * coeff_g ** (-1))
        )
        # Cleaner: psi_n = coeff_g * [(iω/α)^{2n+3}/α² - (iω/β)^{2n+3}/β²] / (2n+3)!
        psi[n] = (
            coeff_g
            * (ik_alpha ** (2 * n + 3) / alpha**2 - ik_beta ** (2 * n + 3) / beta**2)
            / factorial(2 * n + 3)
        )

    return phi, psi


def _eval_poly_and_derivs(u: np.ndarray, coeffs: np.ndarray):
    """
    Evaluate a polynomial P(u) = Σ c_n u^n and its first two
    derivatives P'(u) and P''(u) at array of u values.

    Returns (P, Pp, Ppp) — polynomial, first deriv, second deriv.
    """
    N = len(coeffs)
    P = np.zeros_like(u, dtype=complex)
    Pp = np.zeros_like(u, dtype=complex)
    Ppp = np.zeros_like(u, dtype=complex)

    un = np.ones_like(u, dtype=complex)  # u^0
    for n in range(N):
        P += coeffs[n] * un
        if n >= 1:
            # P'(u) has coefficient n*c_n for u^{n-1}
            Pp += n * coeffs[n] * (un / u if n >= 1 else np.zeros_like(u))
        if n >= 2:
            Ppp += (
                n * (n - 1) * coeffs[n] * (un / (u * u) if n >= 2 else np.zeros_like(u))
            )
        un = un * u

    return P, Pp, Ppp


def _eval_poly_and_derivs_safe(u: np.ndarray, coeffs: np.ndarray):
    """
    Evaluate P(u), P'(u), P''(u) safely (avoid division by zero at u=0).

    Uses Horner-like evaluation for derivatives.
    """
    N = len(coeffs)

    # P(u) = c_0 + c_1 u + c_2 u² + ...
    # P'(u) = c_1 + 2c_2 u + 3c_3 u² + ...
    # P''(u) = 2c_2 + 6c_3 u + 12c_4 u² + ...

    # Build derivative coefficient arrays
    c_p = np.zeros(N, dtype=complex)  # P' coefficients
    c_pp = np.zeros(N, dtype=complex)  # P'' coefficients

    for n in range(1, N):
        c_p[n - 1] = n * coeffs[n]
    for n in range(2, N):
        c_pp[n - 2] = n * (n - 1) * coeffs[n]

    # Evaluate using Horner's method
    P = np.zeros_like(u, dtype=complex)
    Pp = np.zeros_like(u, dtype=complex)
    Ppp = np.zeros_like(u, dtype=complex)

    for n in range(N - 1, -1, -1):
        P = P * u + coeffs[n]
    for n in range(N - 2, -1, -1):
        Pp = Pp * u + c_p[n]
    for n in range(N - 3, -1, -1):
        Ppp = Ppp * u + c_pp[n]

    return P, Pp, Ppp


def _static_eshelby_ABC(
    alpha: float, beta: float, rho: float
) -> Tuple[complex, complex, complex]:
    """
    Static Eshelby depolarization tensor for a cube: A_stat, B_stat, C_stat.

    The Taylor expansion method captures only the smooth (radiation)
    part of ∫G_{ij,kl} d³x.  The static 1/r³ singularity must be handled
    separately via the divergence theorem, converting the PV volume
    integral into a regular surface integral over the six cube faces.

    Geometric constants (dimensionless surface integrals over [-1,1]²):
      j₁ = ∫ du dv / (1+u²+v²)^{3/2}           = 2π/3
      j₂ = ∫ u² du dv / (1+u²+v²)^{5/2}         = -2(√3 - π)/9
      k₁ = ∫ du dv / (1+u²+v²)^{5/2}             = 2(2√3 + π)/9

    Static Kelvin Green's tensor: G^K_{ij} = a₀/r δ_{ij} + b₀ x_i x_j/r³
      a₀ = (α² + β²) / (8π ρ α² β²)
      b₀ = (α² - β²) / (8π ρ α² β²)

    Reference: CubicTMatrix_FullGreensTensor.nb, Section 5b.
    """
    a0 = (alpha**2 + beta**2) / (8.0 * np.pi * rho * alpha**2 * beta**2)
    b0 = (alpha**2 - beta**2) / (8.0 * np.pi * rho * alpha**2 * beta**2)

    sqrt3 = np.sqrt(3.0)
    j1 = 2.0 * np.pi / 3.0
    j2 = -2.0 * (sqrt3 - np.pi) / 9.0
    k1 = 2.0 * (2.0 * sqrt3 + np.pi) / 9.0

    A_stat = 2.0 * (-a0 * j1 - 3.0 * b0 * j2)
    B_stat = 2.0 * b0 * (j1 - 3.0 * j2)
    C_stat = 6.0 * b0 * (3.0 * j2 - k1)

    return A_stat, B_stat, C_stat


def _compute_ABC_polynomial(
    omega: float,
    a: float,
    alpha: float,
    beta: float,
    rho: float,
    n_gauss: int = N_GAUSS,
    n_taylor: int = N_TAYLOR,
) -> Tuple[complex, complex, complex]:
    """
    Compute A^c, B^c, C^c = static Eshelby + smooth radiation corrections.

    The full second-derivative integral tensor decomposes as:
      I_{ijkl} = I^stat_{ijkl} + I^smooth_{ijkl}

    The static part comes from the 1/r Kelvin Green's tensor whose
    second derivatives have a 1/r³ (Eshelby) singularity.  This is
    evaluated analytically via surface integrals (divergence theorem).

    The smooth part comes from the Taylor-expanded polynomial Green's
    tensor G^s_{ij}(x) = δ_{ij} Φ(r²) + x_i x_j Ψ(r²), whose second
    derivatives are regular polynomials integrated by GL quadrature.

    The three needed integrals are:
      I_{1111} = ∫ G_{11,11} d³x = A + 2B + C
      I_{1122} = ∫ G_{11,22} d³x = A
      I_{1212} = ∫ G_{12,12} d³x = B

    Smooth-part formulae (G^s_{ij,kl} = ∂²/∂x_k∂x_l [δ_{ij}Φ + x_ix_jΨ]):

      G^s_{11,11} = 4x₁²Φ'' + 2Φ' + 2Ψ + 10x₁²Ψ' + 4x₁⁴Ψ''
      G^s_{11,22} = 4x₂²Φ'' + 2Φ' + 2x₁²Ψ' + 4x₁²x₂²Ψ''
      G^s_{12,12} = Ψ + 2(x₁²+x₂²)Ψ' + 4x₁²x₂²Ψ''

    where Φ', Φ'', Ψ', Ψ'' are derivatives w.r.t. u = r².
    """
    # ── Static Eshelby depolarization (frequency-independent, real) ──
    A_stat, B_stat, C_stat = _static_eshelby_ABC(alpha, beta, rho)

    # ── Smooth radiation corrections (frequency-dependent, imaginary) ──
    phi, psi = _compute_taylor_coefficients(omega, alpha, beta, rho, n_taylor)

    # Set up quadrature
    x1, x2, x3, r, wt, mask = _setup_cube_quadrature(a, n_gauss)

    # u = r² at all points (including origin — polynomial is regular there)
    u = x1**2 + x2**2 + x3**2

    # Evaluate Φ and derivatives w.r.t. u
    Phi, Phi_p, Phi_pp = _eval_poly_and_derivs_safe(u, phi)

    # Evaluate Ψ and derivatives w.r.t. u
    Psi, Psi_p, Psi_pp = _eval_poly_and_derivs_safe(u, psi)

    x1sq = x1**2
    x2sq = x2**2

    # G^s_{11,11}
    G_1111 = (
        4.0 * x1sq * Phi_pp
        + 2.0 * Phi_p
        + 2.0 * Psi
        + 10.0 * x1sq * Psi_p
        + 4.0 * x1sq**2 * Psi_pp
    )

    # G^s_{11,22}
    G_1122 = (
        4.0 * x2sq * Phi_pp
        + 2.0 * Phi_p
        + 2.0 * x1sq * Psi_p
        + 4.0 * x1sq * x2sq * Psi_pp
    )

    # G^s_{12,12}
    G_1212 = Psi + 2.0 * (x1sq + x2sq) * Psi_p + 4.0 * x1sq * x2sq * Psi_pp

    I_1111_smooth = np.sum(wt * G_1111)
    I_1122_smooth = np.sum(wt * G_1122)
    I_1212_smooth = np.sum(wt * G_1212)

    A_smooth = I_1122_smooth
    B_smooth = I_1212_smooth
    C_smooth = I_1111_smooth - I_1122_smooth - 2.0 * I_1212_smooth

    # ── Full = static + smooth ──
    Ac = A_stat + A_smooth
    Bc = B_stat + B_smooth
    Cc = C_stat + C_smooth

    return Ac, Bc, Cc


# ================================================================
# T-matrix coefficients from integral decomposition
# ================================================================


def _compute_T123(
    Ac: complex, Bc: complex, Cc: complex, Dlambda: float, Dmu: float
) -> Tuple[complex, complex, complex]:
    """
    Compute T1^c, T2^c, T3^c from the A^c, B^c, C^c decomposition.

    T_{mnlp} = Σ_{jk} S_{mnjk} Δc_{jklp}
    where S_{mnjk} = (I_{mjkn} + I_{njkm}) / 2.
    """

    def I_tens(i, j, k, l):
        """I_{ijkl} = Ac δ_{ij}δ_{kl} + Bc(δ_{ik}δ_{jl}+δ_{il}δ_{jk}) + Cc E_{ijkl}."""
        iso = Ac * (1 if i == j else 0) * (1 if k == l else 0) + Bc * (
            (1 if i == k else 0) * (1 if j == l else 0)
            + (1 if i == l else 0) * (1 if j == k else 0)
        )
        cubic = Cc * (1 if i == j == k == l else 0)
        return iso + cubic

    def S_tens(m, n, j, k):
        return 0.5 * (I_tens(m, j, k, n) + I_tens(n, j, k, m))

    def Dc(j, k, l, p):
        return Dlambda * (1 if j == k else 0) * (1 if l == p else 0) + Dmu * (
            (1 if j == l else 0) * (1 if k == p else 0)
            + (1 if j == p else 0) * (1 if k == l else 0)
        )

    def T_tens(m, n, l, p):
        return sum(
            S_tens(m, n, j, k) * Dc(j, k, l, p) for j in range(3) for k in range(3)
        )

    T1c = T_tens(0, 0, 1, 1)
    T2c = T_tens(0, 1, 0, 1)
    T3c = T_tens(0, 0, 0, 0) - T1c - 2.0 * T2c

    return T1c, T2c, T3c


# ================================================================
# Amplification factors and effective contrasts
# ================================================================


def _compute_amplification_factors(
    T1c: complex, T2c: complex, T3c: complex, Gamma0: complex, omega: float, Drho: float
) -> Tuple[complex, complex, complex, complex]:
    """
    Four self-consistent amplification factors (Eqs 42-45).

    A_u     = 1 / (1 − ω²Δρ·Γ₀)
    A_θ     = 1 / (1 − 3T1 − 2T2 − T3)
    A_e^off  = 1 / (1 − 2T2)
    A_e^diag = 1 / (1 − 2T2 − T3)
    """
    amp_u = 1.0 / (1.0 - omega**2 * Drho * Gamma0)
    amp_theta = 1.0 / (1.0 - 3.0 * T1c - 2.0 * T2c - T3c)
    amp_e_off = 1.0 / (1.0 - 2.0 * T2c)
    amp_e_diag = 1.0 / (1.0 - 2.0 * T2c - T3c)
    return amp_u, amp_theta, amp_e_off, amp_e_diag


def _compute_effective_contrasts(
    Dlambda: float,
    Dmu: float,
    Drho: float,
    amp_u: complex,
    amp_theta: complex,
    amp_e_off: complex,
    amp_e_diag: complex,
) -> Tuple[complex, complex, complex, complex]:
    """
    Effective contrasts (Eqs 47-50).

    Δρ*      = Δρ · A_u
    Δμ*_off  = Δμ · A_e^off
    Δμ*_diag = Δμ · A_e^diag
    Δλ*      = (Δλ + ⅔Δμ)·A_θ − ⅓Δμ(A_e^diag + A_e^off)
    """
    Drho_star = Drho * amp_u
    Dmu_star_off = Dmu * amp_e_off
    Dmu_star_diag = Dmu * amp_e_diag
    Dlambda_star = (Dlambda + 2.0 / 3.0 * Dmu) * amp_theta - Dmu / 3.0 * (
        amp_e_diag + amp_e_off
    )
    return Drho_star, Dlambda_star, Dmu_star_off, Dmu_star_diag


# ================================================================
# Main computation function
# ================================================================


def compute_cube_tmatrix(
    omega: float,
    a: float,
    ref: ReferenceMedium,
    contrast: MaterialContrast,
    n_gauss: int = N_GAUSS,
    n_taylor: int = N_TAYLOR,
) -> CubeTMatrixResult:
    """
    Compute the full self-consistent cubic T-matrix for a single scatterer.

    Uses a hybrid approach:
      - Γ₀: 3D GL quadrature of the full Green's tensor (convergent).
      - A, B, C: polynomial Taylor expansion (smooth part, no Eshelby singularity).

    Parameters
    ----------
    omega : float
        Angular frequency (rad/s).
    a : float
        Cube half-width (m). Cube extends from [-a, a]^3.
    ref : ReferenceMedium
        Background elastic medium.
    contrast : MaterialContrast
        Material property contrasts (Δλ, Δμ, Δρ).
    n_gauss : int
        Number of GL quadrature points per dimension (default 32).
    n_taylor : int
        Number of Taylor series terms for polynomial method (default 8).

    Returns
    -------
    CubeTMatrixResult
        Complete T-matrix result including effective contrasts.
    """
    alpha, beta, rho = ref.alpha, ref.beta, ref.rho

    # Step 1: Green's tensor volume integral (Γ₀) — full quadrature
    Gamma0 = _compute_Gamma0_numerical(omega, a, alpha, beta, rho, n_gauss)

    # Step 2: Second-derivative integrals (A^c, B^c, C^c) — polynomial method
    Ac, Bc, Cc = _compute_ABC_polynomial(omega, a, alpha, beta, rho, n_gauss, n_taylor)

    # Step 3: T-matrix coupling coefficients
    T1c, T2c, T3c = _compute_T123(Ac, Bc, Cc, contrast.Dlambda, contrast.Dmu)

    # Step 4: Amplification factors
    amp_u, amp_theta, amp_e_off, amp_e_diag = _compute_amplification_factors(
        T1c, T2c, T3c, Gamma0, omega, contrast.Drho
    )

    # Step 5: Effective contrasts
    Drho_star, Dlambda_star, Dmu_star_off, Dmu_star_diag = _compute_effective_contrasts(
        contrast.Dlambda,
        contrast.Dmu,
        contrast.Drho,
        amp_u,
        amp_theta,
        amp_e_off,
        amp_e_diag,
    )

    return CubeTMatrixResult(
        Gamma0=Gamma0,
        Ac=Ac,
        Bc=Bc,
        Cc=Cc,
        T1c=T1c,
        T2c=T2c,
        T3c=T3c,
        amp_u=amp_u,
        amp_theta=amp_theta,
        amp_e_off=amp_e_off,
        amp_e_diag=amp_e_diag,
        Drho_star=Drho_star,
        Dlambda_star=Dlambda_star,
        Dmu_star_off=Dmu_star_off,
        Dmu_star_diag=Dmu_star_diag,
    )
