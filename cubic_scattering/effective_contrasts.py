"""
effective_contrasts.py
Compute the self-consistent cubic T-matrix effective contrasts.

All integrals are computed analytically:

  Γ₀ (Green's tensor integral):
    Static part via geometric constant g₀ = ∫_{[-1,1]³} 1/|x| d³x
    (computed analytically via divergence theorem).
    Smooth part via cube monomial moments S₀, S₁.

  A^c, B^c, C^c (second-derivative integrals):
    Static Eshelby part via surface integrals (divergence theorem).
    Smooth radiation part via exact polynomial integration using
    cube monomial moments and the trinomial expansion of u^m.

Reference: TMatrix_Derivation.pdf (Part II, Sections 11-16).

Coordinate system: generic (1,2,3) Cartesian — the coordinate
relabelling to (z,x,y) is handled in voigt_tmatrix.py.

All computation is pure NumPy (no SymPy dependency).
"""

from dataclasses import dataclass
from math import factorial
from typing import Tuple

import numpy as np

# Default Taylor expansion order (number of phi/psi terms)
N_TAYLOR = 8  # number of Taylor terms for phi and psi
N_GAUSS = 32  # kept for backward-compatible function signatures


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
# Γ₀ computation: analytical (static + smooth polynomial)
# ================================================================

# Geometric constant: g₀ = ∫_{[-1,1]³} 1/|x| d³x  (PDF Eq 33)
# Exact Mathematica result: g₀ = -(2/3)(3π + 2ln(70226 - 40545√3))
# Using Pell identity 70226² − 3·40545² = 1 for numerical stability:
G0_CUBE = (4.0 / 3.0) * np.log(70226 + 40545 * np.sqrt(3.0)) - 2.0 * np.pi


def _compute_Gamma0_analytical(
    omega: float,
    a: float,
    alpha: float,
    beta: float,
    rho: float,
    n_taylor: int = N_TAYLOR,
) -> complex:
    """
    Compute Γ₀^cube = ∫_cube G_{11}(x) d³x analytically.

    Splits into static + smooth parts (PDF Eqs 34-35):

      Γ₀^stat = a²(2α² + β²)/(12πρα²β²) · g₀

    where a₀, b₀ are static Kelvin coefficients and
    g₀ = -(2/3)(3π + 2ln(70226 - 40545√3)) (PDF Eq 33).

      Γ₀^smooth = Σ_n φ_n S₀(n) + Σ_n ψ_n S₁(n)

    from the Taylor-expanded polynomial part G^s_{11} = Φ(u) + x₁²Ψ(u).
    """
    # Static Kelvin coefficients
    a0 = (alpha**2 + beta**2) / (8.0 * np.pi * rho * alpha**2 * beta**2)
    b0 = (alpha**2 - beta**2) / (8.0 * np.pi * rho * alpha**2 * beta**2)

    Gamma0_stat = a**2 * (a0 + b0 / 3.0) * G0_CUBE

    # Smooth part via cube moments
    phi, psi = _compute_taylor_coefficients(omega, alpha, beta, rho, n_taylor)
    S0, S1, S2, S11 = _compute_cube_moments(a, n_taylor - 1)

    N = n_taylor
    Gamma0_smooth = complex(np.sum(phi[:N] * S0[:N]) + np.sum(psi[:N] * S1[:N]))

    return Gamma0_stat + Gamma0_smooth


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


def _compute_cube_moments(
    a: float, n_max: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute exact monomial moments of the cube [-a,a]³.

    The 1D even moment is μ_k = ∫_{-a}^{a} t^{2k} dt = 2a^{2k+1}/(2k+1).
    Using the trinomial expansion u^m = (x₁²+x₂²+x₃²)^m, we compute:

        S₀(m) = ∫ u^m dV,    S₁(m) = ∫ x₁² u^m dV,
        S₂(m) = ∫ x₁⁴ u^m dV,  S₁₁(m) = ∫ x₁²x₂² u^m dV

    for m = 0, ..., n_max via trinomial sums over (p,q,r) with p+q+r = m.

    Returns (S0, S1, S2, S11) each of length n_max + 1.
    """
    # 1D moments: μ_k = 2a^{2k+1}/(2k+1) for k = 0, ..., n_max+2
    max_k = n_max + 2
    mu = np.array([2.0 * a ** (2 * k + 1) / (2 * k + 1) for k in range(max_k + 1)])

    S0 = np.zeros(n_max + 1)
    S1 = np.zeros(n_max + 1)
    S2 = np.zeros(n_max + 1)
    S11 = np.zeros(n_max + 1)

    for m in range(n_max + 1):
        for p in range(m + 1):
            for q in range(m - p + 1):
                r = m - p - q
                coeff = factorial(m) / (factorial(p) * factorial(q) * factorial(r))
                mu_p_q_r = mu[p] * mu[q] * mu[r]
                S0[m] += coeff * mu_p_q_r
                S1[m] += coeff * mu[p + 1] * mu[q] * mu[r]
                S2[m] += coeff * mu[p + 2] * mu[q] * mu[r]
                S11[m] += coeff * mu[p + 1] * mu[q + 1] * mu[r]

    return S0, S1, S2, S11


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

    The smooth part uses exact analytical integration of polynomial
    integrands via cube monomial moments. The Taylor-expanded smooth
    Green's tensor G^s_{ij} = δ_{ij}Φ(u) + x_ix_jΨ(u) with u = r²
    has polynomial second derivatives.  Each term u^m · x_i^{2p}
    integrates exactly over [-a,a]³ via factorised 1D moments
    μ_k = 2a^{2k+1}/(2k+1) and the trinomial expansion of u^m.

    The three smooth integrals reduce to:
      A_smooth = 4Σ n(n-1)φ_n S₁(n-2) + 2Σ nφ_n S₀(n-1)
               + 2Σ nψ_n S₁(n-1) + 4Σ n(n-1)ψ_n S₁₁(n-2)
      B_smooth = Σ ψ_n S₀(n) + 4Σ nψ_n S₁(n-1)
               + 4Σ n(n-1)ψ_n S₁₁(n-2)
      C_smooth = 4Σ n(n-1)ψ_n [S₂(n-2) − 3S₁₁(n-2)]

    Note: C_smooth depends only on ψ (not φ), confirming O((ka)⁷) scaling.
    """
    # ── Static Eshelby depolarization (frequency-independent, real) ──
    A_stat, B_stat, C_stat = _static_eshelby_ABC(alpha, beta, rho)

    # ── Smooth radiation corrections (exact analytical moments) ──
    phi, psi = _compute_taylor_coefficients(omega, alpha, beta, rho, n_taylor)
    S0, S1, S2, S11 = _compute_cube_moments(a, n_taylor - 1)

    N = n_taylor
    n_idx = np.arange(N)

    # A_smooth = I_{1122}
    A_smooth = complex(0)
    if N > 2:
        nn = n_idx[2:]
        A_smooth += np.sum(4 * nn * (nn - 1) * phi[2:] * S1[: N - 2])
        A_smooth += np.sum(4 * nn * (nn - 1) * psi[2:] * S11[: N - 2])
    if N > 1:
        nn = n_idx[1:]
        A_smooth += np.sum(2 * nn * phi[1:] * S0[: N - 1])
        A_smooth += np.sum(2 * nn * psi[1:] * S1[: N - 1])

    # B_smooth = I_{1212}
    B_smooth = complex(np.sum(psi * S0[:N]))
    if N > 1:
        nn = n_idx[1:]
        B_smooth += np.sum(4 * nn * psi[1:] * S1[: N - 1])
    if N > 2:
        nn = n_idx[2:]
        B_smooth += np.sum(4 * nn * (nn - 1) * psi[2:] * S11[: N - 2])

    # C_smooth = I_{1111} − I_{1122} − 2I_{1212}  (depends only on ψ)
    C_smooth = complex(0)
    if N > 2:
        nn = n_idx[2:]
        C_smooth += np.sum(
            4 * nn * (nn - 1) * psi[2:] * (S2[: N - 2] - 3 * S11[: N - 2])
        )

    # ── Full = static + smooth ──
    return A_stat + A_smooth, B_stat + B_smooth, C_stat + C_smooth


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
    Δλ*      = (Δλ + ⅔Δμ)·A_θ − ⅔Δμ·A_e^diag
    """
    Drho_star = Drho * amp_u
    Dmu_star_off = Dmu * amp_e_off
    Dmu_star_diag = Dmu * amp_e_diag
    Dlambda_star = (
        Dlambda + 2.0 / 3.0 * Dmu
    ) * amp_theta - 2.0 / 3.0 * Dmu * amp_e_diag
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

    All integrals computed analytically (no quadrature):
      - Γ₀: static via g₀ geometric constant + smooth via S₀, S₁ moments.
      - A, B, C: static Eshelby + smooth via S₀, S₁, S₂, S₁₁ moments.

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

    # Step 1: Green's tensor volume integral (Γ₀) — analytical
    Gamma0 = _compute_Gamma0_analytical(omega, a, alpha, beta, rho, n_taylor)

    # Step 2: Second-derivative integrals (A^c, B^c, C^c) — analytical moments
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
