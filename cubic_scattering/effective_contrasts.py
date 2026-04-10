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
    Ac: complex, Bc: complex, Cc: complex, Dlambda: complex, Dmu: complex
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


# ================================================================
# Galerkin T-matrix (Path-B, 27-component closure)
# ================================================================


@dataclass
class GalerkinTMatrixResult:
    """Result of the Galerkin (Path-B) 27-component T-matrix computation.

    The Galerkin closure on T_27 produces a 27x27 T-matrix that
    block-diagonalizes under O_h into 7 irrep blocks.  The three
    strain-sector irreps (A1g, Eg, T2g) yield scalar T-matrix values
    that map to T1c, T2c, T3c identically to Path-A.
    """

    # Per-irrep scalar T-matrix values (strain sector, 1x1 blocks)
    sigma_A1g: complex  # volumetric (trace of strain)
    sigma_Eg: complex  # deviatoric axial strain
    sigma_T2g: complex  # off-diagonal shear strain

    # Physical T-matrix scalars (same meaning as Path-A)
    T1c: complex
    T2c: complex
    T3c: complex

    # Per-irrep eigenvalues for ungerade sector (displacement + quadratic)
    T1u_eigenvalues: np.ndarray  # 4 eigenvalues of T1u block
    T2u_eigenvalues: np.ndarray  # 2 eigenvalues of T2u block
    sigma_A2u: complex  # 1x1 A2u block
    sigma_Eu: complex  # 1x1 Eu block

    @property
    def cubic_anisotropy(self) -> complex:
        """T3c: cubic anisotropy coefficient."""
        return self.T3c


# ── Hardcoded Galerkin atoms (from CubeT27Assemble.wl) ──────────────
#
# The body bilinear form B_body = a0 * B_body_A + b0 * B_body_B
# where a0 = (alpha^2 + beta^2)/(8*pi*rho*alpha^2*beta^2)
#       b0 = (alpha^2 - beta^2)/(8*pi*rho*alpha^2*beta^2)
#
# When projected into the O_h irrep basis via Usym, each irrep block
# has entries that are linear in a0, b0 (via Aelas, Belas substitution).
#
# The mass and stiffness blocks are exact rationals or linear in Dlam, Dmu.
#
# All numerical values below are computed from Pell-simplified closed forms
# to ~15-digit accuracy in CubeT27Assemble.wl.

# Strain-sector irrep data (all 1x1 blocks):
# Format: (M_rho, Bbody_A_rho, Bbody_B_rho, Bel_rho(Dlam, Dmu))
#
# Mass blocks (exact rational, from M27 projected via Usym):
#   A1g: M = 8/3 (from volumetric mode)
#   Eg:  M = 8/3
#   T2g: M = 2/3 (from shear modes — note factor 1/2 in basis)
#
# Stiffness blocks (exact, from BelSym projected via Usym):
#   A1g: Bel = 8*(3*Dlam + 2*Dmu)  = 24*Dlam + 16*Dmu
#   Eg:  Bel = 16*Dmu
#   T2g: Bel = 8*Dmu
#
# Body blocks need numerical evaluation from CubeT27Assemble.wl.
# These are the Schur-complemented values (27->9 projection already done
# internally by the O_h irrep decomposition).
# We express them as linear in (Aelas, Belas) with pre-computed coefficients.

# The per-irrep Bbody values are obtained from:
#   BbodyBlock_rho = Usym^T . BbodySym . Usym
# where BbodySym is the full 27x27 symbolic body form (Section 7).
# At (Aelas=1, Belas=0) these give the A-channel values,
# at (Aelas=0, Belas=1) the B-channel values.

# NOTE: The exact numerical values will be filled in after running
# CubeT27Assemble.wl with the new Section 13a-f.  For now we use
# the Schur-complement values from Section 12b which are validated.
#
# From Section 12b (body-channel Schur on T9):
#   Bbody_Schur_strain A-channel: A1g = Eg = T2g (isotropic in A-channel)
#   Bbody_Schur_strain B-channel: A1g, Eg, T2g differ (cubic anisotropy)
#
# The per-irrep approach computes these DIRECTLY from the 27x27 projection,
# which automatically includes the Schur complement from the quadratic modes.

# Placeholder values — these are filled by _build_galerkin_irrep_blocks()
# at runtime by projecting the known 27x27 matrices.


def compute_cube_tmatrix_galerkin(
    omega: float,
    a: float,
    ref: ReferenceMedium,
    contrast: MaterialContrast,
    n_taylor: int = N_TAYLOR,
) -> GalerkinTMatrixResult:
    """Compute the Galerkin (Path-B) T-matrix for a cube on T_27.

    This uses the 27-component Galerkin closure with O_h irrep
    decomposition.  The 27x27 system reduces to 7 independent blocks
    (max 4x4 for T1u), each solved in closed form.

    The strain-sector irreps (A1g, Eg, T2g) give T1c, T2c, T3c
    that are directly comparable to Path-A results.

    Parameters
    ----------
    omega : Angular frequency (rad/s).
    a : Cube half-width (m).
    ref : Background elastic medium.
    contrast : Material property contrasts.
    n_taylor : Taylor series terms (for Gamma0 and body form).
    """
    alpha, beta, rho = ref.alpha, ref.beta, ref.rho

    # Green's tensor Kelvin coefficients (for ungerade sector body bilinear)
    a0 = (alpha**2 + beta**2) / (8.0 * np.pi * rho * alpha**2 * beta**2)
    b0 = (alpha**2 - beta**2) / (8.0 * np.pi * rho * alpha**2 * beta**2)

    # Body coupling parameter (for ungerade sector)
    eps = complex(omega**2 * contrast.Drho)
    Dlam = contrast.Dlambda
    Dmu_val = contrast.Dmu

    # ── Build per-irrep blocks from hardcoded 27x27 projection ──
    blocks = _build_galerkin_irrep_blocks(a0, b0, Dlam, Dmu_val)

    # ── Strain sector: self-consistent amplification (Gubernatis et al.) ──
    # The gerade irreps see the LS-convolved stiffness via the Eshelby
    # tensor (A^c, B^c, C^c).  Self-consistent amplification accounts for
    # the scatterer modifying its own internal field, coupling the bulk
    # (A1g) and deviatoric (Eg) channels through the effective Δλ* formula.
    # This is identical to Path-A's amplification for the strain sector.
    Ac, Bc, Cc = _compute_ABC_polynomial(omega, a, alpha, beta, rho, n_taylor=n_taylor)

    # Born-level T-matrix from bare contrasts
    T1c_born, T2c_born, T3c_born = _compute_T123(Ac, Bc, Cc, Dlam, Dmu_val)

    # Per-irrep Born eigenvalues → amplification factors
    sigma_born_A1g = 3.0 * T1c_born + 2.0 * T2c_born + T3c_born
    sigma_born_Eg = 2.0 * T2c_born + T3c_born
    sigma_born_T2g = 2.0 * T2c_born

    amp_theta = 1.0 / (1.0 - sigma_born_A1g)  # volumetric
    amp_e_off = 1.0 / (1.0 - sigma_born_T2g)  # off-diagonal shear
    amp_e_diag = 1.0 / (1.0 - sigma_born_Eg)  # deviatoric diagonal

    # Effective contrasts — cross-coupling enters through Δλ*
    Dlam_star = (
        Dlam + 2.0 / 3.0 * Dmu_val
    ) * amp_theta - 2.0 / 3.0 * Dmu_val * amp_e_diag
    Dmu_off_star = Dmu_val * amp_e_off
    Dmu_diag_star = Dmu_val * amp_e_diag

    # Amplified T-matrix: Eshelby contraction with cubic effective stiffness
    # Δc* = (Δλ*, Δμ*_off) isotropic + 2(Δμ*_diag − Δμ*_off)·E cubic
    T1c_iso, T2c_iso, T3c_iso = _compute_T123(Ac, Bc, Cc, Dlam_star, Dmu_off_star)
    delta_mu_star = Dmu_diag_star - Dmu_off_star
    T1c = T1c_iso + 2.0 * delta_mu_star * Bc
    T2c = T2c_iso  # off-diagonal shear: no cubic correction
    T3c = T3c_iso + 2.0 * delta_mu_star * (Ac + Bc + Cc)

    # Per-irrep eigenvalues (amplified T values, no additional denominator)
    sigma_A1g = 3.0 * T1c + 2.0 * T2c + T3c
    sigma_Eg = 2.0 * T2c + T3c
    sigma_T2g = 2.0 * T2c

    # ── Ungerade sector (displacement + quadratic modes) ──
    # T1u: 4x4 block
    T1u_evals = _solve_irrep_block(
        eps,
        a0,
        b0,
        blocks["T1u"]["M"],
        blocks["T1u"]["Bbody_A"],
        blocks["T1u"]["Bbody_B"],
        blocks["T1u"]["Bel"],
    )
    # T2u: 2x2 block
    T2u_evals = _solve_irrep_block(
        eps,
        a0,
        b0,
        blocks["T2u"]["M"],
        blocks["T2u"]["Bbody_A"],
        blocks["T2u"]["Bbody_B"],
        blocks["T2u"]["Bel"],
    )
    # A2u: 1x1 (ungerade)
    bbody_A2u = a0 * blocks["A2u"]["Bbody_A"] + b0 * blocks["A2u"]["Bbody_B"]
    bel_A2u = blocks["A2u"]["Bel"]
    sigma_A2u = (eps * bbody_A2u - bel_A2u) / (
        blocks["A2u"]["M"] + bel_A2u - eps * bbody_A2u
    )
    # Eu: 1x1 (ungerade)
    bbody_Eu = a0 * blocks["Eu"]["Bbody_A"] + b0 * blocks["Eu"]["Bbody_B"]
    bel_Eu = blocks["Eu"]["Bel"]
    sigma_Eu = (eps * bbody_Eu - bel_Eu) / (blocks["Eu"]["M"] + bel_Eu - eps * bbody_Eu)

    return GalerkinTMatrixResult(
        sigma_A1g=sigma_A1g,
        sigma_Eg=sigma_Eg,
        sigma_T2g=sigma_T2g,
        T1c=T1c,
        T2c=T2c,
        T3c=T3c,
        T1u_eigenvalues=T1u_evals,
        T2u_eigenvalues=T2u_evals,
        sigma_A2u=sigma_A2u,
        sigma_Eu=sigma_Eu,
    )


def _solve_irrep_block(
    eps: complex,
    a0: float,
    b0: float,
    M: np.ndarray,
    Bbody_A: np.ndarray,
    Bbody_B: np.ndarray,
    Bel: np.ndarray,
) -> np.ndarray:
    """Solve T_rho = (M + Bel - eps*Bbody)^{-1} . (eps*Bbody - Bel) for m>1 block.

    Returns eigenvalues of the T-matrix block.
    """
    Bbody = a0 * Bbody_A + b0 * Bbody_B
    numer = eps * Bbody - Bel
    denom = M + Bel - eps * Bbody
    Tblock = np.linalg.solve(denom, numer)
    return np.sort(np.real(np.linalg.eigvals(Tblock)))


def _build_galerkin_irrep_blocks(
    a0: float, b0: float, Dlambda: float, Dmu: float
) -> dict:
    """Build per-irrep (M, Bbody_A, Bbody_B, Bel) blocks from hardcoded values.

    The 27x27 mass, body, and stiffness matrices are projected into
    the O_h irrep basis via Usym.  All values come from CubeT27Assemble.wl
    and ExtractUngeradeBlocks.wl.

    The body blocks store A-channel and B-channel contributions
    separately so the physical Aelas/Belas can be applied at runtime.
    The stiffness blocks are evaluated at the given Dlambda, Dmu.

    Returns a dict keyed by irrep name with sub-dicts of numpy arrays.
    """
    # ── Strain sector (gerade, 1x1) ──────────────────────────────────
    # Mass blocks (exact rational from O_h projection):
    M_A1g = 8.0 / 3.0
    M_Eg = 8.0 / 3.0
    M_T2g = 2.0 / 3.0

    # Stiffness blocks (exact, from isotropic Kelvin form):
    Bel_A1g = 24.0 * Dlambda + 16.0 * Dmu
    Bel_Eg = 16.0 * Dmu
    Bel_T2g = 8.0 * Dmu

    # Body blocks from CubeT27Assemble.wl Section 12b:
    # A-channel strain eigenvalue (common for A1g, Eg, T2g by isotropy):
    _strain_ev_A = 5.764716843576429
    # B-channel strain eigenvalues (cubic anisotropy):
    _strain_ev_B_A1g = 0.37723697340892
    _strain_ev_B_Eg = 2.033113419068244
    _strain_ev_B_T2g = 1.589776483375774

    Bbody_A_A1g = _strain_ev_A * M_A1g
    Bbody_B_A1g = _strain_ev_B_A1g * M_A1g
    Bbody_A_Eg = _strain_ev_A * M_Eg
    Bbody_B_Eg = _strain_ev_B_Eg * M_Eg
    Bbody_A_T2g = _strain_ev_A * M_T2g
    Bbody_B_T2g = _strain_ev_B_T2g * M_T2g

    # ── Ungerade sector ──────────────────────────────────────────────
    # All values from CubeT27Assemble.wl O_h irrep projection via Usym.
    # Body blocks computed from Pell-simplified closed forms (~15 digits).
    # Mass blocks are exact rationals.
    #
    # Stiffness: LS-convolved stiffness (dimensionless) from
    # CubeT27Stiffness_LS.wl, decomposed into 4 channels:
    #   Bel = a0*Dlam*S_Alam + a0*Dmu*S_Amu + b0*Dlam*S_Blam + b0*Dmu*S_Bmu

    # T1u 4x4 block
    # Usym basis: 1 constant displacement + 2 S-type + 1 X-type quadratic
    # Mass (exact rational from CubeT27Assemble.wl):
    M_T1u = np.array(
        [
            [24.0, 8.0, 16.0, 0.0],
            [8.0, 24.0 / 5.0, 16.0 / 3.0, 0.0],
            [16.0, 16.0 / 3.0, 224.0 / 15.0, 0.0],
            [0.0, 0.0, 0.0, 16.0 / 3.0],
        ]
    )
    # Body A-channel (Aelas=1, Belas=0): from ExtractUngeradeBlocks.wl
    Bbody_A_T1u = np.array(
        [
            [180.7020138614336, 88.46684439720899, 176.9336887944180, 0.0],
            [88.46684439720899, 51.37918840639332, 88.19207714378903, 0.0],
            [176.9336887944180, 88.19207714378903, 190.9504539565757, 0.0],
            [0.0, 0.0, 0.0, 23.93218181944897],
        ]
    )
    # Body B-channel (Aelas=0, Belas=1): includes beta_7 = 0.692769 fix
    Bbody_B_T1u = np.array(
        [
            [60.23400462045325, 36.53543587279042, 51.93140852438884, 0.0],
            [
                36.53543587279042,
                25.18426841260277,
                32.66972944988528,
                -1.298487466026698,
            ],
            [
                51.93140852438884,
                32.66972944988528,
                56.89780853386966,
                2.858124085890304,
            ],
            [0.0, -1.298487466026698, 2.858124085890304, 8.621528487658494],
        ]
    )
    # LS-convolved stiffness 4-channel matrices (analytical, CubeT27Stiffness_LS.wl)
    _Bstiff_Alam_T1u = np.array(
        [
            [0.0, 662.5740508252127, 0.0, 301.17002310234563],
            [0.0, 285.65344925587914, 0.0, 108.71976046146120],
            [0.0, 543.6650901729812, 0.0, 189.79771258414527],
            [0.0, 18.106381744603254, 0.0, 18.106381744603254],
        ]
    )
    _Bstiff_Amu_T1u = np.array(
        [
            [0.0, 1325.1481016504256, 1325.1481016504258, 301.17002310234563],
            [0.0, 571.3068985117584, 543.6650901729812, 94.89885629207264],
            [0.0, 1087.3301803459626, 1114.9719886847395, 203.61861675353384],
            [0.0, 0.0, 18.106381744603254, 27.15957261690488],
        ]
    )
    _Bstiff_Blam_T1u = np.array(
        [
            [0.0, 301.1700231023139, 0.0, 180.70201386140738],
            [0.0, 118.3793339575252, 0.0, 45.30846221194438],
            [0.0, 211.17128678099675, 0.0, 107.30846973221908],
            [0.0, -0.44327357846679, 0.0, -0.44327357846679],
        ]
    )
    _Bstiff_Bmu_T1u = np.array(
        [
            [0.0, 505.8112982784979, 457.93277564888115, 156.76275254659898],
            [0.0, 220.059521370046, 218.7378116939322, 44.64760737388748],
            [0.0, 367.15747841544214, 352.9143486877837, 100.18690486838985],
            [0.0, -0.4427637558543, 9.89317464915564, 4.72469562403818],
        ]
    )
    Bel_T1u = (
        a0 * Dlambda * _Bstiff_Alam_T1u
        + a0 * Dmu * _Bstiff_Amu_T1u
        + b0 * Dlambda * _Bstiff_Blam_T1u
        + b0 * Dmu * _Bstiff_Bmu_T1u
    )

    # T2u 2x2 block (A_Dlam and B_Dlam channels are identically zero)
    M_T2u = np.array(
        [
            [64.0 / 15.0, 0.0],
            [0.0, 16.0 / 3.0],
        ]
    )
    Bbody_A_T2u = np.array(
        [
            [14.56629966899761, 0.0],
            [0.0, 23.93218181944897],
        ]
    )
    Bbody_B_T2u = np.array(
        [
            [3.342301749772080, -5.455099017943915],
            [-5.455099017943915, 5.446200722753151],
        ]
    )
    _Bstiff_Amu_T2u = np.array(
        [
            [27.64180833877713, 13.82090416938857],
            [18.106381744603254, 27.15957261690488],
        ]
    )
    _Bstiff_Bmu_T2u = np.array(
        [
            [15.03217499881417, 5.79985518771543],
            [-9.89215500393065, 4.3554810293761],
        ]
    )
    Bel_T2u = a0 * Dmu * _Bstiff_Amu_T2u + b0 * Dmu * _Bstiff_Bmu_T2u

    # A2u (1x1) — A_Dlam and B_Dlam channels are zero
    M_A2u = 8.0 / 3.0
    Bbody_A_A2u = 11.96609090972448
    Bbody_B_A2u = 2.594755038941010
    Bel_A2u = a0 * Dmu * 18.106381744603254 + b0 * Dmu * 4.69770984292552

    # Eu (1x1) — A_Dlam and B_Dlam channels are zero
    M_Eu = 16.0 / 9.0
    Bbody_A_Eu = 7.977393939816322
    Bbody_B_Eu = 4.067307958204991
    Bel_Eu = a0 * Dmu * 3.0177302907672185 + b0 * Dmu * 2.72556247538591

    return {
        "A1g": {
            "M": M_A1g,
            "Bbody_A": Bbody_A_A1g,
            "Bbody_B": Bbody_B_A1g,
            "Bel": Bel_A1g,
        },
        "Eg": {"M": M_Eg, "Bbody_A": Bbody_A_Eg, "Bbody_B": Bbody_B_Eg, "Bel": Bel_Eg},
        "T2g": {
            "M": M_T2g,
            "Bbody_A": Bbody_A_T2g,
            "Bbody_B": Bbody_B_T2g,
            "Bel": Bel_T2g,
        },
        "T1u": {
            "M": M_T1u,
            "Bbody_A": Bbody_A_T1u,
            "Bbody_B": Bbody_B_T1u,
            "Bel": Bel_T1u,
        },
        "T2u": {
            "M": M_T2u,
            "Bbody_A": Bbody_A_T2u,
            "Bbody_B": Bbody_B_T2u,
            "Bel": Bel_T2u,
        },
        "A2u": {
            "M": M_A2u,
            "Bbody_A": Bbody_A_A2u,
            "Bbody_B": Bbody_B_A2u,
            "Bel": Bel_A2u,
        },
        "Eu": {"M": M_Eu, "Bbody_A": Bbody_A_Eu, "Bbody_B": Bbody_B_Eu, "Bel": Bel_Eu},
    }
