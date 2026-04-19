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

    # Physical effective stiffness contrasts (Pa) for far-field computation
    Dlambda_star: complex
    Dmu_star_diag: complex
    Dmu_star_off: complex

    # Full per-irrep T-matrix blocks (for T27 assembly)
    T1u_block: np.ndarray  # (4, 4) T1u block matrix
    T2u_block: np.ndarray  # (2, 2) T2u block matrix

    @property
    def cubic_anisotropy(self) -> complex:
        """T3c: cubic anisotropy coefficient."""
        return self.T3c


@dataclass
class GalerkinTMatrixResult57:
    """Result of the Galerkin (Path-B) 57-component T-matrix computation.

    The T₅₇ closure on T_57 = T_27 ⊕ 30 cubic modes produces a 57×57 T-matrix
    that block-diagonalizes under O_h into 9 irrep blocks.

    The ungerade sector (displacement + quadratic, 21D) is IDENTICAL to T₂₇.
    The gerade sector (strain + cubic, 36D) is enlarged:
      A1g: 3×3 (1 strain + 2 cubic)
      A2g: 1×1 (NEW, purely cubic)
      Eg:  4×4 (1 strain + 3 cubic)
      T1g: 3×3 (NEW, purely cubic)
      T2g: 5×5 (1 strain + 4 cubic)

    The physical T1c, T2c, T3c come from the [0,0] entry of each gerade
    block (the strain basis function is always index 0 within each block).
    """

    # ── Ungerade sector (identical to T₂₇) ──
    T1u_eigenvalues: np.ndarray  # 4 eigenvalues of T1u block
    T2u_eigenvalues: np.ndarray  # 2 eigenvalues of T2u block
    sigma_A2u: complex  # 1×1 A2u block
    sigma_Eu: complex  # 1×1 Eu block
    T1u_block: np.ndarray  # (4, 4)
    T2u_block: np.ndarray  # (2, 2)

    # ── Gerade sector (enlarged) ──
    A1g_block: np.ndarray  # (3, 3)
    A1g_eigenvalues: np.ndarray  # 3 eigenvalues
    sigma_A2g: complex  # 1×1 (NEW)
    Eg_block: np.ndarray  # (4, 4)
    Eg_eigenvalues: np.ndarray  # 4 eigenvalues
    T1g_block: np.ndarray  # (3, 3) (NEW)
    T1g_eigenvalues: np.ndarray  # 3 eigenvalues
    T2g_block: np.ndarray  # (5, 5)
    T2g_eigenvalues: np.ndarray  # 5 eigenvalues

    # ── Per-irrep strain-sector scalars (from [0,0] of gerade blocks) ──
    sigma_A1g: complex
    sigma_Eg: complex
    sigma_T2g: complex

    # ── Physical T-matrix scalars ──
    T1c: complex
    T2c: complex
    T3c: complex

    # ── Physical effective stiffness contrasts ──
    Dlambda_star: complex
    Dmu_star_diag: complex
    Dmu_star_off: complex

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

    # ── Gerade sector: Galerkin body bilinear + LS stiffness + smooth ──
    # Compute smooth body bilinear for the 6 strain-only gerade modes.
    # T₂₇ gerade blocks are 1×1, so we compute the smooth correction
    # as a scalar per irrep from the projected smooth body bilinear.
    from cubic_scattering.compute_gerade_blocks import (
        _build_basis_components as _build_basis_57,
    )
    from cubic_scattering.tmatrix_assembly import _build_usym_27

    basis_27 = _build_basis_57()  # reuse the 57-element basis builder
    Usym_27 = _build_usym_27()
    # T₂₇ gerade indices: strain modes 3-8
    gerade_27 = list(range(3, 9))
    g_idx_27 = np.array(gerade_27)
    # Usym₂₇ gerade columns: 21-26 (6 columns for A1g, Eg, T2g)
    usym_g_27 = Usym_27[np.ix_(g_idx_27, np.arange(21, 27))]

    phi, psi = _compute_taylor_coefficients(omega, alpha, beta, rho, n_taylor)
    from cubic_scattering.compute_gerade_blocks import compute_smooth_body_bilinear

    Bsmooth_6 = np.zeros((6, 6), dtype=complex)
    for n in range(n_taylor):
        BA_n, BB_n = compute_smooth_body_bilinear(basis_27, gerade_27, n)
        Bsmooth_6 += phi[n] * a ** (2 * n) * BA_n
        Bsmooth_6 += psi[n] * a ** (2 * n + 2) * BB_n

    # Project to irrep basis (6×6 → diagonal blocks: A1g=1, Eg=2, T2g=3)
    Bsmooth_proj_27 = usym_g_27.T @ Bsmooth_6 @ usym_g_27
    # Extract 1×1 gerade blocks:
    #   A1g: col 0, Eg: cols 1-2 (take [1,1]), T2g: cols 3-5 (take [3,3])
    smooth_A1g = complex(Bsmooth_proj_27[0, 0])
    smooth_Eg = complex(Bsmooth_proj_27[1, 1])
    smooth_T2g = complex(Bsmooth_proj_27[3, 3])

    # LS-convolved stiffness for strain-only gerade blocks (dimensionless).
    # These are the [0,0] entries of the T₅₇ 4-channel stiffness matrices.
    # Strain modes have zero volume force → stiffness = surface traction only.
    _S_Alam_A1g = 3.01170023102118e01
    _S_Amu_A1g = 2.00780015401412e01
    _S_Blam_A1g = -3.01170023102345e01
    _S_Bmu_A1g = -2.00780015401564e01

    _S_Amu_Eg = 2.00780015401412e01
    _S_Bmu_Eg = 4.05418544137612e00

    _S_Amu_T2g = 1.00390007700706e01
    _S_Bmu_T2g = 1.99493844290071e00

    bel_A1g = (
        a0 * Dlam * _S_Alam_A1g
        + a0 * Dmu_val * _S_Amu_A1g
        + b0 * Dlam * _S_Blam_A1g
        + b0 * Dmu_val * _S_Bmu_A1g
    )
    bel_Eg = a0 * Dmu_val * _S_Amu_Eg + b0 * Dmu_val * _S_Bmu_Eg
    bel_T2g = a0 * Dmu_val * _S_Amu_T2g + b0 * Dmu_val * _S_Bmu_T2g

    # Solve gerade 1×1 blocks with smooth correction
    bbody_A1g = (
        a0 * blocks["A1g"]["Bbody_A"] + b0 * blocks["A1g"]["Bbody_B"] + smooth_A1g
    )
    sigma_A1g = (eps * bbody_A1g - bel_A1g) / (
        blocks["A1g"]["M"] + bel_A1g - eps * bbody_A1g
    )
    bbody_Eg = a0 * blocks["Eg"]["Bbody_A"] + b0 * blocks["Eg"]["Bbody_B"] + smooth_Eg
    sigma_Eg = (eps * bbody_Eg - bel_Eg) / (blocks["Eg"]["M"] + bel_Eg - eps * bbody_Eg)
    bbody_T2g = (
        a0 * blocks["T2g"]["Bbody_A"] + b0 * blocks["T2g"]["Bbody_B"] + smooth_T2g
    )
    sigma_T2g = (eps * bbody_T2g - bel_T2g) / (
        blocks["T2g"]["M"] + bel_T2g - eps * bbody_T2g
    )

    # Extract T1c, T2c, T3c from per-irrep strain eigenvalues
    T2c = sigma_T2g / 2.0
    T3c = sigma_Eg - sigma_T2g
    T1c = (sigma_A1g - sigma_Eg) / 3.0

    # Effective contrasts from amplified σ values
    amp_theta = 1.0 / (1.0 - sigma_A1g)
    amp_e_off = 1.0 / (1.0 - sigma_T2g)
    amp_e_diag = 1.0 / (1.0 - sigma_Eg)

    Dlam_star = (
        Dlam + 2.0 / 3.0 * Dmu_val
    ) * amp_theta - 2.0 / 3.0 * Dmu_val * amp_e_diag
    Dmu_off_star = Dmu_val * amp_e_off
    Dmu_diag_star = Dmu_val * amp_e_diag

    # ── Ungerade sector (displacement + quadratic modes) ──
    # T1u: 4x4 block
    T1u_evals, T1u_block = _solve_irrep_block(
        eps,
        a0,
        b0,
        blocks["T1u"]["M"],
        blocks["T1u"]["Bbody_A"],
        blocks["T1u"]["Bbody_B"],
        blocks["T1u"]["Bel"],
    )
    # T2u: 2x2 block
    T2u_evals, T2u_block = _solve_irrep_block(
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
        Dlambda_star=Dlam_star,
        Dmu_star_diag=Dmu_diag_star,
        Dmu_star_off=Dmu_off_star,
        T1u_block=T1u_block,
        T2u_block=T2u_block,
    )


def _solve_irrep_block(
    eps: complex,
    a0: float,
    b0: float,
    M: np.ndarray,
    Bbody_A: np.ndarray,
    Bbody_B: np.ndarray,
    Bel: np.ndarray,
    Bbody_smooth: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve T_rho = (M + Bel - eps*Bbody)^{-1} . (eps*Bbody - Bel) for m>1 block.

    Parameters
    ----------
    Bbody_smooth : optional complex array
        Smooth (dynamic) body bilinear correction. When provided, adds
        frequency-dependent radiation damping to the body bilinear.

    Returns (eigenvalues, Tblock) tuple. Eigenvalues sorted by real part.
    """
    Bbody = a0 * Bbody_A + b0 * Bbody_B
    if Bbody_smooth is not None:
        Bbody = Bbody + Bbody_smooth
    numer = eps * Bbody - Bel
    denom = M + Bel - eps * Bbody
    Tblock = np.linalg.solve(denom, numer)
    evals = np.linalg.eigvals(Tblock)
    # Sort by real part (eigenvalues may be complex with smooth correction)
    order = np.argsort(evals.real)
    return evals[order], Tblock


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

    # Body blocks: A-channel strain eigenvalue (common for A1g, Eg, T2g by isotropy)
    # and B-channel strain eigenvalues (cubic anisotropy).
    # Validated by 6D Monte Carlo (500k samples, ratio = 0.9982) and O_h commutation.
    # NOTE: Earlier values from CubeGalerkin27.wl had a parity bug — the octant
    # fold incorrectly removed the K1at(1,0,0) term (odd u-power, valid on [0,1]³
    # but vanishes on [-1,1]³).  The symmetrized approach in compute_gerade_blocks.py
    # correctly sums σ=±1 contributions per axis.  These corrected values do NOT
    # affect T₂₇ results because the solver uses Path-A (Eshelby) for the gerade
    # sector, but they ARE used by T₅₇.
    _strain_ev_A = 1.755317619825370
    # B-channel strain eigenvalues (cubic anisotropy):
    _strain_ev_B_A1g = -1.611924358631187
    _strain_ev_B_Eg = 0.043992031944107
    _strain_ev_B_T2g = 0.579676728845940

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


# ================================================================
# Galerkin T-matrix (Path-B, 57-component closure)
# ================================================================
#
# The T₅₇ extends T₂₇ with 30 cubic modes (10 monomials × 3 directions).
# All cubic modes are GERADE, so the ungerade sector is unchanged.
# The gerade sector enlarges from 6D (strain only) to 36D (strain + cubic).
#
# Per-irrep block sizes (gerade):
#   A1g: 3×3 (1 strain + 2 cubic)
#   A2g: 1×1 (purely cubic, NEW)
#   Eg:  4×4 (1 strain + 3 cubic)
#   T1g: 3×3 (purely cubic, NEW)
#   T2g: 5×5 (1 strain + 4 cubic)
#
# All gerade numerical values computed by compute_gerade_blocks.py:
# body bilinear (symmetrized A+B channels) + LS-convolved stiffness (volume + surface).


def _build_galerkin_irrep_blocks_57(
    a0: float, b0: float, Dlambda: float, Dmu: float
) -> dict:
    """Build per-irrep (M, Bbody_A, Bbody_B, Bel) blocks for T₅₇.

    Gerade blocks are enlarged; ungerade blocks are identical to T₂₇.
    All gerade numerical values from compute_gerade_blocks.py.

    The body blocks store A-channel and B-channel contributions separately
    so the physical a0/b0 can be applied at runtime.
    The stiffness blocks are evaluated at the given Dlambda, Dmu via
    the 4-channel decomposition: Bel = a0*Dlam*S1 + a0*Dmu*S2 + b0*Dlam*S3 + b0*Dmu*S4.
    """
    # ── Ungerade sector (identical to T₂₇) ──
    # Reuse the T₂₇ block builder for ungerade irreps
    blocks_27 = _build_galerkin_irrep_blocks(a0, b0, Dlambda, Dmu)

    # ── Gerade sector: enlarged blocks ──
    # Basis ordering within each irrep block:
    #   Index 0 = strain basis function
    #   Indices 1+ = cubic basis functions (from Usym₅₇ column ordering)

    # A1g: 3×3 (1 strain + 2 cubic A1g modes)
    # From compute_gerade_blocks.py (symmetrized body bilinear + LS surface stiffness)
    M_A1g = np.array(
        [
            [2.66666666666667e00, -1.25707872210942e00, -1.60000000000000e00],
            [-1.25707872210942e00, 8.29629629629630e-01, 7.54247233265651e-01],
            [-1.60000000000000e00, 7.54247233265651e-01, 1.14285714285714e00],
        ]
    )
    Bbody_A_A1g = np.array(
        [
            [4.68084698620099e00, -1.95578297974614e00, -2.58786335304064e00],
            [-1.95578297974614e00, 9.28934515705655e-01, 1.08299025655914e00],
            [-2.58786335304064e00, 1.08299025655914e00, 1.49705056933698e00],
        ]
    )
    Bbody_B_A1g = np.array(
        [
            [-4.29846495634983e00, 1.60737927922178e00, 2.43676151203526e00],
            [1.60737927922178e00, -6.05599520647960e-01, -9.45258445511073e-01],
            [2.43676151203526e00, -9.45258445511073e-01, -1.43348342046600e00],
        ]
    )
    _Bstiff_Alam_A1g = np.array(
        [
            [3.01170023102118e01, -3.58676752494104e01, -7.60868291827650e01],
            [-1.26464244377180e01, 1.54366188578036e01, 3.27460136188353e01],
            [-1.85903903199458e01, 2.12901201490966e01, 4.51631649891076e01],
        ]
    )
    _Bstiff_Amu_A1g = np.array(
        [
            [2.00780015401412e01, -5.28532577458219e01, -1.16404168454835e02],
            [-8.43094962514530e00, 2.44897172136286e01, 4.87622446323896e01],
            [-1.23935935466305e01, 2.98233469930484e01, 6.82351408763794e01],
        ]
    )
    _Bstiff_Blam_A1g = np.array(
        [
            [-3.01170023102345e01, 3.37045958389208e01, 7.14982448245584e01],
            [1.26464244377339e01, -1.34657499039296e01, -2.85651692124922e01],
            [1.85903903199907e01, -2.04353590577232e01, -4.33499428970940e01],
        ]
    )
    _Bstiff_Bmu_A1g = np.array(
        [
            [-2.00780015401564e01, 4.58232496616640e01, 1.11815584096667e02],
            [8.43094962515592e00, -1.79799202231185e01, -4.45814002261291e01],
            [1.23935935466605e01, -2.68821440516555e01, -6.64219187844046e01],
        ]
    )
    Bel_A1g = (
        a0 * Dlambda * _Bstiff_Alam_A1g
        + a0 * Dmu * _Bstiff_Amu_A1g
        + b0 * Dlambda * _Bstiff_Blam_A1g
        + b0 * Dmu * _Bstiff_Bmu_A1g
    )

    # A2g: 1×1 (purely cubic)
    M_A2g = 2.37037037037037e-01
    Bbody_A_A2g = 1.10038661873406e-01
    Bbody_B_A2g = 9.77720372336787e-02
    _Bstiff_Amu_A2g = 3.94955670903389e-01
    _Bstiff_Bmu_A2g = 2.84556049127716e-01
    Bel_A2g = a0 * Dmu * _Bstiff_Amu_A2g + b0 * Dmu * _Bstiff_Bmu_A2g

    # Eg: 4×4 (1 strain + 3 cubic)
    M_Eg = np.array(
        [
            [2.66666666666667e00, 0.0, -1.25707872210942e00, -1.60000000000000e00],
            [0.0, 2.37037037037037e-01, 0.0, 0.0],
            [-1.25707872210942e00, 0.0, 8.29629629629630e-01, 7.54247233265651e-01],
            [-1.60000000000000e00, 0.0, 7.54247233265651e-01, 1.14285714285714e00],
        ]
    )
    Bbody_A_Eg = np.array(
        [
            [4.68084698620099e00, 0.0, -1.95578297974614e00, -2.58786335304064e00],
            [0.0, 1.10038661873405e-01, 0.0, 0.0],
            [-1.95578297974614e00, 0.0, 9.28934515705657e-01, 1.08299025655915e00],
            [-2.58786335304064e00, 0.0, 1.08299025655915e00, 1.49705056933698e00],
        ]
    )
    Bbody_B_Eg = np.array(
        [
            [
                1.17312085184287e-01,
                -2.27723589429496e-01,
                1.08036919621390e-01,
                -1.55454618317817e-01,
            ],
            [
                -2.27723589429496e-01,
                1.89124462676295e-02,
                6.81141074695810e-02,
                1.01241825389441e-01,
            ],
            [
                1.08036919621390e-01,
                6.81141074695810e-02,
                -3.84645094176192e-02,
                -8.87809165930631e-03,
            ],
            [
                -1.55454618317817e-01,
                1.01241825389441e-01,
                -8.87809165930631e-03,
                9.09582223434946e-02,
            ],
        ]
    )
    _Bstiff_Alam_Eg = np.array(
        [
            [0.0, -2.01101931105757e01, 1.16106254058462e01, -4.92597117497441e01],
            [0.0, 2.96216753177545e-01, -1.71020822185532e-01, 7.25579898548937e-01],
            [0.0, 8.24607521806736e00, -4.76087374690909e00, 2.01986766648745e01],
            [0.0, 1.16737672598868e01, -6.73985266995266e00, 2.85947731627308e01],
        ]
    )
    _Bstiff_Amu_Eg = np.array(
        [
            [
                2.00780015401412e01,
                -1.92354236636229e01,
                -1.95365266553145e01,
                -1.16404168454835e02,
            ],
            [0.0, 1.85367997060265e00, 8.42194867104820e-01, 0.0],
            [
                -8.43094962514530e00,
                8.89688233214019e00,
                9.07986498539988e00,
                4.87622446323896e01,
            ],
            [
                -1.23935935466305e01,
                1.06592155146445e01,
                1.13610441528577e01,
                6.82351408763794e01,
            ],
        ]
    )
    _Bstiff_Blam_Eg = np.array(
        [
            [0.0, -3.39588974863570e00, 1.96061786051311e00, -8.31819710690567e00],
            [0.0, 8.19396439793163e-01, -4.73078755087604e-01, 2.00710317454640e00],
            [0.0, 8.05994825014866e-01, -4.65341329187778e-01, 1.97427605661024e00],
            [0.0, 1.76569120418585e00, -1.01942229204245e00, 4.32504249357570e00],
        ]
    )
    _Bstiff_Bmu_Eg = np.array(
        [
            [
                4.05418544137612e00,
                -3.88090249700993e00,
                -2.28132960295010e00,
                -1.35703013463398e01,
            ],
            [
                -3.50324453643484e-01,
                8.41179947521715e-01,
                5.88274724755393e-01,
                3.78365643408440e00,
            ],
            [
                -1.32448912448191e00,
                1.04212538538011e00,
                6.84097768342287e-01,
                2.67702433798905e00,
            ],
            [
                -1.84490896310383e00,
                2.28055817536835e00,
                1.36852486679632e00,
                7.40018230912525e00,
            ],
        ]
    )
    Bel_Eg = (
        a0 * Dlambda * _Bstiff_Alam_Eg
        + a0 * Dmu * _Bstiff_Amu_Eg
        + b0 * Dlambda * _Bstiff_Blam_Eg
        + b0 * Dmu * _Bstiff_Bmu_Eg
    )

    # T1g: 3×3 (purely cubic, NEW — no Δλ channels: S_Alam=S_Blam=0)
    M_T1g = np.array(
        [
            [1.14285714285714e00, -5.33333333333333e-01, 5.33333333333333e-01],
            [-5.33333333333333e-01, 5.33333333333333e-01, -2.96296296296296e-01],
            [5.33333333333333e-01, -2.96296296296296e-01, 5.33333333333333e-01],
        ]
    )
    Bbody_A_T1g = np.array(
        [
            [1.49705056933698e00, -7.65789754371929e-01, 7.65789754371929e-01],
            [-7.65789754371929e-01, 5.19486588789531e-01, -4.09447926916124e-01],
            [7.65789754371929e-01, -4.09447926916124e-01, 5.19486588789528e-01],
        ]
    )
    Bbody_B_T1g = np.array(
        [
            [1.46526699490391e00, -6.90238833868902e-01, 7.43949577305885e-01],
            [-6.90238833868902e-01, 3.28295573863060e-01, -3.54280322608062e-01],
            [7.43949577305885e-01, -3.54280322608062e-01, 4.97677817279916e-01],
        ]
    )
    _Bstiff_Amu_T1g = np.array(
        [
            [1.30675930444869e01, -2.69910828583313e00, 3.52748631699770e00],
            [-5.68872004250335e00, 3.81768804108661e00, -1.88448116116102e00],
            [6.28115354885843e00, -1.67524447270256e00, 2.85696402762719e00],
        ]
    )
    _Bstiff_Bmu_T1g = np.array(
        [
            [1.35208985675449e01, -1.56116790781987e00, 3.49229319631476e00],
            [-6.83586613207803e00, 1.23122804657989e00, -1.79426694668734e00],
            [6.61215917471479e00, -9.28291104083037e-01, 2.72709804552766e00],
        ]
    )
    Bel_T1g = a0 * Dmu * _Bstiff_Amu_T1g + b0 * Dmu * _Bstiff_Bmu_T1g

    # T2g: 5×5 (1 strain + 4 cubic)
    M_T2g = np.array(
        [
            [
                1.33333333333333e00,
                -1.13137084989848e00,
                -6.28539361054709e-01,
                -6.28539361054709e-01,
                0.0,
            ],
            [
                -1.13137084989848e00,
                1.14285714285714e00,
                5.33333333333333e-01,
                5.33333333333333e-01,
                0.0,
            ],
            [
                -6.28539361054709e-01,
                5.33333333333333e-01,
                5.33333333333333e-01,
                2.96296296296296e-01,
                0.0,
            ],
            [
                -6.28539361054709e-01,
                5.33333333333333e-01,
                2.96296296296296e-01,
                5.33333333333333e-01,
                0.0,
            ],
            [0.0, 0.0, 0.0, 0.0, 2.96296296296296e-01],
        ]
    )
    Bbody_A_T2g = np.array(
        [
            [
                2.34042349310049e00,
                -1.82989572571919e00,
                -9.77891489873070e-01,
                -9.77891489873068e-01,
                0.0,
            ],
            [
                -1.82989572571919e00,
                1.49705056933698e00,
                7.65789754371929e-01,
                7.65789754371929e-01,
                0.0,
            ],
            [
                -9.77891489873070e-01,
                7.65789754371929e-01,
                5.19486588789531e-01,
                4.09447926916124e-01,
                0.0,
            ],
            [
                -9.77891489873068e-01,
                7.65789754371929e-01,
                4.09447926916124e-01,
                5.19486588789528e-01,
                0.0,
            ],
            [0.0, 0.0, 0.0, 0.0, 1.62289486529769e-01],
        ]
    )
    Bbody_B_T2g = np.array(
        [
            [
                7.72902305127920e-01,
                -5.54490754792847e-01,
                -4.74394515841571e-01,
                -3.07625040577538e-01,
                1.43176101621980e-01,
            ],
            [
                -5.54490754792847e-01,
                4.48972566364247e-01,
                3.31488506160633e-01,
                2.19872041042896e-01,
                -1.12371093307849e-01,
            ],
            [
                -4.74394515841571e-01,
                3.31488506160633e-01,
                1.91615436363944e-01,
                1.91521849186632e-01,
                -1.95813897568651e-02,
            ],
            [
                -3.07625040577538e-01,
                2.19872041042896e-01,
                1.91521849186632e-01,
                2.03694886648106e-01,
                -1.95813897568653e-02,
            ],
            [
                1.43176101621980e-01,
                -1.12371093307849e-01,
                -1.95813897568651e-02,
                -1.95813897568653e-02,
                5.40964955103471e-02,
            ],
        ]
    )
    _Bstiff_Alam_T2g = np.array(
        [
            [0.0, 0.0, -3.54505886428956e01, 0.0, 1.25336758132231e01],
            [0.0, 0.0, 2.77578794583206e01, 0.0, -9.81389239816862e00],
            [0.0, 0.0, 1.67775073153162e01, 0.0, -5.93174459703350e00],
            [0.0, 0.0, 1.48325415823838e01, 0.0, -5.24409536756753e00],
            [0.0, 0.0, -8.41543973898678e-01, 0.0, 2.97530725305215e-01],
        ]
    )
    _Bstiff_Amu_T2g = np.array(
        [
            [
                1.00390007700706e01,
                -6.73374525290926e01,
                -5.78964061525932e01,
                -2.19407691340205e01,
                1.25336758132232e01,
            ],
            [
                -8.76359404009229e00,
                5.51675478318925e01,
                4.61470620689514e01,
                1.75608045794662e01,
                -9.81389239816862e00,
            ],
            [
                -4.21547481257265e00,
                2.87913938029367e01,
                2.63746385829618e01,
                9.19041674373540e00,
                -5.24409536756754e00,
            ],
            [
                -4.21547481257265e00,
                2.81989602965815e01,
                2.42321950145777e01,
                1.05578552811049e01,
                -5.93174459703350e00,
            ],
            [0.0, 0.0, 0.0, -8.41543973898676e-01, 1.19012290122086e00],
        ]
    )
    _Bstiff_Blam_T2g = np.array(
        [
            [0.0, 0.0, -1.73843021724031e01, 0.0, 6.14627897615112e00],
            [0.0, 0.0, 1.27065009258415e01, 0.0, -4.49242648490785e00],
            [0.0, 0.0, 7.86860347339083e00, 0.0, -2.78197143725134e00],
            [0.0, 0.0, 6.98339736183948e00, 0.0, -2.46900381513847e00],
            [0.0, 0.0, -1.58823504275468e00, 0.0, 5.61525884424971e-01],
        ]
    )
    _Bstiff_Bmu_T2g = np.array(
        [
            [
                1.99493844290071e00,
                -1.99085280701090e01,
                -2.28981288225635e01,
                -7.22394286274971e00,
                5.36332733566354e00,
            ],
            [
                -2.05149958771621e00,
                1.58683727680466e01,
                1.71162848348287e01,
                5.50013495434587e00,
                -3.94557240431241e00,
            ],
            [
                -1.16592181364682e00,
                1.15809226764477e01,
                1.15987767287491e01,
                4.19573376456785e00,
                -2.73338459696195e00,
            ],
            [
                -7.61401978417260e-01,
                7.85717093612986e00,
                9.16062144821175e00,
                3.52818924836924e00,
                -2.42041697484909e00,
            ],
            [
                6.47536944966110e-01,
                -3.96215609745521e00,
                -2.50903009607504e00,
                -9.20795053320365e-01,
                8.59056609731881e-01,
            ],
        ]
    )
    Bel_T2g = (
        a0 * Dlambda * _Bstiff_Alam_T2g
        + a0 * Dmu * _Bstiff_Amu_T2g
        + b0 * Dlambda * _Bstiff_Blam_T2g
        + b0 * Dmu * _Bstiff_Bmu_T2g
    )

    return {
        # Gerade (enlarged)
        "A1g": {
            "M": M_A1g,
            "Bbody_A": Bbody_A_A1g,
            "Bbody_B": Bbody_B_A1g,
            "Bel": Bel_A1g,
        },
        "A2g": {
            "M": M_A2g,
            "Bbody_A": Bbody_A_A2g,
            "Bbody_B": Bbody_B_A2g,
            "Bel": Bel_A2g,
        },
        "Eg": {"M": M_Eg, "Bbody_A": Bbody_A_Eg, "Bbody_B": Bbody_B_Eg, "Bel": Bel_Eg},
        "T1g": {
            "M": M_T1g,
            "Bbody_A": Bbody_A_T1g,
            "Bbody_B": Bbody_B_T1g,
            "Bel": Bel_T1g,
        },
        "T2g": {
            "M": M_T2g,
            "Bbody_A": Bbody_A_T2g,
            "Bbody_B": Bbody_B_T2g,
            "Bel": Bel_T2g,
        },
        # Ungerade (unchanged from T₂₇)
        "T1u": blocks_27["T1u"],
        "T2u": blocks_27["T2u"],
        "A2u": blocks_27["A2u"],
        "Eu": blocks_27["Eu"],
    }


def _compute_smooth_body_bilinear_projected(
    omega: float,
    a: float,
    alpha: float,
    beta: float,
    rho: float,
    n_taylor: int,
    gerade_indices: list[int],
    usym_gerade: np.ndarray,
    basis,
) -> np.ndarray:
    """Compute projected smooth body bilinear for gerade sector.

    Combines all Taylor orders into a single complex 36×36 matrix,
    then projects through Usym to the irrep basis.

    The smooth body bilinear is:
      Bbody_smooth = Σ_n [φ_n·a^{2n}·BA_smooth_n + ψ_n·a^{2n+2}·BB_smooth_n]

    where BA_smooth_n and BB_smooth_n are geometry-only matrices with
    prefactors 8·4^n and 32·4^n already built in.

    Returns the 36×36 projected smooth body bilinear (complex).
    """
    from cubic_scattering.compute_gerade_blocks import compute_smooth_body_bilinear

    phi, psi = _compute_taylor_coefficients(omega, alpha, beta, rho, n_taylor)

    n_ger = len(gerade_indices)
    Bbody_smooth_raw = np.zeros((n_ger, n_ger), dtype=complex)

    for n in range(n_taylor):
        BA_n, BB_n = compute_smooth_body_bilinear(basis, gerade_indices, n)
        Bbody_smooth_raw += phi[n] * a ** (2 * n) * BA_n
        Bbody_smooth_raw += psi[n] * a ** (2 * n + 2) * BB_n

    # Project through Usym to per-irrep basis
    return usym_gerade.T @ Bbody_smooth_raw @ usym_gerade


def compute_cube_tmatrix_galerkin_57(
    omega: float,
    a: float,
    ref: ReferenceMedium,
    contrast: MaterialContrast,
    n_taylor: int = N_TAYLOR,
) -> GalerkinTMatrixResult57:
    """Compute the Galerkin (Path-B) T-matrix for a cube on T₅₇.

    Extends T₂₇ with 30 cubic trial functions for (ka)⁴ accuracy.
    The ungerade sector (displacement + quadratic, 21D) is identical to T₂₇.
    The gerade sector (strain + cubic, 36D) uses enlarged per-irrep blocks.

    The strain-sector T1c, T2c, T3c are extracted from the [0,0] entry
    of each gerade block, incorporating (ka)⁴ corrections from cubic modes.

    Parameters
    ----------
    omega : Angular frequency (rad/s).
    a : Cube half-width (m).
    ref : Background elastic medium.
    contrast : Material property contrasts.
    n_taylor : Taylor series terms (for Gamma0 and body form).
    """
    alpha, beta, rho = ref.alpha, ref.beta, ref.rho

    a0 = (alpha**2 + beta**2) / (8.0 * np.pi * rho * alpha**2 * beta**2)
    b0 = (alpha**2 - beta**2) / (8.0 * np.pi * rho * alpha**2 * beta**2)

    eps = complex(omega**2 * contrast.Drho)
    Dlam = contrast.Dlambda
    Dmu_val = contrast.Dmu

    blocks = _build_galerkin_irrep_blocks_57(a0, b0, Dlam, Dmu_val)

    # ── Smooth body bilinear (dynamic correction for radiation damping) ──
    from cubic_scattering.compute_gerade_blocks import (
        GERADE_INDICES,
        _build_basis_components,
        _extract_irrep_block,
    )
    from cubic_scattering.tmatrix_assembly import _build_usym_57

    basis = _build_basis_components()
    Usym = _build_usym_57()
    g_idx = np.array(GERADE_INDICES)
    usym_gerade = Usym[np.ix_(g_idx, np.arange(21, 57))]

    Bsmooth_proj = _compute_smooth_body_bilinear_projected(
        omega,
        a,
        alpha,
        beta,
        rho,
        n_taylor,
        list(GERADE_INDICES),
        usym_gerade,
        basis,
    )

    # Extract per-irrep smooth blocks
    smooth_blocks: dict[str, np.ndarray] = {}
    for irrep in ["A1g", "A2g", "Eg", "T1g", "T2g"]:
        smooth_blocks[irrep] = _extract_irrep_block(Bsmooth_proj, irrep)

    # ── Ungerade sector (identical to T₂₇) ──
    T1u_evals, T1u_block = _solve_irrep_block(
        eps,
        a0,
        b0,
        blocks["T1u"]["M"],
        blocks["T1u"]["Bbody_A"],
        blocks["T1u"]["Bbody_B"],
        blocks["T1u"]["Bel"],
    )
    T2u_evals, T2u_block = _solve_irrep_block(
        eps,
        a0,
        b0,
        blocks["T2u"]["M"],
        blocks["T2u"]["Bbody_A"],
        blocks["T2u"]["Bbody_B"],
        blocks["T2u"]["Bel"],
    )
    # A2u: 1×1
    bbody_A2u = a0 * blocks["A2u"]["Bbody_A"] + b0 * blocks["A2u"]["Bbody_B"]
    bel_A2u = blocks["A2u"]["Bel"]
    sigma_A2u = (eps * bbody_A2u - bel_A2u) / (
        blocks["A2u"]["M"] + bel_A2u - eps * bbody_A2u
    )
    # Eu: 1×1
    bbody_Eu = a0 * blocks["Eu"]["Bbody_A"] + b0 * blocks["Eu"]["Bbody_B"]
    bel_Eu = blocks["Eu"]["Bel"]
    sigma_Eu = (eps * bbody_Eu - bel_Eu) / (blocks["Eu"]["M"] + bel_Eu - eps * bbody_Eu)

    # ── Gerade sector (enlarged blocks with smooth correction) ──
    # A1g: 3×3
    A1g_evals, A1g_block = _solve_irrep_block(
        eps,
        a0,
        b0,
        blocks["A1g"]["M"],
        blocks["A1g"]["Bbody_A"],
        blocks["A1g"]["Bbody_B"],
        blocks["A1g"]["Bel"],
        Bbody_smooth=smooth_blocks["A1g"],
    )
    sigma_A1g = complex(A1g_block[0, 0])  # strain entry

    # A2g: 1×1 (with smooth correction)
    bbody_A2g = (
        a0 * blocks["A2g"]["Bbody_A"]
        + b0 * blocks["A2g"]["Bbody_B"]
        + complex(smooth_blocks["A2g"][0, 0])
    )
    bel_A2g = blocks["A2g"]["Bel"]
    denom_A2g = blocks["A2g"]["M"] + bel_A2g - eps * bbody_A2g
    sigma_A2g = (
        (eps * bbody_A2g - bel_A2g) / denom_A2g
        if abs(denom_A2g) > 1e-30
        else 0.0 + 0.0j
    )

    # Eg: 4×4
    Eg_evals, Eg_block = _solve_irrep_block(
        eps,
        a0,
        b0,
        blocks["Eg"]["M"],
        blocks["Eg"]["Bbody_A"],
        blocks["Eg"]["Bbody_B"],
        blocks["Eg"]["Bel"],
        Bbody_smooth=smooth_blocks["Eg"],
    )
    sigma_Eg = complex(Eg_block[0, 0])  # strain entry

    # T1g: 3×3
    T1g_evals, T1g_block = _solve_irrep_block(
        eps,
        a0,
        b0,
        blocks["T1g"]["M"],
        blocks["T1g"]["Bbody_A"],
        blocks["T1g"]["Bbody_B"],
        blocks["T1g"]["Bel"],
        Bbody_smooth=smooth_blocks["T1g"],
    )

    # T2g: 5×5
    T2g_evals, T2g_block = _solve_irrep_block(
        eps,
        a0,
        b0,
        blocks["T2g"]["M"],
        blocks["T2g"]["Bbody_A"],
        blocks["T2g"]["Bbody_B"],
        blocks["T2g"]["Bel"],
        Bbody_smooth=smooth_blocks["T2g"],
    )
    sigma_T2g = complex(T2g_block[0, 0])  # strain entry

    # ── Extract T1c, T2c, T3c from per-irrep strain eigenvalues ──
    # Same relations as T₂₇:
    #   σ_A1g = 3T1c + 2T2c + T3c
    #   σ_Eg  = 2T2c + T3c
    #   σ_T2g = 2T2c
    T2c = sigma_T2g / 2.0
    T3c = sigma_Eg - sigma_T2g
    T1c = (sigma_A1g - sigma_Eg) / 3.0

    # ── Effective stiffness contrasts ──
    # These follow the same self-consistent amplification as T₂₇
    # but the σ values now include (ka)⁴ corrections from cubic modes.
    amp_theta = 1.0 / (1.0 - sigma_A1g)
    amp_e_off = 1.0 / (1.0 - sigma_T2g)
    amp_e_diag = 1.0 / (1.0 - sigma_Eg)

    Dlam_star = (
        Dlam + 2.0 / 3.0 * Dmu_val
    ) * amp_theta - 2.0 / 3.0 * Dmu_val * amp_e_diag
    Dmu_off_star = Dmu_val * amp_e_off
    Dmu_diag_star = Dmu_val * amp_e_diag

    return GalerkinTMatrixResult57(
        # Ungerade
        T1u_eigenvalues=T1u_evals,
        T2u_eigenvalues=T2u_evals,
        sigma_A2u=sigma_A2u,
        sigma_Eu=sigma_Eu,
        T1u_block=T1u_block,
        T2u_block=T2u_block,
        # Gerade (enlarged)
        A1g_block=A1g_block,
        A1g_eigenvalues=A1g_evals,
        sigma_A2g=sigma_A2g,
        Eg_block=Eg_block,
        Eg_eigenvalues=Eg_evals,
        T1g_block=T1g_block,
        T1g_eigenvalues=T1g_evals,
        T2g_block=T2g_block,
        T2g_eigenvalues=T2g_evals,
        # Strain-sector scalars
        sigma_A1g=sigma_A1g,
        sigma_Eg=sigma_Eg,
        sigma_T2g=sigma_T2g,
        # Physical
        T1c=T1c,
        T2c=T2c,
        T3c=T3c,
        Dlambda_star=Dlam_star,
        Dmu_star_diag=Dmu_diag_star,
        Dmu_star_off=Dmu_off_star,
    )
