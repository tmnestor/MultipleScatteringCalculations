"""Analytic ka→0 Mie reference for a spherical inclusion (finite contrast).

Closed-form expressions transcribed from
`Mathematica/MieAsymptotic.wl` (validated against textbook Eshelby in
`Mathematica/MieAsymptoticVerify.wl`).

All formulas are exact in (Δλ, Δμ, Δρ) and leading-order in ka. They
serve as the gold-standard reference for the near-field comparison
between exact Mie, T₀ point-scatterer, and CPA mean-field models.

Conventions match `Mathematica/MieAsymptotic.wl`:

  * Non-dimensionalisation: a = μ₀ = ρ₀ = 1, λ₀ = lam0 = λ_bg/μ_bg.
  * Small parameter: w = ω·a/β = ka_S (the S-wave ka).
  * Outside P-wavenumber: kP·a = w / √(λ₀+2).
  * Contrasts: dlam = Δλ/μ₀, dmu = Δμ/μ₀, drho = Δρ/ρ₀.

Three key identities (cross-checked symbolically — see
MieAsymptoticVerify.wl):

  * E₀ = (λ₀+2) / (λ₀+2 + ΔK)              — bulk Eshelby
  * E₁ = 1                                  — no Eshelby for translation
  * E₂ = 1 / (1 + β_E·Δμ)                   — shear Eshelby
        β_E = 2(8+3λ₀) / (15(λ₀+2)) ≡ 2(4-5ν)/(15(1-ν))
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

# =====================================================================
# Material parameter dataclass (non-dimensionalised)
# =====================================================================


@dataclass(frozen=True)
class NondimContrast:
    """Non-dimensionalised contrasts (everything in units of μ₀, ρ₀, a)."""

    lam0: float  # background λ/μ ratio
    dlam: float  # Δλ / μ₀
    dmu: float  # Δμ / μ₀
    drho: float  # Δρ / ρ₀

    @classmethod
    def from_physical(
        cls,
        alpha: float,
        beta: float,
        rho: float,
        Dlambda: float,
        Dmu: float,
        Drho: float,
    ) -> "NondimContrast":
        """Build from physical (SI) values. mu0 = rho*beta², lam0_phys = rho*alpha² - 2 mu0."""
        mu0 = rho * beta**2
        lam0_phys = rho * alpha**2 - 2.0 * mu0
        return cls(
            lam0=lam0_phys / mu0,
            dlam=Dlambda / mu0,
            dmu=Dmu / mu0,
            drho=Drho / rho,
        )


# =====================================================================
# Eshelby concentration factors — exact, finite-contrast
# =====================================================================


def beta_E(lam0: float) -> float:
    """Shear Eshelby coefficient β_E = 2(8+3λ₀) / (15(λ₀+2))."""
    return 2.0 * (8.0 + 3.0 * lam0) / (15.0 * (lam0 + 2.0))


def alpha_E(lam0: float) -> float:
    """Bulk Eshelby coefficient α_E = 3K₀/(3K₀+4μ₀) (in nondim units)."""
    K0 = lam0 + 2.0 / 3.0
    return 3.0 * K0 / (3.0 * K0 + 4.0)


def E_0(c: NondimContrast) -> float:
    """E₀ = (λ₀+2) / (λ₀+2 + ΔK), ΔK = Δλ + 2 Δμ/3 (bulk Eshelby)."""
    DK = c.dlam + 2.0 * c.dmu / 3.0
    return (c.lam0 + 2.0) / (c.lam0 + 2.0 + DK)


def E_1(c: NondimContrast) -> float:
    """E₁ ≡ 1 (no Eshelby correction for the translation mode)."""
    return 1.0


def E_2(c: NondimContrast) -> float:
    """E₂ = 1 / (1 + β_E·Δμ) (shear Eshelby concentration factor)."""
    return 1.0 / (1.0 + beta_E(c.lam0) * c.dmu)


# =====================================================================
# Scattered partial-wave amplitudes (leading-order in w = ka_S)
# =====================================================================
# Transcribed verbatim from Section 5 of MieAsymptotic.wl:
#
#   a_0 = w² · (-3 dlam - 2 dmu) / [3·(λ₀+2)·(6 + 3 dlam + 2 dmu + 3 λ₀)]
#   a_1 = w² · (i/3) · drho / (λ₀+2)
#   a_2 = w² · 20 dmu / [3·(λ₀+2)·(15(λ₀+2) + 2 dmu (8+3 λ₀))]
#   b_1 = w² · (i/3) · drho
#   b_2 = w² · 10 dmu √(λ₀+2) / [3·(15(λ₀+2) + 2 dmu (8+3 λ₀))]


def a_0_analytic(c: NondimContrast, w: float) -> complex:
    """Leading scattered-P amplitude, monopole."""
    num = -(3.0 * c.dlam + 2.0 * c.dmu)
    den = 3.0 * (c.lam0 + 2.0) * (6.0 + 3.0 * c.dlam + 2.0 * c.dmu + 3.0 * c.lam0)
    return complex(w**2 * num / den)


def a_1_analytic(c: NondimContrast, w: float) -> complex:
    """Leading scattered-P amplitude, dipole."""
    return 1j * w**2 * c.drho / (3.0 * (c.lam0 + 2.0))


def a_2_analytic(c: NondimContrast, w: float) -> complex:
    """Leading scattered-P amplitude, quadrupole (shear-driven)."""
    den = (
        3.0
        * (c.lam0 + 2.0)
        * (15.0 * (c.lam0 + 2.0) + 2.0 * c.dmu * (8.0 + 3.0 * c.lam0))
    )
    return complex(w**2 * 20.0 * c.dmu / den)


def b_1_analytic(c: NondimContrast, w: float) -> complex:
    """Leading scattered-S amplitude, dipole."""
    return 1j * w**2 * c.drho / 3.0


def b_2_analytic(c: NondimContrast, w: float) -> complex:
    """Leading scattered-S amplitude, quadrupole."""
    den = 3.0 * (15.0 * (c.lam0 + 2.0) + 2.0 * c.dmu * (8.0 + 3.0 * c.lam0))
    return complex(w**2 * 10.0 * c.dmu * sqrt(c.lam0 + 2.0) / den)


# =====================================================================
# Interior near-field (leading-order)
# =====================================================================
# From Section 7 of MieAsymptotic.wl, validated to reproduce static
# Eshelby in MieAsymptoticVerify.wl (V4, V5).
#
# Interior displacement decomposes as
#   u_r^{int}(r, θ)     = Σ_n  U_r^{(n)}(r, w; c)  · P_n(cos θ)
#   u_θ^{int}(r, θ)     = Σ_n  U_θ^{(n)}(r, w; c)  · ∂_θ P_n(cos θ)
#
# Leading-order coefficients (in ka):
#   U_r^{(0)} = i √(λ₀+2) · r · w / (6 + 3 Δλ + 2 Δμ + 3 λ₀)
#   U_r^{(2)} = 10 i √(λ₀+2) · r · w / (30 + 16 Δμ + 15 λ₀ + 6 Δμ λ₀)
#   U_θ^{(2)} = 5  i √(λ₀+2) · r · w / (30 + 16 Δμ + 15 λ₀ + 6 Δμ λ₀)


def U_r_n0_leading(c: NondimContrast, r: float, w: float) -> complex:
    """Interior u_r leading coefficient, monopole."""
    den = 6.0 + 3.0 * c.dlam + 2.0 * c.dmu + 3.0 * c.lam0
    return 1j * sqrt(c.lam0 + 2.0) * r * w / den


def U_r_n2_leading(c: NondimContrast, r: float, w: float) -> complex:
    """Interior u_r leading coefficient, quadrupole shear mode.

    The denominator factor is 15(λ₀+2)·(1 + β_E·Δμ) — the static Eshelby
    shear concentration falls out of the Mie ka→0 limit at finite Δμ.
    """
    den = 30.0 + 16.0 * c.dmu + 15.0 * c.lam0 + 6.0 * c.dmu * c.lam0
    return 10.0j * sqrt(c.lam0 + 2.0) * r * w / den


def U_theta_n2_leading(c: NondimContrast, r: float, w: float) -> complex:
    """Interior u_θ leading coefficient, quadrupole shear mode."""
    den = 30.0 + 16.0 * c.dmu + 15.0 * c.lam0 + 6.0 * c.dmu * c.lam0
    return 5.0j * sqrt(c.lam0 + 2.0) * r * w / den


# =====================================================================
# Convenience: extract effective Δμ from a₂ and verify the Eshelby form
# =====================================================================


def Dmu_star_from_a2(a2: complex, c: NondimContrast, w: float) -> complex:
    """Extract effective shear contrast Δμ* from a₂ via Δμ* = 9(λ₀+2)²·a₂/(4 w²).

    Derivation: with C_P = a³/(3 ρ α²) = 1/(3(λ₀+2)) in nondim units and
    k_P² = w²/(λ₀+2), the textbook formula Δμ* = 3 a₂/(4 C_P k_P²)
    reduces to Δμ* = 9(λ₀+2)²·a₂/(4 w²).

    For the analytic a₂, this should equal Δμ / (1 + β_E·Δμ).
    """
    return 9.0 * (c.lam0 + 2.0) ** 2 * a2 / (4.0 * w**2)
