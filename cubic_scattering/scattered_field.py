"""
scattered_field.py
Far-field scattering amplitudes from the T27 cube T-matrix.

Physics:
  The Lippmann-Schwinger source for scattered waves is:
    s_j(r) = ω²Δρ u_total_j(r) + ∂_k(Δc_jklm ε_total_lm(r))

  In the Rayleigh limit (ka << 1), the far-field integral factorizes into:
    Force monopole: F_i = ω²Δρ ∫ u_total_i d³r  (→ dipole l=1 pattern)
    Stress dipole: Δσ = self-consistent effective stress (→ monopole l=0 + quadrupole l=2)

  The TOTAL field volume integral ∫u_total d³r = c_total[0:3] = c_inc[0:3] + c_sc[0:3],
  where c = T27 @ c_inc gives the scattered overlap.

  The stress dipole uses the Voigt T-matrix (T1c,T2c,T3c) applied to the
  incident strain, which already includes self-consistent amplification.

Sign convention:
  f_P = -Q_P/(4πρα²) where Q_P = r̂·F - ikP V(r̂·Δσ·r̂)
  This matches the Mie far-field convention with the (-1)^n correction.
"""

from typing import TYPE_CHECKING

import numpy as np

from .effective_contrasts import (
    GalerkinTMatrixResult,
    MaterialContrast,
    ReferenceMedium,
)
from .voigt_tmatrix import effective_stiffness_voigt

if TYPE_CHECKING:
    from .resonance_tmatrix import ResonanceTmatrixResult

# ================================================================
# Far-field scattering amplitudes
# ================================================================


def _incident_voigt_strain(k_vec: np.ndarray, pol: np.ndarray) -> np.ndarray:
    """Compute the Voigt strain vector for a plane wave u = pol exp(ik·r).

    The strain is ε_ij = (1/2)(ik_i pol_j + ik_j pol_i).
    Voigt ordering: (ε11, ε22, ε33, 2ε23, 2ε13, 2ε12).
    """
    eps_V = np.zeros(6, dtype=complex)
    # Diagonal: ε_ii = ik_i pol_i
    for i in range(3):
        eps_V[i] = 1j * k_vec[i] * pol[i]
    # Off-diagonal (with factor 2 for Voigt):
    # 2ε23 = i(k2 pol3 + k3 pol2)
    eps_V[3] = 1j * (k_vec[1] * pol[2] + k_vec[2] * pol[1])
    # 2ε13 = i(k1 pol3 + k3 pol1)
    eps_V[4] = 1j * (k_vec[0] * pol[2] + k_vec[2] * pol[0])
    # 2ε12 = i(k1 pol2 + k2 pol1)
    eps_V[5] = 1j * (k_vec[0] * pol[1] + k_vec[1] * pol[0])
    return eps_V


def _voigt_to_tensor(sigma_V: np.ndarray) -> np.ndarray:
    """Convert Voigt stress vector (6,) to symmetric tensor (3,3)."""
    S = np.zeros((3, 3), dtype=complex)
    S[0, 0] = sigma_V[0]
    S[1, 1] = sigma_V[1]
    S[2, 2] = sigma_V[2]
    S[1, 2] = S[2, 1] = sigma_V[3]
    S[0, 2] = S[2, 0] = sigma_V[4]
    S[0, 1] = S[1, 0] = sigma_V[5]
    return S


def cube_far_field(
    c_inc: np.ndarray,
    c_sc: np.ndarray,
    theta: np.ndarray | float,
    ref: ReferenceMedium,
    galerkin: GalerkinTMatrixResult,
    contrast: MaterialContrast,
    omega: float,
    a: float,
    k_vec: np.ndarray | None = None,
    pol: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute far-field scattering amplitudes from T27 results.

    Uses the correct Lippmann-Schwinger far-field formula:
      f_P = -[r̂·F - ikP V(r̂·Δσ·r̂)] / (4πρα²)

    where:
      F = ω²Δρ (c_inc[0:3] + c_sc[0:3])  — force monopole from total displacement
      Δσ = T_Voigt @ ε_inc  — self-consistent stress dipole

    Parameters
    ----------
    c_inc : Incident overlap integrals (27,).
    c_sc : Scattered overlap integrals (27,) from T27 @ c_inc.
    theta : Scattering angle(s) from forward direction.
    ref : Background medium.
    galerkin : Galerkin T-matrix result (for T1c, T2c, T3c).
    contrast : Material contrasts.
    omega : Angular frequency.
    a : Cube half-width.
    k_vec : Incident wave vector (default: kP ẑ for P-wave).
    pol : Incident polarization (default: ẑ for P-wave).

    Returns
    -------
    f_P, f_SV, f_SH : Scattering amplitudes vs angle.
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))

    kP = omega / ref.alpha
    kS = omega / ref.beta
    rho = ref.rho
    V = (2.0 * a) ** 3

    if k_vec is None:
        k_vec = np.array([0.0, 0.0, kP])
    if pol is None:
        pol = k_vec / np.linalg.norm(k_vec)

    k_hat = k_vec / np.linalg.norm(k_vec)

    # ── Force monopole from density contrast ──
    # F_i = ω²Δρ × ∫ u_total_i d³r = ω²Δρ × (c_inc[i] + c_sc[i])
    c_total = c_inc + c_sc
    F = omega**2 * contrast.Drho * c_total[:3]

    # ── Stress dipole from stiffness contrast ──
    # Δσ = Δc* @ ε_inc where Δc* is the PHYSICAL effective stiffness (Pa)
    Dc_star = effective_stiffness_voigt(
        galerkin.Dlambda_star, galerkin.Dmu_star_diag, galerkin.Dmu_star_off
    )
    eps_inc_V = _incident_voigt_strain(k_vec, pol)
    dsigma_V = Dc_star @ eps_inc_V  # 6-component Voigt stress perturbation (Pa)
    dsigma = _voigt_to_tensor(dsigma_V)  # 3×3 tensor

    # ── Scattering plane basis ──
    if abs(k_hat[0]) < 0.9:
        ref_vec = np.array([1.0, 0.0, 0.0])
    else:
        ref_vec = np.array([0.0, 1.0, 0.0])
    perp1 = ref_vec - np.dot(ref_vec, k_hat) * k_hat
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(k_hat, perp1)

    f_P = np.zeros_like(theta, dtype=complex)
    f_SV = np.zeros_like(theta, dtype=complex)
    f_SH = np.zeros_like(theta, dtype=complex)

    for idx, th in enumerate(theta):
        r_hat = np.sin(th) * perp1 + np.cos(th) * k_hat
        sv_hat = np.cos(th) * perp1 - np.sin(th) * k_hat
        sh_hat = perp2

        rF = np.dot(r_hat, F)
        rSr = r_hat @ (V * dsigma) @ r_hat
        Sr = (V * dsigma) @ r_hat

        # f_P = -[r̂·F - ikP(r̂·VΔσ·r̂)] / (4πρα²)
        Q_P = rF - 1j * kP * rSr
        f_P[idx] = -Q_P / (4.0 * np.pi * rho * ref.alpha**2)

        # f_S = -[(I-r̂r̂)·F - ikS(I-r̂r̂)·VΔσ·r̂] / (4πρβ²)
        F_perp = F - rF * r_hat
        S_perp = Sr - rSr * r_hat
        Q_S = F_perp - 1j * kS * S_perp
        u_S = -Q_S / (4.0 * np.pi * rho * ref.beta**2)

        f_SV[idx] = np.dot(sv_hat, u_S)
        f_SH[idx] = np.dot(sh_hat, u_S)

    return f_P, f_SV, f_SH


# ================================================================
# Resonance (multi-cell) far-field
# ================================================================


def resonance_far_field(
    res: "ResonanceTmatrixResult",
    theta: np.ndarray | float,
    ref: ReferenceMedium,
    contrast: MaterialContrast,
    omega: float,
    a: float,
    k_vec: np.ndarray | None = None,
    pol: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Far-field scattering amplitudes from the resonance (multi-cell) T-matrix.

    Sums per-sub-cell contributions using the far-field Green's tensor,
    including force monopole and stress dipole with inter-cell phase factors.
    Converges to `cube_far_field` (T27) in the Rayleigh limit as n_sub → ∞.

    The ``ResonanceTmatrixResult`` must have been computed with a ``wave_type``
    matching the incident wave (``'P'`` or ``'S'``).

    Parameters
    ----------
    res : Resonance T-matrix result (must include psi_exc, centres, T_loc_9x9).
    theta : Scattering angle(s) from forward direction.
    ref : Background medium.
    contrast : Material contrast (unused; stiffness is baked into T_loc_9x9).
    omega : Angular frequency (rad/s).
    a : Parent cube half-width (m).
    k_vec : Incident wave vector (default: kP ẑ for P-wave along axis 2).
    pol : Incident polarization (default: k̂ for P-wave).

    Returns
    -------
    f_P, f_SV, f_SH : Scattering amplitudes vs angle.
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))

    kP = omega / ref.alpha
    kS = omega / ref.beta
    rho = ref.rho

    if k_vec is None:
        k_vec = np.array([0.0, 0.0, kP])
    if pol is None:
        pol = k_vec / np.linalg.norm(k_vec)

    k_hat = k_vec / np.linalg.norm(k_vec)

    # 9-component incident vector: [displacement, Voigt strain]
    inc_vec = np.zeros(9, dtype=complex)
    inc_vec[:3] = pol
    inc_vec[3:] = _incident_voigt_strain(k_vec, pol)

    # Sub-cell data from result
    N = res.n_sub**3
    centres = res.centres
    psi_exc = res.psi_exc
    T_loc = res.T_loc_9x9

    # Precompute per-cell scattered sources: source_n = T_loc @ psi_exc_n @ inc_vec
    sources = np.zeros((N, 9), dtype=complex)
    for n in range(N):
        psi_n = psi_exc[9 * n : 9 * n + 9, :] @ inc_vec
        sources[n] = T_loc @ psi_n

    # Scattering plane basis (same as cube_far_field)
    if abs(k_hat[0]) < 0.9:
        ref_vec = np.array([1.0, 0.0, 0.0])
    else:
        ref_vec = np.array([0.0, 1.0, 0.0])
    perp1 = ref_vec - np.dot(ref_vec, k_hat) * k_hat
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(k_hat, perp1)

    # Precompute per-cell stress tensors
    forces = sources[:, :3]  # (N, 3)
    sigmas = np.array([_voigt_to_tensor(sources[n, 3:]) for n in range(N)])  # (N, 3, 3)

    f_P = np.zeros_like(theta, dtype=complex)
    f_SV = np.zeros_like(theta, dtype=complex)
    f_SH = np.zeros_like(theta, dtype=complex)

    for idx, th in enumerate(theta):
        r_hat = np.sin(th) * perp1 + np.cos(th) * k_hat
        sv_hat = np.cos(th) * perp1 - np.sin(th) * k_hat
        sh_hat = perp2

        # Outgoing phase: exp(-i k_out · r_n) for each cell
        r_dot_centres = centres @ r_hat  # (N,)
        phase_P = np.exp(-1j * kP * r_dot_centres)  # (N,)
        phase_S = np.exp(-1j * kS * r_dot_centres)  # (N,)

        # Sum over sub-cells with phase factors
        # P-wave: Q_P_n = r̂·F_n - ikP(r̂·σ_n·r̂)
        rF = forces @ r_hat  # (N,)
        rSr = np.einsum("i,nij,j->n", r_hat, sigmas, r_hat)  # (N,)
        Q_P_total = np.sum((rF - 1j * kP * rSr) * phase_P)
        f_P[idx] = -Q_P_total / (4.0 * np.pi * rho * ref.alpha**2)

        # S-wave: Q_S = F_perp - ikS(σ·r̂)_perp
        Sr = np.einsum("nij,j->ni", sigmas, r_hat)  # (N, 3)
        F_perp = forces - np.outer(rF, r_hat)  # (N, 3)
        S_perp = Sr - np.outer(rSr, r_hat)  # (N, 3)
        Q_S_per_cell = F_perp - 1j * kS * S_perp  # (N, 3)
        Q_S_total = np.sum(Q_S_per_cell * phase_S[:, None], axis=0)  # (3,)
        u_S = -Q_S_total / (4.0 * np.pi * rho * ref.beta**2)

        f_SV[idx] = np.dot(sv_hat, u_S)
        f_SH[idx] = np.dot(sh_hat, u_S)

    return f_P, f_SV, f_SH


# ================================================================
# Scattering cross-section
# ================================================================


def scattering_cross_section(
    c_inc: np.ndarray,
    c_sc: np.ndarray,
    ref: ReferenceMedium,
    galerkin: GalerkinTMatrixResult,
    contrast: MaterialContrast,
    omega: float,
    a: float,
    k_vec: np.ndarray | None = None,
    pol: np.ndarray | None = None,
    n_theta: int = 200,
) -> float:
    """Compute total scattering cross-section by angular integration.

    σ_sc = 2π ∫₀^π (kP/kI |f_P|² + kS/kI (|f_SV|² + |f_SH|²)) sin(θ) dθ
    """
    kP = omega / ref.alpha
    kS = omega / ref.beta

    if k_vec is None:
        k_vec = np.array([0.0, 0.0, kP])
    if pol is None:
        pol = k_vec / np.linalg.norm(k_vec)

    # Determine incident wavenumber
    k_hat = k_vec / np.linalg.norm(k_vec)
    dot = abs(np.dot(pol, k_hat))
    kI = kP if dot > 0.5 else kS

    theta = np.linspace(0, np.pi, n_theta)
    f_P, f_SV, f_SH = cube_far_field(
        c_inc, c_sc, theta, ref, galerkin, contrast, omega, a, k_vec, pol
    )

    integrand = (
        kP / kI * np.abs(f_P) ** 2 + kS / kI * (np.abs(f_SV) ** 2 + np.abs(f_SH) ** 2)
    ) * np.sin(theta)

    return float(2.0 * np.pi * np.trapezoid(integrand, theta))


# ================================================================
# Optical theorem check
# ================================================================


def optical_theorem_check(
    T27: np.ndarray,
    ref: ReferenceMedium,
    galerkin: GalerkinTMatrixResult,
    contrast: MaterialContrast,
    omega: float,
    a: float,
    k_vec: np.ndarray | None = None,
    pol: np.ndarray | None = None,
) -> tuple[float, float]:
    """Verify the optical theorem: σ_ext = (4π/k) Im[f(θ=0)].

    Returns (sigma_ext, sigma_sc).
    """
    from .incident_field import cube_overlap_integrals

    kP = omega / ref.alpha
    if k_vec is None:
        k_vec = np.array([0.0, 0.0, kP])
    if pol is None:
        pol = k_vec / np.linalg.norm(k_vec)

    kI = np.linalg.norm(k_vec)
    c_inc = cube_overlap_integrals(k_vec, pol, a)
    c_sc = T27 @ c_inc

    f_P_fwd, _, _ = cube_far_field(
        c_inc, c_sc, np.array([0.0]), ref, galerkin, contrast, omega, a, k_vec, pol
    )

    # Note: our far-field uses f_P = -Q_P/(4πρα²), so the optical theorem
    # σ_ext = (4π/k) Im[f(0)] picks up the opposite sign from the standard
    # convention. We flip to get σ_ext > 0 for passive scatterers.
    sigma_ext = -4.0 * np.pi / kI * np.imag(f_P_fwd[0])
    sigma_sc = scattering_cross_section(
        c_inc, c_sc, ref, galerkin, contrast, omega, a, k_vec, pol
    )

    return float(sigma_ext), float(sigma_sc)
