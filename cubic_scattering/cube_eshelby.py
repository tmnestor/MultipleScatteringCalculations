"""Cube Eshelby concentration factors for elastic cube scattering.

Computes Eshelby concentration factors for cubic cells by comparing
full T-matrix effective contrasts at finite contrast against the
Born (linearized) limit. The cube has 4 independent channels:

    amp_theta  : volumetric strain amplification
    amp_u      : displacement/density amplification
    amp_e_off  : off-diagonal shear amplification
    amp_e_diag : diagonal deviatoric amplification

The concentration ratio E = full_contrast / Born_contrast quantifies
how self-consistent renormalization modifies each channel beyond the
Born approximation.

For a sphere, these reduce to the 3 classical Eshelby factors:
    E_0 (monopole/bulk) ~ amp_theta
    E_1 (dipole/density) ~ amp_u
    E_2 (quadrupole/shear) ~ amp_e_off

The cube adds a 4th channel (amp_e_diag) that captures cubic anisotropy
in the shear response, absent for a sphere.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .effective_contrasts import (
    CubeTMatrixResult,
    MaterialContrast,
    ReferenceMedium,
    compute_cube_tmatrix,
)
from .resonance_tmatrix import (
    ResonanceTmatrixResult,
    compute_resonance_tmatrix,
)


@dataclass
class CubeEshelbyResult:
    """Result from cube Eshelby concentration factor computation.

    Attributes:
        amp_u: Displacement amplification factor.
        amp_theta: Volumetric strain amplification factor.
        amp_e_off: Off-diagonal shear amplification factor.
        amp_e_diag: Diagonal deviatoric amplification factor.
        Drho_star: Effective density contrast.
        Dlambda_star: Effective lambda contrast.
        Dmu_star_off: Effective off-diagonal shear modulus contrast.
        Dmu_star_diag: Effective diagonal shear modulus contrast.
        Drho_star_born: Born-limit density contrast.
        Dlambda_star_born: Born-limit lambda contrast.
        Dmu_star_off_born: Born-limit off-diagonal shear contrast.
        Dmu_star_diag_born: Born-limit diagonal shear contrast.
        E_u: Concentration ratio for density channel.
        E_theta: Concentration ratio for volumetric strain channel.
        E_e_off: Concentration ratio for off-diagonal shear channel.
        E_e_diag: Concentration ratio for diagonal deviatoric channel.
        T1c: T-matrix coupling coefficient T1.
        T2c: T-matrix coupling coefficient T2.
        T3c: T-matrix coupling coefficient T3.
        Gamma0: Green's tensor volume integral.
        T_comp_9x9: Full 9x9 composite T-matrix (None for Rayleigh-only).
        cubic_anisotropy: Dmu_star_diag - Dmu_star_off.
        ka: Dimensionless frequency used.
        ref: Background medium.
        contrast: Material contrast.
    """

    # Amplification factors
    amp_u: complex
    amp_theta: complex
    amp_e_off: complex
    amp_e_diag: complex

    # Effective contrasts (full)
    Drho_star: complex
    Dlambda_star: complex
    Dmu_star_off: complex
    Dmu_star_diag: complex

    # Born-limit effective contrasts
    Drho_star_born: complex
    Dlambda_star_born: complex
    Dmu_star_off_born: complex
    Dmu_star_diag_born: complex

    # Concentration ratios E = full / Born
    E_u: complex
    E_theta: complex
    E_e_off: complex
    E_e_diag: complex

    # T-matrix internals
    T1c: complex
    T2c: complex
    T3c: complex
    Gamma0: complex
    T_comp_9x9: NDArray[np.complexfloating] | None

    # Cubic anisotropy
    cubic_anisotropy: complex

    # Metadata
    ka: float
    ref: ReferenceMedium
    contrast: MaterialContrast


@dataclass
class CubeConvergenceResult:
    """Result from cube Eshelby convergence study.

    Attributes:
        ka_values: Array of ka values tested.
        amp_u: Displacement amplification at each ka.
        amp_theta: Volumetric strain amplification at each ka.
        amp_e_off: Off-diagonal shear amplification at each ka.
        amp_e_diag: Diagonal deviatoric amplification at each ka.
        E_u: Concentration ratio for density at each ka.
        E_theta: Concentration ratio for volumetric strain at each ka.
        E_e_off: Concentration ratio for off-diagonal shear at each ka.
        E_e_diag: Concentration ratio for diagonal deviatoric at each ka.
        E_static: Static Eshelby factors (CubeEshelbyResult at low ka).
        used_resonance: Boolean array, True where resonance solver was used.
    """

    ka_values: NDArray[np.floating]
    amp_u: NDArray[np.complexfloating]
    amp_theta: NDArray[np.complexfloating]
    amp_e_off: NDArray[np.complexfloating]
    amp_e_diag: NDArray[np.complexfloating]
    E_u: NDArray[np.complexfloating]
    E_theta: NDArray[np.complexfloating]
    E_e_off: NDArray[np.complexfloating]
    E_e_diag: NDArray[np.complexfloating]
    E_static: CubeEshelbyResult
    used_resonance: NDArray[np.bool_]


def _safe_ratio(numerator: complex, denominator: complex) -> complex:
    """Compute ratio with protection against near-zero denominators."""
    if abs(denominator) < 1e-30:
        return complex(1.0)
    return numerator / denominator


def compute_cube_born_tmatrix(
    omega: float,
    a: float,
    ref: ReferenceMedium,
    contrast: MaterialContrast,
    epsilon: float = 1e-4,
) -> CubeTMatrixResult:
    """Compute Born (linearized) cube T-matrix via finite-difference scaling.

    Runs compute_cube_tmatrix at epsilon-scaled contrast and divides
    effective contrasts by epsilon. This gives the linear response
    for each channel.

    Args:
        omega: Angular frequency (rad/s).
        a: Cube half-width (m).
        ref: Background medium.
        contrast: Material contrasts (full, unscaled).
        epsilon: Perturbation scale for Born limit.

    Returns:
        CubeTMatrixResult with Born-limit effective contrasts.
    """
    weak = MaterialContrast(
        Dlambda=contrast.Dlambda * epsilon,
        Dmu=contrast.Dmu * epsilon,
        Drho=contrast.Drho * epsilon,
    )
    result = compute_cube_tmatrix(omega, a, ref, weak)

    return CubeTMatrixResult(
        Gamma0=result.Gamma0,
        Ac=result.Ac,
        Bc=result.Bc,
        Cc=result.Cc,
        T1c=result.T1c / epsilon,
        T2c=result.T2c / epsilon,
        T3c=result.T3c / epsilon,
        amp_u=result.amp_u,
        amp_theta=result.amp_theta,
        amp_e_off=result.amp_e_off,
        amp_e_diag=result.amp_e_diag,
        Drho_star=result.Drho_star / epsilon,
        Dlambda_star=result.Dlambda_star / epsilon,
        Dmu_star_off=result.Dmu_star_off / epsilon,
        Dmu_star_diag=result.Dmu_star_diag / epsilon,
    )


def compute_cube_eshelby_factors(
    ref: ReferenceMedium,
    contrast: MaterialContrast,
    a: float = 1.0,
    ka: float = 0.01,
    ka_resonance_threshold: float = 0.3,
) -> CubeEshelbyResult:
    """Compute cube Eshelby concentration factors at a given ka.

    At low ka (< ka_resonance_threshold): uses Rayleigh T-matrix.
    At high ka: uses resonance T-matrix with Foldy-Lax subdivision.

    Args:
        ref: Background medium.
        contrast: Material contrasts.
        a: Cube half-width (m).
        ka: Target dimensionless frequency (k_S * a).
        ka_resonance_threshold: Switch to resonance solver above this.

    Returns:
        CubeEshelbyResult with all 4 concentration factors.
    """
    omega = ka * ref.beta / a

    # Born T-matrix for comparison
    born = compute_cube_born_tmatrix(omega, a, ref, contrast)

    # Full T-matrix
    T_comp_9x9 = None
    if ka < ka_resonance_threshold:
        full = compute_cube_tmatrix(omega, a, ref, contrast)
    else:
        res = compute_resonance_tmatrix(omega, a, ref, contrast)
        full = res.rayleigh_result
        # Extract effective contrasts from resonance composite T-matrix
        full = _extract_rayleigh_from_resonance(res)
        T_comp_9x9 = res.T_comp_9x9

    # Concentration ratios
    E_u = _safe_ratio(full.Drho_star, born.Drho_star)
    E_theta = _safe_ratio(full.Dlambda_star, born.Dlambda_star)
    E_e_off = _safe_ratio(full.Dmu_star_off, born.Dmu_star_off)
    E_e_diag = _safe_ratio(full.Dmu_star_diag, born.Dmu_star_diag)

    return CubeEshelbyResult(
        amp_u=full.amp_u,
        amp_theta=full.amp_theta,
        amp_e_off=full.amp_e_off,
        amp_e_diag=full.amp_e_diag,
        Drho_star=full.Drho_star,
        Dlambda_star=full.Dlambda_star,
        Dmu_star_off=full.Dmu_star_off,
        Dmu_star_diag=full.Dmu_star_diag,
        Drho_star_born=born.Drho_star,
        Dlambda_star_born=born.Dlambda_star,
        Dmu_star_off_born=born.Dmu_star_off,
        Dmu_star_diag_born=born.Dmu_star_diag,
        E_u=E_u,
        E_theta=E_theta,
        E_e_off=E_e_off,
        E_e_diag=E_e_diag,
        T1c=full.T1c,
        T2c=full.T2c,
        T3c=full.T3c,
        Gamma0=full.Gamma0,
        T_comp_9x9=T_comp_9x9,
        cubic_anisotropy=full.Dmu_star_diag - full.Dmu_star_off,
        ka=ka,
        ref=ref,
        contrast=contrast,
    )


def _extract_rayleigh_from_resonance(res: ResonanceTmatrixResult) -> CubeTMatrixResult:
    """Extract effective contrasts from a resonance result's composite T-matrix.

    At resonance, the Rayleigh-level amplification factors from the
    sub-cell computation don't capture the full physics. We use the
    9x9 composite T-matrix to extract effective contrasts.

    For the density channel, the force monopole block gives:
        T_comp[:3,:3] = omega^2 * Drho_star_eff * V * I_3

    For the stiffness channels, the stress dipole block gives
    the effective Voigt stiffness contrast.
    """
    omega = res.omega
    a = res.a
    V = (2.0 * a) ** 3
    T_comp = res.T_comp_9x9

    # Density: from force monopole (average diagonal of displacement block)
    Drho_star_eff = complex(np.mean(np.diag(T_comp[:3, :3])) / (omega**2 * V))

    # Stiffness: from stress dipole block (6x6)
    # The Voigt stiffness block has structure:
    #   [[D*, 0], [0, S*]] where D = diag(lam+2mu_diag) + off-diag(lam)
    #   and S = diag(2*mu_off)
    stress_block = T_comp[3:, 3:] / V
    Dmu_star_off_eff = complex(stress_block[3, 3] / 2.0)
    Dlambda_plus_2mu_diag = complex(stress_block[0, 0])
    Dlambda_eff = complex(stress_block[0, 1])
    Dmu_star_diag_eff = complex((Dlambda_plus_2mu_diag - Dlambda_eff) / 2.0)
    Dlambda_star_eff = Dlambda_eff

    # Reconstruct amplification factors
    contrast = res.contrast
    amp_u = _safe_ratio(Drho_star_eff, complex(contrast.Drho))
    amp_theta_num = Dlambda_star_eff + 2.0 / 3.0 * Dmu_star_diag_eff
    amp_theta_denom = contrast.Dlambda + 2.0 / 3.0 * contrast.Dmu
    amp_theta = _safe_ratio(amp_theta_num, complex(amp_theta_denom))
    amp_e_off = _safe_ratio(Dmu_star_off_eff, complex(contrast.Dmu))
    amp_e_diag = _safe_ratio(Dmu_star_diag_eff, complex(contrast.Dmu))

    rayleigh = res.rayleigh_result
    return CubeTMatrixResult(
        Gamma0=rayleigh.Gamma0,
        Ac=rayleigh.Ac,
        Bc=rayleigh.Bc,
        Cc=rayleigh.Cc,
        T1c=rayleigh.T1c,
        T2c=rayleigh.T2c,
        T3c=rayleigh.T3c,
        amp_u=amp_u,
        amp_theta=amp_theta,
        amp_e_off=amp_e_off,
        amp_e_diag=amp_e_diag,
        Drho_star=Drho_star_eff,
        Dlambda_star=Dlambda_star_eff,
        Dmu_star_off=Dmu_star_off_eff,
        Dmu_star_diag=Dmu_star_diag_eff,
    )


def compute_cube_eshelby(
    ref: ReferenceMedium,
    contrast: MaterialContrast,
    a: float = 1.0,
    ka_static: float = 0.01,
    ka_dynamic: float = 0.3,
) -> CubeEshelbyResult:
    """Compute cube Eshelby concentration factors (main entry point).

    Combines static (ka -> 0) and dynamic (finite ka) computation.
    Returns the result at ka_dynamic, which includes both static
    and dynamic amplification factors.

    Args:
        ref: Background medium.
        contrast: Material contrasts.
        a: Cube half-width (m).
        ka_static: Target ka for static limit.
        ka_dynamic: Target ka for dynamic computation.

    Returns:
        CubeEshelbyResult at the dynamic ka value.
    """
    return compute_cube_eshelby_factors(ref, contrast, a=a, ka=ka_dynamic)


def cube_convergence_study(
    ref: ReferenceMedium,
    contrast: MaterialContrast,
    ka_values: NDArray[np.floating] | None = None,
    a: float = 1.0,
    ka_resonance_threshold: float = 0.3,
) -> CubeConvergenceResult:
    """Sweep ka values tracking all 4 amplification factors and concentration ratios.

    Auto-selects Rayleigh vs resonance solver per ka value.

    Args:
        ref: Background medium.
        contrast: Material contrasts.
        ka_values: Array of ka_S values. Default: logspace from 0.05 to 1.5.
        a: Cube half-width (m).
        ka_resonance_threshold: Switch to resonance solver above this.

    Returns:
        CubeConvergenceResult with per-channel arrays.
    """
    if ka_values is None:
        ka_values = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5])

    ka_values = np.asarray(ka_values, dtype=float)
    n_ka = len(ka_values)

    amp_u = np.zeros(n_ka, dtype=complex)
    amp_theta = np.zeros(n_ka, dtype=complex)
    amp_e_off = np.zeros(n_ka, dtype=complex)
    amp_e_diag = np.zeros(n_ka, dtype=complex)
    E_u = np.zeros(n_ka, dtype=complex)
    E_theta = np.zeros(n_ka, dtype=complex)
    E_e_off = np.zeros(n_ka, dtype=complex)
    E_e_diag = np.zeros(n_ka, dtype=complex)
    used_resonance = np.zeros(n_ka, dtype=bool)

    # Static reference
    E_static = compute_cube_eshelby_factors(ref, contrast, a=a, ka=0.01)

    for i, ka in enumerate(ka_values):
        result = compute_cube_eshelby_factors(
            ref,
            contrast,
            a=a,
            ka=ka,
            ka_resonance_threshold=ka_resonance_threshold,
        )
        amp_u[i] = result.amp_u
        amp_theta[i] = result.amp_theta
        amp_e_off[i] = result.amp_e_off
        amp_e_diag[i] = result.amp_e_diag
        E_u[i] = result.E_u
        E_theta[i] = result.E_theta
        E_e_off[i] = result.E_e_off
        E_e_diag[i] = result.E_e_diag
        used_resonance[i] = ka >= ka_resonance_threshold

    return CubeConvergenceResult(
        ka_values=ka_values,
        amp_u=amp_u,
        amp_theta=amp_theta,
        amp_e_off=amp_e_off,
        amp_e_diag=amp_e_diag,
        E_u=E_u,
        E_theta=E_theta,
        E_e_off=E_e_off,
        E_e_diag=E_e_diag,
        E_static=E_static,
        used_resonance=used_resonance,
    )
