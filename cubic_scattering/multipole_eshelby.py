"""Multipole Eshelby concentration factors for elastic sphere scattering.

Computes and validates higher-order (n>=3) Eshelby concentration factors
by comparing full Mie coefficients at finite contrast against the
Born (linearized) limit. Quantifies far-field accuracy improvement from
including octupole and higher multipoles.

The concentration factor E_n relates the full scattering coefficient a_n
to the Born coefficient a_n^Born:
    a_n(full) = E_n * a_n^Born

For n=0,1,2 these match the classical Eshelby amplification factors:
    E_0 = K_0 / (K_0 + alpha_E * DK)     (monopole amplification)
    E_1 = 1.0                              (dipole — no static correction)
    E_2 = 1 / (1 + beta_E * Dmu/mu_0)     (quadrupole amplification)

where alpha_E = 3K_0/(3K_0+4mu_0) and beta_E = 6(K_0+2mu_0)/(5(3K_0+4mu_0))
are the Eshelby depolarization factors.

For n>=3, these are new results extracted numerically.
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .effective_contrasts import MaterialContrast, ReferenceMedium
from .sphere_scattering import MieResult, compute_elastic_mie, mie_far_field


@dataclass
class MultipoleEshelbyResult:
    """Result from multipole Eshelby concentration factor computation.

    Attributes:
        n_max: Maximum multipole order computed.
        E_n_static: Static Eshelby concentration factors, shape (n_max+1,).
        E_n_dynamic: Dynamic (finite ka) concentration factors, shape (n_max+1,).
        a_n_born: Born limit scattering coefficients, shape (n_max+1,).
        a_n_full: Full Mie scattering coefficients, shape (n_max+1,).
        ka_P: Dimensionless P-wave frequency for dynamic computation.
        ref: Background medium.
        contrast: Material contrast.
    """

    n_max: int
    E_n_static: NDArray[np.complexfloating]
    E_n_dynamic: NDArray[np.complexfloating]
    a_n_born: NDArray[np.complexfloating]
    a_n_full: NDArray[np.complexfloating]
    ka_P: float
    ref: ReferenceMedium
    contrast: MaterialContrast


def compute_born_coefficients(
    omega: float,
    radius: float,
    ref: ReferenceMedium,
    contrast: MaterialContrast,
    n_max: int | None = None,
    epsilon: float = 1e-4,
) -> NDArray[np.complexfloating]:
    """Compute Born (linearized) Mie coefficients via finite-difference scaling.

    Runs compute_elastic_mie at epsilon-scaled contrast and divides by epsilon.
    This gives the linear response coefficient for each multipole order.

    Args:
        omega: Angular frequency (rad/s).
        radius: Sphere radius (m).
        ref: Background medium.
        contrast: Material contrasts (full, unscaled).
        n_max: Maximum angular order. If None, auto-selected.
        epsilon: Perturbation scale for Born limit. Default 1e-4 provides
            good numerical stability for coefficients up to n=5.

    Returns:
        Born coefficients a_n^Born, shape (n_max+1,).
    """
    weak = MaterialContrast(
        Dlambda=contrast.Dlambda * epsilon,
        Dmu=contrast.Dmu * epsilon,
        Drho=contrast.Drho * epsilon,
    )
    mie_weak = compute_elastic_mie(omega, radius, ref, weak, n_max=n_max)
    return mie_weak.a_n / epsilon


def compute_static_eshelby_factors(
    ref: ReferenceMedium,
    contrast: MaterialContrast,
    n_max: int = 5,
    radius: float = 1.0,
    ka_static: float = 0.01,
) -> NDArray[np.complexfloating]:
    """Compute static Eshelby concentration factors E_n for a sphere.

    At low frequency (ka ~ 0.01), the ratio a_n(full) / a_n(Born)
    converges to the static Eshelby concentration factor with O(ka^2)
    dynamic corrections (~0.01% at ka=0.01).

    Args:
        ref: Background medium.
        contrast: Material contrasts.
        n_max: Maximum multipole order.
        radius: Sphere radius (m). Result is scale-independent.
        ka_static: Target ka for static limit computation. Must be large
            enough for higher-order coefficients to be above numerical noise.

    Returns:
        E_n_static, shape (n_max+1,). Real-valued in the static limit.
    """
    omega = ka_static * ref.beta / radius

    mie_full = compute_elastic_mie(omega, radius, ref, contrast, n_max=n_max)
    a_n_born = compute_born_coefficients(omega, radius, ref, contrast, n_max=n_max)

    E_n = np.ones(n_max + 1, dtype=complex)
    for n in range(n_max + 1):
        if abs(a_n_born[n]) > 1e-30:
            E_n[n] = mie_full.a_n[n] / a_n_born[n]

    return E_n


def compute_multipole_eshelby(
    ref: ReferenceMedium,
    contrast: MaterialContrast,
    n_max: int = 5,
    radius: float = 1.0,
    ka_dynamic: float = 0.3,
    ka_static: float = 0.01,
) -> MultipoleEshelbyResult:
    """Compute static and dynamic Eshelby concentration factors.

    Combines static (ka→0) and dynamic (finite ka) concentration factors
    for all multipole orders up to n_max.

    Args:
        ref: Background medium.
        contrast: Material contrasts.
        n_max: Maximum multipole order.
        radius: Sphere radius (m).
        ka_dynamic: Target ka for dynamic computation.
        ka_static: Target ka for static limit.

    Returns:
        MultipoleEshelbyResult with all factors and coefficients.
    """
    E_n_static = compute_static_eshelby_factors(
        ref, contrast, n_max=n_max, radius=radius, ka_static=ka_static
    )

    omega_dyn = ka_dynamic * ref.beta / radius
    mie_full = compute_elastic_mie(omega_dyn, radius, ref, contrast, n_max=n_max)
    a_n_born = compute_born_coefficients(omega_dyn, radius, ref, contrast, n_max=n_max)

    E_n_dynamic = np.ones(n_max + 1, dtype=complex)
    for n in range(n_max + 1):
        if abs(a_n_born[n]) > 1e-30:
            E_n_dynamic[n] = mie_full.a_n[n] / a_n_born[n]

    return MultipoleEshelbyResult(
        n_max=n_max,
        E_n_static=E_n_static,
        E_n_dynamic=E_n_dynamic,
        a_n_born=a_n_born,
        a_n_full=mie_full.a_n.copy(),
        ka_P=mie_full.ka_P,
        ref=ref,
        contrast=contrast,
    )


def far_field_truncation_error(
    mie_result: MieResult,
    n_trunc: int,
    n_theta: int = 181,
) -> float:
    """Compute far-field L2 relative error from truncating at order n_trunc.

    Compares the full Mie far-field against a version with a_n, b_n zeroed
    for n > n_trunc.

    Args:
        mie_result: Full Mie result (untruncated).
        n_trunc: Maximum order to retain.
        n_theta: Number of scattering angles for error evaluation.

    Returns:
        L2 relative error (scalar, real).
    """
    theta = np.linspace(0, np.pi, n_theta)

    f_P_full, f_SV_full, _ = mie_far_field(mie_result, theta, incident_type="P")

    # Create truncated copy
    trunc = MieResult(
        a_n=mie_result.a_n.copy(),
        b_n=mie_result.b_n.copy(),
        c_n=mie_result.c_n.copy(),
        a_n_sv=mie_result.a_n_sv.copy(),
        b_n_sv=mie_result.b_n_sv.copy(),
        n_max=mie_result.n_max,
        omega=mie_result.omega,
        radius=mie_result.radius,
        ref=mie_result.ref,
        contrast=mie_result.contrast,
        ka_P=mie_result.ka_P,
        ka_S=mie_result.ka_S,
    )
    trunc.a_n[n_trunc + 1 :] = 0.0
    trunc.b_n[n_trunc + 1 :] = 0.0

    f_P_trunc, f_SV_trunc, _ = mie_far_field(trunc, theta, incident_type="P")

    # L2 relative error combining P and SV channels
    diff_P = f_P_full - f_P_trunc
    diff_SV = f_SV_full - f_SV_trunc
    norm_full = np.sqrt(np.sum(np.abs(f_P_full) ** 2 + np.abs(f_SV_full) ** 2))

    if norm_full < 1e-30:
        return 0.0

    norm_diff = np.sqrt(np.sum(np.abs(diff_P) ** 2 + np.abs(diff_SV) ** 2))
    return float(norm_diff / norm_full)


@dataclass
class ConvergenceResult:
    """Result from convergence study.

    Attributes:
        ka_values: Array of ka values tested.
        n_trunc_values: Array of truncation orders tested.
        errors: 2D error array, shape (len(ka_values), len(n_trunc_values)).
        E_n_static: Static Eshelby factors for reference.
        ka_thresholds: Dict mapping (n_trunc, error_pct) → max ka below threshold.
    """

    ka_values: NDArray[np.floating]
    n_trunc_values: NDArray[np.signedinteger]
    errors: NDArray[np.floating]
    E_n_static: NDArray[np.complexfloating]
    ka_thresholds: dict[tuple[int, float], float] = field(default_factory=dict)


def convergence_study(
    ref: ReferenceMedium,
    contrast: MaterialContrast,
    ka_values: NDArray[np.floating] | None = None,
    n_trunc_values: NDArray[np.signedinteger] | None = None,
    radius: float = 1.0,
    n_max_mie: int = 15,
) -> ConvergenceResult:
    """Sweep over ka × n_trunc to quantify far-field truncation error.

    For each (ka, n_trunc) pair, computes the L2 relative error of the
    truncated far-field vs the full Mie solution.

    Args:
        ref: Background medium.
        contrast: Material contrasts.
        ka_values: Array of ka_S values. Default: logspace from 0.05 to 2.0.
        n_trunc_values: Array of truncation orders. Default: [2, 3, 4, 5].
        radius: Sphere radius (m).
        n_max_mie: Maximum Mie order for the reference solution.

    Returns:
        ConvergenceResult with error table and ka thresholds.
    """
    if ka_values is None:
        ka_values = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0])
    if n_trunc_values is None:
        n_trunc_values = np.array([2, 3, 4, 5])

    ka_values = np.asarray(ka_values, dtype=float)
    n_trunc_values = np.asarray(n_trunc_values, dtype=int)

    errors = np.zeros((len(ka_values), len(n_trunc_values)))

    for i, ka in enumerate(ka_values):
        omega = ka * ref.beta / radius
        mie = compute_elastic_mie(omega, radius, ref, contrast, n_max=n_max_mie)
        for j, n_trunc in enumerate(n_trunc_values):
            errors[i, j] = far_field_truncation_error(mie, int(n_trunc))

    # Compute static Eshelby factors
    E_n_static = compute_static_eshelby_factors(
        ref, contrast, n_max=max(int(n_trunc_values.max()), 5)
    )

    # Compute ka thresholds
    ka_thresholds: dict[tuple[int, float], float] = {}
    for pct in [1.0, 5.0, 10.0]:
        threshold = pct / 100.0
        for j, n_trunc in enumerate(n_trunc_values):
            below = ka_values[errors[:, j] < threshold]
            ka_thresholds[(int(n_trunc), pct)] = (
                float(below[-1]) if len(below) > 0 else 0.0
            )

    return ConvergenceResult(
        ka_values=ka_values,
        n_trunc_values=n_trunc_values,
        errors=errors,
        E_n_static=E_n_static,
        ka_thresholds=ka_thresholds,
    )
