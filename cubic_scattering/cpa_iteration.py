"""Coherent Potential Approximation (CPA) for space-filling cubic lattices.

Iterates the single-cube T-matrix computation (from effective_contrasts.py)
with an updating effective medium until the CPA condition <T> = 0 is
satisfied. This achieves simultaneous static (Eshelby) and dynamic
self-consistency.

For a space-filling lattice of cubes (volume fraction phi=1, lattice
spacing d=2a), the effective medium has cubic symmetry with 3 independent
moduli: lambda*, mu*_off, mu*_diag.

The iteration:
  1. Initialize C* = <C> (Voigt average)
  2. For each phase n: compute T_n = T(C_n - C*, C*) using compute_cube_tmatrix
  3. Enforce <T> = sum_n phi_n T_n = 0 by updating C*
  4. Repeat until convergence

Reference: Sabina & Willis (1993), adapted from spheres to cubes.
"""

from dataclasses import dataclass

import numpy as np

from .effective_contrasts import (
    CubeTMatrixResult,
    MaterialContrast,
    ReferenceMedium,
    compute_cube_tmatrix,
)


@dataclass
class Phase:
    """A distinct material phase in the composite.

    Attributes:
        lam: Lame parameter lambda (Pa).
        mu: Shear modulus mu (Pa).
        rho: Density (kg/m^3).
        volume_fraction: Volume fraction of this phase (0 < phi <= 1).
    """

    lam: float
    mu: float
    rho: float
    volume_fraction: float

    @property
    def alpha(self) -> float:
        """P-wave velocity."""
        return np.sqrt((self.lam + 2.0 * self.mu) / self.rho)

    @property
    def beta(self) -> float:
        """S-wave velocity."""
        return np.sqrt(self.mu / self.rho)

    @property
    def bulk_modulus(self) -> float:
        """Bulk modulus K = lambda + 2mu/3."""
        return self.lam + 2.0 * self.mu / 3.0


@dataclass
class CubicEffectiveMedium:
    """Effective medium with cubic symmetry (3 independent moduli).

    For a cube lattice the effective medium inherits the cubic symmetry
    of the microstructure: mu*_off != mu*_diag in general.

    Attributes:
        lam: Effective Lame parameter lambda* (Pa).
        mu_off: Effective off-diagonal shear modulus mu*_off (Pa).
        mu_diag: Effective diagonal shear modulus mu*_diag (Pa).
        rho: Effective density (kg/m^3).
    """

    lam: float
    mu_off: float
    mu_diag: float
    rho: float

    @property
    def mu_iso(self) -> float:
        """Isotropic part of shear modulus (average)."""
        return (self.mu_off + self.mu_diag) / 2.0

    @property
    def bulk_modulus(self) -> float:
        """Bulk modulus K = lambda + 2mu_diag/3."""
        return self.lam + 2.0 * self.mu_diag / 3.0

    @property
    def cubic_anisotropy(self) -> float:
        """mu_diag - mu_off: measure of cubic anisotropy."""
        return self.mu_diag - self.mu_off

    def as_reference_medium(self) -> ReferenceMedium:
        """Convert to isotropic ReferenceMedium for T-matrix computation.

        Uses mu_off as the isotropic shear modulus. The cubic anisotropy
        (mu_diag - mu_off) enters through the C^c coefficient in the
        T-matrix computation.
        """
        mu = self.mu_off
        rho = self.rho
        beta = np.sqrt(mu / rho)
        lam = self.lam
        alpha = np.sqrt((lam + 2.0 * mu) / rho)
        return ReferenceMedium(alpha=alpha, beta=beta, rho=rho)


@dataclass
class CPAResult:
    """Result of CPA iteration.

    Attributes:
        effective_medium: Converged effective medium.
        n_iterations: Number of iterations to convergence.
        converged: Whether the iteration converged.
        residual_history: Max residual at each iteration.
        tmatrix_results: Per-phase T-matrix results at convergence.
        phases: Input phases.
        omega: Angular frequency used.
        a: Cube half-width used.
    """

    effective_medium: CubicEffectiveMedium
    n_iterations: int
    converged: bool
    residual_history: list[float]
    tmatrix_results: list[CubeTMatrixResult]
    phases: list[Phase]
    omega: float
    a: float


def voigt_average(phases: list[Phase]) -> CubicEffectiveMedium:
    """Compute Voigt (upper bound) average of phases.

    C*_Voigt = sum_n phi_n C_n

    This is the natural starting point for CPA iteration.

    Args:
        phases: List of material phases with volume fractions.

    Returns:
        CubicEffectiveMedium at the Voigt average.
    """
    lam_avg = sum(p.lam * p.volume_fraction for p in phases)
    mu_avg = sum(p.mu * p.volume_fraction for p in phases)
    rho_avg = sum(p.rho * p.volume_fraction for p in phases)
    return CubicEffectiveMedium(lam=lam_avg, mu_off=mu_avg, mu_diag=mu_avg, rho=rho_avg)


def _phase_contrast(phase: Phase, eff: CubicEffectiveMedium) -> MaterialContrast:
    """Compute contrast of a phase relative to the effective medium.

    DC_n = C_n - C*

    For the isotropic part we use mu_off as reference.
    The diagonal deviation (mu_diag - mu_off) enters separately
    through the C^c channel.
    """
    return MaterialContrast(
        Dlambda=phase.lam - eff.lam,
        Dmu=phase.mu - eff.mu_off,
        Drho=phase.rho - eff.rho,
    )


def _compute_cpa_update(
    phases: list[Phase],
    eff: CubicEffectiveMedium,
    omega: float,
    a: float,
) -> tuple[CubicEffectiveMedium, float, list[CubeTMatrixResult]]:
    """One CPA iteration step.

    Computes T_n for each phase in the current effective medium,
    then updates C* so that <T> = 0.

    The CPA condition is enforced on the 3 isotropic channels:
        <Drho*> = sum phi_n Drho*_n = 0
        <Dlambda*> = sum phi_n Dlambda*_n = 0
        <Dmu*_off> = sum phi_n Dmu*_off_n = 0

    The diagonal shear modulus mu*_diag is a derived quantity:
    since compute_cube_tmatrix takes an isotropic reference, we
    cannot independently control the diagonal channel. The cubic
    anisotropy (mu*_diag - mu*_off) emerges from the C^c coefficient
    and is computed as a diagnostic after convergence.

    Returns:
        Updated effective medium, max residual, per-phase T-matrix results.
    """
    ref = eff.as_reference_medium()

    avg_Drho = 0.0
    avg_Dlambda = 0.0
    avg_Dmu_off = 0.0
    avg_Dmu_diag = 0.0
    results: list[CubeTMatrixResult] = []

    for phase in phases:
        contrast = _phase_contrast(phase, eff)
        result = compute_cube_tmatrix(omega, a, ref, contrast)
        results.append(result)

        phi = phase.volume_fraction
        avg_Drho += phi * result.Drho_star.real
        avg_Dlambda += phi * result.Dlambda_star.real
        avg_Dmu_off += phi * result.Dmu_star_off.real
        avg_Dmu_diag += phi * result.Dmu_star_diag.real

    # Update isotropic channels: lambda*, mu*_off, rho*
    new_mu_off = eff.mu_off + avg_Dmu_off

    # mu*_diag is derived: mu*_off + cubic anisotropy from <Dmu*_diag - Dmu*_off>
    new_mu_diag = new_mu_off + (avg_Dmu_diag - avg_Dmu_off) + eff.cubic_anisotropy

    new_eff = CubicEffectiveMedium(
        lam=eff.lam + avg_Dlambda,
        mu_off=new_mu_off,
        mu_diag=new_mu_diag,
        rho=eff.rho + avg_Drho,
    )

    # Residual: only the 3 isotropic channels (not diagonal shear)
    scale_modulus = max(abs(eff.lam) + 2.0 * abs(eff.mu_off), 1.0)
    scale_rho = max(abs(eff.rho), 1.0)
    residual = max(
        abs(avg_Drho) / scale_rho,
        abs(avg_Dlambda) / scale_modulus,
        abs(avg_Dmu_off) / scale_modulus,
    )

    return new_eff, residual, results


def compute_cpa(
    phases: list[Phase],
    omega: float,
    a: float,
    max_iter: int = 100,
    tol: float = 1e-10,
    damping: float = 1.0,
    initial_medium: CubicEffectiveMedium | None = None,
) -> CPAResult:
    """Run the CPA iteration to find the self-consistent effective medium.

    Args:
        phases: List of material phases. Volume fractions must sum to 1.
        omega: Angular frequency (rad/s).
        a: Cube half-width (m).
        max_iter: Maximum number of iterations.
        tol: Convergence tolerance on normalised residual.
        damping: Under-relaxation factor (0 < damping <= 1).
            Use < 1 for stability at high contrast.
        initial_medium: Starting effective medium. Default: Voigt average.

    Returns:
        CPAResult with the converged effective medium and diagnostics.
    """
    # Validate volume fractions
    total_phi = sum(p.volume_fraction for p in phases)
    if abs(total_phi - 1.0) > 1e-10:
        msg = f"Volume fractions must sum to 1.0, got {total_phi}"
        raise ValueError(msg)

    # Initialize
    if initial_medium is not None:
        eff = initial_medium
    else:
        eff = voigt_average(phases)

    residual_history: list[float] = []
    results: list[CubeTMatrixResult] = []

    for iteration in range(max_iter):
        new_eff, residual, results = _compute_cpa_update(phases, eff, omega, a)
        residual_history.append(residual)

        if residual < tol:
            return CPAResult(
                effective_medium=new_eff,
                n_iterations=iteration + 1,
                converged=True,
                residual_history=residual_history,
                tmatrix_results=results,
                phases=phases,
                omega=omega,
                a=a,
            )

        # Under-relaxation for stability
        if damping < 1.0:
            eff = CubicEffectiveMedium(
                lam=eff.lam + damping * (new_eff.lam - eff.lam),
                mu_off=eff.mu_off + damping * (new_eff.mu_off - eff.mu_off),
                mu_diag=eff.mu_diag + damping * (new_eff.mu_diag - eff.mu_diag),
                rho=eff.rho + damping * (new_eff.rho - eff.rho),
            )
        else:
            eff = new_eff

    return CPAResult(
        effective_medium=eff,
        n_iterations=max_iter,
        converged=False,
        residual_history=residual_history,
        tmatrix_results=results,
        phases=phases,
        omega=omega,
        a=a,
    )


def phases_from_two_phase(
    ref: ReferenceMedium,
    contrast: MaterialContrast,
    volume_fraction: float,
) -> list[Phase]:
    """Create a two-phase system from reference + contrast.

    Phase 1 (matrix): properties = ref, volume fraction = 1 - phi
    Phase 2 (inclusion): properties = ref + contrast, volume fraction = phi

    Args:
        ref: Background/matrix medium.
        contrast: Inclusion contrast relative to background.
        volume_fraction: Volume fraction of inclusions (0 < phi < 1).

    Returns:
        List of two Phase objects.
    """
    matrix = Phase(
        lam=ref.lam,
        mu=ref.mu,
        rho=ref.rho,
        volume_fraction=1.0 - volume_fraction,
    )
    inclusion = Phase(
        lam=ref.lam + contrast.Dlambda,
        mu=ref.mu + contrast.Dmu,
        rho=ref.rho + contrast.Drho,
        volume_fraction=volume_fraction,
    )
    return [matrix, inclusion]


def compute_cpa_two_phase(
    ref: ReferenceMedium,
    contrast: MaterialContrast,
    volume_fraction: float,
    omega: float,
    a: float,
    **kwargs,
) -> CPAResult:
    """Convenience wrapper for two-phase CPA.

    Args:
        ref: Background/matrix medium.
        contrast: Inclusion contrast.
        volume_fraction: Inclusion volume fraction.
        omega: Angular frequency (rad/s).
        a: Cube half-width (m).
        **kwargs: Passed to compute_cpa (max_iter, tol, damping).

    Returns:
        CPAResult with the converged effective medium.
    """
    phases = phases_from_two_phase(ref, contrast, volume_fraction)
    return compute_cpa(phases, omega, a, **kwargs)
