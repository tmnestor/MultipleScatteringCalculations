"""
cubic_scattering — Elastic multiple scattering from cubic heterogeneities.
This package provides:
T₀ (Rayleigh) : Self-consistent T-matrix for cubic elastic scatterers (ka << 1)
T₀ (Resonance) : Full-wave T-matrix via internal Foldy-Lax subdivision (ka ~ O(1))
G₀ : Lattice Green's tensor for inter-site coupling
Coordinate system: z (down), x (right), y (out of page) — right-handed.
Rayleigh regime (ka << 1)
--------------------------
Uses a Taylor-expanded Green's tensor with analytic self-consistent
amplification factors (Gubernatis et al. 1977 + Eshelby correction).
Entry points:
compute_cube_tmatrix() — full T-matrix computation
voigt_tmatrix_6x6() — 6×6 Voigt strain-space T-matrix
Resonance regime (ka ~ O(1))
-----------------------------
Subdivides the cube into n³ Rayleigh sub-cells and solves the internal
Foldy-Lax system with the full elastodynamic Green's tensor (near- +
intermediate- + far-field coupling). Reduces to the Rayleigh result at n=1.
Entry points:
compute_resonance_tmatrix() — full-wave T-matrix computation
suggest_n_subcells() — auto-select n for ka_sub < 0.3
voigt_tmatrix_from_resonance_result() — 6×6 Voigt T-matrix (approximate;
see TODO in resonance_tmatrix.py)
scattering_order_decomposition() — Neumann-series order analysis
validate_rayleigh_limit() — regression: resonance → Rayleigh at n=1
Regime selection guide
----------------------
ka = (ω/β) · a Regime Module
─────────────────────────────────────────────────────────────────────
ka < 0.3 Rayleigh effective_contrasts.py (fast)
0.3 ≤ ka < 1.0 Transition resonance_tmatrix.py (n=2–4)
ka ≥ 1.0 Resonance resonance_tmatrix.py (n≥4)
For seismic exploration (10–200 Hz) with metre-scale cubes in typical
crustal velocities (β ≈ 1500–3500 m/s):
a = 0.1 m → ka ≈ 0.002–0.08 (deep Rayleigh)
a = 1.0 m → ka ≈ 0.02–0.8 (Rayleigh to transition)
a = 5.0 m → ka ≈ 0.09–4.2 (transition to resonance)
a = 10 m → ka ≈ 0.18–8.4 (full resonance regime)
"""

from .cpa_iteration import (
    CPAResult,
    CubicEffectiveMedium,
    Phase,
    compute_cpa,
    compute_cpa_two_phase,
    phases_from_two_phase,
    voigt_average,
)
from .cube_eshelby import (
    CubeConvergenceResult,
    CubeEshelbyResult,
    compute_cube_eshelby,
    compute_cube_eshelby_factors,
    cube_convergence_study,
)
from .effective_contrasts import (
    CubeTMatrixResult,
    GalerkinTMatrixResult,
    GalerkinTMatrixResult57,
    MaterialContrast,
    ReferenceMedium,
    compute_cube_tmatrix,
    compute_cube_tmatrix_galerkin,
    compute_cube_tmatrix_galerkin_57,
)
from .horizontal_greens import exact_propagator_9x9
from .incident_field import (
    cube_overlap_integrals,
    cube_overlap_integrals_57,
    plane_wave_PSV_SH,
)
from .kennett_layers import (
    FluidLayer,
    IsotropicLayer,
    KennettResult,
    LayerStack,
    PSVCoefficients,
    SHCoefficients,
    cpa_stack_from_phases,
    cubic_to_isotropic_layer,
    kennett_layers,
    kennett_reflectivity_batch,
    psv_fluid_solid,
    random_heterogeneous_stack,
    random_velocity_stack,
)
from .lattice_greens import LatticeGreens
from .multipole_eshelby import (
    ConvergenceResult,
    MultipoleEshelbyResult,
    compute_multipole_eshelby,
    convergence_study,
)
from .resonance_tmatrix import (
    ResonanceTmatrixResult,
    compute_resonance_tmatrix,
    elastodynamic_greens,
    elastodynamic_greens_deriv,
    scattering_order_decomposition,
    sub_cell_centres,
    suggest_n_subcells,
    validate_rayleigh_limit,
    voigt_tmatrix_from_resonance_result,
)
from .scattered_field import (
    cube_far_field,
    optical_theorem_check,
    resonance_far_field,
    scattering_cross_section,
)
from .seismic_survey import (
    GatherConfig,
    ShotGatherResult,
    SurveyConfig,
    bessel_summation,
    bessel_summation_gpu,
    build_survey_stack,
    compute_shot_gather,
    free_surface_reverberations,
    load_survey_config,
    receiver_ghost,
    ricker_source_spectrum,
    source_ghost,
)
from .slab_scattering import (
    SlabGeometry,
    SlabMaterial,
    SlabResult,
    compute_slab_scattering,
    compute_slab_tmatrices,
    random_slab_material,
    slab_reflected_field,
    uniform_slab_material,
)
from .slab_scattering_gpu import compute_slab_scattering_gpu
from .solver_config import (
    ScatteringConfig,
    load_config,
    run_from_config,
    validate_config,
)
from .sphere_scattering import (
    MieEffectiveContrasts,
    MieResult,
    SphereDecompositionResult,
    compute_elastic_mie,
    compute_sphere_foldy_lax,
    decompose_SV_SH,
    foldy_lax_far_field,
    mie_extract_effective_contrasts,
    mie_far_field,
    mie_scattered_displacement,
    sphere_sub_cell_centres,
)
from .sphere_scattering_fft import compute_sphere_foldy_lax_fft
from .sphere_scattering_fft_gpu import compute_sphere_foldy_lax_fft_gpu
from .tmatrix_assembly import (
    assemble_tmatrix_27,
    assemble_tmatrix_57,
    tmatrix_27_to_voigt_6x6,
    tmatrix_57_to_voigt_6x6,
)
from .torch_gmres import get_device, select_dtype, torch_gmres
from .voigt_tmatrix import (
    effective_stiffness_voigt,
    scattered_stress_voigt,
    strain_from_displacement_traction,
    tmatrix_displacement_traction,
    traction_from_strain,
    voigt_tmatrix_6x6,
    voigt_tmatrix_from_result,
)

__all__ = [
    # ── Rayleigh regime ──────────────────────────────────────────────────
    "ReferenceMedium",
    "MaterialContrast",
    "CubeTMatrixResult",
    "compute_cube_tmatrix",
    "GalerkinTMatrixResult",
    "GalerkinTMatrixResult57",
    "compute_cube_tmatrix_galerkin",
    "compute_cube_tmatrix_galerkin_57",
    "voigt_tmatrix_6x6",
    "voigt_tmatrix_from_result",
    "effective_stiffness_voigt",
    "strain_from_displacement_traction",
    "traction_from_strain",
    "tmatrix_displacement_traction",
    "scattered_stress_voigt",
    # ── T27 assembly and far-field ──────────────────────────────────────
    "assemble_tmatrix_27",
    "assemble_tmatrix_57",
    "tmatrix_27_to_voigt_6x6",
    "tmatrix_57_to_voigt_6x6",
    "cube_overlap_integrals",
    "cube_overlap_integrals_57",
    "plane_wave_PSV_SH",
    "cube_far_field",
    "resonance_far_field",
    "scattering_cross_section",
    "optical_theorem_check",
    # ── Resonance regime ─────────────────────────────────────────────────
    "ResonanceTmatrixResult",
    "compute_resonance_tmatrix",
    "suggest_n_subcells",
    "validate_rayleigh_limit",
    "scattering_order_decomposition",
    "voigt_tmatrix_from_resonance_result",
    # ── Green's tensor (shared) ──────────────────────────────────────────
    "elastodynamic_greens",
    "elastodynamic_greens_deriv",
    "sub_cell_centres",
    # ── Horizontal Green's tensor ────────────────────────────────────────
    "exact_propagator_9x9",
    # ── Lattice ──────────────────────────────────────────────────────────
    "LatticeGreens",
    # ── Sphere scattering (validation) ────────────────────────────────
    "SphereDecompositionResult",
    "MieResult",
    "MieEffectiveContrasts",
    "mie_extract_effective_contrasts",
    "compute_sphere_foldy_lax",
    "compute_elastic_mie",
    "mie_far_field",
    "mie_scattered_displacement",
    "foldy_lax_far_field",
    "decompose_SV_SH",
    "sphere_sub_cell_centres",
    # ── Sphere scattering FFT (scalable) ────────────────────────────
    "compute_sphere_foldy_lax_fft",
    # ── Multipole Eshelby concentration factors ────────────────────
    "MultipoleEshelbyResult",
    "ConvergenceResult",
    "compute_multipole_eshelby",
    "convergence_study",
    # ── Cube Eshelby concentration factors ────────────────────────
    "CubeEshelbyResult",
    "CubeConvergenceResult",
    "compute_cube_eshelby",
    "compute_cube_eshelby_factors",
    "cube_convergence_study",
    # ── CPA iteration (self-consistent effective medium) ────────────
    "Phase",
    "CubicEffectiveMedium",
    "CPAResult",
    "compute_cpa",
    "compute_cpa_two_phase",
    "phases_from_two_phase",
    "voigt_average",
    # ── Kennett layer stacking ────────────────────────────────────────
    "FluidLayer",
    "IsotropicLayer",
    "LayerStack",
    "PSVCoefficients",
    "SHCoefficients",
    "KennettResult",
    "kennett_layers",
    "kennett_reflectivity_batch",
    "psv_fluid_solid",
    "cubic_to_isotropic_layer",
    "cpa_stack_from_phases",
    "random_heterogeneous_stack",
    "random_velocity_stack",
    # ── Seismic survey simulation ──────────────────────────────────────
    "SurveyConfig",
    "GatherConfig",
    "ShotGatherResult",
    "source_ghost",
    "receiver_ghost",
    "free_surface_reverberations",
    "ricker_source_spectrum",
    "bessel_summation",
    "bessel_summation_gpu",
    "build_survey_stack",
    "compute_shot_gather",
    "load_survey_config",
    # ── Slab Foldy-Lax multiple scattering ───────────────────────────
    "SlabGeometry",
    "SlabMaterial",
    "SlabResult",
    "compute_slab_scattering",
    "compute_slab_tmatrices",
    "slab_reflected_field",
    "random_slab_material",
    "uniform_slab_material",
    # ── GPU-accelerated solvers ─────────────────────────────────────
    "compute_slab_scattering_gpu",
    "compute_sphere_foldy_lax_fft_gpu",
    "torch_gmres",
    "get_device",
    "select_dtype",
    # ── YAML configuration ──────────────────────────────────────────
    "ScatteringConfig",
    "load_config",
    "validate_config",
    "run_from_config",
]
