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

from .effective_contrasts import (
    CubeTMatrixResult,
    MaterialContrast,
    ReferenceMedium,
    compute_cube_tmatrix,
)
from .horizontal_greens import exact_propagator_9x9
from .lattice_greens import LatticeGreens
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
    "voigt_tmatrix_6x6",
    "voigt_tmatrix_from_result",
    "effective_stiffness_voigt",
    "strain_from_displacement_traction",
    "traction_from_strain",
    "tmatrix_displacement_traction",
    "scattered_stress_voigt",
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
]
