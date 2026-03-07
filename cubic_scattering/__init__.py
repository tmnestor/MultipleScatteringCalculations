"""
cubic_scattering — Elastic multiple scattering from cubic heterogeneities.

This package provides:
  T₀: Self-consistent T-matrix for cubic elastic scatterers
  G₀: Lattice Green's tensor for inter-site coupling

Coordinate system: z (down), x (right), y (out of page) — right-handed.

Main entry points:
  compute_cube_tmatrix()  — full T-matrix computation
  voigt_tmatrix_6x6()     — 6×6 Voigt strain-space T-matrix
  LatticeGreens           — lattice Green's tensor (spatial, spectral, hybrid, FCC)
"""

from .effective_contrasts import (
    CubeTMatrixResult,
    MaterialContrast,
    ReferenceMedium,
    compute_cube_tmatrix,
)
from .lattice_greens import LatticeGreens
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
    "LatticeGreens",
]
