"""FFTProp: 2.5D spectral scattering using FFT-based plane-wave decomposition.

This package converts the Fortran program FFTPROP.F to modern Python 3.12
with NumPy. It computes the scattered wavefield in a 2D heterogeneous medium
embedded in a 3D homogeneous plane-layer reference medium, using cylindrical
harmonic expansions (m=-2..+2) for P and SV waves.

Main components:
  - ReferenceMedium, GridConfig: Model and grid parameters
  - SpectralArrays: Wavenumber grid, phase factors, free-surface reflection,
    plane-wave ↔ cylindrical-harmonic transforms
  - Propagation sweeps: Source/receiver projection, up/down/left/right sweeps
  - compute_wavefield: Top-level driver reproducing full FFTPROP.F
"""

from __future__ import annotations

from .fftprop_driver import (
    compute_wavefield,
    default_source_array,
)
from .medium import (
    GridConfig,
    ReferenceMedium,
    SourceReceiverConfig,
    default_grid,
    default_medium,
    default_source_receiver,
)
from .propagation import (
    PropagationResult,
    downsweep,
    free_surface_reflect,
    left_sweep,
    receiver_downsweep,
    right_sweep,
    source_downsweep,
    upsweep,
)
from .spectral_arrays import (
    SpectralArrays,
    build_spectral_arrays,
    vertical_slowness,
)

__version__ = "1.0.0"
__author__ = "Converted from FFTPROP.F (2.5D spectral scattering)"

__all__ = [
    # Medium and grid
    "ReferenceMedium",
    "GridConfig",
    "SourceReceiverConfig",
    "default_medium",
    "default_grid",
    "default_source_receiver",
    # Spectral arrays
    "vertical_slowness",
    "SpectralArrays",
    "build_spectral_arrays",
    # Propagation
    "PropagationResult",
    "source_downsweep",
    "receiver_downsweep",
    "upsweep",
    "free_surface_reflect",
    "downsweep",
    "right_sweep",
    "left_sweep",
    # Driver
    "compute_wavefield",
    "default_source_array",
]
