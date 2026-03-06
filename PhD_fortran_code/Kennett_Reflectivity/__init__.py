"""
Kennett_Reflectivity: Kennett's recursive reflectivity method for stratified elastic media.

This package converts the Fortran program ``kennetslo.f`` to modern Python 3.12 with NumPy.
It computes plane wave reflectivity responses of stratified elastic half-spaces using
Kennett's Addition Formulae.

Main components:
  - LayerModel: Representation of stratified elastic media
  - Scattering matrices: P-SV interfacial reflection/transmission coefficients
  - Kennett reflectivity: Recursive algorithm for reflectivity computation
  - Seismogram generation: Synthetic seismogram computation and visualization
"""

from __future__ import annotations

from .kennett_reflectivity import inv2x2, kennett_reflectivity
from .layer_model import LayerModel, complex_slowness, vertical_slowness
from .scattering_matrices import (
    ScatteringCoefficients,
    ocean_bottom_interface,
    solid_solid_interface,
)
from .source import ricker_spectrum, ricker_wavelet

__version__ = "1.0.0"
__author__ = "Converted from kennetslo.f (Kennett's method)"

__all__ = [
    "LayerModel",
    "complex_slowness",
    "vertical_slowness",
    "ScatteringCoefficients",
    "solid_solid_interface",
    "ocean_bottom_interface",
    "kennett_reflectivity",
    "inv2x2",
    "ricker_spectrum",
    "ricker_wavelet",
    "compute_seismogram",
    "compute_gather",
    "plot_gather",
    "default_ocean_crust_model",
    "compute_gather_gpu",
    "kennett_reflectivity_batch",
    "get_device",
]


def __getattr__(name: str):
    """Lazy-load submodules to avoid conflicts with ``python -m``."""
    _seismogram_names = ("compute_seismogram", "default_ocean_crust_model")
    _gather_names = ("compute_gather", "plot_gather")

    if name in _seismogram_names:
        from .kennett_seismogram import compute_seismogram, default_ocean_crust_model

        return {
            "compute_seismogram": compute_seismogram,
            "default_ocean_crust_model": default_ocean_crust_model,
        }[name]
    if name in _gather_names:
        from .kennett_gather import compute_gather, plot_gather

        return {"compute_gather": compute_gather, "plot_gather": plot_gather}[name]

    _gpu_names = ("compute_gather_gpu", "kennett_reflectivity_batch", "get_device")
    if name in _gpu_names:
        from .kennett_gather_gpu import compute_gather_gpu
        from .kennett_reflectivity_gpu import get_device, kennett_reflectivity_batch

        return {
            "compute_gather_gpu": compute_gather_gpu,
            "kennett_reflectivity_batch": kennett_reflectivity_batch,
            "get_device": get_device,
        }[name]

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
