"""Main driver for FFTPROP: 2.5D spectral scattering computation.

Faithful translation of the FFTPROP.F main program orchestration.

This module ties together the spectral array setup, source/receiver
projection, and all propagation sweeps to compute the scattered
wavefield at each scatterer position.

The algorithm:
    1. Build spectral arrays (wavenumber grid, phase, free surface, PC)
    2. Initialise scattering source array SY
    3. Source downsweep → Svec
    4. Receiver downsweep → Rvec
    5. Upsweep (bottom to top) → PSY, PU/SU at free surface
    6. Free-surface reflection → PD/SD
    7. Downsweep (top to bottom) → PSY continued
    8. For each depth layer: right sweep + left sweep → PSY lateral
    9. Return PSY, Svec, Rvec
"""

from __future__ import annotations

import logging

import numpy as np

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
from .spectral_arrays import build_spectral_arrays

__all__ = ["compute_wavefield", "default_source_array"]

logger = logging.getLogger(__name__)


def default_source_array(grid: GridConfig) -> np.ndarray:
    """Create the default scattering source array SY from FFTPROP.F.

    Fortran line 114: SY(Nscatx, -2, 1, Nscatz) = C1
    This places a unit P-wave source at the bottom-right corner of the
    scatterer array (deepest layer, rightmost position, m=-2 harmonic).

    Parameters
    ----------
    grid : GridConfig
        Grid configuration.

    Returns
    -------
    SY : np.ndarray
        Scattering source array, shape (Nscatx, 5, 2, Nscatz).
        All zeros except SY[Nscatx-1, 0, 0, Nscatz-1] = 1.0.
    """
    SY = np.zeros((grid.Nscatx, 5, 2, grid.Nscatz), dtype=np.complex128)
    # Fortran: SY(Nscatx, -2, 1, Nscatz) = C1
    # Python:  SY[Nscatx-1, 0, 0, Nscatz-1] = 1.0
    #   jscat: Nscatx → Nscatx-1 (0-based)
    #   m=-2: m_idx = m+2 = 0
    #   wavetype=1 (P): index 0
    #   iscat=Nscatz → Nscatz-1 (0-based)
    SY[grid.Nscatx - 1, 0, 0, grid.Nscatz - 1] = 1.0 + 0.0j
    return SY


def compute_wavefield(
    medium: ReferenceMedium | None = None,
    grid: GridConfig | None = None,
    src: SourceReceiverConfig | None = None,
    freq: float = 2.0,
    atten_imag: float = 0.2,
    SY: np.ndarray | None = None,
) -> PropagationResult:
    """Compute the 2.5D spectral scattering wavefield.

    This is the top-level function reproducing the full FFTPROP.F program.

    Parameters
    ----------
    medium : ReferenceMedium, optional
        Reference medium. Default: alpha=5.0, rho=3.0, Q=50.0.
    grid : GridConfig, optional
        Grid configuration. Default: Nk=4096, Nscatx=81, Nscatz=2, jskip=8.
    src : SourceReceiverConfig, optional
        Source/receiver setup. Default: Xs=10, Xr=0, P-source, P-receiver.
    freq : float
        Frequency in Hz, default 2.0.
    atten_imag : float
        Imaginary part of complex omega, default 0.2.
    SY : np.ndarray, optional
        Custom scattering source array. Default: unit source at bottom-right
        corner (Fortran convention).

    Returns
    -------
    PropagationResult
        Contains SY, PSY, Svec, Rvec arrays.
    """
    if medium is None:
        medium = default_medium()
    if grid is None:
        grid = default_grid()
    if src is None:
        src = default_source_receiver()

    logger.info(
        f"FFTPROP: alpha={medium.alpha}, rho={medium.rho}, Q={medium.Q}, "
        f"freq={freq}, Nk={grid.Nk}, Nscatx={grid.Nscatx}, "
        f"Nscatz={grid.Nscatz}, jskip={grid.jskip}"
    )

    # Step 1: Build spectral arrays
    logger.info("Building spectral arrays...")
    sa = build_spectral_arrays(medium, grid, freq, atten_imag)

    # Step 2: Initialise scattering sources and result arrays
    result = PropagationResult(grid)
    if SY is None:
        result.SY = default_source_array(grid)
    else:
        result.SY = SY.copy()

    # Step 3: Source downsweep (Fortran loop 6000)
    logger.info("Source downsweep...")
    source_downsweep(sa, grid, src, result)

    # Step 4: Receiver downsweep (Fortran loop 8000)
    logger.info("Receiver downsweep...")
    receiver_downsweep(sa, grid, src, result)

    # Step 5: Upsweep (Fortran loop 1000)
    logger.info("Upsweep...")
    PU, SU = upsweep(sa, grid, result)

    # Step 6: Free-surface reflection (Fortran lines 367-370)
    logger.info("Free-surface reflection...")
    PD, SD = free_surface_reflect(sa, PU, SU)

    # Step 7: Downsweep (Fortran loop 2000)
    logger.info("Downsweep...")
    downsweep(sa, grid, PD, SD, result)

    # Step 8: Lateral sweeps per depth layer (Fortran loop 5000)
    logger.info("Lateral sweeps...")
    for iscat in range(grid.Nscatz - 1, -1, -1):
        logger.info(f"  Depth layer {iscat + 1}: right sweep...")
        right_sweep(sa, grid, iscat, result)
        logger.info(f"  Depth layer {iscat + 1}: left sweep...")
        left_sweep(sa, grid, iscat, result)

    logger.info("FFTPROP computation complete.")
    return result


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Running FFTPROP with default parameters...")
    result = compute_wavefield()

    # Print output matching Fortran (lines 515–527)
    j = 0
    for iscat in range(default_grid().Nscatz):
        for jscat in range(default_grid().Nscatx):
            for i_wt in range(2):  # wavetype: 0=P, 1=SV
                for m_idx in range(5):  # m=-2..+2
                    j += 1
                    rvec_val = result.Rvec[jscat, m_idx, i_wt, iscat]
                    svec_val = result.Svec[jscat, m_idx, i_wt, iscat]
                    print(
                        f"Rvec[{jscat + 1},{m_idx - 2},{i_wt + 1},{iscat + 1}] = "
                        f"{j} ({rvec_val.real:+.6e},{rvec_val.imag:+.6e}j)"
                    )
                    print(
                        f"Svec[{jscat + 1},{m_idx - 2},{i_wt + 1},{iscat + 1}] = "
                        f"{j} ({svec_val.real:+.6e},{svec_val.imag:+.6e}j)"
                    )

    print("normal completion")
