"""Propagation sweeps: source/receiver projection and scattering iterations.

Faithful translation of FFTPROP.F sweep loops (lines 200–528).

The algorithm performs five propagation passes:

1. **Source downsweep** (Fortran loop 6000): Projects source field through
   free surface, transforms k→x at each scatterer depth to build Svec.

2. **Receiver downsweep** (Fortran loop 8000): Same as source downsweep
   but for receiver field, building Rvec.

3. **Upsweep** (Fortran loop 1000): Upward sweep from deepest scatterer
   layer, accumulating cylindrical-harmonic fields PSY via k→x transforms
   and scattering sources SY via x→k transforms.

4. **Free-surface reflection**: Applies Rayleigh reflection to convert
   upgoing (PU, SU) to downgoing (PD, SD).

5. **Downsweep** (Fortran loop 2000): Downward sweep from shallowest layer,
   continuing the PSY accumulation and scattering.

6. **Lateral sweeps** (Fortran loops 3000, 4000): Right and left sweeps
   within each depth layer for horizontal scattering.

FFT convention:
    Fortran FFT(X, N, +1.0) = sum_k X(k)*exp(+i*2π*n*k/N) = N * np.fft.ifft(X)
    Fortran FFT(X, N, -1.0) = sum_k X(k)*exp(-i*2π*n*k/N) = np.fft.fft(X)

    signi=+1 (k→x): Python equivalent = Nk * np.fft.ifft(X)
    signi=-1 (x→k): Python equivalent = np.fft.fft(X)

Note on Fortran equivalence:
    The Fortran uses `equivalence (Mu1, Md1)` etc. to share memory between
    up-sweep and down-sweep arrays. In Python, we simply reuse variables.

Note on right/left sweep stride:
    Fortran loops 3000/4000 use `Do ik=1,Nk,2` (stride 2), processing only
    odd-indexed wavenumbers. This is preserved as array slicing [::2].
"""

from __future__ import annotations

import logging

import numpy as np

from .medium import GridConfig, SourceReceiverConfig
from .spectral_arrays import SpectralArrays

__all__ = [
    "PropagationResult",
    "source_downsweep",
    "receiver_downsweep",
    "upsweep",
    "free_surface_reflect",
    "downsweep",
    "right_sweep",
    "left_sweep",
]

logger = logging.getLogger(__name__)


class PropagationResult:
    """Container for all propagation output arrays.

    Attributes
    ----------
    SY : np.ndarray
        Scattering source amplitudes, shape (Nscatx, 5, 2, Nscatz).
        Index 1: m+2 for m=-2..+2. Index 2: 0=P, 1=SV.
    PSY : np.ndarray
        Scattered wavefield amplitudes, same shape as SY.
    Svec : np.ndarray
        Source projection onto scatterer basis, same shape as SY.
    Rvec : np.ndarray
        Receiver projection onto scatterer basis, same shape as SY.
    """

    def __init__(self, grid: GridConfig):
        Nscatx = grid.Nscatx
        Nscatz = grid.Nscatz

        self.SY = np.zeros((Nscatx, 5, 2, Nscatz), dtype=np.complex128)
        self.PSY = np.zeros((Nscatx, 5, 2, Nscatz), dtype=np.complex128)
        self.Svec = np.zeros((Nscatx, 5, 2, Nscatz), dtype=np.complex128)
        self.Rvec = np.zeros((Nscatx, 5, 2, Nscatz), dtype=np.complex128)


def _fft_k2x(X: np.ndarray, Nk: int) -> np.ndarray:
    """Fortran FFT(X, Nk, +1.0): k→x transform (unnormalized).

    Equivalent to Nk * np.fft.ifft(X).
    """
    return Nk * np.fft.ifft(X)


def _fft_x2k(X: np.ndarray) -> np.ndarray:
    """Fortran FFT(X, Nk, -1.0): x→k transform.

    Equivalent to np.fft.fft(X).
    """
    return np.fft.fft(X)


def _project_source_or_receiver(
    sa: SpectralArrays,
    grid: GridConfig,
    X_position: float,
    wave_type: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project source/receiver through free surface into downgoing plane waves.

    Faithful translation of Fortran lines 200–212 and 248–260.

    Parameters
    ----------
    sa : SpectralArrays
        Precomputed spectral arrays.
    grid : GridConfig
        Grid configuration.
    X_position : float
        Horizontal position (Xs or Xr).
    wave_type : int
        1 for P-source/receiver, 2 for SV-source/receiver.

    Returns
    -------
    PD : np.ndarray
        Downgoing P-wave in wavenumber domain, shape (Nk,).
    SD : np.ndarray
        Downgoing S-wave in wavenumber domain, shape (Nk,).
    """
    Nk = grid.Nk
    temp = np.exp(-1j * sa.kxvec * X_position)

    PD = np.zeros(Nk, dtype=np.complex128)
    SD = np.zeros(Nk, dtype=np.complex128)

    if wave_type == 1:
        # Fortran: is=1 branch (P-source)
        PD = np.sqrt(sa.dkx / (np.pi * sa.kzavec)) * sa.rtEa * sa.W11 * temp
        SD = np.sqrt(sa.dkx / (np.pi * sa.kzbvec)) * sa.rtEb * sa.W21 * temp
    else:
        # Fortran: is=2 branch (SV-source)
        PD = np.sqrt(sa.dkx / (np.pi * sa.kzavec)) * sa.rtEa * sa.W12 * temp
        SD = np.sqrt(sa.dkx / (np.pi * sa.kzbvec)) * sa.rtEb * sa.W22 * temp

    return PD, SD


def source_downsweep(
    sa: SpectralArrays,
    grid: GridConfig,
    src: SourceReceiverConfig,
    result: PropagationResult,
) -> None:
    """Source field downsweep: project source through free surface to scatterers.

    Faithful translation of Fortran loop 6000 (lines 200–245).

    Builds result.Svec: source coupling at each scatterer position.
    """
    Nk = grid.Nk
    jskip = grid.jskip

    # Project source through free surface (Fortran lines 200–212)
    PD, SD = _project_source_or_receiver(sa, grid, src.Xs, src.is_type)

    # Downsweep through depth layers (Fortran Do 6000 iscat=1,Nscatz)
    for iscat in range(grid.Nscatz):
        logger.debug(f"Source downsweep: iscat={iscat + 1}")

        # Transform to x-space for each harmonic order m
        # Fortran Do 700 m=-2,2
        for m_idx in range(5):  # m_idx = m + 2, so m = m_idx - 2
            # Multiply by PC in k-space, then k→x transform
            # Fortran: Md1(ik) = PC(ik,m,1,iscat) * PD(ik)
            Md1 = sa.PC[:, m_idx, 0, iscat] * PD
            Md2 = sa.PC[:, m_idx, 1, iscat] * SD

            # Fortran: call FFT(Md1, Nk, 1.0) — k→x
            Md1 = _fft_k2x(Md1, Nk)
            Md2 = _fft_k2x(Md2, Nk)

            # Sample at scatterer positions (Fortran: j=1, j=j+jskip)
            j = 0  # Python 0-based (Fortran j=1)
            for jscat in range(grid.Nscatx):
                result.Svec[jscat, m_idx, 0, iscat] += Md1[j]
                result.Svec[jscat, m_idx, 1, iscat] += Md2[j]
                j += jskip

        # Phase advance for next depth layer
        # Fortran: PD(ik) = Eavec(ik)*PD(ik), SD(ik) = Ebvec(ik)*SD(ik)
        PD = sa.Eavec * PD
        SD = sa.Ebvec * SD


def receiver_downsweep(
    sa: SpectralArrays,
    grid: GridConfig,
    src: SourceReceiverConfig,
    result: PropagationResult,
) -> None:
    """Receiver field downsweep: project receiver through free surface.

    Faithful translation of Fortran loop 8000 (lines 248–293).

    Builds result.Rvec: receiver coupling at each scatterer position.
    """
    Nk = grid.Nk
    jskip = grid.jskip

    # Project receiver through free surface (Fortran lines 248–260)
    PD, SD = _project_source_or_receiver(sa, grid, src.Xr, src.ir_type)

    # Downsweep through depth layers (Fortran Do 8000 iscat=1,Nscatz)
    for iscat in range(grid.Nscatz):
        logger.debug(f"Receiver downsweep: iscat={iscat + 1}")

        for m_idx in range(5):
            Md1 = sa.PC[:, m_idx, 0, iscat] * PD
            Md2 = sa.PC[:, m_idx, 1, iscat] * SD

            Md1 = _fft_k2x(Md1, Nk)
            Md2 = _fft_k2x(Md2, Nk)

            j = 0
            for jscat in range(grid.Nscatx):
                result.Rvec[jscat, m_idx, 0, iscat] += Md1[j]
                result.Rvec[jscat, m_idx, 1, iscat] += Md2[j]
                j += jskip

        PD = sa.Eavec * PD
        SD = sa.Ebvec * SD


def upsweep(
    sa: SpectralArrays,
    grid: GridConfig,
    result: PropagationResult,
) -> tuple[np.ndarray, np.ndarray]:
    """Upward sweep from deepest scatterer layer to free surface.

    Faithful translation of Fortran loop 1000 (lines 303–364).

    Accumulates PSY (scattered field at each scatterer) and returns
    the upgoing wavefield (PU, SU) at the free surface.

    Returns
    -------
    PU : np.ndarray
        Upgoing P-wave in wavenumber domain, shape (Nk,).
    SU : np.ndarray
        Upgoing S-wave in wavenumber domain, shape (Nk,).
    """
    Nk = grid.Nk
    jskip = grid.jskip

    # Initialise upgoing field to zero (Fortran lines 304–307)
    PU = np.zeros(Nk, dtype=np.complex128)
    SU = np.zeros(Nk, dtype=np.complex128)

    # Upward sweep: iscat = Nscatz down to 1 (Fortran Do 1000 iscat=Nscatz,1,-1)
    for iscat in range(grid.Nscatz - 1, -1, -1):
        logger.debug(f"Upsweep: iscat={iscat + 1}")

        # Phase advance (Fortran lines 313–316)
        PU = sa.Eavec * PU
        SU = sa.Ebvec * SU

        # --- Accumulate incoming field into PSY (Fortran Do 50 m=-2,2) ---
        # Note: Fortran uses PC(ik, -m, ...) for the projection
        for m_idx in range(5):
            neg_m_idx = 4 - m_idx  # -m mapped to index: if m_idx=m+2, then -m+2=4-m_idx

            # Fortran: Mu1(ik) = PC(ik,-m,1,iscat)*PU(ik)
            Mu1 = sa.PC[:, neg_m_idx, 0, iscat] * PU
            Mu2 = sa.PC[:, neg_m_idx, 1, iscat] * SU

            # k→x transform
            Mu1 = _fft_k2x(Mu1, Nk)
            Mu2 = _fft_k2x(Mu2, Nk)

            # Sample at scatterer positions
            j = 0
            for jscat in range(grid.Nscatx):
                result.PSY[jscat, m_idx, 0, iscat] += Mu1[j]
                result.PSY[jscat, m_idx, 1, iscat] += Mu2[j]
                j += jskip

        # --- Scatter sources back to k-space (Fortran Do 60 m=-2,2) ---
        for m_idx in range(5):
            # Place scattering sources onto FFT grid
            SigPU = np.zeros(Nk, dtype=np.complex128)
            SigSU = np.zeros(Nk, dtype=np.complex128)

            j = 0
            for jscat in range(grid.Nscatx):
                SigPU[j] = result.SY[jscat, m_idx, 0, iscat]
                SigSU[j] = result.SY[jscat, m_idx, 1, iscat]
                j += jskip

            # x→k transform
            SigPU = _fft_x2k(SigPU)
            SigSU = _fft_x2k(SigSU)

            # Accumulate into upgoing field
            # Fortran: PU(ik) = PU(ik) + PC(ik,m,1,iscat)*SigPU(ik)
            PU = PU + sa.PC[:, m_idx, 0, iscat] * SigPU
            SU = SU + sa.PC[:, m_idx, 1, iscat] * SigSU

    return PU, SU


def free_surface_reflect(
    sa: SpectralArrays,
    PU: np.ndarray,
    SU: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply free-surface Rayleigh reflection.

    Faithful translation of Fortran lines 367–370.

    Converts upgoing (PU, SU) to downgoing (PD, SD) via:
        PD = Rpp*PU - Rsp*SU
        SD = Rsp*PU + Rss*SU

    Parameters
    ----------
    sa : SpectralArrays
        Contains Rpp, Rsp, Rss reflection coefficients.
    PU, SU : np.ndarray
        Upgoing P and S wavefields, shape (Nk,).

    Returns
    -------
    PD, SD : np.ndarray
        Downgoing P and S wavefields after free-surface reflection.
    """
    PD = sa.Rpp * PU - sa.Rsp * SU
    SD = sa.Rsp * PU + sa.Rss * SU
    return PD, SD


def downsweep(
    sa: SpectralArrays,
    grid: GridConfig,
    PD: np.ndarray,
    SD: np.ndarray,
    result: PropagationResult,
) -> None:
    """Downward sweep from shallowest to deepest scatterer layer.

    Faithful translation of Fortran loop 2000 (lines 372–429).

    Continues PSY accumulation with the free-surface-reflected field.
    """
    Nk = grid.Nk
    jskip = grid.jskip

    for iscat in range(grid.Nscatz):
        logger.debug(f"Downsweep: iscat={iscat + 1}")

        # --- Accumulate incoming field into PSY (Fortran Do 70) ---
        for m_idx in range(5):
            Md1 = sa.PC[:, m_idx, 0, iscat] * PD
            Md2 = sa.PC[:, m_idx, 1, iscat] * SD

            Md1 = _fft_k2x(Md1, Nk)
            Md2 = _fft_k2x(Md2, Nk)

            j = 0
            for jscat in range(grid.Nscatx):
                result.PSY[jscat, m_idx, 0, iscat] += Md1[j]
                result.PSY[jscat, m_idx, 1, iscat] += Md2[j]
                j += jskip

        # --- Scatter sources back to k-space (Fortran Do 80) ---
        # Note: Fortran uses PC(ik, -m, ...) for the back-projection
        for m_idx in range(5):
            neg_m_idx = 4 - m_idx

            SigPD = np.zeros(Nk, dtype=np.complex128)
            SigSD = np.zeros(Nk, dtype=np.complex128)

            j = 0
            for jscat in range(grid.Nscatx):
                SigPD[j] = result.SY[jscat, m_idx, 0, iscat]
                SigSD[j] = result.SY[jscat, m_idx, 1, iscat]
                j += jskip

            SigPD = _fft_x2k(SigPD)
            SigSD = _fft_x2k(SigSD)

            # Fortran: PD(ik) = PD(ik) + PC(ik,-m,...)*SigPD(ik)
            PD = PD + sa.PC[:, neg_m_idx, 0, iscat] * SigPD
            SD = SD + sa.PC[:, neg_m_idx, 1, iscat] * SigSD

        # Phase advance
        PD = sa.Eavec * PD
        SD = sa.Ebvec * SD


def right_sweep(
    sa: SpectralArrays,
    grid: GridConfig,
    iscat: int,
    result: PropagationResult,
) -> None:
    """Right-to-left horizontal sweep at a single depth layer.

    Faithful translation of Fortran loop 3000 (lines 445–469).

    Note: Uses stride-2 wavenumber sampling (Fortran: Do ik=1,Nk,2),
    processing only odd-indexed (Fortran 1-based) = even-indexed (Python
    0-based) wavenumber components.

    Parameters
    ----------
    sa : SpectralArrays
        Precomputed spectral arrays.
    grid : GridConfig
        Grid configuration.
    iscat : int
        Depth layer index (0-based).
    result : PropagationResult
        Contains SY (scattering sources) and PSY (accumulated field).
    """
    Nk = grid.Nk

    # Initialise upgoing field (stride 2) (Fortran lines 438–441)
    PU = np.zeros(Nk, dtype=np.complex128)
    SU = np.zeros(Nk, dtype=np.complex128)

    # Right sweep: jscat = 1 to Nscatx (Fortran Do 3000)
    for jscat in range(grid.Nscatx):
        logger.debug(f"Right sweep: jscat={jscat + 1}")

        # Phase advance (stride 2)
        PU[::2] = sa.Eavec[::2] * PU[::2]
        SU[::2] = sa.Ebvec[::2] * SU[::2]

        # --- Accumulate into PSY (Fortran Do 90 m=-2,2) ---
        for m_idx in range(5):
            m = m_idx - 2  # actual m value
            neg_m_idx = 4 - m_idx

            # Fortran: im = 2.0 * ((Ci)**m)
            im_factor = 2.0 * (1j**m)

            # Fortran: PSY(jscat,m,1,iscat) += im*PC(ik,-m,1,iscat)*PU(ik)
            result.PSY[jscat, m_idx, 0, iscat] += np.sum(
                im_factor * sa.PC[::2, neg_m_idx, 0, iscat] * PU[::2]
            )
            result.PSY[jscat, m_idx, 1, iscat] += np.sum(
                im_factor * sa.PC[::2, neg_m_idx, 1, iscat] * SU[::2]
            )

        # --- Scatter from SY into wavefield (Fortran Do 30 m=-2,2) ---
        for m_idx in range(5):
            m = m_idx - 2

            # Fortran: im = (-Ci)**m
            im_factor = (-1j) ** m

            # Fortran: PU(ik) += im*PC(ik,m,1,iscat)*SY(jscat,m,1,iscat)
            PU[::2] += (
                im_factor
                * sa.PC[::2, m_idx, 0, iscat]
                * result.SY[jscat, m_idx, 0, iscat]
            )
            SU[::2] += (
                im_factor
                * sa.PC[::2, m_idx, 1, iscat]
                * result.SY[jscat, m_idx, 1, iscat]
            )


def left_sweep(
    sa: SpectralArrays,
    grid: GridConfig,
    iscat: int,
    result: PropagationResult,
) -> None:
    """Left-to-right horizontal sweep at a single depth layer.

    Faithful translation of Fortran loop 4000 (lines 477–506).

    Same stride-2 convention as right_sweep.

    Parameters
    ----------
    sa : SpectralArrays
        Precomputed spectral arrays.
    grid : GridConfig
        Grid configuration.
    iscat : int
        Depth layer index (0-based).
    result : PropagationResult
        Contains SY (scattering sources) and PSY (accumulated field).
    """
    Nk = grid.Nk

    # Initialise downgoing field (Fortran lines 471–474)
    PD = np.zeros(Nk, dtype=np.complex128)
    SD = np.zeros(Nk, dtype=np.complex128)

    # Left sweep: jscat = Nscatx down to 1 (Fortran Do 4000 jscat=Nscatx,1,-1)
    for jscat in range(grid.Nscatx - 1, -1, -1):
        logger.debug(f"Left sweep: jscat={jscat + 1}")

        # --- Accumulate into PSY (Fortran Do 110 m=-2,2) ---
        for m_idx in range(5):
            m = m_idx - 2

            # Fortran: im = 2.0 * ((Ci)**m)
            im_factor = 2.0 * (1j**m)

            # Fortran: PSY(jscat,m,1,iscat) += im*PC(ik,m,1,iscat)*PD(ik)
            # NOTE: For left sweep, Fortran uses PC(ik, m, ...) not PC(ik, -m, ...)
            result.PSY[jscat, m_idx, 0, iscat] += np.sum(
                im_factor * sa.PC[::2, m_idx, 0, iscat] * PD[::2]
            )
            result.PSY[jscat, m_idx, 1, iscat] += np.sum(
                im_factor * sa.PC[::2, m_idx, 1, iscat] * SD[::2]
            )

        # --- Scatter from SY into wavefield (Fortran Do 120 m=-2,2) ---
        for m_idx in range(5):
            m = m_idx - 2
            neg_m_idx = 4 - m_idx

            # Fortran: im = (-Ci)**m
            im_factor = (-1j) ** m

            # Fortran: PD(ik) += im*PC(ik,-m,1,iscat)*SY(jscat,m,1,iscat)
            PD[::2] += (
                im_factor
                * sa.PC[::2, neg_m_idx, 0, iscat]
                * result.SY[jscat, m_idx, 0, iscat]
            )
            SD[::2] += (
                im_factor
                * sa.PC[::2, neg_m_idx, 1, iscat]
                * result.SY[jscat, m_idx, 1, iscat]
            )

        # Phase advance (stride 2)
        PD[::2] = sa.Eavec[::2] * PD[::2]
        SD[::2] = sa.Ebvec[::2] * SD[::2]
