"""Reference medium and scatterer grid parameters.

Faithful translation of FFTPROP.F parameter block and DATA statements.

Fortran variables mapped:
    alpha, rho, Q  → ReferenceMedium
    Nk, Nscatx, Nscatz, jskip → GridConfig
    Xs, Xr, is, ir → SourceReceiverConfig
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "ReferenceMedium",
    "GridConfig",
    "SourceReceiverConfig",
    "default_medium",
    "default_grid",
    "default_source_receiver",
]


@dataclass
class ReferenceMedium:
    """Homogeneous elastic reference medium with constant-Q attenuation.

    Fortran DATA: alpha=5.0, rho=3.0, Q=50.0
    S-wave velocity: beta = alpha / sqrt(3)  (Poisson solid)

    Attributes
    ----------
    alpha : float
        P-wave velocity.
    rho : float
        Density.
    Q : float
        Quality factor for attenuation.
    """

    alpha: float
    rho: float
    Q: float

    # --- derived complex quantities (Fortran lines 46–56) ---

    @property
    def alpha_imag(self) -> float:
        """Fortran: ai = alpha / (2.0*Q)."""
        return self.alpha / (2.0 * self.Q)

    @property
    def complex_alpha(self) -> complex:
        """Complex P-velocity a0 = alpha + i*ai.

        Fortran: a0 = Cmplx(alpha, ai).
        """
        return complex(self.alpha, self.alpha_imag)

    @property
    def complex_slowness_p(self) -> complex:
        """Complex P-slowness ca0 = 1/a0.

        Fortran: ca0 = C1/a0.
        """
        return 1.0 / self.complex_alpha

    @property
    def complex_slowness_s(self) -> complex:
        """Complex S-slowness cb0 = ca0/sqrt(3).

        Fortran: cb0 = ca0/sqrt(3.0).
        """
        return self.complex_slowness_p / np.sqrt(3.0)

    def complex_omega(self, freq: float, atten_imag: float = 0.2) -> complex:
        """Complex angular frequency.

        Fortran: w = Cmplx(2.0*Pi*freq, .2).

        Parameters
        ----------
        freq : float
            Frequency in Hz.
        atten_imag : float
            Imaginary part of omega (frequency damping), default 0.2.
        """
        return complex(2.0 * np.pi * freq, atten_imag)

    def ka0(self, freq: float, atten_imag: float = 0.2) -> complex:
        """Complex P-wavenumber ka0 = w * a0.

        Fortran: ka0 = w*a0.
        """
        return self.complex_omega(freq, atten_imag) * self.complex_alpha

    def kb0(self, freq: float, atten_imag: float = 0.2) -> complex:
        """Complex S-wavenumber kb0 = ka0 * sqrt(3).

        Fortran: kb0 = ka0*sqrt(3.0).
        """
        return self.ka0(freq, atten_imag) * np.sqrt(3.0)


@dataclass
class GridConfig:
    """Spatial and spectral grid configuration.

    Fortran PARAMETER: Nk=4096, Nscatx=81, Nscatz=2, jskip=8

    Attributes
    ----------
    Nk : int
        Number of FFT points (power of 2).
    Nscatx : int
        Number of horizontal scatterer positions.
    Nscatz : int
        Number of depth layers of scatterers.
    jskip : int
        Stride in FFT grid between adjacent scatterer positions.
    """

    Nk: int = 4096
    Nscatx: int = 81
    Nscatz: int = 2
    jskip: int = 8

    @property
    def Nh(self) -> int:
        """Fortran: Nh = Nk/2."""
        return self.Nk // 2

    @property
    def Nhm1(self) -> int:
        """Fortran: Nhm1 = Nh - 1."""
        return self.Nh - 1


@dataclass
class SourceReceiverConfig:
    """Source and receiver parameters.

    Fortran DATA: Xs=10.0, Xr=0.0, is=1, ir=1

    Attributes
    ----------
    Xs : float
        Source horizontal position.
    Xr : float
        Receiver horizontal position.
    is_type : int
        Source type (1=P, 2=SV).
    ir_type : int
        Receiver type (1=P, 2=SV).
    """

    Xs: float = 10.0
    Xr: float = 0.0
    is_type: int = 1
    ir_type: int = 1


def default_medium() -> ReferenceMedium:
    """Default reference medium from FFTPROP.F.

    Homogeneous half-space: alpha=5.0, rho=3.0, Q=50.0,
    beta = alpha/sqrt(3) ≈ 2.887 (Poisson solid).
    """
    return ReferenceMedium(alpha=5.0, rho=3.0, Q=50.0)


def default_grid() -> GridConfig:
    """Default grid from FFTPROP.F: Nk=4096, Nscatx=81, Nscatz=2, jskip=8."""
    return GridConfig(Nk=4096, Nscatx=81, Nscatz=2, jskip=8)


def default_source_receiver() -> SourceReceiverConfig:
    """Default source/receiver from FFTPROP.F: Xs=10, Xr=0, P-source, P-receiver."""
    return SourceReceiverConfig(Xs=10.0, Xr=0.0, is_type=1, ir_type=1)
