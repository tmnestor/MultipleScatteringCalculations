"""Spectral grid, phase factors, free-surface reflection, and PC transforms.

Faithful translation of FFTPROP.F array initialization (lines 58–193).

Fortran subroutine / variable mapping:
    VERTICALSLO(K, Kx)    → vertical_slowness(k, kx)
    kxvec, kzavec, kzbvec → SpectralArrays.kxvec, .kzavec, .kzbvec
    rtEa, rtEb, Eavec, Ebvec → SpectralArrays.rtEa, .rtEb, .Eavec, .Ebvec
    Rpp, Rsp, Rss         → SpectralArrays.Rpp, .Rsp, .Rss
    W11, W12, W21, W22    → SpectralArrays.W11, .W12, .W21, .W22
    PC(Nk,-2:2,2,Nscatz)  → SpectralArrays.PC  shape (Nk, 5, 2, Nscatz)
    fPC(Kx,Kz,dkx,Ka0,m)  → (inlined in PC construction)

Bugs fixed from Fortran:
    1. Lines 190-193: Loop variable `i` but body uses `ik`.
       Fix: Compute Eavec = rtEa**2 directly (vectorised).
    2. Lines 148-150: Eavec/Ebvec used in free-surface Rsp/Rss BEFORE
       being computed at line 190.
       Fix: Compute Eavec/Ebvec before free-surface section.

Note on branch cut convention:
    FFTPROP.F VERTICALSLO uses strict .LT. (less than) for the imaginary
    part check, unlike kennetslo.f which uses .LE. (less than or equal).
    We preserve this difference.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .medium import GridConfig, ReferenceMedium

__all__ = [
    "vertical_slowness",
    "SpectralArrays",
    "build_spectral_arrays",
]

logger = logging.getLogger(__name__)


def vertical_slowness(k: complex, kx: np.ndarray) -> np.ndarray:
    """Compute vertical wavenumber kz = sqrt(k² - kx²).

    Faithful translation of FFTPROP.F VERTICALSLO function.

    Parameters
    ----------
    k : complex
        Complex total wavenumber (ka0 or kb0).
    kx : np.ndarray
        Horizontal wavenumber array, shape (Nk,).

    Returns
    -------
    np.ndarray
        Vertical wavenumber array with Im(kz) >= 0, shape (Nk,).

    Notes
    -----
    Fortran:
        T = CMPLX((K+Kx)*(K-Kx))
        VERTICALSLO = csqrt(T)
        IF(AIMAG(VERTICALSLO) .LT. 0.0) VERTICALSLO = -VERTICALSLO

    Uses strict less-than (.LT.) for the branch cut, NOT less-than-or-equal.
    """
    kx = np.asarray(kx, dtype=np.complex128)
    T = (k + kx) * (k - kx)
    kz = np.sqrt(T)
    # Fortran: .LT. 0.0 (strict less than)
    mask = kz.imag < 0.0
    kz[mask] = -kz[mask]
    return kz


@dataclass
class SpectralArrays:
    """All precomputed spectral arrays for FFTPROP.

    Groups wavenumber grid, vertical wavenumbers, phase factors,
    free-surface Rayleigh reflection, W source-coupling matrices,
    and plane-wave ↔ cylindrical-harmonic (PC) transform arrays.

    Attributes
    ----------
    kxvec : np.ndarray
        FFT-ordered horizontal wavenumber, shape (Nk,).
    kzavec : np.ndarray
        Vertical P-wavenumber, shape (Nk,).
    kzbvec : np.ndarray
        Vertical S-wavenumber, shape (Nk,).
    rtEa, rtEb : np.ndarray
        Half-step phase factors exp(i·kz·zh), shape (Nk,).
    Eavec, Ebvec : np.ndarray
        Full-step phase factors exp(i·kz·z) = rtE², shape (Nk,).
    Rpp, Rsp, Rss : np.ndarray
        Free-surface Rayleigh reflection coefficients, shape (Nk,).
        Include phase factors Eavec/Ebvec per Fortran convention.
    W11, W12, W21, W22 : np.ndarray
        Source coupling through free surface, shape (Nk,).
    PC : np.ndarray
        Plane-wave ↔ cylindrical-harmonic transform.
        Shape (Nk, 5, 2, Nscatz). Index 1 maps m=-2..+2 → 0..4.
        Index 2: 0=P, 1=SV. Index 3: depth layer.
    dkx : float
        Wavenumber spacing.
    dx : float
        Spatial grid spacing.
    z : float
        Scatterer vertical spacing (= jskip * dx).
    intfac : float
        Integration factor sqrt(dkx / (2π)).
    ka0 : complex
        Complex P-wavenumber.
    kb0 : complex
        Complex S-wavenumber.
    omega : complex
        Complex angular frequency.
    """

    # Wavenumber grid
    kxvec: np.ndarray
    kzavec: np.ndarray
    kzbvec: np.ndarray

    # Phase factors
    rtEa: np.ndarray
    rtEb: np.ndarray
    Eavec: np.ndarray
    Ebvec: np.ndarray

    # Free-surface reflection
    Rpp: np.ndarray
    Rsp: np.ndarray
    Rss: np.ndarray
    W11: np.ndarray
    W12: np.ndarray
    W21: np.ndarray
    W22: np.ndarray

    # Plane-cylindrical transform
    PC: np.ndarray

    # Scalar parameters
    dkx: float
    dx: float
    z: float
    intfac: float
    ka0: complex
    kb0: complex
    omega: complex


def build_spectral_arrays(
    medium: ReferenceMedium,
    grid: GridConfig,
    freq: float = 2.0,
    atten_imag: float = 0.2,
) -> SpectralArrays:
    """Build all spectral arrays for FFTPROP computation.

    Faithful translation of FFTPROP.F initialisation (lines 58–193).

    Parameters
    ----------
    medium : ReferenceMedium
        Reference medium properties.
    grid : GridConfig
        Grid configuration.
    freq : float
        Frequency in Hz, default 2.0.
    atten_imag : float
        Imaginary part of complex omega, default 0.2.

    Returns
    -------
    SpectralArrays
        All precomputed arrays ready for propagation sweeps.
    """
    Nk = grid.Nk
    Nh = grid.Nh
    Nhm1 = grid.Nhm1
    Nscatz = grid.Nscatz

    # Complex wavenumbers (Fortran lines 46–56)
    w = medium.complex_omega(freq, atten_imag)
    ka0 = medium.ka0(freq, atten_imag)
    kb0 = medium.kb0(freq, atten_imag)
    cb0 = medium.complex_slowness_s  # Fortran: cb0 = ca0/sqrt(3)

    logger.info(f"w = {w}")
    logger.info(f"ka0 = {ka0}")
    logger.info(f"kb0 = {kb0}")

    # Phase space dimensions (Fortran lines 60–84)
    wavelength = 2.0 * np.pi / abs(ka0)
    logger.info(f"lambda = {wavelength}")

    kxmax = 32.0 * ka0.real  # Fortran: kxmax = 32.*Real(ka0)
    dkx = 2.0 * kxmax / float(Nk)  # Fortran: dkx = 2.*kxmax/float(Nk)
    intfac = np.sqrt(dkx / (2.0 * np.pi))
    dx = np.pi / kxmax  # Fortran: dx = Pi/kxmax
    z = float(grid.jskip) * dx  # Fortran: z = float(jskip)*dx

    logger.info(f"kxmax = {kxmax}")
    logger.info(f"kxmax/real(ka0) = {kxmax / ka0.real}")
    logger.info(f"dx = {dx}")
    logger.info(f"dkx = {dkx}")
    logger.info(f"lambda/z = {wavelength / z}")
    logger.info(f"z/dx = {z / dx}")
    logger.info(f"lambda/dx = {wavelength / dx}")
    logger.info(f"G = {2.0 * np.pi / (z * ka0.real)}")

    # --- FFT-ordered kx grid (Fortran lines 118–123) ---
    # Fortran 1-based: kxvec(1)=0, kxvec(i+1)=i*dkx, kxvec(i+1+Nh)=(i-Nh)*dkx
    # Python 0-based: kxvec[0]=0, kxvec[i]=i*dkx, kxvec[i+Nh]=(i-Nh)*dkx
    kxvec = np.zeros(Nk, dtype=np.complex128)
    for i in range(1, Nhm1 + 1):
        kxvec[i] = float(i) * dkx
        kxvec[i + Nh] = float(i - Nh) * dkx
    kxvec[0] = 0.0
    kxvec[Nh] = -Nh * dkx

    # --- Vertical wavenumbers and half-step phase (Fortran lines 125–131) ---
    zh = 0.5 * z
    kzavec = vertical_slowness(ka0, kxvec)
    kzbvec = vertical_slowness(kb0, kxvec)
    rtEa = np.exp(1j * kzavec * abs(zh))
    rtEb = np.exp(1j * kzbvec * abs(zh))

    # --- Full-step phase factors (Fortran lines 190–193, BUG FIXED) ---
    # Original Fortran has loop var `i` but uses `ik` in body.
    # Correct: Eavec = rtEa², Ebvec = rtEb²
    Eavec = rtEa * rtEa
    Ebvec = rtEb * rtEb

    # --- Free-surface Rayleigh reflection (Fortran lines 134–156) ---
    # NOTE: In Fortran, Eavec/Ebvec are used here (line 148) before being
    # computed (line 190). We fix this by computing them above.
    cb02 = cb0**2  # Fortran: cb02 = cb0**2
    cb04 = cb02**2  # Fortran: cb04 = cb02**2

    p = kxvec / w  # Fortran: p = kxvec(ik)/w
    p2 = p * p  # Fortran: p2 = p*p
    qa = kzavec / w  # Fortran: qa = kzavec(ik)/w
    qb = kzbvec / w  # Fortran: qb = kzbvec(ik)/w
    twortqa = np.sqrt(2.0 * qa)  # Fortran: CSqrt(2.0*qa)
    twortqb = np.sqrt(2.0 * qb)  # Fortran: CSqrt(2.0*qb)

    R1 = 1.0 - 2.0 * cb02 * p2  # Fortran: R1 = (1.0 - 2*cb02*p2)
    R12 = R1**2  # Fortran: R12 = R1**2
    R2 = 4.0 * cb04 * p2 * qa * qb  # Fortran: R2 = 4.0*cb04*p2*qa*qb
    Rayleigh = R12 + R2  # Fortran: Rayleigh = (R12 + R2)

    # IMPORTANT: Fortran ordering matters here.
    # Line 147: Rpp(ik) = (-R12 + R2) / Rayleigh       → base Rpp
    # Line 148: Rsp(ik) = ... * Eavec * Ebvec / Rayleigh
    # Line 149: Rss(ik) = Ebvec(ik) * Rpp(ik)          → uses BASE Rpp
    # Line 150: Rpp(ik) = Eavec(ik) * Rpp(ik)          → modifies Rpp
    Rpp_base = (-R12 + R2) / Rayleigh
    Rsp = -4.0 * cb04 * p * R1 * np.sqrt(qa * qb) * Eavec * Ebvec / Rayleigh
    Rss = Ebvec * Rpp_base  # uses base Rpp (before Eavec multiply)
    Rpp = Eavec * Rpp_base  # now modified Rpp

    # W source-coupling matrices (Fortran lines 151–154)
    sqrt_rho = np.sqrt(medium.rho)
    W11 = -1j * twortqa * R1 / (Rayleigh * sqrt_rho)
    W12 = -2.0 * 1j * cb02 * twortqa * qb * p / (Rayleigh * sqrt_rho)
    W21 = -2.0 * 1j * cb02 * qa * twortqb * p / (Rayleigh * sqrt_rho)
    W22 = 1j * twortqb * R1 / (Rayleigh * sqrt_rho)

    # --- Plane-wave ↔ cylindrical-harmonic transform (Fortran lines 158–185) ---
    # PC(ik, m, wavetype, depth) where m ∈ {-2,-1,0,+1,+2}
    # Python indexing: m stored as m+2 → indices 0..4
    # Wavetype: 0=P, 1=SV (Fortran: 1=P, 2=SV)
    PC = np.zeros((Nk, 5, 2, Nscatz), dtype=np.complex128)

    for i_depth in range(Nscatz):
        # P-wave (Fortran wavetype=1, Python index=0)
        # Fortran lines 161–171
        kz_p = kzavec
        PC[:, 2, 0, i_depth] = np.sqrt(dkx / (np.pi * kz_p))  # m=0

        temp_pos = (kxvec / ka0) + 1j * (kz_p / ka0)
        PC[:, 3, 0, i_depth] = PC[:, 2, 0, i_depth] * temp_pos  # m=+1
        PC[:, 4, 0, i_depth] = PC[:, 3, 0, i_depth] * temp_pos  # m=+2

        temp_neg = (kxvec / ka0) - 1j * (kz_p / ka0)
        PC[:, 1, 0, i_depth] = PC[:, 2, 0, i_depth] * temp_neg  # m=-1
        PC[:, 0, 0, i_depth] = PC[:, 1, 0, i_depth] * temp_neg  # m=-2

        # SV-wave (Fortran wavetype=2, Python index=1)
        # Fortran lines 174–184
        kz_s = kzbvec
        PC[:, 2, 1, i_depth] = np.sqrt(dkx / (np.pi * kz_s))  # m=0

        temp_pos = (kxvec / kb0) + 1j * (kz_s / kb0)
        PC[:, 3, 1, i_depth] = PC[:, 2, 1, i_depth] * temp_pos  # m=+1
        PC[:, 4, 1, i_depth] = PC[:, 3, 1, i_depth] * temp_pos  # m=+2

        temp_neg = (kxvec / kb0) - 1j * (kz_s / kb0)
        PC[:, 1, 1, i_depth] = PC[:, 2, 1, i_depth] * temp_neg  # m=-1
        PC[:, 0, 1, i_depth] = PC[:, 1, 1, i_depth] * temp_neg  # m=-2

    logger.info(
        f"Spectral arrays built: Nk={Nk}, Nscatz={Nscatz}, "
        f"dkx={dkx:.6f}, dx={dx:.6f}, z={z:.6f}"
    )

    return SpectralArrays(
        kxvec=kxvec,
        kzavec=kzavec,
        kzbvec=kzbvec,
        rtEa=rtEa,
        rtEb=rtEb,
        Eavec=Eavec,
        Ebvec=Ebvec,
        Rpp=Rpp,
        Rsp=Rsp,
        Rss=Rss,
        W11=W11,
        W12=W12,
        W21=W21,
        W22=W22,
        PC=PC,
        dkx=dkx,
        dx=dx,
        z=z,
        intfac=intfac,
        ka0=ka0,
        kb0=kb0,
        omega=w,
    )
