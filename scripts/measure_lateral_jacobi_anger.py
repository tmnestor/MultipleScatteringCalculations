"""Measure: the azimuthal integrals S_n(q; Delta) of the cells' lateral moments by the Jacobi-Anger expansion.

Paper: LatexPDFs/InterfaceCouplingIntegrals, Section "Verification" (the wavenumber route).
Plan: docs/2026-10-10-stratified-reference-legendre-cells-3d.md (12 October: checked, not built).

S_n(q; Delta) = (1/4 pi^2) int dphi  L_a(k) L_c(k) e^{i k.Delta} e^{i n phi}, |n| <= 4, the lateral moments of a
field function and a source monomial (``interface_spectral._lateral_harmonics_at``, a trapezoid rule whose size
grows with q |Delta|).  With e^{i k.Delta} = sum_p i^p J_p(q |Delta|) e^{i p (phi - phi_Delta)},

    S_n(q; Delta) = (1/2 pi) sum_p i^p J_p(q |Delta|) e^{-i p phi_Delta} c_{-n-p}(q),

c_m(q) the azimuthal Fourier coefficients of L_a L_c.  The product of the lateral moments is band-limited in m by
the cell, not by the offset, so the c_m are computed once for all offsets.

Part 1: the bandwidth of c_m (largest |m| with |c_m| above a tolerance, relative to the largest).
Part 2: S_n by the expansion against the direct rule, 4 field functions x 10 source monomials, error relative
to max |S_n|.  Expected (12 October 2026): |m| <= 13, 32, 51, 84, 148 at qh = 1, 5, 10, 20, 40 (tolerance
1e-12); agreement 4e-16 .. 3e-13.

Run:  PYTHONPATH=. python scripts/measure_lateral_jacobi_anger.py
"""

import numpy as np
from scipy.special import jv

from cubic_scattering import interface_spectral as isp

H = 0.05


def coefficients(qh: float, n_phi: int, n_test: int, n_source: int) -> np.ndarray:
    """c_m of L_a L_c at q h on an n_phi-point azimuthal grid, indexed by m mod n_phi, shape (n_phi, a, c)."""
    phi = 2 * np.pi * np.arange(n_phi) / n_phi
    q = qh / H
    kx, ky = q * np.cos(phi), q * np.sin(phi)
    ef = isp.SOURCE_EXPONENTS_QUARTIC[:n_test]
    es = isp.SOURCE_EXPONENTS_QUARTIC[:n_source]
    f = (
        isp._lateral(kx, ky, H, ef, True)[:, :, None]
        * isp._lateral(kx, ky, H, es, False)[:, None, :]
    )
    return np.fft.fft(f, axis=0) / n_phi


def bandwidth() -> None:
    print("Part 1: bandwidth of c_m (10 field functions x 35 source monomials)")
    for qh in (1.0, 5.0, 10.0, 20.0, 40.0):
        n = 1024
        c = coefficients(qh, n, 10, 35)
        mag = np.abs(c).max(axis=(1, 2))
        mag /= mag.max()
        m = np.abs(np.fft.fftfreq(n, 1 / n).astype(int))
        print(
            f"  qh = {qh:4.0f}: |m| <= {m[mag > 1e-12].max():3d} (1e-12), "
            f"bound 3.5 qh + 20 = {3.5 * qh + 20:.0f}"
        )


def harmonics(qh: float, off: tuple, n_test: int = 4, n_source: int = 10) -> np.ndarray:
    """S_n by the Jacobi-Anger expansion, shape (9, n_test, n_source)."""
    big_m = int(3.5 * qh + 30)
    n = 4 * big_m
    c = coefficients(qh, n, n_test, n_source)
    dist, ang = np.hypot(*off), np.arctan2(off[1], off[0])
    p = np.arange(-big_m - 4, big_m + 5)
    g = 1j**p * jv(p, qh * dist) * np.exp(-1j * p * ang)
    return np.stack([np.einsum("p,pac->ac", g, c[(-k - p) % n]) for k in isp._HARM]) / (
        2 * np.pi
    )


def agreement() -> None:
    print("Part 2: S_n by the expansion against the direct azimuthal rule (4 x 10)")
    ef, es = isp.SOURCE_EXPONENTS_QUARTIC[:4], isp.SOURCE_EXPONENTS_QUARTIC[:10]
    worst = 0.0
    for qh in (0.7, 4.3, 9.9):
        for off in ((0, 0), (2, 0), (6, -4), (24, 10), (40, 0)):
            ref = isp._lateral_harmonics_at(qh, off, H, ef, es)
            err = np.abs(harmonics(qh, off) - ref).max() / np.abs(ref).max()
            worst = max(worst, err)
            print(f"  qh = {qh}, offset {off} h: {err:.1e}")
    print(f"  worst {worst:.1e}")


if __name__ == "__main__":
    bandwidth()
    agreement()
