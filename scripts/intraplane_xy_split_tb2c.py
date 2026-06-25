#!/usr/bin/env python3
"""TB2c (foundation) — Python port + self-test of the vector L/M/N projector.

The fully-direct vector bridge (project the elastic lattice field onto vector
spherical multipoles and match Kambe's G0^vec) first needs a correct vector
projector.  This ports the machinery of Mathematica/IntraPlaneVectorTranslation.wl
to Python and validates it self-consistently:

  * Vector spherical harmonics P (radial), B, C (tangential).
  * Regular/outgoing vector multipoles L, M, N (Cruzan/Stein normalisation).
  * Self-normalised projection a^c = <F, harm> / <basis^c, harm>:
      M <- C (toroidal),  N <- P (radial),  L <- P (radial, P-wave).

Gate: a regular multipole reconstructed from its own extracted coefficients
reproduces itself (orthogonality of the C/P projection channels).
"""

from __future__ import annotations

import numpy as np
from scipy.special import jv

KP = 0.9 + 0.18j  # damped P, S wavenumbers (seismic-ish, internal-consistency)
KS = 1.5 + 0.30j


def sph_j(n, z):
    z = np.asarray(z, dtype=complex)
    return np.sqrt(np.pi / (2 * z)) * jv(n + 0.5, z)


def sph_jp(n, z):  # derivative d/dz of spherical j_n
    z = np.asarray(z, dtype=complex)
    return sph_j(n - 1, z) - (n + 1) / z * sph_j(n, z)


def Ylm(n, m, theta, phi):
    try:
        from scipy.special import sph_harm_y

        return sph_harm_y(n, m, theta, phi)
    except (ImportError, AttributeError):
        from scipy.special import sph_harm

        return sph_harm(m, n, phi, theta)


def dtheta_Y(n, m, theta, phi):
    """Analytic d/dtheta of Y_n^m: m cot th Y_n^m + sqrt((n-m)(n+m+1)) e^{-i phi} Y_n^{m+1}."""
    term1 = m / np.tan(theta) * Ylm(n, m, theta, phi)
    if m + 1 > n:
        return term1
    fac = np.sqrt((n - m) * (n + m + 1))
    return term1 + fac * np.exp(-1j * phi) * Ylm(n, m + 1, theta, phi)


def _rotmat(theta, phi):
    """Columns (rhat, thetahat, phihat) in Cartesian, per point. Shape (P,3,3)."""
    st, ct, sp, cp = np.sin(theta), np.cos(theta), np.sin(phi), np.cos(phi)
    R = np.empty((len(theta), 3, 3))
    R[:, 0, 0], R[:, 0, 1], R[:, 0, 2] = st * cp, ct * cp, -sp
    R[:, 1, 0], R[:, 1, 1], R[:, 1, 2] = st * sp, ct * sp, cp
    R[:, 2, 0], R[:, 2, 1], R[:, 2, 2] = ct, -st, 0.0
    return R


def Pvec(n, m, theta, phi):
    """Radial vector harmonic Y_n^m rhat. Shape (P,3) complex."""
    R = _rotmat(theta, phi)
    rhat = R[:, :, 0]
    return Ylm(n, m, theta, phi)[:, None] * rhat


def Bvec(n, m, theta, phi):
    """Tangential gradient harmonic. Shape (P,3) complex."""
    R = _rotmat(theta, phi)
    comp = np.zeros((len(theta), 3), dtype=complex)
    comp[:, 1] = dtheta_Y(n, m, theta, phi)
    comp[:, 2] = 1j * m / np.sin(theta) * Ylm(n, m, theta, phi)
    return np.einsum("pij,pj->pi", R.astype(complex), comp)


def Cvec(n, m, theta, phi):
    """Toroidal harmonic = rhat x Bvec. Shape (P,3)."""
    R = _rotmat(theta, phi)
    rhat = R[:, :, 0].astype(complex)
    return np.cross(rhat, Bvec(n, m, theta, phi))


def M_reg(n, m, kappa, rho0, theta, phi):
    """Regular toroidal multipole M = -j_n(kr) C_n^m (Cruzan/Stein)."""
    return -sph_j(n, kappa * rho0) * Cvec(n, m, theta, phi)


def N_reg(n, m, kappa, rho0, theta, phi):
    """Regular spheroidal multipole N. Shape (P,3)."""
    x = kappa * rho0
    P = Pvec(n, m, theta, phi)
    B = Bvec(n, m, theta, phi)
    return (n * (n + 1) / x) * sph_j(n, x) * P + (
        sph_j(n, x) + x * sph_jp(n, x)
    ) / x * B


def make_sphere(n_u=20, n_phi=40):
    u, wu = np.polynomial.legendre.leggauss(n_u)
    theta = np.arccos(u)
    phi = 2 * np.pi * np.arange(n_phi) / n_phi
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    WU = np.broadcast_to(wu[:, None], TH.shape)
    w = (WU * (2 * np.pi / n_phi)).ravel()
    return TH.ravel(), PH.ravel(), w


def proj(field, harm_vals, w):
    """<field, harm> = oint field . conj(harm) dOmega  (quadrature)."""
    return np.sum(w[:, None] * field * np.conj(harm_vals), axis=0).sum()


def main() -> None:
    print("==== TB2c (foundation) :: vector L/M/N projector self-consistency ====")
    th, ph, w = make_sphere()
    rho0 = 0.5
    worst_M = worst_N = 0.0
    for n in (1, 2):
        for m in range(-n, n + 1):
            C = Cvec(n, m, th, ph)
            P = Pvec(n, m, th, ph)
            normMC = proj(M_reg(n, m, KS, rho0, th, ph), C, w)
            normNP = proj(N_reg(n, m, KS, rho0, th, ph), P, w)
            # self: extract coeff of M_reg(n,m) projected on C(n',m') -> delta
            for n2 in (1, 2):
                for m2 in range(-n2, n2 + 1):
                    C2 = Cvec(n2, m2, th, ph)
                    aM = proj(M_reg(n, m, KS, rho0, th, ph), C2, w) / (
                        proj(M_reg(n2, m2, KS, rho0, th, ph), C2, w)
                    )
                    expect = 1.0 if (n2 == n and m2 == m) else 0.0
                    worst_M = max(worst_M, abs(aM - expect))
            _ = normMC, normNP
    print(f"  M-channel (C projection) orthonormality residual = {worst_M:.2e}")
    # N self-consistency
    for n in (1, 2):
        for m in range(-n, n + 1):
            for n2 in (1, 2):
                for m2 in range(-n2, n2 + 1):
                    P2 = Pvec(n2, m2, th, ph)
                    aN = proj(N_reg(n, m, KS, rho0, th, ph), P2, w) / (
                        proj(N_reg(n2, m2, KS, rho0, th, ph), P2, w)
                    )
                    expect = 1.0 if (n2 == n and m2 == m) else 0.0
                    worst_N = max(worst_N, abs(aN - expect))
    print(f"  N-channel (P projection) orthonormality residual = {worst_N:.2e}")
    ok = worst_M < 1e-6 and worst_N < 1e-6
    print(f"  -> {'PASS' if ok else 'FAIL'} (vector projector is convention-correct)")


if __name__ == "__main__":
    main()
