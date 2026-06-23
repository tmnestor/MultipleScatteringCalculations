"""Phase 3b cycle 1: undamped multipole structure constants D[q,s] cross-check.

``Mathematica/IntraPlaneKambe.wl`` computes the undamped (kappa real) planar lattice
structure constants D[q,s] by multipole-projecting the undamped scalar Ewald field, and
dumps them to ``IntraPlaneKambe_reference.json``.  This module independently recomputes the
general-z Ewald field and its multipole projection (scipy) and asserts: (1) the undamped
D[q,s] are eta-independent, (2) they match the dump.

Notes on the independent recompute (vs. the plan's reference code):
- The scalar Ewald erfc arguments are complex, so ``scipy.special.erfc`` (real-only) cannot
  be used; the complex complementary error function is evaluated via the Faddeeva function
  ``wofz`` (``erfc(z) = exp(-z**2) * w(i z)``).
- The lattice field is the R != 0 sum (the structure constant excludes the origin site), so
  the bare self-term ``g(r) = e^{i kappa |r|}/(4 pi |r|)`` is subtracted from the full Ewald
  sum, matching ``ewTot3`` in the Mathematica twin.
- The projection mirrors ``Dproj``: project onto ``conj(Y_q^{-s})`` then apply ``(-1)^s`` so
  the result is ``D_struct[q,s] = (-1)^s Dbar[q,-s]``.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.special import sph_harm_y, spherical_jn, wofz

REF = (
    Path(__file__).resolve().parents[2]
    / "Mathematica"
    / "IntraPlaneKambe_reference.json"
)


@pytest.fixture(scope="module")
def dump():
    assert REF.exists(), f"missing {REF} (run IntraPlaneKambe.wl first)"
    return json.loads(REF.read_text())


def _cerfc(z):
    """Complex complementary error function via the Faddeeva function.

    ``w(i z) = exp(z**2) erfc(z)`` so ``erfc(z) = exp(-z**2) w(i z)``; valid for complex z,
    unlike ``scipy.special.erfc`` which accepts real arguments only.
    """
    z = np.asarray(z, dtype=complex)
    return np.exp(-(z**2)) * wofz(1j * z)


def _ewald_total_z(kappa, r, eta, aL, kx, ky, Rc, Gc):
    """General-z scalar Ewald field over R != 0 (real + reciprocal halves), Bloch-phased."""
    x, y, z = r
    real = 0.0 + 0j
    for i in range(-Rc, Rc + 1):
        for j in range(-Rc, Rc + 1):
            d = math.hypot(math.hypot(x - aL * i, y - aL * j), z)
            ph = np.exp(1j * aL * (kx * i + ky * j))
            real += (
                ph
                / d
                * sum(
                    np.exp(s * 1j * kappa * d)
                    * _cerfc(d * eta + s * 1j * kappa / (2 * eta))
                    for s in (-1, 1)
                )
            )
    real *= 1.0 / (8 * math.pi)
    A = aL * aL
    recipB = 2 * math.pi / aL
    rec = 0.0 + 0j
    for m in range(-Gc, Gc + 1):
        for n in range(-Gc, Gc + 1):
            kpg = np.array([kx + recipB * m, ky + recipB * n])
            kz = np.sqrt(kappa**2 - kpg @ kpg + 0j)
            az = abs(z)
            rec += (
                np.exp(1j * (kpg[0] * x + kpg[1] * y))
                / kz
                * (
                    np.exp(-1j * kz * az) * _cerfc(az * eta + kz / (2j * eta))
                    + np.exp(1j * kz * az) * _cerfc(-az * eta + kz / (2j * eta))
                )
            )
    rec *= 1j / (4 * A)
    rn = math.sqrt(x * x + y * y + z * z)
    self_term = np.exp(1j * kappa * rn) / (4 * math.pi * rn)  # R=0 bare GF, excluded
    return real + rec - self_term


def _sph_pts(rho0, nu=16, nphi=32):
    nodes, w = np.polynomial.legendre.leggauss(nu)
    pts, wts = [], []
    for u, wu in zip(nodes, w, strict=False):
        st = math.sqrt(1 - u * u)
        for jj in range(nphi):
            ph = 2 * math.pi * jj / nphi
            pts.append(rho0 * np.array([st * math.cos(ph), st * math.sin(ph), u]))
            wts.append(wu * (2 * math.pi / nphi))
    return pts, wts


def _Yc(q, s, d):
    """Orthonormal spherical harmonic Y_q^s; scipy ``sph_harm_y(n, m, polar, azimuth)``."""
    theta = math.acos(d[2] / np.linalg.norm(d))  # polar
    phi = math.atan2(d[1], d[0])  # azimuth
    return sph_harm_y(q, s, theta, phi)


def _Dstruct_proj(field, q, s, kappa, rho0):
    pts, wts = _sph_pts(rho0)
    integ = sum(
        w * field(p) * np.conj(_Yc(q, -s, p)) for p, w in zip(pts, wts, strict=False)
    )
    return (-1) ** s * integ / (1j * kappa * spherical_jn(q, kappa * rho0))


def test_undamped_eta_independence(dump):
    par = dump["params"]
    aL, kx, ky, k = par["aL"], par["kx"], par["ky"], par["kappa"]
    for q in range(0, 4):
        for s in range(-q, q + 1):
            d1 = _Dstruct_proj(
                lambda r: _ewald_total_z(k, r, 0.7, aL, kx, ky, 6, 6),
                q,
                s,
                k,
                par["rho0"],
            )
            d2 = _Dstruct_proj(
                lambda r: _ewald_total_z(k, r, 1.15, aL, kx, ky, 6, 6),
                q,
                s,
                k,
                par["rho0"],
            )
            assert abs(d1 - d2) < 1e-5, f"D[{q},{s}] eta-dependent"


def test_matches_mathematica(dump):
    par = dump["params"]
    aL, kx, ky, k = par["aL"], par["kx"], par["ky"], par["kappa"]
    by = {(e["q"], e["s"]): complex(*e["val"]) for e in dump["Dstruct"]}
    for q in range(0, 4):
        for s in range(-q, q + 1):
            mine = _Dstruct_proj(
                lambda r: _ewald_total_z(k, r, 0.7, aL, kx, ky, 6, 6),
                q,
                s,
                k,
                par["rho0"],
            )
            assert abs(mine - by[(q, s)]) < 1e-4, f"D[{q},{s}] != Mathematica"
