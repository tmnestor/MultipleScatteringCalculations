"""Phase 3b cycle 2: undamped vector G0 cross-check.

``Mathematica/IntraPlaneKambeVector.wl`` builds the undamped vector coupling G0^vec by
contracting the cycle-1 scalar structure constants D[q,s] (L via kappa_P, M/N via
extracted coeff_q at kappa_S) and dumps it to ``IntraPlaneKambeVector_reference.json``.
This module independently recomputes the scalar D[q,s] (general-z Ewald + multipole
projection, scipy) and the L-block scalar-Gaunt contraction, asserts the L sub-block of
the dump matches, and checks the dump's structural invariants (L-block reciprocity,
finite coupling, isolated-limit residual).
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.special import sph_harm_y, spherical_jn, wofz
from sympy.physics.wigner import gaunt as _wig_gaunt

REF = (
    Path(__file__).resolve().parents[2]
    / "Mathematica"
    / "IntraPlaneKambeVector_reference.json"
)


@pytest.fixture(scope="module")
def dump():
    assert REF.exists(), f"missing {REF} (run IntraPlaneKambeVector.wl first)"
    return json.loads(REF.read_text())


def _cerfc(z):
    z = np.asarray(z, dtype=complex)
    return np.exp(-(z**2)) * wofz(1j * z)


def _ewald_total_z(kappa, r, eta, aL, kx, ky, Rc, Gc):
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
    return real + rec - np.exp(1j * kappa * rn) / (4 * math.pi * rn)


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
    theta = math.acos(d[2] / np.linalg.norm(d))
    phi = math.atan2(d[1], d[0])
    return sph_harm_y(q, s, theta, phi)


def _Dstruct(field, q, s, kappa, rho0):
    pts, wts = _sph_pts(rho0)
    integ = sum(
        w * field(p) * np.conj(_Yc(q, -s, p)) for p, w in zip(pts, wts, strict=False)
    )
    return (-1) ** s * integ / (1j * kappa * spherical_jn(q, kappa * rho0))


def _gaunt(l1, m1, l2, m2, l3, m3):
    if m1 + m2 + m3 != 0 or abs(m1) > l1 or abs(m2) > l2 or abs(m3) > l3:
        return 0.0
    return float(_wig_gaunt(l1, l2, l3, m1, m2, m3))


def _g0LL(n, m, nu, mu, Dfun):
    tot = 0.0 + 0j
    for q in range(abs(n - nu), n + nu + 1):
        tot += (
            1j ** (nu + q - n)
            * (-1) ** q
            * Dfun(q, m - mu)
            * _gaunt(n, m, nu, -mu, q, mu - m)
        )
    return 4 * math.pi * (-1) ** m * tot


def test_L_block_matches(dump):
    par = dump["params"]
    aL, kx, ky, kP, eta, rho0 = (
        par["aL"],
        par["kx"],
        par["ky"],
        par["kappaP"],
        par["eta"],
        par["rho0"],
    )
    idx = [tuple(e) for e in dump["idx"]]
    G0 = [[complex(*c) for c in row] for row in dump["G0vec"]]
    Dcache: dict[tuple[int, int], complex] = {}

    def Dfun(q, s):
        if (q, s) not in Dcache:
            Dcache[(q, s)] = _Dstruct(
                lambda r: _ewald_total_z(kP, r, eta, aL, kx, ky, 6, 6), q, s, kP, rho0
            )
        return Dcache[(q, s)]

    lpos = [i for i, e in enumerate(idx) if e[2] == "L"]
    for i in lpos:
        for j in lpos:
            ni, mi, _ = idx[i]
            nj, mj, _ = idx[j]
            mine = _g0LL(nj, mj, ni, mi, Dfun)  # receiver i <- source j
            assert abs(mine - G0[i][j]) < 1e-4, f"L G0[{idx[i]}<-{idx[j]}] != dump"


def test_dump_invariants(dump):
    """Structural-presence and magnitude-bound checks on the dump's reported invariants.

    Verifies that each key is present, is a finite real number, and falls within
    physics-meaningful bounds.  The substantive cross-language L-block validation
    lives in ``test_L_block_matches``; these checks guard the dump's summary
    statistics only.
    """
    for key in ("iso_dev", "recip_resid", "coupling"):
        assert key in dump, f"dump missing key '{key}'"
        assert math.isfinite(dump[key]), f"dump['{key}'] = {dump[key]!r} is not finite"

    iso_dev = dump["iso_dev"]
    assert iso_dev >= 0.0, f"iso_dev must be non-negative, got {iso_dev}"
    assert iso_dev < 1e-12, f"iso_dev = {iso_dev} exceeds threshold 1e-12"

    recip_resid = dump["recip_resid"]
    assert recip_resid >= 0.0, f"recip_resid must be non-negative, got {recip_resid}"
    assert recip_resid < 1e-9, f"recip_resid = {recip_resid} exceeds threshold 1e-9"

    coupling = dump["coupling"]
    assert 0.001 < coupling < 1.0, (
        f"coupling = {coupling} outside expected band (0.001, 1.0); "
        "actual collective effect should be ~0.02"
    )
