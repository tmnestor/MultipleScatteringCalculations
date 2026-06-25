"""Independent Python cross-check of the Phase-0 elastic translation operator.

The Mathematica notebook ``Mathematica/IntraPlaneTranslation.wl`` builds the scalar
separation matrix ``beta^c_{nm,nu mu}(d)`` (singular->regular addition theorem) in
closed form and dumps it to ``Mathematica/IntraPlaneTranslation_reference.json``.

Here we recompute the same coefficients by two *independent* routes in Python and
assert they match Mathematica:

  1. spherical-harmonic PROJECTION quadrature (scipy only) -- a different method,
  2. the Gaunt / Wigner-3j CLOSED FORM (sympy) -- a cross-language check of the formula.

Conventions match CartesianT0.wl exactly: time e^{-i w t}, outgoing h_n^(1), orthonormal
spherical harmonics with Condon-Shortley phase, direction triples (x,y,z) with z the
polar (3rd) component; ang[d] = (arccos(d3/|d|), atan2(d2, d1)).

Run:  conda run -n seismic pytest cubic_scattering/tests/test_intraplane_translation.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import scipy.special as sp

REF_JSON = (
    Path(__file__).resolve().parents[2]
    / "Mathematica"
    / "IntraPlaneTranslation_reference.json"
)


# --------------------------------------------------------------------------- helpers
def ynm(n: int, m: int, theta, phi):
    """Orthonormal Condon-Shortley spherical harmonic; theta polar, phi azimuth."""
    try:  # scipy >= 1.15
        return sp.sph_harm_y(n, m, theta, phi)
    except AttributeError:  # classic signature sph_harm(m, n, azimuth, polar)
        return sp.sph_harm(m, n, phi, theta)


def jn(n: int, x):
    return sp.spherical_jn(n, x)


def hn(n: int, x):
    return sp.spherical_jn(n, x) + 1j * sp.spherical_yn(n, x)


def angles(v: np.ndarray):
    """(r, theta, phi) from a Cartesian triple, matching CartesianT0.wl ang[]."""
    r = np.sqrt(np.sum(v * v, axis=-1))
    theta = np.arccos(v[..., 2] / r)
    phi = np.arctan2(v[..., 1], v[..., 0])
    return r, theta, phi


@pytest.fixture(scope="module")
def ref():
    assert REF_JSON.exists(), (
        f"missing reference JSON: {REF_JSON} (run IntraPlaneTranslation.wl first)"
    )
    return json.loads(REF_JSON.read_text())


# --------------------------------------------------------------- projection quadrature
def _quad_grid(dvec: np.ndarray, rho: float, n_gl: int = 48, n_phi: int = 96):
    """Precompute the sphere quadrature grid about B=origin and the source-relative
    angles of p - A (A = dvec).  Returns everything reusable across (n,m,nu,mu)."""
    u, w = np.polynomial.legendre.leggauss(n_gl)  # nodes/weights on [-1, 1]
    phis = 2 * np.pi * np.arange(n_phi) / n_phi
    st = np.sqrt(1.0 - u * u)
    # shat[i, j] on the sphere; p = rho * shat ; rel = p - A
    shat = np.stack(
        [
            np.outer(st, np.cos(phis)),
            np.outer(st, np.sin(phis)),
            np.broadcast_to(u[:, None], (n_gl, n_phi)),
        ],
        axis=-1,
    )
    p = rho * shat
    rel = p - dvec  # A = dvec, B = origin
    r_src, th_src, ph_src = angles(rel)
    th_tgt = np.broadcast_to(np.arccos(u)[:, None], (n_gl, n_phi))
    ph_tgt = np.broadcast_to(phis[None, :], (n_gl, n_phi))
    weight = w[:, None] * (2 * np.pi / n_phi)
    return dict(
        r_src=r_src,
        th_src=th_src,
        ph_src=ph_src,
        th_tgt=th_tgt,
        ph_tgt=ph_tgt,
        weight=weight,
    )


def proj_beta(n, m, nu, mu, k, rho, grid):
    """beta by projecting the outgoing multipole (about A) onto Y_nu^mu on a sphere
    about B (origin); exact and independent of the closed form."""
    f = hn(n, k * grid["r_src"]) * ynm(n, m, grid["th_src"], grid["ph_src"])
    integrand = (
        grid["weight"] * f * np.conj(ynm(nu, mu, grid["th_tgt"], grid["ph_tgt"]))
    )
    return integrand.sum() / jn(nu, k * rho)


# --------------------------------------------------------------------- the cross-checks
def test_projection_matches_mathematica(ref):
    """scipy projection quadrature vs Mathematica closed form, for every dumped entry."""
    dvec = np.array(ref["dvec"], float)
    rho = 0.3 * np.linalg.norm(dvec)
    grid = _quad_grid(dvec, rho)
    worst = 0.0
    for key, k in (("betaP", ref["kP"]), ("betaS", ref["kS"])):
        for e in ref[key]:
            b_py = proj_beta(e["n"], e["m"], e["nu"], e["mu"], k, rho, grid)
            b_mma = complex(e["re"], e["im"])
            worst = max(worst, abs(b_py - b_mma))
    print(
        f"\n  max |beta_proj(py) - beta_closed(mma)| over {len(ref['betaP']) + len(ref['betaS'])} entries = {worst:.3e}"
    )
    assert worst < 1e-8


def test_closedform_matches_mathematica(ref):
    """Independent Python Gaunt/Wigner-3j closed form vs Mathematica, for every entry."""
    wigner = pytest.importorskip("sympy.physics.wigner", reason="sympy not installed")
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def gaunt(l1, m1, l2, m2, l3, m3):
        if m1 + m2 + m3 != 0 or abs(m1) > l1 or abs(m2) > l2 or abs(m3) > l3:
            return 0.0
        pref = np.sqrt((2 * l1 + 1) * (2 * l2 + 1) * (2 * l3 + 1) / (4 * np.pi))
        return (
            pref
            * float(wigner.wigner_3j(l1, l2, l3, 0, 0, 0))
            * float(wigner.wigner_3j(l1, l2, l3, m1, m2, m3))
        )

    def cf_beta(n, m, nu, mu, dvec, k):
        dl = np.linalg.norm(dvec)
        _, thd, phd = angles(np.asarray(dvec, float))
        tot = 0.0 + 0j
        for q in range(abs(n - nu), n + nu + 1):
            g = gaunt(n, m, nu, -mu, q, mu - m)
            if g == 0.0:
                continue
            tot += (
                (1j) ** (nu + q - n)
                * (-1.0) ** q
                * hn(q, k * dl)
                * ynm(q, m - mu, thd, phd)
                * g
            )
        return 4 * np.pi * (-1.0) ** m * tot

    dvec = np.array(ref["dvec"], float)
    worst = 0.0
    for key, k in (("betaP", ref["kP"]), ("betaS", ref["kS"])):
        for e in ref[key]:
            b_py = cf_beta(e["n"], e["m"], e["nu"], e["mu"], dvec, k)
            worst = max(worst, abs(b_py - complex(e["re"], e["im"])))
    print(
        f"\n  max |beta_closed(py) - beta_closed(mma)| over {len(ref['betaP']) + len(ref['betaS'])} entries = {worst:.3e}"
    )
    assert worst < 1e-10


def test_python_field_reconstruction(ref):
    """End-to-end in Python: the outgoing scalar multipole about A, evaluated near B,
    equals the regular-multipole sum about B using the projection coefficients."""
    dvec = np.array(ref["dvec"], float)
    k = ref["kP"]
    d_len = np.linalg.norm(dvec)
    rho_proj = 0.3 * d_len
    grid = _quad_grid(dvec, rho_proj)
    nmax = 8  # low-nu projection is stable; rho/d below keeps truncation tiny
    rng = np.random.default_rng(20260618)
    # field points near B with |r-B| = 0.2 d
    pts = []
    for _ in range(8):
        uu = rng.uniform(-1, 1)
        ph = rng.uniform(0, 2 * np.pi)
        st = np.sqrt(1 - uu * uu)
        pts.append(0.2 * d_len * np.array([st * np.cos(ph), st * np.sin(ph), uu]))
    # precompute projection coefficients up to nmax (independent of field point)
    betas = {}
    for nu in range(nmax + 1):
        for mu in range(-nu, nu + 1):
            betas[(nu, mu)] = proj_beta(
                0, 0, nu, mu, k, rho_proj, grid
            )  # source (n,m)=(0,0)
    worst = 0.0
    for p in pts:
        lhs = hn(0, k * np.linalg.norm(p - dvec)) * ynm(0, 0, *angles(p - dvec)[1:])
        rhs = 0.0 + 0j
        r_b, th_b, ph_b = angles(p)
        for (nu, mu), b in betas.items():
            rhs += b * jn(nu, k * r_b) * ynm(nu, mu, th_b, ph_b)
        worst = max(worst, abs(lhs - rhs))
    print(
        f"\n  Python field-reconstruction residual (monopole source, {len(pts)} pts) = {worst:.3e}"
    )
    assert worst < 1e-7
