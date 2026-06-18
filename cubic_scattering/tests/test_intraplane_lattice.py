"""Independent Python cross-check of the Phase-1 Ewald intra-plane lattice sum.

The Mathematica notebook ``Mathematica/IntraPlaneLatticeSum.wl`` Ewald-accelerates the
conditionally convergent intra-plane Helmholtz sum (EwaldIntraPlanePropagator.tex,
Eqs. real + recip) and dumps sample values to
``Mathematica/IntraPlaneLatticeSum_reference.json``.

Here Python reimplements the real-space and reciprocal-space halves and verifies,
independently and cross-language:
  1. agreement with the Mathematica Ewald values (real and damped kappa),
  2. eta-split independence,
  3. agreement with the damped direct real-space sum (where the latter converges).

Complex erfc is built from the Faddeeva function: erfc(z) = exp(-z^2) wofz(i z).
Conventions match the notebook: square lattice spacing aL, A = aL^2, reciprocal
spacing 2 pi / aL, Bloch phase e^{i k_par . R}, Im k_zG >= 0.

Run:  conda run -n seismic pytest cubic_scattering/tests/test_intraplane_lattice.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import wofz

REF_JSON = (
    Path(__file__).resolve().parents[2]
    / "Mathematica"
    / "IntraPlaneLatticeSum_reference.json"
)


def erfc_c(z):
    """Complex complementary error function via Faddeeva: erfc(z) = exp(-z^2) w(i z)."""
    z = np.asarray(z, dtype=complex)
    return np.exp(-(z**2)) * wofz(1j * z)


@pytest.fixture(scope="module")
def ref():
    assert REF_JSON.exists(), f"missing {REF_JSON} (run IntraPlaneLatticeSum.wl first)"
    return json.loads(REF_JSON.read_text())


def _ewald(ref):
    """Return real/recip/direct closures bound to the reference geometry."""
    aL = ref["aL"]
    kx, ky = ref["kx"], ref["ky"]
    area = aL**2
    recipB = 2 * np.pi / aL

    def real_half(kappa, rho, eta, rc):
        i = np.arange(-rc, rc + 1)
        ii, jj = np.meshgrid(i, i, indexing="ij")
        rx, ry = aL * ii, aL * jj
        d = np.sqrt((rho[0] - rx) ** 2 + (rho[1] - ry) ** 2)
        bloch = np.exp(1j * aL * (kx * ii + ky * jj))
        term = np.zeros_like(d, dtype=complex)
        for s in (-1, 1):
            term += np.exp(s * 1j * kappa * d) * erfc_c(
                d * eta + s * 1j * kappa / (2 * eta)
            )
        return (1 / (8 * np.pi)) * np.sum(bloch / d * term)

    def recip_half(kappa, rho, eta, gc):
        m = np.arange(-gc, gc + 1)
        mm, nn = np.meshgrid(m, m, indexing="ij")
        kpgx, kpgy = kx + recipB * mm, ky + recipB * nn
        kz = np.sqrt(
            kappa**2 - (kpgx**2 + kpgy**2) + 0j
        )  # Im kz >= 0 (principal branch)
        phase = np.exp(1j * (kpgx * rho[0] + kpgy * rho[1]))
        return (1j / (2 * area)) * np.sum(phase / kz * erfc_c(kz / (2j * eta)))

    def total(kappa, rho, eta, rc, gc):
        return real_half(kappa, rho, eta, rc) + recip_half(kappa, rho, eta, gc)

    def direct(kappa, rho, lbig):
        i = np.arange(-lbig, lbig + 1)
        ii, jj = np.meshgrid(i, i, indexing="ij")
        d = np.sqrt((rho[0] - aL * ii) ** 2 + (rho[1] - aL * jj) ** 2)
        bloch = np.exp(1j * aL * (kx * ii + ky * jj))
        return np.sum(np.exp(1j * kappa * d) / (4 * np.pi * d) * bloch)

    return total, direct


def test_ewald_matches_mathematica(ref):
    total, _ = _ewald(ref)
    eta, rc, gc = ref["eta1"], ref["RcE"], ref["GcE"]
    kappaR = ref["kReal"]
    kappaD = ref["kDampRe"] + 1j * ref["kDampIm"]
    worst = 0.0
    for key, kappa in (("ewaldReal", kappaR), ("ewaldDamp", kappaD)):
        for rho, (re, im) in zip(ref["rhoPts"], ref[key], strict=True):
            got = total(kappa, np.array(rho), eta, rc, gc)
            worst = max(worst, abs(got - complex(re, im)))
    print(
        f"\n  max |Ewald(py) - Ewald(mma)| over {2 * len(ref['rhoPts'])} samples = {worst:.3e}"
    )
    assert worst < 1e-9


def test_ewald_eta_independence(ref):
    total, _ = _ewald(ref)
    rc, gc = ref["RcE"], ref["GcE"]
    kappaD = ref["kDampRe"] + 1j * ref["kDampIm"]
    worst = 0.0
    for kappa in (ref["kReal"], kappaD):
        for rho in ref["rhoPts"]:
            a = total(kappa, np.array(rho), 0.7, rc, gc)
            b = total(kappa, np.array(rho), 1.15, rc, gc)
            worst = max(worst, abs(a - b))
    print(f"\n  max Ewald eta-independence |G(eta1)-G(eta2)| = {worst:.3e}")
    assert worst < 1e-11


def test_ewald_vs_direct(ref):
    total, direct = _ewald(ref)
    eta, rc, gc = ref["eta1"], ref["RcE"], ref["GcE"]
    kappaD = ref["kDampRe"] + 1j * ref["kDampIm"]
    worst = 0.0
    for rho in ref["rhoPts"]:
        e = total(kappaD, np.array(rho), eta, rc, gc)
        d = direct(kappaD, np.array(rho), 40)
        worst = max(worst, abs(e - d))
    print(f"\n  max |Ewald - damped direct| = {worst:.3e}")
    assert worst < 1e-6
