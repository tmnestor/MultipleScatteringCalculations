"""Independent Python cross-check of the Phase-2 collective Foldy-Lax results.

``Mathematica/IntraPlaneVectorLattice.wl`` dumps the lattice-summed multi-channel
vector coupling ``G0^vec(k_par)`` (plus the L/M/N basis index and the lattice/damping
parameters) to ``Mathematica/IntraPlaneVectorLattice_reference.json``. Here Python
independently verifies, cross-language:

1. **Symplectic-J reciprocity** (the resolution of the T0/G0 metric mismatch, from
   ``IntraPlaneCollectiveReciprocity.wl``): with the symplectic channel metric
   ``D = diag(d_L=(kS/kP)^3/2, d_M=I*sqrt(n(n+1)), d_N=sqrt(n(n+1)))`` and the m-flip
   conjugation ``J0 = diag((-1)^(n+m)) (x) (m->-m)``, the dumped ``G0^vec`` satisfies
   ``J0 (D G0 D^-1) J0 = (D G0 D^-1)^T``. Without the metric (D=I) it does NOT.
2. The **L-block** (scalar P channel) is sigma-reciprocal on its own.
3. The **L-block values**, recomputed from scratch in Python from the scalar damped
   lattice structure constants ``D[q,s] = sum_R h_q(kappa|R|) Y_q^s(^R) e^{i k.R}`` and
   the Gaunt contraction, match the dumped ``G0^vec`` L-block.

Run:  conda run -n seismic pytest cubic_scattering/tests/test_intraplane_collective.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import hankel1, sph_harm_y
from sympy.physics.wigner import wigner_3j

REF = (
    Path(__file__).resolve().parents[2]
    / "Mathematica"
    / "IntraPlaneVectorLattice_reference.json"
)


@pytest.fixture(scope="module")
def ref():
    assert REF.exists(), f"missing {REF} (run IntraPlaneVectorLattice.wl first)"
    d = json.loads(REF.read_text())
    d["G0"] = np.array(
        [[complex(re, im) for (re, im) in row] for row in d["G0vec"]], dtype=complex
    )
    d["idx"] = [(int(n), int(m), str(c)) for (n, m, c) in d["idx"]]
    return d


# ---------------------------------------------------------------- metric / J0
def _metric(idx, kPo, kSo):
    def dwt(e):
        n, _m, c = e
        if c == "L":
            return (kSo / kPo) ** 1.5
        if c == "N":
            return np.sqrt(n * (n + 1))
        return 1j * np.sqrt(n * (n + 1))  # M: SH/SV I-phase

    return np.diag([dwt(e) for e in idx]).astype(complex)


def _J0(idx):
    nD = len(idx)
    J = np.zeros((nD, nD), dtype=complex)
    pos = {e: i for i, e in enumerate(idx)}
    for k, (n, m, c) in enumerate(idx):
        J[pos[(n, -m, c)], k] = (-1) ** (n + m)
    return J


def _recip(A, D, J0):
    B = D @ A @ np.linalg.inv(D)
    return float(np.max(np.abs(J0 @ B @ J0 - B.T)))


def test_symplectic_reciprocity(ref):
    """G0 is sigma-symmetric in the symplectic metric (item d), but NOT without it."""
    idx = ref["idx"]
    J0 = _J0(idx)
    D = _metric(idx, ref["kPo"], ref["kSo"])
    eye = np.eye(len(idx), dtype=complex)
    with_metric = _recip(ref["G0"], D, J0)
    without = _recip(ref["G0"], eye, J0)
    print(
        f"\n  G0 sigma-reciprocity: with symplectic D = {with_metric:.3e}"
        f",  with D=I = {without:.3e}"
    )
    assert with_metric < 1e-10  # reciprocal in the symplectic metric
    assert without > 1e-2  # and the metric is doing real work


def test_Lblock_reciprocity(ref):
    """The scalar P (L) block is sigma-reciprocal on its own."""
    idx = ref["idx"]
    Lpos = [i for i, e in enumerate(idx) if e[2] == "L"]
    Lidx = [idx[i] for i in Lpos]
    G0LL = ref["G0"][np.ix_(Lpos, Lpos)]
    nD = len(Lpos)
    J = np.zeros((nD, nD), dtype=complex)
    pos = {e: i for i, e in enumerate(Lidx)}
    for k, (n, m, c) in enumerate(Lidx):
        J[pos[(n, -m, c)], k] = (-1) ** (n + m)
    r = float(np.max(np.abs(J @ G0LL @ J - G0LL.T)))
    print(f"\n  G0 L-block sigma-reciprocity = {r:.3e}")
    assert r < 1e-10


# ------------------------------------------------ independent L-block recompute
def _sph_h(q, z):
    """Outgoing spherical Hankel h_q^(1)(z) for complex z, via cylindrical hankel1."""
    return np.sqrt(np.pi / (2.0 * z)) * hankel1(q + 0.5, z)


def _gaunt(l1, m1, l2, m2, l3, m3):
    if m1 + m2 + m3 != 0 or abs(m1) > l1 or abs(m2) > l2 or abs(m3) > l3:
        return 0.0
    pref = np.sqrt((2 * l1 + 1) * (2 * l2 + 1) * (2 * l3 + 1) / (4 * np.pi))
    return (
        float(pref)
        * float(wigner_3j(l1, l2, l3, 0, 0, 0))
        * float(wigner_3j(l1, l2, l3, m1, m2, m3))
    )


def _structD(q, s, kappa, aL, kx, ky, Lrad):
    ij = np.array(
        [
            (i, j)
            for i in range(-Lrad, Lrad + 1)
            for j in range(-Lrad, Lrad + 1)
            if (i, j) != (0, 0)
        ]
    )
    i, j = ij[:, 0], ij[:, 1]
    Rn = aL * np.sqrt(i * i + j * j)
    phi = np.arctan2(j, i)
    Y = sph_harm_y(q, s, np.pi / 2, phi)
    bloch = np.exp(1j * aL * (kx * i + ky * j))
    return np.sum(_sph_h(q, kappa * Rn) * Y * bloch)


def _g0LL(n, m, nu, mu, kappa, aL, kx, ky, Lrad):
    tot = 0.0 + 0j
    for q in range(abs(n - nu), n + nu + 1):
        g = _gaunt(n, m, nu, -mu, q, mu - m)
        if g == 0.0:
            continue
        tot += (
            (1j) ** (nu + q - n)
            * (-1) ** q
            * _structD(q, m - mu, kappa, aL, kx, ky, Lrad)
            * g
        )
    return 4 * np.pi * (-1) ** m * tot


def test_Lblock_values(ref):
    """Recompute L-block entries from the scalar structure constants; match the dump."""
    idx, G0 = ref["idx"], ref["G0"]
    kappaP = ref["kPo"] + 1j * ref["dampIm"]
    aL, kx, ky, Lrad = ref["aL"], ref["kx"], ref["ky"], int(ref["Lrad"])
    pos = {e: i for i, e in enumerate(idx)}
    # (source (n,m,L)) -> (target (nu,mu,L)); G0[target, source]
    tests = [
        ((0, 0), (0, 0)),
        ((1, 0), (1, 0)),
        ((1, -1), (2, 1)),
        ((2, 0), (1, 1)),
        ((2, 2), (2, -1)),
    ]
    worst = 0.0
    for (n, m), (nu, mu) in tests:
        got = G0[pos[(nu, mu, "L")], pos[(n, m, "L")]]
        want = _g0LL(n, m, nu, mu, kappaP, aL, kx, ky, Lrad)
        worst = max(worst, abs(got - want))
    print(f"\n  max |G0_LL(py) - G0_LL(mma)| over {len(tests)} entries = {worst:.3e}")
    assert worst < 1e-9
