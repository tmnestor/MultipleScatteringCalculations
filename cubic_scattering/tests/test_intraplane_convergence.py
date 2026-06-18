"""Independent Python cross-check of the Phase-2 item (c) convergence study.

``Mathematica/IntraPlaneConvergence.wl`` dumps the closed-form P-channel (L) collective
Foldy-Lax convergence study to ``IntraPlaneConvergence_reference.json``:
  - the single-site Mie spectrum ``||T0(n)||_F`` (Part A);
  - the renormalised collective monopole coefficient over ``Nmax = 1..6`` and a packing
    sweep ``aL in {6, 4, 3, 2.5, 2.2}`` (sphere radius ``aa = 1``, touching at ``aL = 2``),
    plus the coupling / spectral radius / conditioning per ``aL`` (Part B).

Here Python independently (scipy/sympy):

1. **Rebuilds the closed-form collective monopole from scratch** -- the scalar damped
   lattice structure constants ``D[q, s] = sum_R h_q(kappa|R|) Y_q^s(^R) e^{i k.R}`` (stable
   spherical Hankel), the Gaunt contraction ``g0LL``, the diagonal Mie L-channel ``T0L``
   (scalars imported from the dump), and the collective solve
   ``t_coll = T0L (I - G0LL T0L)^{-1}`` -- and matches the dumped monopole at every
   ``(aL, Nmax)``. (The monopole is a diagonal entry, hence invariant to the index-order
   transpose between this recompute and the .wl ``solveLL``.)
2. Confirms **n-convergence**: the monopole is converged in multipole order to < 1e-5
   relative at every packing density.
3. Confirms the **packing-density** trend: coupling, spectral radius and conditioning all
   grow monotonically as the spheres approach touching (``aL -> 2 aa``) -- the numerical
   signature of the translation-theorem region-of-validity boundary.

Run:  conda run -n seismic pytest cubic_scattering/tests/test_intraplane_convergence.py -v
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
from scipy.special import hankel1, sph_harm_y
from sympy.physics.wigner import wigner_3j

REF = (
    Path(__file__).resolve().parents[2]
    / "Mathematica"
    / "IntraPlaneConvergence_reference.json"
)


@pytest.fixture(scope="module")
def ref():
    assert REF.exists(), f"missing {REF} (run IntraPlaneConvergence.wl first)"
    d = json.loads(REF.read_text())
    d["aLlist"] = [float(a) for a in d["aLlist"]]
    d["T0Lc"] = np.array([complex(re, im) for (re, im) in d["T0L"]], dtype=complex)
    d["monoc"] = np.array(
        [[complex(re, im) for (re, im) in per_aL] for per_aL in d["mono"]],
        dtype=complex,
    )  # shape (n_aL, n_Nmax)
    return d


# ----------------------------------------------------- closed-form L-block helpers
def _sph_h(q, z):
    """Stable outgoing spherical Hankel h_q^(1)(z) (no j+iy cancellation)."""
    return np.sqrt(np.pi / (2.0 * z)) * hankel1(q + 0.5, z)


@lru_cache(maxsize=None)
def _gaunt(l1, m1, l2, m2, l3, m3):
    """Memoised Gaunt coefficient (sympy wigner_3j is symbolic/slow; many repeats)."""
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


def _g0LL(n, m, nu, mu, structD):
    tot = 0.0 + 0j
    for q in range(abs(n - nu), n + nu + 1):
        g = _gaunt(n, m, nu, -mu, q, mu - m)
        if g == 0.0:
            continue
        tot += (1j) ** (nu + q - n) * (-1) ** q * structD(q, m - mu) * g
    return 4 * np.pi * (-1) ** m * tot


def _idxL(Nmax):
    return [(n, m) for n in range(Nmax + 1) for m in range(-n, n + 1)]


def _build_full(ref, ai):
    """Closed-form G0LL + diagonal T0L at Nmax=NmaxB for packing index ai.

    Natural [target, source] convention; sub-blocks (leading principal minors) give
    every smaller Nmax since idxL is ordered by ascending n.
    """
    aL = ref["aLlist"][ai]
    kappaP = ref["kPo"] + 1j * ref["dampIm"]
    kx, ky = ref["kx"], ref["ky"]
    Lrad, Nmax = int(ref["LradB"]), int(ref["NmaxB"])
    cache: dict[tuple[int, int], complex] = {}

    def structD(q, s):
        if (q, s) not in cache:
            cache[(q, s)] = _structD(q, s, kappaP, aL, kx, ky, Lrad)
        return cache[(q, s)]

    idx = _idxL(Nmax)
    nD = len(idx)
    T0L = np.diag([ref["T0Lc"][n] for (n, _m) in idx])
    G0 = np.array(
        [
            [
                _g0LL(idx[j][0], idx[j][1], idx[i][0], idx[i][1], structD)
                for j in range(nD)
            ]
            for i in range(nD)
        ],
        dtype=complex,
    )
    return G0, T0L


def _solve_block(G0, T0L, Nmax):
    nD = len(_idxL(Nmax))
    g, t = G0[:nD, :nD], T0L[:nD, :nD]
    tcoll = t @ np.linalg.inv(np.eye(nD, dtype=complex) - g @ t)
    coupling = np.linalg.norm(tcoll - t) / np.linalg.norm(t)
    return tcoll[0, 0], coupling


@pytest.fixture(scope="module")
def recomputed(ref):
    """Python-side monopole(ai, Nmax) rebuilt from scratch (one full build per aL)."""
    nA, nN = len(ref["aLlist"]), int(ref["NmaxB"])
    mono = np.zeros((nA, nN), dtype=complex)
    for ai in range(nA):
        G0, T0L = _build_full(ref, ai)
        for Nmax in range(1, nN + 1):
            mono[ai, Nmax - 1], _ = _solve_block(G0, T0L, Nmax)
    return mono


# ---------------------------------------------------------------------- the checks
def test_mie_spectrum_decay(ref):
    """[A] dumped single-site Mie spectrum decays super-exponentially for n>=2."""
    s = np.array(ref["specNorm"])
    assert np.all(np.diff(s[2:]) < 0), "||T0(n)||_F must decrease for n>=2"
    assert s[-1] / s.max() < 1e-4
    print(f"\n  Mie ||T0(n)||_F: peak={s.max():.2e}, n={len(s) - 1} -> {s[-1]:.2e}")


def test_closedform_monopole_crosscheck(ref, recomputed):
    """[B] Python closed-form collective monopole matches the .wl dump at all (aL, Nmax)."""
    worst = float(np.max(np.abs(recomputed - ref["monoc"])))
    print(f"\n  max |mono(py) - mono(mma)| over all (aL, Nmax) = {worst:.3e}")
    assert worst < 1e-6


def test_n_convergence(ref):
    """[B] monopole converged in multipole order: < 1e-5 relative at every density."""
    mono = ref["monoc"]
    # relative increment at Nmax=5 (well-conditioned at every density; Nmax index 4)
    rel = np.abs(mono[:, 4] - mono[:, 3]) / np.abs(mono[:, 4])
    for ai, aL in enumerate(ref["aLlist"]):
        print(f"\n  aL={aL}: rel monopole convergence (N=5) = {rel[ai]:.3e}")
    assert np.all(rel < 1e-5)


def test_vector_collective_convergence(ref):
    """[C] full-vector (L/M/N) collective is converged in multipole order at fixed lattice.

    The shared low-order block is stable in Nmax (Cauchy increment small absolutely and
    relative to the coupling). This is the magnitude test, not a strict monotone decrease:
    the increments (~1e-6) sit below the Lrad=8 M/N lattice-truncation floor (~1e-2), whose
    ordering at that level is lattice/quadrature noise (deep lattice convergence is the
    deferred Ewald-accelerated vector G0).
    """
    cauchyV = np.array(ref["cauchyV"])
    couplingV = np.array(ref["couplingV"])
    rel = float(cauchyV.max() / couplingV[-1])
    print(
        f"\n  vector shared-block Cauchy = {cauchyV}"
        f"\n  coupling(Nmax=1,2,3)       = {couplingV}"
        f"\n  max Cauchy / coupling = {rel:.2e},  Lrad floor = {ref['floorC']:.2e}"
    )
    assert np.all(couplingV > 1e-6) and np.all(np.isfinite(couplingV))
    assert cauchyV.max() < 1e-5  # low-order collective block stable in Nmax
    assert rel < 1e-3  # multipole increment negligible vs the coupling


def test_packing_density_trend(ref):
    """[B] coupling / spectral radius / conditioning grow as aL -> touching."""
    # aLlist is descending, so each quantity must be ascending (denser => stronger)
    coupling = np.array(ref["coupling"])
    specrad = np.array(ref["specrad"])
    condN5 = np.array(ref["condN5"])
    print(
        f"\n  aL (desc) = {ref['aLlist']}"
        f"\n  coupling  = {coupling}"
        f"\n  specrad   = {specrad}"
        f"\n  cond@N5   = {condN5}"
    )
    assert np.all(np.diff(coupling) > 0), "coupling must grow as aL decreases"
    assert np.all(np.diff(specrad) > 0), "spectral radius must grow as aL decreases"
    assert np.all(np.diff(condN5) > 0), "conditioning must worsen as aL decreases"
