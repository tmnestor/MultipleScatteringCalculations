"""The point Green's tensor and its derivatives at small k r, against a 50-digit evaluation.

G_ij = (1 / 4 pi mu) [delta_ij g_S + d_i d_j (g_S - g_P) / k_S^2], g_k = e^{ikr} / r. The closed form of
(g_S - g_P) / k_S^2 cancels its 1/r term between the P and S parts, so evaluated as written it loses
digits as eps / (k_S r)^2: 4e-7 relative at k_S r = 1e-4. ``elastodynamic_greens_deriv`` and the 9 x 9
block built on it must hold round-off accuracy at every k_S r.

The reference evaluates the same closed form at 50 digits, where the cancellation costs nothing.
"""

import itertools

import mpmath as mp
import numpy as np
import pytest
import sympy as sp

from cubic_scattering.effective_contrasts import ReferenceMedium
from cubic_scattering.graded_voxel.kernel import kernel_9x9
from cubic_scattering.resonance_tmatrix import _propagator_block_9x9, elastodynamic_greens_deriv

REF = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
OMEGA = 2.0 * np.pi * 10.0
ETAS = [0.0, 0.05, 0.5]  # attenuation: lattice_greens passes omega (1 + i eta)
DIRECTION = np.array([0.6, 0.48, 0.64])
KR_VALUES = [1e-4, 1e-3, 1e-2, 0.1, 0.49, 0.51, 1.0, 10.0]
RTOL = 5e-14


def _radial_F_mpmath():
    r, k = sp.symbols("r k", positive=True)
    exprs = [sp.exp(sp.I * k * r) / r]
    for _ in range(4):
        exprs.append(sp.diff(exprs[-1], r) / r)
    return [sp.lambdify((r, k), e, "mpmath") for e in exprs]


def _reference_tensors(x: np.ndarray, omega: complex) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """G, d_k G_ij and d_k d_l G_ij at 50 digits, from F_q = ((1/r) d/dr)^q of the radial parts."""
    mp.mp.dps = 50
    k_p = mp.mpc(complex(omega)) / mp.mpf(REF.alpha)
    k_s = mp.mpc(complex(omega)) / mp.mpf(REF.beta)
    funcs = _radial_F_mpmath()
    xm = [mp.mpf(float(c)) for c in x]
    r = mp.sqrt(sum(c * c for c in xm))
    fa = [f(r, k_s) for f in funcs]
    fb = [(f(r, k_s) - f(r, k_p)) / k_s**2 for f in funcs]

    def dl(a: int, b: int) -> int:
        return 1 if a == b else 0

    def d(fq: list, idx: tuple[int, ...]):
        if len(idx) == 0:
            return fq[0]
        if len(idx) == 1:
            return xm[idx[0]] * fq[1]
        if len(idx) == 2:
            i, j = idx
            return dl(i, j) * fq[1] + xm[i] * xm[j] * fq[2]
        if len(idx) == 3:
            i, j, k = idx
            sym = dl(i, j) * xm[k] + dl(i, k) * xm[j] + dl(j, k) * xm[i]
            return sym * fq[2] + xm[i] * xm[j] * xm[k] * fq[3]
        i, j, k, m = idx
        pairs = dl(i, j) * dl(k, m) + dl(i, k) * dl(j, m) + dl(i, m) * dl(j, k)
        mixed = (
            dl(i, j) * xm[k] * xm[m]
            + dl(i, k) * xm[j] * xm[m]
            + dl(i, m) * xm[j] * xm[k]
            + dl(j, k) * xm[i] * xm[m]
            + dl(j, m) * xm[i] * xm[k]
            + dl(k, m) * xm[i] * xm[j]
        )
        return pairs * fq[2] + mixed * fq[3] + xm[i] * xm[j] * xm[k] * xm[m] * fq[4]

    c = 1 / (4 * mp.pi * mp.mpf(REF.mu))
    G = np.zeros((3, 3), complex)
    Gd = np.zeros((3, 3, 3), complex)
    Gdd = np.zeros((3, 3, 3, 3), complex)
    for i, j in itertools.product(range(3), repeat=2):
        G[i, j] = complex(c * (dl(i, j) * d(fa, ()) + d(fb, (i, j))))
        for k in range(3):
            Gd[i, j, k] = complex(c * (dl(i, j) * d(fa, (k,)) + d(fb, (i, j, k))))
            for m in range(3):
                Gdd[i, j, k, m] = complex(c * (dl(i, j) * d(fa, (k, m)) + d(fb, (i, j, k, m))))
    return G, Gd, Gdd


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


@pytest.mark.parametrize("eta", ETAS)
@pytest.mark.parametrize("kr", KR_VALUES)
def test_greens_deriv_holds_round_off_at_every_kr(kr: float, eta: float) -> None:
    omega = OMEGA * (1.0 + 1j * eta) if eta else OMEGA
    x = DIRECTION * (kr / abs(omega / REF.beta))
    ref_tensors = _reference_tensors(x, omega)
    got = elastodynamic_greens_deriv(x, omega, REF)
    for name, g, r in zip(("G", "Gd", "Gdd"), got, ref_tensors, strict=True):
        assert _rel(g, r) < RTOL, f"{name} at |k_S| r = {kr}, eta = {eta}: relative error {_rel(g, r):.2e}"


@pytest.mark.parametrize("kr", KR_VALUES)
def test_block_9x9_is_the_vectorised_kernel(kr: float) -> None:
    x = DIRECTION * (kr / (OMEGA / REF.beta))
    np.testing.assert_allclose(
        _propagator_block_9x9(x, OMEGA, REF), kernel_9x9(x[None], OMEGA, REF)[0], rtol=1e-15, atol=0.0
    )


def test_zero_separation_returns_zeros() -> None:
    G, Gd, Gdd = elastodynamic_greens_deriv(np.zeros(3), OMEGA, REF)
    assert not G.any() and not Gd.any() and not Gdd.any()
    assert not _propagator_block_9x9(np.zeros(3), OMEGA, REF).any()
