"""The graded single site T36."""

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from cubic_scattering import MaterialContrast, ReferenceMedium
from cubic_scattering.graded_voxel.basis import TEST_EXPONENTS, gram_test, monomials
from cubic_scattering.graded_voxel.blocks import near_block, static_term_integral
from cubic_scattering.graded_voxel.site import (
    cell_contrast_coefficients,
    contrast_operator,
    single_site_t36,
)

REF = ReferenceMedium(5000.0, 3000.0, 2500.0)
CON = MaterialContrast(Dlambda=2.0e9, Dmu=1.0e9, Drho=100.0)
OMEGA, H = 150.0, 1.25
ROOT = Path(__file__).resolve().parents[2]


def _delta(grad=(0.0, 0.0, 0.0)):
    base = contrast_operator(CON.Dlambda, CON.Dmu, CON.Drho, OMEGA)
    return np.array([base] + [g * base for g in grad])


def _parity(i, a):
    comp = -1 if i < 3 else 1  # displacement odd, strain even
    return comp * (1 if a == 0 else -1)


def test_parity_split_without_gradient():
    t = single_site_t36(H, _delta(), near_block((0, 0, 0), H, OMEGA, REF))
    par = np.array([_parity(i, a) for a in range(4) for i in range(9)])
    mixed = np.abs(t[np.ix_(par == 1, par == -1)]).max() + np.abs(t[np.ix_(par == -1, par == 1)]).max()
    assert mixed < 1e-12 * np.abs(t).max()
    assert (par == 1).sum() == 15 and (par == -1).sum() == 21


def test_gradient_couples_the_parity_sets():
    t = single_site_t36(H, _delta((0.1, 0.0, 0.0)), near_block((0, 0, 0), H, OMEGA, REF))
    par = np.array([_parity(i, a) for a in range(4) for i in range(9)])
    assert np.abs(t[np.ix_(par == 1, par == -1)]).max() > 1e-4 * np.abs(t).max()


def _voigt_rep(q, engineering):
    pairs = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    rep = np.zeros((6, 6))
    for b, (m, n) in enumerate(pairs):
        e = np.zeros((3, 3))
        val = 0.5 if (engineering and m != n) else 1.0
        e[m, n] = e[n, m] = val
        er = q @ e @ q.T
        for a, (i, j) in enumerate(pairs):
            rep[a, b] = er[i, j] * (2.0 if (engineering and i != j) else 1.0)
    return rep


def _rep36(q, strain):
    r9 = np.zeros((9, 9))
    r9[:3, :3] = q
    r9[3:, 3:] = _voigt_rep(q, engineering=strain)
    ra = np.zeros((4, 4))
    ra[0, 0] = 1.0
    ra[1:, 1:] = q
    return np.kron(ra, r9)


def test_c4v_about_the_gradient_axis():
    # a gradient along axis 0 is invariant under the quarter turn about axis 0: (z, x, y) -> (z, -y, x)
    q = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])
    t = single_site_t36(H, _delta((0.1, 0.0, 0.0)), near_block((0, 0, 0), H, OMEGA, REF))
    rep_in, rep_out = _rep36(q, strain=True), _rep36(q, strain=False)
    np.testing.assert_allclose(rep_out @ t, t @ rep_in, atol=1e-11 * np.abs(t).max())


def test_born_limit_without_gradient_is_exact():
    weak = 1e-9 * _delta()
    t = single_site_t36(H, weak, near_block((0, 0, 0), H, OMEGA, REF))
    m = np.kron(np.diag([8 * H**3] + [8 * H**3 / 3] * 3), np.eye(9))
    np.testing.assert_allclose(t, m @ np.kron(np.eye(4), weak[0]), rtol=1e-8, atol=1e-8 * np.abs(t).max())


def test_born_error_with_gradient_scales_as_kh_squared():
    # G2: with a gradient, T36 drops the incident field's quadratic moments, so its Born error is
    # O(Delta gh (kh)^2): the error must fall by 4 when kh halves
    x, w = leggauss(12)
    xi = np.stack(np.meshgrid(x, x, x, indexing="ij"), -1).reshape(-1, 3)
    ww = np.einsum("i,j,k->ijk", w, w, w).ravel()
    lt = monomials(TEST_EXPONENTS, xi)
    amp = np.arange(1, 10) + 0.5j
    errs = []
    for kh in (0.05, 0.1):
        omega = kh / H * REF.beta
        base = 1e-9 * contrast_operator(CON.Dlambda, CON.Dmu, CON.Drho, omega)
        delta = np.array([base, 0.2 * base, 0.0 * base, 0.0 * base])
        k_vec = np.array([1.0, 0.0, 0.0]) * kh / H
        field = np.exp(1j * H * xi @ k_vec)  # e^{i k.x} at the nodes (cell centred at 0)
        coeff = np.linalg.solve(gram_test(H), H**3 * (lt * ww) @ field)
        t = single_site_t36(H, delta, near_block((0, 0, 0), H, omega, REF))
        got = t @ np.kron(coeff, amp)
        dfun = np.einsum("an,aij->nij", lt, delta)  # Delta(xi) at the nodes
        exact = H**3 * np.einsum("an,n,nij,j->ai", lt * ww, field, dfun, amp).ravel()
        errs.append(np.linalg.norm(got - exact) / np.linalg.norm(exact))
    slope = np.log(errs[1] / errs[0]) / np.log(2.0)
    assert abs(slope - 2.0) < 0.15, (errs, slope)


def test_matches_mathematica_term_integrals():
    # G3: the singular term integrals by integration by parts onto the cells (Mathematica) against the
    # s-form with derivatives moved onto the autocorrelation (Python)
    path = ROOT / "Mathematica" / "GradedVoxel_term_integrals.json"
    data = json.loads(path.read_text())
    assert len(data["terms"]) == 128
    for t in data["terms"]:
        want = t["value"][0] + 1j * t["value"][1]
        got = static_term_integral(t["m"], tuple(t["idx"]), tuple(t["offset"]), float(data["h"]), 16)[
            t["a"], t["c"]
        ]
        scale = max(abs(want), 1e-3)
        assert abs(got - want) / scale < 1e-10, t


def test_unphysical_projected_contrast_fails_fast():
    def steep(pos):
        return 1.0 + 40.0 * pos[0]

    with pytest.raises(ValueError, match="52%") as err:
        cell_contrast_coefficients(steep, np.zeros(3), H, CON, REF, OMEGA, degree=1)
    msg = str(err.value)
    for part in ("cell centre", "allowed", "Fix"):
        assert part in msg
