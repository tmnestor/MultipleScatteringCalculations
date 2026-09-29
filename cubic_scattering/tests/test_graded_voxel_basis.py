"""Local polynomial bases of the graded voxel."""

import numpy as np
from numpy.polynomial.legendre import leggauss

from cubic_scattering.graded_voxel.basis import (
    SOURCE_EXPONENTS,
    TEST_EXPONENTS,
    gram_test,
    gram_test_source,
    monomials,
    product_table,
    source_expansion,
)

H = 0.7


def _gauss_cell(n=6):
    x, w = leggauss(n)
    xi = np.stack(np.meshgrid(x, x, x, indexing="ij"), -1).reshape(-1, 3)
    ww = np.einsum("i,j,k->ijk", w, w, w).ravel()
    return xi, ww


def test_grams_match_quadrature():
    xi, w = _gauss_cell()
    lt, ls = monomials(TEST_EXPONENTS, xi), monomials(SOURCE_EXPONENTS, xi)
    np.testing.assert_allclose(gram_test(H), H**3 * (lt * w) @ lt.T, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(gram_test_source(H), H**3 * (lt * w) @ ls.T, rtol=1e-14, atol=1e-15)


def test_gram_test_is_diagonal():
    g = gram_test(H)
    np.testing.assert_allclose(g, np.diag([8 * H**3] + [8 * H**3 / 3] * 3), rtol=1e-15)


def test_product_table_reproduces_products():
    xi = np.random.default_rng(0).uniform(-1, 1, (50, 3))
    lt, ls = monomials(TEST_EXPONENTS, xi), monomials(SOURCE_EXPONENTS, xi)
    prod = np.einsum("cab,cn->abn", product_table(), ls)
    np.testing.assert_allclose(prod, lt[:, None, :] * lt[None, :, :], rtol=1e-15)


def test_source_expansion_of_a_linear_contrast():
    rng = np.random.default_rng(1)
    delta = rng.normal(size=(4, 9, 9)) + 1j * rng.normal(size=(4, 9, 9))
    xi = rng.uniform(-1, 1, (20, 3))
    lt, ls = monomials(TEST_EXPONENTS, xi), monomials(SOURCE_EXPONENTS, xi)
    e = source_expansion(delta)
    for b in range(4):
        want = np.einsum("an,aij->nij", lt, delta) * lt[b][:, None, None]  # Delta(xi) L_b(xi)
        got = np.einsum("cn,cij->nij", ls, e[:, b])
        np.testing.assert_allclose(got, want, rtol=1e-13, atol=1e-13)
