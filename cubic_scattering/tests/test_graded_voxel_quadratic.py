"""The graded voxel with a contrast quadratic in the cell (r = 2): twenty source monomials."""

import itertools

import numpy as np
import pytest
import sympy as sp

from cubic_scattering import MaterialContrast, ReferenceMedium
from cubic_scattering.graded_voxel.basis import (
    CONTRAST_BASIS,
    SOURCE_EXPONENTS,
    SOURCE_EXPONENTS_CUBIC,
    TEST_EXPONENTS,
    contrast_values,
    monomials,
    product_table,
    source_expansion,
)
from cubic_scattering.graded_voxel.blocks import coupling_block, far_block, near_block
from cubic_scattering.graded_voxel.fft import offset_blocks, solve_graded_sphere_fft
from cubic_scattering.graded_voxel.site import cell_contrast_coefficients
from cubic_scattering.graded_voxel.solver import solve_graded_sphere

REF = ReferenceMedium(5000.0, 3000.0, 2500.0)
CON = MaterialContrast(Dlambda=2.0e9, Dmu=1.0e9, Drho=100.0)
KHAT = np.array([1.0, 0.0, 0.0])
OMEGA, H = 150.0, 1.25


def _gauss_cell(n=6):
    x, w = np.polynomial.legendre.leggauss(n)
    xi = np.stack(np.meshgrid(x, x, x, indexing="ij"), -1).reshape(-1, 3)
    return xi, np.einsum("i,j,k->ijk", w, w, w).ravel()


def test_there_are_twenty_source_monomials_and_the_first_ten_are_unchanged():
    assert len(SOURCE_EXPONENTS_CUBIC) == 20
    assert SOURCE_EXPONENTS_CUBIC[:10] == SOURCE_EXPONENTS
    assert sorted(SOURCE_EXPONENTS_CUBIC[10:]) == sorted(
        e for e in itertools.product(range(4), repeat=3) if sum(e) == 3
    )


def test_contrast_basis_is_orthogonal_and_starts_with_the_test_basis():
    xi, w = _gauss_cell()
    q = contrast_values(xi)
    assert q.shape == (10, len(xi)) == (len(CONTRAST_BASIS), len(xi))
    gram = (q * w) @ q.T
    np.testing.assert_allclose(gram, np.diag(np.diag(gram)), atol=1e-14)
    np.testing.assert_allclose(q[:4], monomials(TEST_EXPONENTS, xi), rtol=1e-15)


def test_product_table_reproduces_contrast_times_test():
    xi = np.random.default_rng(0).uniform(-1, 1, (50, 3))
    table = product_table(10)
    assert table.shape == (20, 10, 4)
    prod = np.einsum("cab,cn->abn", table, monomials(SOURCE_EXPONENTS_CUBIC, xi))
    expect = contrast_values(xi)[:, None, :] * monomials(TEST_EXPONENTS, xi)[None, :, :]
    np.testing.assert_allclose(prod, expect, rtol=1e-13, atol=1e-14)


def test_source_expansion_sizes_follow_the_contrast_degree():
    rng = np.random.default_rng(1)
    assert source_expansion(rng.normal(size=(4, 9, 9))).shape == (10, 4, 9, 9)
    assert source_expansion(rng.normal(size=(10, 9, 9))).shape == (20, 4, 9, 9)


def test_a_quadratic_profile_is_projected_exactly():
    centre = np.array([3.0, -2.0, 1.0])

    def profile(pos):
        d = (pos - centre) / H
        return 0.5 + 0.1 * d[0] - 0.05 * d[2] + 0.04 * d[0] * d[1] - 0.03 * d[1] ** 2 + 0.02 * d[2] ** 2

    delta = cell_contrast_coefficients(profile, centre, H, CON, REF, OMEGA, degree=2)
    assert delta.shape == (10, 9, 9)
    xi = np.random.default_rng(2).uniform(-1, 1, (30, 3))
    fit = np.einsum("a,an->n", delta[:, 0, 0].real / delta[0, 0, 0].real, contrast_values(xi))
    exact = np.array([profile(centre + H * p) for p in xi]) / (0.5 - 0.01 + 0.02 / 3)
    np.testing.assert_allclose(fit, exact, rtol=1e-12)
    # the orthogonal basis leaves the mean and the linear coefficients those of the linear projection
    lin = cell_contrast_coefficients(profile, centre, H, CON, REF, OMEGA, degree=1)
    np.testing.assert_allclose(delta[:4], lin, rtol=1e-13)


def test_default_blocks_are_unchanged_and_are_the_first_ten_columns():
    for off in ((0, 0, 0), (1, 0, -1), (2, 1, 0)):
        k10 = coupling_block(off, H, OMEGA, REF)
        k20 = coupling_block(off, H, OMEGA, REF, n_source=20)
        assert k10.shape == (4, 10, 9, 9)
        assert k20.shape == (4, 20, 9, 9)
        assert np.linalg.norm(k20[:, :10] - k10) / np.linalg.norm(k10) < 1e-13


def test_cubic_columns_match_the_6d_reference_beyond_contact():
    for off in ((2, 0, 0), (2, -2, 1)):
        a = near_block(off, H, OMEGA, REF, n_q=14, n_source=20)
        b = far_block(off, H, OMEGA, REF, 10, n_source=20)
        c = coupling_block(off, H, OMEGA, REF, n_source=20)
        for col in range(10, 20):
            scale = np.linalg.norm(b[:, col])
            assert np.linalg.norm(a[:, col] - b[:, col]) / scale < 1e-9, (off, col)
            assert np.linalg.norm(c[:, col] - b[:, col]) / scale < 1e-9, (off, col)


def _expand_on_sub(exps, sigma):
    xs = sp.symbols("s0 s1 s2")
    out = np.zeros((len(exps), len(exps)))
    for i, e in enumerate(exps):
        expr = sp.expand(sp.Mul(*[((xs[k] + sigma[k]) / 2) ** e[k] for k in range(3)]))
        poly = sp.Poly(expr, *xs)
        for mono, coef in zip(poly.monoms(), poly.coeffs(), strict=True):
            out[i, exps.index(tuple(mono))] = float(coef)
    return out


@pytest.mark.parametrize("off", [(0, 0, 0), (1, 0, 0), (1, 1, 1)])
def test_subdivision_identity_with_cubic_sources(off):
    # a block between cells of half-width h is the sum of the blocks between their 2^3 sub-cells, with the
    # polynomials re-expressed on the sub-cells: exact, and it exercises the self and touching blocks
    sigmas = list(itertools.product((-1, 1), repeat=3))
    c_t = {s: _expand_on_sub(TEST_EXPONENTS, s) for s in sigmas}
    d_s = {s: _expand_on_sub(SOURCE_EXPONENTS_CUBIC, s) for s in sigmas}
    big = coupling_block(off, H, OMEGA, REF, n_source=20)
    acc = np.zeros_like(big)
    cache: dict = {}
    for s, s2 in itertools.product(sigmas, sigmas):
        rsub = tuple(int(2 * o + (a - b) // 2) for o, a, b in zip(off, s, s2, strict=True))
        if rsub not in cache:
            cache[rsub] = coupling_block(rsub, H / 2, OMEGA, REF, n_source=20)
        acc += np.einsum("ab,bdij,cd->acij", c_t[s], cache[rsub], d_s[s2])
    worst = max(
        np.linalg.norm(acc[a, c] - big[a, c]) / np.linalg.norm(big[0, 0])
        for a in range(4)
        for c in range(20)
    )
    assert worst < 1e-7, worst


def test_symmetry_generated_blocks_with_cubic_sources_equal_direct_ones():
    blocks = offset_blocks(3, H, OMEGA, REF, n_source=20)
    for off in ((1, 0, -1), (-1, 1, 1), (2, -1, 0), (0, -2, 1)):
        direct = coupling_block(off, H, OMEGA, REF, n_source=20)
        assert np.linalg.norm(blocks[off] - direct) / np.linalg.norm(direct) < 1e-11, off


def _profile(pos):
    return 1.0 - 0.004 * float(pos @ pos)


def test_fft_solve_equals_the_dense_solve_with_a_quadratic_contrast():
    dense = solve_graded_sphere(OMEGA, 10.0, REF, CON, 4, _profile, KHAT, KHAT, "P", p=1, r=2)
    fft = solve_graded_sphere_fft(
        OMEGA, 10.0, REF, CON, 4, _profile, KHAT, KHAT, "P", p=1, r=2, gmres_tol=1e-13
    )
    assert dense.delta.shape[1] == 10
    rel = np.linalg.norm(fft.psi - dense.psi) / np.linalg.norm(dense.psi)
    assert rel < 1e-10, rel


def test_a_quadratic_profile_solved_with_r2_differs_from_r1():
    # _profile is exactly quadratic: r = 2 represents it exactly, r = 1 does not
    r1 = solve_graded_sphere_fft(OMEGA, 10.0, REF, CON, 4, _profile, KHAT, KHAT, "P", p=1, r=1)
    r2 = solve_graded_sphere_fft(OMEGA, 10.0, REF, CON, 4, _profile, KHAT, KHAT, "P", p=1, r=2)
    rel = np.linalg.norm(r2.psi - r1.psi) / np.linalg.norm(r1.psi)
    assert 1e-8 < rel < 1e-2, rel
