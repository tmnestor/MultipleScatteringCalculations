"""The graded voxel with a field quadratic in the cell (p = 2): ten field functions, 35 source monomials."""

import itertools

import numpy as np
import pytest

from cubic_scattering import MaterialContrast, ReferenceMedium
from cubic_scattering.graded_voxel.basis import (
    SOURCE_EXPONENTS_CUBIC,
    contrast_values,
    gram_test,
    gram_test_source,
    monomials,
    product_table,
    source_expansion,
    source_exponents,
)
from cubic_scattering.graded_voxel.blocks import coupling_block, far_block, near_block
from cubic_scattering.graded_voxel.fft import offset_blocks, solve_graded_sphere_fft
from cubic_scattering.graded_voxel.solver import plane_wave_moments, solve_graded_sphere

REF = ReferenceMedium(5000.0, 3000.0, 2500.0)
CON = MaterialContrast(Dlambda=2.0e9, Dmu=1.0e9, Drho=100.0)
KHAT = np.array([1.0, 0.0, 0.0])
OMEGA, H = 150.0, 1.25


def _gauss_cell(n=8):
    x, w = np.polynomial.legendre.leggauss(n)
    xi = np.stack(np.meshgrid(x, x, x, indexing="ij"), -1).reshape(-1, 3)
    return xi, np.einsum("i,j,k->ijk", w, w, w).ravel()


def test_there_are_35_source_monomials_and_the_first_twenty_are_unchanged():
    exps = source_exponents(35)
    assert len(exps) == 35
    assert exps[:20] == SOURCE_EXPONENTS_CUBIC
    assert sorted(exps[20:]) == sorted(e for e in itertools.product(range(5), repeat=3) if sum(e) == 4)


def test_quadratic_grams_match_quadrature():
    xi, w = _gauss_cell()
    q = contrast_values(xi)
    np.testing.assert_allclose(gram_test(H, 10), H**3 * (q * w) @ q.T, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(gram_test(H, 10)[:4, :4], gram_test(H), rtol=1e-15)
    ms = monomials(source_exponents(35), xi)
    np.testing.assert_allclose(gram_test_source(H, 35, 10), H**3 * (q * w) @ ms.T, rtol=1e-13, atol=1e-14)


def test_product_table_reproduces_field_times_contrast():
    xi = np.random.default_rng(0).uniform(-1, 1, (60, 3))
    q = contrast_values(xi)
    for n_contrast, n_field, n_src in ((10, 10, 35), (4, 10, 20), (10, 4, 20), (4, 4, 10)):
        table = product_table(n_contrast, n_field)
        assert table.shape == (n_src, n_contrast, n_field)
        prod = np.einsum("cab,cn->abn", table, monomials(source_exponents(n_src), xi))
        np.testing.assert_allclose(prod, q[:n_contrast, None] * q[None, :n_field], rtol=1e-12, atol=1e-13)


def test_source_expansion_with_a_quadratic_field():
    rng = np.random.default_rng(1)
    assert source_expansion(rng.normal(size=(10, 9, 9)), 10).shape == (35, 10, 9, 9)
    assert source_expansion(rng.normal(size=(4, 9, 9)), 10).shape == (20, 10, 9, 9)
    assert source_expansion(rng.normal(size=(4, 9, 9))).shape == (10, 4, 9, 9)


def test_plane_wave_moments_of_the_quadratic_field_match_quadrature():
    xi, w = _gauss_cell(12)
    centres = np.array([[1.0, -2.0, 0.5]])
    k = np.array([0.31, -0.17, 0.23])
    amp = np.arange(1, 10) + 0.5j
    got = plane_wave_moments(centres, H, k, amp, 10)
    assert got.shape == (1, 10, 9)
    phase = np.exp(1j * (centres[0] + H * xi) @ k)
    ref = H**3 * (contrast_values(xi) * w) @ phase
    np.testing.assert_allclose(got[0], ref[:, None] * amp[None, :], rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(got[:, :4], plane_wave_moments(centres, H, k, amp), rtol=1e-14)


def test_the_linear_field_rows_are_unchanged():
    for off in ((0, 0, 0), (1, 0, -1), (2, 1, 0)):
        k4 = coupling_block(off, H, OMEGA, REF, n_source=20)
        k10 = coupling_block(off, H, OMEGA, REF, n_source=35, n_test=10)
        assert k10.shape == (10, 35, 9, 9)
        assert np.linalg.norm(k10[:4, :20] - k4) / np.linalg.norm(k4) < 1e-12


def test_quadratic_rows_match_the_6d_reference_beyond_contact():
    for off in ((2, 0, 0), (2, -2, 1)):
        a = near_block(off, H, OMEGA, REF, n_q=14, n_source=35, n_test=10)
        b = far_block(off, H, OMEGA, REF, 10, n_source=35, n_test=10)
        c = coupling_block(off, H, OMEGA, REF, n_source=35, n_test=10)
        for row in range(4, 10):
            scale = np.linalg.norm(b[row])
            assert np.linalg.norm(a[row] - b[row]) / scale < 1e-8, (off, row)
            assert np.linalg.norm(c[row] - b[row]) / scale < 1e-8, (off, row)


def _reexpand(values, n_out, sigma):
    """R with f_a((xi_s + sigma) / 2) = sum_b R[a, b] f_b(xi_s), by least squares on random points."""
    xi_s = np.random.default_rng(7).uniform(-1, 1, (200, 3))
    big = values((xi_s + np.asarray(sigma)) / 2.0)[:n_out]
    sub = values(xi_s)[:n_out]
    return np.linalg.lstsq(sub.T, big.T, rcond=None)[0].T


@pytest.mark.parametrize("off", [(0, 0, 0), (1, 0, 0), (1, 1, 1)])
def test_subdivision_identity_with_a_quadratic_field(off):
    sigmas = list(itertools.product((-1, 1), repeat=3))
    exps = source_exponents(35)
    c_t = {s: _reexpand(contrast_values, 10, s) for s in sigmas}
    d_s = {s: _reexpand(lambda x: monomials(exps, x), 35, s) for s in sigmas}
    big = coupling_block(off, H, OMEGA, REF, n_source=35, n_test=10)
    acc = np.zeros_like(big)
    cache: dict = {}
    for s, s2 in itertools.product(sigmas, sigmas):
        rsub = tuple(int(2 * o + (a - b) // 2) for o, a, b in zip(off, s, s2, strict=True))
        if rsub not in cache:
            cache[rsub] = coupling_block(rsub, H / 2, OMEGA, REF, n_source=35, n_test=10)
        acc += np.einsum("ab,bdij,cd->acij", c_t[s], cache[rsub], d_s[s2])
    worst = max(
        np.linalg.norm(acc[a, c] - big[a, c]) / np.linalg.norm(big[0, 0])
        for a in range(10)
        for c in range(35)
    )
    assert worst < 1e-7, worst


def test_symmetry_generated_blocks_with_a_quadratic_field_equal_direct_ones():
    blocks = offset_blocks(3, H, OMEGA, REF, n_source=35, n_test=10)
    for off in ((1, 0, -1), (-1, 1, 1), (2, -1, 0), (0, -2, 1)):
        direct = coupling_block(off, H, OMEGA, REF, n_source=35, n_test=10)
        assert np.linalg.norm(blocks[off] - direct) / np.linalg.norm(direct) < 1e-11, off


def _profile(pos):
    return 1.0 - 0.004 * float(pos @ pos)


def test_fft_solve_equals_the_dense_solve_with_a_quadratic_field():
    dense = solve_graded_sphere(OMEGA, 10.0, REF, CON, 3, _profile, KHAT, KHAT, "P", p=2, r=2)
    fft = solve_graded_sphere_fft(
        OMEGA, 10.0, REF, CON, 3, _profile, KHAT, KHAT, "P", p=2, r=2, gmres_tol=1e-13
    )
    assert dense.psi.shape[1:] == (10, 9)
    rel = np.linalg.norm(fft.psi - dense.psi) / np.linalg.norm(dense.psi)
    assert rel < 1e-10, rel
