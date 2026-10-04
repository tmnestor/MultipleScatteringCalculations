"""Distant cells: the coupling block as a multipole series about the centre separation."""

import numpy as np
import pytest

from cubic_scattering import ReferenceMedium
from cubic_scattering.graded_voxel.blocks import coupling_block, family_tables
from cubic_scattering.graded_voxel.kernel import _helmholtz_F, kernel_9x9
from cubic_scattering.graded_voxel.multipole import (
    b_scalar_F,
    cell_pair_moments,
    far_block_multipole,
    helmholtz_F,
    piece_moments_1d,
    piecewise_multipole_block,
    radial_derivative_array,
    radial_derivatives,
    truncation_order,
)

REF = ReferenceMedium(5000.0, 3000.0, 2500.0)
OMEGA, H = 150.0, 1.25


def test_helmholtz_F_by_hankel_functions_matches_the_closed_forms():
    r = np.array([3.0, 11.0, 40.0])
    for k in (0.03, 0.4):
        want = _helmholtz_F(k, r)
        for i, ri in enumerate(r):
            got = helmholtz_F(k, float(ri), 4)
            for q in range(5):
                assert abs(got[q] / want[q][i] - 1) < 1e-11, (k, ri, q)


def test_radial_derivatives_match_finite_differences():
    k, x = 0.3, np.array([7.0, -4.0, 5.5])
    d = radial_derivatives(k, x, 3)
    eps = 1e-3
    for beta in ((1, 0, 0), (0, 2, 0), (1, 1, 1), (2, 0, 1)):
        i = int(np.argmax(beta))
        low = tuple(b - (1 if j == i else 0) for j, b in enumerate(beta))
        e = np.zeros(3)
        e[i] = eps
        fd = (radial_derivatives(k, x + e, 2)[low] - radial_derivatives(k, x - e, 2)[low]) / (2 * eps)
        assert abs(d[beta] / fd - 1) < 1e-6, beta


def test_the_propagator_is_the_family_tables_on_the_two_scalars():
    # P = (1 / 4 pi mu) sum_idx (TA[idx] d^idx g_S + TB[idx] d^idx (g_S - g_P) / k_S^2): the zeroth term
    x = np.array([9.0, 2.0, -6.0])
    ka, kb = OMEGA / REF.alpha, OMEGA / REF.beta
    ds, dp = radial_derivatives(kb, x, 4), radial_derivatives(ka, x, 4)
    ta, tb = family_tables()
    p = np.zeros((9, 9), complex)

    def beta_of(idx):
        return (idx.count(0), idx.count(1), idx.count(2))

    for idx, coef in ta.items():
        p += coef * ds[beta_of(idx)]
    for idx, coef in tb.items():
        p += coef * (ds[beta_of(idx)] - dp[beta_of(idx)]) / kb**2
    p /= 4 * np.pi * REF.mu
    want = kernel_9x9(x[None], OMEGA, REF)[0]
    assert np.abs(p - want).max() / np.abs(want).max() < 1e-11


def test_cell_pair_moments_match_quadrature():
    x, w = np.polynomial.legendre.leggauss(6)
    mu = cell_pair_moments(4, 10, 4)
    for gamma, a, c, e_t, e_s in (
        ((2, 0, 1), 1, 4, (1, 0, 0), (2, 0, 0)),
        ((0, 3, 0), 2, 2, (0, 1, 0), (0, 1, 0)),
    ):
        ref = 1.0
        for i in range(3):
            ref *= sum(
                wu * wv * u ** e_t[i] * v ** e_s[i] * (u - v) ** gamma[i]
                for u, wu in zip(x, w, strict=True)
                for v, wv in zip(x, w, strict=True)
            )
        assert abs(mu[gamma][a, c] - ref) < 1e-12 * max(1.0, abs(ref))


@pytest.mark.parametrize("off", [(8, 0, 0), (9, 5, 3), (12, -7, 0)])
def test_multipole_block_equals_the_quadrature_block(off):
    want = coupling_block(off, H, OMEGA, REF)
    got = far_block_multipole(off, H, OMEGA, REF, tol=1e-12)
    assert np.linalg.norm(got - want) / np.linalg.norm(want) < 1e-9, off


def test_multipole_block_with_a_quadratic_field():
    off = (10, 4, -2)
    want = coupling_block(off, H, 600.0, REF, n_source=35, n_test=10)
    got = far_block_multipole(off, H, 600.0, REF, n_source=35, n_test=10, tol=1e-12)
    assert np.linalg.norm(got - want) / np.linalg.norm(want) < 1e-9


def test_the_series_converges_geometrically_in_the_ratio_of_cell_to_distance():
    off = (8, 0, 0)
    want = coupling_block(off, H, OMEGA, REF)
    errs = [
        np.linalg.norm(far_block_multipole(off, H, OMEGA, REF, order=n) - want) / np.linalg.norm(want)
        for n in (4, 8, 12)
    ]
    assert errs[0] > errs[1] > errs[2]
    # the ratio of the source-plus-field cell extent to the distance: (2 sqrt 3 h) / (16 h)
    rho = 2 * np.sqrt(3) / 16
    assert errs[2] < 50 * rho**13


def test_truncation_order_and_refusal_near_contact():
    assert truncation_order((8, 0, 0), 1e-12) < truncation_order((4, 0, 0), 1e-12)
    with pytest.raises(ValueError, match="too close"):
        far_block_multipole((1, 1, 0), H, OMEGA, REF)


def test_radial_derivative_array_equals_the_dictionary():
    k, x = 0.3, np.array([7.0, -4.0, 5.5])
    arr = radial_derivative_array(k, x, 6)
    for beta, val in radial_derivatives(k, x, 6).items():
        assert abs(arr[beta] / val - 1) < 1e-12, beta


def test_piece_moments_match_quadrature():
    # int over a sub-interval of the cross-correlation w(sigma) / h times (sigma - centre)^g
    from cubic_scattering.graded_voxel.blocks import autocorrelation_1d

    x, w = np.polynomial.legendre.leggauss(30)
    for e_t, e_s in ((0, 0), (1, 2), (2, 4)):
        left, right = autocorrelation_1d(e_t, e_s)
        mom = piece_moments_1d(e_t, e_s, splits=1, order=9)  # (4 sub-intervals, 10)
        assert mom.shape == (4, 10)
        for sub, (lo, hi, poly) in enumerate(((-2, -1, left), (-1, 0, left), (0, 1, right), (1, 2, right))):
            c, r = (lo + hi) / 2, (hi - lo) / 2
            for g in (0, 3, 9):
                ref = r * sum(wi * poly(c + r * xi) * (r * xi) ** g for xi, wi in zip(x, w, strict=True))
                assert abs(mom[sub, g] - ref) < 1e-13 * max(1.0, abs(ref)), (e_t, e_s, sub, g)


@pytest.mark.parametrize("off", [(2, 0, 0), (2, 1, 1), (2, -2, 2), (3, 1, 0), (5, -2, 1)])
def test_piecewise_multipole_block_equals_the_quadrature_block(off):
    # every non-touching offset, including the nearest ones, where the series about the cell centres fails
    for omega in (150.0, 600.0):
        want = coupling_block(off, H, omega, REF)
        got = piecewise_multipole_block(off, H, omega, REF, tol=1e-11)
        assert np.linalg.norm(got - want) / np.linalg.norm(want) < 2e-9, (off, omega)


def test_piecewise_multipole_block_with_a_quadratic_field():
    off = (2, 1, 0)
    want = coupling_block(off, H, 600.0, REF, n_source=35, n_test=10)
    got = piecewise_multipole_block(off, H, 600.0, REF, n_source=35, n_test=10, tol=1e-11)
    assert np.linalg.norm(got - want) / np.linalg.norm(want) < 2e-9


def test_piecewise_multipole_refuses_touching_cells():
    with pytest.raises(ValueError, match="touch"):
        piecewise_multipole_block((1, 0, 0), H, OMEGA, REF)


@pytest.mark.parametrize("ks_r", [1e-6, 1e-3, 0.3, 0.99, 1.5])
def test_b_scalar_derivatives_have_no_cancellation(ks_r):
    # F_q of B = (g_S - g_P) / k_S^2 against 40-digit arithmetic: the difference of Hankel forms loses
    # eps / (k r)^2, so at k_S r = 1e-3 it was wrong by 1e-10 relative
    import mpmath as mp

    mp.mp.dps = 40
    r = 2.5
    ks = ks_r / r
    kp = ks * REF.beta / REF.alpha
    q_max = 8
    got = b_scalar_F(ks, kp, r, q_max)

    def f_q(k, q):
        t = mp.mpf(r)
        g = lambda x: mp.exp(1j * k * x) / x  # noqa: E731
        # (r^-1 d/dr)^q applied symbolically through mpmath differentiation of g(sqrt(2 u)) in u = r^2 / 2
        return mp.diff(lambda u: g(mp.sqrt(2 * u)), t * t / 2, q)

    for q in range(q_max + 1):
        want = (f_q(mp.mpf(ks), q) - f_q(mp.mpf(kp), q)) / mp.mpf(ks) ** 2
        assert abs(got[q] - complex(want)) <= 1e-13 * abs(complex(want)), (q, got[q], want)


@pytest.mark.parametrize("off", [(4, 3, 2), (8, 0, 0)])
def test_multipole_block_at_low_frequency(off):
    # k_S h = 1e-4: the old difference of Hankel forms gave 3e-10 and 1e-10 here
    omega = 1e-4 * REF.beta / H
    want = coupling_block(off, H, omega, REF)
    got = far_block_multipole(off, H, omega, REF, tol=1e-13)
    assert np.linalg.norm(got - want) / np.linalg.norm(want) < 1e-13


def test_piecewise_block_bisects_rather_than_raise_the_order():
    # (2, 1, 1) at k_S h = 0.5: the unbisected plan of order 48 stalled at 2.2e-12
    from cubic_scattering.graded_voxel.blocks import _sform, _to_field_rows

    h, off = 1.0, (2, 1, 1)
    omega = 0.5 * REF.beta / h
    ref = _sform(off, h, (0, 0, 0), lambda X: kernel_9x9(X, omega, REF).reshape(len(X), 81), 28, 10, 4)
    want = _to_field_rows(ref.reshape(4, 10, 9, 9))
    got = piecewise_multipole_block(off, h, omega, REF, tol=1e-13)
    assert np.linalg.norm(got - want) / np.linalg.norm(want) < 1e-13
