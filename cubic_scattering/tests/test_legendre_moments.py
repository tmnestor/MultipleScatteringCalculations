"""Legendre (modified) moments of S^m over boxes off the singular point, against 40-digit quadrature."""

import math

import mpmath as mp
import numpy as np
import pytest

from cubic_scattering.graded_voxel.legendre_moments import (
    box_moment_derivatives,
    box_moments,
    legendre_product_matrix,
)

mp.mp.dps = 30


def _edge_exact(k: int, a2: float, l: float, m: int) -> float:
    f = lambda t: mp.legendre(k, 2 * (t - l) - 1) * (a2 + t * t) ** (mp.mpf(m) / 2)  # noqa: E731
    return float(mp.quad(f, [l, l + 1]))


@pytest.mark.parametrize(
    ("a2", "l", "m"),
    [
        (2.0, 1.0, -1),  # rho = sqrt(3): backward marching would do
        (1.0, 1.0, 1),  # rho = sqrt(2): both marching directions fail for the face analogue
        (2.0, 0.0, -1),
        (0.0, 1.0, -1),  # S = t, rho = 1
        (5.0, 1.0, -3),
        (1.0, 0.0, -1),
        (8.0, 1.0, 1),
    ],
)
def test_edge_moments_to_full_relative_accuracy(a2, l, m):
    # every moment, the tiny high-order ones included, to round-off of its own size
    lam = box_moments((l,), a2, m, 32)
    for k in range(17):
        ex = _edge_exact(k, a2, l, m)
        err = abs(lam[k] - ex)
        assert err <= 5e-15 * abs(lam[0]) and err <= 1e-10 * abs(ex) + 1e-30, (k, ex)


@pytest.mark.parametrize(
    ("c", "ly", "lz", "m"),
    [(1, 1, 0, -1), (0, 1, 1, -1), (1, 0, 1, 1), (1, 0, 0, -1), (2, 1, 1, -3)],
)
def test_face_moments_including_the_near_band(c, ly, lz, m):
    # the faces at rho = sqrt(2) and 1 from the singular point, where forward and backward marching of the
    # monomial recurrences both fail
    lam = box_moments((float(ly), float(lz)), float(c * c), m, 24)
    mp.mp.dps = 18
    for q, r in ((0, 0), (2, 3), (6, 1)):
        f = lambda y, z, q=q, r=r: (  # noqa: E731
            mp.legendre(q, 2 * (y - ly) - 1)
            * mp.legendre(r, 2 * (z - lz) - 1)
            * (c * c + y * y + z * z) ** (mp.mpf(m) / 2)
        )
        ex = float(mp.quad(f, [ly, ly + 1], [lz, lz + 1]))
        assert abs(lam[q, r] - ex) <= 1e-15 * abs(lam[0, 0]) + 1e-12 * abs(ex), (q, r)
    mp.mp.dps = 30


def test_moments_refuse_a_box_with_a_corner_at_the_singular_point():
    with pytest.raises(ValueError, match="pyramids"):
        box_moments((0.0, 0.0, 0.0), 0.0, -1, 8)


def test_product_matrix_multiplies_in_the_legendre_basis():
    t = legendre_product_matrix(6, (0.5, 0.25, 0.125))
    x = np.linspace(-1, 1, 7)
    for k in range(6):
        lhs = (0.5 + 0.25 * x + 0.125 * x**2) * np.polynomial.legendre.legval(x, np.eye(6)[k])
        rhs = np.polynomial.legendre.legval(x, t[:, k])
        np.testing.assert_allclose(lhs, rhs, atol=1e-14)


def test_point_moment_is_the_corner_value():
    assert math.isclose(float(box_moments((), 4.0, -3, 5)), 4.0**-1.5)


def _mp_point_derivative(z, b, m):
    """d^b (sum z_f^2 + rest)^(m/2) w.r.t. z at 30 digits: the Hermite-type sum, exact in mp."""

    def value(z, rest2):
        s2 = sum(v * v for v in z) + rest2
        total = mp.mpf(0)
        ranges = [range(bf // 2 + 1) for bf in b]
        import itertools

        for k in itertools.product(*ranges):
            coef = mp.mpf(1)
            for bf, kf in zip(b, k, strict=True):
                coef *= mp.factorial(bf) / (mp.factorial(kf) * mp.factorial(bf - 2 * kf) * 2**kf)
            q = sum(b) - sum(k)
            ladder = mp.mpf(1)
            for j in range(q):
                ladder *= m - 2 * j
            mono = mp.mpf(1)
            for zf, bf, kf in zip(z, b, k, strict=True):
                mono *= zf ** (bf - 2 * kf)
            total += coef * mono * ladder * s2 ** ((mp.mpf(m) - 2 * q) / 2)
        return total

    return value


#: the relative floor of the moments, of the leading moment. Round-off for a few derivatives. With many
#: derivatives, or more derivatives than a small positive power, the derivative is a small difference of
#: the Leibniz terms that the relation subtracts, and the floor is the size of that difference: 2e-14 for
#: d^5 (1/r) on the near face, 4e-15 for d^4 S on a face, 6e-14 for d d^4 S^3 on an edge at |z| = 3. None
#: of these changes with the truncation (24 to 56 indices).
ROUND_OFF, SMALL_DIFFERENCE = 4e-15, 1e-13


@pytest.mark.parametrize(
    ("lows", "z", "b", "m", "tol"),
    [
        ((2.0, 0.0), (1.0,), (5,), -1, ROUND_OFF),  # a face of the face neighbour, fifth normal derivative
        ((0.0, 0.0), (1.0,), (5,), -1, SMALL_DIFFERENCE),  # the near face, from the foot of the normal
        ((0.0, 0.0), (3.0,), (3,), -3, ROUND_OFF),  # the far face
        ((2.0,), (1.0, 1.0), (2, 3), -1, ROUND_OFF),  # an edge: derivatives along both fixed axes
        ((1.0, 0.0), (1.0,), (4,), 1, SMALL_DIFFERENCE),
        ((0.0,), (1.0, 3.0), (1, 4), 3, SMALL_DIFFERENCE),
    ],
)
def test_derivative_moments_equal_high_precision_quadrature(lows, z, b, m, tol) -> None:
    """The moments of d_z^b S^m from the differentiated relation, against 30-digit Gauss."""
    n = 28
    lam = box_moment_derivatives(lows, z, b, m, n)
    f = _mp_point_derivative([mp.mpf(v) for v in z], b, m)
    xs, ws = zip(*mp.calculus.quadrature.GaussLegendre(mp.mp).calc_nodes(5, mp.mp.prec), strict=True)
    d = len(lows)
    ks = [(0,) * d, (1,) * d, (3,) + (2,) * (d - 1), (6,) * d]
    scale = 0.0
    for k in ks:
        exact = mp.mpf(0)
        import itertools

        for idx in itertools.product(range(len(xs)), repeat=d):
            u = [xs[i] for i in idx]
            w = mp.fprod(ws[i] / 2 for i in idx)
            t = [mp.mpf(l) + (uj + 1) / 2 for l, uj in zip(lows, u, strict=True)]
            leg = mp.fprod(mp.legendre(kj, uj) for kj, uj in zip(k, u, strict=True))
            exact += w * leg * f(z, sum(tj * tj for tj in t))
        if k == (0,) * d:
            scale = abs(float(exact))
        got = float(lam[k])
        assert abs(got - float(exact)) <= tol * scale, (k, got, float(exact))


def test_derivative_moments_reduce_to_box_moments_without_derivatives() -> None:
    lam = box_moment_derivatives((1.0, 0.0), (1.0, 2.0), (0, 0), -1, 12)
    assert np.array_equal(lam, box_moments((1.0, 0.0), 5.0, -1, 12))
