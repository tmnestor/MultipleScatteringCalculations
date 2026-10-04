"""Legendre (modified) moments of S^m over boxes off the singular point, against 40-digit quadrature."""

import math

import mpmath as mp
import numpy as np
import pytest

from cubic_scattering.graded_voxel.legendre_moments import box_moments, legendre_product_matrix

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
