"""Tests for cubic_scattering.special: spherical_jn safe against a complex argument's negative zero."""

import numpy as np

from cubic_scattering.special import spherical_jn
from cubic_scattering.stratified_march import receiver_moments, source_moments_local


def _closed(n: int, z):
    """j_0 .. j_2 in closed form."""
    z = np.asarray(z, dtype=complex)
    s, c = np.sin(z), np.cos(z)
    return {0: s / z, 1: s / z**2 - c / z, 2: (3 / z**2 - 1) * s / z - 3 * c / z**2}[n]


def test_a_negative_zero_imaginary_part_does_not_flip_the_sign() -> None:
    """The case SciPy gets wrong: -x - 0j on the negative real axis, every order, scalars and arrays."""
    pts = np.array(
        [complex(-0.3, -0.0), complex(-2.5, -0.0), complex(-7.1, -0.0), -0.3 + 0j, 0.3 + 0j, 1.5 - 0.4j]
    )
    for n in range(3):
        got = spherical_jn(n, pts)
        assert np.allclose(got, _closed(n, pts), rtol=1e-12, atol=1e-14)
        assert np.isclose(spherical_jn(n, complex(-0.3, -0.0)), _closed(n, -0.3), rtol=1e-12)


def test_real_arguments_are_untouched() -> None:
    x = np.linspace(-6.0, 6.0, 41)
    from scipy.special import spherical_jn as scipy_jn

    for n in range(4):
        assert np.array_equal(spherical_jn(n, x), scipy_jn(n, x))


def test_the_cell_moments_match_brute_force_at_negated_wavenumbers() -> None:
    """The moments that broke: a source's exp(-i k.x) takes -k h, so a positive lateral wavenumber held
    as complex arrives as -kx - 0j. Both moments against Gauss quadrature, real and evanescent."""
    from numpy.polynomial.legendre import Legendre, leggauss

    h = 0.05
    x, w = leggauss(30)
    k = np.array(
        [[4.0 + 0j, 7.0 + 0j, 3.0 + 0j], [-5.0 + 0j, 2.0 + 0j, -6.0 + 0j], [30.0j, 11.0 + 0j, 9.0 + 0j]]
    )
    rec = receiver_moments(k, h, 4)
    src = source_moments_local(k, h, 10)
    expo_r = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    for p, kv in enumerate(k):
        one = [np.exp(1j * kv[i] * h * x) for i in range(3)]
        for a, e in enumerate(expo_r):
            want = h**3 * np.prod([np.sum(w * Legendre.basis(e[i])(x) * one[i]) for i in range(3)])
            assert np.isclose(rec[p, a], want, rtol=1e-11, atol=0)
        want0 = h**3 * np.prod([np.sum(w * np.exp(-1j * kv[i] * h * x)) for i in range(3)])
        assert np.isclose(src[p, 0], want0, rtol=1e-11, atol=0)
