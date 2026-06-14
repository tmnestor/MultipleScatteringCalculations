"""test_cube_multipole_far_field.py

Tests for the validated cube radiation primitives (radiation_I1d, radiation_monomial).
"""

import numpy as np

from cubic_scattering.cube_radiation import (
    radiation_I1d,
    radiation_monomial,
)


def _quad_I1d(p: int, k: float, a: float) -> complex:
    """Reference 1D integral int_{-a}^{a} t**p exp(-i k t) dt via dense quadrature."""
    t = np.linspace(-a, a, 200001)
    return np.trapezoid(t**p * np.exp(-1j * k * t), t)


def test_radiation_I1d_matches_quadrature():
    for p in (0, 1, 2):
        for k in (0.0, 1e-7, 0.013, 0.37, 2.4, -1.1):
            a = 1.7
            got = radiation_I1d(p, k, a)
            if k == 0.0:
                exact = {0: 2.0 * a, 1: 0.0, 2: 2.0 * a**3 / 3.0}[p]
                assert abs(got - exact) < 1e-12, (p, k, got, exact)
            else:
                ref = _quad_I1d(p, k, a)
                assert abs(got - ref) < 1e-8, (p, k, got, ref)


def test_radiation_I1d_smallk_limits_exact():
    a = 0.9
    assert abs(radiation_I1d(0, 0.0, a) - 2.0 * a) < 1e-14
    assert abs(radiation_I1d(1, 0.0, a) - 0.0) < 1e-14
    assert abs(radiation_I1d(2, 0.0, a) - 2.0 * a**3 / 3.0) < 1e-14


def _quad_monomial(exp, k_sc, a, n=121):
    """3D Gauss-Legendre reference for int_cube r^exp exp(-i k_sc . r') d3r'."""
    nodes, weights = np.polynomial.legendre.leggauss(n)
    t = nodes * a
    w = weights * a
    e1, e2, e3 = exp
    f0 = (t**e1) * np.exp(-1j * k_sc[0] * t) * w
    f1 = (t**e2) * np.exp(-1j * k_sc[1] * t) * w
    f2 = (t**e3) * np.exp(-1j * k_sc[2] * t) * w
    return f0.sum() * f1.sum() * f2.sum()


def test_radiation_monomial_matches_gauss():
    a = 1.3
    k_sc = np.array([0.41, -0.22, 0.67])
    for exp in [
        (0, 0, 0),
        (1, 0, 0),
        (0, 2, 0),
        (2, 0, 0),
        (1, 1, 0),
        (0, 1, 1),
        (2, 0, 0),
        (0, 0, 2),
        (1, 0, 1),
    ]:
        got = radiation_monomial(exp, k_sc, a)
        ref = _quad_monomial(exp, k_sc, a)
        assert abs(got - ref) < 1e-10, (exp, got, ref)


def test_radiation_monomial_zero_k_is_volume_moment():
    a = 2.0
    k0 = np.array([0.0, 0.0, 0.0])
    # const -> volume (2a)**3; r0**2 -> (2a)**2 * 2a**3/3
    assert abs(radiation_monomial((0, 0, 0), k0, a) - (2 * a) ** 3) < 1e-12
    assert (
        abs(radiation_monomial((2, 0, 0), k0, a) - (2 * a) ** 2 * (2 * a**3 / 3))
        < 1e-10
    )
