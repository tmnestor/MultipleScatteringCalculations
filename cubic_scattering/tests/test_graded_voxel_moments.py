"""Closed-form master integrals of the graded voxel: monomials times odd powers of the distance."""

import json
from pathlib import Path

import mpmath as mp
import numpy as np
import pytest
from scipy import integrate

from cubic_scattering import ReferenceMedium
from cubic_scattering.graded_voxel.blocks import (
    family_tables,
    near_block,
    near_block_series,
    radial_monomials,
    static_term_integral,
    static_term_integral_closed,
    static_term_table,
)
from cubic_scattering.graded_voxel.moments import box_integral, face_integral, line_integral

MASTER = Path(__file__).resolve().parents[2] / "Mathematica" / "GradedVoxel_master_integrals.json"
REF = ReferenceMedium(5000.0, 3000.0, 2500.0)


@pytest.fixture(autouse=True)
def _forty_digits():
    with mp.workdps(40):
        yield


@pytest.mark.parametrize(
    ("r", "a2", "c", "m"),
    [(0, 5, 2, -1), (0, 5, 2, -3), (0, 4, 4, 1), (0, 8, 2, 3), (1, 4, 2, -3), (3, 20, 4, -1), (6, 4, 2, 1)]
    + [(2, 0, 2, -1), (4, 0, 4, -3), (0, 0, 2, 1)],
)
def test_line_integral_matches_quadrature(r, a2, c, m):
    ref, _ = integrate.quad(lambda z: z**r * (a2 + z * z) ** (m / 2), 0, c, epsabs=0, epsrel=1e-13)
    assert abs(float(line_integral(r, a2, c, m)) / ref - 1) < 1e-11


@pytest.mark.parametrize(
    ("q", "r", "a", "b", "c", "m"),
    [
        (0, 0, 2, 2, 4, -3),
        (0, 0, 2, 2, 4, -1),
        (0, 0, 4, 2, 2, 1),
        (1, 0, 2, 4, 2, -3),
        (2, 1, 2, 2, 2, -1),
        (3, 4, 4, 2, 4, -3),
        (5, 2, 2, 4, 4, 1),
        (0, 3, 2, 2, 2, -3),
        (0, 0, 0, 2, 4, -1),
        (1, 1, 0, 2, 2, -3),
        (2, 3, 0, 4, 2, -3),
    ],
)
def test_face_integral_matches_quadrature(q, r, a, b, c, m):
    def f(z, y):
        return y**q * z**r * (a * a + y * y + z * z) ** (m / 2)

    ref, _ = integrate.dblquad(f, 0, b, 0, c, epsabs=0, epsrel=1e-11)
    assert abs(float(face_integral(q, r, a, b, c, m)) / ref - 1) < 1e-8


@pytest.mark.parametrize(
    ("p", "q", "r", "a", "b", "c", "m"),
    [(1, 1, 0, 2, 2, 2, -1), (2, 0, 1, 2, 4, 2, -3), (1, 2, 3, 4, 2, 2, -3), (0, 2, 2, 2, 2, 4, 1)],
)
def test_box_integral_matches_quadrature(p, q, r, a, b, c, m):
    def f(z, y, x):
        return x**p * y**q * z**r * (x * x + y * y + z * z) ** (m / 2)

    ref, _ = integrate.tplquad(f, 0, a, 0, b, 0, c, epsabs=0, epsrel=1e-9)
    assert abs(float(box_integral(p, q, r, a, b, c, m)) / ref - 1) < 1e-7


def test_box_integrals_are_symmetric_under_permuting_the_axes():
    v = box_integral(3, 1, 2, 2, 4, 2, -3)
    assert mp.almosteq(v, box_integral(1, 3, 2, 4, 2, 2, -3), rel_eps=mp.mpf(10) ** -25)
    assert mp.almosteq(v, box_integral(2, 1, 3, 2, 4, 2, -3), rel_eps=mp.mpf(10) ** -25)


def test_homogeneity_in_the_box_size():
    # x^p y^q z^r R^m has degree p + q + r + m; the box integral scales with one more power per axis
    small = box_integral(1, 2, 0, 1, 1, 2, -1)
    big = box_integral(1, 2, 0, 2, 2, 4, -1)
    assert mp.almosteq(big, small * mp.mpf(2) ** (1 + 2 + 0 - 1 + 3), rel_eps=mp.mpf(10) ** -25)


def test_coulomb_self_energy_of_the_cube_in_closed_form():
    # int int_{V x V} 1/|x - x'| over the cube of half-width 1 is int prod_i (2 - |s_i|) / |s| ds over
    # [-2, 2]^3: eight octants, each a sum of box integrals over [0, 2]^3 with p, q, r <= 1
    total = mp.mpf(0)
    for p in (0, 1):
        for q in (0, 1):
            for r in (0, 1):
                coef = mp.mpf(2) ** (3 - p - q - r) * (-1) ** (p + q + r)
                total += coef * box_integral(p, q, r, 2, 2, 2, -1)
    total *= 8
    s2, s3 = mp.sqrt(2), mp.sqrt(3)
    exact = 2 * ((1 + s2 - 2 * s3) / 5 - mp.pi / 3 + mp.log((1 + s2) * (2 + s3))) * mp.mpf(2) ** 5
    assert mp.almosteq(total, exact, rel_eps=mp.mpf(10) ** -25)


def test_the_solid_angle_of_the_octant():
    # sum_i d_i d_i (1/r) = -4 pi delta: by the divergence theorem the flux of x/r^3 through the three
    # far faces of [0, a]^3 is the octant's solid angle pi/2, whatever the box
    for a, b, c in ((2, 2, 2), (2, 4, 2), (4, 2, 4)):
        flux = (
            a * face_integral(0, 0, a, b, c, -3)
            + b * face_integral(0, 0, b, a, c, -3)
            + c * face_integral(0, 0, c, a, b, -3)
        )
        assert mp.almosteq(flux, mp.pi / 2, rel_eps=mp.mpf(10) ** -25)


def test_divergent_integrals_are_refused():
    with pytest.raises(ValueError, match="diverges"):
        box_integral(0, 0, 0, 2, 2, 2, -3)
    with pytest.raises(ValueError, match="diverges"):
        face_integral(0, 0, 0, 2, 2, -3)
    with pytest.raises(ValueError, match="odd"):
        box_integral(0, 0, 0, 2, 2, 2, -2)


def test_values_are_floats_to_double_precision():
    assert np.isfinite(float(box_integral(4, 4, 4, 4, 4, 4, -3)))


def test_master_integrals_match_direct_integration():
    # the same integrals taken directly, one variable at a time, with none of the reductions used in
    # ``moments``: symbolically where that finishes, else numerically (GradedVoxel_MasterIntegrals.wl)
    rows = json.loads(MASTER.read_text())
    assert len(rows) >= 20
    for row in rows:
        fn = box_integral if row["kind"] == "box" else face_integral
        got = fn(*row["args"])
        want = mp.mpf(row["value"].split("`")[0])
        # the symbolic results are exact; the numerical fallback is good to about 13 digits
        tol = mp.mpf(10) ** (-25 if row["how"] == "symbolic" else -12)
        assert mp.almosteq(got, want, rel_eps=tol), row


@pytest.mark.parametrize("off", [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, -1, 1)])
def test_closed_static_terms_equal_the_quadrature(off):
    # every singular static term d^idx r^m of the Kelvin kernel, for all 4 x 20 polynomial pairs
    h = 1.25
    worst = 0.0
    for m, idx in static_term_table(REF.alpha, REF.beta, REF.rho):
        quad = static_term_integral(m, idx, off, h, 14, 20)
        closed = static_term_integral_closed(m, idx, off, h, 20)
        scale = np.abs(quad).max()
        if scale > 0:
            worst = max(worst, np.abs(closed - quad).max() / scale)
    assert worst < 1e-9, worst


def test_closed_static_terms_with_a_quadratic_field():
    h = 0.7
    for m, idx in ((1, (0, 0, 1, 2)), (-1, (0, 1)), (1, (2, 2)), (1, (0, 1, 1))):
        quad = static_term_integral(m, idx, (1, 0, -1), h, 14, 35, 10)
        closed = static_term_integral_closed(m, idx, (1, 0, -1), h, 35, 10)
        assert np.abs(closed - quad).max() / np.abs(quad).max() < 1e-9, (m, idx)


def test_near_block_by_closed_forms_equals_the_quadrature_block():
    for off in ((0, 0, 0), (1, 1, 0)):
        quad = near_block(off, 1.25, 150.0, REF, n_q=14)
        closed = near_block(off, 1.25, 150.0, REF, n_q=14, static="closed")
        assert np.linalg.norm(closed - quad) / np.linalg.norm(quad) < 1e-10, off


def test_radial_monomials_reproduce_the_derivatives():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(6, 3))
    r = np.linalg.norm(x, axis=1)
    eps = 1e-4
    for m, idx in ((1, (0, 1)), (3, (0, 0, 2)), (5, (0, 1, 1, 2)), (-1, (2,))):
        got = sum(c * np.prod(x ** np.array(al), axis=1) * r**n for c, al, n in radial_monomials(m, idx))
        # one central difference of the lower-order derivative
        lower = radial_monomials(m, idx[:-1])
        e = np.zeros(3)
        e[idx[-1]] = eps

        def val(y, lower=lower):
            ry = np.linalg.norm(y, axis=1)
            return sum(c * np.prod(y ** np.array(al), axis=1) * ry**n for c, al, n in lower)

        fd = (val(x + e) - val(x - e)) / (2 * eps)
        np.testing.assert_allclose(got, fd, rtol=1e-6, atol=1e-8)


def test_family_tables_rebuild_the_static_table():
    ta, tb = family_tables()
    pref = 1.0 / (4.0 * np.pi * REF.mu)
    b2 = -(1.0 - REF.beta**2 / REF.alpha**2) / 2.0
    static = static_term_table(REF.alpha, REF.beta, REF.rho)
    for (m, idx), coef in static.items():
        want = pref * ta[idx] if m == -1 else pref * b2 * tb[idx]
        np.testing.assert_allclose(coef, want, rtol=1e-14, atol=1e-30)
    assert len(static) == len(ta) + len(tb)


@pytest.mark.parametrize("off", [(0, 0, 0), (1, 0, 0), (1, 1, -1)])
def test_series_near_block_equals_the_quadrature_block(off):
    # static and dynamic parts both from the universal moments, against Duffy and Gauss quadrature
    for h, omega in ((1.25, 150.0), (1.25, 1200.0)):
        quad = near_block(off, h, omega, REF, n_q=14)
        series = near_block_series(off, h, omega, REF)
        assert np.linalg.norm(series - quad) / np.linalg.norm(quad) < 1e-10, (off, omega)


def test_series_near_block_with_a_quadratic_field():
    quad = near_block((1, 1, 0), 1.25, 600.0, REF, n_q=14, n_source=35, n_test=10)
    series = near_block_series((1, 1, 0), 1.25, 600.0, REF, n_source=35, n_test=10)
    assert np.linalg.norm(series - quad) / np.linalg.norm(quad) < 1e-10
