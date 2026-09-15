"""Tests for the 9x9 Bubnov-Galerkin inter-cell propagator.

The load-bearing tests are the two invariants the construction does NOT impose:
transpose symmetry under R -> -R, and agreement of the constant-mode block with
the doubly-averaged operator. Both emerge from the integration rather than being
built in, so they catch a wrong autocorrelation, a wrong trial function, or a
mishandled singularity.
"""

import numpy as np
import pytest

from cubic_scattering.effective_contrasts import ReferenceMedium
from cubic_scattering.galerkin_propagator import (
    autocorrelation,
    basis_terms,
    far_field_moment,
    galerkin_block_9x9,
    gram_diagonal,
)
from cubic_scattering.slab_scattering import _cell_averaged_propagator

REF = ReferenceMedium(5000.0, 3000.0, 2500.0)
OMEGA, D = 60.0, 1.0
A = 0.5 * D


def test_shear_trial_functions_are_two_term() -> None:
    """Engineering shear needs 0.5(s_q e_p + s_p e_q), not a single monomial.

    The general derivation writes basis functions as p_alpha(r) e_{d(alpha)} --
    one monomial, one direction. Taken literally for the shear modes that would
    silently HALVE the shear sector, which is the channel the residual lives in.
    """
    for alpha in range(3):
        assert len(basis_terms(alpha)) == 1
    for alpha in (3, 4, 5):  # axial: zz, xx, yy
        assert len(basis_terms(alpha)) == 1
    for alpha in (6, 7, 8):  # shear
        terms = basis_terms(alpha)
        assert len(terms) == 2
        assert all(abs(c - 0.5) < 1e-15 for c, _, _ in terms)


def test_trial_functions_give_the_engineering_voigt_identity() -> None:
    """Symmetric gradients of the strain modes must be the Voigt unit vectors."""

    def phi(alpha: int, s: np.ndarray) -> np.ndarray:
        v = np.zeros(3)
        for c, e, dirn in basis_terms(alpha):
            v[dirn] += c * (s[0] ** e[0]) * (s[1] ** e[1]) * (s[2] ** e[2])
        return v

    h = 1e-6
    for alpha in range(3, 9):
        g = np.zeros((3, 3))
        for j in range(3):
            e = np.zeros(3)
            e[j] = h
            g[:, j] = (phi(alpha, e) - phi(alpha, -e)) / (2 * h)
        eps = 0.5 * (g + g.T)
        voigt = np.array([eps[0, 0], eps[1, 1], eps[2, 2], 2 * eps[1, 2], 2 * eps[0, 2], 2 * eps[0, 1]])
        want = np.zeros(6)
        want[alpha - 3] = 1.0
        np.testing.assert_allclose(voigt, want, atol=1e-6)


def test_autocorrelation_matches_the_derivations_worked_forms() -> None:
    """All three closed forms in the derivation, at machine precision."""
    rng = np.random.default_rng(7)
    u = rng.uniform(-2 * A, 2 * A, size=(200, 3))
    b = A - 0.5 * np.abs(u)

    tent = np.prod(2.0 * A - np.abs(u), axis=-1)
    np.testing.assert_allclose(autocorrelation((0, 0, 0), (0, 0, 0), u, A), tent, rtol=1e-14)

    mixed = -u[:, 1] * b[:, 1] * (2 * b[:, 0]) * (2 * b[:, 2])
    np.testing.assert_allclose(autocorrelation((0, 0, 0), (0, 1, 0), u, A), mixed, rtol=1e-14)

    axial = (2 * b[:, 1] ** 3 / 3 - u[:, 1] ** 2 * b[:, 1] / 2) * (2 * b[:, 0]) * (2 * b[:, 2])
    np.testing.assert_allclose(autocorrelation((0, 1, 0), (0, 1, 0), u, A), axial, rtol=1e-14)


@pytest.mark.parametrize(
    "r_vec",
    [
        np.array([0.0, 2.0 * D, 0.0]),
        np.array([D, 2.0 * D, 0.0]),
        np.array([0.0, D, 0.0]),  # face contact
        np.array([0.0, D, D]),  # edge contact
    ],
)
def test_reciprocity_bare_transpose(r_vec: np.ndarray) -> None:
    """Gamma(-R) = Gamma(R)^T, with NO metric.

    A Bubnov form uses the same family on both sides, so the bilinear form is
    symmetric under (alpha, R) <-> (beta, -R) directly. Decorating this with the
    9-component metric M = Sigma J -- which belongs to Foldy-Lax self-adjointness,
    not to the bare propagator -- gives a statement that cannot hold, since it
    would require M to commute with Gamma.
    """
    fwd = galerkin_block_9x9(r_vec, D, OMEGA, REF, n_quad=10)
    bwd = galerkin_block_9x9(-r_vec, D, OMEGA, REF, n_quad=10)
    assert np.abs(bwd - fwd.T).max() / np.abs(fwd).max() < 1e-12


def test_constant_block_equals_the_double_average() -> None:
    """The G block must coincide with the moment operator, and only the G block.

    Constant x constant autocorrelation IS the tent IS the double cell average,
    so the two constructions must agree there exactly -- while the linear modes
    must NOT, which is the whole point of the build.
    """
    r_vec = np.array([0.0, 2.0 * D, 0.0])
    gal = galerkin_block_9x9(r_vec, D, OMEGA, REF, n_quad=12)
    mom = _cell_averaged_propagator(r_vec, D, OMEGA, REF, 8)
    a_, b_ = gal[:3, :3].ravel(), mom[:3, :3].ravel()
    c = complex(np.vdot(a_, b_) / np.vdot(a_, a_))
    assert abs(c - 1.0) < 1e-9
    assert np.abs(a_ * c - b_).max() / np.abs(b_).max() < 1e-10

    s_g, s_m = gal[3:, 3:].ravel(), mom[3:, 3:].ravel()
    cs = complex(np.vdot(s_g, s_m) / np.vdot(s_g, s_g))
    assert np.abs(s_g * cs - s_m).max() / np.abs(s_m).max() > 1e-3


def test_contact_quadrature_is_converged() -> None:
    """The apex-pyramid rule must handle the 1/rho singularity, not tolerate it."""
    r_face = np.array([0.0, D, 0.0])
    a_ = galerkin_block_9x9(r_face, D, OMEGA, REF, n_quad=8)
    b_ = galerkin_block_9x9(r_face, D, OMEGA, REF, n_quad=14)
    assert np.abs(a_ - b_).max() / np.abs(b_).max() < 1e-10


def test_basis_index_out_of_range_is_diagnostic() -> None:
    with pytest.raises(ValueError, match="basis index must be in 0..26"):
        basis_terms(27)


# ---------------------------------------------------------------------------
# T27 quadratic tier
# ---------------------------------------------------------------------------


def test_basis_is_identical_to_the_one_the_tmatrix_is_built_on() -> None:
    """All 27 trial functions must BE `_build_basis_components`'s first 27.

    The propagator and T0 have to live on one basis or the Foldy-Lax product is
    meaningless. Comparing against the actual builder rather than a second
    hardcoded list is the point: two copies of the same table drift silently,
    and a permutation here would reorder every 27-mode result without any check
    noticing.

    It also pins the fact the far-field limit turns on: the quadratic functions
    are RAW monomials, not mean-subtracted, so `s_p^2` carries a monopole
    `Int s_p^2 = V d^2/12` in T0's basis as well as in the propagator's.
    """
    from cubic_scattering.compute_gerade_blocks import _build_basis_components

    ref_basis = _build_basis_components()
    for alpha in range(27):
        # the builder stores (component, exponents, coeff); this module stores
        # (coeff, exponents, direction).
        want = sorted((c, e, d) for d, e, c in ref_basis[alpha])
        got = sorted(basis_terms(alpha))
        assert got == want, f"basis mode {alpha} differs from the T-matrix basis"


@pytest.mark.parametrize("pair", [(2, 0), (0, 2), (2, 1), (1, 2), (2, 2)])
def test_degree_two_autocorrelations_against_numerical_integration(
    pair: tuple[int, int],
) -> None:
    """The degree-2 closed forms are new algebra; nothing else validates them."""
    from scipy.integrate import quad

    pa, pb = pair
    rng = np.random.default_rng(11)
    for _ in range(4):
        uk = float(rng.uniform(-2 * A, 2 * A))
        b = A - 0.5 * abs(uk)
        num = quad(
            lambda x, p=pa, q=pb, h=uk / 2: (x + h) ** p * (x - h) ** q,
            -b,
            b,
            epsabs=1e-13,
            epsrel=1e-13,
        )[0]
        got = float(autocorrelation((pa, 0, 0), (pb, 0, 0), np.array([[uk, 0.0, 0.0]]), A)[0])
        got /= (2 * A) ** 2  # strip the two trivial axes, each contributing 2A
        assert got == pytest.approx(num, rel=1e-10, abs=1e-14)


def test_unsupported_monomial_degree_is_diagnostic() -> None:
    """Degree 3 (the T57 tier) must fail loudly, not silently mis-integrate."""
    with pytest.raises(ValueError, match="unsupported monomial pair"):
        autocorrelation((3, 0, 0), (0, 0, 0), np.array([[0.1, 0.0, 0.0]]), A)


@pytest.mark.parametrize("r_vec", [np.array([0.0, 2.0 * D, 0.0]), np.array([0.0, D, 0.0])])
def test_tiers_nest_exactly(r_vec: np.ndarray) -> None:
    """The 9x9 sub-block of the 27x27 must be the standalone 9x9, bit for bit."""
    g9 = galerkin_block_9x9(r_vec, D, OMEGA, REF, n_quad=8, n_modes=9)
    g27 = galerkin_block_9x9(r_vec, D, OMEGA, REF, n_quad=8, n_modes=27)
    np.testing.assert_array_equal(g27[:9, :9], g9)


def _moment(e: tuple[int, int, int], a: float) -> float:
    """Int s^e over [-a,a]^3, by elementary integration -- no shared code."""
    out = 1.0
    for n in e:
        if n % 2:
            return 0.0
        out *= 2.0 * a ** (n + 1) / (n + 1)
    return out


def test_gram_diagonal_is_the_integral_of_phi_dot_phi() -> None:
    """Both tiers, computed term by term rather than from the closed forms."""
    g = gram_diagonal(D, 27)
    assert g.shape == (27,)
    for alpha in range(27):
        want = sum(
            ca * cb * _moment(tuple(x + y for x, y in zip(ea, eb, strict=True)), A)
            for ca, ea, da in basis_terms(alpha)
            for cb, eb, db in basis_terms(alpha)
            if da == db
        )
        assert g[alpha] == pytest.approx(want, rel=1e-14)
    np.testing.assert_allclose(g[:9], gram_diagonal(D), rtol=1e-15)


def test_far_field_moment_is_the_lowest_surviving_moment() -> None:
    """The source scale is the lowest non-vanishing multipole, by construction.

    This is the half of the normalisation the far-field limit pins, and the half
    that is NOT the Gram -- see `scripts/gate_t27_far_field.py`.
    """
    m = far_field_moment(D, 27)
    for alpha in range(27):
        _, e, _ = basis_terms(alpha)[0]
        for order in range(3):
            vals = [
                _moment(tuple(e[k] + sum(1 for i in idx if i == k) for k in range(3)), A)
                for idx in np.ndindex(*(3,) * order)
            ]
            if any(abs(v) > 0 for v in vals):
                assert m[alpha] == pytest.approx(max(abs(v) for v in vals), rel=1e-14)
                break
        else:
            pytest.fail(f"mode {alpha} has no surviving moment up to the quadrupole")


def test_the_quadratic_tier_carries_two_different_source_scales() -> None:
    """s_p^2 is a MONOPOLE; s_p s_q is a quadrupole. One tier, two channels.

    If this ever collapses to a single scale, the quadratic modes have stopped
    radiating into the displacement channel and every 27-mode far field is wrong.
    """
    m = far_field_moment(D, 27)
    v = D**3
    for k in range(3):
        for mono in range(3):  # s_p^2
            assert m[9 + 6 * k + mono] == pytest.approx(v * D**2 / 12.0, rel=1e-14)
        for mono in range(3, 6):  # s_p s_q
            assert m[9 + 6 * k + mono] == pytest.approx(v * D**4 / 144.0, rel=1e-14)
    # ... and the monopole one coincides with the LINEAR modes' dipole scale,
    # which is why a tier-wide scale cannot be read off the tier index.
    assert m[9] == pytest.approx(m[3], rel=1e-14)


@pytest.mark.parametrize("fn", [gram_diagonal, far_field_moment])
def test_scale_helpers_reject_an_unsupported_tier(fn) -> None:
    with pytest.raises(ValueError, match="n_modes must be 9 or 27"):
        fn(D, 57)


def test_reciprocity_holds_for_the_full_27x27() -> None:
    """Bubnov symmetry is a property of the form, not of the tier."""
    for r_vec in (np.array([0.0, 2.0 * D, 0.0]), np.array([0.0, D, 0.0])):
        fwd = galerkin_block_9x9(r_vec, D, OMEGA, REF, n_quad=8, n_modes=27)
        bwd = galerkin_block_9x9(-r_vec, D, OMEGA, REF, n_quad=8, n_modes=27)
        assert np.abs(bwd - fwd.T).max() / np.abs(fwd).max() < 1e-12
