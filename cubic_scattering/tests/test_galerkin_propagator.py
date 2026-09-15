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
    galerkin_block_9x9,
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
    with pytest.raises(ValueError, match="basis index must be in 0..8"):
        basis_terms(9)
