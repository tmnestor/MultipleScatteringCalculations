"""Tests for the Kupradze derivative ladder.

The load-bearing test is equivalence with `_propagator_block_9x9`, which reached
the same 9x9 by a different route (hand-derived radial phi/psi with 3 and 7
tensor structures). Agreement at machine precision exercises the derivative
ladder to FOURTH order -- the order the strain-strain block of the lattice sum
needs and the order where hand transcription fails silently.
"""

import math

import numpy as np
import pytest

from cubic_scattering.effective_contrasts import ReferenceMedium
from cubic_scattering.kupradze_derivatives import (
    _delta_x_structure,
    _pairings,
    greens_from_scalars,
    propagator_block_9x9_kupradze,
    radial_ladder,
    scalar_derivative_tensors,
)
from cubic_scattering.resonance_tmatrix import _propagator_block_9x9

REF = ReferenceMedium(5000.0, 3000.0, 2500.0)
OMEGA = 60.0
SEPARATIONS = [
    np.array([0.0, 200.0, 0.0]),
    np.array([200.0, 200.0, 0.0]),
    np.array([200.0, 400.0, 600.0]),
    np.array([-300.0, 150.0, -450.0]),
]


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_pairing_multiplicities(n: int) -> None:
    """Distinct delta/x structures number n! / (2^m m! (n-2m)!)."""
    for m in range(n // 2 + 1):
        expected = math.factorial(n) // (2**m * math.factorial(m) * math.factorial(n - 2 * m))
        assert len(_pairings(list(range(n)), m)) == expected


@pytest.mark.parametrize("n", [2, 3, 4])
def test_structures_fully_symmetric(n: int) -> None:
    """Derivative tensors of a scalar are symmetric in every index pair."""
    x = np.array([0.31, -0.72, 0.55], dtype=complex)
    for m in range(n // 2 + 1):
        struct = _delta_x_structure(n, m, x)
        for a in range(n):
            for b in range(a + 1, n):
                axes = list(range(n))
                axes[a], axes[b] = axes[b], axes[a]
                np.testing.assert_allclose(struct, struct.transpose(axes), atol=1e-14)


def test_radial_ladder_base_case() -> None:
    """f_0 is the scalar Helmholtz kernel itself."""
    r, kappa = 137.0, 0.02
    expected = np.exp(1j * kappa * r) / (4.0 * np.pi * r)
    assert radial_ladder(r, kappa, 0)[0] == pytest.approx(expected, rel=1e-13)


def test_ladder_against_a_polynomial_identity() -> None:
    """(1/r d/dr) applied numerically must reproduce the ladder's next rung.

    This checks the Rayleigh/Hankel recursion independently of the tensor
    structures, using a plain derivative of the previous rung.
    """
    r, kappa, h = 137.0, 0.02, 1e-3
    for k in range(4):
        lo = radial_ladder(r - h, kappa, k)[k]
        hi = radial_ladder(r + h, kappa, k)[k]
        numeric = (hi - lo) / (2.0 * h) / r
        assert radial_ladder(r, kappa, k + 1)[k + 1] == pytest.approx(numeric, rel=1e-6)


@pytest.mark.parametrize("r_vec", SEPARATIONS)
def test_matches_validated_point_propagator(r_vec: np.ndarray) -> None:
    """The load-bearing equivalence, to fourth order, at machine precision."""
    mine = propagator_block_9x9_kupradze(r_vec, OMEGA, REF)
    theirs = _propagator_block_9x9(r_vec, OMEGA, REF)
    rel = np.abs(mine - theirs).max() / np.abs(theirs).max()
    assert rel < 1e-13


def test_complex_omega_is_finite_and_tends_to_lossless() -> None:
    """Attenuative media must work, and depart from lossless as 1/Q."""
    products = []
    for q in (200.0, 2000.0, 20000.0):
        damped = propagator_block_9x9_kupradze(SEPARATIONS[2], OMEGA * (1 + 0.5j / q), REF)
        lossless = propagator_block_9x9_kupradze(SEPARATIONS[2], OMEGA, REF)
        assert np.all(np.isfinite(damped))
        products.append(np.abs(damped - lossless).max() / np.abs(lossless).max() * q)
    assert products[-1] == pytest.approx(products[0], rel=0.05)


def test_zero_separation_is_rejected_with_a_diagnostic() -> None:
    """R = 0 is singular; it must fail loudly, not return a plausible number."""
    with pytest.raises(ValueError, match="separation is zero"):
        scalar_derivative_tensors(np.zeros(3), 0.02)


def test_greens_from_scalars_is_linear_in_the_scalar_tensors() -> None:
    """The Kupradze operator has constant coefficients, so it must be linear.

    This is what lets the SAME assembly consume Bloch-summed scalars: summation
    commutes with a linear operator. If this failed, the lattice route would be
    invalid however good the scalar sums were.
    """
    r_a, r_b = SEPARATIONS[1], SEPARATIONS[2]
    kp, ks = OMEGA / REF.alpha, OMEGA / REF.beta
    a_p = scalar_derivative_tensors(r_a, kp)
    a_s = scalar_derivative_tensors(r_a, ks)
    b_p = scalar_derivative_tensors(r_b, kp)
    b_s = scalar_derivative_tensors(r_b, ks)

    summed = greens_from_scalars(
        [x + y for x, y in zip(a_p, b_p, strict=True)],
        [x + y for x, y in zip(a_s, b_s, strict=True)],
        OMEGA,
        REF,
    )
    separate = [
        u + v
        for u, v in zip(
            greens_from_scalars(a_p, a_s, OMEGA, REF),
            greens_from_scalars(b_p, b_s, OMEGA, REF),
            strict=True,
        )
    ]
    for got, want in zip(summed, separate, strict=True):
        np.testing.assert_allclose(got, want, rtol=1e-13, atol=0.0)
