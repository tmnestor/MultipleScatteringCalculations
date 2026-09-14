"""Tests for the Bloch-summed Kupradze derivative tensors.

The sharp test is ETA-INDEPENDENCE at every derivative order. The Ewald split is
exact for any eta -- it only moves work between the real-space and reciprocal
halves -- so the total cannot depend on it, and a wrong fourth-derivative
recursion breaks that at order 4 while leaving the scalar untouched.

Two traps are guarded explicitly because both are already on this project's
record: the tests use z != 0 (the reciprocal erfc pairing is invisible at z = 0)
and k_par != 0 (a wrong Bloch phase sign agrees exactly at k_par = 0).
"""

import math

import numpy as np
import pytest

from cubic_scattering.effective_contrasts import ReferenceMedium
from cubic_scattering.kupradze_derivatives import radial_ladder
from cubic_scattering.lattice_kupradze import (
    direct_scalar_tensors,
    ladder_from_plain,
    lattice_block_9x9,
    lattice_scalar_tensors,
    origin_scalar_tensors,
)
from cubic_scattering.planar_ewald import ewald_total

A_L = 2.0
K_PAR = np.array([0.2, 0.1])
ETA1, ETA2 = 0.7, 1.15
RC, GC = 8, 8
KAPPA = 1.5
KAPPA_DAMPED = 1.5 + 0.25j
REF = ReferenceMedium(5.0, 3.0, 2.5)
OMEGA = 4.5

R_OFFPLANE = np.array([0.31, 0.5, 0.0])  # (z, x, y)
R_INPLANE = np.array([0.0, 0.35, 0.28])


def test_ladder_bridge_on_a_terminating_monomial() -> None:
    """(1/d d/dd)^k d^6 = 6d^4, 24d^2, 48, 0 -- exact, no special functions."""
    d = 1.37
    plain = [complex(v) for v in (d**6, 6 * d**5, 30 * d**4, 120 * d**3, 360 * d**2)]
    expected = [d**6, 6 * d**4, 24 * d**2, 48.0, 0.0]
    got = ladder_from_plain(plain, d)
    for k in range(5):
        assert got[k] == pytest.approx(expected[k], abs=1e-9)


def test_ladder_bridge_reproduces_the_hankel_ladder() -> None:
    """Plain derivatives in closed form, bridged, must match Rayleigh's formula.

    The two routes share nothing: Leibniz on exp(i kappa d) * (1/d) against the
    spherical-Hankel recursion.
    """
    d = 1.37
    plain = [
        complex(
            sum(
                math.comb(j, m)
                * (1j * KAPPA) ** (j - m)
                * np.exp(1j * KAPPA * d)
                * (-1) ** m
                * math.factorial(m)
                / d ** (m + 1)
                for m in range(j + 1)
            )
            / (4.0 * np.pi)
        )
        for j in range(5)
    ]
    bridged = ladder_from_plain(plain, d)
    closed = radial_ladder(d, KAPPA, 4)
    for k in range(5):
        assert bridged[k] == pytest.approx(closed[k], rel=1e-12)


@pytest.mark.parametrize("r_vec", [R_OFFPLANE, R_INPLANE])
def test_order_zero_matches_the_gated_scalar_ewald(r_vec: np.ndarray) -> None:
    """The n = 0 tensor IS the scalar lattice sum -- and checks the (z,x,y) order.

    `planar_ewald` is (x, y, z); this module is (z, x, y). A permutation slip
    would show up here and nowhere else.
    """
    mine = lattice_scalar_tensors(r_vec, KAPPA, ETA1, RC, GC, A_L, K_PAR, order=0)[0]
    xyz = np.array([r_vec[1], r_vec[2], r_vec[0]])
    theirs = ewald_total(KAPPA, xyz, ETA1, RC, GC, A_L, K_PAR)
    assert complex(mine) == pytest.approx(theirs, rel=1e-12)


@pytest.mark.parametrize("order", [0, 1, 2, 3, 4])
def test_eta_independence_at_every_order(order: int) -> None:
    """The load-bearing invariant: the split parameter cannot affect the total."""
    a = lattice_scalar_tensors(R_OFFPLANE, KAPPA, ETA1, RC, GC, A_L, K_PAR, order)[order]
    b = lattice_scalar_tensors(R_OFFPLANE, KAPPA, ETA2, RC, GC, A_L, K_PAR, order)[order]
    assert np.abs(a - b).max() / np.abs(a).max() < 1e-11


@pytest.mark.parametrize("order", [0, 2, 4])
def test_against_a_converged_damped_direct_sum(order: int) -> None:
    """An independent construction, shown converged in its own radius first."""
    near = direct_scalar_tensors(R_OFFPLANE, KAPPA_DAMPED, 30, A_L, K_PAR, order)[order]
    far = direct_scalar_tensors(R_OFFPLANE, KAPPA_DAMPED, 45, A_L, K_PAR, order)[order]
    arbiter_drift = np.abs(far - near).max() / np.abs(far).max()
    assert arbiter_drift < 1e-6, "arbiter not converged; it cannot convict anything"

    ewald = lattice_scalar_tensors(R_OFFPLANE, KAPPA_DAMPED, ETA1, RC, GC, A_L, K_PAR, order)[order]
    assert np.abs(ewald - far).max() / np.abs(far).max() < 1e-9


def test_z_parity_holds_at_the_same_plane_point() -> None:
    """Every lattice vector is in-plane, so the sum depends on z only via z^2.

    At z = 0 every tensor entry with an odd number of z indices must vanish.
    Nothing in the construction imposes this, so it is genuine evidence.
    """
    tens = lattice_scalar_tensors(R_INPLANE, KAPPA, ETA1, RC, GC, A_L, K_PAR)
    for order in range(1, 5):
        t = tens[order]
        scale = np.abs(t).max()
        for idx in np.ndindex(*((3,) * order)):
            if idx.count(0) % 2 == 1:
                assert abs(complex(t[idx])) / scale < 1e-12


@pytest.mark.parametrize("order", [0, 1, 2, 3, 4])
def test_origin_value_is_eta_independent(order: int) -> None:
    """The diagonal term, where both singular pieces cancel analytically."""
    a = origin_scalar_tensors(KAPPA, ETA1, RC, GC, A_L, K_PAR, order)[order]
    b = origin_scalar_tensors(KAPPA, ETA2, RC, GC, A_L, K_PAR, order)[order]
    assert np.abs(a - b).max() / np.abs(a).max() < 1e-11


@pytest.mark.parametrize("order", [0, 2, 4])
def test_r_to_zero_approaches_the_origin_value_first_order(order: int) -> None:
    """The gap must HALVE as |r| halves -- being close is not enough.

    A wrong constant plus cancellation noise also looks 'close'; only the rate
    shows the analytic limit is the right one.
    """
    origin = origin_scalar_tensors(KAPPA, ETA1, RC, GC, A_L, K_PAR)[order]
    scale = np.abs(origin).max()
    gaps = []
    for frac in (0.05, 0.025, 0.0125):
        near = lattice_scalar_tensors(np.array([0.0, frac * A_L, 0.0]), KAPPA, ETA1, RC, GC, A_L, K_PAR)[
            order
        ]
        gaps.append(np.abs(near - origin).max() / scale)
    for i in range(len(gaps) - 1):
        assert 1.7 < gaps[i] / gaps[i + 1] < 2.3


def test_origin_z_parity_and_bloch_phase_signature() -> None:
    """z-parity holds always; odd ORDERS vanish only at k_par = 0.

    The contrast matters: a check run at k_par = 0 alone would also pass with
    the Bloch phase sign reversed, which is a defect already on this project's
    record.
    """
    origin = origin_scalar_tensors(KAPPA, ETA1, RC, GC, A_L, K_PAR)
    for order in range(1, 5):
        t = origin[order]
        scale = np.abs(t).max()
        for idx in np.ndindex(*((3,) * order)):
            if idx.count(0) % 2 == 1:
                assert abs(complex(t[idx])) / scale < 1e-12

    at_gamma = origin_scalar_tensors(KAPPA, ETA1, RC, GC, A_L, np.zeros(2))
    for order in (1, 3):
        assert np.abs(at_gamma[order]).max() / np.abs(at_gamma[order - 1]).max() < 1e-12
        assert np.abs(origin[order]).max() / np.abs(origin[order - 1]).max() > 1e-3


def test_block_9x9_dispatches_to_the_origin_limit_at_zero() -> None:
    """r = 0 must be finite -- the un-regularised path would divide by zero."""
    blk = lattice_block_9x9(np.zeros(3), OMEGA, REF, ETA1, RC, GC, A_L, K_PAR)
    assert np.all(np.isfinite(blk))
    other = lattice_block_9x9(np.zeros(3), OMEGA, REF, ETA2, RC, GC, A_L, K_PAR)
    assert np.abs(blk - other).max() / np.abs(blk).max() < 1e-10


def test_same_plane_9x9_is_finite_and_eta_independent() -> None:
    """The assembled block, which is the object build_slab_kernels needs."""
    a = lattice_block_9x9(R_INPLANE, OMEGA, REF, ETA1, RC, GC, A_L, K_PAR)
    b = lattice_block_9x9(R_INPLANE, OMEGA, REF, ETA2, RC, GC, A_L, K_PAR)
    assert np.all(np.isfinite(a))
    assert np.abs(a - b).max() / np.abs(a).max() < 1e-10
