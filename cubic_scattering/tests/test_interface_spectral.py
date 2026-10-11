"""Tests for the frequency-sweep form of the interface blocks (``interface_spectral``).

Validates:
  1. The vertical exponential moments against direct quadrature, on both branches and both signs
  2. The harmonic form of the 9-state rotation, at azimuths it was not sampled at
  3. The radial rule integrates a square-root branch point to rounding
  4. The Chebyshev table of the lateral harmonics S_n against direct evaluation, between its nodes
  5. The frame-and-harmonics route of D against the mode sum at every (q, phi), reflected and transmitted
  6. Identical media: a transmitted block equals Paper 2's homogeneous block (slow)
  7. A source in B (mirrored onto A) equals Paper 2's block at the opposite offset (slow)
"""

import mpmath as mp
import numpy as np
import pytest

from cubic_scattering import ReferenceMedium
from cubic_scattering import interface_spectral as isp
from cubic_scattering.graded_voxel.blocks import coupling_block

A = ReferenceMedium(5.0, 3.0, 2.5)
B = ReferenceMedium(6.5, 3.6, 2.8)
H = 0.05


@pytest.mark.parametrize("c", [0.0, 0.3, -2.0, 7.9, 8.1, -15.0, 40.0, -300.0])
def test_exp_moments(c):
    got = isp._exp_moments(4, np.array([c]))[:, 0]
    sg = 1.0 if c >= 0 else -1.0
    for n in range(5):
        with mp.workdps(30):
            ref = float(mp.quad(lambda v, n=n: v**n * mp.exp(c * (v - sg)), [-1, 1]))
        assert abs(got[n] - ref) <= 1e-14 * max(1.0, abs(ref)) + 1e-15


def test_harmonic_rotation():
    rng = np.random.default_rng(1)
    G = rng.normal(size=(9, 9)) + 1j * rng.normal(size=(9, 9))
    H5 = isp._harmonic_rotation()
    for phi in (0.123, 1.0, 2.9, -2.2):
        T = isp.rotation(np.array([phi]))[0]
        direct = T @ G @ T.T
        harm = np.einsum("n,nadbc,bc->ad", np.exp(1j * isp._HARM * phi), H5, G)
        assert np.abs(harm - direct).max() < 1e-13 * np.abs(direct).max()


def test_radial_rule_branch_point():
    # int_0^3 1/sqrt|q - 1| dq = 2 + 2 sqrt(2)
    x, w = isp.radial_rule(3.0, [1.0], width=0.5)
    assert abs(np.sum(w / np.sqrt(np.abs(x - 1.0))) - (2 + 2 * np.sqrt(2))) < 1e-12


@pytest.mark.parametrize("off", [(0, 0), (2, 0), (2, -4)])
def test_lateral_table(off):
    tab = isp.lateral_harmonics([off], H, 10.0, n_source=10, n_test=4)
    qh = np.array([0.0, 0.37, 1.9, 4.444, 7.01, 9.99])
    got = isp._interpolate(tab[off], qh)
    exps_f, exps_s = (
        isp.SOURCE_EXPONENTS_QUARTIC[:4],
        isp.SOURCE_EXPONENTS_QUARTIC[:10],
    )
    ref = np.stack([isp._lateral_harmonics_at(x, off, H, exps_f, exps_s) for x in qh])
    assert np.abs(got - ref).max() < 1e-13 * np.abs(ref).max()


@pytest.mark.parametrize("pair", [((-1, 2, 0), (-1, 0, 0)), ((1, 2, 2), (-1, 0, 0))])
def test_frame_route_against_direct(pair):
    new = isp.truncated_blocks([pair], H, 10.0, A, B, q0h=10.0)[0]
    old = isp._truncated_blocks_direct([pair], H, 10.0, A, B, q0h=10.0)[0]
    # the reflected mode sums carry ~2e-11 of rounding (P and SV cancel), in either route
    assert np.abs(new - old).max() < 1e-10 * np.abs(old).max()


@pytest.mark.slow
def test_identical_media_transmitted():
    rec, src = (1, 0, 0), (-1, 0, 0)
    k = isp.interface_block(rec, src, H, 10.0, A, A, n_source=10, n_test=1, q0h=10.0)
    ref = coupling_block((1, 0, 0), H, 10.0, A, n_source=10, n_test=1)
    assert np.abs(k - ref).max() < 1e-9 * np.abs(ref).max()


@pytest.mark.slow
def test_mirror_identical_media():
    # a source in B, mirrored onto A by the code, against Paper 2's block at the opposite offset
    k = isp.interface_block(
        (-1, 0, 0), (1, 0, 0), H, 10.0, A, A, n_source=10, n_test=1, q0h=10.0
    )
    ref = coupling_block((-1, 0, 0), H, 10.0, A, n_source=10, n_test=1)
    assert np.abs(k - ref).max() < 1e-9 * np.abs(ref).max()
