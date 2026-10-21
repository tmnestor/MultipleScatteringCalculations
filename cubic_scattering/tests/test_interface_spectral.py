"""Tests for the frequency-sweep form of the interface blocks (``interface_spectral``).

Validates:
  1. The vertical exponential moments against direct quadrature, on both branches and both signs
  2. The harmonic form of the 9-state rotation, at azimuths it was not sampled at
  3. The radial rule integrates a square-root branch point to rounding
  4. The Chebyshev table of the lateral harmonics S_n against direct evaluation, between its nodes
  5. The frame-and-harmonics route of D against the mode sum at every (q, phi), reflected and transmitted
  5a. The smooth windows: 0 below the ramp, 1 above Q0, monotone; the radial rule's plain breaks
  5b. The windowed split moves exactly the ramp integrals from D to A0, A1: D - D_chi = int chi (Y0 + w^2 Y1)
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


@pytest.mark.parametrize(
    "win",
    [isp.Window(0.5, "cinf"), isp.Window(0.25, "cinf"), isp.Window(0.5, "poly", 3)],
)
def test_window_shape(win):
    qh = np.linspace(0.0, 12.0, 1201)
    chi = win(qh, 10.0)
    assert np.all(chi[qh <= win.start * 10.0] == 0.0)
    assert np.all(chi[qh >= 10.0] == 1.0)
    assert np.all(np.diff(chi) >= 0.0)
    # symmetric about the middle of the ramp
    mid = (1 + win.start) / 2 * 10.0
    d = np.linspace(0.0, (1 - win.start) / 2 * 10.0, 7)
    assert np.allclose(win(mid + d, 10.0) + win(mid - d, 10.0), 1.0, atol=1e-15)


def test_window_rejects_bad_arguments():
    with pytest.raises(ValueError):
        isp.Window(1.0)
    with pytest.raises(ValueError):
        isp.Window(0.5, "gauss")


def test_window_ramp_must_clear_the_branch_points():
    # omega h / beta = 1 puts the last branch-point panel at q h = 2 > 0.15 * 10
    with pytest.raises(ValueError):
        isp.truncated_blocks(
            [((-1, 0, 0), (-1, 0, 0))], H, 60.0, A, B, window=isp.Window(0.15)
        )


def test_radial_rule_breaks():
    x, w = isp.radial_rule(10.0, [0.5], width=2.0, breaks=(5.0, 0.2))
    # a kink at 5 is integrated exactly by panels that end there
    assert abs(np.sum(w * np.abs(x - 5.0)) - 25.0) < 1e-12
    assert abs(np.sum(w) - 10.0) < 1e-13


@pytest.mark.slow
@pytest.mark.parametrize(
    "pair", [((-1, 0, 0), (-1, 0, 0)), ((1, 2, 0), (-1, 0, 0)), ((-1, 0, 0), (1, 2, 0))]
)
@pytest.mark.parametrize("win", [isp.Window(0.5, "cinf"), isp.Window(0.5, "poly", 3)])
def test_window_moves_the_ramp(pair, win):
    rec, src = pair
    om, q0 = 60.0, 10.0  # omega h / beta = 1: D's mode sums carry little rounding
    args = (H, om, A, B, 10, 4, q0)
    # a source in B goes through interface_block, which mirrors it onto A (A0 = A1 = 0 isolates D)
    if src[0] > 0:
        sharp = isp.interface_block(rec, src, H, om, A, B, q0h=q0, parts=(0, 0))
        win_d = isp.interface_block(
            rec, src, H, om, A, B, q0h=q0, parts=(0, 0), window=win
        )
    else:
        sharp = isp.truncated_blocks([pair], *args)[0]
        win_d = isp.truncated_blocks([pair], *args, window=win)[0]
    r0, r1 = isp.window_parts(rec, src, H, A, B, win, q0h=q0)
    moved = r0 + om**2 * r1
    assert np.abs(moved).max() > 1e-3 * np.abs(sharp).max()  # the test is not vacuous
    assert np.abs(sharp - win_d - moved).max() < 1e-11 * np.abs(sharp).max()


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


@pytest.mark.slow
@pytest.mark.parametrize("zr", [-1, 1])
def test_low_and_high_bands_make_up_d(zr):
    """D_low (per offset, band below the window) + D_high (sampled on a periodic grid) = D, reflected and
    transmitted, at offsets to 4 cells; the images of the 40-cell grid are 36 cells away."""
    omega = A.beta / (6 * H)
    window, low = isp.Window(0.25), isp.Window(0.35 / 2.5)
    offs = [(0, 0), (1, 0), (2, 1), (4, 3)]
    pairs = [((zr, 2 * a, 2 * b), (-1, 0, 0)) for a, b in offs]
    full = isp.truncated_blocks(pairs, H, omega, A, B, 10, 4, 10.0, window=window)
    table = isp.lateral_harmonics([(2 * a, 2 * b) for a, b in offs], H, 2.5, 10, 4)
    d_low = isp.truncated_blocks(
        pairs, H, omega, A, B, 10, 4, 2.5, table=table, low_pass=low
    )
    d_high = isp.high_band_blocks(
        zr,
        -1,
        40,
        H,
        omega,
        A,
        B,
        10,
        4,
        10.0,
        window=window,
        low_pass=low,
        band_qh=2.5,
    )
    k0 = np.abs(isp.image_block((zr, 0, 0), (-1, 0, 0), H, A, B, 10, 4)).max()
    for (a, b), f, lo in zip(offs, full, d_low, strict=True):
        assert np.abs(lo + d_high[a, b] - f).max() < 1e-8 * k0


def test_bands_refuse_bad_arguments():
    omega = A.beta / (2 * H)  # branch points to omega h / beta_A = 0.5
    with pytest.raises(ValueError, match="low_pass"):
        isp.truncated_blocks(
            [((-1, 0, 0), (-1, 0, 0))],
            H,
            omega,
            A,
            B,
            window=isp.Window(0.25),
            low_pass=isp.Window(0.4),
        )
    with pytest.raises(ValueError, match="branch points"):
        isp.high_band_blocks(
            -1, -1, 8, H, omega, A, B, low_pass=isp.Window(0.1), band_qh=2.5
        )


@pytest.mark.slow
def test_high_band_by_symmetry_is_the_whole_grid():
    """The wedge 0 <= ky <= kx and the eight lateral symmetries give the whole grid's D_high, at canonical and
    other offsets, reflected and transmitted."""
    omega = A.beta / (6 * H)
    kw = {"window": isp.Window(0.25), "low_pass": isp.Window(0.4), "band_qh": 2.5}
    offs = [(0, 0), (1, 0), (2, 1), (3, 3), (5, 2), (-2, 1), (1, -4)]
    for zr in (-1, 1):
        whole = isp.high_band_blocks(zr, -1, 16, H, omega, A, B, 10, 4, 10.0, **kw)
        wedge = isp.high_band_blocks(
            zr, -1, 16, H, omega, A, B, 10, 4, 10.0, offsets=offs, **kw
        )
        scale = np.abs(whole).max()
        for (a, b), blk in zip(offs, wedge, strict=True):
            assert np.abs(blk - whole[a % 16, b % 16]).max() < 1e-13 * scale
