"""Tests of the interface coupling applied through plane-wave modes (``stratified_apply``)."""

import itertools

import numpy as np
import pytest

from cubic_scattering import interface_spectral as isp
from cubic_scattering import stratified_apply as sa
from cubic_scattering.effective_contrasts import ReferenceMedium

pytest.importorskip("finufft")

A = ReferenceMedium(5.0, 3.0, 2.5)
B = ReferenceMedium(6.5, 3.6, 2.8)
H = 0.05
OMEGA = A.beta / (6 * H)
WINDOW = isp.Window(0.25)
LOW = isp.Window(0.35 / 2.5)  # the band [0.35, 2.5] / h at omega h / beta_A = 1/6


def test_factors_rebuild_the_integrand():
    """The mode and static pseudo-mode factors give ``integrand_points`` (reflected and transmitted, in the band
    and on the window's ramp), to the rounding of the modes' cancellation (the terms reach ~1e5 of the sum)."""
    rng = np.random.default_rng(1)
    qh = np.concatenate([rng.uniform(0.4, 2.5, 20), rng.uniform(2.5, 10.0, 40)])
    phi = rng.uniform(0, 2 * np.pi, len(qh))
    kx, ky = qh / H * np.cos(phi), qh / H * np.sin(phi)
    rec, src = [-1, -3, 1, 5], [-1, -5]
    f = sa.mode_factors(rec, src, kx, ky, H, OMEGA, A, B, 10, 4, 10.0, WINDOW)
    for zr, zs in itertools.product(rec, src):
        ref = isp.integrand_points(zr, zs, kx, ky, H, OMEGA, A, B, 10, 4, 10.0, WINDOW)
        got = sa.integrand_from_factors(f, zr, zs)
        assert np.abs(got - ref).max() < 1e-9 * np.abs(ref).max()


def _dense(blocks_of, x, layers, shape):
    """sum_s K(r - s) x_s over a box, K from blocks_of(zr, zs) -> callable (dx, dy)."""
    nx, ny = shape
    out = {z: np.zeros((nx, ny, 4, 9), dtype=complex) for z in layers}
    for zr, zs in itertools.product(layers, repeat=2):
        k = blocks_of(zr, zs)
        for (ir, jr), (i_s, j_s) in itertools.product(
            itertools.product(range(nx), range(ny)), repeat=2
        ):
            out[zr][ir, jr] += np.einsum(
                "acij,cj->ai", k(ir - i_s, jr - j_s), x[zs][i_s, j_s]
            )
    return out


def test_low_band_by_nufft_is_the_per_offset_product():
    """D_low x by NUFFTs over the band's polar quadrature equals the product of the per-offset blocks of
    ``truncated_blocks(low_pass=...)``, sources on both sides."""
    shape, layers = (3, 2), [-1, 1]
    rng = np.random.default_rng(3)
    x = {
        z: rng.standard_normal(shape + (10, 9))
        + 1j * rng.standard_normal(shape + (10, 9))
        for z in layers
    }
    got = sa.apply_low_band(x, layers, shape, H, OMEGA, A, B, 2.5, LOW)
    offs = [(a, b) for a in range(-2, 3) for b in range(-1, 2)]
    table = isp.lateral_harmonics([(2 * a, 2 * b) for a, b in offs], H, 2.5, 10, 4)

    def blocks_of(zr, zs):
        if zs < 0:
            got_ = isp.truncated_blocks(
                [((zr, 2 * a, 2 * b), (zs, 0, 0)) for a, b in offs],
                H,
                OMEGA,
                A,
                B,
                10,
                4,
                2.5,
                table,
                low_pass=LOW,
            )
        else:
            got_ = [
                isp._mirror(m)
                for m in isp.truncated_blocks(
                    [((-zr, 2 * a, 2 * b), (-zs, 0, 0)) for a, b in offs],
                    H,
                    OMEGA,
                    B,
                    A,
                    10,
                    4,
                    2.5,
                    table,
                    low_pass=LOW,
                )
            ]
        k = dict(zip(offs, got_, strict=True))
        return lambda a, b: k[a, b]

    ref = _dense(blocks_of, x, layers, shape)
    scale = max(np.abs(ref[z]).max() for z in layers)
    assert all(np.abs(got[z] - ref[z]).max() < 1e-12 * scale for z in layers)


def test_high_band_at_the_box_wavenumbers_is_the_periodic_kernel():
    """D_high x at the wavenumbers of a periodic box (every alias a node) equals the product of
    ``high_band_blocks`` on that box, sources on both sides."""
    shape, layers, n = (3, 2), [-1, 1], 12
    rng = np.random.default_rng(4)
    x = {
        z: rng.standard_normal(shape + (10, 9))
        + 1j * rng.standard_normal(shape + (10, 9))
        for z in layers
    }
    got = sa.apply_high_band(
        x, layers, shape, n, H, OMEGA, A, B, 10.0, WINDOW, 2.5, LOW
    )

    def blocks_of(zr, zs):
        if zs < 0:
            g = isp.high_band_blocks(
                zr, zs, n, H, OMEGA, A, B, 10, 4, 10.0, window=WINDOW, low_pass=LOW
            )
            return lambda a, b: g[a % n, b % n]
        g = isp.high_band_blocks(
            -zr, -zs, n, H, OMEGA, B, A, 10, 4, 10.0, window=WINDOW, low_pass=LOW
        )
        return lambda a, b: isp._mirror(g[a % n, b % n])

    ref = _dense(blocks_of, x, layers, shape)
    scale = max(np.abs(ref[z]).max() for z in layers)
    # the modes' cancellation at large q (measured 1.2e-11 at omega h / beta = 1/6 on a 40-cell box)
    assert all(np.abs(got[z] - ref[z]).max() < 1e-10 * scale for z in layers)


def test_whole_space_march_is_paper_2_between_planes_apart():
    """The march in depth over the low band's nodes and the box's lattice reproduces Paper 2's blocks between
    planes two or more apart (planes 0 and 1 apart are left out), to D1: the lattice's images 40 cells away cost
    ~1e-10 at this band."""
    from scipy.fft import next_fast_len

    from cubic_scattering.graded_voxel.blocks import coupling_block

    shape, layers = (2, 2), [-4, -3, -2, -1]
    rng = np.random.default_rng(5)
    x = {
        z: rng.standard_normal(shape + (10, 9))
        + 1j * rng.standard_normal(shape + (10, 9))
        for z in layers
    }
    kx, ky, w = sa.whole_space_nodes(
        2.0 * np.hypot(1, 1), next_fast_len(2 + 40), H, OMEGA, A, 2.5, LOW
    )
    got = sa.WholeSpaceMarch(layers, shape, kx, ky, w, H, OMEGA, A).apply(x)
    ref = {z: np.zeros(shape + (4, 9), dtype=complex) for z in layers}
    for zr, zs in itertools.product(layers, repeat=2):
        if abs(zr - zs) < 2:
            continue
        for (ir, jr), (i_s, j_s) in itertools.product(
            itertools.product(range(2), range(2)), repeat=2
        ):
            k = coupling_block((zr - zs, ir - i_s, jr - j_s), H, OMEGA, A, 10, 4)
            ref[zr][ir, jr] += np.einsum("acij,cj->ai", k, x[zs][i_s, j_s])
    scale = max(np.abs(ref[z]).max() for z in layers)
    assert all(np.abs(got[z] - ref[z]).max() < 1e-8 * scale for z in layers)
