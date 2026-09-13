"""Tests for the Cartesian directional sweeps."""

import numpy as np
import pytest

from cubic_scattering.directional_sweeps import make_sweep_grid, sweep_x
from cubic_scattering.effective_contrasts import ReferenceMedium
from cubic_scattering.sweep_kernels import lateral_split_9x9

REF = ReferenceMedium(alpha=5.0, beta=3.0, rho=2.5)
OMEGA = 2 * np.pi * (1.0 + 0.03j)
PITCH = 0.25


def _splits(grid):
    """Both lateral splits for a grid."""
    right = lateral_split_9x9(grid.ky, grid.kz_nodes, grid.pitch, OMEGA, REF, direction="right")
    left = lateral_split_9x9(grid.ky, grid.kz_nodes, grid.pitch, OMEGA, REF, direction="left")
    return right, left


def _pairwise_lateral(sources, grid, split_right, split_left):
    """O(N^2) reference: sum the split kernel over every ordered pair.

    Deliberately a different algorithm from the sweep -- a double loop over
    pairs, not a running accumulation -- so agreement is evidence.
    """
    n_z, n_x, _ = sources.shape
    out = np.zeros_like(sources)
    w = grid.kz_weights
    for i in range(n_x):
        for j in range(n_x):
            if i == j:
                continue
            n = abs(i - j)
            split = split_right if j < i else split_left
            kern = np.einsum("abk,k->abk", split.amp_p, split.phase_p**n) + np.einsum(
                "abk,k->abk", split.amp_s, split.phase_s**n
            )
            block = np.einsum("k,abk->ab", w, kern)
            out[:, i, :] += sources[:, j, :] @ block.T
    return out


def test_sweep_x_equals_pairwise_sum_distinct_sources() -> None:
    """RUNG 2: the running sweep resums the pairwise double sum exactly.

    DISTINCT source at every site -- that is the disorder-resolved property
    being claimed. See test_uniform_source_control_would_be_vacuous.
    """
    rng = np.random.default_rng(20260913)
    n_z, n_x = 2, 9
    grid = make_sweep_grid(n_z, n_x, PITCH, ky=0.6, n_kz=512)
    right, left = _splits(grid)

    sources = rng.standard_normal((n_z, n_x, 9)) + 1j * rng.standard_normal((n_z, n_x, 9))

    got = sweep_x(sources, grid, right, left)
    want = _pairwise_lateral(sources, grid, right, left)
    # 1e-12, not 1e-14: both paths sum n_kz quadrature terms in different orders,
    # and the residual grows as sqrt(n_kz) (measured: 4e-14 at n_kz=64 rising to
    # 2e-13 at n_kz=2048) while staying FLAT in n_x. That is round-off in the
    # quadrature, not error in the resummation -- which is what
    # test_residual_is_flat_in_lattice_size pins down.
    assert np.abs(got - want).max() / np.abs(want).max() < 1e-12


def test_residual_is_flat_in_lattice_size() -> None:
    """RUNG 2, the discriminating half: the residual must not grow with n_x.

    The sweep replaces an O(n_x^2) pairwise sum by two O(n_x) recursions. If the
    recursion were wrong -- a factor accumulated that should be applied once, a
    term added before being read instead of after -- the discrepancy would grow
    with the number of sites swept. Holding n_kz fixed isolates that from the
    quadrature round-off.
    """
    rng = np.random.default_rng(99)
    residuals = []
    for n_x in (4, 8, 16, 32):
        grid = make_sweep_grid(1, n_x, PITCH, ky=0.6, n_kz=128)
        right, left = _splits(grid)
        sources = rng.standard_normal((1, n_x, 9)) + 1j * rng.standard_normal((1, n_x, 9))
        got = sweep_x(sources, grid, right, left)
        want = _pairwise_lateral(sources, grid, right, left)
        residuals.append(np.abs(got - want).max() / np.abs(want).max())

    # An eight-fold increase in lattice size must not move the residual by an
    # order of magnitude.
    assert max(residuals) / min(residuals) < 10.0
    assert max(residuals) < 1e-12


def test_uniform_source_control_would_be_vacuous() -> None:
    """MANDATORY CONTROL: show the uniform-source version cannot discriminate.

    An implementation that averaged the sites before sweeping still matches the
    pairwise sum when every source is identical. This test asserts that the weak
    version of rung 2 passes for a deliberately WRONG implementation, so a
    future edit cannot quietly downgrade the real test to the weak one.
    """
    n_z, n_x = 1, 6
    grid = make_sweep_grid(n_z, n_x, PITCH, ky=0.6, n_kz=256)
    right, left = _splits(grid)

    uniform = np.ones((n_z, n_x, 9), dtype=complex)
    averaged = np.broadcast_to(uniform.mean(axis=1, keepdims=True), uniform.shape).copy()

    a = sweep_x(uniform, grid, right, left)
    b = sweep_x(averaged, grid, right, left)
    assert np.abs(a - b).max() == 0.0  # indistinguishable -- hence vacuous

    rng = np.random.default_rng(7)
    varied = rng.standard_normal((n_z, n_x, 9)) + 0j
    v_avg = np.broadcast_to(varied.mean(axis=1, keepdims=True), varied.shape).copy()
    c = sweep_x(varied, grid, right, left)
    d = sweep_x(v_avg, grid, right, left)
    assert np.abs(c - d).max() / np.abs(c).max() > 1e-2  # the real test discriminates


def test_self_term_is_never_formed() -> None:
    """A source at one site alone must produce no field AT that site."""
    n_z, n_x = 1, 5
    grid = make_sweep_grid(n_z, n_x, PITCH, ky=0.6, n_kz=256)
    right, left = _splits(grid)

    sources = np.zeros((n_z, n_x, 9), dtype=complex)
    sources[0, 2, :] = 1.0
    out = sweep_x(sources, grid, right, left)
    assert np.abs(out[0, 2, :]).max() == 0.0
    assert np.abs(out[0, 1, :]).max() > 0.0  # neighbours DO see it
    assert np.abs(out[0, 3, :]).max() > 0.0


def test_lateral_coupling_decays_with_separation() -> None:
    """Sanity: a single source is felt less further away, in both directions."""
    n_z, n_x = 1, 7
    grid = make_sweep_grid(n_z, n_x, PITCH, ky=0.6, n_kz=256)
    right, left = _splits(grid)

    sources = np.zeros((n_z, n_x, 9), dtype=complex)
    sources[0, 3, :] = 1.0
    out = np.abs(sweep_x(sources, grid, right, left)[0]).max(axis=1)
    assert out[4] > out[5] > out[6]
    assert out[2] > out[1] > out[0]


def test_grid_rejects_single_site_row() -> None:
    with pytest.raises(ValueError, match="n_x"):
        make_sweep_grid(2, 1, PITCH, ky=0.6)


def test_sweep_rejects_mismatched_pitch() -> None:
    grid = make_sweep_grid(1, 4, PITCH, ky=0.6, n_kz=64)
    wrong_r = lateral_split_9x9(grid.ky, grid.kz_nodes, 2 * PITCH, OMEGA, REF, direction="right")
    wrong_l = lateral_split_9x9(grid.ky, grid.kz_nodes, 2 * PITCH, OMEGA, REF, direction="left")
    with pytest.raises(ValueError, match="pitch"):
        sweep_x(np.zeros((1, 4, 9), dtype=complex), grid, wrong_r, wrong_l)


def test_sweep_rejects_two_splits_of_the_same_direction() -> None:
    """Passing the same split twice silently symmetrises the field."""
    grid = make_sweep_grid(1, 4, PITCH, ky=0.6, n_kz=64)
    right, _ = _splits(grid)
    with pytest.raises(ValueError, match="direction"):
        sweep_x(np.zeros((1, 4, 9), dtype=complex), grid, right, right)


def test_sweep_rejects_wrong_source_shape() -> None:
    grid = make_sweep_grid(1, 4, PITCH, ky=0.6, n_kz=64)
    right, left = _splits(grid)
    with pytest.raises(ValueError, match="sources"):
        sweep_x(np.zeros((1, 4, 6), dtype=complex), grid, right, left)
