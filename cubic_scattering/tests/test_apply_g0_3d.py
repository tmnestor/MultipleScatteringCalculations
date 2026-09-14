"""Tests for the composed three-dimensional G0.

Two independent claims, because they fail in different ways:

  * VALUE -- the composed operator equals an explicit O(N^2) pairwise sum over
    the closed form. Catches a wrong propagator, a wrong shift, a transposed
    separation index.
  * PARTITION -- every ordered pair is reached exactly ONCE and the self-site
    exactly as often as the physics says. This is a support COUNT, an integer
    identity, not a tolerance. A pair counted twice or dropped changes the
    answer by an O(1) amount that no norm test distinguishes from a slightly
    wrong kernel, so it is checked separately and exactly.

The self-site deserves its own sentence. The whole-space same-depth table
poisons dx = dy = 0 with NaN because that term belongs inside T0. The layered
diagonal DOES carry it, because the self-return off a layer boundary is not in
T0 either. So without a model the self-site must be reached zero times, and
with one, exactly once -- and if the whole-space self-term were ever summed,
the NaN would propagate and every test here would fail loudly rather than
quietly absorbing it.
"""

import numpy as np
import pytest

from cubic_scattering.directional_sweeps import (
    G0Cache3D,
    SweepGrid3D,
    apply_g0_3d,
    build_g0_cache_3d,
)
from cubic_scattering.effective_contrasts import ReferenceMedium
from cubic_scattering.horizontal_greens import exact_propagator_9x9

REF = ReferenceMedium(5.0, 3.0, 2.5)
OM = 2 * np.pi * 6.0 * (1 + 0.03j)
PITCH = 0.5


def _grid(n_z: int = 3, n_x: int = 3, n_y: int = 2) -> SweepGrid3D:
    return SweepGrid3D(n_z=n_z, n_x=n_x, n_y=n_y, pitch=PITCH)


def test_matches_an_explicit_pairwise_sum() -> None:
    """VALUE: whole-space case against a brute-force O(N^2) sum."""
    grid = _grid()
    cache = build_g0_cache_3d(grid, REF, OM)

    rng = np.random.default_rng(20260914)
    shape = (grid.n_z, grid.n_x, grid.n_y, 9)
    src = rng.normal(size=shape) + 1j * rng.normal(size=shape)

    got = apply_g0_3d(src, cache)

    want = np.zeros_like(got)
    for lz in range(grid.n_z):
        for ix in range(grid.n_x):
            for iy in range(grid.n_y):
                for mz in range(grid.n_z):
                    for jx in range(grid.n_x):
                        for jy in range(grid.n_y):
                            if (lz, ix, iy) == (mz, jx, jy):
                                continue
                            p = exact_propagator_9x9(
                                (ix - jx) * PITCH,
                                (iy - jy) * PITCH,
                                (lz - mz) * PITCH,
                                OM,
                                REF,
                            )
                            want[lz, ix, iy] += p @ src[mz, jx, jy]

    assert np.isfinite(got).all(), "NaN leaked -- the poisoned self-term was summed"
    err = np.abs(got - want).max() / np.abs(want).max()
    assert err < 1e-13, f"composed G0 != pairwise sum: {err:.3e}"


def _counting_cache(grid: SweepGrid3D, *, with_reverberation: bool) -> G0Cache3D:
    """A cache whose every table entry is the all-ones 9x9.

    Turns the operator into a pure contribution COUNTER: applying it to a
    source that is 1 in component 0 at every site makes each output component
    the integer number of (target, source) contributions.
    """
    ones = np.ones((9, 9), dtype=complex)
    sd = np.broadcast_to(ones, (2 * grid.n_x - 1, 2 * grid.n_y - 1, 9, 9)).copy()
    sd[grid.n_x - 1, grid.n_y - 1] = np.nan  # keep the poison
    ls = np.zeros((grid.n_z, grid.n_z, 2 * grid.n_x - 1, 2 * grid.n_y - 1, 9, 9), dtype=complex)
    for lz in range(grid.n_z):
        for mz in range(grid.n_z):
            if lz == mz and not with_reverberation:
                continue
            ls[lz, mz] = ones
    return G0Cache3D(grid=grid, same_depth=sd, layered=ls)


@pytest.mark.parametrize("with_reverberation", [False, True])
def test_partition_counts_every_pair_exactly_once(with_reverberation: bool) -> None:
    """PARTITION: an exact integer identity, not a tolerance.

    COUNTS CONTRIBUTIONS, NOT PAIRS, and the two differ on purpose. A same-plane
    pair is split across two tables -- the whole-space half from `same_depth`,
    the reverberation half from the layered diagonal -- so it contributes TWICE.
    A different-plane pair contributes once, because `layered_stack_table`
    already adds the closed form back in real space inside its own block. The
    self-site contributes once with a layered background (reverberation only)
    and not at all without one.

    So per target, with n_p = n_x * n_y sites in a plane and N = n_z * n_p:

        (N - n_p) * 1        different plane, one block each
      + (n_p - 1) * 2        same plane, whole space + reverberation
      + 1                    the self-return, if there is layering

    Getting this wrong in the obvious way -- expecting N - 1, as though each
    pair contributed once -- is what the first draft of this test did.
    """
    grid = _grid()
    cache = _counting_cache(grid, with_reverberation=with_reverberation)
    n_p = grid.n_x * grid.n_y
    n_sites = grid.n_z * n_p

    src = np.zeros((grid.n_z, grid.n_x, grid.n_y, 9), dtype=complex)
    src[..., 0] = 1.0

    got = apply_g0_3d(src, cache)
    assert np.isfinite(got).all(), "NaN leaked -- the poisoned self-term was summed"

    if with_reverberation:
        expected = (n_sites - n_p) + 2 * (n_p - 1) + 1
    else:
        expected = (n_sites - n_p) + (n_p - 1)
    counts = np.rint(got.real).astype(int)
    assert (counts == expected).all(), (
        f"partition broken: expected {expected} contributions per target, "
        f"got min {counts.min()} max {counts.max()}"
    )
    assert np.abs(got.imag).max() < 1e-9


def test_rejects_a_source_of_the_wrong_shape() -> None:
    """Fail fast, with the four-element diagnostic the project requires."""
    grid = _grid()
    cache = build_g0_cache_3d(grid, REF, OM)
    with pytest.raises(ValueError) as exc:
        apply_g0_3d(np.zeros((grid.n_z, grid.n_x, grid.n_y, 6), dtype=complex), cache)
    msg = str(exc.value)
    assert "Where:" in msg
    assert "Valid:" in msg
    assert "Fix:" in msg
