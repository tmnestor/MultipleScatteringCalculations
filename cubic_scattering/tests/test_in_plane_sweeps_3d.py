"""Tests for the 3-D in-plane sweeps (the thesis's Algorithm 5.2): sweep_x_3d, sweep_y_3d."""

import numpy as np
import pytest

from cubic_scattering import directional_sweeps as ds
from cubic_scattering.effective_contrasts import ReferenceMedium
from cubic_scattering.pair_propagators import same_depth_table, separation_index
from cubic_scattering.sweep_kernels import lateral_split_9x9
from cubic_scattering.sweep_solver import solve_foldy_lax_3d

PITCH = 0.25
OMEGA = 2 * np.pi * 6.0 * (1.0 + 0.03j)
REFS = [ReferenceMedium(5.0, 3.0, 2.5), ReferenceMedium(6.5, 3.6, 2.8)]
SHAPE = (2, 3, 4)
CHUNK = 8192


def _sources(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(SHAPE + (9,)) + 1j * rng.standard_normal(SHAPE + (9,))


def _pairwise(src: np.ndarray, part: str) -> np.ndarray:
    """O(N^2) reference from the closed form: a different algorithm from the sweep, so agreement is
    evidence."""
    n_z, n_x, n_y, _ = src.shape
    out = np.zeros_like(src)
    for z in range(n_z):
        tab = same_depth_table(n_x, n_y, PITCH, OMEGA, REFS[z])
        for i, j, i2, j2 in np.ndindex(n_x, n_y, n_x, n_y):
            dx, dy = i - i2, j - j2
            if (dx, dy) == (0, 0) or (part == "x" and dx == 0) or (part == "y" and dx != 0):
                continue
            out[z, i, j] += tab[separation_index(dx, n_x), separation_index(dy, n_y)] @ src[z, i2, j2]
    return out


def test_pi9_is_the_x_y_exchange() -> None:
    """An involution that swaps u_x/u_y, e_xx/e_yy and 2e_zy/2e_zx, and fixes the rest."""
    assert np.array_equal(ds.PI9[ds.PI9], np.arange(9))
    assert [int(i) for i in ds.PI9] == [0, 2, 1, 3, 5, 4, 6, 8, 7]


def test_lateral_split_takes_one_ky_per_node() -> None:
    """A per-node ky is the scalar call at each node; a mismatched shape is refused."""
    ky = np.array([0.3, -1.1, 2.0])
    kz = np.array([0.5, 0.0, -1.7])
    got = lateral_split_9x9(ky, kz, PITCH, OMEGA, REFS[0])
    for n in range(3):
        one = lateral_split_9x9(float(ky[n]), kz[n : n + 1], PITCH, OMEGA, REFS[0])
        assert np.allclose(got.amp_p[..., n], one.amp_p[..., 0], rtol=1e-14, atol=0)
        assert np.allclose(got.phase_s[n], one.phase_s[0], rtol=1e-14, atol=0)
    with pytest.raises(ValueError, match="ky has shape"):
        lateral_split_9x9(ky[:2], kz, PITCH, OMEGA, REFS[0])


def test_a_unit_source_reaches_exactly_the_other_sites_of_its_plane() -> None:
    """The partition as a support count: every other site of the plane once, never the self-site or
    another plane."""
    nodes = ds.make_transverse_nodes(20.0 / PITCH, 96)
    for z, i, j in ((0, 1, 2), (1, 0, 0)):
        src = np.zeros(SHAPE + (9,), dtype=complex)
        src[z, i, j, 0] = 1.0
        reached_x = np.abs(ds.sweep_x_3d(src, PITCH, OMEGA, REFS, nodes, chunk=CHUNK)).max(axis=3) > 0
        reached_y = np.abs(ds.sweep_y_3d(src, PITCH, OMEGA, REFS, nodes, chunk=CHUNK)).max(axis=3) > 0
        want_x = np.zeros(SHAPE, dtype=bool)
        want_x[z] = True
        want_x[z, i, :] = False  # the left-right sweep: dx != 0 only
        want_y = np.zeros(SHAPE, dtype=bool)
        want_y[z, i, :] = True
        want_y[z, i, j] = False  # the in-out sweep: dx = 0, dy != 0
        assert np.array_equal(reached_x, want_x)
        assert np.array_equal(reached_y, want_y)


@pytest.mark.parametrize("part", ["x", "y"])
def test_each_sweep_converges_to_the_closed_form(part: str) -> None:
    """Distinct sources, two plane media: the error is the quadrature's and falls as the rule is refined."""
    src = _sources(20261018)
    want = _pairwise(src, part)
    sweep = ds.sweep_x_3d if part == "x" else ds.sweep_y_3d
    errors = []
    for n_axis in (256, 512):
        nodes = ds.make_transverse_nodes(20.0 / PITCH, n_axis)
        got = sweep(src, PITCH, OMEGA, REFS, nodes, chunk=CHUNK)
        errors.append(np.abs(got - want).max() / np.abs(want).max())
    assert errors[1] < errors[0] / 10
    assert errors[1] < 3e-4


def test_chunking_does_not_change_the_field() -> None:
    """The quadrature is a sum over nodes: any chunking gives the same field to round-off."""
    src = _sources(7)
    nodes = ds.make_transverse_nodes(20.0 / PITCH, 128)
    one = ds.in_plane_sweeps_3d(src, PITCH, OMEGA, REFS, nodes, chunk=nodes.kt.size)
    many = ds.in_plane_sweeps_3d(src, PITCH, OMEGA, REFS, nodes, chunk=1000)
    assert np.abs(one - many).max() < 1e-12 * np.abs(one).max()


def test_planes_in_one_medium_share_a_kernel() -> None:
    """Grouping planes by medium is an economy, not an approximation: the same field plane by plane."""
    src = _sources(3)
    nodes = ds.make_transverse_nodes(20.0 / PITCH, 96)
    same = [REFS[0], REFS[0]]
    both = ds.in_plane_sweeps_3d(src, PITCH, OMEGA, same, nodes, chunk=CHUNK)
    for z in range(2):
        alone = ds.in_plane_sweeps_3d(src[z : z + 1], PITCH, OMEGA, [REFS[0]], nodes, chunk=CHUNK)
        assert np.abs(both[z] - alone[0]).max() < 1e-12 * np.abs(both).max()


def test_sweeps_3d_refuse_bad_input() -> None:
    nodes = ds.make_transverse_nodes(20.0 / PITCH, 16)
    src = _sources(1)
    with pytest.raises(ValueError, match="expected \\(n_z, n_x, n_y, 9\\)"):
        ds.sweep_x_3d(src[..., :6], PITCH, OMEGA, REFS, nodes, chunk=CHUNK)
    with pytest.raises(ValueError, match="plane media"):
        ds.sweep_y_3d(src, PITCH, OMEGA, REFS[:1], nodes, chunk=CHUNK)
    with pytest.raises(ValueError, match="chunk must be"):
        ds.sweep_x_3d(src, PITCH, OMEGA, REFS, nodes, chunk=0)
    with pytest.raises(ValueError, match="kr_max > 0"):
        ds.make_transverse_nodes(0.0, 16)


def _solve_setup(scale: float):
    """The 3-D Foldy-Lax set-up of test_sweep_solver_3d: one medium, 2 x 3 x 2 voxels at pitch 0.5."""
    grid = ds.SweepGrid3D(n_z=2, n_x=3, n_y=2, pitch=0.5)
    cache = ds.build_g0_cache_3d(grid, REFS[0], OMEGA)
    rng = np.random.default_rng(20260914)
    n = (grid.n_z, grid.n_x, grid.n_y)
    t0 = scale * (rng.normal(size=(*n, 9, 9)) + 1j * rng.normal(size=(*n, 9, 9)))
    psi = rng.normal(size=(*n, 9)) + 1j * rng.normal(size=(*n, 9))
    sweeps = ds.InPlaneSweeps(
        OMEGA, (REFS[0],) * grid.n_z, ds.make_transverse_nodes(20.0 / 0.5, 512), CHUNK
    )
    return grid, cache, t0, psi, sweeps


def test_g0_with_the_sweeps_is_g0_with_the_table() -> None:
    """The sweeps replace only the same-depth table: the whole G0 agrees to the quadrature's error."""
    grid, cache, _, _, sweeps = _solve_setup(0.1)
    src = _sources(11)[:, : grid.n_x, : grid.n_y]
    table = ds.apply_g0_3d(src, cache)
    swept = ds.apply_g0_3d(src, cache, in_plane=sweeps)
    assert np.abs(swept - table).max() < 1e-3 * np.abs(table).max()
    with pytest.raises(ValueError, match="plane media"):
        ds.apply_g0_3d(src, cache, in_plane=ds.InPlaneSweeps(OMEGA, (REFS[0],), sweeps.nodes, CHUNK))


@pytest.mark.slow
def test_a_solve_with_the_sweeps_matches_the_solve_with_the_table() -> None:
    """GMRES through the sweeps reaches the table route's solution, within the operator's error amplified
    by the solve."""
    _, cache, t0, psi, sweeps = _solve_setup(0.1)
    table = solve_foldy_lax_3d(cache, t0, psi, tol=1e-10)
    swept = solve_foldy_lax_3d(cache, t0, psi, tol=1e-10, in_plane=sweeps)
    assert np.abs(swept.psi - table.psi).max() < 1e-3 * np.abs(table.psi).max()
    assert abs(swept.n_matvec - table.n_matvec) <= 2
