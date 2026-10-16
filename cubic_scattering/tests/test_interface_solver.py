"""Tests of the Galerkin solve near a welded interface (``interface_solver``); the full gate is
scripts/gate_interface_solve.py."""

import numpy as np
import pytest

from cubic_scattering import interface_solver as iso
from cubic_scattering import interface_spectral as isp
from cubic_scattering.effective_contrasts import MaterialContrast, ReferenceMedium
from cubic_scattering.graded_voxel.blocks import coupling_block
from cubic_scattering.stratified_march import traction_rows

A = ReferenceMedium(5.0, 3.0, 2.5)
B = ReferenceMedium(6.5, 3.6, 2.8)
H = 0.05
OMEGA = A.beta / (6 * H)


@pytest.mark.parametrize("wave", ["P", "SV", "SH"])
def test_reference_field_is_continuous(wave):
    """The reference waves' displacement and traction agree on both sides of z = 0."""
    kx, ky = 0.3 * OMEGA / A.alpha, -0.2 * OMEGA / A.alpha
    sides = {True: np.zeros(6, complex), False: np.zeros(6, complex)}
    for in_a, _, d in iso.reference_waves(kx, ky, OMEGA, A, B, wave):
        sides[in_a] += traction_rows(d, A if in_a else B)
    scale = np.abs(sides[True]).max()
    assert np.abs(sides[True] - sides[False]).max() < 1e-13 * scale


def test_grid_units_and_sides():
    g = iso.InterfaceGrid(np.array([[-1, 2, -3], [0, 0, 1]]), H, A, B)
    assert g.units.tolist() == [[-1, 4, -6], [1, 0, 2]]
    assert g.in_above.tolist() == [True, False]
    with pytest.raises(ValueError):
        iso.InterfaceGrid(np.array([[0, 0, 0], [0, 0, 0]]), H, A, B)


def test_symmetry_map_matches_the_direct_block():
    """A pair with lateral offset (-2, 1) cells across the interface: its block by the lateral symmetry from the
    canonical one equals the block computed directly."""
    g = iso.InterfaceGrid(np.array([[0, -2, 1], [-1, 0, 0]]), H, A, B)
    c = iso.InterfaceCouplings(g, workers=1)
    got = c.interface_block(0, 1, OMEGA)
    direct = isp.interface_block((1, -4, 2), (-1, 0, 0), H, OMEGA, A, B, 10, 4)
    assert np.abs(got - direct).max() < 1e-13 * np.abs(direct).max()


def test_identical_media_solve_is_the_homogeneous_one():
    """A = B: two cells touching across the interface solve as in the whole space (Paper 2's blocks)."""
    g = iso.InterfaceGrid(np.array([[-1, 0, 0], [0, 0, 0]]), H, A, A)
    c = iso.InterfaceCouplings(g)
    ctr = [MaterialContrast(0.1 * A.lam, 0.1 * A.mu, 0.1 * A.rho)] * 2
    kx, ky = 0.3 * OMEGA / A.alpha, 0.1 * OMEGA / A.alpha
    res = iso.solve_interface(c, OMEGA, ctr, kx, ky, "P", p=1)
    blocks = {
        (m, n): coupling_block(
            tuple(int(v) for v in g.idx[m] - g.idx[n]), H, OMEGA, A, 10, 4
        )
        for m in range(2)
        for n in range(2)
    }
    a = iso.assemble(g, blocks, res.delta, 1, 0)
    psi = np.linalg.solve(a, res.reference.ravel()).reshape(res.psi.shape)
    assert np.abs(res.psi - psi).max() < 1e-8 * np.abs(psi).max()


def test_solve_refuses_too_small_couplings():
    g = iso.InterfaceGrid(np.array([[-1, 0, 0]]), H, A, B)
    c = iso.InterfaceCouplings(g, workers=1)
    with pytest.raises(ValueError, match="p = 2"):
        iso.solve_interface(c, OMEGA, [None], 0.0, 0.0, "P", p=2)


def test_apply_coupling_is_the_assembled_coupling():
    """M x - apply_coupling(x) equals the assembled matrix times x."""
    g = iso.InterfaceGrid(np.array([[-1, 0, 0], [0, 1, 0]]), H, A, B)
    c = iso.InterfaceCouplings(g, workers=1)
    ctr = [
        MaterialContrast(0.1 * A.lam, 0.1 * A.mu, 0.1 * A.rho),
        MaterialContrast(0.1 * B.lam, 0.0, 0.1 * B.rho),
    ]
    delta = iso.cell_contrasts(g, OMEGA, ctr, 1)
    blocks = c.blocks(OMEGA)
    x = np.random.default_rng(3).standard_normal((2, 4, 9)) + 0j
    m = np.diag(iso.gram_test(H, 4))
    got = m[None, :, None] * x - iso.apply_coupling(g, blocks, delta, x, 1, 1)
    want = (iso.assemble(g, blocks, delta, 1, 1) @ x.ravel()).reshape(x.shape)
    assert np.abs(got - want).max() < 1e-13 * np.abs(want).max()


def test_blocks_match_the_direct_blocks():
    """The batched per-frequency blocks (series, symmetry, mirroring) equal the blocks pair by pair:
    Paper 2's Gauss block plus ``interface_block``, or ``interface_block`` alone across the interface."""
    g = iso.InterfaceGrid(np.array([[-1, 0, 0], [0, 1, 0], [-2, -1, 1]]), H, A, B)
    c = iso.InterfaceCouplings(g)
    got = c.blocks(OMEGA)
    u = g.units
    for (m, n), k in got.items():
        want = isp.interface_block(
            tuple(int(v) for v in u[m]),
            tuple(int(v) for v in u[n]),
            H,
            OMEGA,
            A,
            B,
            10,
            4,
        )
        if g.in_above[m] == g.in_above[n]:
            off = tuple(int(v) for v in g.idx[m] - g.idx[n])
            want = want + coupling_block(off, H, OMEGA, g.medium(m), 10, 4)
        assert np.abs(k - want).max() < 1e-11 * np.abs(want).max(), (m, n)


def test_window_needs_the_omega2_split():
    g = iso.InterfaceGrid(np.array([[-1, 0, 0]]), H, A, B)
    with pytest.raises(ValueError, match="omega4=False"):
        iso.InterfaceCouplings(g, window=isp.Window(0.5), omega4=True, workers=1)


def test_radius_needs_a_window():
    g = iso.InterfaceGrid(np.array([[-1, 0, 0]]), H, A, B)
    with pytest.raises(ValueError, match="smooth window"):
        iso.InterfaceCouplings(g, radius=16.0, workers=1)


def test_one_lateral_table_per_offset():
    """The classes of an offset share one S_n table, built as for the offset alone; beyond the radius a class has
    no corrections."""
    g = iso.InterfaceGrid(
        np.array([[-1, 0, 0], [-1, 1, 0], [0, 0, 0], [0, 1, 0]]), H, A, B
    )
    c = iso.InterfaceCouplings(g, window=isp.Window(0.25), radius=1.0)
    assert len(c.classes) == 8  # four depth pairs, offsets (0, 0) and (2, 0)
    for off in ((0, 0), (2, 0)):
        group = [v for k, v in c.classes.items() if k[2:] == off]
        assert len(group) == 4
        assert all(t is group[0][1] for _, t in group)
        want = isp.lateral_harmonics([off], H, c.q0h, c.n_source, c.n_test)
        e, co = group[0][1][off]
        assert np.array_equal(e, want[off][0]) and np.array_equal(co, want[off][1])
        assert all((p is None) == (off != (0, 0)) for p, _ in group)


def test_lateral_fft_product_and_gmres_match_the_dense_solve():
    """Cells scattered in a box with gaps, on both sides: the FFT product equals ``apply_coupling`` and the GMRES
    solve equals the dense one."""
    g = iso.InterfaceGrid(
        np.array([[-1, 0, 0], [0, 1, 0], [-2, -1, 1], [1, 2, 2], [0, 0, 0]]), H, A, B
    )
    c = iso.InterfaceCouplings(g)
    ctr = [
        MaterialContrast(
            0.1 * g.medium(m).lam, 0.1 * g.medium(m).mu, 0.1 * g.medium(m).rho
        )
        for m in range(5)
    ]
    ctr[3] = None  # a cell of the reference medium
    kx, ky = 0.3 * OMEGA / A.alpha, 0.1 * OMEGA / A.alpha
    op = iso.LateralOperator(c, OMEGA, p=1, r=1)
    delta = iso.cell_contrasts(g, OMEGA, ctr, 1)
    x = np.random.default_rng(5).standard_normal((5, 4, 9)) + 0j
    blocks = c.blocks(OMEGA)
    want = iso.apply_coupling(g, blocks, delta, x, 1, 1)
    assert np.abs(op.coupling(delta, x) - want).max() < 1e-12 * np.abs(want).max()
    dense = iso.solve_interface(c, OMEGA, ctr, kx, ky, "P", p=1, r=1, blocks=blocks)
    it, n_it = iso.solve_interface_gmres(
        c, OMEGA, ctr, kx, ky, "P", p=1, r=1, operator=op
    )
    assert np.abs(it.psi - dense.psi).max() < 1e-10 * np.abs(dense.psi).max()
    assert n_it < 50
