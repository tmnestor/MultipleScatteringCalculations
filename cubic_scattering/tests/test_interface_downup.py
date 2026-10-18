"""Tests of the two-way march preconditioner (``interface_downup``)."""

import numpy as np
import pytest

from cubic_scattering import interface_downup as idu
from cubic_scattering import interface_solver as iso
from cubic_scattering.effective_contrasts import MaterialContrast, ReferenceMedium
from cubic_scattering.graded_voxel.basis import source_expansion

A = ReferenceMedium(5.0, 3.0, 2.5)
B = ReferenceMedium(6.5, 3.6, 2.8)
H = 0.05
OMEGA = A.beta / (2 * H)


def test_fgmres_solves_a_complex_system():
    rng = np.random.default_rng(1)
    n = 60
    a = np.eye(n) * 4 + rng.standard_normal((n, n)) / np.sqrt(n) * (1 + 1j)
    b = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    want = np.linalg.solve(a, b)
    x, n_mv, hist = idu.fgmres(lambda v: a @ v, lambda v: v, b, tol=1e-12, restart=7)
    assert np.abs(x - want).max() < 1e-10 * np.abs(want).max()
    assert hist[-1] <= 1e-12 or np.linalg.norm(a @ x - b) <= 1e-12 * np.linalg.norm(b)
    inv = np.linalg.inv(a)
    x, n_mv, _ = idu.fgmres(lambda v: a @ v, lambda v: inv @ v, b, tol=1e-12)
    assert n_mv <= 3 and np.abs(x - want).max() < 1e-10 * np.abs(want).max()


@pytest.fixture(scope="module")
def box():
    """Four planes (two each side) of 2 x 2 cells, a gap in one, at omega h / beta_A = 1/2."""
    cells = [
        (iz, ix, iy) for iz in (-2, -1, 0, 1) for ix in range(2) for iy in range(2)
    ]
    cells.remove((0, 1, 1))
    g = iso.InterfaceGrid(np.array(cells), H, A, B)
    c = iso.InterfaceCouplings(g)
    op = iso.LateralOperator(c, OMEGA, p=1, r=0, parts=True)
    return g, c, op


def _contrasts(g, f):
    return [
        MaterialContrast(f * g.medium(m).lam, f * g.medium(m).mu, f * g.medium(m).rho)
        for m in range(len(g.idx))
    ]


def test_the_split_adds_up_to_the_coupling(box):
    g, _, op = box
    e = np.zeros((len(g.idx), op.n_source, op.na, 9, 9), dtype=complex)
    du = idu.DownUp(op, e)
    for zr in op.layers:
        for zs in op.layers:
            got = sum(parts.get((zr, zs), 0) for parts in (du.lo, du.up, du.refl)) + (
                du.k0[zr] if zr == zs else 0
            )
            assert (
                np.abs(got - op.kernels[zr, zs]).max()
                <= 1e-12 * np.abs(op.kernels[zr, zs]).max()
            )
    # forward is downward in L, upward in U; R stays on one side and off the diagonal
    assert all(zs < zr for zr, zs in du.lo) and all(zs > zr for zr, zs in du.up)
    assert all((zr < 0) == (zs < 0) and zr != zs for zr, zs in du.refl)


def test_downup_passes_converge_to_the_dense_solve(box):
    """As a stationary iteration with exact plane solves, DownUp converges to the solution at small contrast."""
    g, c, op = box
    ctr = _contrasts(g, 0.1)
    kx, ky = 0.3 * OMEGA / A.alpha, 0.1 * OMEGA / A.alpha
    dense = iso.solve_interface(c, OMEGA, ctr, kx, ky, "P", p=1, r=0)
    delta = iso.cell_contrasts(g, OMEGA, ctr, 0)
    e = np.array(
        [source_expansion(d, op.n_field)[: op.n_source, : op.na] for d in delta]
    )
    rhs = iso.reference_moments(g, OMEGA, kx, ky, "P", op.n_field)[:, : op.na]
    du = idu.DownUp(op, e, inner_tol=1e-14)
    got = du(rhs.ravel()).reshape(rhs.shape)
    err1 = np.abs(got - dense.psi[:, : op.na]).max()
    du.passes = 8
    got = du(rhs.ravel()).reshape(rhs.shape)
    err8 = np.abs(got - dense.psi[:, : op.na]).max()
    scale = np.abs(dense.psi).max()
    assert err8 < 1e-9 * scale
    assert err8 < 1e-3 * err1


@pytest.mark.parametrize("pre", ["downup", "jacobi"])
def test_fgmres_matches_the_dense_solve(box, pre):
    g, c, op = box
    ctr = _contrasts(g, 0.5)
    kx, ky = 0.3 * OMEGA / A.alpha, 0.1 * OMEGA / A.alpha
    dense = iso.solve_interface(c, OMEGA, ctr, kx, ky, "P", p=1, r=0)
    got, stats = idu.solve_interface_fgmres(
        c, OMEGA, ctr, kx, ky, "P", p=1, r=0, operator=op, preconditioner=pre, tol=1e-11
    )
    assert np.abs(got.psi - dense.psi).max() < 1e-9 * np.abs(dense.psi).max()
    assert stats["outer"] < 60
    if pre == "downup":
        assert stats["planes"] > 0 and stats["inner"] >= stats["planes"]


def test_averaged_sweep_is_the_gram_inverse_at_zero_contrast(box):
    """With no contrast the averaged system is M alone: Algorithm 5.1's sweep applies M^-1 cell by cell."""
    g, _, op = box
    e = np.zeros((len(g.idx), op.n_source, op.na, 9, 9), dtype=complex)
    pre = idu.AveragedSweep(op, e)
    v = np.random.default_rng(2).standard_normal((len(g.idx), op.na, 9)) + 0j
    want = np.einsum("ab,nbi->nai", np.linalg.inv(op.m9), v)
    got = pre(v.ravel()).reshape(v.shape)
    assert np.abs(got - want).max() < 1e-12 * np.abs(want).max()


def test_fgmres_with_the_averaged_sweep_matches_the_dense_solve(box):
    g, c, op = box
    ctr = _contrasts(g, 0.5)
    kx, ky = 0.3 * OMEGA / A.alpha, 0.1 * OMEGA / A.alpha
    dense = iso.solve_interface(c, OMEGA, ctr, kx, ky, "P", p=1, r=0)
    got, stats = idu.solve_interface_fgmres(
        c,
        OMEGA,
        ctr,
        kx,
        ky,
        "P",
        p=1,
        r=0,
        operator=op,
        preconditioner="averaged",
        tol=1e-11,
    )
    assert np.abs(got.psi - dense.psi).max() < 1e-9 * np.abs(dense.psi).max()
    assert stats["outer"] < 60


def test_dtn_sweep_inverts_a_single_plane():
    """One plane: no coupling between planes to average, so the DtN form with a tight inner solve is the plane's
    exact inverse, its box's edges and its varying contrast included."""
    cells = [(-1, ix, iy) for ix in range(3) for iy in range(2)]
    g = iso.InterfaceGrid(np.array(cells), H, A, B)
    c = iso.InterfaceCouplings(g, workers=1)
    op = iso.LateralOperator(c, OMEGA, p=1, r=0)
    f = np.random.default_rng(9).uniform(0.1, 0.5, len(cells))
    ctr = [
        MaterialContrast(
            fm * g.medium(m).lam, fm * g.medium(m).mu, fm * g.medium(m).rho
        )
        for m, fm in enumerate(f)
    ]
    delta = iso.cell_contrasts(g, OMEGA, ctr, 0)
    e = np.array(
        [source_expansion(d, op.n_field)[: op.n_source, : op.na] for d in delta]
    )
    x = np.random.default_rng(5).standard_normal((len(cells), op.na, 9)) + 0j
    src = np.einsum("ncbij,nbj->nci", e, x).reshape(len(cells), -1)
    ax = np.einsum("ab,nbi->nai", op.m9, x) - op._convolve(src)
    got = idu.DtNSweep(op, e, inner_tol=1e-13)(ax.ravel()).reshape(x.shape)
    assert np.abs(got - x).max() < 1e-9 * np.abs(x).max()


def test_fgmres_with_the_dtn_sweep_matches_the_dense_solve(box):
    """A contrast that varies within the planes: the pivots carry the fluctuation."""
    g, c, op = box
    rng = np.random.default_rng(8)
    ctr = [
        MaterialContrast(f * g.medium(m).lam, f * g.medium(m).mu, f * g.medium(m).rho)
        for m, f in enumerate(rng.uniform(0.1, 0.5, len(g.idx)))
    ]
    kx, ky = 0.3 * OMEGA / A.alpha, 0.1 * OMEGA / A.alpha
    dense = iso.solve_interface(c, OMEGA, ctr, kx, ky, "P", p=1, r=0)
    got, stats = idu.solve_interface_fgmres(
        c,
        OMEGA,
        ctr,
        kx,
        ky,
        "P",
        p=1,
        r=0,
        operator=op,
        preconditioner="dtn",
        tol=1e-11,
    )
    assert np.abs(got.psi - dense.psi).max() < 1e-9 * np.abs(dense.psi).max()
    assert stats["outer"] < 60 and stats["planes"] > 0


def test_dtn_window_system_is_the_true_padded_system(box):
    """The window's operator, the averaged trailing system less the contrast fluctuation, is the true system
    M - K E on the padded grid (E zero outside the cells): over all the planes it is assembled independently."""
    g, _, op = box
    rng = np.random.default_rng(12)
    ctr = [
        MaterialContrast(f * g.medium(m).lam, f * g.medium(m).mu, f * g.medium(m).rho)
        for m, f in enumerate(rng.uniform(0.1, 0.5, len(g.idx)))
    ]
    delta = iso.cell_contrasts(g, OMEGA, ctr, 0)
    e = np.array(
        [source_expansion(d, op.n_field)[: op.n_source, : op.na] for d in delta]
    )
    du = idu.DtNSweep(op, e)
    win = list(range(len(op.layers)))
    rows, nf = op.na * 9, op.kernels[op.layers[0], op.layers[0]].shape[-1]
    v = [
        rng.standard_normal((rows, nf)) + 1j * rng.standard_normal((rows, nf))
        for _ in win
    ]
    got = du._trailing(win, v)
    for a, za in enumerate(op.layers):
        for b, zb in enumerate(op.layers):
            got[a] = got[a] - np.einsum(
                "rcf,cf->rf", op.kernels[za, zb], du._fluctuation(zb, v[b])
            )
    m_full = np.kron(op.m9, np.eye(9))
    for a, za in enumerate(op.layers):
        want = m_full @ v[a]
        for b, zb in enumerate(op.layers):
            src = np.zeros((op.n_source * 9,) + op.shape, dtype=complex)
            cells = op.by_layer[zb]
            ug = np.fft.ifft2(v[b].reshape((rows,) + op.shape), axes=(1, 2))
            src[:, op.pos[cells, 0], op.pos[cells, 1]] = np.einsum(
                "ncr,rn->cn", du.e_cells[zb], ug[:, op.pos[cells, 0], op.pos[cells, 1]]
            )
            want = want - np.einsum(
                "rcf,cf->rf",
                op.kernels[za, zb],
                np.fft.fft2(src, axes=(1, 2)).reshape(src.shape[0], -1),
            )
        assert np.abs(got[a] - want).max() < 1e-10 * np.abs(want).max()


def test_fgmres_with_full_dtn_windows_matches_the_dense_solve(box):
    g, c, op = box
    rng = np.random.default_rng(8)
    ctr = [
        MaterialContrast(f * g.medium(m).lam, f * g.medium(m).mu, f * g.medium(m).rho)
        for m, f in enumerate(rng.uniform(0.1, 0.5, len(g.idx)))
    ]
    kx, ky = 0.3 * OMEGA / A.alpha, 0.1 * OMEGA / A.alpha
    dense = iso.solve_interface(c, OMEGA, ctr, kx, ky, "P", p=1, r=0)
    got, stats = idu.solve_interface_fgmres(
        c,
        OMEGA,
        ctr,
        kx,
        ky,
        "P",
        p=1,
        r=0,
        operator=op,
        preconditioner="dtn",
        dtn_window=None,
        tol=1e-11,
    )
    assert np.abs(got.psi - dense.psi).max() < 1e-9 * np.abs(dense.psi).max()
    assert stats["outer"] < 60
