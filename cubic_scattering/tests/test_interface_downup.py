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
    cells = [(iz, ix, iy) for iz in (-2, -1, 0, 1) for ix in range(2) for iy in range(2)]
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


@pytest.mark.slow
def test_the_split_adds_up_to_the_coupling(box):
    g, _, op = box
    e = np.zeros((len(g.idx), op.n_source, op.na, 9, 9), dtype=complex)
    du = idu.DownUp(op, e)
    for zr in op.layers:
        for zs in op.layers:
            got = sum(parts.get((zr, zs), 0) for parts in (du.lo, du.up, du.refl)) + (
                du.k0[zr] if zr == zs else 0
            )
            assert np.abs(got - op.kernels[zr, zs]).max() <= 1e-12 * np.abs(op.kernels[zr, zs]).max()
    # forward is downward in L, upward in U; R stays on one side and off the diagonal
    assert all(zs < zr for zr, zs in du.lo) and all(zs > zr for zr, zs in du.up)
    assert all((zr < 0) == (zs < 0) and zr != zs for zr, zs in du.refl)


@pytest.mark.slow
def test_downup_passes_converge_to_the_dense_solve(box):
    """As a stationary iteration with exact plane solves, DownUp converges to the solution at small
    contrast."""
    g, c, op = box
    ctr = _contrasts(g, 0.1)
    kx, ky = 0.3 * OMEGA / A.alpha, 0.1 * OMEGA / A.alpha
    dense = iso.solve_interface(c, OMEGA, ctr, kx, ky, "P", p=1, r=0)
    delta = iso.cell_contrasts(g, OMEGA, ctr, 0)
    e = np.array([source_expansion(d, op.n_field)[: op.n_source, : op.na] for d in delta])
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


@pytest.mark.slow
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


@pytest.mark.slow
def test_averaged_sweep_is_the_gram_inverse_at_zero_contrast(box):
    """With no contrast the averaged system is M alone: Algorithm 5.1's sweep applies M^-1 cell by cell."""
    g, _, op = box
    e = np.zeros((len(g.idx), op.n_source, op.na, 9, 9), dtype=complex)
    pre = idu.AveragedSweep(op, e)
    v = np.random.default_rng(2).standard_normal((len(g.idx), op.na, 9)) + 0j
    want = np.einsum("ab,nbi->nai", np.linalg.inv(op.m9), v)
    got = pre(v.ravel()).reshape(v.shape)
    assert np.abs(got - want).max() < 1e-12 * np.abs(want).max()


@pytest.mark.slow
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


@pytest.mark.slow
def test_dtn_sweep_inverts_a_single_plane():
    """One plane: no coupling between planes to average, so the DtN form with a tight inner solve is the
    plane's exact inverse, its box's edges and its varying contrast included."""
    cells = [(-1, ix, iy) for ix in range(3) for iy in range(2)]
    g = iso.InterfaceGrid(np.array(cells), H, A, B)
    c = iso.InterfaceCouplings(g, workers=1)
    op = iso.LateralOperator(c, OMEGA, p=1, r=0)
    f = np.random.default_rng(9).uniform(0.1, 0.5, len(cells))
    ctr = [
        MaterialContrast(fm * g.medium(m).lam, fm * g.medium(m).mu, fm * g.medium(m).rho)
        for m, fm in enumerate(f)
    ]
    delta = iso.cell_contrasts(g, OMEGA, ctr, 0)
    e = np.array([source_expansion(d, op.n_field)[: op.n_source, : op.na] for d in delta])
    x = np.random.default_rng(5).standard_normal((len(cells), op.na, 9)) + 0j
    src = np.einsum("ncbij,nbj->nci", e, x).reshape(len(cells), -1)
    ax = np.einsum("ab,nbi->nai", op.m9, x) - op._convolve(src)
    got = idu.DtNSweep(op, e, inner_tol=1e-13)(ax.ravel()).reshape(x.shape)
    assert np.abs(got - x).max() < 1e-9 * np.abs(x).max()


@pytest.mark.slow
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


@pytest.mark.slow
def test_dtn_window_system_is_the_true_padded_system(box):
    """The window's operator, the averaged trailing system less the contrast fluctuation, is the true system
    M - K E on the padded grid (E zero outside the cells): over all the planes it is assembled
    independently."""
    g, _, op = box
    rng = np.random.default_rng(12)
    ctr = [
        MaterialContrast(f * g.medium(m).lam, f * g.medium(m).mu, f * g.medium(m).rho)
        for m, f in enumerate(rng.uniform(0.1, 0.5, len(g.idx)))
    ]
    delta = iso.cell_contrasts(g, OMEGA, ctr, 0)
    e = np.array([source_expansion(d, op.n_field)[: op.n_source, : op.na] for d in delta])
    du = idu.DtNSweep(op, e)
    win = list(range(len(op.layers)))
    rows, nf = op.na * 9, op.kernels[op.layers[0], op.layers[0]].shape[-1]
    v = [rng.standard_normal((rows, nf)) + 1j * rng.standard_normal((rows, nf)) for _ in win]
    got = du._trailing(win, v)
    for a, za in enumerate(op.layers):
        for b, zb in enumerate(op.layers):
            got[a] = got[a] - np.einsum("rcf,cf->rf", op.kernels[za, zb], du._fluctuation(zb, v[b]))
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


@pytest.mark.slow
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


def _varying(g, seed):
    rng = np.random.default_rng(seed)
    return [
        MaterialContrast(f * g.medium(m).lam, f * g.medium(m).mu, f * g.medium(m).rho)
        for m, f in enumerate(rng.uniform(0.1, 0.5, len(g.idx)))
    ]


@pytest.mark.slow
def test_true_sweep_with_full_windows_is_the_exact_inverse(box):
    """The leading solves of the true system, each over all the planes above and solved tightly: the block
    LDU solve of the true system, so the preconditioner inverts A (contrast varying within the planes)."""
    g, _, op = box
    delta = iso.cell_contrasts(g, OMEGA, _varying(g, 13), 0)
    e = np.array([source_expansion(d, op.n_field)[: op.n_source, : op.na] for d in delta])
    x = np.random.default_rng(6).standard_normal((len(g.idx), op.na, 9)) + 0j
    src = np.einsum("ncbij,nbj->nci", e, x).reshape(len(g.idx), -1)
    ax = np.einsum("ab,nbi->nai", op.m9, x) - op._convolve(src)
    got = idu.TrueSweep(op, e, inner_tol=1e-13, window=None)(ax.ravel()).reshape(x.shape)
    assert np.abs(got - x).max() < 1e-9 * np.abs(x).max()


@pytest.mark.slow
def test_fgmres_with_the_true_sweep_matches_the_dense_solve(box):
    g, c, op = box
    ctr = _varying(g, 8)
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
        preconditioner="true-sweep",
        dtn_window=0,
        tol=1e-11,
    )
    assert np.abs(got.psi - dense.psi).max() < 1e-9 * np.abs(dense.psi).max()
    assert stats["outer"] < 60


@pytest.mark.slow
def test_leading_sweep_is_the_exact_inverse(box):
    """Route A of the matrix-free true sweep, run here on the stored kernels: the leading systems solved
    tightly, the solved planes entering through the true coupling, invert A (contrast varying within the
    planes)."""
    g, _, op = box
    delta = iso.cell_contrasts(g, OMEGA, _varying(g, 13), 0)
    e = np.array([source_expansion(d, op.n_field)[: op.n_source, : op.na] for d in delta])
    x = np.random.default_rng(6).standard_normal((len(g.idx), op.na, 9)) + 0j
    src = np.einsum("ncbij,nbj->nci", e, x).reshape(len(g.idx), -1)
    ax = np.einsum("ab,nbi->nai", op.m9, x) - op._convolve(src)
    sweep = idu.LeadingSweep(op, e, inner_tol=1e-13)
    got = sweep(ax.ravel()).reshape(x.shape)
    assert np.abs(got - x).max() < 1e-9 * np.abs(x).max()
    assert sweep.inner_solves == len(op.layers)  # one leading solve per plane
    assert sweep.outer_products == len(op.layers) - 1  # one coupling product per solved plane


@pytest.mark.slow
def test_fgmres_with_the_leading_sweep_matches_the_dense_solve(box):
    g, c, op = box
    ctr = _varying(g, 8)
    kx, ky = 0.3 * OMEGA / A.alpha, 0.1 * OMEGA / A.alpha
    dense = iso.solve_interface(c, OMEGA, ctr, kx, ky, "P", p=1, r=0)
    got, stats = idu.solve_interface_fgmres(
        c, OMEGA, ctr, kx, ky, "P", p=1, r=0, operator=op, preconditioner="leading", tol=1e-11
    )
    assert np.abs(got.psi - dense.psi).max() < 1e-9 * np.abs(dense.psi).max()
    assert stats["outer"] < 30


def test_fgmres_from_an_initial_guess():
    """An exact x0 costs its residual's product alone; a near one fewer products than from zero."""
    rng = np.random.default_rng(2)
    n = 60
    a = np.eye(n) * 4 + rng.standard_normal((n, n)) / np.sqrt(n) * (1 + 1j)
    b = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    want = np.linalg.solve(a, b)
    x, n_mv, hist = idu.fgmres(lambda v: a @ v, lambda v: v, b, tol=1e-12, x0=want)
    assert n_mv == 1 and hist[0] <= 1e-12 and np.array_equal(x, want)
    _, n_cold, _ = idu.fgmres(lambda v: a @ v, lambda v: v, b, tol=1e-12)
    near = want + 1e-6 * np.abs(want).max() * rng.standard_normal(n)
    x, n_warm, _ = idu.fgmres(lambda v: a @ v, lambda v: v, b, tol=1e-12, x0=near)
    assert n_warm < n_cold and np.abs(x - want).max() < 1e-10 * np.abs(want).max()


@pytest.mark.slow
def test_a_preconditioner_built_once_serves_other_solves_and_frequencies(box):
    """A DtNSweep built at omega, passed as ``precondition``: the dense solution at omega and at 0.95 omega,
    the counts of each solve its own; and the solution at omega as x0 needs only its residual's product."""
    g, c, op = box
    ctr = _varying(g, 8)
    e = idu.source_expansions(op, iso.cell_contrasts(g, OMEGA, ctr, 0))
    pre = idu.DtNSweep(op, e, inner_tol=1e-2)
    counts = []
    for omega in (OMEGA, 0.95 * OMEGA):
        kx, ky = 0.3 * omega / A.alpha, 0.1 * omega / A.alpha
        dense = iso.solve_interface(c, omega, ctr, kx, ky, "P", p=1, r=0)
        got, stats = idu.solve_interface_fgmres(
            c,
            omega,
            ctr,
            kx,
            ky,
            "P",
            p=1,
            r=0,
            operator=op if omega == OMEGA else None,
            precondition=pre,
            tol=1e-11,
        )
        assert np.abs(got.psi - dense.psi).max() < 1e-9 * np.abs(dense.psi).max()
        counts.append(stats["planes"])
        if omega == OMEGA:
            first = got
    assert counts[0] > 0 and counts[1] > 0 and pre.inner_solves == sum(counts)
    kx, ky = 0.3 * OMEGA / A.alpha, 0.1 * OMEGA / A.alpha
    again, stats = idu.solve_interface_fgmres(
        c, OMEGA, ctr, kx, ky, "P", p=1, r=0, operator=op, precondition=pre, tol=1e-9, x0=first.psi
    )
    assert stats["outer"] == 1 and stats["planes"] == 0
    assert np.abs(again.psi - first.psi).max() == 0


def _varying_e(g, op, seed):
    return idu.source_expansions(op, iso.cell_contrasts(g, OMEGA, _varying(g, seed), 0))


@pytest.mark.slow
def test_dtn_sweep_with_no_inner_iterations_is_the_averaged_sweep(box):
    """S_z^-1 alone as each pivot's inverse: U^-1 S^-1 L^-1 is the averaged system's inverse."""
    g, _, op = box
    e = _varying_e(g, op, 3)
    v = np.random.default_rng(4).standard_normal(len(g.idx) * op.na * 9) + 0j
    want = idu.AveragedSweep(op, e)(v)
    pre = idu.DtNSweep(op, e, inner_iterations=0)
    got = pre(v)
    assert np.abs(got - want).max() < 1e-11 * np.abs(want).max()
    assert pre.inner_products == 0 and pre.inner_solves == len(op.layers)


@pytest.mark.slow
def test_dtn_sweep_depth_band(box):
    """A band over every plane is the full factorisation; band 1 is the exact inverse of the averaged system
    with its blocks between planes more than one apart dropped, assembled and inverted independently."""
    g, _, op = box
    e = _varying_e(g, op, 3)
    v = np.random.default_rng(4).standard_normal(len(g.idx) * op.na * 9) + 0j
    n_l = len(op.layers)
    full = idu.DtNSweep(op, e, inner_tol=1e-6)(v)
    for band in (n_l - 1, n_l + 2):
        assert np.array_equal(idu.DtNSweep(op, e, inner_tol=1e-6, depth_band=band)(v), full)
    a, _ = idu._averaged_system(op, e)
    rows = op.na * 9
    for i in range(n_l):
        for j in range(n_l):
            if abs(i - j) > 1:
                a[:, i * rows : (i + 1) * rows, j * rows : (j + 1) * rows] = 0
    want_pre = idu.AveragedSweep(op, e)
    want_pre.inv = np.linalg.inv(a)
    want = want_pre(v)
    got = idu.DtNSweep(op, e, inner_iterations=0, depth_band=1)(v)
    assert np.abs(got - want).max() < 1e-11 * np.abs(want).max()
    assert np.abs(want - idu.AveragedSweep(op, e)(v)).max() > 1e-6 * np.abs(want).max()  # the band matters


@pytest.mark.slow
def test_dtn_sweep_fixed_inner_iterations(box):
    """k inner iterations per pivot: the work grows with k, and many iterations approach the tight solve."""
    g, _, op = box
    e = _varying_e(g, op, 3)
    v = np.random.default_rng(4).standard_normal(len(g.idx) * op.na * 9) + 0j
    tight = idu.DtNSweep(op, e, inner_tol=1e-12)(v)
    products, errors = [], []
    for k in (1, 2, 4, 24):
        pre = idu.DtNSweep(op, e, inner_iterations=k)
        got = pre(v)
        products.append(pre.inner_products)
        errors.append(np.abs(got - tight).max() / np.abs(tight).max())
    assert products == sorted(products) and products[0] < products[-1]
    assert errors[-1] < 1e-8 and errors[0] > errors[-1]
    with pytest.raises(ValueError, match="inner_iterations"):
        idu.DtNSweep(op, e, inner_iterations=-1)


@pytest.mark.slow
def test_true_sweep_refuses_a_depth_band(box):
    g, _, op = box
    with pytest.raises(ValueError, match="depth_band applies to DtNSweep alone"):
        idu.TrueSweep(op, _varying_e(g, op, 3), depth_band=1)


@pytest.mark.slow
def test_sweeps_sharing_one_factorisation_are_the_sweeps_built_alone(box):
    """``sharing``: the same factors, settings of its own, counts from zero; equal to building it afresh."""
    g, _, op = box
    e = _varying_e(g, op, 3)
    v = np.random.default_rng(4).standard_normal(len(g.idx) * op.na * 9) + 0j
    base = idu.DtNSweep(op, e, inner_tol=1e-2)
    base(v)
    for cls, settings in [
        (idu.TrueSweep, {"window": None, "inner_tol": 1e-2}),
        (idu.TrueSweep, {"window": 1, "inner_iterations": 1}),
        (None, {"inner_iterations": 0}),
    ]:
        shared = base.sharing(cls, **settings)
        alone = (cls or idu.DtNSweep)(op, e, **settings)
        assert type(shared) is type(alone) and shared.inner_products == 0
        assert np.array_equal(shared(v), alone(v))
        assert shared.inner_products == alone.inner_products
        assert shared.lo is base.lo  # shared, not copied
    assert base.window == 0 and base.inner_iterations is None  # the original is untouched
    with pytest.raises(ValueError, match="unknown settings"):
        base.sharing(depth_band=1)
    with pytest.raises(ValueError, match="needs the full factorisation"):
        idu.DtNSweep(op, e, depth_band=1).sharing(idu.TrueSweep)
