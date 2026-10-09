#!/usr/bin/env python3
"""The graded sphere's Legendre-cell model expanded in powers of the frequency, against the exact series.

Every coupling block of the Legendre (Galerkin) cell is a power series in k_S with frequency-independent
coefficients: the touching blocks from the universal moments (``blocks.universal_moment``), every other
block from the far coefficients of the s-form (as ``blocks.far_series_coefficients``). With
G = (1 / 4 pi mu)[delta_ij g_S + k_S^-2 d_i d_j (g_S - g_P)] and g_k = sum_t (i k)^t r^(t-1) / t!, the
coefficient of k_S^j of a block is

    K_j = (1 / 4 pi mu) [ i^j / j! . A(r^(j-1))  +  i^(j+2) (1 - gamma^(j+2)) / (j+2)! . B(r^(j+1)) ],

A and B the blocks of delta_ij r^m and d_i d_j r^m, gamma = beta / alpha. The contrast is
Delta = Delta_0 + beta^2 k_S^2 Delta_1 (the density enters as omega^2 Drho), and the incident wave's cell
moments are a power series in k_P = gamma k_S. So the system (M - K E) psi = rhs is a power series in k_S,
and so is its solution psi = sum_j k_S^j psi_j, each order one solve with the STATIC operator:

    (M - K_0 E_0) psi_j = rhs_j + sum_(i>=1) K_i E_0 psi_(j-i) + beta^2 sum_(i>=0) K_i E_1 psi_(j-2-i).

The far field r e^(-i k r) u is expanded the same way (the phase e^(-i k x.r_hat) and the source), so every
power of the cell model's amplitude is obtained directly: no frequency is sampled and nothing is fitted.
The exact sphere's coefficients are those of ``measure_graded_sphere_frequency_series.exact_series``.

STORAGE. The blocks of a power are convolved by FFT on the (2n - 1)^3 grid. Under the reflection of axis m,
K(R o) = D K(o) D' with D and D' diagonal signs (``fft.symmetry_reps``), so the transform satisfies
K^(R k) = D K^(k) D' and only the octant k in [0, n)^3 is stored, computed from the octant of offsets by a
cosine or sine transform along each axis according to each entry's parity: an eighth of the memory.

Run:  python -u scripts/measure_graded_sphere_legendre_series.py [--p=1] [--r=1] [--J=6] [--check] n1 n2 ...
"""

import functools
import itertools
import math
import multiprocessing
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.sparse.linalg import LinearOperator, gmres

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import measure_ball_gradient_hierarchy as hier  # noqa: E402
import measure_graded_sphere_frequency_series as fs  # noqa: E402
import measure_graded_sphere_gradient_hierarchy as gs  # noqa: E402
from gate_sphere_cell_average_vs_mie import obs_points  # noqa: E402

from cubic_scattering.graded_voxel import blocks as gb  # noqa: E402
from cubic_scattering.graded_voxel.basis import (  # noqa: E402
    SOURCE_EXPONENTS,
    SOURCE_EXPONENTS_QUARTIC,
    gram_test,
    monomials,
    source_expansion,
)
from cubic_scattering.graded_voxel.farfield import graded_far_field  # noqa: E402
from cubic_scattering.graded_voxel.fft import _transform, signed_permutations, solve_graded_sphere_fft, symmetry_reps  # noqa: E402
from cubic_scattering.graded_voxel.site import cell_contrast_coefficients  # noqa: E402
from cubic_scattering.graded_voxel.solver import field_sizes  # noqa: E402
from cubic_scattering.sphere_scattering import _plane_wave_strain_voigt, _voigt_to_tensor  # noqa: E402
from cubic_scattering.sphere_scattering_fft import _build_grid_index_map  # noqa: E402

REF, CONTRAST = hier.REF, hier.CONTRAST
GAMMA = REF.beta / REF.alpha
PREF = 1.0 / (4.0 * math.pi * REF.mu)
#: GMRES relative tolerance of every solve (series orders and the direct check)
TOL = float(os.environ.get("SERIES_GMRES_TOL", "1e-13"))
CACHE = ROOT / "scripts" / "data" / "legendre_far_series"


# ------------------------------------------------------------------ the blocks, power by power
def near_powers(offset, h, n_source, n_test, J) -> list[np.ndarray]:
    """K_j for a touching offset (or the self cell), j = 0..J, from the universal moments."""
    ta, tb = gb.family_tables()
    out = []
    for j in range(J + 1):
        acc = np.zeros((n_test, n_source, 9, 9), dtype=complex)
        for table, m, w in (
            (ta, j - 1, PREF * 1j**j / math.factorial(j)),
            (tb, j + 1, PREF * 1j ** (j + 2) * (1.0 - GAMMA ** (j + 2)) / math.factorial(j + 2)),
        ):
            for idx, coef in table.items():
                u = gb.universal_moment(m, idx, offset, n_source, n_test)
                acc += w * h ** (6 + m - len(idx)) * u[:, :, None, None] * coef[None, None]
        out.append(gb._to_field_rows(acc))
    return out


def far_coefficients(job):
    """The far s-form coefficients S[t, family, a, c, 81] on the unit cell for t = 0..tmax."""
    offset, n_source, n_test, tmax = job
    ts = range(tmax + 1)
    part = gb._sform(
        offset, 1.0, (0, 0, 0), functools.partial(gb._far_family_kernels, ts=ts), gb.gauss_order(offset),
        n_source, n_test,
    )
    return offset, np.moveaxis(part.real.reshape(n_test, n_source, len(ts), 2, 81), (2, 3), (0, 1))


def far_powers(coeffs, h, J) -> list[np.ndarray]:
    """K_j for a non-touching offset from its far coefficients, j = 0..J."""
    order = gb._far_component_order()
    n_test, n_source = coeffs.shape[2:4]
    out = []
    for j in range(J + 1):
        wa = PREF * 1j**j / math.factorial(j) * h ** (5 + j)
        t = j + 2
        wb = PREF * 1j**t * (1.0 - GAMMA**t) / math.factorial(t) * h ** (5 + t) * h**-2.0
        blk = (wa * coeffs[j, 0] + wb * coeffs[t, 1]) * h ** (-order)
        out.append(gb._to_field_rows(blk.reshape(n_test, n_source, 9, 9)))
    return out


def all_far_coefficients(n_sub, n_source, n_test, J, workers=4) -> dict:
    """Far coefficients for every non-touching orbit of the grid, cached on disk across grids."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"far_ns{n_source}_nt{n_test}_t{J + 2}.pkl"
    store = pickle.loads(path.read_bytes()) if path.exists() else {}
    reps = {tuple(sorted(o, reverse=True)) for o in itertools.product(range(n_sub), repeat=3)}
    todo = sorted(r for r in reps if max(r) > 1 and r not in store)
    if todo:
        env = {v: os.environ.get(v) for v in fs.THREAD_VARS}
        os.environ.update({v: "1" for v in fs.THREAD_VARS})
        try:
            with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                store.update(pool.map(far_coefficients, [(r, n_source, n_test, J + 2) for r in todo], chunksize=2))
        finally:
            for v, val in env.items():
                os.environ.pop(v, None) if val is None else os.environ.__setitem__(v, val)
        path.write_bytes(pickle.dumps(store, protocol=pickle.HIGHEST_PROTOCOL))
    return store


# ------------------------------------------------------------------ the octant transform
def reflection_signs(n_source, n_test):
    """Row and column sign vectors (rows: test fn x 9, cols: source monomial x 9) of each axis reflection."""
    rows, cols = [], []
    for m in range(3):
        q = np.eye(3)
        q[m, m] = -1.0
        t, u, s = symmetry_reps(q, n_source, n_test)
        assert np.allclose(t, np.diag(np.diag(t))) and np.allclose(u, np.diag(np.diag(u)))
        rows.append(np.kron(np.diag(t), np.diag(s)))
        cols.append(np.kron(np.diag(u), np.diag(s)))
    return rows, cols


def octant_transform(blocks_oct, n_sub, rs, cs):
    """K^(k) for k in [0, n)^3 from K(o), o in [0, n)^3, (rows, cols) per point; DFT size 2n - 1."""
    npad = 2 * n_sub - 1
    k = np.arange(n_sub)
    theta = 2.0 * np.pi * np.outer(k, k) / npad  # [k, o]
    even = np.where(np.arange(n_sub)[None, :] == 0, 1.0, 2.0 * np.cos(theta))
    odd = np.where(np.arange(n_sub)[None, :] == 0, 0.0, -2.0j * np.sin(theta))
    out = blocks_oct.astype(complex)
    for m in range(3):
        par = np.outer(rs[m], cs[m])  # +1: entry even in axis m, -1: odd
        moved = np.moveaxis(out, m, 0)  # (n, n, n, rows, cols) with axis m first
        tr = np.where(par > 0, np.tensordot(even, moved, axes=(1, 0)), np.tensordot(odd, moved, axes=(1, 0)))
        out = np.moveaxis(tr, 0, m)
    return out  # (n, n, n, rows, cols)


# ------------------------------------------------------------------ the series solve
def solve_series(n_sub: int, p: int, r: int, J: int, profile):
    na, n_field, _, n_source = field_sizes(p, r)
    radius = gs.RADIUS
    h_cell = radius / n_sub
    grid_idx, centres, h = _build_grid_index_map(
        radius, n_sub, lambda q: bool(np.linalg.norm(q) < radius + np.sqrt(3.0) * h_cell)
    )
    n = len(centres)
    rows, cols = na * 9, n_source * 9
    # contrast: Delta = Delta_0 + beta^2 k_S^2 Delta_1 (omega = beta k_S)
    d0 = np.array([cell_contrast_coefficients(profile, c, h, CONTRAST, REF, 0.0, degree=r) for c in centres])
    d1 = np.array([cell_contrast_coefficients(profile, c, h, CONTRAST, REF, 1.0, degree=r) for c in centres]) - d0
    e0 = np.array([source_expansion(d, n_field)[:, :na] for d in d0])  # (N, nc, na, 9, 9)
    e1 = np.array([source_expansion(d, n_field)[:, :na] for d in d1])
    # blocks per power, octant of the transform
    t0 = time.perf_counter()
    far = all_far_coefficients(n_sub, n_source, n_field, J)
    t_far = time.perf_counter() - t0
    qs = signed_permutations()
    canon = {}
    for o in itertools.product(range(n_sub), repeat=3):
        c = tuple(sorted(o, reverse=True))
        if c not in canon:
            canon[c] = near_powers(c, h, n_source, n_field, J) if max(c) <= 1 else far_powers(far[c], h, J)
    rs, cs = reflection_signs(n_source, n_field)
    rs = [x[:rows] for x in rs]  # the first na test functions
    k_oct = []
    for j in range(J + 1):
        bo = np.zeros((n_sub, n_sub, n_sub, rows, cols), dtype=complex)
        for o in itertools.product(range(n_sub), repeat=3):
            c = tuple(sorted(o, reverse=True))
            q = next(q for q in qs if np.array_equal(q @ np.array(c, float), np.array(o, float)))
            blk = _transform(q, canon[c][j])[:na]
            bo[o] = blk.transpose(0, 2, 1, 3).reshape(rows, cols)
        k_oct.append(octant_transform(bo, n_sub, rs, cs))
        del bo
    t_blocks = time.perf_counter() - t0
    npad = 2 * n_sub - 1
    patterns = []
    for sig in itertools.product((1, -1), repeat=3):
        tgt, src, dr, dc = [], [], np.ones(rows), np.ones(cols)
        for m, sg in enumerate(sig):
            if sg > 0:
                tgt.append(np.arange(0, n_sub))
                src.append(np.arange(0, n_sub))
            else:
                tt = np.arange(n_sub, npad)
                tgt.append(tt)
                src.append(npad - tt)
                dr, dc = dr * rs[m], dc * cs[m]
        patterns.append((np.ix_(*tgt), np.ix_(*src), dr, dc))
    g0, g1, g2 = grid_idx.T

    def kconv(j: int, src: np.ndarray) -> np.ndarray:
        """sum_n K_j(g_m - g_n) src_n for source coefficients src (N, cols): (N, rows)."""
        grid = np.zeros((npad, npad, npad, cols), dtype=complex)
        grid[g0, g1, g2] = src
        sh = np.fft.fftn(grid, axes=(0, 1, 2))
        yh = np.zeros((npad, npad, npad, rows), dtype=complex)
        for tgt, srci, dr, dc in patterns:
            yh[tgt] = dr * np.einsum("xyzrc,xyzc->xyzr", k_oct[j][srci], sh[tgt] * dc)
        return np.fft.ifftn(yh, axes=(0, 1, 2))[g0, g1, g2]

    def apply_e(e, psi):  # psi (N, na, 9) -> source coefficients (N, cols)
        return np.einsum("ncbij,nbj->nci", e, psi).reshape(n, cols)

    m9 = gram_test(h, n_field)[:na, :na]
    dim = n * rows

    def a0(x):
        psi = x.reshape(n, na, 9)
        return (np.einsum("ab,nbi->nai", m9, psi).reshape(n, rows) - kconv(0, apply_e(e0, psi))).ravel()

    k0 = near_powers((0, 0, 0), h, n_source, n_field, 0)[0][:na]
    pre = np.array([
        np.linalg.inv(np.kron(m9, np.eye(9)) - np.einsum("acij,cbjk->aibk", k0, en).reshape(rows, rows))
        for en in e0
    ])  # fmt: skip
    a_op = LinearOperator((dim, dim), matvec=a0, dtype=complex)
    m_op = LinearOperator((dim, dim), matvec=lambda x: np.einsum("nij,nj->ni", pre, x.reshape(n, rows)).ravel(),
                          dtype=complex)  # fmt: skip
    # incident wave: amp9(k) = a0 + k_P a1, phase sum_m (i k_P k.x)^m / m!
    amp0 = np.concatenate([gs.POL.astype(complex), np.zeros(6)])
    amp1 = np.concatenate([np.zeros(3), _plane_wave_strain_voigt(gs.K_HAT, gs.POL, 1.0)])
    xg, wg = leggauss(12)
    xi = np.stack(np.meshgrid(xg, xg, xg, indexing="ij"), -1).reshape(-1, 3)
    wq = np.einsum("i,j,k->ijk", wg, wg, wg).ravel() * h**3
    # the field functions as products of Legendre polynomials, as ``solver.plane_wave_moments`` takes them
    leg = [np.ones_like(xi), xi, 0.5 * (3.0 * xi**2 - 1.0)]
    lvals = np.array([np.prod([leg[e[i]][:, i] for i in range(3)], axis=0) for e in SOURCE_EXPONENTS[:na]])
    proj = (centres @ gs.K_HAT)[:, None] + h * (xi @ gs.K_HAT)[None, :]  # (N, G)
    mom = [np.einsum("ag,ng->na", lvals * wq, proj**m) for m in range(J + 1)]  # <L_a, (k.x)^m>
    rhs = []
    for j in range(J + 1):
        v = GAMMA**j * (1j**j / math.factorial(j)) * mom[j][:, :, None] * amp0
        if j >= 1:
            v = v + GAMMA**j * (1j ** (j - 1) / math.factorial(j - 1)) * mom[j - 1][:, :, None] * amp1
        rhs.append(v)
    psi = []
    t1 = time.perf_counter()
    for j in range(J + 1):
        b = rhs[j].reshape(n, rows).copy()
        for i in range(1, j + 1):
            b += kconv(i, apply_e(e0, psi[j - i]))
        for i in range(0, j - 1):
            b += REF.beta**2 * kconv(i, apply_e(e1, psi[j - 2 - i]))
        sol, info = gmres(a_op, b.ravel(), M=m_op, rtol=TOL, atol=0.0, restart=200, maxiter=50)
        assert info == 0, (j, info)
        psi.append(sol.reshape(n, na, 9))
    t_solve = time.perf_counter() - t1
    return dict(centres=centres, h=h, e0=e0, e1=e1, psi=psi, na=na, n_field=n_field, n_source=n_source,
                d0=d0, d1=d1, times=(t_far, t_blocks, t_solve))  # fmt: skip


# ------------------------------------------------------------------ the far field, power by power
def far_series(sol, obs, J):
    """Coefficients of k_S^p, p = 0..J + 1, of r e^(-i k r) u in each direction, P and S."""
    xg, wg = leggauss(8)  # exact for the source (degree <= 4) times the phase's powers (degree <= 7)
    xi = np.stack(np.meshgrid(xg, xg, xg, indexing="ij"), -1).reshape(-1, 3)
    wq = np.einsum("i,j,k->ijk", wg, wg, wg).ravel() * sol["h"] ** 3
    n_src = sol["n_source"]
    ms = monomials(SOURCE_EXPONENTS_QUARTIC[:n_src], xi)  # (n_src, G)
    dirs = obs / np.linalg.norm(obs, axis=1)[:, None]
    n_pow = J + 2
    out = {m: np.zeros((n_pow, len(obs), 3), dtype=complex) for m in ("P", "S")}
    # sources per order: s_j = E_0 psi_j + beta^2 E_1 psi_(j-2), at the Gauss nodes
    # s_(J+1) is kept too: its force, beta^2 E_1 psi_(J-1), radiates at k_S^(J+1); its stress E_0 psi_(J+1)
    # only from k_S^(J+2) on, beyond the powers kept, so psi_(J+1) is not needed
    srcs = []
    n_psi = len(sol["psi"])
    for j in range(n_psi + 1):
        sc = np.einsum("ncbij,nbj->nci", sol["e0"], sol["psi"][j]) if j < n_psi else 0.0
        if j >= 2:
            sc = sc + REF.beta**2 * np.einsum("ncbij,nbj->nci", sol["e1"], sol["psi"][j - 2])
        srcs.append(np.einsum("cg,nci->ngi", ms, sc) * wq[None, :, None])  # (N, G, 9)
    pts = sol["centres"][:, None, :] + sol["h"] * xi[None]
    for o, rh in enumerate(dirs):
        proj = pts @ rh  # (N, G)
        for mode, kr, speed in (("P", GAMMA, REF.alpha), ("S", 1.0, REF.beta)):
            pref = 1.0 / (4.0 * math.pi * REF.rho * speed**2)
            for j, s in enumerate(srcs):
                force = s[..., :3]
                sig = np.array([_voigt_to_tensor(v) for v in s[..., 3:].reshape(-1, 6)]).reshape(*s.shape[:2], 3, 3)
                sr = sig @ rh
                for m in range(n_pow - j):
                    ph = (-1j * kr * proj) ** m / math.factorial(m)
                    for p_, amp in ((j + m, np.einsum("ng,ngi->i", ph, force)),
                                    (j + m + 1, 1j * kr * np.einsum("ng,ngi->i", ph, sr))):  # fmt: skip
                        if p_ < n_pow:
                            amp = rh * (rh @ amp) if mode == "P" else amp - rh * (rh @ amp)
                            out[mode][p_, o] += pref * amp
    return out


def profile_fn(pos) -> float:
    return float(gs.smoothstep(np.array([np.linalg.norm(pos)]))[0])


def check(n_sub: int, p: int, r: int, J: int, obs) -> None:
    """The series summed at k_S a = 0.02 against the direct FFT solve, and K_j against the blocks."""
    na, n_field, _, n_source = field_sizes(p, r)
    h = gs.RADIUS / n_sub
    k = 1e-4 / h
    for off in ((0, 0, 0), (1, 0, 0), (1, 1, 1)):
        ser = sum(k**j * b for j, b in enumerate(near_powers(off, h, n_source, n_field, J)))
        ref = gb.near_block_series(off, h, k * REF.beta, REF, n_source, n_field)
        print(f"[check] near K_j summed vs near_block_series {off}: {np.abs(ser - ref).max() / np.abs(ref).max():.1e}")
    for off in ((2, 0, 0), (3, 2, 1)):
        c = far_coefficients((off, n_source, n_field, J + 2))[1]
        ser = sum(k**j * b for j, b in enumerate(far_powers(c, h, J)))
        ref = gb.far_block_series(off, h, k * REF.beta, REF, n_source, n_field)
        print(f"[check] far K_j summed vs far_block_series {off}: {np.abs(ser - ref).max() / np.abs(ref).max():.1e}")
    sol = solve_series(n_sub, p, r, J, profile_fn)
    ser = far_series(sol, obs, J)
    for ka in (0.02, 0.05):
        ks = ka / gs.RADIUS
        omega = ks * REF.beta
        # the direct solve with every block from the package's series at this frequency (exact for the
        # touching blocks), so that the comparison isolates the expansion in the frequency
        canon, blocks = {}, {}
        for off in itertools.product(range(-(n_sub - 1), n_sub), repeat=3):
            c = tuple(sorted((abs(o) for o in off), reverse=True))
            if c not in canon:
                fn = gb.near_block_series if max(c) <= 1 else gb.far_block_series
                canon[c] = fn(c, h, omega, REF, n_source, n_field)
            q = next(q for q in signed_permutations() if np.array_equal(q @ np.array(c, float), np.array(off, float)))
            blocks[off] = _transform(q, canon[c])
        res = solve_graded_sphere_fft(omega, gs.RADIUS, REF, CONTRAST, n_sub, profile_fn, gs.K_HAT, gs.POL, "P",
                                      p=p, r=r, gmres_tol=TOL, blocks=blocks, max_cycles=60)  # fmt: skip
        # the 1/r far-field formula at r = 1: r e^(-i k r) u is then the formula itself, with no phase
        # e^(i k r) at a large r to cancel (at r = 5e8 a, k r ~ 1e7 would cost ~1e-9 of the amplitude)
        up, us = graded_far_field(res, obs, 1.0, gs.K_HAT, gs.POL, "P", n_gauss=8)
        up = up * np.exp(-1j * omega / REF.alpha)
        us = us * np.exp(-1j * ks)
        sp = sum(ser["P"][q] * ks**q for q in range(J + 2))
        ss = sum(ser["S"][q] * ks**q for q in range(J + 2))
        scale = max(np.abs(up).max(), np.abs(us).max())
        print(f"[check] n {n_sub}, k_S a = {ka}: series summed vs direct solve, P {np.abs(sp - up).max() / scale:.1e}, "
              f"S {np.abs(ss - us).max() / scale:.1e}", flush=True)  # fmt: skip


def main() -> int:
    args = sys.argv[1:]
    opts = {k: int(v) for k, v in (a[2:].split("=") for a in args if a.startswith("--") and "=" in a)}
    p, r, J = opts.get("p", 1), opts.get("r", 1), opts.get("J", 6)
    do_check = "--check" in args
    ladder = [int(a) for a in args if not a.startswith("--")] or [4, 6]
    gs.CORE = 0.1 * gs.RADIUS
    gs.set_profile("smoothstep")
    obs = obs_points(gs.R_FAR, gs.THETA)
    print(f"Legendre cell p = {p}, contrast degree r = {r}, powers of k_S kept J = {J}", flush=True)
    if do_check:
        check(ladder[0], p, r, J, obs)
        return 0
    ex = fs.exact_series(obs, J + 2)
    a = gs.RADIUS
    lead = max(np.abs(ex[m][2]).max() for m in ("P", "S")) * a**-2
    for n in ladder:
        sol = solve_series(n, p, r, J, profile_fn)
        vox = far_series(sol, obs, J)
        row = []
        for q in range(2, J + 2):
            diff = max(np.abs(vox[m][q] - ex[m][q]).max() for m in ("P", "S")) * a**-q
            size = max(np.abs(ex[m][q]).max() for m in ("P", "S")) * a**-q
            rel = diff / size if size > 1e-12 * lead else diff / lead
            row.append(f"(ka)^{q}: {rel:.3e} [{size / lead:.1e}]")
        tf, tb, tsv = sol["times"]
        print(f"  n {n:2d} cells {len(sol['centres']):5d}: " + "  ".join(row)
              + f"   (far coefficients {tf:.0f} s, blocks {tb:.0f} s, solves {tsv:.0f} s)", flush=True)  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
