#!/usr/bin/env python3
"""The graded sphere's cell model expanded in powers of the frequency, against the exact series.

Every coupling table of the Cartesian multipole hierarchy is a power series in k with frequency-independent
coefficients (``derivatives.moment_table_kseries``; the self table likewise, ``tensor_moment``), the density
contrast enters as omega^2 Drho = beta^2 Drho k_S^2, and the incident plane wave's derivatives at the cell
centres are a power series in k_P = (beta / alpha) k_S. So the system psi - M (B * psi) = rhs is a power
series in k_S, and so is its solution:

    A_0 psi_j = rhs_j + sum_(i=1..j) M (B_i * psi_(j-i)),        A_0 = I - M B_0 * ,

each order one solve with the STATIC operator. The far field is expanded the same way (the phase
exp(-i k x.r_hat) and the omega^2 force), so every power of the cell model's far-field amplitude is
obtained directly: no frequency is sampled and nothing is fitted.

The exact graded sphere's amplitude has the same expansion in closed form from its per-order T-matrix
series (``Mathematica/GradedSphere_LowFrequency.json``): with h_n(z) ~ (-i)^(n+1) e^(iz) / z and the plane
wave's (2n+1) i^n / (i k_P),

    r e^(-i k_P r) u_r     = (-i / k_P) sum_n (2n+1) T^PP_n(w) P_n(cos theta),
    r e^(-i k_S r) u_theta = (-i / k_P) sum_n (2n+1) T^SP_n(w) dP_n/dtheta,         w = k_P a,

checked here against ``mie_scattered_displacement`` at one real frequency before it is used.

Tables: touching and near orbits (Chebyshev distance <= 2) from the closed coefficients (cached by
``measure_graded_sphere_low_frequency.py``), farther ones from the Gauss coefficients of
``derivatives.kseries_coefficients``.

Run:  python -u scripts/measure_graded_sphere_frequency_series.py [--gauss] [--rc=2] [--profile=sin2] n1 ...
"""

import itertools
import json
import math
import multiprocessing
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import legval
from scipy.sparse.linalg import LinearOperator, gmres

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import measure_ball_gradient_hierarchy as hier  # noqa: E402
import measure_graded_sphere_gradient_hierarchy as gs  # noqa: E402
import measure_graded_sphere_gradient_hierarchy_fft as gfft  # noqa: E402
import measure_graded_sphere_low_frequency as lowf  # noqa: E402
from cubic_scattering.graded_voxel import derivatives as gd  # noqa: E402
from cubic_scattering.sphere_scattering import mie_scattered_displacement  # noqa: E402
from gate_sphere_cell_average_vs_mie import obs_points  # noqa: E402

J = 6  # powers of k_S kept in the cell model; the far field to (k a)^(2 + J)
AXES = (0, 1, 2)
REF, CONTRAST = hier.REF, hier.CONTRAST
MU = REF.rho * REF.beta**2
RATIO = REF.beta / REF.alpha  # k_P / k_S
#: k-series coefficients computed ahead (``prefetch_kseries``): {canonical offset: (u_all[:J + 3], a_list)}.
#: Only the first J + 3 powers are kept, against the 49 that ``kseries_coefficients`` caches per offset.
KSERIES: dict = {}
#: if set, a directory to which each power's transformed blocks are written as they are built and from which
#: they are read through a memory map; only B_0, which GMRES applies at every iteration, is then held in RAM
SPILL_DIR: Path | None = None


# ------------------------------------------------------------------ per-power tables
def table_powers(units: tuple[int, int, int], side: float, d_list, w_list) -> list[np.ndarray]:
    """T_j[i, n, D, W], the coefficient of k_S^j of the table between two different cells, j = 0..J."""
    if max(abs(u) for u in units) <= lowf.NEAR:
        u_all, a_list = lowf.closed_coefficients(units)
    elif units in KSERIES:
        u_all, a_list = KSERIES[units]
    else:
        u_all, a_list = gd.kseries_coefficients(units, tuple(d_list), tuple(w_list))
    rows_d, rows_b = gd._kseries_rows(tuple(d_list), a_list)
    a_deg = np.array([sum(a) for a in a_list], dtype=float)
    w_deg = np.array([sum(w) for w in w_list], dtype=float)
    scale = side ** (2.0 - a_deg[:, None] + w_deg[None, :])
    out = []
    for j in range(J + 1):
        v1 = (1j**j) * side**j / math.factorial(j) * u_all[j] * scale
        t2 = j + 2  # the B family's term t carries k_S^(t - 2)
        v2 = (1j**t2) * (1.0 - RATIO**t2) * side**t2 / math.factorial(t2) * u_all[t2] * scale
        tab = np.zeros((3, 3, len(d_list), len(w_list)), dtype=complex)
        for i in AXES:
            for n in range(i, 3):
                val = v2[rows_b[i][n]]
                if i == n:
                    val = val + v1[rows_d]
                tab[i, n] = tab[n, i] = val
        out.append(tab / (4.0 * math.pi * MU))
    return out


def self_powers(side: float, d_list, w_list) -> list[np.ndarray]:
    """The self table's coefficients of k_S^j, from the stored moments of the cube (``tensor_moment``)."""
    hier.scalar_moment = gs.cube.cube_scalar_moment
    out = []
    for j in range(J + 1):
        tab = np.zeros((3, 3, len(d_list), len(w_list)), dtype=complex)
        for di, ds in enumerate(d_list):
            for wi, w in enumerate(w_list):
                if (len(ds) + len(w)) % 2:
                    continue
                for i in AXES:
                    for n in range(i, 3):
                        val = 0.0j
                        if i == n:
                            val += (1j**j) / math.factorial(j) * hier.scalar_moment(side, j - 1, ds, w)
                        t2 = j + 2
                        val += (
                            (1j**t2)
                            * (1.0 - RATIO**t2)
                            / math.factorial(t2)
                            * hier.scalar_moment(side, t2 - 1, (*ds, i, n), w)
                        )
                        tab[i, n, di, wi] = tab[n, i, di, wi] = (-1) ** len(ds) * val / (4.0 * math.pi * MU)
        out.append(tab)
    return out


THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def _kseries_one(job):
    rep, d_exp, w_exp = job
    u_all, a_list = gd.kseries_coefficients.__wrapped__(rep, d_exp, w_exp)
    return rep, (u_all[: J + 3].copy(), a_list)


def prefetch_kseries(n_sub: int, d_list, w_list, workers: int, cache: Path | None = None) -> None:
    """Fill ``KSERIES`` for every orbit of an n_sub grid farther than ``lowf.NEAR``, in parallel.

    The coefficients are on the unit cube, so they do not depend on the grid: ``cache`` (a pickle per
    contrast degree and J) carries them from one grid to the next.
    """
    d_exp = tuple(gd.as_exponents(d) for d in d_list)
    w_exp = tuple(gd.as_exponents(w) for w in w_list)
    if cache is not None and cache.exists() and not KSERIES:
        KSERIES.update(pickle.loads(cache.read_bytes()))
    reps = {gd.canonical_offset(k) for k in itertools.product(range(n_sub), repeat=3)}
    todo = sorted(r for r in reps if max(r) > lowf.NEAR and r not in KSERIES)
    if not todo:
        return
    # fresh single-threaded workers: one BLAS thread each, so that they do not oversubscribe the cores
    saved = {v: os.environ.get(v) for v in THREAD_VARS}
    os.environ.update({v: "1" for v in THREAD_VARS})
    try:
        with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            KSERIES.update(pool.map(_kseries_one, [(r, d_exp, w_exp) for r in todo], chunksize=4))
    finally:
        for v, val in saved.items():
            if val is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = val
    if cache is not None:
        cache.write_bytes(pickle.dumps(KSERIES, protocol=pickle.HIGHEST_PROTOCOL))


# ------------------------------------------------------------------ the series solve
def solve_series(n_sub: int):
    side, centres, coefs, grid = gs.build_cells(n_sub, gfft.R_C)
    asm_s = gs.Assembler(3, gfft.R_C, 0.0, CONTRAST)  # a_rho = 0: the strain part of the blocks
    asm_r = gs.Assembler(3, gfft.R_C, REF.beta, CONTRAST)  # a_rho = beta^2 Drho: strain plus density / k^2
    v_exp = [gd.as_exponents(v) for v in asm_s.v_list]
    u_exp = [gd.as_exponents(u) for u in asm_s.u_list]
    d_exp = [gd.as_exponents(d) for d in asm_s.d_list]
    w_exp = [gd.as_exponents(w) for w in asm_s.w_list]
    nu3, n_v, n_fft = 3 * asm_s.nu, len(asm_s.v_list), 2 * n_sub
    # blocks per power: B_j = S(T_j) + beta^2 Drho D(T_(j-2)), D(T) = blocks_r(T) - blocks_s(T)
    canon: dict[tuple[int, int, int], list[np.ndarray]] = {}

    def power_at(key, j):
        rep = gd.canonical_offset(key)
        if rep not in canon:
            tabs = (
                self_powers(side, asm_s.d_list, asm_s.w_list)
                if rep == (0, 0, 0)
                else table_powers(rep, side, d_exp, w_exp)
            )
            strain = [asm_s.blocks(t) for t in tabs]
            dens = [asm_r.blocks(t) - s for t, s in zip(tabs, strain, strict=True)]
            canon[rep] = [strain[t] + (dens[t - 2] if t >= 2 else 0.0) for t in range(J + 1)]
        if key == (0, 0, 0):
            return canon[rep][j]
        pi, sigma = gd.mapping_to(key)
        return gd.transform_block(canon[rep][j], pi, sigma, v_exp, u_exp)

    d, s = gfft.parity_signs(asm_s)
    k_half = n_fft // 2 + 1
    b_oct = []
    # one power at a time, and the FFT a third of the rows at a time, so that the real-space blocks of
    # only one power are held: the peak memory is then set by the transformed blocks themselves
    for j in range(J + 1):
        octant = {key: power_at(key, j) for key in itertools.product(range(n_sub), repeat=3)}
        for blocks in canon.values():  # this power's blocks now live in the octant
            blocks[j] = None
        shape = (n_v, k_half, k_half, k_half, nu3, nu3)
        if SPILL_DIR is None:
            bo = np.zeros(shape, dtype=complex)
        else:
            bo = np.lib.format.open_memmap(SPILL_DIR / f"b_oct_{j}.npy", mode="w+", dtype=complex, shape=shape)
        for vi in range(n_v):
            for r0 in range(0, nu3, nu3 // 3):
                rows = slice(r0, r0 + nu3 // 3)
                spatial = gfft._place(octant, n_fft, d, s, vi, rows, n_v)
                bo[vi, ..., rows, :] = np.fft.fftn(spatial, axes=(0, 1, 2))[:k_half, :k_half, :k_half]
        del octant, spatial
        if SPILL_DIR is not None:
            bo.flush()
        b_oct.append(bo)
    canon.clear()
    if SPILL_DIR is not None:
        b_oct[0] = np.array(b_oct[0])  # applied at every GMRES iteration: held in RAM
    patterns = []
    for sig in itertools.product((1, -1), repeat=3):
        tgt, src, dd, sv = [], [], np.ones(nu3), np.ones(n_v)
        for m, sg in enumerate(sig):
            if sg > 0:
                tgt.append(np.arange(0, k_half))
                src.append(np.arange(0, k_half))
            else:
                t_idx = np.arange(k_half, n_fft)
                tgt.append(t_idx)
                src.append(n_fft - t_idx)
                dd, sv = dd * d[m], sv * s[m]
        if all(len(t) for t in tgt):
            patterns.append((np.ix_(*tgt), np.ix_(*src), dd, sv))
    coef_grid = np.zeros((n_sub, n_sub, n_sub, n_v))
    mask = np.zeros((n_sub, n_sub, n_sub), dtype=bool)
    gi = tuple(grid.T)
    coef_grid[gi] = coefs
    mask[gi] = True

    def conv(j: int, x: np.ndarray) -> np.ndarray:
        """M (B_j * x)."""
        acc = np.zeros((n_fft, n_fft, n_fft, nu3), dtype=complex)
        for vi in range(n_v):
            z = np.zeros((n_fft, n_fft, n_fft, nu3), dtype=complex)
            z[:n_sub, :n_sub, :n_sub] = coef_grid[..., vi, None] * x
            z_hat = np.fft.fftn(z, axes=(0, 1, 2))
            for tgt, src, dd, sv in patterns:
                zz = z_hat[tgt] * dd
                acc[tgt] += sv[vi] * dd * np.matmul(b_oct[j][vi][src], zz[..., None])[..., 0]
        return mask[..., None] * np.fft.ifftn(acc, axes=(0, 1, 2))[:n_sub, :n_sub, :n_sub]

    size = n_sub**3 * nu3
    a0 = LinearOperator((size, size), dtype=complex,
                        matvec=lambda v: (v.reshape(n_sub, n_sub, n_sub, nu3) - conv(0, v.reshape(
                            n_sub, n_sub, n_sub, nu3))).ravel())  # fmt: skip
    # the incident wave's coefficient of k_S^j: POL_i (i r k_hat)^p (i r k_hat . x_c)^(j-|p|) / (j-|p|)!
    rhs = [np.zeros((n_sub, n_sub, n_sub, nu3), dtype=complex) for _ in range(J + 1)]
    for c, xc in enumerate(centres):
        kx = float(gs.K_HAT @ xc)
        for pi, p_idx in enumerate(asm_s.u_list):
            kp_pow = np.prod([gs.K_HAT[ax] for ax in p_idx]) if p_idx else 1.0
            for j in range(len(p_idx), J + 1):
                m = j - len(p_idx)
                val = (1j * RATIO) ** j * kp_pow * kx**m / math.factorial(m)
                for i in AXES:
                    rhs[j][tuple(grid[c])][i * asm_s.nu + pi] += gs.POL[i] * val
    psi = []
    for j in range(J + 1):
        b = rhs[j].copy()
        for i in range(1, j + 1):
            b += conv(i, psi[j - i])
        sol, info = gmres(a0, b.ravel(), rtol=1e-13, atol=0.0, restart=200, maxiter=50)
        assert info == 0, (j, info)
        psi.append(sol.reshape(n_sub, n_sub, n_sub, nu3))
    sols = [np.array([p[tuple(g)] for g in grid]) for p in psi]  # per power: (cells, nu3)
    return side, centres, coefs, asm_s, sols


# ------------------------------------------------------------------ far-field series
def far_series(side, centres, coefs, asm, sols, obs):
    """Coefficients of k_S^j (j = 0..J + 2) of r e^(-i k r) u at each observation direction, P and S."""
    x1, w1 = np.polynomial.legendre.leggauss(6)
    x1, w1 = 0.5 * side * x1, 0.5 * side * w1
    xi = np.stack(np.meshgrid(x1, x1, x1, indexing="ij"), -1).reshape(-1, 3)
    wts = np.einsum("i,j,k->ijk", w1, w1, w1).ravel()
    mono_u = np.stack([gs.monomial(xi, w) for w in asm.u_list])
    mono_v = np.stack([gs.monomial(xi, v) for v in asm.v_list])
    grad_u = np.zeros((3, asm.nu, len(xi)))
    for wi, w in enumerate(asm.u_list):
        for r in set(w):
            rest = list(w)
            rest.remove(r)
            grad_u[r, wi] = w.count(r) * gs.monomial(xi, tuple(rest))
    dc = hier.stiffness(CONTRAST)
    dirs = obs / np.linalg.norm(obs, axis=1)[:, None]
    n_pow = J + 3
    out = {
        "P": np.zeros((n_pow, len(obs), 3), dtype=complex),
        "S": np.zeros((n_pow, len(obs), 3), dtype=complex),
    }
    for c, (xc, cf) in enumerate(zip(centres, coefs, strict=True)):
        prof = cf @ mono_v
        pts = xc[None, :] + xi
        for j, sol in enumerate(sols):
            s = sol[c].reshape(3, asm.nu)
            u = np.einsum("jw,w,wg->gj", s, asm.coef, mono_u)
            grad = np.einsum("jw,w,rwg->grj", s, asm.coef, grad_u)
            force = REF.beta**2 * CONTRAST.Drho * prof[:, None] * u  # times k_S^2
            tau = prof[:, None, None] * np.einsum("nkrj,grj->gnk", dc, grad)
            for o, rh in enumerate(dirs):
                proj = pts @ rh
                for mode, kr, speed in (("P", RATIO, REF.alpha), ("S", 1.0, REF.beta)):
                    pref = 1.0 / (4.0 * math.pi * REF.rho * speed**2)
                    # exp(-i k_c x.rh) = sum_m (-i kr x.rh)^m / m!  (k_c = kr k_S)
                    for m in range(n_pow - j):
                        ph = (-1j * kr * proj) ** m / math.factorial(m) * wts
                        # the force term carries k_S^2, the stress term i k_c = i kr k_S
                        terms = []
                        if j + m + 2 < n_pow:
                            terms.append((j + m + 2, ph @ force))
                        if j + m + 1 < n_pow:
                            terms.append((j + m + 1, 1j * kr * np.einsum("g,gnk,k->n", ph, tau, rh)))
                        for p, amp in terms:
                            amp = rh * (rh @ amp) if mode == "P" else amp - rh * (rh @ amp)
                            out[mode][p, o] += pref * amp
    return out


def exact_series(obs, n_pow: int = J + 3):
    """The exact amplitude's coefficients of k_S^j, from the T-matrix series, P and S."""
    data = json.loads(lowf.series_path().read_text())
    a = gs.RADIUS
    dirs = obs / np.linalg.norm(obs, axis=1)[:, None]
    out = {
        "P": np.zeros((n_pow, len(obs), 3), dtype=complex),
        "S": np.zeros((n_pow, len(obs), 3), dtype=complex),
    }
    for o, rh in enumerate(dirs):
        cos_t = float(np.clip(rh[0], -1.0, 1.0))
        theta = math.acos(cos_t)
        # the polar unit vector in the plane of z (axis 0) and rh
        perp = rh - cos_t * np.array([1.0, 0.0, 0.0])
        e_theta = -np.array([1.0, 0.0, 0.0]) * math.sin(theta) + (
            cos_t * perp / np.linalg.norm(perp) if np.linalg.norm(perp) > 1e-12 else 0.0
        )
        for n, order in enumerate(data["orders"]):
            pn = legval(cos_t, [0] * n + [1])
            dpn = _dpn_dtheta(n, theta)
            for k, mat in enumerate(order["Tpsv"]):
                # T w^k with w = k_P a = RATIO a k_S; (-i / k_P) T = -i a RATIO^(k-1) a^(k-1) k_S^(k-1) t_k
                if k < 1 or k - 1 >= n_pow:
                    continue
                fac = -1j * a * (RATIO * a) ** (k - 1)
                t_pp = mat[0][0][0] + 1j * mat[0][0][1]
                t_sp = mat[1][0][0] + 1j * mat[1][0][1]
                out["P"][k - 1, o] += fac * (2 * n + 1) * t_pp * pn * rh
                out["S"][k - 1, o] += fac * (2 * n + 1) * t_sp * dpn * e_theta
    return out


def _dpn_dtheta(n: int, theta: float) -> float:
    """d P_n(cos theta) / d theta = -sin theta P_n'(cos theta)."""
    c = np.zeros(n + 1)
    c[n] = 1.0
    return float(
        -math.sin(theta) * np.polynomial.legendre.legval(math.cos(theta), np.polynomial.legendre.legder(c))
    )


def check_exact_formula(obs) -> float:
    """The asymptotic amplitude formula against mie_scattered_displacement at k_S a = 0.2, where the
    double-precision Mie evaluation is accurate, with twenty terms so that truncation is negligible."""
    ka_s, n_pow = 0.2, 20
    omega = ka_s * REF.beta / gs.RADIUS
    ks, kp = omega / REF.beta, omega / REF.alpha
    direct = mie_scattered_displacement(lowf.exact_result(omega), obs)
    ser = exact_series(obs, n_pow)
    dist = np.linalg.norm(obs, axis=1)[:, None]
    summed = (
        sum(ser["P"][p] * ks**p for p in range(n_pow)) * np.exp(1j * kp * dist) / dist
        + sum(ser["S"][p] * ks**p for p in range(n_pow)) * np.exp(1j * ks * dist) / dist
    )
    return float(np.abs(summed - direct).max() / np.abs(direct).max())


def main() -> int:
    args = sys.argv[1:]
    if "--gauss" in args:  # every orbit from the Gauss coefficients: a check of the series solve alone
        lowf.NEAR = -1
        args.remove("--gauss")
    for flag in [x for x in args if x.startswith("--rc=")]:  # the degree of the contrast in each cell
        gfft.R_C = int(flag.split("=", 1)[1])
        args.remove(flag)
    gs.CORE = 0.1 * gs.RADIUS
    profile = "smoothstep"
    for flag in [x for x in args if x.startswith("--profile=")]:
        profile = flag.split("=", 1)[1]
        args.remove(flag)
    gs.set_profile(profile)
    ladder = [int(v) for v in args] or [4, 6]
    obs = obs_points(gs.R_FAR, gs.THETA)
    print(f"profile {profile}, contrast degree {gfft.R_C}", flush=True)
    print(f"[formula] exact amplitude series vs mie_scattered_displacement at k_S a = 0.2: "
          f"{check_exact_formula(obs):.1e}", flush=True)  # fmt: skip
    ex = exact_series(obs)
    a = gs.RADIUS
    for n in ladder:
        side, centres, coefs, asm, sols = solve_series(n)
        vox = far_series(side, centres, coefs, asm, sols, obs)
        # the coefficient of (k_S a)^p is c_p a^-p; its relative error, and its size against the leading
        # term's (in brackets). The (ka)^3 coefficient is zero for the exact sphere (its T-matrix has no
        # w^4 term), so for it the error is reported against the leading term instead.
        lead = max(np.abs(ex[m][2]).max() for m in ("P", "S")) * a**-2
        row = []
        for p in range(2, J + 2):  # (ka)^(J+2) would need psi_(J+1) in its stress term
            diff = max(np.abs(vox[m][p] - ex[m][p]).max() for m in ("P", "S")) * a**-p
            size = max(np.abs(ex[m][p]).max() for m in ("P", "S")) * a**-p
            rel = diff / size if size > 1e-12 * lead else diff / lead
            row.append(f"(ka)^{p}: {rel:.2e} [{size / lead:.1e}]")
        print(f"  n_sub {n:2d}: " + "  ".join(row), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
