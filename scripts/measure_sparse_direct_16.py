#!/usr/bin/env python3
"""MEASUREMENT, not a gate: is an exact sparse solve of the sparsified system affordable at 16^3?

WHY.  Preconditioning A = I - G0 T0 with M = H^{-1} P, H the sparse 27-voxel
system solved EXACTLY, cuts GMRES iterations ~20x at 8^3
(``measure_sparsify_preconditioner.py``); every cheap sweep of H has so far
failed.  But H is sparse, and nested-dissection sparse LU costs ~N^2 to set up
and ~N^{4/3} per solve in 3-D.  At the lattices this project runs, that may
already be cheap next to building and applying G0.  This measures it.

AT 16^3 (36,864 unknowns) A is never dense (22 GB): it is applied matrix-free
through ``apply_g0_3d`` on the whole-space cache, as the solver applies it.  H
is assembled sparse from LOCAL kernel blocks (offsets <= 2), and the stencils
are fitted against a far window of R pitches; R = 7 (as at 8^3) and R = 15 (the
whole domain) are compared.  The kernel is the closed-form whole-space
propagator, CHECKED against the solver's cache on a small window before use.
Fits accumulate K_o K_o^H and K_c K_o^H in chunks, so memory stays bounded.

MEASURED: cache build, one A matvec, stencil fit, sparse LU (time, fill), one
preconditioner application, GMRES iterations with and without M.

RESULT (2026-09-27, 16^3; validated at 6^3, where it reproduces 10 / 16 / 34
exactly).  Contrast-independent costs: cache 25 s; one G0 matvec 3.7 s;
stencil fit 1 s (R = 7), 12 s (R = 15); sparse LU of H 7.5 min, 1.76e8 fill,
~3.5 GB; one preconditioner solve 0.6 s.  R = 15 buys nothing over R = 7.
Iterations were measured only at strength 10 (none 33, R = 7: 18, R = 15: 20)
and strength 30 (none 414; stopped before the preconditioned runs).  BOTH ARE
UNPHYSICAL: strength 10 is Delta lambda / lambda = 114%, strength 30 343% of
background, beyond the |Delta| < 52% validity floor.  The iteration counts are
not evidence about physical problems; re-measure at strength <= ~4 with the
scattering raised by lattice size.

Run:  conda run -n seismic python scripts/measure_sparse_direct_16.py [N]
SI units, whole space, as measure_sparsify_preconditioner.py.
"""

import itertools
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator, gmres, splu

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubic_scattering.directional_sweeps import SweepGrid3D, apply_g0_3d, build_g0_cache_3d  # noqa: E402
from cubic_scattering.horizontal_greens import exact_propagator_9x9  # noqa: E402
from scripts import measure_sparsify_preconditioner as mp  # noqa: E402

N = mp.N  # first argument; 16 for the measurement
P = mp.PITCH
D9 = np.r_[np.ones(3), np.full(6, P)]
TOL = 1e-8
T_START = time.perf_counter()


def stage(label: str) -> None:
    """Print one timed stage."""
    print(f"  [{time.perf_counter() - T_START:7.1f} s] {label}", flush=True)


def g_block(dz: int, dx: int, dy: int) -> NDArray:
    """G0 block at offset (receiver - source) in pitches; zero at the self term (it belongs to T0)."""
    if dz == 0 and dx == 0 and dy == 0:
        return np.zeros((9, 9), dtype=complex)
    return exact_propagator_9x9(dx * P, dy * P, dz * P, mp.OM, mp.REF)


def check_kernel() -> float:
    """Closed-form blocks against the solver's cache (apply_g0_3d) on a 5^3 window."""
    ker = mp.kernel_window(2)  # [dz+2, dx+2, dy+2], from apply_g0_3d
    worst = 0.0
    for dz, dx, dy in itertools.product(range(-2, 3), repeat=3):
        a, b = g_block(dz, dx, dy), ker[dz + 2, dx + 2, dy + 2]
        worst = max(worst, float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300)))
    return worst


def kernel_table(r: int) -> NDArray:
    """D9-scaled G0 blocks at every offset |d|_inf <= r + 1: tab[dz + c, dx + c, dy + c], c = r + 1."""
    c = r + 1
    n = 2 * c + 1
    tab = np.zeros((n, n, n, 9, 9), dtype=complex)
    for dz, dx, dy in itertools.product(range(-c, c + 1), repeat=3):
        tab[dz + c, dx + c, dy + c] = D9[:, None] * g_block(dz, dx, dy)
    return tab


def fit_types(r: int, lam: float = 1e-3) -> dict:
    """Centre-anchored stencils per clipping type, far window |d| <= r, chunked accumulation."""
    tab = kernel_table(r)
    c = r + 1
    out = {}
    for typ in itertools.product((-1, 0, 1), repeat=3):

        def ok(o: tuple, lim: int, t: tuple = typ) -> bool:
            return all(
                (-lim if s >= 0 else 0) <= v <= (lim if s <= 0 else 0) for v, s in zip(o, t, strict=True)
            )

        near = [o for o in itertools.product((-1, 0, 1), repeat=3) if ok(o, 1)]
        near.sort(key=lambda o: o != (0, 0, 0))
        far = [
            o for o in itertools.product(range(-r, r + 1), repeat=3) if ok(o, r) and max(map(abs, o)) > 1
        ]
        nn = 9 * len(near)
        gram = np.zeros((nn - 9, nn - 9), dtype=complex)
        cross = np.zeros((9, nn - 9), dtype=complex)
        for start in range(0, len(far), 400):
            chunk = far[start : start + 400]
            k = np.zeros((nn, 9 * len(chunk)), dtype=complex)
            for j, f in enumerate(chunk):
                for i, o in enumerate(near):
                    k[9 * i : 9 * i + 9, 9 * j : 9 * j + 9] = tab[
                        o[0] - f[0] + c, o[1] - f[1] + c, o[2] - f[2] + c
                    ]
            k_c, k_o = k[:9], k[9:]
            gram += k_o @ k_o.conj().T
            cross += k_c @ k_o.conj().T
        reg = lam * float(np.linalg.eigvalsh(gram)[-1]) * np.eye(gram.shape[0])
        xh = -np.linalg.solve((gram + reg).T, cross.T).T
        out[typ] = (near, np.vstack([np.eye(9), xh.conj().T]))
    return out


def assemble_sparse(types: dict, t_blocks: NDArray) -> tuple[sp.csc_matrix, sp.csr_matrix]:
    """H and P from local blocks only: H row = alpha^H D psi_mu - (alpha^H D G0[mu, mu]) T0 psi_mu."""
    size = 9 * N**3
    near_g = {(a, b, c): g_block(a, b, c) for a, b, c in itertools.product(range(-2, 3), repeat=3)}
    rows, cols, hv, pv = [], [], [], []
    for z, x, y in itertools.product(range(N), repeat=3):
        typ = tuple(-1 if v == 0 else (1 if v == N - 1 else 0) for v in (z, x, y))
        near, alpha = types[typ]
        ah = alpha.conj().T * np.tile(D9, len(near))[None, :]
        r0 = 9 * ((z * N + x) * N + y)
        for j, o in enumerate(near):
            zz, xx, yy = z + o[0], x + o[1], y + o[2]
            c0 = 9 * ((zz * N + xx) * N + yy)
            beta = np.zeros((9, 9), dtype=complex)
            for i2, o2 in enumerate(near):
                beta += ah[:, 9 * i2 : 9 * i2 + 9] @ near_g[o2[0] - o[0], o2[1] - o[1], o2[2] - o[2]]
            blk_h = ah[:, 9 * j : 9 * j + 9] - beta @ t_blocks[zz, xx, yy]
            blk_p = ah[:, 9 * j : 9 * j + 9]
            ri, ci = np.meshgrid(np.arange(r0, r0 + 9), np.arange(c0, c0 + 9), indexing="ij")
            rows.append(ri.ravel())
            cols.append(ci.ravel())
            hv.append(blk_h.ravel())
            pv.append(blk_p.ravel())
    ri, ci = np.concatenate(rows), np.concatenate(cols)
    h = sp.csc_matrix((np.concatenate(hv), (ri, ci)), shape=(size, size))
    p = sp.csr_matrix((np.concatenate(pv), (ri, ci)), shape=(size, size))
    return h, p


def main() -> int:
    """Costs and iterations at N^3.

    Returns:
        0.
    """
    print("=" * 78)
    print(f"EXACT SPARSE SOLVE OF THE SPARSIFIED SYSTEM AT {N}^3 ({9 * N**3} unknowns), whole space")
    print("=" * 78)
    err = check_kernel()
    stage(f"closed-form kernel vs the solver's cache on a 5^3 window: {err:.1e}")
    if not err < 1e-12:
        msg = (
            f"the closed-form kernel disagrees with apply_g0_3d by {err:.2e}.\n"
            "  Where: scripts/measure_sparse_direct_16.py, g_block\n"
            "  Valid: agreement to round-off (< 1e-12)\n"
            "  Fix:   check the (x, y, z) argument order of exact_propagator_9x9."
        )
        raise RuntimeError(msg)
    grid = SweepGrid3D(n_z=N, n_x=N, n_y=N, pitch=P)
    cache = build_g0_cache_3d(grid, mp.REF, mp.OM)
    stage("whole-space cache built")
    rng = np.random.default_rng(1)
    b = rng.standard_normal(9 * N**3) + 1j * rng.standard_normal(9 * N**3)
    t0 = time.perf_counter()
    apply_g0_3d(b.reshape(N, N, N, 9), cache)
    t_matvec = time.perf_counter() - t0
    stage(f"one G0 matvec: {t_matvec:.2f} s")
    fits = {}
    for r in (7, 15) if N > 8 else (7,):
        t0 = time.perf_counter()
        fits[r] = fit_types(r)
        stage(f"27 stencil types fitted, R = {r}: {time.perf_counter() - t0:.0f} s")
    size = 9 * N**3
    for strength in (10.0, 30.0, 60.0):
        tb = mp.t_blocks(strength)

        def a_mv(v: NDArray, tb=tb) -> NDArray:
            psi = v.reshape(N, N, N, 9)
            return (psi - apply_g0_3d(np.einsum("zxyab,zxyb->zxya", tb, psi), cache)).ravel()

        a_op = LinearOperator((size, size), matvec=a_mv, dtype=complex)
        its0: list[float] = []
        restart = 500
        gmres(a_op, b, rtol=TOL, restart=restart, maxiter=4, callback=its0.append, callback_type="pr_norm")
        print(f"\n  strength {strength:g}: GMRES, no preconditioner: {len(its0)} iterations", flush=True)
        for r, types in fits.items():
            t0 = time.perf_counter()
            h, p = assemble_sparse(types, tb)
            t_asm = time.perf_counter() - t0
            t0 = time.perf_counter()
            lu = splu(h)
            t_lu = time.perf_counter() - t0
            fill = lu.L.nnz + lu.U.nnz
            t0 = time.perf_counter()
            lu.solve(p @ b)
            t_solve = time.perf_counter() - t0
            m_op = LinearOperator((size, size), matvec=lambda v, s=lu, pp=p: s.solve(pp @ v), dtype=complex)
            am = LinearOperator((size, size), matvec=lambda v, mo=m_op: a_mv(mo.matvec(v)), dtype=complex)
            its: list[float] = []
            y, info = gmres(
                am, b, rtol=TOL, restart=restart, maxiter=4, callback=its.append, callback_type="pr_norm"
            )
            true = float(np.linalg.norm(a_mv(m_op.matvec(y)) - b) / np.linalg.norm(b))
            print(
                f"    R = {r:2d}: assemble {t_asm:.0f} s, LU {t_lu:.1f} s (fill {fill / 1e6:.1f}M nnz, "
                f"~{fill * 20 / 1e9:.2f} GB), one solve {t_solve:.2f} s; "
                f"GMRES {len(its) if info == 0 else -1} iterations (true residual {true:.1e})",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
