#!/usr/bin/env python3
"""The gradient hierarchy on the graded sphere on FINER grids: an iterative solve with an FFT product.

``measure_graded_sphere_gradient_hierarchy.py`` solves the scheme densely and stops at 8 voxels across
for third gradients. Here the same system is solved by GMRES with the product done by fast Fourier
transform, which the scheme allows exactly: the block that couples two voxels depends on their offset
alone,

    field derivatives at c = sum_{c'} sum_V c^{c'}_V B_V(c - c') U^{c'} ,

a convolution over the lattice for each monomial V of the contrast. Nothing is approximated by the
transform; the answer is the dense solve's to the tolerance of the iteration, which [check] verifies.

Three economies make the finer grids affordable, none of them an approximation of the scheme:

* The contrast is held to degree R_C = 1 in each voxel instead of the field's degree. The order of a
  scheme is the smaller of the field's and the medium's, and a projected linear contrast is fourth order
  (its error is orthogonal to the linear functions), so nothing is lost at fourth order. It cuts the
  number of blocks B_V from 20 to 4.
* The blocks are computed for offsets in one octant only. A mirror of the offset in axis m multiplies
  the entry [(i, P), (j, W)] of B_V by (-1) to the number of times m occurs among i, P, j, W and V,
  because every factor (the Green's tensor, its derivatives, the weights) has a definite parity.
  [parity] checks this against a block computed directly.
* The self block (zero offset) is the cube's moments, computed once per grid.

Measured as before: far field at nine angles against the exact graded sphere.

Run:  python -u scripts/measure_graded_sphere_gradient_hierarchy_fft.py --core=1 --q=3 --check 4 6
      python -u scripts/measure_graded_sphere_gradient_hierarchy_fft.py --core=1 --q=3 8 10 12
"""

import itertools
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import measure_ball_gradient_hierarchy as hier  # noqa: E402
import measure_graded_sphere_gradient_hierarchy as gs  # noqa: E402
from crosscheck_graded_sphere import graded_mie_result  # noqa: E402
from cubic_scattering.sphere_scattering import mie_scattered_displacement  # noqa: E402
from gate_sphere_cell_average_vs_mie import obs_points  # noqa: E402

AXES = (0, 1, 2)
R_C = 1


def parity_signs(asm: gs.Assembler):
    """d[m, (i, P)] = (-1)^([i == m] + count of m in P);  s[m, V] = (-1)^(count of m in V)."""
    d = np.ones((3, 3, asm.nu))
    for m in AXES:
        for i in AXES:
            for wi, w in enumerate(asm.u_list):
                d[m, i, wi] = (-1.0) ** ((i == m) + w.count(m))
    s = np.array([[(-1.0) ** v.count(m) for v in asm.v_list] for m in AXES])
    return d.reshape(3, 3 * asm.nu), s


def octant_blocks(n_sub: int, side: float, omega: float, asm: gs.Assembler) -> dict:
    """The blocks of every offset in the first octant: one integrated per orbit of the cube group."""
    at = gs.symmetric_blocks(asm, side, omega)
    return {key: at(key) for key in itertools.product(range(n_sub), repeat=3)}


def check_parity(side: float, omega: float, asm: gs.Assembler) -> bool:
    d, s = parity_signs(asm)
    key, flipped = (1, 2, 0), (-1, 2, 0)

    def at(offset: tuple[int, int, int]) -> np.ndarray:
        delta = side * np.array(offset, float)
        return asm.blocks(gs.coupling_array(delta, side, omega, asm.d_list, asm.w_list, 12))

    base, direct = at(key), at(flipped)
    derived = base * s[0][:, None, None] * d[0][None, :, None] * d[0][None, None, :]
    err = float(np.abs(direct - derived).max() / np.abs(direct).max())
    ok = err < 1e-10
    verdict = "PASS" if ok else "FAIL"
    print(f"  [parity] mirrored block from the octant block: {err:.1e}   {verdict}", flush=True)
    return ok


def _place(
    octant: dict, n_fft: int, d: np.ndarray, s: np.ndarray, vi: int, rows: slice, n_v: int
) -> np.ndarray:
    """B_V on the periodic offset grid for the rows ``rows`` of the block, from the octant blocks.

    The blocks of the other octants follow by the mirrors (``parity_signs``).
    """
    nu3 = d.shape[1]
    n_rows = len(range(nu3)[rows])
    out = np.zeros((n_fft, n_fft, n_fft, n_rows, nu3), dtype=complex)
    for key, blk in octant.items():
        for flips in itertools.product((False, True), repeat=3):
            if any(f and k == 0 for f, k in zip(flips, key, strict=True)):
                continue
            dd = np.ones(nu3)
            sv = 1.0
            idx = []
            for m in AXES:
                if flips[m]:
                    dd = dd * d[m]
                    sv *= s[m][vi]
                    idx.append((-key[m]) % n_fft)
                else:
                    idx.append(key[m])
            out[idx[0], idx[1], idx[2]] = blk[vi, rows] * sv * dd[rows, None] * dd[None, :]
    return out


#: rows of the 60 x 60 blocks transformed at a time, which bounds the transient memory of the build
ROW_CHUNK = 12


def solve_fft(n_sub: int, q: int, omega: float, contrast, tol: float = 1e-10, check_octant: bool = False):
    side, centres, coefs, grid = gs.build_cells(n_sub, R_C)
    asm = gs.Assembler(q, R_C, omega, contrast)
    nu3 = 3 * asm.nu
    n_v = len(asm.v_list)
    n_fft = 2 * n_sub
    t0 = time.perf_counter()
    octant = octant_blocks(n_sub, side, omega, asm)
    t_tab = time.perf_counter() - t0
    d, s = parity_signs(asm)

    # The blocks' transform over the three offset axes, kept for ONE OCTANT of wavevectors only. A mirror M
    # of the offsets maps B_V(o) to s_V D B_V(o) D (``parity_signs``), so the transform obeys
    # B^_V(M k) = s_V D B^_V(k) D, and the other octants follow from the stored one in the product:
    # memory falls from (2n)^3 to (n + 1)^3 wavevectors. Built a few rows at a time.
    k_half = n_fft // 2 + 1
    b_oct = np.zeros((n_v, k_half, k_half, k_half, nu3, nu3), dtype=complex)
    for vi in range(n_v):
        for r0 in range(0, nu3, ROW_CHUNK):
            rows = slice(r0, min(r0 + ROW_CHUNK, nu3))
            spatial = _place(octant, n_fft, d, s, vi, rows, n_v)
            b_oct[vi, :, :, :, rows] = np.fft.fftn(spatial, axes=(0, 1, 2))[:k_half, :k_half, :k_half]
    # the eight reflections of the octant, on disjoint index ranges: '+' takes 0 .. n, '-' takes n+1 .. 2n-1
    # and reads the stored octant at 2n - k
    patterns = []
    for sigma in itertools.product((1, -1), repeat=3):
        tgt, src, dd = [], [], np.ones(nu3)
        sv = np.ones(n_v)
        for m, sg in enumerate(sigma):
            if sg > 0:
                tgt.append(np.arange(0, k_half))
                src.append(np.arange(0, k_half))
            else:
                t_idx = np.arange(k_half, n_fft)
                tgt.append(t_idx)
                src.append(n_fft - t_idx)
                dd = dd * d[m]
                sv = sv * s[m]
        if all(len(t) for t in tgt):
            patterns.append((np.ix_(*tgt), np.ix_(*src), dd, sv))

    coef_grid = np.zeros((n_sub, n_sub, n_sub, n_v))
    mask = np.zeros((n_sub, n_sub, n_sub), dtype=bool)
    gi = tuple(grid.T)
    coef_grid[gi] = coefs
    mask[gi] = True

    kp = omega / hier.REF.alpha
    rhs = np.zeros((n_sub, n_sub, n_sub, nu3), dtype=complex)
    for c, xc in enumerate(centres):
        phase = np.exp(1j * kp * (gs.K_HAT @ xc))
        for pi, p_idx in enumerate(asm.u_list):
            der = np.prod([1j * kp * gs.K_HAT[ax] for ax in p_idx]) if p_idx else 1.0
            for i in AXES:
                rhs[tuple(grid[c])][i * asm.nu + pi] = gs.POL[i] * der * phase

    def matvec(vec: np.ndarray) -> np.ndarray:
        x = vec.reshape(n_sub, n_sub, n_sub, nu3)
        acc = np.zeros((n_fft, n_fft, n_fft, nu3), dtype=complex)
        for vi in range(n_v):
            z = np.zeros((n_fft, n_fft, n_fft, nu3), dtype=complex)
            z[:n_sub, :n_sub, :n_sub] = coef_grid[..., vi, None] * x
            z_hat = np.fft.fftn(z, axes=(0, 1, 2))
            for tgt, src, dd, sv in patterns:
                zz = z_hat[tgt] * dd
                acc[tgt] += sv[vi] * dd * np.matmul(b_oct[vi][src], zz[..., None])[..., 0]
        y = np.fft.ifftn(acc, axes=(0, 1, 2))[:n_sub, :n_sub, :n_sub]
        return (x - mask[..., None] * y).ravel()

    if check_octant:
        # the product with the blocks stored for every wavevector (as before this storage), on a random
        # vector
        full = np.zeros((n_v, n_fft, n_fft, n_fft, nu3, nu3), dtype=complex)
        for vi in range(n_v):
            full[vi] = np.fft.fftn(_place(octant, n_fft, d, s, vi, slice(0, nu3), n_v), axes=(0, 1, 2))
        rng = np.random.default_rng(5)
        vec = rng.normal(size=n_sub**3 * nu3) + 1j * rng.normal(size=n_sub**3 * nu3)
        x = vec.reshape(n_sub, n_sub, n_sub, nu3)
        acc = np.zeros((n_fft, n_fft, n_fft, nu3), dtype=complex)
        for vi in range(n_v):
            z = np.zeros((n_fft, n_fft, n_fft, nu3), dtype=complex)
            z[:n_sub, :n_sub, :n_sub] = coef_grid[..., vi, None] * x
            acc += np.matmul(full[vi], np.fft.fftn(z, axes=(0, 1, 2))[..., None])[..., 0]
        y_full = (x - mask[..., None] * np.fft.ifftn(acc, axes=(0, 1, 2))[:n_sub, :n_sub, :n_sub]).ravel()
        diff = float(np.abs(matvec(vec) - y_full).max() / np.abs(y_full).max())
        verdict = "PASS" if diff < 1e-12 else "FAIL"
        print(
            f"  [octant] product from one octant of wavevectors against all of them: {diff:.1e}"
            f"   {verdict}",
            flush=True,
        )
        del full
        if diff >= 1e-12:
            raise RuntimeError(f"the octant storage changes the product by {diff:.1e}")

    size = n_sub**3 * nu3
    its = [0]

    def count(_res) -> None:
        its[0] += 1

    t0 = time.perf_counter()
    sol, info = gmres(
        LinearOperator((size, size), matvec=matvec, dtype=complex),
        rhs.ravel(),
        rtol=tol,
        atol=0.0,
        restart=60,
        maxiter=5,
        callback=count,
        callback_type="pr_norm",
    )
    if info != 0:
        raise RuntimeError(f"GMRES did not reach {tol:g} in 300 iterations (info = {info})")
    t_solve = time.perf_counter() - t0
    sol = sol.reshape(n_sub, n_sub, n_sub, nu3)[gi].reshape(len(centres), 3, asm.nu)
    return side, centres, coefs, asm, sol, {"tables_s": t_tab, "solve_s": t_solve, "iterations": its[0]}


def main() -> int:
    ka_s, qs, ladder, check = 0.5, [3], [], False
    for a in sys.argv[1:]:
        if a.startswith("--ka="):
            ka_s = float(a.split("=", 1)[1])
        elif a.startswith("--core="):
            gs.CORE = float(a.split("=", 1)[1])
        elif a.startswith("--q="):
            qs = [int(v) for v in a.split("=", 1)[1].split(",")]
        elif a == "--check":
            check = True
        elif a.startswith("--profile="):
            gs.set_profile(a.split("=", 1)[1])
        else:
            ladder.append(int(a))
    contrast = hier.CONTRAST
    omega = ka_s * hier.REF.beta / gs.RADIUS
    obs = obs_points(gs.R_FAR, gs.THETA)
    n_max = max(8, int(np.ceil(ka_s + 4 * ka_s ** (1 / 3) + 6)))
    mie = graded_mie_result(omega, gs.RADIUS, gs.CORE, hier.REF, contrast, n_max)
    exact = mie_scattered_displacement(mie, obs)
    peak = float(np.max(np.abs(exact)))
    shell = gs.RADIUS - gs.CORE
    print(
        f"graded sphere by the gradient hierarchy (FFT product), k_S a = {ka_s}, shell {shell} m, "
        f"{gs.PROFILE}, "
        f"contrast degree {R_C}",
        flush=True,
    )
    ok = True
    for q in qs:
        rows = []
        for n in ladder or [4]:
            if n == (ladder or [4])[0]:
                side0 = 2.0 * gs.RADIUS / n
                ok = check_parity(side0, omega, gs.Assembler(q, R_C, omega, contrast)) and ok
            side, centres, coefs, asm, sol, info = solve_fft(n, q, omega, contrast, check_octant=check)
            got = gs.far_field(side, centres, coefs, asm, sol, omega, contrast, obs)
            err = float(np.max(np.abs(got - exact)) / peak)
            rows.append((n, err))
            note = ""
            if check:
                _, _, _, _, dense = gs.solve(n, q, omega, contrast, r_c=R_C)
                diff = float(np.abs(sol - dense).max() / np.abs(dense).max())
                good = diff < 1e-7
                ok = ok and good
                note = f"   [check] equals the dense solve to {diff:.1e} {'PASS' if good else 'FAIL'}"
            print(
                f"  q = {q}  n_sub {n:2d}  voxels {len(centres):5d}  "
                f"unknowns {3 * asm.nu * len(centres):6d}  error {err:.3e}   "
                f"({shell / side:.1f} voxels across the shell; tables {info['tables_s']:.0f} s, "
                f"{info['iterations']} iterations in {info['solve_s']:.0f} s){note}",
                flush=True,
            )
        for (n1, e1), (n2, e2) in zip(rows, rows[1:], strict=False):
            print(f"  q = {q}  apparent order {n1} -> {n2}: {math.log(e1 / e2) / math.log(n2 / n1):.2f}")
    print("CHECKS PASS" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
