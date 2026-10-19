"""FFT matrix-vector solver for the graded voxel on the sphere grid.

The coupling blocks K(o) depend only on the grid offset o, and each cell's contrast E_n acts on its own
cell, so the Galerkin operator (M - K E) is a block-Toeplitz convolution followed by a cell-local map:

    y_m = M psi_m - sum_n K(g_m - g_n) s_n,     s_n = E_n psi_n  (the source monomial coefficients),

which is evaluated with FFTs on the padded grid of (2n - 1)^3 points (circular embedding of the offsets).

THE BLOCKS BY THE CUBE'S SYMMETRY.  The background is isotropic, so P(Q r) = S P(r) S^T for every signed
permutation Q (48 of them), S = diag(Q, R6(Q)); for signed permutations the tensor and engineering Voigt
representations coincide.  The cube maps to itself, so with xi = Q eta, L_a(Q eta) = sum_b T_ab L_b(eta),
m_c(Q eta) = sum_d U_cd m_d(eta):

    K(Q o)_ac = sum_bd T_ab U_cd  S K_bd(o) S^T.

Only one offset per orbit is computed (``coupling_block``); the rest follow exactly.

THE SOLVE.  GMRES with a block-Jacobi preconditioner: each cell's own (M - K(0) E_n)^-1, which carries the
self-interaction that dominates the second-order term.  scipy's ``maxiter`` counts RESTART CYCLES, so the
total number of iterations is bounded explicitly as restart * max_cycles.
"""

import itertools
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator, gmres

from ..effective_contrasts import MaterialContrast, ReferenceMedium
from ..sphere_scattering import _plane_wave_strain_voigt
from ..sphere_scattering_fft import _build_grid_index_map
from .basis import (
    contrast_values,
    gram_test,
    monomials,
    source_expansion,
    source_exponents,
)
from .blocks import coupling_block
from .site import cell_contrast_coefficients
from .solver import GradedVoxelResult, field_sizes, plane_wave_moments

VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


@lru_cache(maxsize=1)
def signed_permutations() -> tuple[NDArray, ...]:
    """The 48 signed permutation matrices of the cube group."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            q = np.zeros((3, 3))
            for i, j in enumerate(perm):
                q[i, j] = signs[i]
            out.append(q)
    return tuple(out)


def _voigt_rep(q: NDArray) -> NDArray:
    """R6 with (Q e Q^T) in Voigt = R6 e in Voigt, for a signed permutation Q."""
    rep = np.zeros((6, 6))
    for b, (m, n) in enumerate(VOIGT_PAIRS):
        e = np.zeros((3, 3))
        e[m, n] = e[n, m] = 1.0
        er = q @ e @ q.T
        for a, (i, j) in enumerate(VOIGT_PAIRS):
            rep[a, b] = er[i, j]
    return rep


def symmetry_reps(
    q: NDArray, n_source: int = 10, n_test: int = 4
) -> tuple[NDArray, NDArray, NDArray]:
    """(T, U, S): the field-function (n_test square), source-monomial (n_source square) and 9-component
    (9 x 9) representations of Q."""
    key = (q.tobytes(), n_source, n_test)
    if key not in _REPS:
        _REPS[key] = _symmetry_reps(q, n_source, n_test)
    return _REPS[key]


_REPS: dict[tuple, tuple[NDArray, NDArray, NDArray]] = {}


def _symmetry_reps(
    q: NDArray, n_source: int, n_test: int
) -> tuple[NDArray, NDArray, NDArray]:
    rng = np.random.default_rng(12345)
    eta = rng.uniform(-1.0, 1.0, (96, 3))
    fa = contrast_values(eta @ q.T)[:n_test]  # Q_a(Q eta_k); the first four are L_a
    fb = contrast_values(eta)[:n_test]
    t = np.round(np.linalg.lstsq(fb.T, fa.T, rcond=None)[0].T)  # exactly 0 or +-1
    exps = source_exponents(n_source)
    a = monomials(exps, eta @ q.T)  # m_c(Q eta_k)
    b = monomials(exps, eta)  # m_d(eta_k)
    u = np.linalg.lstsq(b.T, a.T, rcond=None)[0].T
    u = np.round(u)  # exactly 0 or +-1 for a signed permutation
    s = np.zeros((9, 9))
    s[:3, :3] = q
    s[3:, 3:] = _voigt_rep(q)
    return t, u, s


def _signed_permutation(m: NDArray) -> tuple[NDArray, NDArray] | None:
    """(perm, sign) with m[i, perm[i]] = sign[i] the only nonzero of row i, or None if m is not one."""
    perm = np.argmax(np.abs(m), axis=1)
    sign = m[np.arange(len(m)), perm]
    if not (
        np.all(np.abs(sign) == 1.0)
        and np.count_nonzero(m) == len(m)
        and len(set(perm)) == len(m)
    ):
        return None
    return perm, sign


def _transform(q: NDArray, block: NDArray) -> NDArray:
    """T K S^T under the cube symmetry q, with U and S on the source and 9-component indices.

    Every representation of a signed permutation is itself one, so the product is a reindexing with signs:
    no arithmetic beyond the signs (the dense einsum it replaces cost 0.05 s per offset at degree two).
    """
    t, u, s = symmetry_reps(q, block.shape[1], block.shape[0])
    sp = [_signed_permutation(m) for m in (t, u, s)]
    if any(x is None for x in sp):
        tmp = np.einsum("ij,bdjk,lk->bdil", s, block, s)
        return np.einsum("ab,cd,bdil->acil", t, u, tmp)
    (pt, st), (pu, su), (ps, ss) = sp
    out = block[np.ix_(pt, pu, ps, ps)]
    return out * (
        st[:, None, None, None]
        * su[None, :, None, None]
        * ss[None, None, :, None]
        * ss[None, None, None, :]
    )


def offset_blocks(
    n_sub: int,
    h: float,
    omega: float,
    ref: ReferenceMedium,
    n_source: int = 10,
    n_test: int = 4,
) -> dict[tuple[int, int, int], NDArray]:
    """K(o) for every offset o in [-(n-1), n-1]^3: one ``coupling_block`` per cube-group orbit."""
    canonical = canonical_blocks(n_sub, h, omega, ref, n_source, n_test)
    return {off: _transform(q, canonical[c]) for off, c, q in _orbit_map(n_sub)}


def canonical_blocks(
    n_sub: int,
    h: float,
    omega: float,
    ref: ReferenceMedium,
    n_source: int = 10,
    n_test: int = 4,
) -> dict[tuple[int, int, int], NDArray]:
    """One ``coupling_block`` per cube-group orbit of the offsets in [-(n-1), n-1]^3, keyed by the orbit's
    canonical offset (a >= b >= d >= 0): about 1/48 of the offsets, so the kernel need not be held per
    offset (``transformed_kernel``)."""
    rng = range(n_sub)
    return {
        c: coupling_block(c, h, omega, ref, n_source, n_test)
        for c in itertools.product(rng, rng, rng)
        if c[0] >= c[1] >= c[2]
    }


@lru_cache(maxsize=4)
def _orbit_map(n_sub: int) -> tuple[tuple, ...]:
    """(offset, canonical offset, the signed permutation taking the one to the other) for every offset."""
    qs = signed_permutations()
    out = []
    rng = range(-(n_sub - 1), n_sub)
    for off in itertools.product(rng, rng, rng):
        c = tuple(sorted((abs(o) for o in off), reverse=True))
        cv, target = np.array(c, dtype=float), np.array(off, dtype=float)
        out.append((off, c, next(q for q in qs if np.array_equal(q @ cv, target))))
    return tuple(out)


def transformed_kernel(
    canonical: dict[tuple[int, int, int], NDArray],
    n_sub: int,
    na: int,
    nc: int,
    dtype: type = np.complex128,
    path: Path | None = None,
    budget_bytes: float = 2e9,
) -> NDArray:
    """The coupling's kernel on the (2n - 1)^3 circular embedding, transformed: kh[row, col, i, j, k] with
    row = (a, i9) of the test functions and col = (c, j9) of the source monomials, as the FFT solve uses it.

    Built a group of rows at a time from the canonical blocks (``canonical_blocks``), each group's
    real-space kernel assembled by the cube's symmetry, transformed in double precision and stored in
    ``dtype``; the group's size keeps its working arrays within ``budget_bytes``.  In memory, or
    memory-mapped at ``path`` (created, overwritten), so that the model's size is bounded by the disk, not
    the memory.
    """
    npad = 2 * n_sub - 1
    rows, cols = na * 9, nc * 9
    shape = (rows, cols, npad, npad, npad)
    if path is None:
        kh = np.empty(shape, dtype=dtype)
    else:
        kh = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    per_row = 2 * cols * npad**3 * 16  # the group's real-space array and its transform, complex128
    group = max(1, min(rows, int(budget_bytes // per_row)))
    orbits = _orbit_map(n_sub)
    for r0 in range(0, rows, group):
        sel = slice(r0, min(rows, r0 + group))
        real = np.zeros((sel.stop - sel.start, cols, npad, npad, npad), dtype=complex)
        for off, c, q in orbits:
            blk = _transform(q, canonical[c])[:na, :nc].transpose(0, 2, 1, 3).reshape(rows, cols)
            real[:, :, off[0] % npad, off[1] % npad, off[2] % npad] = blk[sel]
        kh[sel] = np.fft.fftn(real, axes=(2, 3, 4))
        del real
    if path is not None:
        kh.flush()
    return kh


def solve_graded_sphere_fft(
    omega: float,
    radius: float,
    ref: ReferenceMedium,
    contrast: MaterialContrast,
    n_sub: int,
    profile: Callable[[NDArray], float],
    k_hat: NDArray,
    pol: NDArray,
    wave_type: str,
    p: int = 1,
    r: int = 1,
    gmres_tol: float = 1e-11,
    restart: int = 100,
    max_cycles: int = 20,
    blocks: dict | None = None,
    kernel_dtype: type = np.complex128,
    kernel_path: Path | None = None,
) -> GradedVoxelResult:
    """The graded-voxel solve of ``solver.solve_graded_sphere``, with an FFT matvec and GMRES.

    Same cells, contrast projection and right-hand side as the dense solver (every cell overlapping the
    sphere is kept).  The transformed kernel is built from the canonical blocks (``transformed_kernel``),
    stored in ``kernel_dtype`` (complex64 halves its memory; the product is then that of a fixed operator
    perturbed by ~1e-7, so GMRES converges as before and the solution moves by about that) and, with
    ``kernel_path``, memory-mapped to that file rather than held in memory.  ``blocks`` (every offset's
    block, ``offset_blocks``) is the former path, kept.

    Raises:
        RuntimeError: when GMRES does not reach ``gmres_tol`` within restart * max_cycles iterations.
    """
    h_cell = radius / n_sub
    grid_idx, centres, h = _build_grid_index_map(
        radius,
        n_sub,
        lambda q: bool(np.linalg.norm(q) < radius + np.sqrt(3.0) * h_cell),
    )
    n = len(centres)
    na, n_field, _, n_source = field_sizes(p, r)
    nc = 1 if (p == 0 and r == 0) else n_source
    delta = np.array(
        [
            cell_contrast_coefficients(profile, c, h, contrast, ref, omega, degree=r)
            for c in centres
        ]
    )
    e_cells = np.array(
        [source_expansion(d, n_field)[:nc, :na] for d in delta]
    )  # (N, nc, na, 9, 9)
    npad = 2 * n_sub - 1
    rows, cols = na * 9, nc * 9
    if blocks is None:
        # the canonical blocks alone; the kernel assembled from them a group of rows at a time
        canonical = canonical_blocks(n_sub, h, omega, ref, n_source, n_field)
        kh = transformed_kernel(canonical, n_sub, na, nc, kernel_dtype, kernel_path)
        k0 = canonical[(0, 0, 0)][:na, :nc]
        del canonical
    else:
        kh = np.zeros((rows, cols, npad, npad, npad), dtype=complex)
        for off, blk in blocks.items():
            kh[:, :, off[0] % npad, off[1] % npad, off[2] % npad] = (
                blk[:na, :nc].transpose(0, 2, 1, 3).reshape(rows, cols)
            )
        for row in range(rows):
            kh[row] = np.fft.fftn(kh[row], axes=(1, 2, 3))
        kh = kh.astype(kernel_dtype, copy=False)
        k0 = blocks[(0, 0, 0)][:na, :nc]
    g0, g1, g2 = grid_idx[:, 0], grid_idx[:, 1], grid_idx[:, 2]
    m9 = gram_test(h, n_field)[:na, :na]

    def matvec(x: NDArray) -> NDArray:
        psi = x.reshape(n, na, 9)
        src = np.einsum("ncbij,nbj->nci", e_cells, psi).reshape(n, cols)
        grid = np.zeros((cols, npad, npad, npad), dtype=complex)
        grid[:, g0, g1, g2] = src.T
        sh = np.fft.fftn(grid, axes=(1, 2, 3)).reshape(cols, -1)
        yh = np.empty((rows, sh.shape[1]), dtype=complex)
        for row in range(rows):
            yh[row] = np.einsum("cf,cf->f", np.asarray(kh[row]).reshape(cols, -1), sh)
        y = np.fft.ifftn(yh.reshape(rows, npad, npad, npad), axes=(1, 2, 3))[
            :, g0, g1, g2
        ].T
        return (np.einsum("ab,nbi->nai", m9, psi).reshape(n, rows) - y).ravel()

    pre = np.array(
        [
            np.linalg.inv(
                np.kron(m9, np.eye(9))
                - np.einsum("acij,cbjk->aibk", k0, en).reshape(rows, rows)
            )
            for en in e_cells
        ]
    )

    def precondition(x: NDArray) -> NDArray:
        return np.einsum("nij,nj->ni", pre, x.reshape(n, rows)).ravel()

    dim = n * rows
    a_op = LinearOperator((dim, dim), matvec=matvec, dtype=complex)
    m_op = LinearOperator((dim, dim), matvec=precondition, dtype=complex)
    k_mag = omega / (ref.alpha if wave_type == "P" else ref.beta)
    k_hat = np.asarray(k_hat, dtype=float) / np.linalg.norm(k_hat)
    amp = np.concatenate(
        [np.asarray(pol, dtype=complex), _plane_wave_strain_voigt(k_hat, pol, k_mag)]
    )
    rhs = plane_wave_moments(centres, h, k_mag * k_hat, amp, n_field)[:, :na].ravel()
    sol, info = gmres(
        a_op,
        rhs,
        x0=precondition(rhs),
        M=m_op,
        rtol=gmres_tol,
        atol=0.0,
        restart=restart,
        maxiter=max_cycles,
    )
    if info != 0:
        raise RuntimeError(
            f"solve_graded_sphere_fft: GMRES did not reach rtol {gmres_tol:g} "
            f"within {restart * max_cycles} iterations (restart {restart} x {max_cycles} cycles; "
            f"info {info}) at n_sub = {n_sub}. Fix: raise "
            "max_cycles or restart, or loosen gmres_tol."
        )
    full = np.zeros((n, n_field, 9), dtype=complex)
    full[:, :na] = sol.reshape(n, na, 9)
    return GradedVoxelResult(centres, grid_idx, h, omega, ref, delta, full, p, r)
