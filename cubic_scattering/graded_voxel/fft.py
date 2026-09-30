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

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator, gmres

from ..effective_contrasts import MaterialContrast, ReferenceMedium
from ..sphere_scattering import _plane_wave_strain_voigt
from ..sphere_scattering_fft import _build_grid_index_map
from .basis import gram_test, monomials, source_expansion, source_exponents
from .blocks import coupling_block
from .site import cell_contrast_coefficients
from .solver import GradedVoxelResult, plane_wave_moments

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


def symmetry_reps(q: NDArray, n_source: int = 10) -> tuple[NDArray, NDArray, NDArray]:
    """(T, U, S): the test (4 x 4), source-monomial (n_source square) and 9-component (9 x 9) reps of Q."""
    t = np.zeros((4, 4))
    t[0, 0] = 1.0
    t[1:, 1:] = q  # L_{i+1}(Q eta) = (Q eta)_i
    rng = np.random.default_rng(12345)
    eta = rng.uniform(-1.0, 1.0, (48, 3))
    exps = source_exponents(n_source)
    a = monomials(exps, eta @ q.T)  # m_c(Q eta_k)
    b = monomials(exps, eta)  # m_d(eta_k)
    u = np.linalg.lstsq(b.T, a.T, rcond=None)[0].T
    u = np.round(u)  # exactly 0 or +-1 for a signed permutation
    s = np.zeros((9, 9))
    s[:3, :3] = q
    s[3:, 3:] = _voigt_rep(q)
    return t, u, s


def _transform(q: NDArray, block: NDArray) -> NDArray:
    t, u, s = symmetry_reps(q, block.shape[1])
    tmp = np.einsum("ij,bdjk,lk->bdil", s, block, s)
    return np.einsum("ab,cd,bdil->acil", t, u, tmp)


def offset_blocks(
    n_sub: int, h: float, omega: float, ref: ReferenceMedium, n_source: int = 10
) -> dict[tuple[int, int, int], NDArray]:
    """K(o) for every offset o in [-(n-1), n-1]^3: one ``coupling_block`` per cube-group orbit."""
    qs = signed_permutations()
    canonical: dict[tuple[int, int, int], NDArray] = {}
    out: dict[tuple[int, int, int], NDArray] = {}
    rng = range(-(n_sub - 1), n_sub)
    for off in itertools.product(rng, rng, rng):
        a, b, d = sorted((abs(o) for o in off), reverse=True)
        c = (a, b, d)
        if c not in canonical:
            canonical[c] = coupling_block(c, h, omega, ref, n_source)
        cv = np.array(c, dtype=float)
        target = np.array(off, dtype=float)
        q = next(q for q in qs if np.array_equal(q @ cv, target))
        out[off] = _transform(q, canonical[c])
    return out


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
) -> GradedVoxelResult:
    """The graded-voxel solve of ``solver.solve_graded_sphere``, with an FFT matvec and GMRES.

    Same cells, contrast projection and right-hand side as the dense solver (every cell overlapping the
    sphere is kept).

    Raises:
        RuntimeError: when GMRES does not reach ``gmres_tol`` within restart * max_cycles iterations.
    """
    h_cell = radius / n_sub
    grid_idx, centres, h = _build_grid_index_map(
        radius, n_sub, lambda q: bool(np.linalg.norm(q) < radius + np.sqrt(3.0) * h_cell)
    )
    n = len(centres)
    na = 1 if p == 0 else 4
    nc = 1 if (p == 0 and r == 0) else 20 if r == 2 else 10
    delta = np.array(
        [cell_contrast_coefficients(profile, c, h, contrast, ref, omega, degree=r) for c in centres]
    )
    e_cells = np.array([source_expansion(d)[:nc, :na] for d in delta])  # (N, nc, na, 9, 9)
    blocks = offset_blocks(n_sub, h, omega, ref, 20 if r == 2 else 10) if blocks is None else blocks
    npad = 2 * n_sub - 1
    rows, cols = na * 9, nc * 9
    kh = np.zeros((rows, cols, npad, npad, npad), dtype=complex)
    for off, blk in blocks.items():
        kh[:, :, off[0] % npad, off[1] % npad, off[2] % npad] = (
            blk[:na, :nc].transpose(0, 2, 1, 3).reshape(rows, cols)
        )
    for row in range(rows):
        kh[row] = np.fft.fftn(kh[row], axes=(1, 2, 3))
    g0, g1, g2 = grid_idx[:, 0], grid_idx[:, 1], grid_idx[:, 2]
    m9 = gram_test(h)[:na, :na]

    def matvec(x: NDArray) -> NDArray:
        psi = x.reshape(n, na, 9)
        src = np.einsum("ncbij,nbj->nci", e_cells, psi).reshape(n, cols)
        grid = np.zeros((cols, npad, npad, npad), dtype=complex)
        grid[:, g0, g1, g2] = src.T
        sh = np.fft.fftn(grid, axes=(1, 2, 3)).reshape(cols, -1)
        yh = np.empty((rows, sh.shape[1]), dtype=complex)
        for row in range(rows):
            yh[row] = np.einsum("cf,cf->f", kh[row].reshape(cols, -1), sh)
        y = np.fft.ifftn(yh.reshape(rows, npad, npad, npad), axes=(1, 2, 3))[:, g0, g1, g2].T
        return (np.einsum("ab,nbi->nai", m9, psi).reshape(n, rows) - y).ravel()

    k0 = blocks[(0, 0, 0)][:na, :nc]
    pre = np.array(
        [
            np.linalg.inv(np.kron(m9, np.eye(9)) - np.einsum("acij,cbjk->aibk", k0, en).reshape(rows, rows))
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
    amp = np.concatenate([np.asarray(pol, dtype=complex), _plane_wave_strain_voigt(k_hat, pol, k_mag)])
    rhs = plane_wave_moments(centres, h, k_mag * k_hat, amp)[:, :na].ravel()
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
    full = np.zeros((n, 4, 9), dtype=complex)
    full[:, :na] = sol.reshape(n, na, 9)
    return GradedVoxelResult(centres, grid_idx, h, omega, ref, delta, full, p, r)
