#!/usr/bin/env python3
"""GMRES Foldy-Lax solve around the directional-sweep G0.

Solves (I - G0 T0) psi = psi_inc for the exciting field, then T = T0 psi. T0 is
block-diagonal and local -- the cube T-matrix with its self-term already closed
-- so all the cost is one apply_g0 per iteration. Every order of multiple
scattering is built by the Krylov iterations; the propagator itself stays a
pure forward summation.

Coordinates: z = axis 0 (down), x = axis 1 (right), y = axis 2 (out).
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator, gmres

from .directional_sweeps import G0Cache, G0Cache3D, apply_g0, apply_g0_3d


@dataclass(frozen=True)
class SweepSolveResult:
    """Outcome of the Foldy-Lax solve.

    Attributes:
        psi: Exciting field, shape (n_z, n_x, 9).
        n_matvec: Number of apply_g0 calls -- the honest cost measure. GMRES
            restarts make this larger than the outer iteration count.
        residual: Final relative residual ||(I - G0 T0) psi - psi_inc|| / ||psi_inc||.
    """

    psi: NDArray
    n_matvec: int
    residual: float


def solve_sweep_foldy_lax(
    cache: G0Cache,
    t0_blocks: NDArray,
    psi_inc: NDArray,
    *,
    tol: float = 1e-8,
    max_iter: int = 500,
) -> SweepSolveResult:
    """Solve (I - G0 T0) psi = psi_inc by GMRES.

    Args:
        cache: Precomputed sweep kernels from build_g0_cache.
        t0_blocks: Local T-matrices, shape (n_z, n_x, 9, 9).
        psi_inc: Incident field, shape (n_z, n_x, 9).
        tol: Relative residual tolerance.
        max_iter: Iteration cap.

    Returns:
        A SweepSolveResult.

    Raises:
        ValueError: on a shape mismatch.
        RuntimeError: if GMRES does not converge -- never a silent partial answer.
    """
    n_z, n_x = cache.grid.n_z, cache.grid.n_x
    shape = (n_z, n_x, 9)
    if t0_blocks.shape != (n_z, n_x, 9, 9):
        msg = (
            f"t0_blocks has shape {t0_blocks.shape}, expected {(n_z, n_x, 9, 9)}.\n"
            "  Where: cubic_scattering/sweep_solver.py, solve_sweep_foldy_lax(t0_blocks=...)\n"
            "  Valid: one 9x9 block per site, e.g. np.zeros((n_z, n_x, 9, 9), complex)\n"
            "  Fix:   the 6x6 Voigt T-matrix must be embedded in the 9-component state\n"
            "         first -- see voigt_tmatrix.voigt_tmatrix_from_result."
        )
        raise ValueError(msg) from None
    if psi_inc.shape != shape:
        msg = (
            f"psi_inc has shape {psi_inc.shape}, expected {shape}.\n"
            "  Where: cubic_scattering/sweep_solver.py, solve_sweep_foldy_lax(psi_inc=...)\n"
            "  Valid: a complex array of shape (n_z, n_x, 9)\n"
            "  Fix:   evaluate the incident field on the same grid as the cache."
        )
        raise ValueError(msg) from None

    size = n_z * n_x * 9
    count = {"n": 0}

    def matvec(v: NDArray) -> NDArray:
        count["n"] += 1
        psi = v.reshape(shape)
        return (psi - apply_g0(np.einsum("zxab,zxb->zxa", t0_blocks, psi), cache)).ravel()

    op = LinearOperator((size, size), matvec=matvec, dtype=complex)
    b = psi_inc.ravel()

    with np.errstate(over="ignore", invalid="ignore"):
        sol, info = gmres(op, b, rtol=tol, maxiter=max_iter)
        norm_b = float(np.linalg.norm(b))
        residual = float(np.linalg.norm(op.matvec(sol) - b) / norm_b) if norm_b > 0 else 0.0

    # Written as "not (residual <= ...)" so that a NaN residual -- which an
    # unstable operator produces -- is treated as failure rather than silently
    # passing a ">" comparison.
    if info != 0 or not (residual <= max(tol, 1e-6)):
        msg = (
            f"GMRES did not converge: info={info}, relative residual={residual:.3e} "
            f"after {count['n']} matvecs.\n"
            "  Where: cubic_scattering/sweep_solver.py, solve_sweep_foldy_lax\n"
            f"  Valid: a residual below tol={tol:g}\n"
            "  Fix:   reduce the contrast, refine the pitch, or raise max_iter. A\n"
            "         partially converged field is not a physical answer and is not\n"
            "         returned."
        )
        raise RuntimeError(msg) from None

    return SweepSolveResult(psi=sol.reshape(shape), n_matvec=count["n"], residual=residual)


def solve_foldy_lax_3d(
    cache: G0Cache3D,
    t0_blocks: NDArray,
    psi_inc: NDArray,
    *,
    tol: float = 1e-8,
    max_iter: int = 500,
) -> SweepSolveResult:
    """Solve (I - G0 T0) psi = psi_inc by GMRES, three-dimensionally.

    The 3-D counterpart of ``solve_sweep_foldy_lax``. The only difference is the
    state shape and which G0 is called: the multiple scattering still lives
    entirely in the Krylov iterations, and the propagator remains a pure forward
    summation.

    Args:
        cache: Precomputed real-space tables from build_g0_cache_3d.
        t0_blocks: Local T-matrices, shape (n_z, n_x, n_y, 9, 9).
        psi_inc: Incident field, shape (n_z, n_x, n_y, 9).
        tol: Relative residual tolerance.
        max_iter: Iteration cap.

    Returns:
        A SweepSolveResult whose ``psi`` has shape (n_z, n_x, n_y, 9).

    Raises:
        ValueError: on a shape mismatch.
        RuntimeError: if GMRES does not converge -- never a silent partial answer.
    """
    g = cache.grid
    shape = (g.n_z, g.n_x, g.n_y, 9)
    if t0_blocks.shape != (*shape, 9):
        msg = (
            f"t0_blocks has shape {t0_blocks.shape}, expected {(*shape, 9)}.\n"
            "  Where: cubic_scattering/sweep_solver.py, solve_foldy_lax_3d(t0_blocks=...)\n"
            f"  Valid: one 9x9 block per site, e.g. np.zeros({(*shape, 9)}, complex)\n"
            "  Fix:   the 6x6 Voigt T-matrix must be embedded in the 9-component state\n"
            "         first -- see voigt_tmatrix.voigt_tmatrix_from_result."
        )
        raise ValueError(msg) from None
    if psi_inc.shape != shape:
        msg = (
            f"psi_inc has shape {psi_inc.shape}, expected {shape}.\n"
            "  Where: cubic_scattering/sweep_solver.py, solve_foldy_lax_3d(psi_inc=...)\n"
            f"  Valid: a complex array of shape (n_z, n_x, n_y, 9) = {shape}\n"
            "  Fix:   evaluate the incident field on the same grid as the cache."
        )
        raise ValueError(msg) from None

    size = int(np.prod(shape))
    count = {"n": 0}

    def matvec(v: NDArray) -> NDArray:
        count["n"] += 1
        psi = v.reshape(shape)
        return (psi - apply_g0_3d(np.einsum("zxyab,zxyb->zxya", t0_blocks, psi), cache)).ravel()

    op = LinearOperator((size, size), matvec=matvec, dtype=complex)
    b = psi_inc.ravel()

    with np.errstate(over="ignore", invalid="ignore"):
        sol, info = gmres(op, b, rtol=tol, maxiter=max_iter)
        norm_b = float(np.linalg.norm(b))
        residual = float(np.linalg.norm(op.matvec(sol) - b) / norm_b) if norm_b > 0 else 0.0

    # "not (residual <= ...)" so a NaN residual counts as failure rather than
    # silently passing a ">" comparison.
    if info != 0 or not (residual <= max(tol, 1e-6)):
        msg = (
            f"GMRES did not converge: info={info}, relative residual={residual:.3e} "
            f"after {count['n']} matvecs.\n"
            "  Where: cubic_scattering/sweep_solver.py, solve_foldy_lax_3d\n"
            f"  Valid: a residual below tol={tol:g}\n"
            "  Fix:   reduce the contrast, refine the pitch, or raise max_iter. A\n"
            "         partially converged field is not a physical answer and is not\n"
            "         returned."
        )
        raise RuntimeError(msg) from None

    return SweepSolveResult(psi=sol.reshape(shape), n_matvec=count["n"], residual=residual)
