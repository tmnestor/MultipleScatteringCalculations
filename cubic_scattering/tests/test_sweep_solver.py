"""Tests for the directional-sweep Foldy-Lax solver."""

import numpy as np
import pytest

from cubic_scattering.directional_sweeps import apply_g0, build_g0_cache, make_sweep_grid
from cubic_scattering.effective_contrasts import ReferenceMedium
from cubic_scattering.sweep_solver import SweepSolveResult, solve_sweep_foldy_lax

REF = ReferenceMedium(alpha=5.0, beta=3.0, rho=2.5)
OMEGA = 2 * np.pi * (1.0 + 0.03j)
PITCH = 0.25


def _cache(n_z: int = 2, n_x: int = 6):
    grid = make_sweep_grid(n_z, n_x, PITCH, ky=0.6, n_kz=128, n_kx=128)
    return build_g0_cache(grid, REF, OMEGA)


def test_zero_t0_returns_the_incident_field() -> None:
    """With T0 = 0 the system is the identity: psi == psi_inc."""
    cache = _cache()
    n_z, n_x = cache.grid.n_z, cache.grid.n_x
    rng = np.random.default_rng(3)
    psi_inc = rng.standard_normal((n_z, n_x, 9)) + 1j * rng.standard_normal((n_z, n_x, 9))
    t0 = np.zeros((n_z, n_x, 9, 9), dtype=complex)

    res = solve_sweep_foldy_lax(cache, t0, psi_inc)
    assert isinstance(res, SweepSolveResult)
    assert np.abs(res.psi - psi_inc).max() < 1e-12


def test_born_limit_matches_one_forward_application() -> None:
    """At weak contrast, psi - psi_inc == G0 T0 psi_inc to first order.

    A knowable answer that does NOT go through the solver's own machinery: the
    right-hand side is one explicit forward sweep, not a second solve.
    """
    cache = _cache()
    n_z, n_x = cache.grid.n_z, cache.grid.n_x
    rng = np.random.default_rng(5)
    psi_inc = rng.standard_normal((n_z, n_x, 9)) + 0j
    t0 = 1e-7 * (rng.standard_normal((n_z, n_x, 9, 9)) + 0j)

    res = solve_sweep_foldy_lax(cache, t0, psi_inc, tol=1e-12)
    born = apply_g0(np.einsum("zxab,zxb->zxa", t0, psi_inc), cache)
    err = np.abs(res.psi - psi_inc - born).max() / np.abs(born).max()
    assert err < 1e-5  # second order in a 1e-7 contrast


def test_solution_satisfies_its_own_equation() -> None:
    """Residual check: (I - G0 T0) psi == psi_inc for the returned psi."""
    cache = _cache()
    n_z, n_x = cache.grid.n_z, cache.grid.n_x
    rng = np.random.default_rng(17)
    psi_inc = rng.standard_normal((n_z, n_x, 9)) + 0j
    t0 = 1e-3 * (rng.standard_normal((n_z, n_x, 9, 9)) + 0j)

    res = solve_sweep_foldy_lax(cache, t0, psi_inc, tol=1e-10)
    lhs = res.psi - apply_g0(np.einsum("zxab,zxb->zxa", t0, res.psi), cache)
    assert np.abs(lhs - psi_inc).max() / np.abs(psi_inc).max() < 1e-9
    assert res.n_matvec > 0


def test_rejects_mismatched_t0_shape() -> None:
    cache = _cache()
    bad = np.zeros((cache.grid.n_z, cache.grid.n_x, 6, 6), dtype=complex)
    psi_inc = np.zeros((cache.grid.n_z, cache.grid.n_x, 9), dtype=complex)
    with pytest.raises(ValueError, match="t0_blocks"):
        solve_sweep_foldy_lax(cache, bad, psi_inc)


def test_rejects_mismatched_psi_inc_shape() -> None:
    cache = _cache()
    t0 = np.zeros((cache.grid.n_z, cache.grid.n_x, 9, 9), dtype=complex)
    with pytest.raises(ValueError, match="psi_inc"):
        solve_sweep_foldy_lax(cache, t0, np.zeros((cache.grid.n_z, cache.grid.n_x, 3), dtype=complex))


def test_reports_non_convergence_rather_than_returning_silently() -> None:
    """A partially converged field is not a physical answer and is not returned."""
    cache = _cache()
    n_z, n_x = cache.grid.n_z, cache.grid.n_x
    huge = 1e12 * np.ones((n_z, n_x, 9, 9), dtype=complex)
    psi_inc = np.ones((n_z, n_x, 9), dtype=complex)
    with pytest.raises(RuntimeError, match="did not converge"):
        solve_sweep_foldy_lax(cache, huge, psi_inc, max_iter=3)
