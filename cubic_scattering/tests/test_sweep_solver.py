"""Tests for the directional-sweep Foldy-Lax solver."""

import numpy as np
import pytest

from cubic_scattering.directional_sweeps import (
    LayeredBackground,
    apply_g0,
    build_g0_cache,
    make_sweep_grid,
)
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


def test_solves_on_a_stratified_background() -> None:
    """End to end with the production vertical operator (thesis Ch.5 Q^d).

    The layered solve must differ from the whole-space one -- otherwise the
    background is being ignored -- while still satisfying its own equation.
    """
    import sys

    sibling = "/Users/tod/Desktop/SeismicInversion"
    if sibling not in sys.path:
        sys.path.insert(0, sibling)
    pytest.importorskip("GlobalMatrix.layered_greens")
    lm = pytest.importorskip("Kennett_Reflectivity.layer_model")

    n_lay, pitch, q = 16, 1.0, 2.0
    al, be, rh = 4.0, 2.22, 2.6
    fast = [1.5, *([al] * n_lay), al]
    fast_b = [0.0, *([be] * n_lay), be]
    fast_r = [1.03, *([rh] * n_lay), rh]
    for lay in (9, 10):
        fast[lay], fast_b[lay], fast_r[lay] = 6.5, 3.7, 3.3
    model = lm.LayerModel.from_arrays(
        alpha=fast,
        beta=fast_b,
        rho=fast_r,
        thickness=[3.0, *([pitch] * n_lay), np.inf],
        Q_alpha=[q] * (n_lay + 2),
        Q_beta=[1e10, *([q] * n_lay), q],
    )

    grid = make_sweep_grid(2, 4, pitch, ky=0.3, n_kz=32, n_kx=64, kx_max=3.0)
    omega = 2 * np.pi * 6.0
    s_p, s_s = model.complex_slowness_p(), model.complex_slowness_s()
    ref = ReferenceMedium(1.0 / s_p[1], 1.0 / s_s[1], model.rho[1])

    background = LayeredBackground(model=model, plane_ifaces=(7, 11))
    cache_l = build_g0_cache(grid, ref, omega, background=background)
    cache_w = build_g0_cache(grid, ref, omega)

    rng = np.random.default_rng(31)
    psi_inc = rng.standard_normal((2, 4, 9)) + 0j
    t0 = 1e-3 * (rng.standard_normal((2, 4, 9, 9)) + 0j)

    res_l = solve_sweep_foldy_lax(cache_l, t0, psi_inc, tol=1e-10)
    res_w = solve_sweep_foldy_lax(cache_w, t0, psi_inc, tol=1e-10)

    # It satisfies its own equation...
    lhs = res_l.psi - apply_g0(np.einsum("zxab,zxb->zxa", t0, res_l.psi), cache_l)
    assert np.abs(lhs - psi_inc).max() / np.abs(psi_inc).max() < 1e-9
    # ...and the layering actually changed the answer.
    assert np.abs(res_l.psi - res_w.psi).max() / np.abs(res_w.psi).max() > 1e-6


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
