"""Tests for the three-dimensional Foldy-Lax solve.

The claim is that GMRES on the matrix-free operator reaches the same answer as
two INDEPENDENT direct methods -- a dense LU of the assembled operator, and an
explicit Neumann series. Three routes agreeing is the evidence standard; GMRES
agreeing with itself is not.

The spectral radius is deliberately pushed near the Neumann convergence limit.
At small contrast every method agrees trivially and the test proves nothing; the
discriminating case is a solve that actually needs the iterations.
"""

import numpy as np
import pytest

from cubic_scattering.directional_sweeps import (
    SweepGrid3D,
    apply_g0_3d,
    build_g0_cache_3d,
)
from cubic_scattering.effective_contrasts import ReferenceMedium
from cubic_scattering.sweep_solver import solve_foldy_lax_3d

REF = ReferenceMedium(5.0, 3.0, 2.5)
OM = 2 * np.pi * 6.0 * (1 + 0.03j)
PITCH = 0.5


def _setup(scale: float):
    grid = SweepGrid3D(n_z=2, n_x=3, n_y=2, pitch=PITCH)
    cache = build_g0_cache_3d(grid, REF, OM)
    rng = np.random.default_rng(20260914)
    n = (grid.n_z, grid.n_x, grid.n_y)
    t0 = scale * (rng.normal(size=(*n, 9, 9)) + 1j * rng.normal(size=(*n, 9, 9)))
    psi = rng.normal(size=(*n, 9)) + 1j * rng.normal(size=(*n, 9))
    return grid, cache, t0, psi


def _dense(grid, cache, t0):
    """Assemble (I - G0 T0) column by column. Independent of the Krylov path."""
    size = grid.n_z * grid.n_x * grid.n_y * 9
    shape = (grid.n_z, grid.n_x, grid.n_y, 9)
    m = np.zeros((size, size), dtype=complex)
    for c in range(size):
        e = np.zeros(size, dtype=complex)
        e[c] = 1.0
        v = e.reshape(shape)
        m[:, c] = (v - apply_g0_3d(np.einsum("zxyab,zxyb->zxya", t0, v), cache)).ravel()
    return m


def test_gmres_matches_dense_lu_and_neumann() -> None:
    """RUNG 5-3D: three routes to the same solve."""
    # Calibrated so rho lands near 0.9: rho is linear in this scale, and at
    # 2e-3 it is 0.005, where every method agrees trivially and the test is
    # vacuous. The assertion below enforces that calibration rather than
    # trusting it.
    grid, cache, t0, psi = _setup(0.33)

    res = solve_foldy_lax_3d(cache, t0, psi, tol=1e-12)

    m = _dense(grid, cache, t0)
    lu = np.linalg.solve(m, psi.ravel()).reshape(psi.shape)

    rho = float(np.max(np.abs(np.linalg.eigvals(np.eye(m.shape[0]) - m))))
    assert 0.5 < rho < 0.98, f"spectral radius {rho:.3f} is not a discriminating test"

    # Explicit Neumann: psi + K psi + K^2 psi + ... with K = G0 T0
    acc = psi.copy()
    term = psi.copy()
    for _ in range(400):
        term = apply_g0_3d(np.einsum("zxyab,zxyb->zxya", t0, term), cache)
        acc = acc + term
    neumann = acc

    e_lu = np.abs(res.psi - lu).max() / np.abs(lu).max()
    e_nm = np.abs(res.psi - neumann).max() / np.abs(lu).max()
    assert e_lu < 1e-10, f"GMRES != dense LU: {e_lu:.3e} at rho={rho:.3f}"
    assert e_nm < 1e-8, f"GMRES != Neumann: {e_nm:.3e} at rho={rho:.3f}"


def test_refuses_to_return_a_partial_answer() -> None:
    """A solve that does not converge must RAISE, never return quietly."""
    grid, cache, t0, psi = _setup(5.0)  # rho ~ 13, far outside any radius
    with pytest.raises(RuntimeError) as exc:
        solve_foldy_lax_3d(cache, t0, psi, tol=1e-14, max_iter=2)
    assert "matvec" in str(exc.value)


def test_rejects_mismatched_shapes() -> None:
    """Fail fast, with the four-element diagnostic the project requires."""
    grid, cache, t0, psi = _setup(1.0e-3)
    with pytest.raises(ValueError) as exc:
        solve_foldy_lax_3d(cache, t0[..., :6, :6], psi)
    msg = str(exc.value)
    assert "Where:" in msg
    assert "Valid:" in msg
    assert "Fix:" in msg
