"""The sphere solver's condition number by LAPACK's estimate, on the factorisation the solve already makes.

The exact 2-norm condition number (np.linalg.cond, a full SVD) cost about 22 times the solve. LAPACK's
zgecon estimates the 1-norm condition number from the LU factors in O(N^2). Its estimate never exceeds
the exact kappa_1 and is in practice within a small factor of it; kappa_1 and kappa_2 differ by at most a
factor N, and for these Foldy-Lax matrices by far less. The solution itself must not change.
"""

import numpy as np
import pytest

from cubic_scattering.effective_contrasts import MaterialContrast, ReferenceMedium
from cubic_scattering.resonance_tmatrix import _solve_with_condition_estimate, _times_block_diagonal
from cubic_scattering.sphere_scattering import compute_sphere_foldy_lax

#: The exact 2-norm condition number (SVD) the solver returned for the sphere case below, before the change.
KAPPA2_N4 = 1.2126129119408393


def _foldy_lax_like(n: int, scale: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.eye(n) + scale * (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / np.sqrt(n)


@pytest.mark.parametrize("scale", [0.1, 0.5, 0.9])
def test_solution_unchanged_and_estimate_bounds_exact_kappa1(scale: float) -> None:
    a = _foldy_lax_like(300, scale, 7)
    rhs = np.random.default_rng(1).normal(size=(300, 4)) + 0j
    sol, cond = _solve_with_condition_estimate(a.copy(), rhs)
    np.testing.assert_allclose(sol, np.linalg.solve(a, rhs), rtol=1e-12, atol=1e-12)
    exact = np.linalg.cond(a, 1)
    assert exact / 3.0 <= cond <= exact * (1 + 1e-12), f"estimate {cond:.4g} vs exact kappa_1 {exact:.4g}"


def test_singular_matrix_is_reported_not_hidden() -> None:
    a = np.ones((5, 5), dtype=complex)
    with pytest.raises(np.linalg.LinAlgError):
        _solve_with_condition_estimate(a, np.ones((5, 1), dtype=complex))


def test_sphere_result_and_estimate() -> None:
    ref = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
    con = MaterialContrast(Dlambda=2e9, Dmu=1e9, Drho=100.0)
    res = compute_sphere_foldy_lax(2 * np.pi * 10.0, 10.0, ref, con, 4, cell_average=False)
    # The previous value, the exact 2-norm condition number, for this configuration (n_sub = 4): the
    # 1-norm estimate must be of the same size.
    assert np.isfinite(res.condition_number) and res.condition_number >= 1.0
    assert 0.2 < res.condition_number / KAPPA2_N4 < 5.0


def test_block_diagonal_product_equals_the_dense_one() -> None:
    rng = np.random.default_rng(5)
    n = 40
    p = rng.normal(size=(9 * n, 9 * n)) + 1j * rng.normal(size=(9 * n, 9 * n))
    t = rng.normal(size=(9, 9)) + 1j * rng.normal(size=(9, 9))
    dense = p @ np.kron(np.eye(n), t)
    np.testing.assert_allclose(
        _times_block_diagonal(p, t), dense, rtol=1e-14, atol=1e-14 * np.max(np.abs(dense))
    )
