"""The hierarchy's frequency-independent table coefficients in closed form, with no quadrature.

U[t, a, w](o) = int_cube d^a r^(t - 1) (o - xi) xi^W d xi on the unit cube, for an integer offset o != 0.
The field point o is never in the closure of the source cube, so after cutting the cube into its eight
half-cubes each piece is a box in one orthant of the separation, away from the singular point, and its
integral is a contraction with the Legendre moments of ``legendre_moments.box_moments``. The reference is a
tensor Gauss rule of 40 points per axis, which shares no step with that route.
"""

import numpy as np
import pytest

from cubic_scattering.effective_contrasts import ReferenceMedium
from cubic_scattering.graded_voxel import derivatives as gd

# every test builds the closed coefficients of at least one offset: minutes each
pytestmark = pytest.mark.slow

REF = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
D_LIST = tuple(gd.multi_indices(4))
W_LIST = tuple(gd.multi_indices(4))


def gauss_coefficients(offset, n_gauss: int) -> np.ndarray:
    """The Gauss route of ``kseries_coefficients`` at a chosen order."""
    saved = gd.kseries_gauss_points
    gd.kseries_coefficients.cache_clear()
    gd.kseries_gauss_points = lambda _o: n_gauss  # type: ignore[assignment]
    try:
        u, _ = gd.kseries_coefficients(tuple(offset), D_LIST, W_LIST)
    finally:
        gd.kseries_gauss_points = saved  # type: ignore[assignment]
        gd.kseries_coefficients.cache_clear()
    return u


@pytest.mark.parametrize("offset", [(1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 1, 0), (0, -1, 1), (3, 0, 0)])
def test_closed_coefficients_equal_high_order_gauss(offset) -> None:
    reference = gauss_coefficients(offset, 40)
    closed, a_list = gd.kseries_coefficients_closed(tuple(offset), D_LIST, W_LIST)
    assert a_list == gd.multi_indices(max(sum(d) for d in D_LIST) + 2)
    assert closed.shape == reference.shape
    for t in range(closed.shape[0]):
        size = np.abs(reference[t]).max()
        err = np.abs(closed[t] - reference[t]).max() / size
        assert err < 1e-13, (offset, t, err)


def test_closed_coefficients_use_no_approximate_quadrature(monkeypatch) -> None:
    """No tensor Gauss rule over the cell; the only Gauss rules are the exact ones for the polynomial (even
    m) terms, whose order is the degree of the polynomial."""

    def refuse(*args, **kwargs):
        raise AssertionError("the closed route called the approximate tensor rule")

    calls = []
    real = gd.leggauss

    def counted(n):
        calls.append(n)
        return real(n)

    monkeypatch.setattr(gd, "cell_rule", refuse)
    monkeypatch.setattr(gd, "leggauss", counted)
    gd._free_moments.cache_clear()
    gd.kseries_coefficients_closed.cache_clear()
    gd.kseries_coefficients_certified.cache_clear()  # the values come from here: recompute them
    gd.kseries_coefficients_closed((1, 1, 0), D_LIST, W_LIST)
    max_w = max(max(w) for w in W_LIST)
    assert calls and all(n <= (gd.KSERIES_T_MAX - 1 + max_w) // 2 + 1 for n in calls)


def test_closed_coefficients_refuse_the_self_cell() -> None:
    with pytest.raises(ValueError, match="self cell"):
        gd.kseries_coefficients_closed((0, 0, 0), D_LIST, W_LIST)


@pytest.mark.parametrize("offset", [(1, 0, 0), (1, 1, 1), (2, 1, 0)])
@pytest.mark.parametrize("ks_side", [1e-4, 0.6, 3.0])
def test_table_from_closed_coefficients_equals_high_order_gauss(offset, ks_side) -> None:
    side = 2.0
    omega = ks_side * REF.beta / side
    reference = gd.moment_table(
        side * np.array(offset, float), side, omega, REF, list(D_LIST), list(W_LIST), 36
    )
    table = gd.moment_table_kseries(
        offset, side, omega, REF, list(D_LIST), list(W_LIST), tol=1e-13,
        coefficients=gd.kseries_coefficients_closed,
    )  # fmt: skip
    assert np.abs(table - reference).max() <= 1e-12 * np.abs(reference).max()
