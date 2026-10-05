"""The moment hierarchy's cell-to-cell tables by a multipole series about sub-cubes, with no quadrature.

T[i, n, D, W](o) = int_cube (d_D G_in)(o - xi) xi^W d xi for a source cube of side d at the origin and a
field point o at the centre of another cell. The field point is at least half a cell outside the source
cube, so the integrand is smooth and the reference is the tensor Gauss rule of ``derivatives.moment_table``
at a high order; the series must reproduce it to its tolerance.
"""

import numpy as np
import pytest

from cubic_scattering.effective_contrasts import ReferenceMedium
from cubic_scattering.graded_voxel import derivatives as gd

REF = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
SIDE = 2.0
D_LIST = list(gd.multi_indices(4))
W_LIST = list(gd.multi_indices(4))


@pytest.mark.parametrize(
    "offset", [(1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 1, 0), (3, 0, 0), (5, 2, 1), (13, 5, 3)]
)
@pytest.mark.parametrize("ks_side", [0.02, 0.6, 3.0])
def test_series_equals_high_order_gauss(offset, ks_side) -> None:
    omega = ks_side * REF.beta / SIDE
    o = SIDE * np.array(offset, float)
    reference = gd.moment_table(o, SIDE, omega, REF, D_LIST, W_LIST, 28)
    series = gd.moment_table_series(o, SIDE, omega, REF, D_LIST, W_LIST, tol=1e-12)
    scale = np.abs(reference).max(axis=(0, 1), keepdims=True)  # per (D, W) column
    assert np.abs(series - reference).max() <= 1e-10 * np.abs(reference).max()
    # each (D, W) column to 1e-9 of its own size, down to round-off of the whole table: some columns nearly
    # cancel by symmetry at a distant offset and carry only round-off in either method
    assert np.all(np.abs(series - reference) <= 1e-9 * scale + 1e-14 * np.abs(reference).max())


def test_series_refuses_the_self_cell() -> None:
    with pytest.raises(ValueError, match="self cell"):
        gd.moment_table_series(np.zeros(3), SIDE, 30.0, REF, D_LIST, W_LIST, tol=1e-12)


def test_series_truncation_follows_the_tolerance() -> None:
    o = SIDE * np.array([1.0, 1.0, 0.0])
    omega = 0.1 * REF.beta / SIDE
    reference = gd.moment_table(o, SIDE, omega, REF, D_LIST, W_LIST, 28)
    for tol in (1e-4, 1e-8):
        series = gd.moment_table_series(o, SIDE, omega, REF, D_LIST, W_LIST, tol=tol)
        assert np.abs(series - reference).max() <= 10 * tol * np.abs(reference).max()


@pytest.mark.parametrize("offset", [(1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 1, 0)])
@pytest.mark.parametrize("ks_side", [0.02, 0.6, 3.0])
def test_wavenumber_series_equals_high_order_gauss(offset, ks_side) -> None:
    """The series in k with frequency-independent coefficients, for touching and near cells."""
    omega = ks_side * REF.beta / SIDE
    o = SIDE * np.array(offset, float)
    reference = gd.moment_table(o, SIDE, omega, REF, D_LIST, W_LIST, 28)
    series = gd.moment_table_kseries(offset, SIDE, omega, REF, D_LIST, W_LIST, tol=1e-13)
    scale = np.abs(reference).max(axis=(0, 1), keepdims=True)
    assert np.abs(series - reference).max() <= 1e-10 * np.abs(reference).max()
    assert np.all(np.abs(series - reference) <= 1e-9 * scale + 1e-14 * np.abs(reference).max())


def test_wavenumber_series_coefficients_are_reused_across_frequencies_and_sizes() -> None:
    """The coefficients depend on neither the frequency nor the cell size: one computation serves all."""
    gd.kseries_coefficients.cache_clear()
    for side, ks_side in ((2.0, 0.1), (0.5, 0.4), (7.0, 1.0)):
        gd.moment_table_kseries((1, 1, 0), side, ks_side * REF.beta / side, REF, D_LIST, W_LIST, tol=1e-12)
    assert gd.kseries_coefficients.cache_info().misses == 1


def test_wavenumber_series_coefficients_are_accurate_at_every_power() -> None:
    """The Gauss rule of the coefficients must integrate the steep high powers too, not only t <= 19.

    At t = 48 the integrand d^a r^47 xi^W behaves like a polynomial of degree about 51 per axis, beyond the
    reach of the 12 points that were calibrated for distant offsets over t <= 19: against 40 points the
    coefficients at (2, 1, 0) were wrong by 2e-13 of each order's largest entry.
    """
    offset = (2, 1, 0)
    d_list, w_list = tuple(D_LIST), tuple(W_LIST)
    gd.kseries_coefficients.cache_clear()
    production, _ = gd.kseries_coefficients(offset, d_list, w_list)
    saved = gd.kseries_gauss_points
    gd.kseries_coefficients.cache_clear()
    gd.kseries_gauss_points = lambda _o: 40  # type: ignore[assignment]
    try:
        reference, _ = gd.kseries_coefficients(offset, d_list, w_list)
    finally:
        gd.kseries_gauss_points = saved  # type: ignore[assignment]
        gd.kseries_coefficients.cache_clear()
    for t in range(40, gd.KSERIES_T_MAX + 1):
        err = np.abs(production[t] - reference[t]).max() / np.abs(reference[t]).max()
        assert err < 1e-14, (t, err)
