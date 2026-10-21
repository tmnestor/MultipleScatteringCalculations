"""The contrast projection through a ``BatchProfile``: every cell's Gauss points in one call of the profile.

A profile given as a function of many points at once is evaluated once for all cells (216 points a cell)
rather than point by point. The projection must equal the per-point one to round-off at every degree,
the batch function must be called once, the wrapper must still answer a single point, and the
physical-range check must fire as before, naming the cell.
"""

import numpy as np
import pytest

from cubic_scattering.effective_contrasts import MaterialContrast, ReferenceMedium
from cubic_scattering.graded_voxel import site

REF = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
CON = MaterialContrast(Dlambda=2.0e9, Dmu=1.0e9, Drho=100.0)
H = 1.25
CENTRES = np.array([[0.0, 0.0, 0.0], [2.5, 0.0, -2.5], [5.0, 2.5, 7.5], [-7.5, 5.0, 2.5]])


def _scalar(x: np.ndarray) -> float:
    return float(np.exp(-0.01 * np.dot(x, x)))


class CountingBatch:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, pts: np.ndarray) -> np.ndarray:
        self.calls += 1
        return np.exp(-0.01 * np.einsum("ni,ni->n", pts, pts))


@pytest.mark.parametrize("degree", [0, 1, 2])
def test_the_batch_projection_equals_the_point_by_point_one(degree: int) -> None:
    batch = CountingBatch()
    got = site.cell_contrast_coefficients_all(site.BatchProfile(batch), CENTRES, H, CON, REF, 300.0, degree)
    want = site.cell_contrast_coefficients_all(_scalar, CENTRES, H, CON, REF, 300.0, degree)
    assert batch.calls == 1
    assert np.abs(got - want).max() <= 1e-14 * np.abs(want).max()


def test_the_wrapper_answers_a_single_point() -> None:
    profile = site.BatchProfile(CountingBatch())
    x = np.array([1.0, -2.0, 0.5])
    assert profile(x) == pytest.approx(_scalar(x), rel=1e-15)
    assert isinstance(profile(x), float)


def test_the_physical_range_check_fires_through_the_batch() -> None:
    strong = site.BatchProfile(lambda pts: np.full(len(pts), 40.0))
    with pytest.raises(ValueError, match=r"cell centre.*below 52% of the background"):
        site.cell_contrast_coefficients_all(strong, CENTRES[:1] + 100.0, H, CON, REF, 300.0, 1)
