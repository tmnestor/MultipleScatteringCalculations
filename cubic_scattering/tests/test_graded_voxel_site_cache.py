"""The contrast projection once per sweep: ``cell_contrast_coefficients_all`` against the per-cell function.

Only the contrast operator depends on the frequency, so the projection of the profile is computed on the
first call and kept. The batched result must equal the per-cell one exactly at every degree, a further
frequency must evaluate the profile not at all and still be exact, and the physical-range check must fire.
"""

import numpy as np
import pytest

from cubic_scattering.effective_contrasts import MaterialContrast, ReferenceMedium
from cubic_scattering.graded_voxel import site

REF = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
CON = MaterialContrast(Dlambda=2.0e9, Dmu=1.0e9, Drho=100.0)
H = 1.25
CENTRES = np.array([[0.0, 0.0, 0.0], [2.5, 0.0, -2.5], [5.0, 2.5, 7.5], [-7.5, 5.0, 2.5]])


class Counting:
    """A smooth profile that counts its evaluations."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, x: np.ndarray) -> float:
        self.calls += 1
        return float(np.exp(-0.01 * np.dot(x, x)))


@pytest.mark.parametrize("degree", [0, 1, 2])
def test_batched_equals_per_cell_exactly(degree: int) -> None:
    profile = Counting()
    got = site.cell_contrast_coefficients_all(profile, CENTRES, H, CON, REF, 300.0, degree)
    want = np.array(
        [site.cell_contrast_coefficients(profile, c, H, CON, REF, 300.0, degree) for c in CENTRES]
    )
    np.testing.assert_array_equal(got, want)


def test_a_further_frequency_reuses_the_projection() -> None:
    profile = Counting()
    site.cell_contrast_coefficients_all(profile, CENTRES, H, CON, REF, 300.0, 1)
    first = profile.calls
    assert first == len(CENTRES) * 216
    again = site.cell_contrast_coefficients_all(profile, CENTRES, H, CON, REF, 900.0, 1)
    assert profile.calls == first  # not evaluated again
    want = np.array([site.cell_contrast_coefficients(profile, c, H, CON, REF, 900.0, 1) for c in CENTRES])
    np.testing.assert_array_equal(again, want)


def test_another_profile_or_grid_is_not_served_the_old_projection() -> None:
    a, b = Counting(), Counting()
    site.cell_contrast_coefficients_all(a, CENTRES, H, CON, REF, 300.0, 1)
    site.cell_contrast_coefficients_all(b, CENTRES, H, CON, REF, 300.0, 1)
    assert b.calls == len(CENTRES) * 216
    site.cell_contrast_coefficients_all(a, CENTRES[:2], H, CON, REF, 300.0, 1)
    assert a.calls == (len(CENTRES) + 2) * 216


def test_the_physical_range_check_still_fires() -> None:
    strong = MaterialContrast(Dlambda=2.0e10, Dmu=2.0e10, Drho=100.0)
    with pytest.raises(ValueError, match="allowed range is below 52%"):
        site.cell_contrast_coefficients_all(Counting(), CENTRES, H, strong, REF, 300.0, 1)
