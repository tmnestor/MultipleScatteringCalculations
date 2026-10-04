"""The Galerkin far blocks as a series in the wavenumber: coefficients independent of frequency and size."""

import numpy as np
import pytest

from cubic_scattering.effective_contrasts import ReferenceMedium
from cubic_scattering.graded_voxel import blocks

REF = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


@pytest.mark.parametrize("off", [(2, 0, 0), (2, 1, -1), (3, 1, 0), (5, 4, 2)])
@pytest.mark.parametrize("ks_h", [1e-6, 0.05, 0.25, 0.5])
def test_far_series_equals_the_gauss_block(off, ks_h):
    # quadrature is linear in the kernel: the series of Gauss coefficients is the Gauss block, truncated
    h = 1.25
    omega = ks_h * REF.beta / h
    series = blocks.far_block_series(off, h, omega, REF)
    gauss = blocks.coupling_block(off, h, omega, REF)
    assert _rel(series, gauss) < 1e-12, (off, ks_h)


def test_far_series_with_a_quadratic_field():
    h, omega = 1.25, 0.3 * REF.beta / 1.25
    series = blocks.far_block_series((2, 1, 0), h, omega, REF, n_source=35, n_test=10)
    gauss = blocks.coupling_block((2, 1, 0), h, omega, REF, n_source=35, n_test=10)
    assert _rel(series, gauss) < 1e-12


def test_far_series_imaginary_part_follows_the_low_frequency_law():
    # Im K / k tends to its limit with an O(k^2) correction: the step 1e-5 -> 1e-4 is 100x the step
    # 1e-6 -> 1e-5 (a truncation two powers early would leave the first step at O(1))
    h, off = 1.25, (3, 1, 0)
    vals = [blocks.far_block_series(off, h, k * REF.beta / h, REF).imag / k for k in (1e-6, 1e-5, 1e-4)]
    step_small = np.linalg.norm(vals[0] - vals[1])
    step_large = np.linalg.norm(vals[1] - vals[2])
    assert 80.0 < step_large / step_small < 120.0


def test_far_series_coefficients_serve_every_frequency_and_cell_size():
    blocks.far_series_coefficients.cache_clear()
    off = (2, 2, 1)
    for h in (0.5, 1.25, 4.0):
        for ks_h in (0.01, 0.2):
            omega = ks_h * REF.beta / h
            assert (
                _rel(blocks.far_block_series(off, h, omega, REF), blocks.coupling_block(off, h, omega, REF))
                < 1e-12
            )
    assert blocks.far_series_coefficients.cache_info().misses == 1


def test_far_series_rejects_touching_offsets():
    with pytest.raises(ValueError, match="touches"):
        blocks.far_block_series((1, 1, 0), 1.25, 100.0, REF)


def test_far_series_refuses_a_frequency_beyond_its_terms():
    with pytest.raises(ValueError, match="coupling_block"):
        blocks.far_block_series((9, 9, 9), 1.25, 3.0 * REF.beta / 1.25, REF)
