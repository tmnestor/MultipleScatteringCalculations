"""The Galerkin coupling blocks of non-touching cells, batched: ``coupling_blocks_far`` against
``coupling_block``, one offset at a time.

The same s-form Gauss rule (same pieces, nodes, weights and order per offset), with the offset-independent
weights built once and the kernel evaluated for many offsets in one compiled call. The blocks must equal the
per-offset ones to round-off at every distance band of ``gauss_order`` and at the frequencies of the
thesis's Chapter 7, in any chunking; touching offsets must be refused, as ``gauss_order`` refuses them.
"""

import numpy as np
import pytest

from cubic_scattering.effective_contrasts import ReferenceMedium
from cubic_scattering.graded_voxel import blocks as B

REF = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
H = 10.0 / 32  # Chapter 7: a = 10 m, N = 32
# one offset per distance band of gauss_order (14 points at two cells, 10 to four, 8 beyond), and the
# far corner
OFFSETS = [(2, 1, 1), (4, 4, 0), (12, 7, 3), (24, 24, 24)]


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


@pytest.mark.parametrize("ka", [0.26, 3.0, 8.9])
def test_batched_blocks_equal_the_per_offset_blocks(ka: float) -> None:
    omega = ka * REF.beta / 10.0
    got = B.coupling_blocks_far(np.array(OFFSETS), H, omega, REF)
    for k, off in enumerate(OFFSETS):
        assert _rel(got[k], B.coupling_block(off, H, omega, REF)) < 1e-13, off


def test_complex_frequency_and_chunking() -> None:
    omega = 3.0 * REF.beta / 10.0 * (1 + 0.05j)
    whole = B.coupling_blocks_far(np.array(OFFSETS), H, omega, REF)
    small = B.coupling_blocks_far(np.array(OFFSETS), H, omega, REF, chunk_points=5000)
    np.testing.assert_allclose(small, whole, rtol=0, atol=1e-15 * np.abs(whole).max())
    for k, off in enumerate(OFFSETS):
        assert _rel(whole[k], B.coupling_block(off, H, omega, REF)) < 1e-13, off


def test_touching_offsets_are_refused() -> None:
    with pytest.raises(ValueError, match="touches"):
        B.coupling_blocks_far(np.array([(3, 0, 0), (1, 1, 0)]), H, 300.0, REF)


def test_canonical_blocks_use_gauss_for_touching_cells_beyond_the_stored_moments(monkeypatch) -> None:
    """Beyond k_S h = 1/2 the series would compute its moments (minutes each): touching blocks are Gauss."""
    from cubic_scattering.graded_voxel import fft as F

    def never(*args, **kwargs):
        raise AssertionError("near_block_series was called beyond its stored moments")

    monkeypatch.setattr(F, "near_block_series", never)
    h = 2.5
    omega = 0.8 * REF.beta / h  # k_S h = 0.8, beyond NEAR_STORE_MAX_KH
    got = F.canonical_blocks(2, h, omega, REF)  # the canonical offsets of a 2-cell grid all touch
    assert sorted(got) == [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)]
    np.testing.assert_array_equal(got[(1, 1, 0)], B.coupling_block((1, 1, 0), h, omega, REF))
