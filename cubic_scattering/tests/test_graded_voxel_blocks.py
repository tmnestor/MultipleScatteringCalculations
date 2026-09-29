"""Coupling blocks of the graded voxel."""

import numpy as np

from cubic_scattering import ReferenceMedium
from cubic_scattering.graded_voxel.blocks import far_block
from cubic_scattering.graded_voxel.kernel import kernel_9x9

REF = ReferenceMedium(5000.0, 3000.0, 2500.0)
OMEGA = 150.0
H = 1.25  # k_S h = 0.0625


def test_far_block_point_limit():
    off = (12, 5, -3)
    k = far_block(off, H, OMEGA, REF, n_gauss=4)
    R = 2 * H * np.array(off, float)
    point = (2 * H) ** 6 * kernel_9x9(R[None], OMEGA, REF)[0]
    assert np.linalg.norm(k[0, 0] - point) / np.linalg.norm(point) < 5e-3


def test_far_block_converges_in_gauss_order():
    off = (3, 1, 0)
    k6, k8 = far_block(off, H, OMEGA, REF, 6), far_block(off, H, OMEGA, REF, 8)
    assert np.linalg.norm(k6 - k8) / np.linalg.norm(k8) < 1e-9


def test_far_block_odd_moments_vanish_on_axis_by_symmetry():
    # offset along axis 0: reflecting axis 1 flips xi_1 (test a = 2) and leaves the G entries with an even
    # number of axis-1 indices unchanged, so those entries of K[2, 0] vanish
    k = far_block((3, 0, 0), H, OMEGA, REF, 6)
    for i, j in ((0, 0), (1, 1), (2, 2), (0, 2), (2, 0)):
        assert abs(k[2, 0][i, j]) < 1e-12 * np.abs(k[0, 0]).max(), (i, j)
