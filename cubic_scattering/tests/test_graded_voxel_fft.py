"""The graded voxel's FFT solver: symmetry-generated blocks and the FFT matrix-vector product."""

import numpy as np

from cubic_scattering import MaterialContrast, ReferenceMedium
from cubic_scattering.graded_voxel.blocks import coupling_block
from cubic_scattering.graded_voxel.fft import (
    offset_blocks,
    signed_permutations,
    solve_graded_sphere_fft,
    symmetry_reps,
)
from cubic_scattering.graded_voxel.kernel import kernel_9x9
from cubic_scattering.graded_voxel.solver import solve_graded_sphere

REF = ReferenceMedium(5000.0, 3000.0, 2500.0)
CON = MaterialContrast(Dlambda=2.0e9, Dmu=1.0e9, Drho=100.0)
KHAT = np.array([1.0, 0.0, 0.0])
OMEGA, H = 150.0, 1.25


def _profile(pos):
    # a smooth graded profile that stays physical: 1 in the middle, falling to ~0.4 at the corners
    return 1.0 - 0.004 * float(pos @ pos)


def test_there_are_48_signed_permutations():
    qs = signed_permutations()
    assert len(qs) == 48
    for q in qs:
        np.testing.assert_array_equal(q @ q.T, np.eye(3))


def test_kernel_is_equivariant_under_the_cube_group():
    # the background is isotropic: P(Q r) = S P(r) S^T with S = diag(Q, R6(Q))
    rng = np.random.default_rng(0)
    r = rng.normal(size=(5, 3)) * 4.0
    p = kernel_9x9(r, OMEGA, REF)
    for q in signed_permutations():
        _, _, s = symmetry_reps(q)
        pq = kernel_9x9(r @ q.T, OMEGA, REF)
        for k in range(5):
            np.testing.assert_allclose(pq[k], s @ p[k] @ s.T, atol=1e-12 * np.abs(p[k]).max())


def test_symmetry_generated_blocks_equal_direct_ones():
    blocks = offset_blocks(4, H, OMEGA, REF)
    for off in ((1, 0, -1), (-1, 1, 1), (2, -1, 0), (-3, 2, 1), (0, -2, 3)):
        direct = coupling_block(off, H, OMEGA, REF)
        assert np.linalg.norm(blocks[off] - direct) / np.linalg.norm(direct) < 1e-11, off


def test_fft_solve_equals_the_dense_solve():
    for p in (0, 1):
        dense = solve_graded_sphere(OMEGA, 10.0, REF, CON, 4, _profile, KHAT, KHAT, "P", p=p, r=p)
        fft = solve_graded_sphere_fft(
            OMEGA, 10.0, REF, CON, 4, _profile, KHAT, KHAT, "P", p=p, r=p, gmres_tol=1e-13
        )
        np.testing.assert_array_equal(fft.centres, dense.centres)
        rel = np.linalg.norm(fft.psi - dense.psi) / np.linalg.norm(dense.psi)
        assert rel < 1e-10, (p, rel)
