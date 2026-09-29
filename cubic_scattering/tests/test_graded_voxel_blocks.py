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


from cubic_scattering.graded_voxel.blocks import (  # noqa: E402
    coupling_block,
    near_block,
    static_term_integral,
)

W9 = np.diag([1.0, 1, 1, 1, 1, 1, 0.5, 0.5, 0.5])
PI9 = np.diag([-1.0, -1, -1, 1, 1, 1, 1, 1, 1])


def test_eshelby_sum_rule():
    # sum_p of the Galerkin double integral of d_p d_p (1/r) over the self cell = -4 pi V
    tot = sum(static_term_integral(-1, (p, p), (0, 0, 0), H, 12)[0, 0] for p in range(3))
    assert abs(tot / (-4 * np.pi * (2 * H) ** 3) - 1) < 1e-11


def test_coulomb_self_energy():
    # int int_{V x V} 1/|x - x'| = C (2h)^5,
    # C = 2[(1 + sqrt2 - 2 sqrt3)/5 - pi/3 + ln((1 + sqrt2)(2 + sqrt3))]
    # (the mean inverse distance in the unit cube; Mathematica re-derives it)
    c = 2 * (
        (1 + np.sqrt(2) - 2 * np.sqrt(3)) / 5 - np.pi / 3 + np.log((1 + np.sqrt(2)) * (2 + np.sqrt(3)))
    )
    got = static_term_integral(-1, (), (0, 0, 0), H, 12)[0, 0]
    assert abs(got / (c * (2 * H) ** 5) - 1) < 1e-11


def test_sform_equals_gauss_beyond_contact():
    # the s-form machinery (moved derivatives, deltas, pieces) on an offset where Gauss is also valid
    for off in ((2, 0, 0), (2, 1, 1), (2, -2, 1)):
        a = near_block(off, H, OMEGA, REF, n_q=14)
        b = far_block(off, H, OMEGA, REF, 10)
        assert np.linalg.norm(a - b) / np.linalg.norm(b) < 1e-9, off


def test_near_block_converges():
    for off in ((0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)):
        a, b = near_block(off, H, OMEGA, REF, 12), near_block(off, H, OMEGA, REF, 16)
        assert np.linalg.norm(a - b) / np.linalg.norm(b) < 1e-11, off


def test_block_reciprocity():
    # W P(r) symmetric at every r  =>  W K_ab(R) symmetric for test functions a, b (source c = b)
    for off in ((0, 0, 0), (1, 0, 0), (1, 1, 1)):
        k = near_block(off, H, OMEGA, REF, 12)
        for a in range(4):
            for b in range(4):
                wk = W9 @ k[a, b]
                assert np.linalg.norm(wk - wk.T) / np.linalg.norm(k[0, 0]) < 1e-11, (off, a, b)


def test_self_block_parity():
    # P(-r) = Pi P(r) Pi. Swapping the names of the two points of the self cell:
    # K_ba(0) = int int L_b(u) P(u - u') L_a(u') = int int L_a(u) P(u' - u) L_b(u') = Pi K_ab(0) Pi
    k = near_block((0, 0, 0), H, OMEGA, REF, 12)
    for a in range(4):
        for b in range(4):
            want = PI9 @ k[a, b] @ PI9
            assert np.linalg.norm(k[b, a] - want) / np.linalg.norm(k[0, 0]) < 1e-11


def test_coupling_block_beyond_contact_matches_the_6d_reference():
    # Review Focus 3, and the s-form far-block ruling: the one-cell-gap and a distant offset
    for off in ((2, 0, 0), (2, 2, 2), (5, -1, 3)):
        a = coupling_block(off, H, OMEGA, REF)
        b = far_block(off, H, OMEGA, REF, 10 if max(map(abs, off)) == 2 else 6)
        assert np.linalg.norm(a - b) / np.linalg.norm(b) < 1e-9, off


def test_static_term_table_reproduces_the_static_kernel():
    # the (m, idx) decomposition that near_block integrates term by term IS the Kelvin 9x9 propagator
    from cubic_scattering.graded_voxel.blocks import static_term_table
    from cubic_scattering.graded_voxel.kernel import power_F, radial_component

    rng = np.random.default_rng(7)
    X = rng.normal(size=(20, 3)) * 2.0
    r = np.linalg.norm(X, axis=1)
    got = np.zeros((20, 9, 9), dtype=complex)
    for (m, idx), coef in static_term_table(REF.alpha, REF.beta, REF.rho).items():
        got += radial_component(power_F(m, r), X, idx)[:, None, None] * coef[None]
    want = kernel_9x9(X, OMEGA, REF, dynamic=False)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12 * np.abs(want).max())
