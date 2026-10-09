"""Radiation of the graded voxel's sources."""

import numpy as np

from cubic_scattering import ReferenceMedium
from cubic_scattering.graded_voxel.farfield import graded_far_field, graded_field, radiate, radiate_exact
from cubic_scattering.graded_voxel.solver import GradedVoxelResult
from cubic_scattering.resonance_tmatrix import elastodynamic_greens_deriv
from cubic_scattering.sphere_scattering import _voigt_to_tensor  # halves the engineering shear entries

REF = ReferenceMedium(5000.0, 3000.0, 2500.0)
OMEGA = 150.0


def _exact(src, x_obs):
    # u_i = G_ij F_j + d_k G_ij sigma_jk (source derivative d' = -d), the Green's tensor evaluated exactly;
    # the source's shear entries are engineering stress 2 sigma_pq
    g, gd, _ = elastodynamic_greens_deriv(x_obs, OMEGA, REF)
    return g @ src[:3] + np.einsum("ijk,jk->i", gd, _voigt_to_tensor(src[3:]))


def test_point_force_and_dipole_far_field():
    rng = np.random.default_rng(0)
    src = rng.normal(size=9) + 1j * rng.normal(size=9)
    dirs = np.array([[1.0, 0, 0], [0, 0.6, 0.8], [-0.36, 0.48, 0.8]])
    rdist = 2e6
    up, us = radiate(np.zeros((1, 3)), src[None], OMEGA, REF, dirs, rdist)
    for d, u in zip(dirs, up + us, strict=True):
        want = _exact(src, d * rdist)
        assert np.abs(u - want).max() / np.abs(want).max() < 1e-4


def test_radiation_matches_the_kernel_for_every_source_component():
    # the far field of a unit source must be the kernel's own field there: the solve uses the kernel, whose
    # shear columns take ENGINEERING stress (2 sigma_pq: the contrast operator maps gamma to 2 dmu gamma).
    # Radiating the shear entries as tensor sigma_pq doubled them (found 2026-09-30: a full-contrast floor)
    from cubic_scattering.graded_voxel.kernel import kernel_9x9

    d = np.array([0.36, 0.48, 0.8])
    d /= np.linalg.norm(d)
    rdist = 4e6
    p = kernel_9x9((d * rdist)[None], OMEGA, REF)[0]
    for k in range(9):
        s = np.zeros(9, complex)
        s[k] = 1.0
        up, us = radiate(np.zeros((1, 3)), s[None], OMEGA, REF, d[None], rdist)
        rad, ker = (up + us)[0], p[:3] @ s
        assert np.abs(rad - ker).max() / np.abs(ker).max() < 1e-4, k


def test_exact_radiation_of_a_point_source_at_finite_distance():
    # the finite-distance readout is the Green tensor itself: displacement rows against the independent
    # point formula, close to the source and far from it
    rng = np.random.default_rng(1)
    src = rng.normal(size=9) + 1j * rng.normal(size=9)
    x0 = np.array([0.3, -0.2, 0.5])
    obs = np.array([[3.0, 1.0, -2.0], [40.0, -25.0, 10.0], [0.9, -0.2, 0.5]])
    u, _ = radiate_exact(x0[None], src[None], OMEGA, REF, obs)
    for x, got in zip(obs, u, strict=True):
        want = _exact(src, x - x0)
        assert np.abs(got - want).max() / np.abs(want).max() < 1e-10


def test_exact_radiation_strain_is_the_gradient_of_its_displacement():
    # the strain rows are checked against a central difference of the displacement rows
    rng = np.random.default_rng(2)
    src = rng.normal(size=9) + 1j * rng.normal(size=9)
    x, eps = np.array([4.0, -3.0, 2.0]), 1e-4
    _, strain = radiate_exact(np.zeros((1, 3)), src[None], OMEGA, REF, x[None])
    grad = np.zeros((3, 3), complex)
    for k in range(3):
        dx = np.zeros(3)
        dx[k] = eps
        up, _ = radiate_exact(np.zeros((1, 3)), src[None], OMEGA, REF, (x + dx)[None])
        um, _ = radiate_exact(np.zeros((1, 3)), src[None], OMEGA, REF, (x - dx)[None])
        grad[:, k] = (up[0] - um[0]) / (2 * eps)
    want = np.array(
        [
            grad[0, 0],
            grad[1, 1],
            grad[2, 2],
            grad[1, 2] + grad[2, 1],
            grad[0, 2] + grad[2, 0],
            grad[0, 1] + grad[1, 0],
        ]
    )
    assert np.abs(strain[0] - want).max() / np.abs(want).max() < 1e-6


def test_exact_radiation_tends_to_the_far_field():
    rng = np.random.default_rng(3)
    pts = rng.normal(size=(5, 3))
    srcs = rng.normal(size=(5, 9)) + 1j * rng.normal(size=(5, 9))
    dirs = np.array([[1.0, 0, 0], [0, 0.6, 0.8], [-0.36, 0.48, 0.8]])
    rdist = 2e6
    up, us = radiate(pts, srcs, OMEGA, REF, dirs, rdist)
    u, _ = radiate_exact(pts, srcs, OMEGA, REF, dirs * rdist)
    assert np.abs(u - (up + us)).max() / np.abs(u).max() < 1e-4


def test_graded_field_tends_to_the_graded_far_field():
    # two cells with arbitrary linear contrast and state: the cell readout at a great distance is the
    # far-field readout of the same cells
    rng = np.random.default_rng(4)
    res = GradedVoxelResult(
        centres=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        grid_idx=np.array([[0, 0, 0], [1, 0, 0]]),
        h=0.5,
        omega=OMEGA,
        ref=REF,
        delta=rng.normal(size=(2, 4, 9, 9)) * 1e9,
        psi=rng.normal(size=(2, 4, 9)) + 1j * rng.normal(size=(2, 4, 9)),
        p=1,
        r=1,
    )
    dirs = np.array([[0.0, 0.6, 0.8], [1.0, 0.0, 0.0]])
    rdist = 2e6
    khat = np.array([1.0, 0.0, 0.0])
    up, us = graded_far_field(res, dirs, rdist, khat, khat, "P")
    u, _ = graded_field(res, dirs * rdist)
    assert np.abs(u - (up + us)).max() / np.abs(u).max() < 1e-4


def test_monomial_fourier_closed_form():
    # int xi^n exp(i q xi) over [-1, 1] through the spherical Bessel functions, for real and complex q,
    # including q -> 0 where the elementary forms cancel
    from numpy.polynomial.legendre import leggauss

    from cubic_scattering.graded_voxel.farfield import monomial_fourier

    x, w = leggauss(40)
    for n in range(5):
        for q in (1e-5, 0.4, 2.0, 0.3 + 0.5j, -0.7j):
            want = np.sum(w * x**n * np.exp(1j * q * x))
            # the Gauss sum cancels for odd n as q -> 0, so the tolerance is set by the integrand's size
            scale = np.sum(w * np.abs(x**n * np.exp(1j * q * x)))
            assert abs(monomial_fourier(n, q) - want) <= 5e-14 * scale


def test_source_moments_match_the_node_sources_at_complex_wave_vectors():
    # the closed-form moments of the cells' polynomial sources equal those of a converged Gauss rule, for a
    # real wave vector and for a complex one (an evanescent direction), at degrees one and two
    from cubic_scattering.graded_voxel.farfield import _node_sources, source_moments

    rng = np.random.default_rng(3)
    for nf, nd in ((4, 4), (10, 10)):
        res = GradedVoxelResult(
            centres=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 1.0]]),
            grid_idx=np.array([[0, 0, 0], [1, 0, 0], [0, -1, 1]]),
            h=0.5,
            omega=OMEGA,
            ref=REF,
            delta=rng.normal(size=(3, nd, 9, 9)) * 1e9,
            psi=rng.normal(size=(3, nf, 9)) + 1j * rng.normal(size=(3, nf, 9)),
            p=1 if nf == 4 else 2,
            r=1 if nd == 4 else 2,
        )
        pts, srcs = _node_sources(res, 12)
        for k_vec in (np.array([0.3, -0.5, 0.8]), np.array([-1.1j, 0.9, 0.2])):
            want = np.exp(-1j * (pts @ k_vec)) @ srcs
            got = source_moments(res, k_vec)
            assert np.abs(got - want).max() <= 1e-12 * np.abs(want).max()
