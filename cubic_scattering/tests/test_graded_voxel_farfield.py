"""Radiation of the graded voxel's sources."""

import numpy as np

from cubic_scattering import ReferenceMedium
from cubic_scattering.graded_voxel.farfield import radiate
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
