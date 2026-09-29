"""Radiation of the graded voxel's sources."""

import numpy as np

from cubic_scattering import ReferenceMedium
from cubic_scattering.graded_voxel.farfield import radiate
from cubic_scattering.resonance_tmatrix import elastodynamic_greens_deriv
from cubic_scattering.scattered_field import _voigt_to_tensor

REF = ReferenceMedium(5000.0, 3000.0, 2500.0)
OMEGA = 150.0


def _exact(src, x_obs):
    # u_i = G_ij F_j + d_k G_ij sigma_jk (source derivative d' = -d), the Green's tensor evaluated exactly
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
