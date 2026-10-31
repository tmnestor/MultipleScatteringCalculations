"""The point formulation for one incident plane wave, as the comparison with the Galerkin voxel needs it.

``solve_point_plane_wave`` solves the single right-hand side of the incident wave on the torch path (the
canonical kernel), for a uniform or a graded contrast; its exciting field must be the plane-wave columns of
the full solver combined with the wave's [p, eps], and its far field (``point_far_field``, vectorised)
that of ``foldy_lax_far_field``.
"""

from typing import Any

import numpy as np
import pytest
import torch

from cubic_scattering.sphere_scattering import (
    _plane_wave_strain_voigt,
    foldy_lax_far_field,
    point_far_field,
)
from cubic_scattering.sphere_scattering_fft import compute_sphere_foldy_lax_fft
from cubic_scattering.sphere_scattering_fft_gpu import point_wave_far_field, solve_point_plane_wave
from cubic_scattering.tests.test_sphere_scattering_fft_gpu import CONTRAST, REF

RADIUS, N_SUB = 10.0, 5
OMEGA = 0.4 * REF.beta / (RADIUS / N_SUB)
K_HAT, POL = np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])
CPU = torch.device("cpu")


def _graded(pos):
    return 1.0 - 0.004 * float(pos @ pos)


def _inc():
    return np.concatenate([POL, _plane_wave_strain_voigt(K_HAT, POL, OMEGA / REF.alpha)])


def _obs():
    th = np.linspace(0.0, np.pi, 5)
    return np.stack([np.cos(th), np.sin(th), 0.0 * th], axis=1)


@pytest.mark.parametrize("profile", [None, _graded])
def test_the_single_wave_is_the_full_solvers_plane_wave_columns(profile) -> None:
    full = compute_sphere_foldy_lax_fft(
        OMEGA,
        RADIUS,
        REF,
        CONTRAST,
        N_SUB,
        k_hat=K_HAT,
        wave_type="P",
        gmres_tol=1e-12,
        gmres_maxiter=400,
        contrast_profile=profile,
    )
    want = (full.psi_pw @ _inc()).reshape(-1, 9)
    timings: dict[str, float] = {}
    got = solve_point_plane_wave(
        OMEGA,
        RADIUS,
        REF,
        CONTRAST,
        N_SUB,
        K_HAT,
        POL,
        "P",
        profile=profile,
        single_site="exact",
        gmres_tol=1e-12,
        device=CPU,
        dtype=torch.complex128,
        timings=timings,
    )
    assert np.abs(got.psi - want).max() <= 1e-9 * np.abs(want).max()
    assert set(timings) >= {"kernel_s", "solve_s", "iterations"}
    r = 5e3 * RADIUS
    u_p, u_s = foldy_lax_far_field(full, _obs(), r, K_HAT, POL, "P")
    v_p, v_s = point_wave_far_field(got, _obs(), r)
    scale = np.abs(u_p + u_s).max()
    assert np.abs(v_p + v_s - u_p - u_s).max() <= 1e-8 * scale


def _loop_far_field(centres, sources, r_hat_arr, r):
    """The far field cell by cell and angle by angle, as written before the vectorised readout."""
    kp, ks = OMEGA / REF.alpha, OMEGA / REF.beta
    u_p = np.zeros((len(r_hat_arr), 3), dtype=complex)
    u_s = np.zeros_like(u_p)
    for m, rh in enumerate(r_hat_arr):
        for x, s in zip(centres, sources, strict=True):
            sigma = np.diag(s[3:6]).astype(complex)
            sigma[1, 2] = sigma[2, 1] = s[6] / 2
            sigma[0, 2] = sigma[2, 0] = s[7] / 2
            sigma[0, 1] = sigma[1, 0] = s[8] / 2
            g_p = np.exp(1j * kp * (r - rh @ x)) / (4 * np.pi * REF.rho * REF.alpha**2 * r)
            g_s = np.exp(1j * ks * (r - rh @ x)) / (4 * np.pi * REF.rho * REF.beta**2 * r)
            u_p[m] += g_p * (rh @ s[:3] + 1j * kp * rh @ sigma @ rh) * rh
            q = s[:3] + 1j * ks * sigma @ rh
            u_s[m] += g_s * (q - (rh @ q) * rh)
    return u_p, u_s


def test_the_vectorised_far_field_equals_the_loop() -> None:
    rng = np.random.default_rng(11)
    centres = rng.normal(size=(40, 3)) * 5.0
    sources = rng.normal(size=(40, 9)) + 1j * rng.normal(size=(40, 9))
    r = 5e3 * RADIUS
    want = _loop_far_field(centres, sources, _obs(), r)
    got = point_far_field(centres, sources, OMEGA, REF, _obs(), r)
    for a, b in zip(got, want, strict=True):
        assert np.abs(a - b).max() <= 1e-13 * np.abs(b).max()


def test_a_batch_profile_is_evaluated_once_for_all_cells() -> None:
    from cubic_scattering.graded_voxel.site import BatchProfile

    calls = []

    def batch(pts):
        calls.append(len(pts))
        return 1.0 - 0.004 * np.einsum("ni,ni->n", pts, pts)

    kw: dict[str, Any] = dict(
        profile=None, single_site="package", gmres_tol=1e-12, device=CPU, dtype=torch.complex128
    )
    args = (OMEGA, RADIUS, REF, CONTRAST, N_SUB, K_HAT, POL, "P")
    got = solve_point_plane_wave(*args, **{**kw, "profile": BatchProfile(batch)})
    want = solve_point_plane_wave(*args, **{**kw, "profile": _graded})
    assert calls == [len(got.centres)]
    assert np.abs(got.psi - want.psi).max() <= 1e-12 * np.abs(want.psi).max()


def test_the_cells_kept_follow_inside() -> None:
    """inside=None keeps the cells whose centre is in the sphere; a rule that says the same gives the same
    solve, and keeping every cell that overlaps the sphere keeps more."""
    h = RADIUS / N_SUB
    kw: dict[str, Any] = dict(
        profile=None, single_site="exact", gmres_tol=1e-12, device=CPU, dtype=torch.complex128
    )
    args = (OMEGA, RADIUS, REF, CONTRAST, N_SUB, K_HAT, POL, "P")
    default = solve_point_plane_wave(*args, **kw)
    same = solve_point_plane_wave(*args, **kw, inside=lambda c: float(np.linalg.norm(c)) < RADIUS)
    wider = solve_point_plane_wave(
        *args, **kw, inside=lambda c: float(np.linalg.norm(np.maximum(np.abs(c) - h, 0.0))) < RADIUS
    )
    assert np.array_equal(same.centres, default.centres)
    assert np.abs(same.psi - default.psi).max() <= 1e-14 * np.abs(default.psi).max()
    assert len(wider.centres) > len(default.centres)
