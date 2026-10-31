"""The FFT embedding at a fast odd size: the smallest odd n >= 2N - 1 whose only factors are 3, 5 and 7.

A prime embedding size forces the FFT onto its general-length algorithm, four times slower per point
(79 points against 81 at N = 40). Any size at least 2N - 1 embeds the convolution without wrap-around, and
an odd one keeps the centred wavenumbers unique, so both solvers embed in the next such size. The size rule
must give the expected sizes, and the solves through a padded embedding must equal those through 2N - 1.
"""

import numpy as np
import pytest
import torch

from cubic_scattering.sphere_scattering_fft import compute_sphere_foldy_lax_fft, fast_odd_size


@pytest.mark.parametrize(
    ("n_min", "want"),
    [(7, 7), (11, 15), (15, 15), (23, 25), (31, 35), (47, 49), (63, 63), (79, 81), (95, 105), (127, 135)],
)
def test_the_fast_odd_size(n_min: int, want: int) -> None:
    got = fast_odd_size(n_min)
    assert got == want
    rest = got
    for p in (3, 5, 7):
        while rest % p == 0:
            rest //= p
    assert rest == 1 and got % 2 == 1


def test_the_galerkin_solve_through_a_padded_embedding(monkeypatch) -> None:
    """n_sub = 6 embeds in 15 rather than 11; the reference backend solves on the full kernel at 11."""
    from cubic_scattering.graded_voxel import fft as F
    from cubic_scattering.tests.test_graded_voxel_fft import CON, KHAT, OMEGA, REF, _profile

    args = (OMEGA, 10.0, REF, CON, 6, _profile, KHAT, KHAT, "P")
    padded = F.solve_graded_sphere_fft(*args, p=1, r=1, gmres_tol=1e-13)
    monkeypatch.setattr(F, "point_kernel_backend", lambda: "python")
    plain = F.solve_graded_sphere_fft(*args, p=1, r=1, gmres_tol=1e-13)
    assert np.linalg.norm(padded.psi - plain.psi) / np.linalg.norm(plain.psi) < 1e-11


def test_the_point_solve_through_a_padded_embedding() -> None:
    """n_sub = 6 embeds in 15 on the torch path; the NumPy solver embeds in 11."""
    from cubic_scattering.sphere_scattering import _plane_wave_strain_voigt
    from cubic_scattering.sphere_scattering_fft_gpu import solve_point_plane_wave
    from cubic_scattering.tests.test_sphere_scattering_fft_gpu import CONTRAST, REF

    omega, k_hat = 0.4 * REF.beta / (10.0 / 6), np.array([1.0, 0.0, 0.0])
    full = compute_sphere_foldy_lax_fft(
        omega, 10.0, REF, CONTRAST, 6, k_hat=k_hat, wave_type="P", gmres_tol=1e-12, gmres_maxiter=400
    )
    inc = np.concatenate([k_hat, _plane_wave_strain_voigt(k_hat, k_hat, omega / REF.alpha)])
    want = (full.psi_pw @ inc).reshape(-1, 9)
    got = solve_point_plane_wave(
        omega,
        10.0,
        REF,
        CONTRAST,
        6,
        k_hat,
        k_hat,
        "P",
        profile=None,
        single_site="exact",
        gmres_tol=1e-12,
        device=torch.device("cpu"),
        dtype=torch.complex128,
    )
    assert np.abs(got.psi - want).max() <= 1e-9 * np.abs(want).max()


@pytest.mark.parametrize("n_sub", [1, 4, 7, 12])
def test_the_vectorised_sphere_grid_equals_the_cell_by_cell_one(n_sub: int) -> None:
    """The default (sphere) cell selection, vectorised, against the cell-by-cell test it replaces."""
    from cubic_scattering.sphere_scattering_fft import _build_grid_index_map

    radius = 10.0
    got_idx, got_c, got_a = _build_grid_index_map(radius, n_sub)
    want_idx, want_c, want_a = _build_grid_index_map(
        radius, n_sub, lambda pos: bool(np.linalg.norm(pos) < radius)
    )
    np.testing.assert_array_equal(got_idx, want_idx)
    np.testing.assert_array_equal(got_c, want_c)
    assert got_a == want_a
