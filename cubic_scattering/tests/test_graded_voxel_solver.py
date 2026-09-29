"""The graded voxel's global solve."""

import numpy as np

from cubic_scattering import MaterialContrast, ReferenceMedium
from cubic_scattering.graded_voxel.basis import TEST_EXPONENTS, monomials
from cubic_scattering.graded_voxel.solver import plane_wave_moments, solve_graded_sphere

REF = ReferenceMedium(5000.0, 3000.0, 2500.0)
CON = MaterialContrast(Dlambda=2.0e9, Dmu=1.0e9, Drho=100.0)
KHAT = np.array([1.0, 0.0, 0.0])


def test_plane_wave_moments_match_quadrature():
    x, w = np.polynomial.legendre.leggauss(10)
    xi = np.stack(np.meshgrid(x, x, x, indexing="ij"), -1).reshape(-1, 3)
    ww = np.einsum("i,j,k->ijk", w, w, w).ravel()
    h, c, k = 1.3, np.array([[0.4, -2.0, 1.1]]), np.array([0.05, -0.02, 0.03])
    amp = np.arange(9) + 1j
    got = plane_wave_moments(c, h, k, amp)[0]
    ph = np.exp(1j * (c[0] + h * xi) @ k)
    want = h**3 * (monomials(TEST_EXPONENTS, xi) * ww) @ ph
    np.testing.assert_allclose(got, want[:, None] * amp[None, :], rtol=1e-13)


def test_zero_contrast_cells_carry_no_source():
    # Review Focus 1: a profile that vanishes on part of the grid
    def half(pos):
        return 1.0 if pos[0] < 0 else 0.0

    res = solve_graded_sphere(150.0, 10.0, REF, CON, 2, half, KHAT, KHAT, "P")
    zero = res.centres[:, 0] > 0
    assert np.abs(res.delta[zero]).max() == 0.0
    assert np.all(np.isfinite(res.psi))


def test_cell_order_does_not_matter():
    # Review Focus 2: permuting the cells permutes the solution (offset signs are consistent)
    def prof(pos):
        return 0.5 + 0.02 * pos[1]

    a = solve_graded_sphere(150.0, 10.0, REF, CON, 2, prof, KHAT, KHAT, "P")
    perm = np.arange(len(a.centres))[::-1]
    b = solve_graded_sphere(150.0, 10.0, REF, CON, 2, prof, KHAT, KHAT, "P", _order=perm)
    np.testing.assert_allclose(b.psi, a.psi[perm], rtol=1e-10, atol=1e-10 * np.abs(a.psi).max())


def test_p0_born_far_field_matches_the_package():
    # the p = 0, r = 0 arm at weak contrast agrees with the package's T9 sphere solver to O((kh)^2)
    from cubic_scattering.graded_voxel.farfield import graded_far_field
    from cubic_scattering.sphere_scattering import foldy_lax_far_field
    from cubic_scattering.sphere_scattering_fft import compute_sphere_foldy_lax_fft

    weak = MaterialContrast(CON.Dlambda * 1e-6, CON.Dmu * 1e-6, CON.Drho * 1e-6)
    omega = 0.1 * REF.beta / 10.0
    dirs = np.array([[1.0, 0, 0], [0, 1.0, 0], [-0.6, 0, 0.8]])
    # the same cells as the FFT solver (centre inside), so only the conventions differ
    g = solve_graded_sphere(
        omega,
        10.0,
        REF,
        weak,
        4,
        lambda _: 1.0,
        KHAT,
        KHAT,
        "P",
        p=0,
        r=0,
        inside=lambda q: bool(np.linalg.norm(q) < 10.0),
    )
    ug = graded_far_field(g, dirs, 5e5, KHAT, KHAT, "P")
    # the FFT solver builds its grid with the same _build_grid_index_map
    fl = compute_sphere_foldy_lax_fft(
        omega, 10.0, REF, weak, n_sub=4, k_hat=KHAT, wave_type="P", gmres_tol=1e-12
    )
    uf = foldy_lax_far_field(fl, dirs, 5e5, KHAT, KHAT, wave_type="P")
    tot_g, tot_f = ug[0] + ug[1], uf[0] + uf[1]
    assert np.abs(tot_g - tot_f).max() / np.abs(tot_f).max() < 1e-3


def test_every_cell_holding_contrast_is_kept():
    # a Galerkin cell carries the exact projection of its contrast, so a cell whose centre lies outside the
    # sphere but which overlaps it must be kept (dropping it truncates real contrast: G4b, 2026-09-29)
    res = solve_graded_sphere(150.0, 10.0, REF, CON, 2, lambda _: 0.1, KHAT, KHAT, "P", p=0, r=0)
    assert len(res.centres) == 8
    from cubic_scattering.sphere_scattering_fft import _build_grid_index_map

    _, centres, h = _build_grid_index_map(10.0, 4, inside=lambda q: True)
    kept = [c for c in centres if np.linalg.norm(c) < 10.0 + np.sqrt(3) * h]
    res4 = solve_graded_sphere(150.0, 10.0, REF, CON, 4, lambda _: 0.0, KHAT, KHAT, "P", p=0, r=0)
    assert len(res4.centres) == len(kept) == 64
