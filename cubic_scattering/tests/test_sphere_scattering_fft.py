"""Unit tests for FFT-accelerated GMRES Foldy-Lax sphere solver.

Validates internal correctness:
    1. Grid index mapping matches sphere_sub_cell_centres
    2. Pack/unpack roundtrip
    3. FFT matvec matches dense matvec
    4. FFT solution matches dense solution at n_sub=3 and n_sub=5
"""

from __future__ import annotations

import numpy as np
import pytest

from cubic_scattering import (
    MaterialContrast,
    ReferenceMedium,
)
from cubic_scattering.cell_averaged_pair import averaged_pair_block_9x9
from cubic_scattering.effective_contrasts import compute_cube_tmatrix
from cubic_scattering.resonance_tmatrix import (
    _propagator_block_9x9,
    _sub_cell_tmatrix_9x9,
)
from cubic_scattering.sphere_scattering import (
    compute_sphere_foldy_lax,
    sphere_sub_cell_centres,
)
from cubic_scattering.sphere_scattering_fft import (
    _build_fft_kernel,
    _build_grid_index_map,
    _matvec_fft,
    _pack,
    _unpack,
    compute_sphere_foldy_lax_fft,
)

REF = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
CONTRAST = MaterialContrast(Dlambda=2.0e9, Dmu=1.0e9, Drho=100.0)
RADIUS = 0.5
OMEGA = 0.1 * REF.alpha / RADIUS  # ka_P = 0.1


class TestGridIndexMapping:
    def test_grid_index_matches_sphere_sub_cell_centres(self) -> None:
        for n_sub in [3, 5, 7]:
            grid_idx, centres_fft, a_sub_fft = _build_grid_index_map(RADIUS, n_sub)
            centres_ref, a_sub_ref = sphere_sub_cell_centres(RADIUS, n_sub)

            assert a_sub_fft == pytest.approx(a_sub_ref)
            assert len(centres_fft) == len(centres_ref)

            order_fft = np.lexsort(centres_fft.T)
            order_ref = np.lexsort(centres_ref.T)
            np.testing.assert_allclose(centres_fft[order_fft], centres_ref[order_ref], atol=1e-14)


def _staircase(radius: float, n_coarse: int):
    """Cell-centre test for the n_coarse sphere staircase, usable on any refinement of its grid."""
    dd = 2.0 * radius / n_coarse

    def inside(pos: np.ndarray) -> bool:
        idx = np.clip(np.floor((pos + radius) / dd), 0, n_coarse - 1)
        return bool(np.linalg.norm(-radius + (idx + 0.5) * dd) < radius)

    return inside


class TestShapeMask:
    """The optional ``inside`` test: the default is the sphere, and a staircase refines exactly."""

    def test_default_is_the_sphere(self) -> None:
        for n_sub in [3, 5]:
            a = _build_grid_index_map(RADIUS, n_sub)
            b = _build_grid_index_map(RADIUS, n_sub, lambda p: bool(np.linalg.norm(p) < RADIUS))
            np.testing.assert_array_equal(a[0], b[0])
            np.testing.assert_allclose(a[1], b[1], atol=0.0)

    def test_staircase_at_its_own_resolution_is_the_sphere(self) -> None:
        for n_sub in [3, 4, 5]:
            a = _build_grid_index_map(RADIUS, n_sub)
            b = _build_grid_index_map(RADIUS, n_sub, _staircase(RADIUS, n_sub))
            np.testing.assert_array_equal(a[0], b[0])

    def test_refined_staircase_has_m_cubed_times_the_cells(self) -> None:
        for n_coarse, m in [(3, 2), (4, 2), (3, 3)]:
            coarse = _build_grid_index_map(RADIUS, n_coarse)
            fine = _build_grid_index_map(RADIUS, n_coarse * m, _staircase(RADIUS, n_coarse))
            assert len(fine[0]) == m**3 * len(coarse[0])
            # the refined staircase fills exactly the coarse staircase's volume
            assert len(fine[0]) * (2 * fine[2]) ** 3 == pytest.approx(len(coarse[0]) * (2 * coarse[2]) ** 3)

    def test_solver_accepts_the_mask(self) -> None:
        a = compute_sphere_foldy_lax_fft(OMEGA, RADIUS, REF, CONTRAST, n_sub=3)
        mask = _staircase(RADIUS, 3)
        b = compute_sphere_foldy_lax_fft(OMEGA, RADIUS, REF, CONTRAST, n_sub=3, inside=mask)
        assert a.n_cells == b.n_cells
        np.testing.assert_allclose(b.T_comp_9x9, a.T_comp_9x9, rtol=1e-12, atol=0.0)


class TestPackUnpack:
    def test_roundtrip(self) -> None:
        grid_idx, centres, a_sub = _build_grid_index_map(RADIUS, 3)
        nC = len(centres)
        rng = np.random.default_rng(123)
        w = rng.standard_normal(9 * nC) + 1j * rng.standard_normal(9 * nC)

        grids = _pack(w, grid_idx, 5)
        w_back = _unpack(grids, grid_idx, nC)
        np.testing.assert_allclose(w_back, w, atol=1e-15)


class TestFFTMatvec:
    @pytest.mark.parametrize("cell_average", [False, True])
    def test_fft_matvec_vs_dense(self, cell_average: bool) -> None:
        """The FFT convolution reproduces the dense assembly of the SAME operator.

        Both propagators are exercised.  The test previously built the kernel
        through ``_build_fft_kernel`` while assembling the dense reference from
        the point propagator explicitly; once the single cell average became the
        default for the kernel, the two sides were different operators and the
        comparison failed at 6.5e-3 -- which is the size of the cell-average
        correction, not a defect in the convolution.  Pinning the test to one
        propagator would have hidden the new default instead of covering it.
        """
        n_sub = 3
        grid_idx, centres, a_sub = _build_grid_index_map(RADIUS, n_sub)
        nC = len(centres)
        nP = 2 * n_sub - 1

        rayleigh_sub = compute_cube_tmatrix(OMEGA, a_sub, REF, CONTRAST)
        T_loc = _sub_cell_tmatrix_9x9(rayleigh_sub, OMEGA, a_sub)
        kernel_hat = _build_fft_kernel(n_sub, a_sub, T_loc, OMEGA, REF, cell_average=cell_average)

        pitch = 2.0 * a_sub
        dim = 9 * nC
        A_dense = np.eye(dim, dtype=complex)
        for m in range(nC):
            for n in range(nC):
                if m != n:
                    r_vec = centres[m] - centres[n]
                    P = (
                        averaged_pair_block_9x9(r_vec, OMEGA, REF, pitch)
                        if cell_average
                        else _propagator_block_9x9(r_vec, OMEGA, REF)
                    )
                    A_dense[9 * m : 9 * m + 9, 9 * n : 9 * n + 9] -= P @ T_loc

        rng = np.random.default_rng(42)
        x_test = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)

        y_fft = _matvec_fft(x_test, kernel_hat, grid_idx, nP, nC)
        y_dense = A_dense @ x_test

        rel_err = np.linalg.norm(y_fft - y_dense) / np.linalg.norm(y_dense)
        assert rel_err < 1e-10, f"FFT matvec rel err = {rel_err:.2e}"


class TestFFTvsDense:
    @pytest.mark.parametrize("n_sub", [3, 5])
    def test_fft_vs_dense(self, n_sub: int) -> None:
        result_dense = compute_sphere_foldy_lax(OMEGA, RADIUS, REF, CONTRAST, n_sub)
        result_fft = compute_sphere_foldy_lax_fft(OMEGA, RADIUS, REF, CONTRAST, n_sub, gmres_tol=1e-12)

        rel_err = np.linalg.norm(result_fft.T_comp_9x9 - result_dense.T_comp_9x9) / np.linalg.norm(
            result_dense.T_comp_9x9
        )
        assert rel_err < 1e-5, f"T_comp rel err at n_sub={n_sub}: {rel_err:.2e}"
