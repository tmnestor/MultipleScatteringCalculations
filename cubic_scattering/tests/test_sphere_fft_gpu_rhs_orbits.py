"""The point solver's right-hand sides by the cube group: one GMRES solve per orbit of columns.

(I - P T) commutes with U_Q, which moves a cell's field from x to Q x and applies S to its 9 components;
the centred sphere grid is mapped to itself. The Taylor columns e_c are permuted with signs by every Q
(orbits: displacements, normal strains, shear strains), the plane-wave columns e_c exp(i k.x) by the Q that
fix k. So the solutions of the other columns follow from one solve per orbit, exactly.
"""

import numpy as np
import pytest
import torch

from cubic_scattering import sphere_scattering_fft_gpu as G
from cubic_scattering.graded_voxel.fft import signed_permutations
from cubic_scattering.sphere_scattering_fft import _build_grid_index_map, compute_sphere_foldy_lax_fft
from cubic_scattering.tests.test_sphere_scattering_fft_gpu import CONTRAST, REF

CPU = torch.device("cpu")
RADIUS, N_SUB = 10.0, 5
OMEGA = 0.3 * REF.beta / (RADIUS / N_SUB)


def test_the_column_orbits_of_the_whole_group_and_of_an_axis() -> None:
    reps, _ = G._column_orbits(signed_permutations())
    assert reps == [0, 3, 6]  # displacement, normal strain, shear strain
    axis = [q for q in signed_permutations() if np.array_equal(q @ [0.0, 0.0, 1.0], [0.0, 0.0, 1.0])]
    assert len(axis) == 8
    reps, derived = G._column_orbits(axis)
    assert reps == [0, 2, 3, 5, 6, 8]  # {ux, uy}, {uz}, {exx, eyy}, {ezz}, {eyz, exz}, {exy}
    assert sorted([c for c, *_ in derived] + reps) == list(range(9))


def test_the_cell_permutation_maps_the_sphere_grid_to_itself() -> None:
    grid_idx, centres, _ = _build_grid_index_map(RADIUS, N_SUB)
    for q in signed_permutations():
        perm = G._cell_permutation(grid_idx, N_SUB, q)
        np.testing.assert_allclose(centres[perm], centres @ q.T, atol=1e-12)


@pytest.mark.parametrize(
    ("k_hat", "n_solves"), [(np.array([0.0, 0.0, 1.0]), 9), (np.array([0.3, 0.5, 0.81]), 12)]
)
def test_the_orbit_solutions_equal_solving_every_column(monkeypatch, k_hat, n_solves) -> None:
    calls = []
    real = G.torch_gmres

    def counted(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(G, "torch_gmres", counted)
    kw = dict(
        omega=OMEGA, radius=RADIUS, ref=REF, contrast=CONTRAST, n_sub=N_SUB, k_hat=k_hat, wave_type="P"
    )
    got = G.compute_sphere_foldy_lax_fft_gpu(**kw, gmres_tol=1e-12, device=CPU, dtype=torch.complex128)
    assert len(calls) == n_solves  # 3 Taylor orbits, and 6 plane-wave orbits on an axis or 9 off it
    want = compute_sphere_foldy_lax_fft(**kw, gmres_tol=1e-12, gmres_maxiter=400)
    for name in ("psi_exc", "psi_pw"):
        a, b = getattr(got, name), getattr(want, name)
        assert np.abs(a - b).max() <= 1e-9 * np.abs(b).max(), name
    scale = np.abs(want.T_comp_9x9).max()
    assert np.abs(got.T_comp_9x9 - want.T_comp_9x9).max() <= 1e-9 * scale
