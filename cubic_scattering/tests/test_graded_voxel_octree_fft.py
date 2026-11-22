"""The matrix-free product for cells of several sizes, against the dense octree solver."""

import numpy as np
import pytest

from cubic_scattering import MaterialContrast, ReferenceMedium
from cubic_scattering.graded_voxel.octree import (
    _assemble_octree,
    born_series_octree,
    refine_leaves,
    solve_graded_octree,
    uniform_leaves,
)
from cubic_scattering.graded_voxel.octree_fft import (
    born_series_octree_fft,
    octree_operator_fft,
    solve_graded_octree_fft,
)

REF = ReferenceMedium(5000.0, 3000.0, 2500.0)
CONTRAST = MaterialContrast(2.0e9, 1.0e9, 100.0)
OMEGA, RADIUS = 150.0, 10.0
K_HAT = np.array([0.0, 0.0, 1.0])


def _profile(pos):
    return float(np.exp(-0.5 * np.dot(pos, pos) / 36.0))


def _three_sizes():
    """Eight cells of half-width 5, one divided, and one of its children divided again."""
    centres, hs = uniform_leaves(RADIUS, 2)
    flag = np.zeros(len(hs), dtype=bool)
    flag[0] = True
    centres, hs = refine_leaves(centres, hs, flag)
    flag = np.zeros(len(hs), dtype=bool)
    flag[-1] = True
    return refine_leaves(centres, hs, flag)


def _dense(centres, hs, p, r):
    return _assemble_octree(OMEGA, REF, CONTRAST, centres, hs, _profile, K_HAT, K_HAT, "P", p, r, None)


@pytest.mark.parametrize(("p", "r"), [(0, 0), (1, 1)])
def test_the_product_is_the_dense_matrix_on_a_tree_of_three_sizes(p, r):
    centres, hs = _three_sizes()
    assert len(np.unique(hs)) == 3
    system = _dense(centres, hs, p, r)
    op = octree_operator_fft(OMEGA, REF, CONTRAST, centres, hs, _profile, K_HAT, K_HAT, "P", p, r)
    np.testing.assert_allclose(op.rhs, system.rhs, rtol=1e-12, atol=1e-14 * np.abs(system.rhs).max())
    rng = np.random.default_rng(3)
    x = rng.standard_normal(len(op.rhs)) + 1j * rng.standard_normal(len(op.rhs))
    want = system.a @ x
    # what the two share is the mass term; the test is of the coupling, so compare with that taken out
    coupling = want - op.mass(x, False)
    got = op.matvec(x) - op.mass(x, False)
    assert np.abs(got - coupling).max() < 1e-11 * np.abs(coupling).max()
    assert op.products[0] == 1


def test_the_solve_and_the_series_are_the_dense_ones():
    centres, hs = _three_sizes()
    dense = solve_graded_octree(OMEGA, REF, CONTRAST, centres, hs, _profile, K_HAT, K_HAT, "P", p=0, r=0)
    timings = {}
    free = solve_graded_octree_fft(
        OMEGA, REF, CONTRAST, centres, hs, _profile, K_HAT, K_HAT, "P", p=0, r=0, timings=timings
    )
    scale = np.abs(dense.psi).max()
    assert np.abs(free.psi - dense.psi).max() < 1e-8 * scale
    assert timings["n_fine"] == 8 and timings["products"] > 0
    args = (OMEGA, REF, CONTRAST, centres, hs, _profile, K_HAT, K_HAT, "P", 2)
    for a, b in zip(
        born_series_octree(*args, p=0, r=0), born_series_octree_fft(*args, p=0, r=0), strict=True
    ):
        assert np.abs(a.psi - b.psi).max() < 1e-7 * np.abs(a.psi).max()


def test_the_series_needs_one_term():
    centres, hs = uniform_leaves(RADIUS, 2)
    with pytest.raises(ValueError, match="at least one term") as err:
        born_series_octree_fft(OMEGA, REF, CONTRAST, centres, hs, _profile, K_HAT, K_HAT, "P", 0, p=0, r=0)
    assert "Fix:" in str(err.value)


def test_leaves_off_the_lattice_are_refused():
    centres, hs = _three_sizes()
    centres = centres.copy()
    centres[0] += 0.3 * hs.min()
    with pytest.raises(ValueError, match="do not lie on the lattice") as err:
        octree_operator_fft(OMEGA, REF, CONTRAST, centres, hs, _profile, K_HAT, K_HAT, "P", 0, 0)
    assert "Fix:" in str(err.value)
