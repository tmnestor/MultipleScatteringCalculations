"""Cells of different sizes: coupling blocks on an octree, by the two-scale relation."""

import itertools

import numpy as np
import pytest

from cubic_scattering import MaterialContrast, ReferenceMedium
from cubic_scattering.graded_voxel.basis import (
    SOURCE_EXPONENTS,
    contrast_values,
    monomials,
    source_exponents,
)
from cubic_scattering.graded_voxel.blocks import coupling_block
from cubic_scattering.graded_voxel.farfield import graded_far_field
from cubic_scattering.graded_voxel.kernel import kernel_9x9
from cubic_scattering.graded_voxel.octree import (
    field_reexpansion,
    octree_block,
    octree_far_field,
    refine_leaves,
    solve_graded_octree,
    source_reexpansion,
    uniform_leaves,
)
from cubic_scattering.graded_voxel.solver import solve_graded_sphere

REF = ReferenceMedium(5000.0, 3000.0, 2500.0)
OMEGA, H = 150.0, 1.25


def test_field_reexpansion_reproduces_the_parent_functions_on_a_descendant():
    rng = np.random.default_rng(0)
    xi = rng.uniform(-1, 1, (40, 3))
    for scale, shift in ((0.5, (0.5, -0.5, 0.5)), (0.25, (-0.75, 0.25, 0.75))):
        for n in (4, 10):
            c = field_reexpansion(n, scale, shift)
            parent = contrast_values(scale * xi + np.array(shift))[:n]
            np.testing.assert_allclose(c @ contrast_values(xi)[:n], parent, rtol=1e-12, atol=1e-13)


def test_source_reexpansion_reproduces_the_parent_monomials_on_a_descendant():
    rng = np.random.default_rng(1)
    xi = rng.uniform(-1, 1, (40, 3))
    for n in (10, 20, 35):
        exps = source_exponents(n)
        d = source_reexpansion(n, 0.5, (-0.5, 0.5, 0.5))
        parent = monomials(exps, 0.5 * xi + np.array([-0.5, 0.5, 0.5]))
        np.testing.assert_allclose(d @ monomials(exps, xi), parent, rtol=1e-12, atol=1e-13)


def _cell_rule(n):
    x, w = np.polynomial.legendre.leggauss(n)
    xi = np.stack([g.ravel() for g in np.meshgrid(x, x, x, indexing="ij")], axis=1)
    return xi, (w[:, None, None] * w[None, :, None] * w[None, None, :]).ravel()


def _gauss_block(c_t, h_t, c_s, h_s, n_test, n_source, n=8):
    """Six-dimensional Gauss quadrature of the block between two separated boxes; the larger box gets the
    higher order (n per axis on the smaller, n scaled by the size ratio on the larger)."""
    big = max(h_t, h_s)
    xt, wt = _cell_rule(round(n * h_t / min(h_t, h_s)) if h_t == big and h_t != h_s else n)
    xs, ws = _cell_rule(round(n * h_s / min(h_t, h_s)) if h_s == big and h_t != h_s else n)
    sep = (np.asarray(c_t) + h_t * xt)[:, None, :] - (np.asarray(c_s) + h_s * xs)[None, :, :]
    p = kernel_9x9(sep.reshape(-1, 3), OMEGA, REF).reshape(len(xt), len(xs), 81)
    lt = contrast_values(xt)[:n_test] * wt
    ls = monomials(source_exponents(n_source), xs) * ws
    out = np.einsum("ap,pqz,cq->acz", lt, p, ls) * h_t**3 * h_s**3
    return out.reshape(n_test, n_source, 9, 9)


@pytest.mark.parametrize(
    ("h_t", "c_s", "h_s"),
    [
        (2 * H, (7 * H, H, -H), H),  # a large field cell, a small source cell
        (H, (7 * H, 3 * H, -H), 2 * H),  # a small field cell, a large source cell
        (4 * H, (9 * H, -3 * H, H), H),  # two levels apart
    ],
)
def test_unequal_cells_match_six_dimensional_quadrature(h_t, c_s, h_s):
    c_t = (0.0, 0.0, 0.0)
    got = octree_block(c_t, h_t, c_s, h_s, OMEGA, REF)
    want = _gauss_block(c_t, h_t, c_s, h_s, 4, 10, n=6)
    assert np.linalg.norm(got - want) / np.linalg.norm(want) < 1e-9


def test_unequal_cells_with_a_quadratic_field():
    c_t, h_t, c_s, h_s = (0.0, 0.0, 0.0), 2 * H, (7 * H, -H, H), H
    got = octree_block(c_t, h_t, c_s, h_s, OMEGA, REF, n_source=35, n_test=10)
    want = _gauss_block(c_t, h_t, c_s, h_s, 10, 35, n=7)
    assert np.linalg.norm(got - want) / np.linalg.norm(want) < 1e-9


def test_equal_cells_are_the_ordinary_block():
    got = octree_block((0.0, 0.0, 0.0), H, (2 * H, -4 * H, 0.0), H, OMEGA, REF)
    want = coupling_block((-1, 2, 0), H, OMEGA, REF)
    np.testing.assert_array_equal(got, want)


def test_touching_unequal_cells_agree_with_one_more_level_of_subdivision():
    # a large cell and a small one sharing part of a face: no quadrature reference exists for the singular
    # integral, so the block is rebuilt from the cells' own children, a level finer on both sides
    c_t, h_t, c_s, h_s = np.zeros(3), 2 * H, np.array([3 * H, H, -H]), H
    got = octree_block(tuple(c_t), h_t, tuple(c_s), h_s, OMEGA, REF)
    acc = np.zeros_like(got)
    signs = list(itertools.product((-1, 1), repeat=3))
    for st in signs:
        ct_child = c_t + 0.5 * h_t * np.array(st)
        c_mat = field_reexpansion(4, 0.5, tuple(0.5 * v for v in st))
        for ss in signs:
            cs_child = c_s + 0.5 * h_s * np.array(ss)
            d_mat = source_reexpansion(10, 0.5, tuple(0.5 * v for v in ss))
            sub = octree_block(tuple(ct_child), h_t / 2, tuple(cs_child), h_s / 2, OMEGA, REF)
            acc += np.einsum("ab,bdij,cd->acij", c_mat, sub, d_mat)
    assert np.linalg.norm(acc - got) / np.linalg.norm(got) < 1e-8
    assert SOURCE_EXPONENTS[0] == (0, 0, 0)


def test_misaligned_cells_are_refused():
    with pytest.raises(ValueError, match="octree"):
        octree_block((0.0, 0.0, 0.0), 2 * H, (5.1 * H, 0.0, 0.0), H, OMEGA, REF)
    with pytest.raises(ValueError, match="power of two"):
        octree_block((0.0, 0.0, 0.0), 3 * H, (9 * H, 0.0, 0.0), H, OMEGA, REF)


CON = MaterialContrast(Dlambda=2.0e9, Dmu=1.0e9, Drho=100.0)
KHAT = np.array([1.0, 0.0, 0.0])
RADIUS = 10.0


def _profile(pos):
    return 1.0 - 0.004 * float(pos @ pos)


def test_uniform_leaves_are_the_uniform_grid():
    centres, hs = uniform_leaves(RADIUS, 4)
    ref = solve_graded_sphere(OMEGA, RADIUS, REF, CON, 4, _profile, KHAT, KHAT, "P")
    np.testing.assert_allclose(centres, ref.centres, atol=1e-13)
    assert np.all(hs == ref.h)


def test_refining_every_leaf_gives_the_finer_uniform_grid():
    c2, h2 = uniform_leaves(RADIUS, 2)
    c4, h4 = refine_leaves(c2, h2, np.ones(len(c2), dtype=bool))
    want, hw = uniform_leaves(RADIUS, 4)
    assert len(c4) == 8 * len(c2) and np.all(h4 == hw[0])
    key = lambda a: sorted(map(tuple, np.round(a, 9)))  # noqa: E731
    assert key(c4) == key(want)


@pytest.mark.parametrize(("p", "r"), [(0, 0), (1, 1)])
def test_octree_solver_on_a_uniform_tree_equals_the_uniform_solver(p, r):
    centres, hs = uniform_leaves(RADIUS, 3)
    oct_res = solve_graded_octree(OMEGA, REF, CON, centres, hs, _profile, KHAT, KHAT, "P", p=p, r=r)
    ref = solve_graded_sphere(OMEGA, RADIUS, REF, CON, 3, _profile, KHAT, KHAT, "P", p=p, r=r)
    assert np.linalg.norm(oct_res.psi - ref.psi) / np.linalg.norm(ref.psi) < 1e-11
    dirs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-0.6, 0.0, 0.8]])
    got = octree_far_field(oct_res, dirs, 1e9)
    want = graded_far_field(ref, dirs, 1e9, KHAT, KHAT, "P")
    for g, w in zip(got, want, strict=True):
        assert np.abs(g - w).max() / np.abs(w).max() < 1e-11


def test_a_mixed_tree_is_closer_to_the_fine_grid_than_the_coarse_grid_is():
    # refine every second leaf of the coarse grid: leaves of two sizes in one system
    c2, h2 = uniform_leaves(RADIUS, 2)
    flag = np.arange(len(c2)) % 2 == 1
    cm, hm = refine_leaves(c2, h2, flag)
    assert len(np.unique(hm)) == 2
    dirs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-0.6, 0.0, 0.8], [0.0, 0.0, -1.0]])

    def field(c, h):
        res = solve_graded_octree(OMEGA, REF, CON, c, h, _profile, KHAT, KHAT, "P")
        up, us = octree_far_field(res, dirs, 1e9)
        return up + us

    coarse, mixed = field(c2, h2), field(cm, hm)
    fine = field(*uniform_leaves(RADIUS, 4))
    assert np.abs(mixed - fine).max() < np.abs(coarse - fine).max()
