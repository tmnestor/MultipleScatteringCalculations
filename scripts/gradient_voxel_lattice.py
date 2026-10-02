#!/usr/bin/env python3
"""The gradient hierarchy on a LATTICE of cubic voxels: self block from the cube moments, coupling from
ordinary integrals.

Each voxel c carries the Taylor coefficients of the displacement about its centre x_c,

    u_j(x_c + xi) = sum_{|W| <= q} xi_W U^c_{j;W} / |W|! ,

and the Lippmann-Schwinger equation and its derivatives to order q are imposed at every centre:

    U^c_{i;P} - sum_{c'} { omega^2 drho sum_W K[i,j; P; W](x_c - x_c') U^{c'}_{j;W} / |W|!
                           + sum_W K[i,n; P+k; W](x_c - x_c') dc_{nkrj} U^{c'}_{j;rW} / |W|! }
            = d_P u0_i (x_c),

    K[i,n; D; W](Delta) = int_cube (d_D G_in)(Delta - xi) xi_W dxi .

* c' = c (Delta = 0): the integrand is singular and the integral is a distribution. It is the moment of the
  hierarchy, K = (-1)^|D| g[i,n; D; W], in closed form on the cube. All the singular work of the scheme is
  here, and it is done once.
* c' != c: the field point is the centre of another voxel, at least half a voxel from the source voxel, so
  the integrand is smooth and a tensor Gauss rule integrates it. The blocks depend on the offset alone.

The derivatives of the Green's tensor, to any order, come from one representation: every derivative of
exp(i k r)/r is a sum of terms x^a y^b z^c r^p exp(i k r), closed under differentiation.

This module provides the pieces; ``measure_lattice_gradient_hierarchy.py`` runs the tests.
"""

import functools
import itertools
import math
import sys
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import measure_ball_gradient_hierarchy as hier  # noqa: E402
import measure_cube_gradient_hierarchy as cube  # noqa: E402

AXES = (0, 1, 2)
Terms = dict[tuple[int, int, int, int], complex]


# ------------------------------------------------------------ derivatives of exp(i k r)/r, to any order
def d_term(f: Terms, axis: int, ik: complex) -> Terms:
    """d/dx_axis of sum c x^a y^b z^c r^p exp(i k r)."""
    out: Terms = {}
    for (a, b, c, p), coef in f.items():
        e = [a, b, c]
        if e[axis] > 0:
            g = list(e)
            g[axis] -= 1
            key = (g[0], g[1], g[2], p)
            out[key] = out.get(key, 0.0) + coef * e[axis]
        g = list(e)
        g[axis] += 1
        if p != 0:
            key = (g[0], g[1], g[2], p - 2)
            out[key] = out.get(key, 0.0) + coef * p
        key = (g[0], g[1], g[2], p - 1)
        out[key] = out.get(key, 0.0) + coef * ik
    return {k: v for k, v in out.items() if v != 0.0}


@functools.cache
def scalar_derivative(
    ds: tuple[int, ...],
) -> tuple[tuple[tuple[int, int, int, int], tuple[complex, ...]], ...]:
    """d_ds [exp(i k r)/r] as terms whose coefficients are polynomials in (i k), one tuple per term."""
    # carry the power of (i k) explicitly so that one table serves every wavenumber
    f: dict[tuple[int, int, int, int], dict[int, float]] = {(0, 0, 0, -1): {0: 1.0}}
    for axis in ds:
        out: dict[tuple[int, int, int, int], dict[int, float]] = {}

        def add(key, power, val, out=out):
            slot = out.setdefault(key, {})
            slot[power] = slot.get(power, 0.0) + val

        for (a, b, c, p), poly in f.items():
            e = [a, b, c]
            for power, coef in poly.items():
                if e[axis] > 0:
                    g = list(e)
                    g[axis] -= 1
                    add((g[0], g[1], g[2], p), power, coef * e[axis])
                g = list(e)
                g[axis] += 1
                if p != 0:
                    add((g[0], g[1], g[2], p - 2), power, coef * p)
                add((g[0], g[1], g[2], p - 1), power + 1, coef)
        f = out
    table = []
    for key, poly in f.items():
        top = max(poly)
        table.append((key, tuple(poly.get(n, 0.0) for n in range(top + 1))))
    return tuple(table)


def eval_scalar(ds: tuple[int, ...], k: float, x: np.ndarray) -> np.ndarray:
    """d_ds [exp(i k r)/r] at the points x (N, 3)."""
    r = np.linalg.norm(x, axis=1)
    ik = 1j * k
    tot = np.zeros(len(x), dtype=complex)
    for (a, b, c, p), poly in scalar_derivative(tuple(sorted(ds))):
        coef = sum(cn * ik**n for n, cn in enumerate(poly))
        if coef != 0.0:
            tot += coef * x[:, 0] ** a * x[:, 1] ** b * x[:, 2] ** c * r ** float(p)
    return tot * np.exp(ik * r)


def green_derivative(i: int, n: int, ds: tuple[int, ...], omega: float, x: np.ndarray) -> np.ndarray:
    """(d_ds G_in)(x) for the elastodynamic tensor of the background, at the points x (N, 3)."""
    ks, kp = omega / hier.REF.beta, omega / hier.REF.alpha
    mu = hier.REF.rho * hier.REF.beta**2
    tot = (eval_scalar((*ds, i, n), ks, x) - eval_scalar((*ds, i, n), kp, x)) / ks**2
    if i == n:
        tot = tot + eval_scalar(ds, ks, x)
    return tot / (4.0 * math.pi * mu)


# ------------------------------------------------------------ the blocks
def sorted_indices(max_len: int) -> list[tuple[int, ...]]:
    return [w for m in range(max_len + 1) for w in itertools.combinations_with_replacement(AXES, m)]


def coupling_table(offset: np.ndarray, side: float, omega: float, q: int, n_gauss: int) -> dict:
    """K[i,n; D; W](offset) for every sorted D (length <= q + 1) and sorted W (length <= q), i <= n."""
    x1, w1 = leggauss(n_gauss)
    x1, w1 = 0.5 * side * x1, 0.5 * side * w1
    xi = np.stack(np.meshgrid(x1, x1, x1, indexing="ij"), -1).reshape(-1, 3)
    wts = np.einsum("i,j,k->ijk", w1, w1, w1).ravel()
    arg = offset[None, :] - xi
    monos = {w: wts * (np.prod([xi[:, ax] for ax in w], axis=0) if w else 1.0) for w in sorted_indices(q)}
    table = {}
    for ds in sorted_indices(q + 1):
        for i in AXES:
            for n in range(i, 3):
                vals = green_derivative(i, n, ds, omega, arg)
                for w, mono in monos.items():
                    table[(i, n, ds, w)] = complex(np.sum(vals * mono))
    return table


def self_table(side: float, omega: float, q: int) -> dict:
    """The same at zero offset: (-1)^|D| times the distributional moment of the cube."""
    hier.scalar_moment = cube.cube_scalar_moment
    table = {}
    for ds in sorted_indices(q + 1):
        for w in sorted_indices(q):
            if (len(ds) + len(w)) % 2:
                continue  # odd grades vanish on a centrosymmetric cell
            for i in AXES:
                for n in range(i, 3):
                    table[(i, n, ds, w)] = (-1) ** len(ds) * hier.tensor_moment(side, omega, i, n, ds, w)
    return table


def block(table: dict, omega: float, contrast, q: int) -> np.ndarray:
    """The matrix that maps one voxel's unknowns to the scattered field's derivatives at a centre.

    Rows (i, P), columns (j, W'), both in the order i * nu + index; the field at the centre is
    u0 + block @ U of the source voxel.
    """
    idx = sorted_indices(q)
    pos = {w: n for n, w in enumerate(idx)}
    nu = len(idx)
    dc = hier.stiffness(contrast)
    a_rho = omega**2 * contrast.Drho
    out = np.zeros((3 * nu, 3 * nu), dtype=complex)

    def look(i, n, ds, w):
        key = (min(i, n), max(i, n), tuple(sorted(ds)), tuple(sorted(w)))
        return table.get(key, 0.0)

    for pi, p_idx in enumerate(idx):
        for i in AXES:
            row = i * nu + pi
            for wlen in range(q + 1):
                fact = math.factorial(wlen)
                for w in itertools.product(AXES, repeat=wlen):
                    col_w = pos[tuple(sorted(w))]
                    if a_rho != 0.0:
                        for j in AXES:
                            out[row, j * nu + col_w] += a_rho * look(i, j, p_idx, w) / fact
                    if wlen <= q - 1:
                        for k, n in itertools.product(AXES, repeat=2):
                            g = look(i, n, (*p_idx, k), w)
                            if g == 0.0:
                                continue
                            for r, j in itertools.product(AXES, repeat=2):
                                if dc[n, k, r, j] != 0.0:
                                    out[row, j * nu + pos[tuple(sorted((r, *w)))]] += (
                                        g * dc[n, k, r, j] / fact
                                    )
    return out


def solve_lattice(
    centres: np.ndarray, side: float, omega: float, contrast, q: int, k_hat, pol, n_gauss_near=20
):
    """Voxels of this side at these centres, uniform contrast: the Taylor coefficients of every voxel."""
    idx = sorted_indices(q)
    nu = len(idx)
    nc = len(centres)
    kp = omega / hier.REF.alpha
    big = np.eye(3 * nu * nc, dtype=complex)
    rhs = np.zeros(3 * nu * nc, dtype=complex)
    blocks: dict[tuple[int, int, int], np.ndarray] = {}
    own = block(self_table(side, omega, q), omega, contrast, q)
    for c, xc in enumerate(centres):
        phase = np.exp(1j * kp * (k_hat @ xc))
        for pi, p_idx in enumerate(idx):
            der = np.prod([1j * kp * k_hat[ax] for ax in p_idx]) if p_idx else 1.0
            for i in AXES:
                rhs[c * 3 * nu + i * nu + pi] = pol[i] * der * phase
        for c2, xc2 in enumerate(centres):
            if c2 == c:
                blk = own
            else:
                key = tuple(int(v) for v in np.rint((xc - xc2) / side))
                if key not in blocks:
                    near = max(abs(v) for v in key) <= 1
                    n_g = n_gauss_near if near else 8
                    blocks[key] = block(coupling_table(xc - xc2, side, omega, q, n_g), omega, contrast, q)
                blk = blocks[key]
            big[c * 3 * nu : (c + 1) * 3 * nu, c2 * 3 * nu : (c2 + 1) * 3 * nu] -= blk
    sol = np.linalg.solve(big, rhs).reshape(nc, 3, nu)
    return idx, sol


def lattice_far_field(centres, side, omega, contrast, idx, sol, obs: np.ndarray) -> np.ndarray:
    """Far field of all the voxels: each voxel's polynomial source, with the phase of its position."""
    hier.ball_quadrature = lambda radius, n=8: cube.cube_quadrature(radius, n)
    out = np.zeros((len(obs), 3), dtype=complex)
    for xc, s in zip(centres, sol, strict=True):
        one = hier.far_field(side, omega, contrast, idx, s, obs)
        for o, x in enumerate(obs):
            rh = x / np.linalg.norm(x)
            # far_field radiates from the origin; a voxel at xc adds the phase exp(-i k rh.xc)
            # for each wave type.
            # The two wave types have different k, so split the field into its P and S parts first.
            up = rh * (rh @ one[o])
            us = one[o] - up
            out[o] += up * np.exp(-1j * omega / hier.REF.alpha * (rh @ xc)) + us * np.exp(
                -1j * omega / hier.REF.beta * (rh @ xc)
            )
    return out
