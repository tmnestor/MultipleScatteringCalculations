"""Cells of different sizes on an octree: coupling blocks by the two-scale relation.

A cube of half-width H divides into 8^j cubes of half-width h = H / 2^j.  On a descendant with centre
c_d, the parent's local coordinate is xi_parent = (h / H) xi_child + (c_d - c_parent) / H, and the
polynomials of the parent are polynomials of the same degree in the child's coordinate:

    Q_a(xi_parent) = sum_b C[a, b] Q_b(xi_child)        (``field_reexpansion``),
    m_c(xi_parent) = sum_d D[c, d] m_d(xi_child)        (``source_reexpansion``).

The block between a field cell and a source cell of different sizes is therefore an exact, finite sum of
blocks between EQUAL cells: the larger cell is replaced by its descendants at the smaller cell's size,

    K_ac(large field, small source) = sum over descendants d of  sum_b C_d[a, b] K_bc(d, source),
    K_ac(small field, large source) = sum over descendants d of  sum_e K_ae(field, d) D_d[c, e].

No new integral is needed: every block on the octree comes from the equal-cell blocks of ``blocks``
(closed forms for touching cells, quadrature or the multipole series otherwise).  The cells must be
aligned as octree cells are: sizes in a ratio 2^j, and the larger cell's descendants on the smaller
cell's lattice.
"""

import itertools
import math
from functools import cache

import numpy as np
from numpy.typing import NDArray

from ..effective_contrasts import ReferenceMedium
from .basis import CONTRAST_NORMS, contrast_values, source_exponents
from .blocks import coupling_block

Triple = tuple[float, float, float]


@cache
def field_reexpansion(n_field: int, scale: float, shift: Triple) -> NDArray:
    """C[a, b] with Q_a(scale * xi + shift) = sum_b C[a, b] Q_b(xi), shape (n_field, n_field).

    By projection onto the orthogonal Q_b with a Gauss rule that is exact for the products.
    """
    x, w = np.polynomial.legendre.leggauss(4)
    xi = np.stack([g.ravel() for g in np.meshgrid(x, x, x, indexing="ij")], axis=1)
    wt = (w[:, None, None] * w[None, :, None] * w[None, None, :]).ravel()
    child = contrast_values(xi)[:n_field]
    parent = contrast_values(scale * xi + np.asarray(shift))[:n_field]
    return (parent * wt) @ child.T / np.array(CONTRAST_NORMS[:n_field])


@cache
def source_reexpansion(n_source: int, scale: float, shift: Triple) -> NDArray:
    """D[c, d] with m_c(scale * xi + shift) = sum_d D[c, d] m_d(xi), shape (n_source, n_source).

    Exactly, by the binomial expansion of (scale * xi_i + shift_i)^e on each axis.
    """
    exps = source_exponents(n_source)
    index = {e: k for k, e in enumerate(exps)}
    out = np.zeros((n_source, n_source))
    for c, e in enumerate(exps):
        axes = [
            [math.comb(e[i], j) * scale**j * shift[i] ** (e[i] - j) for j in range(e[i] + 1)]
            for i in range(3)
        ]
        for j0, j1, j2 in itertools.product(*(range(len(a)) for a in axes)):
            out[c, index[(j0, j1, j2)]] += axes[0][j0] * axes[1][j1] * axes[2][j2]
    return out


def _levels_apart(h_large: float, h_small: float) -> int:
    ratio = h_large / h_small
    j = round(math.log2(ratio))
    if j < 0 or abs(2.0**j - ratio) > 1e-9 * ratio:
        raise ValueError(
            f"octree_block: the cell sizes {h_large} and {h_small} are not in a ratio that is a power "
            f"of two "
            f"(ratio {ratio:.6g}). Fix: use cells of an octree, half-widths H / 2^j."
        )
    return j


def _descendants(centre: NDArray, h_large: float, j: int) -> tuple[NDArray, float]:
    """Centres of the 8^j descendants of half-width h_large / 2^j, shape (8^j, 3), and that half-width."""
    m = 2**j
    h = h_large / m
    steps = (2 * np.arange(m) + 1 - m) * h
    grid = np.stack([g.ravel() for g in np.meshgrid(steps, steps, steps, indexing="ij")], axis=1)
    return centre + grid, h


def _offset(c_field: NDArray, c_source: NDArray, h: float) -> tuple[int, int, int]:
    off = (c_field - c_source) / (2.0 * h)
    nearest = np.round(off)
    if np.abs(off - nearest).max() > 1e-9:
        raise ValueError(
            f"octree_block: the cells are not aligned as octree cells: the separation {c_field - c_source} "
            f"is not a whole number of cell widths {2.0 * h}. Fix: place cell centres on the lattice "
            "of the octree."
        )
    return (int(nearest[0]), int(nearest[1]), int(nearest[2]))


def octree_block(
    c_field: Triple,
    h_field: float,
    c_source: Triple,
    h_source: float,
    omega: float,
    ref: ReferenceMedium,
    n_source: int = 10,
    n_test: int = 4,
) -> NDArray:
    """K[a, c] between two octree cells of any two sizes, shape (n_test, n_source, 9, 9).

    Args:
        c_field: Centre of the field cell (m).
        h_field: Its half-width (m).
        c_source: Centre of the source cell (m).
        h_source: Its half-width (m).
        omega: Angular frequency (rad/s).
        ref: Background medium.
        n_source: Source monomials (10, 20 or 35).
        n_test: Field functions (4 or 10).

    Raises:
        ValueError: when the sizes are not in a ratio 2^j or the cells are not on a common lattice.
    """
    ct, cs = np.asarray(c_field, dtype=float), np.asarray(c_source, dtype=float)
    if h_field >= h_source:
        j = _levels_apart(h_field, h_source)
        if j == 0:
            return coupling_block(_offset(ct, cs, h_source), h_source, omega, ref, n_source, n_test)
        kids, h = _descendants(ct, h_field, j)
        out = np.zeros((n_test, n_source, 9, 9), dtype=complex)
        for kid in kids:
            shift = (kid - ct) / h_field
            c_mat = field_reexpansion(
                n_test, h / h_field, (float(shift[0]), float(shift[1]), float(shift[2]))
            )
            out += np.einsum(
                "ab,bcij->acij", c_mat, coupling_block(_offset(kid, cs, h), h, omega, ref, n_source, n_test)
            )
        return out
    j = _levels_apart(h_source, h_field)
    kids, h = _descendants(cs, h_source, j)
    out = np.zeros((n_test, n_source, 9, 9), dtype=complex)
    for kid in kids:
        shift = (kid - cs) / h_source
        d_mat = source_reexpansion(
            n_source, h / h_source, (float(shift[0]), float(shift[1]), float(shift[2]))
        )
        out += np.einsum(
            "aeij,ce->acij", coupling_block(_offset(ct, kid, h), h, omega, ref, n_source, n_test), d_mat
        )
    return out
