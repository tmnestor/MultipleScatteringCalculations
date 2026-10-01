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

THE SOLVER (``solve_graded_octree``).  The leaves of the tree are cells of mixed sizes.  Each leaf n has
its own half-width h_n, its own projected contrast and its own Gram matrix, and the Galerkin system of
``solver.solve_graded_sphere`` becomes

    M(h_m) psi_m - sum_n K(m, n) E_n psi_n = <Q, psi0>_m,

with K(m, n) from ``octree_block``.  It is assembled densely: the convolution structure that the FFT
solver uses is lost when the cells differ in size.  On a uniform tree it is the uniform solver.
"""

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache

import numpy as np
import scipy.linalg
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray

from ..effective_contrasts import MaterialContrast, ReferenceMedium
from ..sphere_scattering import _plane_wave_strain_voigt
from ..sphere_scattering_fft import _build_grid_index_map
from .basis import (
    CONTRAST_NORMS,
    SOURCE_EXPONENTS_QUARTIC,
    contrast_values,
    gram_test,
    monomials,
    source_expansion,
    source_exponents,
)
from .blocks import coupling_block
from .farfield import radiate
from .site import cell_contrast_coefficients
from .solver import field_sizes, plane_wave_moments

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


# ---------------------------------------------------------------------------
# Leaves
# ---------------------------------------------------------------------------


def uniform_leaves(radius: float, n_sub: int) -> tuple[NDArray, NDArray]:
    """The uniform grid of ``solver.solve_graded_sphere`` as leaves: (centres (N, 3), half-widths (N,)).

    Every cell of the n_sub^3 grid over [-radius, radius]^3 that overlaps the sphere is kept.
    """
    h_cell = radius / n_sub
    _, centres, h = _build_grid_index_map(
        radius, n_sub, lambda q: bool(np.linalg.norm(q) < radius + np.sqrt(3.0) * h_cell)
    )
    return centres, np.full(len(centres), h)


def refine_leaves(centres: NDArray, half_widths: NDArray, flag: NDArray) -> tuple[NDArray, NDArray]:
    """Replace each flagged leaf by its eight children and keep the others: (centres, half-widths)."""
    signs = np.array(list(itertools.product((-1.0, 1.0), repeat=3)))
    out_c, out_h = [centres[~flag]], [half_widths[~flag]]
    for c, h in zip(centres[flag], half_widths[flag], strict=True):
        out_c.append(c + 0.5 * h * signs)
        out_h.append(np.full(8, 0.5 * h))
    return np.concatenate(out_c), np.concatenate(out_h)


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------


@dataclass
class OctreeResult:
    """Solution of the Galerkin system on the leaves of an octree, for one plane wave."""

    centres: NDArray  # (N, 3)
    half_widths: NDArray  # (N,)
    omega: float
    ref: ReferenceMedium
    delta: NDArray  # (N, 4 or 10, 9, 9) contrast operator coefficients per leaf
    psi: NDArray  # (N, 4 or 10, 9) coefficients of the state per leaf
    p: int
    r: int


def solve_graded_octree(
    omega: float,
    ref: ReferenceMedium,
    contrast: MaterialContrast,
    centres: NDArray,
    half_widths: NDArray,
    profile: Callable[[NDArray], float],
    k_hat: NDArray,
    pol: NDArray,
    wave_type: str,
    p: int = 1,
    r: int = 1,
) -> OctreeResult:
    """Solve the Galerkin system on leaves of mixed sizes, densely.

    Args:
        omega: Angular frequency (rad/s).
        ref: Background medium.
        contrast: The contrast that ``profile`` scales.
        centres: Leaf centres, shape (N, 3) (m).
        half_widths: Leaf half-widths, shape (N,) (m), in ratios 2^j.
        profile: Scalar factor of the contrast as a function of position.
        k_hat: Incident direction.
        pol: Incident polarisation.
        wave_type: 'P' or 'S' (the incident speed).
        p: Field degree (0, 1 or 2).
        r: Contrast degree (0, 1 or 2).
    """
    centres = np.asarray(centres, dtype=float)
    half_widths = np.asarray(half_widths, dtype=float)
    n = len(centres)
    na, n_field, _, n_source = field_sizes(p, r)
    delta = np.array(
        [
            cell_contrast_coefficients(profile, c, float(h), contrast, ref, omega, degree=r)
            for c, h in zip(centres, half_widths, strict=True)
        ]
    )
    rows = na * 9
    a = np.zeros((n * rows, n * rows), dtype=complex)
    h_min = float(half_widths.min())
    cache_k: dict[tuple, NDArray] = {}
    expansions = [
        source_expansion(d, n_field)[:, :na].transpose(0, 2, 1, 3).reshape(n_source * 9, rows)
        for d in delta
    ]
    for col in range(n):
        for row in range(n):
            rel = np.round((centres[row] - centres[col]) / h_min).astype(int)
            key = (float(half_widths[row]), float(half_widths[col]), int(rel[0]), int(rel[1]), int(rel[2]))
            if key not in cache_k:
                blk = octree_block(
                    (0.0, 0.0, 0.0),
                    key[0],
                    (-rel[0] * h_min, -rel[1] * h_min, -rel[2] * h_min),
                    key[1],
                    omega,
                    ref,
                    n_source,
                    n_field,
                )
                cache_k[key] = blk[:na].transpose(0, 2, 1, 3).reshape(rows, n_source * 9)
            entry = -(cache_k[key] @ expansions[col])
            if row == col:
                entry = entry + np.kron(gram_test(float(half_widths[row]), n_field)[:na, :na], np.eye(9))
            a[row * rows : (row + 1) * rows, col * rows : (col + 1) * rows] = entry
    k_mag = omega / (ref.alpha if wave_type == "P" else ref.beta)
    k_hat = np.asarray(k_hat, dtype=float) / np.linalg.norm(k_hat)
    amp = np.concatenate([np.asarray(pol, dtype=complex), _plane_wave_strain_voigt(k_hat, pol, k_mag)])
    rhs = np.zeros((n, rows), dtype=complex)
    for h in np.unique(half_widths):
        sel = half_widths == h
        rhs[sel] = plane_wave_moments(centres[sel], float(h), k_mag * k_hat, amp, n_field)[:, :na].reshape(
            -1, rows
        )
    sol = scipy.linalg.solve(a, rhs.ravel(), overwrite_a=True, check_finite=False).reshape(n, na, 9)
    full = np.zeros((n, n_field, 9), dtype=complex)
    full[:, :na] = sol
    return OctreeResult(centres, half_widths, omega, ref, delta, full, p, r)


def octree_far_field(
    res: OctreeResult, directions: NDArray, r_distance: float, n_gauss: int = 4
) -> tuple[NDArray, NDArray]:
    """Far field (u_P, u_S) of the leaves: each leaf's polynomial source radiated from Gauss nodes."""
    x, w = leggauss(n_gauss)
    xi = np.stack(np.meshgrid(x, x, x, indexing="ij"), -1).reshape(-1, 3)
    ww = np.einsum("i,j,k->ijk", w, w, w).ravel()
    ms = monomials(SOURCE_EXPONENTS_QUARTIC, xi)
    pts, srcs = [], []
    for c, h, d, psi in zip(res.centres, res.half_widths, res.delta, res.psi, strict=True):
        coef = np.einsum("cbij,bj->ci", source_expansion(d, psi.shape[0]), psi)
        srcs.append((ms[: len(coef)].T @ coef) * (ww * h**3)[:, None])
        pts.append(c + h * xi)
    return radiate(np.concatenate(pts), np.concatenate(srcs), res.omega, res.ref, directions, r_distance)
