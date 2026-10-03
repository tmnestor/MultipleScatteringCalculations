"""Derivatives of the elastodynamic Green's tensor to any order, accurate at every k r, and the moment
tables of the gradient hierarchy built from them with the cube's full symmetry.

THE DERIVATIVES.  With g_k = e^{ikr}/r and B = (g_S - g_P)/k_S^2,

    d_D G_in = (1 / 4 pi mu) [ delta_in d_D g_S + d_{D + e_i + e_n} B ],

so every derivative of G is a derivative of one of two radial functions. For a radial f, with the ladder
F_q = ((1/r) d/dr)^q f, each axis contributes a Hermite-like sum (d/ds = (1/r) d/dr, s = r^2/2):

    d^a f = sum_{k : 2 k_i <= a_i}  prod_i a_i! / (k_i! (a_i - 2 k_i)! 2^{k_i})  x^{a - 2k}  F_{|a| - |k|},

for the multi-index a = (a_0, a_1, a_2). The ladder comes from the same two branches as the 9 x 9 point
kernel (``kernel.py``): below |k_S| r = SERIES_LIMIT the power series, in which B's shared singularity
(t = 0) is never formed, so nothing cancels; above it the closed forms F_q = e^{ikr} p_q(ikr)/r^{2q+1}
with the reverse Bessel polynomials p_{q+1}(x) = (x - 2q - 1) p_q(x) + x p_q'(x).

``scalar_derivative_fields`` is the NumPy reference; ``derivatives_fortran`` transcribes it, and
``numerics.yml`` (``point_kernel.backend``) chooses between them.

THE MOMENT TABLES.  The hierarchy couples cell c' to the centre of cell c through

    T[i, n, D, W](o) = int_cube (d_D G_in)(o - xi) xi^W d xi ,

o = x_c - x_c'. A signed permutation Q (Q e_a = sigma_a e_{pi(a)}) maps the cube to itself and G to
Q G Q^T, and every index of T is a Cartesian index, so

    T(Q o)[pi(i), pi(n), pi(D), pi(W)] = sigma_i sigma_n sigma^D sigma^W  T(o)[i, n, D, W] .

One table is integrated for each orbit of offsets under the 48 signed permutations; the others follow
exactly (``table_for_offset``).
"""

import itertools
import math
from functools import cache

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray

from ..effective_contrasts import ReferenceMedium
from .kernel import N_SERIES, SERIES_LIMIT, falling, point_kernel_backend

AXES = (0, 1, 2)
Index = tuple[int, int, int]


# ------------------------------------------------------------------ multi-indices and their terms
@cache
def multi_indices(max_order: int) -> tuple[Index, ...]:
    """Every exponent triple a with |a| <= max_order, by order, then reverse lexicographically."""
    out: list[Index] = []
    for m in range(max_order + 1):
        out.extend((a[0], a[1], a[2]) for a in itertools.product(range(m, -1, -1), repeat=3) if sum(a) == m)
    return tuple(out)


@cache
def derivative_terms(a: Index) -> tuple[tuple[float, Index, int], ...]:
    """d^a f = sum (coefficient) x^exponent F_q, as (coefficient, exponent, q)."""
    terms: list[tuple[float, Index, int]] = []
    for k in itertools.product(*(range(ai // 2 + 1) for ai in a)):
        coef = 1.0
        for ai, ki in zip(a, k, strict=True):
            coef *= math.factorial(ai) / (math.factorial(ki) * math.factorial(ai - 2 * ki) * 2**ki)
        exponent = (a[0] - 2 * k[0], a[1] - 2 * k[1], a[2] - 2 * k[2])
        terms.append((coef, exponent, sum(a) - sum(k)))
    return tuple(terms)


@cache
def bessel_polynomials(q_max: int) -> NDArray:
    """p_q(x) = sum_j P[q, j] x^j for q = 0..q_max (integers).

    F_q[e^{ikr}/r] = e^{ikr} p_q(ikr)/r^{2q+1}.
    """
    p = np.zeros((q_max + 1, q_max + 1))
    p[0, 0] = 1.0
    for q in range(q_max):
        nxt = np.zeros(q_max + 1)
        nxt[1:] += p[q, :-1]  # x p_q
        nxt -= (2 * q + 1) * p[q]  # -(2q+1) p_q
        nxt[1:] += np.arange(1, q_max + 1) * p[q, 1:]  # x p_q'
        p[q + 1] = nxt
    return p


# ------------------------------------------------------------------ the two ladders
def radial_ladders(r: NDArray, omega: complex, ref: ReferenceMedium, q_max: int) -> tuple[NDArray, NDArray]:
    """F_q of g_S and of B = (g_S - g_P)/k_S^2 for q = 0..q_max at radii r > 0, shapes (q_max + 1, N)."""
    ks, kp = omega / ref.beta, omega / ref.alpha
    fs = np.zeros((q_max + 1, len(r)), dtype=complex)
    fb = np.zeros((q_max + 1, len(r)), dtype=complex)
    small = abs(ks) * r <= SERIES_LIMIT
    if small.any():
        rs = r[small]
        for t in range(N_SERIES):
            a_t = (1j * ks) ** t / math.factorial(t)
            b_t = ((1j * ks) ** t - (1j * kp) ** t) / (math.factorial(t) * ks**2)
            for q in range(q_max + 1):
                c = falling(t - 1, q)
                if c == 0.0:
                    continue
                power = c * rs ** float(t - 1 - 2 * q)
                fs[q, small] += a_t * power
                if t >= 1:  # t = 0 is the shared singularity, which B never forms
                    fb[q, small] += b_t * power
    large = ~small
    if large.any():
        rl = r[large]
        poly = bessel_polynomials(q_max)
        for k, sink in ((ks, "s"), (kp, "p")):
            x = 1j * k * rl
            phase = np.exp(x)
            for q in range(q_max + 1):
                val = phase * np.polynomial.polynomial.polyval(x, poly[q]) / rl ** (2 * q + 1)
                if sink == "s":
                    fs[q, large] = val
                    fb[q, large] += val / ks**2
                else:
                    fb[q, large] -= val / ks**2
    return fs, fb


def scalar_derivative_fields(
    X: NDArray, omega: complex, ref: ReferenceMedium, n_s: int, n_b: int
) -> tuple[NDArray, NDArray]:
    """d^a g_S for the first n_s and d^a B for the first n_b multi-indices of ``multi_indices``.

    Shapes (n_s, N) and (n_b, N), at separations X (N, 3), none zero. Dispatches on ``numerics.yml``.

    Raises:
        ValueError: at r = 0, where the derivatives are distributions.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if np.any(np.linalg.norm(X, axis=1) == 0.0):
        raise ValueError(
            "scalar_derivative_fields: r = 0 requested. The Green's tensor is a distribution at the "
            "origin; its cube moments come from the closed forms of the hierarchy, never from point values."
        )
    if point_kernel_backend() == "fortran":
        from .derivatives_fortran import scalar_derivative_fields_fortran  # local: it imports this module

        return scalar_derivative_fields_fortran(X, omega, ref, n_s, n_b)
    return scalar_derivative_fields_python(X, omega, ref, n_s, n_b)


def _order_for(n_idx: int) -> int:
    for m in range(64):
        if len(multi_indices(m)) >= n_idx:
            return m
    raise ValueError(f"{n_idx} multi-indices is beyond order 63")


def scalar_derivative_fields_python(
    X: NDArray, omega: complex, ref: ReferenceMedium, n_s: int, n_b: int
) -> tuple[NDArray, NDArray]:
    """The NumPy reference of ``scalar_derivative_fields``."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    order = _order_for(max(n_s, n_b))
    idx = multi_indices(order)
    fs, fb = radial_ladders(np.linalg.norm(X, axis=1), omega, ref, order)
    powers = [np.stack([X[:, ax] ** e for e in range(order + 1)]) for ax in AXES]

    def field(ladder: NDArray, a: Index) -> NDArray:
        out = np.zeros(len(X), dtype=complex)
        for coef, (e0, e1, e2), q in derivative_terms(a):
            out += coef * powers[0][e0] * powers[1][e1] * powers[2][e2] * ladder[q]
        return out

    s = np.stack([field(fs, a) for a in idx[:n_s]]) if n_s else np.zeros((0, len(X)), complex)
    b = np.stack([field(fb, a) for a in idx[:n_b]]) if n_b else np.zeros((0, len(X)), complex)
    return s, b


def green_derivative_fields(X: NDArray, omega: complex, ref: ReferenceMedium, max_order: int) -> NDArray:
    """d_D G_in at separations X (N, 3) for every D in ``multi_indices(max_order)``.

    Shape (3, 3, n_D, N).
    """
    d_list = multi_indices(max_order)
    b_list = multi_indices(max_order + 2)
    s, b = scalar_derivative_fields(X, omega, ref, len(d_list), len(b_list))
    b_pos = {a: n for n, a in enumerate(b_list)}
    out = np.empty((3, 3, len(d_list), s.shape[1]), dtype=complex)
    for i in AXES:
        for n in range(i, 3):
            shift = [0, 0, 0]
            shift[i] += 1
            shift[n] += 1
            rows = [b_pos[(d[0] + shift[0], d[1] + shift[1], d[2] + shift[2])] for d in d_list]
            val = b[rows]
            if i == n:
                val = val + s
            out[i, n] = out[n, i] = val
    return out / (4.0 * np.pi * ref.mu)


# ------------------------------------------------------------------ the moment tables
def as_exponents(sorted_axes: tuple[int, ...]) -> Index:
    """(0, 0, 2) -> (2, 0, 1): a sorted tuple of axes as an exponent triple."""
    return (sorted_axes.count(0), sorted_axes.count(1), sorted_axes.count(2))


def cell_rule(side: float, n_gauss: int) -> tuple[NDArray, NDArray]:
    """Tensor Gauss rule on the cube [-side/2, side/2]^3: points (G, 3) and weights (G,)."""
    x1, w1 = leggauss(n_gauss)
    x1, w1 = 0.5 * side * x1, 0.5 * side * w1
    pts = np.stack(np.meshgrid(x1, x1, x1, indexing="ij"), -1).reshape(-1, 3)
    return pts, np.einsum("i,j,k->ijk", w1, w1, w1).ravel()


def moment_table(
    offset: NDArray,
    side: float,
    omega: complex,
    ref: ReferenceMedium,
    d_list: list[Index],
    w_list: list[Index],
    n_gauss: int,
) -> NDArray:
    """T[i, n, d, w] = int_cube (d_D G_in)(offset - xi) xi^W d xi by a tensor Gauss rule, (3, 3, nD, nW).

    For an offset of at least one cell, where the integrand is smooth. D and W are exponent triples.
    """
    pts, wts = cell_rule(side, n_gauss)
    monos = np.stack([wts * np.prod(pts ** np.array(w, float), axis=1) for w in w_list])  # (nW, G)
    max_order = max(sum(d) for d in d_list)
    full = green_derivative_fields(np.asarray(offset, float)[None, :] - pts, omega, ref, max_order)
    pos = {a: n for n, a in enumerate(multi_indices(max_order))}
    vals = full[:, :, [pos[d] for d in d_list]]  # (3, 3, nD, G)
    return vals @ monos.T


@cache
def signed_permutations() -> tuple[tuple[tuple[int, int, int], tuple[float, float, float]], ...]:
    """The 48 signed permutations as (pi, sigma): Q e_a = sigma_a e_{pi(a)}."""
    return tuple(
        ((perm[0], perm[1], perm[2]), (signs[0], signs[1], signs[2]))
        for perm in itertools.permutations(AXES)
        for signs in itertools.product((1.0, -1.0), repeat=3)
    )


def apply(pi: Index, sigma: tuple[float, ...], v: Index) -> Index:
    """The integer vector Q v."""
    out = [0, 0, 0]
    for a in AXES:
        out[pi[a]] = int(sigma[a] * v[a])
    return (out[0], out[1], out[2])


def canonical_offset(offset: Index) -> Index:
    """The orbit's representative: absolute values in decreasing order."""
    a, b, c = sorted((abs(v) for v in offset), reverse=True)
    return (a, b, c)


def mapping_to(offset: Index) -> tuple[Index, tuple[float, ...]]:
    """A signed permutation (pi, sigma) that carries ``canonical_offset(offset)`` to ``offset``."""
    rep = canonical_offset(offset)
    for pi, sigma in signed_permutations():
        if apply(pi, sigma, rep) == tuple(offset):
            return pi, sigma
    raise AssertionError(f"no signed permutation maps {rep} to {offset}")


def _inverse_perm(pi: Index, a: Index) -> Index:
    """Source exponents of the target exponents a: source axis ax is target axis pi[ax]."""
    return (a[pi[0]], a[pi[1]], a[pi[2]])


@cache
def _index_maps(
    pi: Index, sigma: tuple[float, ...], d_list: tuple[Index, ...], w_list: tuple[Index, ...]
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    """Gather indices and signs that realise T(Q o) from T(o): target index t reads source pi^-1(t)."""
    d_pos = {d: n for n, d in enumerate(d_list)}
    w_pos = {w: n for n, w in enumerate(w_list)}
    inv = [0, 0, 0]
    for a in AXES:
        inv[pi[a]] = a

    def sign_of(a: Index) -> float:
        return float(np.prod([sigma[ax] ** a[ax] for ax in AXES]))

    src_d = [_inverse_perm(pi, d) for d in d_list]
    src_w = [_inverse_perm(pi, w) for w in w_list]
    return (
        np.array(inv),
        np.array([d_pos[d] for d in src_d]),
        np.array([w_pos[w] for w in src_w]),
        np.array([sigma[inv[t]] for t in AXES]),
        np.einsum("d,w->dw", [sign_of(d) for d in src_d], [sign_of(w) for w in src_w]),
    )


def transform_table(table: NDArray, pi: Index, sigma: tuple[float, ...], d_list, w_list) -> NDArray:
    """T(Q o) from T(o), exactly (index gathers and signs)."""
    src_axis, src_d, src_w, sign_axis, sign_dw = _index_maps(pi, tuple(sigma), tuple(d_list), tuple(w_list))
    out = table[np.ix_(src_axis, src_axis, src_d, src_w)]
    return out * sign_axis[:, None, None, None] * sign_axis[None, :, None, None] * sign_dw[None, None]


def table_for_offset(offset: Index, canonical_tables: dict[Index, NDArray], d_list, w_list) -> NDArray:
    """T(offset) from the table of its orbit's representative."""
    pi, sigma = mapping_to(offset)
    return transform_table(canonical_tables[canonical_offset(offset)], pi, sigma, d_list, w_list)


@cache
def _block_maps(
    pi: Index, sigma: tuple[float, ...], v_list: tuple[Index, ...], u_list: tuple[Index, ...]
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Gathers and signs for a block whose rows and columns are (axis, exponent) pairs, axis-major."""
    nu = len(u_list)
    u_pos = {u: n for n, u in enumerate(u_list)}
    v_pos = {v: n for n, v in enumerate(v_list)}
    inv = [0, 0, 0]
    for a in AXES:
        inv[pi[a]] = a

    def sign_of(a: Index) -> float:
        return float(np.prod([sigma[ax] ** a[ax] for ax in AXES]))

    src, sign = np.empty(3 * nu, dtype=int), np.empty(3 * nu)
    for t in AXES:
        for m, u in enumerate(u_list):
            su = _inverse_perm(pi, u)
            src[t * nu + m] = inv[t] * nu + u_pos[su]
            sign[t * nu + m] = sigma[inv[t]] * sign_of(su)
    src_v = np.array([v_pos[_inverse_perm(pi, v)] for v in v_list])
    sign_v = np.array([sign_of(_inverse_perm(pi, v)) for v in v_list])
    return src, sign, src_v, sign_v


def transform_block(block: NDArray, pi: Index, sigma: tuple[float, ...], v_list, u_list) -> NDArray:
    """B(Q o) from B(o) for a block B[V, (i, P), (j, W)] (rows i * nu + P) built covariantly from T.

    Valid when everything else in the block is invariant under the cube group, as an isotropic contrast is.
    """
    src, sign, src_v, sign_v = _block_maps(pi, tuple(sigma), tuple(v_list), tuple(u_list))
    out = block[np.ix_(src_v, src, src)]
    return out * sign_v[:, None, None] * sign[None, :, None] * sign[None, None, :]
