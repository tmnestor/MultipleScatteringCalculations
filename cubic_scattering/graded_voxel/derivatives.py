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
from collections.abc import Callable
from functools import cache, lru_cache

import numpy as np
from numpy.polynomial import Polynomial, legendre
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray

from ..effective_contrasts import ReferenceMedium
from .kernel import N_SERIES, SERIES_LIMIT, falling, point_kernel_backend
from .legendre_moments import (
    box_moment_derivative_magnitudes,
    box_moment_derivatives,
    box_moments,
    point_derivative,
    point_derivative_magnitude,
)

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


#: the largest ratio of a sub-cube's extent to its distance from the field point in moment_table_series
RHO_MAX = 0.3


def _cube_monomial_moments(alphas: list[Index], half_width: float) -> NDArray:
    """int over [-h, h]^3 of eta^alpha, exactly: prod_k 2 h^(a_k + 1) / (a_k + 1) for even a_k, else 0."""
    out = np.ones(len(alphas))
    for k in range(3):
        a = np.array([al[k] for al in alphas], dtype=float)
        out *= np.where(a % 2 == 0, 2.0 * half_width ** (a + 1.0) / (a + 1.0), 0.0)
    return out


def moment_table_series(
    offset: NDArray,
    side: float,
    omega: complex,
    ref: ReferenceMedium,
    d_list: list[Index],
    w_list: list[Index],
    tol: float,
) -> NDArray:
    """T[i, n, d, w] = int_cube (d_D G_in)(offset - xi) xi^W d xi with no quadrature, shape (3, 3, nD, nW).

    The moment hierarchy's table between two different cells: the field point ``offset`` (the centre of
    another cell) lies at least half a cell outside the source cube, so the integrand is smooth and the
    integral is a convergent multipole series. The source cube is cut into m^3 equal sub-cubes, m the
    least for which every sub-cube's extent over its distance, rho = sqrt(3) h_s / |offset - c|, is at most
    RHO_MAX; about each sub-cube centre c,

        int_sub (d_D G)(X - eta) (c + eta)^W d eta
            = sum_V binom(W, V) c^(W - V) sum_gamma (-1)^|gamma| / gamma!  d^(D + gamma) G(X)  M(V + gamma),

    X = offset - c, with M the exact monomial moments of the sub-cube and the derivatives of the Green's
    tensor at X from ``green_derivative_fields``. The term of order N is bounded by
    sum_j rho^(N - j) kappa^j / j!, kappa = sqrt(3) |k_S| h_s, and N is the least order whose next bound is
    below the tolerance.

    Raises:
        ValueError: for the self cell (offset at the origin) or a field point inside the source cube, where
            the integral is a distribution: its moments are the hierarchy's closed forms.
    """
    o = np.asarray(offset, dtype=float)
    h = side / 2.0
    if np.max(np.abs(o)) < h * (1.0 + 1e-12):
        raise ValueError(
            "moment_table_series: the field point lies in the source cube (the self cell). There the table "
            "is "
            "a distribution, given by the closed-form moments of the hierarchy; this series is for two "
            f"different cells. Got offset {tuple(o)} for a cube of side {side}."
        )
    m = 1
    while True:
        h_s = h / m
        grid = (np.arange(m) - (m - 1) / 2.0) * 2.0 * h_s
        centres = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), -1).reshape(-1, 3)
        dist = np.linalg.norm(o[None, :] - centres, axis=1)
        rho = float(np.max(np.sqrt(3.0) * h_s / dist))
        if rho <= RHO_MAX:
            break
        m += 1
    # the bound of the term of order n, b_n = sum_j rho^(n - j) kappa^j / j! with kappa = sqrt(3) |k_S| h_s:
    # geometric in the ratio and factorial in the sub-cube's size over the wavelength (as multipole.py)
    kappa = math.sqrt(3.0) * abs(omega / ref.beta) * h_s
    order = 2
    while sum(rho ** (order + 1 - j) * kappa**j / math.factorial(j) for j in range(order + 2)) >= tol:
        order += 1
    max_d = max(sum(d) for d in d_list)
    max_w = max(sum(w) for w in w_list)
    gammas = list(multi_indices(order))
    v_list = list(multi_indices(max_w))
    full_idx = multi_indices(max_d + order)
    pos = {a: k for k, a in enumerate(full_idx)}
    # C[gamma, V] = (-1)^|gamma| / gamma! M(V + gamma)
    sign_fact = np.array([(-1.0) ** sum(g) / math.prod(math.factorial(x) for x in g) for g in gammas])
    cmat = np.empty((len(gammas), len(v_list)))
    for jv, v in enumerate(v_list):
        alphas = [(g[0] + v[0], g[1] + v[1], g[2] + v[2]) for g in gammas]
        cmat[:, jv] = sign_fact * _cube_monomial_moments(alphas, h_s)
    gather = np.array([[pos[(d[0] + g[0], d[1] + g[1], d[2] + g[2])] for g in gammas] for d in d_list])
    vals = green_derivative_fields(o[None, :] - centres, omega, ref, max_d + order)  # (3, 3, n_full, P)
    out = np.zeros((3, 3, len(d_list), len(w_list)), dtype=complex)
    for p, c in enumerate(centres):
        s_dv = np.einsum("indg,gv->indv", vals[:, :, gather, p], cmat)  # (3, 3, nD, nV)
        bmat = np.zeros((len(w_list), len(v_list)))
        for jw, w in enumerate(w_list):
            for jv, v in enumerate(v_list):
                if all(v[k] <= w[k] for k in range(3)):
                    bmat[jw, jv] = math.prod(
                        math.comb(w[k], v[k]) * c[k] ** (w[k] - v[k]) for k in range(3)
                    )
        out += np.einsum("indv,wv->indw", s_dv, bmat)
    return out


#: the highest power of k kept in the stored coefficients of moment_table_kseries
KSERIES_T_MAX = 48
#: Gauss points per axis for those coefficients, by the shape of the offset: face, edge and corner
#: neighbours, then every cell farther away. Each reaches 1e-14 against 36 points over t = 0 .. 19 and every
#: D, W of the third-gradient system (|D|, |W| <= 4): the face neighbour, whose field point is half a cell
#: from the source cube, converges slowest (2e-12 at 24 points); (2, 1, 0) and beyond are at 5e-15 with 12.
#: That calibration covers t <= 19 only: the high powers are steep, d^a r^(t - 1) xi^W behaving like a
#: polynomial of degree about t - 1 + max W per axis, so the rule is also never smaller than the one exact
#: for that degree (``kseries_gauss_order``). Without that bound the 12 points left the coefficients at
#: (2, 1, 0) wrong by 2e-13 of each order's largest entry at t = 48.
KSERIES_GAUSS = {"face": 28, "edge": 20, "corner": 14, "far": 12}


def kseries_gauss_points(offset_units: Index) -> int:
    """Gauss points per axis for the k-series coefficients of this offset (KSERIES_GAUSS)."""
    a = sorted((abs(o) for o in offset_units), reverse=True)
    if a[0] > 1:
        return KSERIES_GAUSS["far"]
    return KSERIES_GAUSS[("face", "edge", "corner")[sum(a) - 1]]


def kseries_gauss_order(offset_units: Index, w_list: tuple[Index, ...]) -> int:
    """Gauss points per axis for the k-series coefficients: the calibrated rule of the offset, and at least
    the rule exact for the polynomial degree KSERIES_T_MAX - 1 + max W of the steepest power."""
    max_w = max(max(w) for w in w_list)
    return max(kseries_gauss_points(offset_units), (KSERIES_T_MAX - 1 + max_w) // 2 + 1)


@cache
def kseries_coefficients(
    offset_units: Index, d_list: tuple[Index, ...], w_list: tuple[Index, ...]
) -> tuple[NDArray, tuple[Index, ...]]:
    """U[t, a, w] = int_cube d^a r^(t - 1) (o - xi) xi^W d xi on the UNIT cube, o = offset_units.

    For t = 0 .. KSERIES_T_MAX and every multi-index a of order up to max |D| + 2 (the extra two are the
    derivatives that turn B into the Green's tensor). The integrand is smooth for two different cells, and
    a tensor Gauss rule of ``kseries_gauss_order(offset, w_list)`` points integrates it to round-off.
    Independent of the frequency and of the cell size, so computed once per offset. Returns (U, the list of
    a).
    """
    o = np.asarray(offset_units, dtype=float)
    pts, wts = cell_rule(1.0, kseries_gauss_order(offset_units, w_list))
    x = o[None, :] - pts
    r = np.linalg.norm(x, axis=1)
    max_a = max(sum(d) for d in d_list) + 2
    a_list = multi_indices(max_a)
    monos = np.stack([wts * np.prod(pts ** np.array(w, float), axis=1) for w in w_list])  # (nW, G)
    xpow = [np.stack([x[:, ax] ** e for e in range(max_a + 1)]) for ax in range(3)]
    out = np.zeros((KSERIES_T_MAX + 1, len(a_list), len(w_list)))
    for t in range(KSERIES_T_MAX + 1):
        m = t - 1
        rpow: dict[int, NDArray] = {}
        fields = np.zeros((len(a_list), len(r)))
        for ia, a in enumerate(a_list):
            acc = np.zeros(len(r))
            for coef, (e0, e1, e2), q in derivative_terms(a):
                c = coef * falling(m, q)
                if c == 0.0:
                    continue
                p = m - 2 * q
                if p not in rpow:
                    rpow[p] = r ** float(p)
                acc += c * xpow[0][e0] * xpow[1][e1] * xpow[2][e2] * rpow[p]
            fields[ia] = acc
        out[t] = fields @ monos.T
    return out, a_list


#: Legendre indices kept beyond the degree of the weight in the closed coefficients' box moments, as for
#: the touching Galerkin pieces (``blocks.LEGENDRE_MARGIN``) ...
KSERIES_LEGENDRE_MARGIN = 8
#: ... plus one more for every KSERIES_LEGENDRE_GROWTH of the power m: a steep power such as |x|^47 needs a
#: longer truncation of the boundary-value system (at 13 indices its face moments are wrong by 4.5e-9, at 23
#: they are at round-off; |x|^21 is at round-off with 13)
KSERIES_LEGENDRE_GROWTH = 4
#: ... and KSERIES_LEGENDRE_PER_DERIVATIVE more for each derivative left on the fixed axes, which sharpens
#: the kernel: on the near face of the face neighbour the moments of d_z^4 (1/r) are wrong by 5e-12 at 13
#: indices and at round-off at 24; d_z^5 (1/r) by 6e-10 at 13 and 2e-14 at 24 and at 36
KSERIES_LEGENDRE_PER_DERIVATIVE = 3


@cache
def _free_moments(
    offset_units: Index,
    free: tuple[int, ...],
    z_fixed: tuple[float, ...],
    b_fixed: tuple[int, ...],
    m: int,
    max_w: int,
) -> tuple[NDArray, NDArray]:
    """J[w] = int over the free axes of [-1/2, 1/2]^d of (d^b |x|^m) prod_j xi_j^(w_j), (max_w + 1,)^d.

    x = o - xi on the free axes; the fixed axes hold the coordinates z_fixed, and the derivatives b_fixed
    act along them only. The free axes are cut at xi_j = 0 into half-cubes that never straddle x_j = 0;
    each, reflected to x_j >= 0 and scaled by x = t / 2, is the unit box prod [l_j, l_j + 1], on which
    xi_j^w is expanded exactly in Legendre polynomials of the box coordinate u_j = 2 (t_j - l_j) - 1 and
    contracted with the Legendre moments of d_z^b S^m, S^2 = |z|^2 + |t|^2, z = 2 z_fixed
    (``legendre_moments.box_moment_derivatives``): the derivative is carried by the relation, never
    expanded into its terms. Returns (J, its running error magnitude), the second from
    ``legendre_moments.box_moment_derivative_magnitudes`` contracted with the absolute weights.
    """
    d = len(free)
    if d == 0:
        return (
            np.array(point_derivative(z_fixed, b_fixed, m)),
            np.array(point_derivative_magnitude(z_fixed, b_fixed, m)),
        )
    n_leg = (
        max_w
        + 1
        + KSERIES_LEGENDRE_MARGIN
        + max(0, m) // KSERIES_LEGENDRE_GROWTH
        + KSERIES_LEGENDRE_PER_DERIVATIVE * sum(b_fixed)
    )
    z_box = tuple(2.0 * v for v in z_fixed)
    out = np.zeros((max_w + 1,) * d)
    mag = np.zeros((max_w + 1,) * d)
    spec = {1: "ax,x->a", 2: "ax,by,xy->ab", 3: "ax,by,cz,xyz->abc"}[d]
    for half in itertools.product((-1, 1), repeat=d):
        lows, legs = [], []
        for j, side in zip(free, half, strict=True):
            o = offset_units[j]
            ends = (o - 0.5 * (side > 0), o + 0.5 * (side < 0))  # x_j over this half
            sign = 1.0 if ends[0] >= 0 else -1.0
            low = 2.0 * min(abs(ends[0]), abs(ends[1]))
            lows.append(low)
            x_of_u = Polynomial([sign * (2.0 * low + 1.0) / 4.0, sign / 4.0])
            xi_of_u = o - x_of_u
            leg = np.zeros((max_w + 1, n_leg))
            for w in range(max_w + 1):
                coefs = legendre.poly2leg((xi_of_u**w).coef)
                leg[w, : len(coefs)] = coefs
            legs.append(leg)
        lam = box_moment_derivatives(tuple(lows), z_box, b_fixed, m, n_leg)
        lam_mag = box_moment_derivative_magnitudes(tuple(lows), z_box, b_fixed, m, n_leg)
        # d_x^b |x|^m = 2^|b| d_z^b (S / 2)^m, d xi = 2^-d dt
        scale = 2.0 ** (sum(b_fixed) - m - d)
        out += scale * np.einsum(spec, *legs, lam, optimize=True)
        mag += scale * np.einsum(spec, *[np.abs(g) for g in legs], lam_mag, optimize=True)
    return out, mag


@cache
def _reduced_integral(
    offset_units: Index,
    free: tuple[tuple[int, int], ...],
    fixed: tuple[tuple[int, float, int], ...],
    m: int,
    max_w: int,
) -> tuple[float, float]:
    """int over the free axes of d^b r^m prod xi_k^(w_k), with the normal derivatives reduced first.

    ``free`` holds (axis, w) for each free axis, ``fixed`` holds (axis, z, b): the coordinate x = z of a
    fixed axis and the b derivatives along it. A normal derivative of high order is a small difference of
    the terms the Legendre relation subtracts (d_z^5 S vanishes at the foot of the normal), so before the
    relation is used the family r^m is reduced by its Laplacian, nabla^2 r^m = m (m + 1) r^(m - 2):

        d_f^2 r^m = m (m + 1) r^(m - 2) - sum_(g fixed, g != f) d_g^2 r^m - sum_(k free) d_k^2 r^m.

    Each tangential d_k^2 is moved onto xi_k^w by parts over xi_k in [-h, h] (with d/d xi_k = -d/d x_k):

        int (d_k^2 F) xi^w = -[(d_k F) xi^w] - w [F xi^(w-1)] + w (w - 1) int F xi^(w-2),

    the brackets being edges or corners with one or no derivative along k. The reduction is applied to the
    fixed axis f of largest order b_f >= 2 when every other fixed axis has b_g <= b_f - 3, so that the
    exchange term d_g^2 lowers the largest order and the recursion ends; what is left goes to the relation
    (``_free_moments``), and a point (no free axis) to its value.

    Returns (value, running error magnitude): the magnitude is the sum of |coefficient| times the
    magnitude of each term, down to the moments, so that its ratio to |value| counts every cancellation of
    the evaluation, not only the outermost one.
    """
    h = 0.5
    z_fixed = tuple(z for _, z, _ in fixed)
    b_fixed = tuple(b for _, _, b in fixed)
    if not free:
        return point_derivative(z_fixed, b_fixed, m), point_derivative_magnitude(z_fixed, b_fixed, m)
    f = int(np.argmax(b_fixed)) if fixed else 0
    b_f = b_fixed[f] if fixed else 0
    if b_f >= 2 and all(b <= b_f - 3 for i, b in enumerate(b_fixed) if i != f):
        base = list(fixed)
        base[f] = (fixed[f][0], fixed[f][1], b_f - 2)
        terms: list[tuple[float, tuple[float, float]]] = []
        if m * (m + 1) != 0:
            terms.append((m * (m + 1), _reduced_integral(offset_units, free, tuple(base), m - 2, max_w)))
        for g in range(len(fixed)):
            if g != f:
                swapped = list(base)
                swapped[g] = (fixed[g][0], fixed[g][1], fixed[g][2] + 2)
                terms.append((-1.0, _reduced_integral(offset_units, free, tuple(swapped), m, max_w)))
        for q, (k, w) in enumerate(free):
            rest = free[:q] + free[q + 1 :]
            if w >= 2:
                lowered = free[:q] + ((k, w - 2),) + free[q + 1 :]
                terms.append(
                    (-w * (w - 1), _reduced_integral(offset_units, lowered, tuple(base), m, max_w))
                )
            for side in (1, -1):
                z_k = offset_units[k] - side * h
                with_d = tuple(sorted((*base, (k, z_k, 1))))
                terms.append(
                    (side * (side * h) ** w, _reduced_integral(offset_units, rest, with_d, m, max_w))
                )
                if w >= 1:
                    plain = tuple(sorted((*base, (k, z_k, 0))))
                    terms.append(
                        (
                            side * w * (side * h) ** (w - 1),
                            _reduced_integral(offset_units, rest, plain, m, max_w),
                        )
                    )
        value = sum(c * v for c, (v, _) in terms)
        magnitude = sum(abs(c) * mg for c, (_, mg) in terms)
        return value, magnitude
    axes = tuple(k for k, _ in free)
    weights = tuple(w for _, w in free)
    j_val, j_mag = _free_moments(offset_units, axes, z_fixed, b_fixed, m, max_w)
    return float(j_val[weights]), float(j_mag[weights])


#: pieces of the expanded form kept at once: one offset needs at most 8 half-cubes x 31 powers, and each
#: piece holds (max_a + 1)^3 (max_w + 1)^3 numbers, so an unbounded cache would grow with every offset
EXPANDED_PIECE_CACHE = 256


@lru_cache(maxsize=EXPANDED_PIECE_CACHE)
def _expanded_piece(
    offset_units: Index, half: tuple[int, ...], p: int, max_a: int, max_w: int
) -> tuple[NDArray, NDArray]:
    """R[e0, w0, e1, w1, e2, w2] = int over one half-cube of prod_j x_j^e_j xi_j^w_j |x|^p, in units of
    the box (t = 2 x), by the Legendre moments of S^p.

    The truncation is set by the power p of the function whose moments are taken, so that every t whose
    derivative needs |x|^p shares this one solve.
    """
    n_leg = max_a + max_w + 1 + KSERIES_LEGENDRE_MARGIN + max(0, p) // KSERIES_LEGENDRE_GROWTH
    lows, legs = [], []
    for j in AXES:
        ends = (offset_units[j] - 0.5 * (half[j] > 0), offset_units[j] + 0.5 * (half[j] < 0))
        sign = 1.0 if ends[0] >= 0 else -1.0
        low = 2.0 * min(abs(ends[0]), abs(ends[1]))
        lows.append(low)
        x_of_u = Polynomial([sign * (2.0 * low + 1.0) / 4.0, sign / 4.0])
        xi_of_u = offset_units[j] - x_of_u
        leg = np.zeros((max_a + 1, max_w + 1, n_leg))
        for e in range(max_a + 1):
            for w in range(max_w + 1):
                coefs = legendre.poly2leg((x_of_u**e * xi_of_u**w).coef)
                leg[e, w, : len(coefs)] = coefs
        legs.append(leg)
    lam = box_moments(tuple(lows), 0.0, p, n_leg)
    # the relation gives every moment to round-off of the LARGEST one, not of itself, so the magnitude that
    # its rounding acts on is that scale for every index
    lam_mag = np.full_like(lam, np.abs(lam).max())
    spec = "abx,cdy,efz,xyz->abcdef"
    value = np.einsum(spec, legs[0], legs[1], legs[2], lam, optimize=True)
    magnitude = np.einsum(spec, *[np.abs(g) for g in legs], lam_mag, optimize=True)
    return value, magnitude


def _expanded_coefficients(
    offset_units: Index, a_list: tuple[Index, ...], w_list: tuple[Index, ...], m: int
) -> tuple[NDArray, NDArray]:
    """U[a, W] for an odd m with the derivative expanded into its terms, and the sum of |terms|.

    d^a r^m = sum (coefficient) x^e |x|^p (``derivative_terms``), each term integrated over the eight
    half-cubes by Legendre moments of |x|^p (``legendre_moments.box_moments``), the factor x^e being part of
    the polynomial weight. No integration by parts, so no boundary terms; but the terms of a high derivative
    of a singular power cancel. Returns (U, sum of |terms|), both (n_a, n_W): their ratio is the condition
    number of this representation.
    """
    max_a = max(sum(a) for a in a_list)
    max_w = max(max(w) for w in w_list)
    w_arr = np.array(w_list)
    val = np.zeros((len(a_list), len(w_list)))
    mag = np.zeros((len(a_list), len(w_list)))
    for half in itertools.product((-1, 1), repeat=3):
        for ia, a in enumerate(a_list):
            for coef, e, q in derivative_terms(a):
                c = coef * falling(m, q)
                if c == 0.0:
                    continue
                p = m - 2 * q
                moments, moments_mag = _expanded_piece(offset_units, half, p, max_a, max_w)
                pick = (e[0], w_arr[:, 0], e[1], w_arr[:, 1], e[2], w_arr[:, 2])
                scale = c * 2.0 ** (-p - 3)
                val[ia] += scale * moments[pick]
                mag[ia] += abs(scale) * moments_mag[pick]
    return val, mag


def _polynomial_coefficients(
    offset_units: Index, a_list: tuple[Index, ...], w_list: tuple[Index, ...], m: int
) -> tuple[NDArray, NDArray]:
    """U[a, W] = int_cube d^a r^m (o - xi) xi^W d xi for an even m >= 0, exactly: shape (n_a, n_W).

    r^m is then a polynomial, and d^a r^m xi^W has degree at most m + max W on each axis, so a Gauss rule of
    (m + max W) // 2 + 1 points integrates it exactly (as the even terms of the touching blocks). Evaluated
    from derivative_terms at the nodes, d^a r^m vanishes identically for |a| > m through the falling
    factorial: those zeros are exact, which matters because the weight of the B family at t = 1 grows like
    1 / k and would multiply any round-off left in them. Returns (U, sum of |terms|), both (n_a, n_W).
    """
    max_w = max(max(w) for w in w_list)
    nodes, wts = leggauss((m + max_w) // 2 + 1)
    nodes, wts = nodes / 2.0, wts / 2.0
    xi = np.stack(np.meshgrid(nodes, nodes, nodes, indexing="ij"), -1).reshape(-1, 3)
    wt = np.einsum("i,j,k->ijk", wts, wts, wts).ravel()
    x = np.asarray(offset_units, dtype=float)[None, :] - xi
    r2 = np.sum(x * x, axis=1)
    monos = np.stack([wt * np.prod(xi ** np.array(w, float), axis=1) for w in w_list])  # (n_W, G)
    fields = np.zeros((len(a_list), len(wt)))
    fields_abs = np.zeros((len(a_list), len(wt)))
    for ia, a in enumerate(a_list):
        for coef, e, q in derivative_terms(a):
            c = coef * falling(m, q)
            if c != 0.0:
                term = c * np.prod(x ** np.array(e, float), axis=1) * r2 ** ((m - 2 * q) // 2)
                fields[ia] += term
                fields_abs[ia] += np.abs(term)
    return fields @ monos.T, fields_abs @ np.abs(monos).T


@cache
def kseries_coefficients_closed(
    offset_units: Index, d_list: tuple[Index, ...], w_list: tuple[Index, ...]
) -> tuple[NDArray, tuple[Index, ...]]:
    """The values of ``kseries_coefficients_certified``, with the signature of ``kseries_coefficients``."""
    values, _, a_list = kseries_coefficients_certified(offset_units, d_list, w_list)
    return values, a_list


@cache
def kseries_coefficients_certified(
    offset_units: Index, d_list: tuple[Index, ...], w_list: tuple[Index, ...]
) -> tuple[NDArray, NDArray, tuple[Index, ...]]:
    """The coefficients of ``kseries_coefficients`` in closed form: no approximate quadrature.

    EVEN POWERS (t odd, m = t - 1 even) are polynomials and are integrated exactly by
    ``_polynomial_coefficients``; the derivatives of r^m vanish identically for |a| > m, exactly.

    MOVING THE DERIVATIVES (odd m). With x = o - xi, (d^a r^m)(o - xi) = (-1)^|a| d_xi^a [r^m(o - xi)], and
    along each axis n = a_j derivatives are integrated by parts onto the monomial xi_j^w:

        int_{-h}^{h} d^n g . xi^w = sum_{s <= min(n - 1, w)} (-1)^s [d^(n-1-s) g . (xi^w)^(s)]_{-h}^{h}
                                    + (-1)^n int g . (xi^w)^(n),        h = 1/2.

    Each axis is then either interior, with no derivative of the kernel left on it and the monomial
    differentiated, or fixed at a face xi_j = +-h with b_j = n - 1 - s derivatives left along it. The kernel
    that remains, d^b r^m with b only on the fixed axes, is integrated over the free axes. Written out
    (``derivative_terms``) it is a sum of x^e |x|^(m - 2q) whose x^e involves only the fixed coordinates,
    constants; so the expansion that a sixth derivative of 1/r would need, and whose terms cancel by up to
    1e5 once integrated, is never formed, and in the all-interior term the kernel is r^m itself.

    THE INTEGRALS over the free axes are ``_free_moments``: the field point o is never in the closure of
    the source cube for two different cells (o integer, o != 0), and at least one fixed axis keeps the
    lifted distance c2 > 0, so every box lies away from the singular point and its Legendre moments come
    from corner values by the sparse boundary-value solve of ``legendre_moments.box_moments``.

    TWO REPRESENTATIONS, CHOSEN BY THEIR CONDITION NUMBERS. The integration by parts removes the
    cancellation among the terms of a high derivative of a singular power, but against a monomial of high
    degree along the same axis its own boundary terms cancel instead; the expanded form
    (``_expanded_coefficients``) has the opposite strengths. Each is a sum whose condition number
    kappa = sum |terms| / |sum| is computed from its terms, and each entry is taken from the representation
    with the smaller one. At the face neighbour, against a 40-digit reference, every entry is then as
    accurate as Gauss quadrature or more, the remaining loss being the integral's own conditioning
    int |f| / |int f|, which no route avoids.

    Independent of the frequency and of the cell size. Returns (U, M, the list of a): M is the running error
    magnitude of each coefficient, the sum of |terms| down to the moments, so that eps M is an estimate of
    its rounding error and M / |U| its condition number.

    Raises:
        ValueError: for the self cell, where the integral is a distribution.
    """
    if not any(offset_units):
        raise ValueError(
            "kseries_coefficients_closed: offset (0, 0, 0) is the self cell, a distribution given by the "
            "hierarchy's closed-form moments; these coefficients are for two different cells."
        )
    h = 0.5
    max_a = max(sum(d) for d in d_list) + 2
    max_w = max(max(w) for w in w_list)
    a_list = multi_indices(max_a)
    # the options of one axis after the integration by parts: interior with the monomial of degree w', or
    # fixed at xi_j = side h with b_j derivatives of the kernel left along it
    options: list[tuple] = [("i", w) for w in range(max_w + 1)]
    options += [("b", side, b) for side in (1, -1) for b in range(max_a)]
    pos = {o: k for k, o in enumerate(options)}
    # M[n, w, option]: the coefficient of each option for n derivatives against xi^w, with (-1)^n of
    # d^a = (-1)^|a| d_xi^a and (-1)^b of d_xi^b = (-1)^|b| d^b folded in
    mmat = np.zeros((max_a + 1, max_w + 1, len(options)))
    for n in range(max_a + 1):
        for w in range(max_w + 1):
            if n == 0:
                mmat[n, w, pos[("i", w)]] = 1.0
                continue
            for s in range(min(n - 1, w) + 1):
                for side in (1, -1):
                    deriv = math.factorial(w) / math.factorial(w - s) * (side * h) ** (w - s)
                    b = n - 1 - s
                    mmat[n, w, pos[("b", side, b)]] += (-1.0) ** (n + b + s) * side * deriv
            if n <= w:
                mmat[n, w, pos[("i", w - n)]] += (
                    (-1.0) ** (2 * n) * math.factorial(w) / math.factorial(w - n)
                )
    out = np.zeros((KSERIES_T_MAX + 1, len(a_list), len(w_list)))
    out_mag = np.zeros_like(out)
    a_arr, w_arr = np.array(a_list), np.array(w_list)
    for t in range(KSERIES_T_MAX + 1):
        m = t - 1
        if m >= 0 and m % 2 == 0:
            out[t], out_mag[t] = _polynomial_coefficients(offset_units, a_list, w_list, m)
            continue
        kern = np.zeros((len(options),) * 3)
        kern_mag = np.zeros((len(options),) * 3)
        for combo in itertools.product(range(len(options)), repeat=3):
            opts = [options[k] for k in combo]
            fixed = [j for j in AXES if opts[j][0] == "b"]
            b_vec = tuple(opts[j][2] if opts[j][0] == "b" else 0 for j in AXES)
            if sum(b_vec) > max_a - len(fixed):  # never reached: each fixed axis used one derivative
                continue
            free = tuple(j for j in AXES if opts[j][0] == "i")
            fixed_spec = tuple((j, offset_units[j] - opts[j][1] * h, b_vec[j]) for j in fixed)
            free_spec = tuple((j, opts[j][1]) for j in free)
            kern[combo], kern_mag[combo] = _reduced_integral(offset_units, free_spec, fixed_spec, m, max_w)
        # U[a, W] = sum over the options of every axis of prod_j M[a_j, W_j, o_j] K[o_0, o_1, o_2]
        m0 = mmat[a_arr[:, 0]][:, w_arr[:, 0]]  # (n_a, n_W, n_opt)
        m1 = mmat[a_arr[:, 1]][:, w_arr[:, 1]]
        m2 = mmat[a_arr[:, 2]][:, w_arr[:, 2]]
        moved = np.einsum("awx,awy,awz,xyz->aw", m0, m1, m2, kern, optimize=True)
        moved_abs = np.einsum(
            "awx,awy,awz,xyz->aw", np.abs(m0), np.abs(m1), np.abs(m2), kern_mag, optimize=True
        )
        expanded, expanded_abs = _expanded_coefficients(offset_units, a_list, w_list, m)
        # each entry from the representation with the smaller condition number sum |terms| / |sum|, which is
        # computed from the terms themselves: comparing the two sums of |terms| compares the two kappas
        use_expanded = expanded_abs <= moved_abs
        out[t] = np.where(use_expanded, expanded, moved)
        out_mag[t] = np.where(use_expanded, expanded_abs, moved_abs)
    return out, out_mag, a_list


@cache
def _kseries_rows(
    d_list: tuple[Index, ...], a_list: tuple[Index, ...]
) -> tuple[list[int], list[list[list[int]]]]:
    """Rows of kseries_coefficients for each D, and for each D + e_i + e_n."""
    pos = {a: k for k, a in enumerate(a_list)}
    rows_b = [[[0] * len(d_list) for _ in AXES] for _ in AXES]
    for i in AXES:
        for n in AXES:
            for k, d in enumerate(d_list):
                a = [d[0], d[1], d[2]]
                a[i] += 1
                a[n] += 1
                rows_b[i][n][k] = pos[(a[0], a[1], a[2])]
    return [pos[d] for d in d_list], rows_b


def moment_table_kseries(
    offset_units: Index,
    side: float,
    omega: complex,
    ref: ReferenceMedium,
    d_list: list[Index],
    w_list: list[Index],
    tol: float,
    coefficients: Callable[..., tuple[NDArray, tuple[Index, ...]]] = kseries_coefficients,
) -> NDArray:
    """The hierarchy's table between two different cells as a power series in the wavenumber.

    With g_S = sum_t (i k_S)^t r^(t-1) / t! and B = (g_S - g_P) / k_S^2 term by term,

        T[i, n, D, W] = (1 / 4 pi mu) sum_t [ c1(t) delta_in U(t; D, W) + c2(t) U(t; D + e_i + e_n, W) ],

    c1 = (i k_S)^t / t!, c2 = ((i k_S)^t - (i k_P)^t) / (t! k_S^2), and U the coefficients of
    ``kseries_coefficients``, scaled to the cell by their homogeneity, side^((t - 1) - |a| + |W| + 3). The
    coefficients are computed once for each offset and serve every frequency and every cell size, so for a
    sweep in frequency each further frequency costs only this sum. The B part loses two powers of k to the
    division by k_S^2, so its term t is as large as the Green term t - 2 (the radiation part at order k
    comes from t = 1 and t = 3 together). Summed through t = 4 at least, and until the bound
    (|k_S| r_max)^(t - 2) / (t - 2)! falls below tol, r_max the largest distance from the field point to
    the source cube. coefficients chooses how U is obtained: by Gauss rules
    (kseries_coefficients) or in closed form (kseries_coefficients_closed).

    Raises:
        ValueError: for the self cell, or a frequency for which KSERIES_T_MAX terms do not reach tol.
    """
    if not any(offset_units):
        raise ValueError(
            "moment_table_kseries: offset (0, 0, 0) is the self cell, a distribution given by the "
            "hierarchy's closed-form moments; this series is for two different cells."
        )
    u_all, a_list = coefficients(tuple(offset_units), tuple(d_list), tuple(w_list))
    k_s, k_p = omega / ref.beta, omega / ref.alpha
    r_max = (np.linalg.norm(np.asarray(offset_units, float)) + np.sqrt(3.0) / 2.0) * side
    n_terms = None
    for t in range(KSERIES_T_MAX + 1):
        if t > 4 and (abs(k_s) * r_max) ** (t - 2) / math.factorial(t - 2) < tol:
            n_terms = t
            break
    if n_terms is None:
        raise ValueError(
            f"moment_table_kseries: k_S r = {abs(k_s) * r_max:.2f} needs more than {KSERIES_T_MAX} terms "
            f"for tol = {tol:g}. Fix: use moment_table (Gauss) or moment_table_series (multipole) at this "
            "frequency."
        )
    # the sums over t first, for every a at once, with the cell scaling split as
    # side^((t - 1) - |a| + |W| + 3) = side^t side^(2 - |a| + |W|)
    t_idx = np.arange(n_terms)
    fact = np.array([float(math.factorial(t)) for t in t_idx])
    s_t = side**t_idx
    c1 = (1j * k_s) ** t_idx / fact * s_t
    c2 = np.where(t_idx >= 1, ((1j * k_s) ** t_idx - (1j * k_p) ** t_idx) / (fact * k_s**2), 0.0) * s_t
    a_deg = np.array([sum(a) for a in a_list], dtype=float)
    w_deg = np.array([sum(w) for w in w_list], dtype=float)
    scale = side ** (2.0 - a_deg[:, None] + w_deg[None, :])
    u_t = u_all[:n_terms]
    v1 = np.tensordot(c1, u_t, axes=(0, 0)) * scale
    v2 = np.tensordot(c2, u_t, axes=(0, 0)) * scale
    rows_d, rows_b = _kseries_rows(tuple(d_list), a_list)
    out = np.zeros((3, 3, len(d_list), len(w_list)), dtype=complex)
    for i in AXES:
        for n in range(i, 3):
            val = v2[rows_b[i][n]]
            if i == n:
                val = val + v1[rows_d]
            out[i, n] = val
            out[n, i] = val
    return out / (4.0 * np.pi * ref.mu)


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
