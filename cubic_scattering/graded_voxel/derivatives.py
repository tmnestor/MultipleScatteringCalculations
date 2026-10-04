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
KSERIES_GAUSS = {"face": 28, "edge": 20, "corner": 14, "far": 12}


def kseries_gauss_points(offset_units: Index) -> int:
    """Gauss points per axis for the k-series coefficients of this offset (KSERIES_GAUSS)."""
    a = sorted((abs(o) for o in offset_units), reverse=True)
    if a[0] > 1:
        return KSERIES_GAUSS["far"]
    return KSERIES_GAUSS[("face", "edge", "corner")[sum(a) - 1]]


@cache
def kseries_coefficients(
    offset_units: Index, d_list: tuple[Index, ...], w_list: tuple[Index, ...]
) -> tuple[NDArray, tuple[Index, ...]]:
    """U[t, a, w] = int_cube d^a r^(t - 1) (o - xi) xi^W d xi on the UNIT cube, o = offset_units.

    For t = 0 .. KSERIES_T_MAX and every multi-index a of order up to max |D| + 2 (the extra two are the
    derivatives that turn B into the Green's tensor). The integrand is smooth for two different cells, and
    a tensor Gauss rule of ``kseries_gauss_points(offset)`` points integrates it to round-off. Independent
    of the frequency and of the cell size, so computed once per offset. Returns (U, the list of a).
    """
    o = np.asarray(offset_units, dtype=float)
    pts, wts = cell_rule(1.0, kseries_gauss_points(offset_units))
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
    the source cube.

    Raises:
        ValueError: for the self cell, or a frequency for which KSERIES_T_MAX terms do not reach tol.
    """
    if not any(offset_units):
        raise ValueError(
            "moment_table_kseries: offset (0, 0, 0) is the self cell, a distribution given by the "
            "hierarchy's closed-form moments; this series is for two different cells."
        )
    u_all, a_list = kseries_coefficients(tuple(offset_units), tuple(d_list), tuple(w_list))
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
