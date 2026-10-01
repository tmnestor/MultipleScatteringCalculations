"""Galerkin coupling blocks K_ac(R) of the graded voxel.

K[a, c](R) = int_{V_m} int_{V_n} L_a((x - x_m)/h) P(x - x') m_c((x' - x_n)/h) dx dx',  R = x_m - x_n,

a 9 x 9 block for each test function a (4) and source monomial c (n_source: 10 of degree <= 2 for a contrast
linear in the cell, 20 of degree <= 3 for a quadratic one; the first ten columns are the same).  With
n_test = 10 the rows are the ten field functions of a field quadratic in the cell (``basis.CONTRAST_BASIS``;
the first four rows are the same), and the sources may be the 35 monomials of degree <= 4.  The integrals
are taken with the test MONOMIALS of degree <= 2 and recombined into the orthogonal functions.

THE s-FORM.  With x = x_m + u, x' = x_n + u', the kernel sees r = R + s, s = u - u' in [-2h, 2h]^3, and
the double integral is int W_ac(s) P(R + s) ds with the separable autocorrelation
W_ac(s) = prod_i w(e_a,i, e_c,i; s_i),  w(s) = h int v^e_t (v - s/h)^e_s dv  over
v in [max(-1, s/h - 1), min(1, s/h + 1)]: continuous, piecewise polynomial on [-2h, 0] and [0, 2h], zero
at +-2h, with kinks at 0 and +-2h.

* Non-touching offsets (``coupling_block``): the full kernel is smooth on every piece: 3-D Gauss on the
  8 pieces, 8 n^3 kernel points (``far_block``, the 6-D tensor Gauss in (u, u'), is kept as the
  independent reference).
* The self cell and its 26 touching neighbours (``near_block``): the static (Kelvin) part is a
  distribution, a sum over terms coef * d^idx r^m (m = -1 with |idx| <= 2; m = 1 with |idx| <= 4).
  k = max(0, |idx| - (m + 1)) derivatives are moved onto W: int W d^idx F = (-1)^k int (d^moved W) d^rest F.
  Two derivatives on one axis make w'' a piecewise polynomial PLUS plane deltas J_b delta(s - b h),
  b in {-2, 0, 2}, J_b the jump of w' at b; at most one axis carries them (k <= 2).  What remains,
  d^rest r^m, is at most 1/r.  The dynamic remainder of the kernel is at most 1/r and is integrated
  directly.  Every piece is a box (3-D) or, with a delta, a rectangle (2-D) whose corners sit on the
  breakpoints {-2h, 0, 2h}: the singular point s* = -R is a vertex of the piece (Duffy pyramids about it)
  or at distance >= 2h (tensor Gauss).
"""

import itertools
import math
from collections import defaultdict
from collections.abc import Callable
from functools import cache, lru_cache

import numpy as np
import sympy as sp
from numpy.polynomial import Polynomial
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray

from ..effective_contrasts import ReferenceMedium
from .basis import SOURCE_EXPONENTS, field_in_monomials, monomials, source_exponents
from .kernel import kernel_9x9, power_F, radial_component, static_b2, voigt_maps
from .moments import box_integral, face_integral


@lru_cache(maxsize=8)
def _cell_rule(n: int) -> tuple[NDArray, NDArray]:
    x, w = leggauss(n)
    xi = np.stack(np.meshgrid(x, x, x, indexing="ij"), -1).reshape(-1, 3)
    return xi, np.einsum("i,j,k->ijk", w, w, w).ravel()


def gauss_order(offset: tuple[int, int, int]) -> int:
    """Gauss points per axis on each s-form piece for a non-touching offset."""
    d = max(abs(o) for o in offset)
    if d <= 1:
        raise ValueError(f"gauss_order: offset {offset} touches; use near_block")
    return 14 if d == 2 else 10 if d <= 4 else 8


def far_block(
    offset: tuple[int, int, int],
    h: float,
    omega: float,
    ref: ReferenceMedium,
    n_gauss: int,
    n_source: int = 10,
    n_test: int = 4,
) -> NDArray:
    """K[a, c] for a non-touching offset, shape (n_test, n_source, 9, 9)."""
    xi, w = _cell_rule(n_gauss)
    R = 2.0 * h * np.asarray(offset, dtype=float)
    X = (R[None, None, :] + h * (xi[:, None, :] - xi[None, :, :])).reshape(-1, 3)
    P = kernel_9x9(X, omega, ref).reshape(len(xi), len(xi), 81)
    lt = monomials(SOURCE_EXPONENTS[:n_test], xi) * w
    ls = monomials(source_exponents(n_source), xi) * w
    tmp = np.einsum("ap,pqz->aqz", lt, P)
    return _to_field_rows((h**6 * np.einsum("cq,aqz->acz", ls, tmp)).reshape(n_test, n_source, 9, 9))


def _to_field_rows(block: NDArray) -> NDArray:
    """Rows from the test monomials to the orthogonal field functions (the identity for four rows)."""
    n_test = block.shape[0]
    if n_test == 4:
        return block
    return np.einsum("am,m...->a...", field_in_monomials(n_test), block)


# ---------------------------------------------------------------------------
# The s-form
# ---------------------------------------------------------------------------

Part = tuple[str, float, float, Polynomial | float]


@cache
def autocorrelation_1d(e_t: int, e_s: int) -> tuple[Polynomial, Polynomial]:
    """w(s) / h as polynomials in sigma = s / h, on [-2, 0] and on [0, 2]."""
    v, sg = sp.symbols("v sigma", real=True)
    f = v**e_t * (v - sg) ** e_s
    pieces = (sp.expand(sp.integrate(f, (v, -1, sg + 1))), sp.expand(sp.integrate(f, (v, sg - 1, 1))))
    out = []
    for e in pieces:
        coeffs = sp.Poly(e, sg).all_coeffs()[::-1]
        out.append(Polynomial([float(c) for c in coeffs]))
    return out[0], out[1]


def _axis_parts(e_t: int, e_s: int, n_der: int, h: float) -> list[Part]:
    """The n_der-th s-derivative of w on one axis, in physical units: ('piece', lo, hi, poly in sigma), and
    for n_der = 2 the plane deltas ('delta', b, b, jump), b in units of h."""
    left, right = autocorrelation_1d(e_t, e_s)
    scale = h ** (1 - n_der)
    parts: list[Part] = [
        ("piece", -2.0, 0.0, scale * left.deriv(n_der)),
        ("piece", 0.0, 2.0, scale * right.deriv(n_der)),
    ]
    if n_der == 2:
        # zero jumps are kept: every exponent pair then has the same parts list on an axis, so one index
        # (`choice` in _sform) names the same breakpoint for every (a, c)
        dl, dr = left.deriv(1), right.deriv(1)
        for b, jump in ((-2.0, dl(-2.0)), (0.0, dr(0.0) - dl(0.0)), (2.0, -dr(2.0))):
            parts.append(("delta", b, b, float(jump)))
    return parts


@lru_cache(maxsize=16)
def _gauss01(n: int) -> tuple[NDArray, NDArray]:
    x, w = leggauss(n)
    return (x + 1.0) / 2.0, w / 2.0


def _box_rule(lo: NDArray, hi: NDArray, vertex: NDArray | None, n: int) -> tuple[NDArray, NDArray]:
    """Nodes and weights on prod [lo_i, hi_i] (d = 2 or 3 dims); Duffy pyramids about `vertex` if given."""
    d = len(lo)
    t, wt = _gauss01(n)
    grid = np.stack(np.meshgrid(*([t] * d), indexing="ij"), -1).reshape(-1, d)
    wgrid = np.prod(np.stack(np.meshgrid(*([wt] * d), indexing="ij"), -1).reshape(-1, d), axis=1)
    if vertex is None:
        return lo + grid * (hi - lo), wgrid * np.prod(hi - lo)
    opp = np.where(np.isclose(vertex, lo), hi, lo)
    span = opp - vertex
    nodes_all, w_all = [], []
    for lead in range(d):
        y = np.empty_like(grid)
        others = [j for j in range(d) if j != lead]
        y[:, lead] = grid[:, 0]
        for col, j in enumerate(others, start=1):
            y[:, j] = grid[:, 0] * grid[:, col]
        nodes_all.append(vertex + y * span)
        w_all.append(wgrid * grid[:, 0] ** (d - 1) * np.prod(np.abs(span)))
    return np.concatenate(nodes_all), np.concatenate(w_all)


def _sform(
    offset: tuple[int, int, int],
    h: float,
    orders: tuple[int, int, int],
    kernel_at: Callable[[NDArray], NDArray],
    n_q: int,
    n_source: int = 10,
    n_test: int = 4,
) -> NDArray:
    """sum over pieces of int prod_i d^orders_i w_i(s_i) K(R + s) ds for all (a, c), shape
    (n_test, n_source, Z); a runs over the test MONOMIALS SOURCE_EXPONENTS[:n_test].

    kernel_at(X) returns (N, Z) kernel values at the separations X = R + s.
    """
    R = 2.0 * h * np.asarray(offset, dtype=float)
    sstar = -R / h  # the singular point, in units of h
    src = source_exponents(n_source)
    tst = SOURCE_EXPONENTS[:n_test]
    axis_exps = sorted(
        {(tst[a][i], src[c][i]) for a in range(n_test) for c in range(n_source) for i in range(3)}
    )
    parts = {(i, et, es): _axis_parts(et, es, orders[i], h) for i in range(3) for et, es in axis_exps}
    shape = [len(parts[(i, *axis_exps[0])]) for i in range(3)]
    out: NDArray | None = None
    for choice in itertools.product(*[range(s) for s in shape]):
        kinds = [parts[(i, *axis_exps[0])][choice[i]] for i in range(3)]
        delta_axes = [i for i, k in enumerate(kinds) if k[0] == "delta"]
        free = [i for i in range(3) if i not in delta_axes]
        lo = np.array([kinds[i][1] for i in free])
        hi = np.array([kinds[i][2] for i in free])
        at_vertex = all(np.isclose(sstar[i], kinds[i][1]) for i in delta_axes) and all(
            np.isclose(sstar[i], lo[j]) or np.isclose(sstar[i], hi[j]) for j, i in enumerate(free)
        )
        vertex = np.array([sstar[i] for i in free]) if at_vertex else None
        nodes, w = _box_rule(lo, hi, vertex, n_q)
        sig = np.empty((len(nodes), 3))
        for j, i in enumerate(free):
            sig[:, i] = nodes[:, j]
        for i in delta_axes:
            sig[:, i] = kinds[i][1]
        kv = kernel_at(R + h * sig)
        if out is None:
            out = np.zeros((n_test, n_source, kv.shape[1]), dtype=complex)
        jac = w * h ** len(free)
        axis_vals = {}
        for (i, et, es), plist in parts.items():
            kind = plist[choice[i]]
            poly = kind[3]
            axis_vals[(i, et, es)] = (
                np.full(len(sig), float(poly)) if kind[0] == "delta" else poly(sig[:, i])  # type: ignore[operator, arg-type]
            )
        val = np.broadcast_to(jac, (n_test, n_source, len(jac))).copy()
        for i in range(3):
            val *= np.array([[axis_vals[(i, ta[i], sc[i])] for sc in src] for ta in tst])
        out += (val.reshape(n_test * n_source, -1) @ kv).reshape(n_test, n_source, -1)
    assert out is not None
    return out


@lru_cache(maxsize=8)
def static_term_table(alpha: float, beta: float, rho: float) -> dict[tuple[int, tuple[int, ...]], NDArray]:
    """The static propagator as sum over (m, idx) of coef (9 x 9) * d^idx r^m."""
    ref = ReferenceMedium(alpha, beta, rho)
    pref = 1.0 / (4.0 * np.pi * ref.mu)
    b2 = static_b2(ref)
    table: dict[tuple[int, tuple[int, ...]], NDArray] = defaultdict(lambda: np.zeros((9, 9)))

    def g_terms(i: int, j: int) -> list[tuple[int, tuple[int, ...], float]]:
        terms: list[tuple[int, tuple[int, ...], float]] = [(1, (i, j), pref * b2)]
        if i == j:
            terms.append((-1, (), pref))
        return terms

    def add(row: int, col: int, extra: tuple[int, ...], i: int, j: int, c: float) -> None:
        for m, idx, coef in g_terms(i, j):
            table[(m, tuple(sorted(idx + extra)))][row, col] += c * coef

    mc, mh, ms = voigt_maps()
    for i in range(3):
        for j in range(3):
            add(i, j, (), i, j, 1.0)
    for z in range(27):
        i, j, k = (int(v) for v in np.unravel_index(z, (3, 3, 3)))
        for row in range(3):
            for al in range(6):
                if mc[row, al, z]:
                    add(row, 3 + al, (k,), i, j, mc[row, al, z])
        for al in range(6):
            for col in range(3):
                if mh[al, col, z]:
                    add(3 + al, col, (k,), i, j, mh[al, col, z])
    for z in range(81):
        i, j, k, q = (int(v) for v in np.unravel_index(z, (3, 3, 3, 3)))
        for al in range(6):
            for be in range(6):
                if ms[al, be, z]:
                    add(3 + al, 3 + be, (k, q), i, j, ms[al, be, z])
    return dict(table)


def _moves(m: int, idx: tuple[int, ...]) -> int:
    if m >= 0 and m % 2 == 0:
        return 0
    return max(0, len(idx) - (m + 1))


def static_term_integral(
    m: int,
    idx: tuple[int, ...],
    offset: tuple[int, int, int],
    h: float,
    n_q: int,
    n_source: int = 10,
    n_test: int = 4,
) -> NDArray:
    """int W_ac(s) d^idx r^m (R + s) ds for all (a, c), shape (n_test, n_source), distributionally; a runs
    over the test monomials."""
    k = _moves(m, idx)
    moved, rest = idx[:k], idx[k:]
    orders = (moved.count(0), moved.count(1), moved.count(2))

    def kern(X: NDArray) -> NDArray:
        return radial_component(power_F(m, np.linalg.norm(X, axis=1)), X, rest)[:, None]

    return (-1) ** k * _sform(offset, h, orders, kern, n_q, n_source, n_test)[:, :, 0]


# ---------------------------------------------------------------------------
# The static terms in closed form
# ---------------------------------------------------------------------------

_SIDES = (0, 2, 4)  # |t| at the corners of the pieces, in units of h, measured from the singular point


@cache
def _box_table(m: int, e_max: int) -> NDArray:
    """T[x, y, z, i, j, k] = int over [0, a] x [0, b] x [0, c] of t0^i t1^j t2^k |t|^m, with the sides
    a, b, c = _SIDES[x], _SIDES[y], _SIDES[z].

    Zero when a side is zero; NaN where the integral diverges (never read with a non-zero coefficient).
    """
    tab = np.zeros((3, 3, 3, e_max, e_max, e_max))
    for x, y, z in itertools.product((1, 2), repeat=3):
        for i, j, k in itertools.product(range(e_max), repeat=3):
            tab[x, y, z, i, j, k] = (
                float(box_integral(i, j, k, _SIDES[x], _SIDES[y], _SIDES[z], m))
                if i + j + k + m > -3
                else np.nan
            )
    return tab


@cache
def _face_table(m: int, a: int, e_max: int) -> NDArray:
    """T[y, z, j, k] = int over [0, b] x [0, c] of t1^j t2^k (a^2 + t1^2 + t2^2)^(m/2), with the sides
    b, c = _SIDES[y], _SIDES[z]."""
    tab = np.zeros((3, 3, e_max, e_max))
    for y, z in itertools.product((1, 2), repeat=2):
        for j, k in itertools.product(range(e_max), repeat=2):
            tab[y, z, j, k] = (
                float(face_integral(j, k, a, _SIDES[y], _SIDES[z], m))
                if (a > 0 or j + k + m > -2)
                else np.nan
            )
    return tab


def _anchored(poly_t: NDArray, lo: float, hi: float, alpha: int, e_max: int) -> NDArray:
    """V[side, e]: int_lo^hi t^e g(t) dt = sum_side V[side, e] int_0^side t^e g(t) dt, for an integrand
    whose total power of t on this axis is e + alpha (g even in t apart from that power).

    int_lo^hi = F(hi) - F(lo), F(v) = int_0^v; for v < 0, F(v) = (-1)^(e + alpha + 1) int_0^|v|.
    """
    out = np.zeros((3, e_max))
    for v, end_sign in ((hi, 1.0), (lo, -1.0)):
        if v == 0:
            continue
        side = _SIDES.index(round(abs(v)))
        for e, coef in enumerate(poly_t):
            parity = 1.0 if v > 0 else (-1.0) ** (e + alpha + 1)
            out[side, e] += end_sign * parity * coef
    return out


def static_term_integral_closed(
    m: int,
    idx: tuple[int, ...],
    offset: tuple[int, int, int],
    h: float,
    n_source: int = 10,
    n_test: int = 4,
) -> NDArray:
    """``static_term_integral`` with no quadrature: every piece from the master integrals of ``moments``.

    Valid for every odd m >= -1 (the static terms are m = -1 and m = 1; the odd terms of the dynamic series
    are m = 1, 3, 5, ...).  After the moves the kernel d^rest r^m is a sum of monomials times odd powers
    |t|^n, n >= -3 (``radial_monomials``), t = (R + s) / h measured from the singular point: 1/r for
    m = -1, delta_ij / r - t_i t_j / r^3 for m = 1.  Each piece of
    W is a polynomial on a box with corners at t_i in {0, +-2, +-4}; per axis it is the difference of two
    integrals from the origin, so the piece is a signed sum of origin-anchored box integrals.  A plane
    delta fixes one coordinate at t_i = x0 and leaves a face integral at distance |x0|.

    Raises:
        ValueError: when the offset does not touch (the closed forms are tabulated for the near cells), or
            the term is not one of the static table's.
    """
    if max(abs(o) for o in offset) > 1:
        raise ValueError(f"static_term_integral_closed: offset {offset} does not touch; use coupling_block")
    if m % 2 == 0 or m < -1:
        raise ValueError(f"static_term_integral_closed: m must be odd and >= -1, got {m}")
    k = _moves_closed(m, idx)
    if k > 2:
        raise ValueError(
            f"static_term_integral_closed: d^{idx} r^{m} needs {k} > 2 derivatives moved onto W"
        )
    moved, rest = idx[:k], idx[k:]
    orders = (moved.count(0), moved.count(1), moved.count(2))
    kernel = radial_monomials(m, rest)
    src = source_exponents(n_source)
    tst = SOURCE_EXPONENTS[:n_test]
    axis_exps = sorted(
        {(tst[a][i], src[c][i]) for a in range(n_test) for c in range(n_source) for i in range(3)}
    )
    parts = {(i, et, es): _axis_parts(et, es, orders[i], h) for i in range(3) for et, es in axis_exps}
    shape = [len(parts[(i, *axis_exps[0])]) for i in range(3)]
    sstar = [-2 * o for o in offset]  # the singular point in sigma = s / h
    deg = max(len(pl[3].coef) for plist in parts.values() for pl in plist if isinstance(pl[3], Polynomial))
    e_max = deg + 2
    pad = 4  # the kernel's monomials raise an exponent by at most len(rest) <= 4
    out = np.zeros((n_test, n_source))
    for choice in itertools.product(*[range(n) for n in shape]):
        kinds = [parts[(i, *axis_exps[0])][choice[i]] for i in range(3)]
        fixed = [i for i in range(3) if kinds[i][0] == "delta"]
        if len(fixed) > 1:
            raise ValueError("static_term_integral_closed: two plane deltas on one piece")
        free = [i for i in range(3) if i not in fixed]
        for coef, alpha, power in kernel:
            factors = []
            x0 = 0
            for i in range(3):
                vals: dict[tuple[int, int], NDArray | float] = {}
                for et, es in axis_exps:
                    kind = parts[(i, et, es)][choice[i]]
                    poly = kind[3]
                    if not isinstance(poly, Polynomial):
                        x0 = round(kind[1]) - sstar[i]
                        vals[(et, es)] = float(poly) * float(x0) ** alpha[i]
                    else:
                        shifted = poly(Polynomial([sstar[i], 1.0])).coef  # sigma = t + sigma*
                        vals[(et, es)] = _anchored(
                            shifted, kind[1] - sstar[i], kind[2] - sstar[i], alpha[i], e_max
                        )
                factors.append(np.array([[vals[(ta[i], sc[i])] for sc in src] for ta in tst]))
            if fixed:
                f = fixed[0]
                if x0 == 0 and alpha[f] > 0:
                    continue
                j, l = free
                tab = _face_table(power, abs(x0), e_max + pad)[
                    :, :, alpha[j] : alpha[j] + e_max, alpha[l] : alpha[l] + e_max
                ]
                term = np.einsum("ac,acyj,aczk,yzjk->ac", factors[f], factors[j], factors[l], tab)
            else:
                tab = _box_table(power, e_max + pad)[
                    :,
                    :,
                    :,
                    alpha[0] : alpha[0] + e_max,
                    alpha[1] : alpha[1] + e_max,
                    alpha[2] : alpha[2] + e_max,
                ]
                term = np.einsum("acxi,acyj,aczk,xyzijk->ac", factors[0], factors[1], factors[2], tab)
            # physical units: h per free axis, and d^rest r^m is h^(m - |rest|) times its value in t
            out += coef * h ** (len(free) + m - len(rest)) * term
    return (-1) ** k * out


def _moves_closed(m: int, idx: tuple[int, ...]) -> int:
    """Derivatives moved onto W so that d^rest r^m is at most 1/r singular AND its lowest power of the
    distance is r^-3: |rest| <= min(m + 1, (m + 3) // 2)."""
    return max(0, len(idx) - min(m + 1, (m + 3) // 2))


@cache
def radial_monomials(m: int, idx: tuple[int, ...]) -> list[tuple[float, tuple[int, int, int], int]]:
    """d^idx r^m as a list of (coefficient, alpha, n): sum of coef * t^alpha * |t|^n.

    From d_i (t^alpha |t|^n) = alpha_i t^(alpha - e_i) |t|^n + n t^(alpha + e_i) |t|^(n - 2).
    """
    terms: dict[tuple[tuple[int, int, int], int], float] = {((0, 0, 0), m): 1.0}
    for i in idx:
        new: dict[tuple[tuple[int, int, int], int], float] = defaultdict(float)
        for (alpha, n), coef in terms.items():
            if alpha[i] > 0:
                low = tuple(a - (1 if j == i else 0) for j, a in enumerate(alpha))
                new[(low, n)] += coef * alpha[i]  # type: ignore[index]
            if n != 0:
                up = tuple(a + (1 if j == i else 0) for j, a in enumerate(alpha))
                new[(up, n - 2)] += coef * n  # type: ignore[index]
        terms = dict(new)
    return [(coef, alpha, n) for (alpha, n), coef in terms.items() if coef != 0.0]


@cache
def family_tables() -> tuple[dict[tuple[int, ...], NDArray], dict[tuple[int, ...], NDArray]]:
    """(TA, TB): the 9 x 9 coefficient of d^idx f in the propagator built from G_ij = delta_ij f (TA) and
    from G_ij = d_i d_j f (TB), f any radial function.  No prefactor: the medium enters only through the
    scalar weights of each power of r.
    """
    ta: dict[tuple[int, ...], NDArray] = defaultdict(lambda: np.zeros((9, 9)))
    tb: dict[tuple[int, ...], NDArray] = defaultdict(lambda: np.zeros((9, 9)))

    def add(row: int, col: int, extra: tuple[int, ...], i: int, j: int, c: float) -> None:
        tb[tuple(sorted((i, j) + extra))][row, col] += c
        if i == j:
            ta[tuple(sorted(extra))][row, col] += c

    mc, mh, ms = voigt_maps()
    for i in range(3):
        for j in range(3):
            add(i, j, (), i, j, 1.0)
    for z in range(27):
        i, j, k = (int(v) for v in np.unravel_index(z, (3, 3, 3)))
        for row in range(3):
            for al in range(6):
                if mc[row, al, z]:
                    add(row, 3 + al, (k,), i, j, mc[row, al, z])
        for al in range(6):
            for col in range(3):
                if mh[al, col, z]:
                    add(3 + al, col, (k,), i, j, mh[al, col, z])
    for z in range(81):
        i, j, k, q = (int(v) for v in np.unravel_index(z, (3, 3, 3, 3)))
        for al in range(6):
            for be in range(6):
                if ms[al, be, z]:
                    add(3 + al, 3 + be, (k, q), i, j, ms[al, be, z])
    return dict(ta), dict(tb)


@cache
def universal_moment(
    m: int, idx: tuple[int, ...], offset: tuple[int, int, int], n_source: int = 10, n_test: int = 4
) -> NDArray:
    """U[a, c] = int W_ac(s) d^idx r^m (R + s) ds for cells of half-width 1: a pure number per (a, c).

    For half-width h the moment is h^(6 + m - |idx|) U.  Odd m: the master integrals
    (``static_term_integral_closed``).  Even m: r^m is a polynomial, and a Gauss rule of sufficient order
    on the pieces of W is exact.  a runs over the test monomials.
    """
    if m % 2:
        return static_term_integral_closed(m, idx, offset, 1.0, n_source, n_test)
    if len(idx) > m:
        return np.zeros((n_test, n_source))

    def kern(X: NDArray) -> NDArray:
        return radial_component(power_F(m, np.linalg.norm(X, axis=1)), X, idx)[:, None]

    return _sform(offset, 1.0, (0, 0, 0), kern, 6 + m // 2, n_source, n_test)[:, :, 0].real


def series_weights(m: int, omega: float, ref: ReferenceMedium) -> tuple[complex, complex]:
    """(c1, c2): the weights of delta_ij r^m and of d_i d_j r^m in G_ij, g_k = sum_t (i k)^t r^(t-1) / t!.

    G_ij = (1 / 4 pi mu) [delta_ij g_S + k_S^-2 d_i d_j (g_S - g_P)], so with t = m + 1:
    c1 = (i k_S)^t / t!,  c2 = i^t (k_S^t - k_P^t) / (k_S^2 t!), both times 1 / (4 pi mu).
    The static (Kelvin) part is c1 at m = -1 and c2 at m = 1.
    """
    k_p, k_s = omega / ref.alpha, omega / ref.beta
    t = m + 1
    pref = 1.0 / (4.0 * np.pi * ref.mu)
    fact = float(math.factorial(t))
    c1 = pref * (1j * k_s) ** t / fact
    c2 = pref * (1j**t) * (k_s**t - k_p**t) / (k_s**2 * fact)
    return c1, c2


def near_block_series(
    offset: tuple[int, int, int],
    h: float,
    omega: float,
    ref: ReferenceMedium,
    n_source: int = 10,
    n_test: int = 4,
    tol: float = 1e-17,
) -> NDArray:
    """K[a, c] of the self cell or a touching neighbour with no quadrature of a singular or oscillatory
    kernel: the power series of the propagator in the wavenumber, each term a universal moment.

        K = sum_{m >= -1} sum_idx (c1(m) TA[idx] + c2(m) TB[idx]) h^(6 + m - |idx|) U(m, idx, offset),

    c from ``series_weights``, TA and TB from ``family_tables``, U from ``universal_moment``.  The series
    is that of exp(i k r) and converges for every k; it is summed until its terms fall below ``tol``
    relative to the first, with (k_S * 4 sqrt(3) h)^t / t! as the bound on term t.

    Raises:
        ValueError: when the offset does not touch.
    """
    if max(abs(o) for o in offset) > 1:
        raise ValueError(f"near_block_series: offset {offset} does not touch; use coupling_block")
    ta, tb = family_tables()
    reach = omega / ref.beta * 4.0 * np.sqrt(3.0) * h
    out = np.zeros((n_test, n_source, 9, 9), dtype=complex)
    m = -1
    while True:
        c1, c2 = series_weights(m, omega, ref)
        for table, weight in ((ta, c1), (tb, c2)):
            if weight == 0:
                continue
            for idx, coef in table.items():
                u = universal_moment(m, idx, offset, n_source, n_test)
                out += weight * h ** (6 + m - len(idx)) * u[:, :, None, None] * coef[None, None]
        m += 1
        if m > 1 and reach ** (m + 1) / math.factorial(m + 1) < tol:
            break
    return _to_field_rows(out)


_NEAR_CACHE: dict[tuple, NDArray] = {}


def near_block(
    offset: tuple[int, int, int],
    h: float,
    omega: float,
    ref: ReferenceMedium,
    n_q: int = 12,
    n_source: int = 10,
    n_test: int = 4,
    static: str = "quadrature",
) -> NDArray:
    """K[a, c] by the s-form with the distributional static part, shape (n_test, n_source, 9, 9).

    Valid for any offset; required for the self cell and its touching neighbours. Cached.  ``static``
    chooses how the static (Kelvin) terms are integrated: 'quadrature' (Duffy and Gauss rules of order
    n_q) or 'closed' (the master integrals of ``moments``; touching offsets only).  The dynamic remainder
    is integrated by quadrature in both.

    Raises:
        ValueError: when ``static`` is neither 'quadrature' nor 'closed'.
    """
    if static not in ("quadrature", "closed"):
        raise ValueError(f"near_block: static must be 'quadrature' or 'closed', got {static!r}")
    key = (
        tuple(int(o) for o in offset),
        h,
        omega,
        ref.alpha,
        ref.beta,
        ref.rho,
        n_q,
        n_source,
        n_test,
        static,
    )
    if key in _NEAR_CACHE:
        return _NEAR_CACHE[key]
    out = _sform(
        offset,
        h,
        (0, 0, 0),
        lambda X: kernel_9x9(X, omega, ref, static=False).reshape(len(X), 81),
        n_q,
        n_source,
        n_test,
    ).reshape(n_test, n_source, 9, 9)
    for (m, idx), coef in static_term_table(ref.alpha, ref.beta, ref.rho).items():
        if static == "closed":
            term = static_term_integral_closed(m, idx, offset, h, n_source, n_test)
        else:
            term = static_term_integral(m, idx, offset, h, n_q, n_source, n_test)
        out += term[:, :, None, None] * coef[None, None]
    out = _to_field_rows(out)
    _NEAR_CACHE[key] = out
    return out


def coupling_block(
    offset: tuple[int, int, int],
    h: float,
    omega: float,
    ref: ReferenceMedium,
    n_source: int = 10,
    n_test: int = 4,
) -> NDArray:
    """K[a, c] for any offset: near_block when touching, the s-form with the full kernel otherwise."""
    if max(abs(o) for o in offset) <= 1:
        return near_block(offset, h, omega, ref, n_source=n_source, n_test=n_test)
    out = _sform(
        offset,
        h,
        (0, 0, 0),
        lambda X: kernel_9x9(X, omega, ref).reshape(len(X), 81),
        gauss_order(offset),
        n_source,
        n_test,
    )
    return _to_field_rows(out.reshape(n_test, n_source, 9, 9))
