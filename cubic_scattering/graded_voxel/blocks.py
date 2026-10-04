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

import functools
import itertools
import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from functools import cache, lru_cache

import numpy as np
import sympy as sp
from numpy.polynomial import Polynomial, legendre
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray

from ..effective_contrasts import ReferenceMedium
from .basis import SOURCE_EXPONENTS, field_in_monomials, monomials, source_exponents
from .derivatives import derivative_terms
from .kernel import _assemble, falling, kernel_9x9, power_F, radial_component, static_b2, voigt_maps
from .legendre_moments import box_moments


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
# The static terms in closed form, evaluated stably
# ---------------------------------------------------------------------------

#: Legendre degrees beyond a piece's own polynomial degree kept in the moment systems (truncation margin)
LEGENDRE_MARGIN = 16
#: Legendre coefficients kept per face axis for the polynomial a pyramid leaves on its far face
PYRAMID_DEGREE = 16
#: Gauss points per face axis projecting that polynomial onto Legendre polynomials (exact for its degree)
PYRAMID_GAUSS = 24


def _holds_sstar(kinds: list[Part], sstar: list[float]) -> bool:
    """Whether the closure of a piece of W (or a plane-delta face) holds the singular point s*.

    Raises:
        ValueError: when s* lies inside a piece without being a vertex (the pieces' corners sit on the
            breakpoints {-2, 0, 2}, so this cannot happen for touching cells).
    """
    for kind, s in zip(kinds, sstar, strict=True):
        lo, hi = kind[1], kind[2]
        if not lo <= s <= hi:
            return False
        if kind[0] != "delta" and s not in (lo, hi):
            raise ValueError("_holds_sstar: the singular point lies inside a piece, not at a vertex")
    return True


def _padded(c: NDArray, size: int) -> NDArray:
    out = np.zeros(size)
    out[: min(len(c), size)] = c[:size]
    return out


def _far_piece(
    kinds: list[Part],
    choice: tuple[int, ...],
    parts: dict,
    axis_exps: list[tuple[int, int]],
    tst: Sequence[tuple[int, int, int]],
    src: Sequence[tuple[int, int, int]],
    sstar: list[float],
    kernel: list[tuple[float, tuple[int, int, int], int]],
    n_leg: int,
) -> NDArray:
    """A piece of W (or plane-delta face) away from the singular point, by Legendre modified moments.

    In t = sigma - s* the piece is a box of side 2 in one orthant (or a plane at |t_i| = |b - s*_i|);
    reflected to t >= 0 and scaled by t = 2 t~, it is the unit box prod [l_j, l_j + 1]. The weight
    w_j(sigma) on each free axis is written as a Legendre series in that box's own coordinate
    x_j = 2 (t~_j - l_j) - 1, the kernel monomial t^alpha |t|^n by multiplying it in the Legendre basis by
    t~_j^alpha_j, and the integral is the contraction with ``legendre_moments.box_moments`` of S^n,
    S^2 = c2 + |t~_free|^2.
    """
    n_test, n_source = len(tst), len(src)
    free = [i for i in range(3) if kinds[i][0] != "delta"]
    fixed = [i for i in range(3) if kinds[i][0] == "delta"]
    sign: dict[int, float] = {}
    low: dict[int, float] = {}
    for i in free:
        a, b = kinds[i][1] - sstar[i], kinds[i][2] - sstar[i]
        sign[i] = 1.0 if a >= 0 else -1.0
        low[i] = min(abs(a), abs(b)) / 2.0
    t_fixed = {i: kinds[i][1] - sstar[i] for i in fixed}
    c2 = sum((t_fixed[i] / 2.0) ** 2 for i in fixed)
    wleg: dict[tuple[int, int, int], NDArray] = {}
    for i in free:
        to_x = Polynomial([sstar[i] + sign[i] * (2.0 * low[i] + 1.0), sign[i]])  # sigma as a function of x
        for et, es in axis_exps:
            poly = parts[(i, et, es)][choice[i]][3]
            wleg[(i, et, es)] = _padded(legendre.poly2leg(poly(to_x).coef), n_leg)
    fixed_val = np.ones((n_test, n_source))
    for i in fixed:
        fixed_val = fixed_val * np.array(
            [[float(parts[(i, ta[i], sc[i])][choice[i]][3]) for sc in src] for ta in tst]
        )
    out = np.zeros((n_test, n_source))
    for coef, alpha, n in kernel:
        lam = box_moments(tuple(low[i] for i in free), c2, n, n_leg)
        scale = coef * 2.0 ** (n + len(free))  # |t|^n = 2^n |t~|^n, dt = 2^d dt~
        for i in fixed:
            scale *= t_fixed[i] ** alpha[i]
        vecs = []
        for i in free:
            tpow = legendre.poly2leg((Polynomial([low[i] + 0.5, 0.5]) ** alpha[i]).coef)
            fac = (2.0 * sign[i]) ** alpha[i]
            vecs.append(
                np.array(
                    [
                        [_padded(legendre.legmul(wleg[(i, ta[i], sc[i])], tpow), n_leg) * fac for sc in src]
                        for ta in tst
                    ]
                )
            )
        if len(free) == 3:
            val = np.einsum("acx,acy,acz,xyz->ac", *vecs, lam)
        else:
            val = np.einsum("acx,acy,xy->ac", *vecs, lam)
        out += scale * fixed_val * val
    return out


def _vertex_piece(
    kinds: list[Part],
    choice: tuple[int, ...],
    parts: dict,
    axis_exps: list[tuple[int, int]],
    tst: Sequence[tuple[int, int, int]],
    src: Sequence[tuple[int, int, int]],
    sstar: list[float],
    kernel: list[tuple[float, tuple[int, int, int], int]],
    orders: tuple[int, int, int],
) -> NDArray:
    """A piece of W (or plane-delta face) with a corner at the singular point, by Duffy pyramids about it.

    Reflected and scaled (t = 2 t~), the piece is [0,1]^d with d = 3 free axes, or d = 2 for a plane-delta
    face through the singular point. Pyramid k: t~_k = rho, t~_j = rho v_j, v in [0,1]^(d-1),
    dt~ = rho^(d-1) d rho dv and |t~|^n = rho^n (1 + |v|^2)^(n/2), so

        int P(t~) |t~|^n dt~ = sum_k int_v (1 + |v|^2)^(n/2) G_k(v) dv,
        G_k(v) = sum_p c_p(v) / (p + n + d),

    c_p(v) the coefficient of rho^p in P(rho e_k(v)), formed by exact polynomial products at the nodes v.
    The radial integral is exact; G_k is a polynomial on the far face of the pyramid, at distance 1 from
    the singular point, projected onto Legendre polynomials by an exact Gauss rule and contracted with the
    face moments ``box_moments((0,) * (d - 1), 1, n)``. The coefficients of rho^p that vanish because W
    vanishes at the vertex (an outer end sigma = +-2 of the piece, to order 1 - n_der) are set exactly to
    zero, so p + n + d > 0 wherever c_p is not zero.

    Raises:
        ValueError: if a non-zero c_p meets p + n + d <= 0 (a divergent radial integral).
    """
    n_test, n_source = len(tst), len(src)
    free = [i for i in range(3) if kinds[i][0] != "delta"]
    fixed = [i for i in range(3) if kinds[i][0] == "delta"]
    d = len(free)
    sign: dict[int, float] = {}
    vanish: dict[int, int] = {}
    for i in free:
        sign[i] = 1.0 if kinds[i][2] - sstar[i] > 0 else -1.0
        vanish[i] = max(0, 1 - orders[i]) if abs(sstar[i]) == 2.0 else 0
    wpow: dict[tuple[int, int, int], NDArray] = {}
    for i in free:
        to_t = Polynomial([sstar[i], 2.0 * sign[i]])  # sigma = s* + 2 s t~
        for et, es in axis_exps:
            c = parts[(i, et, es)][choice[i]][3](to_t).coef.copy()
            c[: vanish[i]] = 0.0
            wpow[(i, et, es)] = c
    fixed_val = np.ones((n_test, n_source))
    for i in fixed:
        fixed_val = fixed_val * np.array(
            [[float(parts[(i, ta[i], sc[i])][choice[i]][3]) for sc in src] for ta in tst]
        )
    v_nodes, v_wts = _gauss01(PYRAMID_GAUSS)
    p_at = np.array(
        [legendre.legval(2.0 * v_nodes - 1.0, np.eye(PYRAMID_DEGREE)[k]) for k in range(PYRAMID_DEGREE)]
    )
    norm = 2.0 * np.arange(PYRAMID_DEGREE) + 1.0
    out = np.zeros((n_test, n_source))
    for coef, alpha, n in kernel:
        if any(alpha[i] > 0 for i in fixed):
            continue  # t_i = 0 on a plane-delta face through the singular point
        scale = coef * 2.0 ** (n + sum(alpha[i] for i in free) + d)
        for i in free:
            scale *= sign[i] ** alpha[i]
        fco: dict[int, NDArray] = {}
        for i in free:
            size = max(len(wpow[(i, et, es)]) for et, es in axis_exps) + alpha[i]
            arr = np.zeros((n_test, n_source, size))
            for a, ta in enumerate(tst):
                for c, sc in enumerate(src):
                    w = wpow[(i, ta[i], sc[i])]
                    arr[a, c, alpha[i] : alpha[i] + len(w)] = w
            fco[i] = arr
        lam = box_moments((0.0,) * (d - 1), 1.0, n, PYRAMID_DEGREE + LEGENDRE_MARGIN)
        total = np.zeros((n_test, n_source))
        for apex in free:
            others = [i for i in free if i != apex]
            grids = np.meshgrid(*([v_nodes] * len(others)), indexing="ij")
            vv = [g.ravel() for g in grids]
            poly = fco[apex][:, :, None, :] * np.ones((1, 1, len(vv[0]), 1))
            for j, v in zip(others, vv, strict=True):
                fj = fco[j][:, :, None, :] * (v[None, None, :, None] ** np.arange(fco[j].shape[2]))
                p1, p2 = poly.shape[3], fj.shape[3]
                prod = np.zeros(poly.shape[:3] + (p1 + p2 - 1,))
                for q in range(p2):
                    prod[..., q : q + p1] += poly * fj[..., q : q + 1]
                poly = prod
            denom = np.arange(poly.shape[3]) + n + d
            ok = denom > 0
            if np.any(poly[..., ~ok] != 0.0):
                raise ValueError(
                    f"_vertex_piece: a non-zero rho^p with p + n + d <= 0 (n = {n}, d = {d}): the radial "
                    "integral diverges; the vanishing order of W at the vertex is wrong for this piece."
                )
            g = (poly[..., ok] / denom[ok]).sum(axis=3)  # (a, c, nodes)
            if len(others) == 1:
                g_leg = np.einsum("acg,kg,g->ack", g, p_at, v_wts) * norm
                total += np.einsum("ack,k->ac", g_leg, lam[:PYRAMID_DEGREE])
            else:
                gr = g.reshape(n_test, n_source, PYRAMID_GAUSS, PYRAMID_GAUSS)
                g_leg = np.einsum("acgh,kg,lh,g,h->ackl", gr, p_at, p_at, v_wts, v_wts) * np.outer(
                    norm, norm
                )
                total += np.einsum("ackl,kl->ac", g_leg, lam[:PYRAMID_DEGREE, :PYRAMID_DEGREE])
        out += scale * fixed_val * total
    return out


def static_term_integral_closed(
    m: int,
    idx: tuple[int, ...],
    offset: tuple[int, int, int],
    h: float,
    n_source: int = 10,
    n_test: int = 4,
) -> NDArray:
    """``static_term_integral`` with no quadrature of the kernel: every piece in closed form, stably.

    Valid for every odd m >= -1 (the static terms are m = -1 and m = 1; the odd terms of the dynamic series
    are m = 1, 3, 5, ...). After the moves the kernel d^rest r^m is a sum of monomials times odd powers
    |t|^n (``radial_monomials``), t = (R + s) / h measured from the singular point. Each piece of W is a
    polynomial on a box with corners on the breakpoints, or (a plane delta) on a face of one:

    * a piece whose closure holds the singular point (a corner of it) by Duffy pyramids about that corner:
      the radial integral exactly, the rest by Legendre moments on a face at distance 1 (``_vertex_piece``);
    * every other piece by Legendre modified moments on its own box (``_far_piece``,
      ``legendre_moments.box_moments``).

    WHY NOT THE ORIGIN-ANCHORED MASTER INTEGRALS. The master integrals of ``moments`` are anchored at the
    singular point; a piece away from it is then a signed sum of origin-anchored boxes, with W expanded in
    powers of t about the singular point. Both cancel: the power moments reach 1e19 against a piece of order
    one, and in double precision the static block lost 4 to 5 digits (1.7e-11 for a quadratic field at the
    corner neighbour against the 40-digit reference of Mathematica/GradedVoxel_CornerReference.wl). The
    Legendre moments are well conditioned, and the pyramid leaves only a face away from the singular point.
    The blocks now agree with that reference to round-off (3e-15 quadratic, 1e-15 linear), as quadrature
    does.

    Raises:
        ValueError: when the offset does not touch (the closed forms are tabulated for the near cells), or
            the term needs more than two derivatives moved onto W, or two plane deltas meet on one piece.
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
    sstar = [-2.0 * o for o in offset]  # the singular point in sigma = s / h
    deg = max(len(pl[3].coef) for plist in parts.values() for pl in plist if isinstance(pl[3], Polynomial))
    n_leg = deg + 4 + LEGENDRE_MARGIN  # W's degree + the kernel monomial's (|alpha| <= 4) + the margin
    out = np.zeros((n_test, n_source))
    for choice in itertools.product(*[range(n) for n in shape]):
        kinds = [parts[(i, *axis_exps[0])][choice[i]] for i in range(3)]
        n_fixed = sum(1 for kind in kinds if kind[0] == "delta")
        if n_fixed > 1:
            raise ValueError("static_term_integral_closed: two plane deltas on one piece")
        if _holds_sstar(kinds, sstar):
            piece = _vertex_piece(kinds, choice, parts, axis_exps, tst, src, sstar, kernel, orders)
        else:
            piece = _far_piece(kinds, choice, parts, axis_exps, tst, src, sstar, kernel, n_leg)
        # physical units: h per free axis, and d^rest r^m is h^(m - |rest|) times its value in t
        out += h ** (3 - n_fixed + m - len(rest)) * piece
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
    is that of exp(i k r) and converges for every k. The weight c2 loses two powers of k to the division
    by k_S^2, so its term m is as large as the c1 term m - 2 (the radiation part at order k comes from
    m = 0 and m = 2 together); summed through m = 3 at least, and until the bound
    (k_S * 4 sqrt(3) h)^(m - 1) / (m - 1)! falls below ``tol`` relative to the first term.

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
        if m > 3 and reach ** (m - 1) / math.factorial(m - 1) < tol:
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


# ---------------------------------------------------------------------------
# Far blocks as a series in the wavenumber
# ---------------------------------------------------------------------------

#: the highest power t of k kept in ``far_series_coefficients``
FAR_SERIES_T_MAX = 48
#: powers t taken through one s-form at a time (bounds the kernel array of a piece)
_FAR_T_CHUNK = 8


def _power_derivative_tensors(X: NDArray, m: int) -> list[NDArray]:
    """d^a r^m at separations X for |a| = 0 .. 4, tensors of shape (N,), (N, 3) .. (N, 3, 3, 3, 3)."""
    r = np.linalg.norm(X, axis=1)
    values: dict[tuple[int, int, int], NDArray] = {}
    out = []
    for order in range(5):
        tensor = np.zeros((len(X),) + (3,) * order)
        for idx in itertools.product(range(3), repeat=order):
            a = (idx.count(0), idx.count(1), idx.count(2))
            if a not in values:
                acc = np.zeros(len(X))
                for coef, (e0, e1, e2), q in derivative_terms(a):
                    c = coef * falling(m, q)
                    if c != 0.0:
                        acc += c * X[:, 0] ** e0 * X[:, 1] ** e1 * X[:, 2] ** e2 * r ** float(m - 2 * q)
                values[a] = acc
            tensor[(slice(None), *idx)] = values[a]  # type: ignore[index]
        out.append(tensor)
    return out


def _far_family_kernels(X: NDArray, ts: range) -> NDArray:
    """The 9 x 9 propagator of the two families G = delta_ij r^(t-1) and G = d_i d_j r^(t-1), (N, Z).

    Z runs over (t, family, 81) for t in ts; ``kernel._assemble`` is linear in (G, d G, d d G).
    """
    eye = np.eye(3)
    cols = []
    for t in ts:
        f0, f1, f2, f3, f4 = _power_derivative_tensors(X, t - 1)
        g_a = (
            eye * f0[:, None, None],
            eye[None, :, :, None] * f1[:, None, None, :],
            eye[None, :, :, None, None] * f2[:, None, None, :, :],
        )
        g_b = (f2, f3, f4)
        for g, gd, gdd in (g_a, g_b):
            cols.append(_assemble(g, gd, gdd).real.reshape(len(X), 81))
    return np.concatenate(cols, axis=1)


@cache
def _far_component_order() -> NDArray:
    """The number of derivatives of G in each of the 81 components of the 9 x 9 propagator."""
    e = np.zeros((9, 9))
    e[:3, 3:] = e[3:, :3] = 1.0
    e[3:, 3:] = 2.0
    return e.ravel()


@cache
def far_series_coefficients(offset: tuple[int, int, int], n_source: int = 10, n_test: int = 4) -> NDArray:
    """S[t, family, a, c, z]: the s-form on the UNIT half-side (h = 1) of the two families of
    ``_far_family_kernels``, for t = 0 .. FAR_SERIES_T_MAX, shape (T + 1, 2, n_test, n_source, 81).

    Independent of the frequency and of the cell size, so computed once per offset. Taken with the Gauss
    rule of ``gauss_order``: quadrature is linear in the kernel, so the series summed with these
    coefficients equals the Gauss block of ``coupling_block`` up to the truncation of the series.

    Raises:
        ValueError: when the offset touches (``gauss_order``).
    """
    n_q = gauss_order(offset)
    t_all = FAR_SERIES_T_MAX + 1
    out = np.zeros((t_all, 2, n_test, n_source, 81))
    for t0 in range(0, t_all, _FAR_T_CHUNK):
        ts = range(t0, min(t0 + _FAR_T_CHUNK, t_all))
        kernel_at = functools.partial(_far_family_kernels, ts=ts)
        part = _sform(offset, 1.0, (0, 0, 0), kernel_at, n_q, n_source, n_test)
        out[t0 : ts.stop] = np.moveaxis(part.real.reshape(n_test, n_source, len(ts), 2, 81), (2, 3), (0, 1))
    return out


def far_block_series(
    offset: tuple[int, int, int],
    h: float,
    omega: float,
    ref: ReferenceMedium,
    n_source: int = 10,
    n_test: int = 4,
    tol: float = 1e-14,
) -> NDArray:
    """K[a, c] of a non-touching offset as the power series of the propagator in the wavenumber.

        K = sum_t [ c1(t) S_A(t) + c2(t) S_B(t) ],   S(t; h) = h^(6 + (t - 1) - |a|) S(t; 1),

    c1 and c2 the weights of ``series_weights`` (m = t - 1), S the coefficients of
    ``far_series_coefficients`` and |a| the number of derivatives on r^(t - 1) in each component (two more
    for the family d_i d_j). Each further frequency costs only this sum. The weight c2 loses two powers of k
    to the division by k_S^2, so its term t is as large as the c1 term t - 2: summed through t = 4 at least,
    and until (k_S r_max)^(t - 2) / (t - 2)! falls below tol, r_max the largest separation of two points of
    the cells.

    Raises:
        ValueError: when the offset touches, or FAR_SERIES_T_MAX terms do not reach tol at this frequency.
    """
    coeffs = far_series_coefficients(tuple(offset), n_source, n_test)
    r_max = (2.0 * float(np.linalg.norm(offset)) + 2.0 * np.sqrt(3.0)) * h
    k_s = omega / ref.beta
    n_terms = None
    for t in range(FAR_SERIES_T_MAX + 1):
        if t > 4 and (abs(k_s) * r_max) ** (t - 2) / math.factorial(t - 2) < tol:
            n_terms = t
            break
    if n_terms is None:
        raise ValueError(
            f"far_block_series: k_S r = {abs(k_s) * r_max:.2f} needs more than {FAR_SERIES_T_MAX} terms "
            f"for tol = {tol:g}. Fix: use coupling_block (Gauss) at this frequency."
        )
    w = np.array([series_weights(t - 1, omega, ref) for t in range(n_terms)])  # (T, 2)
    w = w * (h ** (5.0 + np.arange(n_terms)))[:, None] * np.array([1.0, h**-2])[None, :]
    out = np.tensordot(w, coeffs[:n_terms], axes=([0, 1], [0, 1])) * h ** (-_far_component_order())
    return _to_field_rows(out.reshape(n_test, n_source, 9, 9))


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
