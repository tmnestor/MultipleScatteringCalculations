"""Galerkin coupling blocks K_ac(R) of the graded voxel.

K[a, c](R) = int_{V_m} int_{V_n} L_a((x - x_m)/h) P(x - x') m_c((x' - x_n)/h) dx dx',  R = x_m - x_n,

a 9 x 9 block for each test function a (4) and source monomial c (10).

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
from collections import defaultdict
from collections.abc import Callable
from functools import cache, lru_cache

import numpy as np
import sympy as sp
from numpy.polynomial import Polynomial
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray

from ..effective_contrasts import ReferenceMedium
from .basis import SOURCE_EXPONENTS, TEST_EXPONENTS, monomials
from .kernel import kernel_9x9, power_F, radial_component, static_b2, voigt_maps


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
    offset: tuple[int, int, int], h: float, omega: float, ref: ReferenceMedium, n_gauss: int
) -> NDArray:
    """K[a, c] for a non-touching offset, shape (4, 10, 9, 9)."""
    xi, w = _cell_rule(n_gauss)
    R = 2.0 * h * np.asarray(offset, dtype=float)
    X = (R[None, None, :] + h * (xi[:, None, :] - xi[None, :, :])).reshape(-1, 3)
    P = kernel_9x9(X, omega, ref).reshape(len(xi), len(xi), 81)
    lt = monomials(TEST_EXPONENTS, xi) * w
    ls = monomials(SOURCE_EXPONENTS, xi) * w
    tmp = np.einsum("ap,pqz->aqz", lt, P)
    return (h**6 * np.einsum("cq,aqz->acz", ls, tmp)).reshape(4, 10, 9, 9)


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


_PAIRS = [(a, c) for a in range(4) for c in range(10)]
_AXIS_EXPS = sorted({(TEST_EXPONENTS[a][i], SOURCE_EXPONENTS[c][i]) for a, c in _PAIRS for i in range(3)})


def _sform(
    offset: tuple[int, int, int],
    h: float,
    orders: tuple[int, int, int],
    kernel_at: Callable[[NDArray], NDArray],
    n_q: int,
) -> NDArray:
    """sum over pieces of int prod_i d^orders_i w_i(s_i) K(R + s) ds for all (a, c), shape (4, 10, Z).

    kernel_at(X) returns (N, Z) kernel values at the separations X = R + s.
    """
    R = 2.0 * h * np.asarray(offset, dtype=float)
    sstar = -R / h  # the singular point, in units of h
    parts = {(i, et, es): _axis_parts(et, es, orders[i], h) for i in range(3) for et, es in _AXIS_EXPS}
    shape = [len(parts[(i, *_AXIS_EXPS[0])]) for i in range(3)]
    out: NDArray | None = None
    for choice in itertools.product(*[range(s) for s in shape]):
        kinds = [parts[(i, *_AXIS_EXPS[0])][choice[i]] for i in range(3)]
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
            out = np.zeros((4, 10, kv.shape[1]), dtype=complex)
        jac = w * h ** len(free)
        axis_vals = {}
        for (i, et, es), plist in parts.items():
            kind = plist[choice[i]]
            poly = kind[3]
            axis_vals[(i, et, es)] = (
                np.full(len(sig), float(poly)) if kind[0] == "delta" else poly(sig[:, i])  # type: ignore[operator, arg-type]
            )
        for a, c in _PAIRS:
            val = jac.copy()
            for i in range(3):
                val = val * axis_vals[(i, TEST_EXPONENTS[a][i], SOURCE_EXPONENTS[c][i])]
            out[a, c] += val @ kv
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
        i, j, k, l = (int(v) for v in np.unravel_index(z, (3, 3, 3, 3)))
        for al in range(6):
            for be in range(6):
                if ms[al, be, z]:
                    add(3 + al, 3 + be, (k, l), i, j, ms[al, be, z])
    return dict(table)


def _moves(m: int, idx: tuple[int, ...]) -> int:
    if m >= 0 and m % 2 == 0:
        return 0
    return max(0, len(idx) - (m + 1))


def static_term_integral(
    m: int, idx: tuple[int, ...], offset: tuple[int, int, int], h: float, n_q: int
) -> NDArray:
    """int W_ac(s) d^idx r^m (R + s) ds for all (a, c), shape (4, 10), distributionally."""
    k = _moves(m, idx)
    moved, rest = idx[:k], idx[k:]
    orders = (moved.count(0), moved.count(1), moved.count(2))

    def kern(X: NDArray) -> NDArray:
        return radial_component(power_F(m, np.linalg.norm(X, axis=1)), X, rest)[:, None]

    return (-1) ** k * _sform(offset, h, orders, kern, n_q)[:, :, 0]


_NEAR_CACHE: dict[tuple, NDArray] = {}


def near_block(
    offset: tuple[int, int, int], h: float, omega: float, ref: ReferenceMedium, n_q: int = 12
) -> NDArray:
    """K[a, c] by the s-form with the distributional static part, shape (4, 10, 9, 9).

    Valid for any offset; required for the self cell and its touching neighbours. Cached.
    """
    key = (tuple(int(o) for o in offset), h, omega, ref.alpha, ref.beta, ref.rho, n_q)
    if key in _NEAR_CACHE:
        return _NEAR_CACHE[key]
    out = _sform(
        offset, h, (0, 0, 0), lambda X: kernel_9x9(X, omega, ref, static=False).reshape(len(X), 81), n_q
    ).reshape(4, 10, 9, 9)
    for (m, idx), coef in static_term_table(ref.alpha, ref.beta, ref.rho).items():
        out += static_term_integral(m, idx, offset, h, n_q)[:, :, None, None] * coef[None, None]
    _NEAR_CACHE[key] = out
    return out


def coupling_block(offset: tuple[int, int, int], h: float, omega: float, ref: ReferenceMedium) -> NDArray:
    """K[a, c] for any offset: near_block when touching, the s-form with the full kernel otherwise."""
    if max(abs(o) for o in offset) <= 1:
        return near_block(offset, h, omega, ref)
    out = _sform(
        offset, h, (0, 0, 0), lambda X: kernel_9x9(X, omega, ref).reshape(len(X), 81), gauss_order(offset)
    )
    return out.reshape(4, 10, 9, 9)
