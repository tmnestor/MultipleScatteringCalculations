"""Distant cells: the coupling block as a multipole series about the centre separation, with no quadrature.

For two cells whose centres are R apart the kernel is analytic over the whole range of s = u - u', and

    K_ac(R) = int int L_a(u) P(R + u - u') m_c(u') du du'
            = sum_gamma  (h^(6 + |gamma|) / gamma!)  mu_ac^gamma  d^gamma P(R),

    mu_ac^gamma = int int xi^(e_a) (xi - xi')^gamma xi'^(e_c) d xi d xi'   over [-1, 1]^3 x [-1, 1]^3,

the moments of the two cells' polynomials: products of one-dimensional integrals of monomials, rational
numbers (``cell_pair_moments``).  The derivatives of the propagator at R follow from those of two radial
scalars, g_S and (g_S - g_P) / k_S^2, through the fixed tables of ``blocks.family_tables``:

    d^gamma P = (1 / 4 pi mu) sum_idx (TA[idx] d^(gamma + idx) g_S
                                      + TB[idx] d^(gamma + idx) (g_S - g_P) / k_S^2).

A derivative of any order of a radial function f is a sum of monomials times F_q = (r^-1 d/dr)^q f, by
d_i (x^alpha F_q) = alpha_i x^(alpha - e_i) F_q + x^(alpha + e_i) F_(q+1) (``derivative_table``), and for
g_k = exp(i k r) / r the F_q are spherical Hankel functions, F_q = i (-1)^q k^(q+1) h_q(k r) / r^q.

CONVERGENCE.  |s| <= 2 sqrt(3) h.  Each derivative of exp(i k r) / r at R brings a factor 1 / R or k, so
with rho = 2 sqrt(3) h / R and kappa = 2 sqrt(3) k_S h the term of order n is bounded by
b_n = sum_j rho^(n - j) kappa^j / j!: geometric in the ratio of the cells' extent to their distance, and
factorially small in the cell size over the wavelength.  It needs rho < 1 and is practical from about
four cells apart; ``truncation_order`` picks the order for a tolerance.  There is no restriction on k R:
this is the treatment for distant cells in a large body, where a power series in the wavenumber is
useless.

EVERY NON-TOUCHING PAIR (``piecewise_multipole_block``).  Close in, the series about the cell centres
converges too slowly: at two cells apart rho = 0.87.  The cross-correlation W of the two cells'
polynomials is itself a polynomial on each of the eight boxes into which the planes s_i = 0 cut its
support [-2h, 2h]^3 (the s-form of ``blocks``), so

    K_ac(R) = sum over boxes B of  int_B W_ac(s) P(R + s) ds,

and on each box the propagator is expanded about the box's own centre s_B: the moments are those of a
polynomial over a box, exact, and the ratio is now (half-diagonal of the box) / |R + s_B|, the distance
from the box's centre to the singular point.  The boxes may be bisected further (W is still a polynomial
on every part): each bisection halves the half-diagonal.  At two cells apart the ratio is 0.52 on the
eight boxes and 0.33 after one bisection.  This is one treatment for every pair of cells that do not
touch; for distant cells it reduces to the series above with a smaller ratio.
"""

import itertools
import math
from functools import cache

import numpy as np
from numpy.polynomial import Polynomial
from numpy.typing import NDArray
from scipy.special import spherical_jn, spherical_yn

from ..effective_contrasts import ReferenceMedium
from .basis import SOURCE_EXPONENTS, moment_1d, source_exponents
from .blocks import _to_field_rows, autocorrelation_1d, family_tables
from .kernel import falling

Beta = tuple[int, int, int]

#: The series is refused when (2 sqrt 3 h) / R exceeds this: it would need hundreds of orders.
MAX_RATIO = 0.6


def helmholtz_F(k: float, r: float, q_max: int) -> NDArray:
    """F_q = (r^-1 d/dr)^q [exp(i k r) / r], q = 0..q_max, as i (-1)^q k^(q+1) h_q(k r) / r^q."""
    q = np.arange(q_max + 1)
    h = spherical_jn(q, k * r) + 1j * spherical_yn(q, k * r)
    return 1j * (-1.0) ** q * k ** (q + 1.0) * h / r**q


#: Below k_S r = this, the derivatives of B = (g_S - g_P) / k_S^2 come from its power series; above it the
#: difference of the two Hankel-function forms loses at most two digits.
B_SERIES_LIMIT = 1.0
#: Terms of that power series (k_S r <= 1: the last is below 1e-60 of the first)
B_SERIES_TERMS = 48


def b_scalar_F(ks: float, kp: float, r: float, q_max: int) -> NDArray:
    """F_q = (r^-1 d/dr)^q B, B = (g_S - g_P) / k_S^2, q = 0..q_max, without cancellation.

    The difference of the two Hankel forms of ``helmholtz_F`` divided by k_S^2 cancels as k r -> 0 (its
    relative error is about eps / (k r)^2). For k_S r <= B_SERIES_LIMIT the series is used instead:
    B = sum_t c2(t) r^(t-1), c2(t) = ((i k_S)^t - (i k_P)^t) / (t! k_S^2), and
    F_q(r^m) = m (m - 2) ... (m - 2q + 2) r^(m - 2q), every term computed directly.
    """
    if ks * r > B_SERIES_LIMIT:
        return (helmholtz_F(ks, r, q_max) - helmholtz_F(kp, r, q_max)) / ks**2
    out = np.zeros(q_max + 1, dtype=complex)
    for t in range(1, B_SERIES_TERMS):
        c2 = ((1j * ks) ** t - (1j * kp) ** t) / (math.factorial(t) * ks**2)
        m = t - 1
        for q in range(q_max + 1):
            f = falling(m, q)
            if f != 0.0:
                out[q] += c2 * f * r ** (m - 2 * q)
    return out


@cache
def derivative_table(order: int) -> dict[Beta, tuple[tuple[Beta, int, float], ...]]:
    """For every multi-index beta with |beta| <= order: d^beta f(r) = sum of coef * x^alpha * F_q."""
    table: dict[Beta, dict[tuple[Beta, int], float]] = {(0, 0, 0): {((0, 0, 0), 0): 1.0}}
    for total in range(1, order + 1):
        for b0 in range(total + 1):
            for b1 in range(total - b0 + 1):
                beta = (b0, b1, total - b0 - b1)
                i = next(j for j in range(3) if beta[j] > 0)
                low = (beta[0] - (i == 0), beta[1] - (i == 1), beta[2] - (i == 2))
                new: dict[tuple[Beta, int], float] = {}
                for (alpha, q), coef in table[low].items():
                    if alpha[i] > 0:
                        down = (alpha[0] - (i == 0), alpha[1] - (i == 1), alpha[2] - (i == 2))
                        new[(down, q)] = new.get((down, q), 0.0) + coef * alpha[i]
                    up = (alpha[0] + (i == 0), alpha[1] + (i == 1), alpha[2] + (i == 2))
                    new[(up, q + 1)] = new.get((up, q + 1), 0.0) + coef
                table[beta] = new
    return {beta: tuple((alpha, q, c) for (alpha, q), c in terms.items()) for beta, terms in table.items()}


def radial_derivatives(k: float, x: NDArray, order: int, f: NDArray | None = None) -> dict[Beta, complex]:
    """d^beta f(r) at the point x, for every |beta| <= order; f(r) = exp(i k r) / r unless its F_q are
    given."""
    x = np.asarray(x, dtype=float)
    if f is None:
        f = helmholtz_F(k, float(np.linalg.norm(x)), order)
    powers = [[float(x[i]) ** e for e in range(order + 1)] for i in range(3)]
    out: dict[Beta, complex] = {}
    for beta, terms in derivative_table(order).items():
        out[beta] = sum(
            c * powers[0][al[0]] * powers[1][al[1]] * powers[2][al[2]] * f[q] for al, q, c in terms
        )
    return out


def _pair_moment_1d(e_t: int, e_s: int, g: int) -> float:
    """int int u^e_t (u - v)^g v^e_s du dv over [-1, 1]^2."""
    return sum(
        math.comb(g, j) * (-1.0) ** (g - j) * moment_1d(e_t + j) * moment_1d(e_s + g - j)
        for j in range(g + 1)
    )


@cache
def cell_pair_moments(order: int, n_source: int = 10, n_test: int = 4) -> dict[Beta, NDArray]:
    """mu^gamma[a, c] for every |gamma| <= order, cells of half-width 1; a over the test MONOMIALS."""
    tst, src = SOURCE_EXPONENTS[:n_test], source_exponents(n_source)
    out: dict[Beta, NDArray] = {}
    for g0, g1, g2 in itertools.product(range(order + 1), repeat=3):
        if g0 + g1 + g2 > order:
            continue
        gam = (g0, g1, g2)
        out[gam] = np.array(
            [
                [math.prod(_pair_moment_1d(ta[i], sc[i], gam[i]) for i in range(3)) for sc in src]
                for ta in tst
            ]
        )
    return out


def truncation_order(offset: tuple[int, int, int], tol: float, kappa: float = 0.0) -> int:
    """The smallest order n whose next term bound b_(n+1) = sum_j rho^(n+1-j) kappa^j / j! is below tol.

    rho = sqrt(3) / |offset| is the ratio of the cells' extent to their distance, kappa = 2 sqrt(3) k_S h
    the cells' extent in units of the inverse wavenumber.

    Raises:
        ValueError: when the cells are too close for the series (rho above MAX_RATIO).
    """
    ratio = math.sqrt(3.0) / math.sqrt(sum(o * o for o in offset)) if any(offset) else math.inf
    if ratio > MAX_RATIO:
        raise ValueError(
            f"truncation_order: offset {offset} is too close for the multipole series: the ratio of the "
            f"cells' extent to their distance is {ratio:.2f}, above {MAX_RATIO}. Fix: use "
            "blocks.coupling_block for offsets within three cells."
        )
    n = 2
    while sum(ratio ** (n + 1 - j) * kappa**j / math.factorial(j) for j in range(n + 2)) >= tol:
        n += 1
    return n


def far_block_multipole(
    offset: tuple[int, int, int],
    h: float,
    omega: float,
    ref: ReferenceMedium,
    n_source: int = 10,
    n_test: int = 4,
    tol: float = 1e-12,
    order: int | None = None,
) -> NDArray:
    """K[a, c] between distant cells from the multipole series, shape (n_test, n_source, 9, 9).

    Args:
        offset: Integer offset between the cells (field minus source), in cell widths.
        h: Cell half-width (m).
        omega: Angular frequency (rad/s).
        ref: Background medium.
        n_source: Source monomials (10, 20 or 35).
        n_test: Field functions (4 or 10).
        tol: Relative tolerance that sets the order when ``order`` is None.
        order: Truncate the series at this total order instead.

    Raises:
        ValueError: when the cells are too close for the series.
    """
    kappa = 2.0 * math.sqrt(3.0) * omega / ref.beta * h
    n = truncation_order(offset, tol, kappa) if order is None else order
    if order is not None:
        truncation_order(offset, 0.5)  # the closeness check only
    big_r = 2.0 * h * np.asarray(offset, dtype=float)
    ka, kb = omega / ref.alpha, omega / ref.beta
    ds = radial_derivatives(kb, big_r, n + 4)
    db = radial_derivatives(kb, big_r, n + 4, f=b_scalar_F(kb, ka, float(np.linalg.norm(big_r)), n + 4))
    ta, tb = family_tables()
    mu = cell_pair_moments(n, n_source, n_test)
    out = np.zeros((n_test, n_source, 9, 9), dtype=complex)
    weights = {
        gam: h ** (6 + sum(gam)) / math.prod(math.factorial(g) for g in gam) * m for gam, m in mu.items()
    }
    for table, family in ((ta, "a"), (tb, "b")):
        for idx, coef in table.items():
            shift = (idx.count(0), idx.count(1), idx.count(2))
            acc = np.zeros((n_test, n_source), dtype=complex)
            for gam, w in weights.items():
                beta = (gam[0] + shift[0], gam[1] + shift[1], gam[2] + shift[2])
                d = ds[beta] if family == "a" else db[beta]
                acc += w * d
            out += acc[:, :, None, None] * coef[None, None]
    return _to_field_rows(out) / (4.0 * np.pi * ref.mu)


@cache
def _flat_derivative_table(order: int) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
    """``derivative_table`` flattened: (beta index, alpha0, alpha1, alpha2, q, coefficient) per term, the
    beta index being b0 * (order + 1)^2 + b1 * (order + 1) + b2."""
    n1 = order + 1
    rows = [
        (b[0] * n1 * n1 + b[1] * n1 + b[2], al[0], al[1], al[2], q, c)
        for b, terms in derivative_table(order).items()
        for al, q, c in terms
    ]
    arr = np.array(rows)
    return tuple(arr[:, j].astype(int) for j in range(5)) + (arr[:, 5],)  # type: ignore[return-value]


def radial_derivative_array(k: float, x: NDArray, order: int, f: NDArray | None = None) -> NDArray:
    """d^beta f(r) at x as an array D[b0, b1, b2], zero where |beta| > order; f(r) = exp(i k r) / r unless
    its F_q are given."""
    x = np.asarray(x, dtype=float)
    if f is None:
        f = helmholtz_F(k, float(np.linalg.norm(x)), order)
    ib, a0, a1, a2, q, coef = _flat_derivative_table(order)
    e = np.arange(order + 1)
    px, py, pz = (float(x[i]) ** e for i in range(3))
    terms = coef * px[a0] * py[a1] * pz[a2] * f[q]
    n1 = order + 1
    out = np.bincount(ib, weights=terms.real, minlength=n1**3) + 1j * np.bincount(
        ib, weights=terms.imag, minlength=n1**3
    )
    return out.reshape(n1, n1, n1)


@cache
def piece_moments_1d(e_t: int, e_s: int, splits: int, order: int) -> NDArray:
    """M[sub, g] = int over sub-interval `sub` of (w / h)(sigma) (sigma - centre)^g d sigma, g <= order.

    w / h is the one-dimensional cross-correlation of xi^e_t and xi^e_s (``blocks.autocorrelation_1d``),
    a polynomial on [-2, 0] and on [0, 2]; each half is cut into 2^splits equal sub-intervals.  With
    tau = sigma - centre and half-length r, int_{-r}^{r} tau^j d tau = 2 r^(j+1) / (j + 1) for even j.
    """
    left, right = autocorrelation_1d(e_t, e_s)
    n_sub = 2**splits
    r = 1.0 / n_sub
    out = np.zeros((2 * n_sub, order + 1))
    for half, poly in enumerate((left, right)):
        for i in range(n_sub):
            centre = -2.0 + 2.0 * half + (2 * i + 1) * r
            shifted = poly(Polynomial([centre, 1.0])).coef  # w(centre + tau) in powers of tau
            for g in range(order + 1):
                out[half * n_sub + i, g] = sum(
                    float(c) * 2.0 * r ** (j + g + 1) / (j + g + 1)
                    for j, c in enumerate(shifted)
                    if (j + g) % 2 == 0
                )
    return out


#: The highest series order a piece may use. At offset (2, 1, 1) the unbisected plan chose orders 48 to 56
#: and stalled at 6e-13 to 2e-12, while one bisection at orders 36 to 40 reached 1e-15 to 4e-15: the
#: rounding of a series this long grows with its order, so a plan that needs more bisects instead.
PIECE_MAX_ORDER = 40


def _piece_plan(offset: tuple[int, int, int], kappa: float, tol: float) -> tuple[int, int]:
    """(splits, order): the bisection level that makes the series cheapest, and its order.

    The ratio is that of the nearest sub-box: its half-diagonal sqrt(3) / 2^splits over the distance from
    its centre to the singular point, both in units of h.
    """
    best: tuple[float, int, int] | None = None
    for splits in range(4):
        r = 1.0 / 2**splits
        d = math.sqrt(sum(max(2 * abs(o) - 2 + r, r) ** 2 for o in offset))
        ratio = math.sqrt(3.0) * r / d
        kap = kappa * r / 2.0  # kappa was defined for the half-diagonal 2 sqrt(3) h
        n = 2
        while sum(ratio ** (n + 1 - j) * kap**j / math.factorial(j) for j in range(n + 2)) >= tol:
            n += 1
            if n > PIECE_MAX_ORDER:
                break
        if n > PIECE_MAX_ORDER:
            continue
        cost = 8.0 ** (splits + 1) * (n + 1) ** 3
        if best is None or cost < best[0]:
            best = (cost, splits, n)
    if best is None:
        raise ValueError(f"_piece_plan: no bisection level reaches tol = {tol:g} for offset {offset}")
    return best[1], best[2]


def piecewise_multipole_block(
    offset: tuple[int, int, int],
    h: float,
    omega: float,
    ref: ReferenceMedium,
    n_source: int = 10,
    n_test: int = 4,
    tol: float = 1e-12,
    splits: int | None = None,
    order: int | None = None,
) -> NDArray:
    """K[a, c] for any pair of cells that do not touch, from multipole series about the centres of the
    polynomial pieces of the cross-correlation W; no quadrature.  Shape (n_test, n_source, 9, 9).

    Args:
        offset: Integer offset between the cells (field minus source), in cell widths.
        h: Cell half-width (m).
        omega: Angular frequency (rad/s).
        ref: Background medium.
        n_source: Source monomials (10, 20 or 35).
        n_test: Field functions (4 or 10).
        tol: Relative tolerance that chooses the bisection level and the order.
        splits: Bisections of each of the eight boxes (None: chosen for least work).
        order: Order of the series on each box (None: from ``tol``).

    Raises:
        ValueError: when the cells touch (use ``blocks.near_block_series``).
    """
    if max(abs(o) for o in offset) <= 1:
        raise ValueError(
            f"piecewise_multipole_block: the cells at offset {offset} touch, so the singular point lies on "
            "a piece of W. Fix: use blocks.near_block_series for the self cell and its 26 neighbours."
        )
    ka, kb = omega / ref.alpha, omega / ref.beta
    auto_splits, auto_order = _piece_plan(offset, 2.0 * math.sqrt(3.0) * kb * h, tol)
    splits = auto_splits if splits is None else splits
    n = auto_order if order is None else order
    tst, src = SOURCE_EXPONENTS[:n_test], source_exponents(n_source)
    # per axis: moments M_i[a, c, sub, g] of the test-source pair's cross-correlation
    mom = [
        np.array([[piece_moments_1d(ta[i], sc[i], splits, n) for sc in src] for ta in tst])
        for i in range(3)
    ]
    n_sub = 2 ** (splits + 1)
    half = 1.0 / 2**splits
    centres = -2.0 + half * (2 * np.arange(n_sub) + 1)
    big_r = 2.0 * h * np.asarray(offset, dtype=float)
    g = np.arange(n + 1)
    fact = np.array([math.factorial(int(v)) for v in g], dtype=float)
    scale = h**g / fact  # h^g / g! per axis
    total = np.add.outer(np.add.outer(g, g), g)
    keep = total <= n
    ta_tab, tb_tab = family_tables()
    out = np.zeros((n_test, n_source, 9, 9), dtype=complex)
    for i0 in range(n_sub):
        for i1 in range(n_sub):
            for i2 in range(n_sub):
                x = big_r + h * np.array([centres[i0], centres[i1], centres[i2]])
                ds = radial_derivative_array(kb, x, n + 4)
                db = radial_derivative_array(
                    kb, x, n + 4, f=b_scalar_F(kb, ka, float(np.linalg.norm(x)), n + 4)
                )
                m0 = mom[0][:, :, i0, :] * scale
                m1 = mom[1][:, :, i1, :] * scale
                m2 = mom[2][:, :, i2, :] * scale
                for table, d in ((ta_tab, ds), (tb_tab, db)):
                    for idx, coef in table.items():
                        s0, s1, s2 = idx.count(0), idx.count(1), idx.count(2)
                        dg = np.where(keep, d[s0 : s0 + n + 1, s1 : s1 + n + 1, s2 : s2 + n + 1], 0.0)
                        acc = np.einsum("aci,acj,ack,ijk->ac", m0, m1, m2, dg, optimize=True)
                        out += acc[:, :, None, None] * coef[None, None]
    return _to_field_rows(out) * h**6 / (4.0 * np.pi * ref.mu)
