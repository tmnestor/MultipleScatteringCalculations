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
"""

import itertools
import math
from functools import cache

import numpy as np
from numpy.typing import NDArray
from scipy.special import spherical_jn, spherical_yn

from ..effective_contrasts import ReferenceMedium
from .basis import SOURCE_EXPONENTS, moment_1d, source_exponents
from .blocks import _to_field_rows, family_tables

Beta = tuple[int, int, int]

#: The series is refused when (2 sqrt 3 h) / R exceeds this: it would need hundreds of orders.
MAX_RATIO = 0.6


def helmholtz_F(k: float, r: float, q_max: int) -> NDArray:
    """F_q = (r^-1 d/dr)^q [exp(i k r) / r], q = 0..q_max, as i (-1)^q k^(q+1) h_q(k r) / r^q."""
    q = np.arange(q_max + 1)
    h = spherical_jn(q, k * r) + 1j * spherical_yn(q, k * r)
    return 1j * (-1.0) ** q * k ** (q + 1.0) * h / r**q


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


def radial_derivatives(k: float, x: NDArray, order: int) -> dict[Beta, complex]:
    """d^beta [exp(i k r) / r] at the point x, for every |beta| <= order."""
    x = np.asarray(x, dtype=float)
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
    dp = radial_derivatives(ka, big_r, n + 4)
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
                d = ds[beta] if family == "a" else (ds[beta] - dp[beta]) / kb**2
                acc += w * d
            out += acc[:, :, None, None] * coef[None, None]
    return _to_field_rows(out) / (4.0 * np.pi * ref.mu)
