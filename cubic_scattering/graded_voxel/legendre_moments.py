"""Legendre (modified) moments of a power of the distance over a box away from the singular point.

    Lam[k] = int_box prod_j P_(k_j)(x_j) S^m dt,   x_j = 2 (t_j - l_j) - 1,   S^2 = c2 + sum_j t_j^2,

over the unit box prod_j [l_j, l_j + 1] in d = 1, 2 or 3 dimensions (l_j >= 0), m odd, the closure of the
box (lifted by c2 >= 0) not holding the singular point S = 0. These are the stable building blocks of the
closed forms of the touching coupling blocks (``blocks.static_term_integral_closed``): power moments of S^m
about the singular point are ill-conditioned in double precision, orthogonal-polynomial moments on the box
are not.

THE RELATION. Along an anchor axis i, the identity (2k+1) P_k = P_(k+1)' - P_(k-1)' and an integration by
parts, applied to g = S^n = (c2 + sum_j t_j^2) S^m with n = m + 2, whose x_i-derivative is (n/2) t_i S^m
(dt_i/dx_i = 1/2 on a unit box), give for every multi-index k (k^ the indices of the other axes)

    k_i >= 1:  (2k_i+1) int P_k (c2 + sum_j t_j^2) S^m
                   + (n/2) int (P_(k_i+1) - P_(k_i-1))(x_i) t_i P_(k^) S^m = 0,
    k_i =  0:  int P_k (c2 + sum_j t_j^2) S^m + (n/2) int P_1(x_i) t_i P_(k^) S^m
                   = (1/2) [ Lam_(d-1)[k^; c2 + (l_i+1)^2, n] + Lam_(d-1)[k^; c2 + l_i^2, n] ].

The boundary terms vanish for k_i >= 1 because P_(k_i+1) - P_(k_i-1) is zero at x_i = +-1; for k_i = 0 they
are the moments of S^n on the two faces normal to axis i, one dimension down, computed by the same
relation. In d = 0 the 'moment' is the corner value c2^(m/2). Multiplication by t_j and t_j^2 is a banded
operator in the Legendre index (t_j = l_j + 1/2 + x_j/2), so each relation couples the indices
k_j - 2 .. k_j + 2 in every axis.

THE SOLUTION. The anchor is the axis of the largest k_i. The relations for k in [0, N)^d, with Lam = 0
beyond, form one sparse linear system: the boundary-value formulation of a recurrence whose wanted solution
is the minimal one (Olver 1967; Gautschi 1967). Marching it in either direction is unstable near the
singular point; the truncated system is not, and the moments come out to full RELATIVE accuracy, the
high-order ones (which decay geometrically for a function analytic on the box) included. No logarithm or
arctangent enters: the only data are the corner values of S^n, the transcendental functions being carried
by the system itself.

Checked against 40-digit quadrature in 1-D and 2-D (tests/test_legendre_moments.py), and through the
coupling blocks against the 40-digit corner and edge reference of
Mathematica/GradedVoxel_CornerReference.wl.
"""

import itertools
import math
from functools import cache, lru_cache

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
from numpy.polynomial import legendre as npleg
from numpy.typing import NDArray


@cache
def legendre_product_matrix(n: int, coeffs: tuple[float, ...]) -> NDArray:
    """T[m, k] with q(x) P_k(x) = sum_m T[m, k] P_m(x), q the polynomial of power coefficients `coeffs`."""
    q_leg = npleg.poly2leg(list(coeffs))
    out = np.zeros((n + len(coeffs), n))
    for k in range(n):
        e = np.zeros(k + 1)
        e[k] = 1.0
        prod = npleg.legmul(q_leg, e)
        out[: len(prod), k] = prod
    return out


#: factorised systems kept at once: box_moment_derivatives reuses one for every derivative order of the same
#: (box, c2, m), while a three-dimensional factorisation fills in to hundreds of megabytes, so the cache is
#: bounded rather than unlimited
RELATION_CACHE = 8


@lru_cache(maxsize=RELATION_CACHE)
def _relation_system(
    lows: tuple[float, ...], c2: float, m: int, n: int
) -> tuple[spla.SuperLU, NDArray, tuple[tuple[int, int, tuple[int, ...]], ...]]:
    """The sparse system of the relation for k in [0, n)^d, factorised once.

    Returns the LU factors, the weight w of each row (2 k_i + 1, or 1 for k_i = 0: the factor that
    multiplies c2 on the diagonal), and for each row with k_i = 0 the triple (row, anchor axis i, k^) whose
    right-hand side is the mean of the moments of S^(m + 2) on the two faces normal to axis i.
    """
    d = len(lows)
    n_rel = m + 2
    shape = (n,) * d
    index = np.arange(n**d).reshape(shape)
    t_mat = [legendre_product_matrix(n, (lj + 0.5, 0.5)) for lj in lows]
    tt_mat = [legendre_product_matrix(n, ((lj + 0.5) ** 2, lj + 0.5, 0.25)) for lj in lows]
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    weights = np.zeros(n**d)
    boundary: list[tuple[int, int, tuple[int, ...]]] = []

    def add(row: int, kvec: tuple[int, ...], axis: int, column: NDArray, coef: float) -> None:
        for mm, v in enumerate(column[:n]):
            if v != 0.0:
                k2 = list(kvec)
                k2[axis] = mm
                rows.append(row)
                cols.append(int(index[tuple(k2)]))
                vals.append(coef * v)

    for kvec in itertools.product(range(n), repeat=d):
        row = int(index[kvec])
        i = int(np.argmax(kvec))
        ki = kvec[i]
        w = 2 * ki + 1 if ki >= 1 else 1
        weights[row] = w
        rows.append(row)
        cols.append(row)
        vals.append(w * c2)
        for j in range(d):
            add(row, kvec, j, tt_mat[j][:, kvec[j]], w)
        if ki >= 1:
            if ki + 1 < n:
                add(row, kvec, i, t_mat[i][:, ki + 1], n_rel / 2)
            add(row, kvec, i, t_mat[i][:, ki - 1], -n_rel / 2)
        else:
            add(row, kvec, i, t_mat[i][:, 1], n_rel / 2)
            boundary.append((row, i, tuple(kvec[j] for j in range(d) if j != i)))
    a = sps.csc_matrix((vals, (rows, cols)), shape=(n**d, n**d))
    return spla.splu(a), weights, tuple(boundary)


@cache
def box_moments(lows: tuple[float, ...], c2: float, m: int, n: int) -> NDArray:
    """Lam[k] for k in [0, n)^d (d = len(lows)), as an array of shape (n,) * d; a scalar for d = 0.

    Raises:
        ValueError: when the closure of the box holds the singular point (c2 = 0 and every l_j = 0): the
            moments of S^m alone diverge there for m <= -d (logarithmically at m = -d), although the
            kernel's integrand t^alpha |t|^m may not.
    """
    d = len(lows)
    if d == 0:
        return np.array(c2 ** (m / 2.0))
    if c2 == 0.0 and not any(lows):
        raise ValueError(
            "box_moments: the box has a corner at the singular point (c2 = 0, lows = 0); its moments are "
            "taken by Duffy pyramids about that corner (blocks._vertex_piece), not by this relation."
        )
    lu, _, boundary = _relation_system(lows, c2, m, n)
    rhs = np.zeros(n**d)
    for row, i, k_hat in boundary:
        other = tuple(lows[j] for j in range(d) if j != i)
        hi = box_moments(other, c2 + (lows[i] + 1.0) ** 2, m + 2, n)
        lo = box_moments(other, c2 + lows[i] ** 2, m + 2, n)
        rhs[row] = 0.5 * (float(hi[k_hat]) + float(lo[k_hat]))
    return lu.solve(rhs).reshape((n,) * d)


def point_derivative(z: tuple[float, ...], b: tuple[int, ...], m: int) -> float:
    """d^b (sum_f z_f^2)^(m/2) at the point z, by the Hermite-type expansion in the radial ladder."""
    s2 = sum(v * v for v in z)
    total = 0.0
    for k in itertools.product(*(range(bf // 2 + 1) for bf in b)):
        coef = 1.0
        for bf, kf in zip(b, k, strict=True):
            coef *= math.factorial(bf) / (math.factorial(kf) * math.factorial(bf - 2 * kf) * 2**kf)
        q = sum(b) - sum(k)
        ladder = 1.0
        for j in range(q):  # F_q of S^m: m (m - 2) ... (m - 2q + 2) S^(m - 2q)
            ladder *= m - 2 * j
        if ladder == 0.0:
            continue
        mono = math.prod(zf ** (bf - 2 * kf) for zf, bf, kf in zip(z, b, k, strict=True))
        total += coef * mono * ladder * s2 ** ((m - 2 * q) / 2.0)
    return total


@cache
def box_moment_derivatives(
    lows: tuple[float, ...], z: tuple[float, ...], b: tuple[int, ...], m: int, n: int
) -> NDArray:
    """Lam^(b)[k] = int_box prod_j P_(k_j)(x_j) d_z^b S^m dt, S^2 = sum_f z_f^2 + sum_j t_j^2.

    The moments over the unit box prod_j [l_j, l_j + 1] of a derivative of S^m with respect to the fixed
    coordinates z (the coordinates normal to a face, an edge or a corner, which the box does not vary),
    without expanding the derivative into its terms. The fixed coordinates enter the relation of
    ``box_moments`` only through c2 = sum_f z_f^2, which multiplies the diagonal; differentiating the
    relation b times by Leibniz's rule leaves the same matrix and moves the derivatives of c2 S^m,

        d^b (c2 S^m) = c2 d^b S^m + sum_f [ 2 b_f z_f d^(b - e_f) S^m + b_f (b_f - 1) d^(b - 2 e_f) S^m ],

    to the right-hand side as moments of lower derivatives, computed first. The boundary data are the same
    derivatives of S^(m + 2) on the faces, one dimension down, and in d = 0 the derivative of S^m at a
    point.
    For b = 0 this is ``box_moments(lows, sum z^2, m, n)``.

    Raises:
        ValueError: for z = 0 (no fixed coordinate away from the singular point) with every l_j = 0.
    """
    c2 = float(sum(v * v for v in z))
    d = len(lows)
    if not any(b):
        return box_moments(lows, c2, m, n)
    if d == 0:
        return np.array(point_derivative(z, b, m))
    if c2 == 0.0 and not any(lows):
        raise ValueError(
            "box_moment_derivatives: the box has a corner at the singular point (z = 0, lows = 0); the "
            "relation needs the singular point outside the closure of the box."
        )
    lu, weights, boundary = _relation_system(lows, c2, m, n)
    rhs = np.zeros(n**d)
    for row, i, k_hat in boundary:
        other = tuple(lows[j] for j in range(d) if j != i)
        hi = box_moment_derivatives(other, (*z, lows[i] + 1.0), (*b, 0), m + 2, n)
        lo = box_moment_derivatives(other, (*z, lows[i]), (*b, 0), m + 2, n)
        rhs[row] = 0.5 * (float(hi[k_hat]) + float(lo[k_hat]))
    for f, bf in enumerate(b):
        if bf >= 1:
            lower = list(b)
            lower[f] -= 1
            rhs -= weights * 2.0 * bf * z[f] * box_moment_derivatives(lows, z, tuple(lower), m, n).ravel()
        if bf >= 2:
            lower = list(b)
            lower[f] -= 2
            rhs -= weights * bf * (bf - 1) * box_moment_derivatives(lows, z, tuple(lower), m, n).ravel()
    return lu.solve(rhs).reshape((n,) * d)


def point_derivative_magnitude(z: tuple[float, ...], b: tuple[int, ...], m: int) -> float:
    """The sum of |terms| of ``point_derivative``: the magnitude its rounding acts on."""
    s2 = sum(v * v for v in z)
    total = 0.0
    for k in itertools.product(*(range(bf // 2 + 1) for bf in b)):
        coef = 1.0
        for bf, kf in zip(b, k, strict=True):
            coef *= math.factorial(bf) / (math.factorial(kf) * math.factorial(bf - 2 * kf) * 2**kf)
        q = sum(b) - sum(k)
        ladder = 1.0
        for j in range(q):
            ladder *= m - 2 * j
        mono = math.prod(zf ** (bf - 2 * kf) for zf, bf, kf in zip(z, b, k, strict=True))
        total += abs(coef * mono * ladder) * s2 ** ((m - 2 * q) / 2.0)
    return total


@cache
def box_moment_derivative_magnitudes(
    lows: tuple[float, ...], z: tuple[float, ...], b: tuple[int, ...], m: int, n: int
) -> NDArray:
    """A running error magnitude for ``box_moment_derivatives``: the size of what its rounding acts on.

    The relation is solved for the moments; what can cancel is its right-hand side, where the boundary data
    and the Leibniz terms of the lower derivatives are combined. This propagates the sum of their absolute
    values through the same factorised system, |A^-1 (sum |rhs terms|)|, starting for b = 0 from the largest
    |Lam| at every index (the undifferentiated relation gives each moment to round-off of the largest one,
    not of itself) and from the sum of |terms| of the Hermite expansion at a point. Its ratio to |Lam^(b)|
    is the condition number of the evaluation.
    """
    c2 = float(sum(v * v for v in z))
    d = len(lows)
    if not any(b):
        lam = box_moments(lows, c2, m, n)
        return np.full_like(lam, np.abs(lam).max())  # round-off of the largest moment, for every index
    if d == 0:
        return np.array(point_derivative_magnitude(z, b, m))
    lu, weights, boundary = _relation_system(lows, c2, m, n)
    rhs = np.zeros(n**d)
    for row, i, k_hat in boundary:
        other = tuple(lows[j] for j in range(d) if j != i)
        hi = box_moment_derivative_magnitudes(other, (*z, lows[i] + 1.0), (*b, 0), m + 2, n)
        lo = box_moment_derivative_magnitudes(other, (*z, lows[i]), (*b, 0), m + 2, n)
        rhs[row] = 0.5 * (float(hi[k_hat]) + float(lo[k_hat]))
    for f, bf in enumerate(b):
        if bf >= 1:
            lower = list(b)
            lower[f] -= 1
            rhs += (
                np.abs(weights * 2.0 * bf * z[f])
                * box_moment_derivative_magnitudes(lows, z, tuple(lower), m, n).ravel()
            )
        if bf >= 2:
            lower = list(b)
            lower[f] -= 2
            rhs += (
                np.abs(weights * bf * (bf - 1))
                * box_moment_derivative_magnitudes(lows, z, tuple(lower), m, n).ravel()
            )
    return np.abs(lu.solve(rhs)).reshape((n,) * d)
