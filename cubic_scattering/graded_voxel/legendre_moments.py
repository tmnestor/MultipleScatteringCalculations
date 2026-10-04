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
from functools import cache

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


@cache
def box_moments(lows: tuple[float, ...], c2: float, m: int, n: int) -> NDArray:
    """Lam[k] for k in [0, n)^d (d = len(lows)), as an array of shape (n,) * d; a scalar for d = 0.

    Raises:
        ValueError: when the closure of the box holds the singular point (c2 = 0 and every l_j = 0), where
            the moments of the faces through it diverge for m <= -3.
    """
    d = len(lows)
    if d == 0:
        return np.array(c2 ** (m / 2.0))
    if c2 == 0.0 and not any(lows):
        raise ValueError(
            "box_moments: the box has a corner at the singular point (c2 = 0, lows = 0); its moments are "
            "taken by Duffy pyramids about that corner (blocks._vertex_piece), not by this relation."
        )
    n_rel = m + 2
    shape = (n,) * d
    index = np.arange(n**d).reshape(shape)
    t_mat = [legendre_product_matrix(n, (lj + 0.5, 0.5)) for lj in lows]
    tt_mat = [legendre_product_matrix(n, ((lj + 0.5) ** 2, lj + 0.5, 0.25)) for lj in lows]
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    rhs = np.zeros(n**d)

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
            other = tuple(lows[j] for j in range(d) if j != i)
            k_hat = tuple(kvec[j] for j in range(d) if j != i)
            hi = box_moments(other, c2 + (lows[i] + 1.0) ** 2, n_rel, n)
            lo = box_moments(other, c2 + lows[i] ** 2, n_rel, n)
            rhs[row] = 0.5 * (float(hi[k_hat]) + float(lo[k_hat]))
    a = sps.csc_matrix((vals, (rows, cols)), shape=(n**d, n**d))
    return spla.spsolve(a, rhs).reshape(shape)
