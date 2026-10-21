"""Distant receiver cells: the cell average by the expansion about the cell centre, all offsets at once.

WHY THIS EXISTS. The wavenumber series (``cell_average_kseries``) is exact and amortised across frequency,
but it cannot reach distant cells at high frequency: once |k_S| r_max passes about 9, 48 powers no longer
reach the tolerance. Those offsets went to the quadrature, which at k_S R = 5 (n_sub = 25) meant 70% of
them and 0.85 s per frequency. For a DISTANT cell the natural expansion is about the cell's centre instead:

    < d^a f >(s) = sum_{beta even} w_beta  d^(a + beta) f(s),    w_beta = prod_i h^(beta_i) / (beta_i + 1)!,

the cube's moments (h the half-width; frequency-independent) times the derivatives of g_S and
B = (g_S - g_P) / k_S^2 at the centre, which ``graded_voxel.derivatives.scalar_derivative_fields`` gives to
any order (compiled, ``fortran/green_derivatives.f90``). It converges as (h / r)^2 and (k_S h)^2, for every
frequency the order cap allows. Item 3's d^2 tail is its first correction.

This is ``graded_voxel.derivatives.moment_table_series`` (weight W = 0, one sub-cube) vectorised over the
offsets: one call of the derivative ladder for every offset, then one weighted sum. Equal to that function,
offset by offset, to round-off (``tests/test_cell_average_far.py``). Per frequency it is analytic but not
amortised: the derivatives depend on the frequency, the weights do not.

WHERE IT APPLIES. One expansion about the centre needs the cell small seen from the source:
rho = sqrt(3) h / r <= ``RHO_MAX`` (0.3, as ``moment_table_series``), so r >= about 2.9 cells. The
truncation order is the least N whose bound sum_j rho^(N + 1 - j) kappa^j / j!, kappa = sqrt(3) |k_S| h,
is below the tolerance (``moment_table_series``' rule); an offset whose N would pass ``far_max_order`` is
left to the quadrature, as are separations off the lattice. Offsets are evaluated in groups of equal N,
one ladder call per group, so a close offset needing a high order does not impose it on distant ones.
"""

import math

import numpy as np
from numpy.typing import NDArray

from .cell_average_fortran import cube_image_blocks, cube_maps
from .effective_contrasts import ReferenceMedium
from .graded_voxel import derivatives as gd
from .graded_voxel.kernel import _assemble as assemble_from_greens

__all__ = ["RHO_MAX", "averaged_pair_blocks_9x9_far", "canonical_greens_far"]

#: the largest sqrt(3) h / r for one expansion about the cell centre (moment_table_series' value)
RHO_MAX = gd.RHO_MAX
_D_LIST = gd.multi_indices(2)


def _orders(
    canonical: NDArray, omega: complex, ref: ReferenceMedium, d: float, tol: float, cap: int
) -> NDArray:
    """Each offset's truncation order by ``moment_table_series``' rule, or -1 where it does not apply."""
    h = 0.5 * d
    r = np.linalg.norm(canonical.astype(float), axis=1) * d
    rho = np.sqrt(3.0) * h / r
    kappa = math.sqrt(3.0) * abs(omega / ref.beta) * h
    out = np.full(len(canonical), -1)
    for j, rj in enumerate(rho):
        if rj > RHO_MAX:
            continue
        order = 2
        while order <= cap:
            bound = sum(rj ** (order + 1 - i) * kappa**i / math.factorial(i) for i in range(order + 2))
            if bound < tol:
                out[j] = order
                break
            order += 1
    return out


def _weights(targets: tuple[gd.Index, ...], order: int, max_order: int) -> NDArray:
    """W[a, k] such that sum_k W[a, k] d^(index k) f = sum_beta w_beta d^(a + beta) f, |beta| <= order."""
    pos = {a: k for k, a in enumerate(gd.multi_indices(max_order))}
    betas = [b for b in gd.multi_indices(order) if all(x % 2 == 0 for x in b)]
    w = np.zeros((len(targets), len(pos)))
    for ia, a in enumerate(targets):
        for b in betas:
            weight = math.prod(0.5**x / math.factorial(x + 1) for x in b)  # h = 1/2 in units of the side
            w[ia, pos[(a[0] + b[0], a[1] + b[1], a[2] + b[2])]] += weight
    return w


def canonical_greens_far(
    canonical: NDArray, omega: complex, ref: ReferenceMedium, d: float, *, tol: float, max_order: int
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """The cell-averaged G, dG, ddG at canonical offsets by the expansion about the cell centre.

    Args:
        canonical: Integer offsets (K, 3).
        omega: Angular frequency.
        ref: Background medium.
        d: Cell side.
        tol: Truncation tolerance (``cell_average.far_tol``).
        max_order: The highest truncation order allowed (``cell_average.far_max_order``).

    Returns:
        (G (K, 3, 3), Gd (K, 3, 3, 3), Gdd (K, 3, 3, 3, 3), ok (K,)); ok is False where the expansion does
        not apply (too close, or an order beyond the cap), and the entries there are zero.
    """
    K = len(canonical)
    G = np.zeros((K, 3, 3), dtype=complex)
    Gd = np.zeros((K, 3, 3, 3), dtype=complex)
    Gdd = np.zeros((K, 3, 3, 3, 3), dtype=complex)
    orders = _orders(canonical, omega, ref, d, tol, max_order)
    ok = orders >= 0
    g_targets = gd.multi_indices(2)  # d^D g_S, |D| <= 2
    b_targets = gd.multi_indices(4)  # d^(D + e_i + e_n) B
    pos_g = {a: k for k, a in enumerate(g_targets)}
    pos_b = {a: k for k, a in enumerate(b_targets)}
    pref = 1.0 / (4.0 * np.pi * ref.mu)
    deg_g = np.array([float(sum(a)) for a in g_targets])
    deg_b = np.array([float(sum(a)) for a in b_targets])

    # one ladder call per distinct order: a close offset needing a high order must not impose it on the
    # distant ones, whose derivatives to that order would cost far more than they need
    for order in sorted(set(orders[ok].tolist())):
        rows = np.flatnonzero(orders == order)
        X = canonical[rows].astype(float) * d
        n_s, n_b = len(gd.multi_indices(2 + order)), len(gd.multi_indices(4 + order))
        f_s, f_b = gd.scalar_derivative_fields(X, omega, ref, n_s, n_b)  # (n_s, K'), (n_b, K')
        # derivatives in physical units, weights in units of the side: d^|beta| restores them
        scale_s = np.array([d ** sum(a) for a in gd.multi_indices(2 + order)])
        scale_b = np.array([d ** sum(a) for a in gd.multi_indices(4 + order)])
        s_g = (_weights(g_targets, order, 2 + order) * scale_s) @ f_s / (d**deg_g)[:, None]  # (10, K')
        s_b = (_weights(b_targets, order, 4 + order) * scale_b) @ f_b / (d**deg_b)[:, None]  # (35, K')

        def entry(i: int, n: int, *axes: int, s_g: NDArray = s_g, s_b: NDArray = s_b) -> NDArray:
            e = [0, 0, 0]
            for ax in axes:
                e[ax] += 1
            b = (e[0] + (i == 0) + (n == 0), e[1] + (i == 1) + (n == 1), e[2] + (i == 2) + (n == 2))
            val = s_b[pos_b[b]]
            if i == n:
                val = val + s_g[pos_g[(e[0], e[1], e[2])]]
            return pref * val

        for i in range(3):
            for n in range(3):
                G[rows, i, n] = entry(i, n)
                for k in range(3):
                    Gd[rows, i, n, k] = entry(i, n, k)
                    for m in range(3):
                        Gdd[rows, i, n, k, m] = entry(i, n, k, m)
    return G, Gd, Gdd, ok


def averaged_pair_blocks_9x9_far(
    S: NDArray, omega: complex, ref: ReferenceMedium, d: float, *, tol: float, max_order: int
) -> tuple[NDArray, NDArray]:
    """The averaged 9 x 9 blocks at separations S (N, 3) by the expansion about the cell centre.

    Returns:
        (blocks (N, 9, 9), covered (N,)): covered is False for a separation off the lattice, too close for
        one expansion, or needing an order beyond ``max_order``; its block is zero, for the caller to fill.
    """
    S = np.atleast_2d(np.asarray(S, dtype=float))
    offsets = np.rint(S / d).astype(int)
    on_lattice = (np.max(np.abs(S - d * offsets), axis=1) <= 1.0e-9 * d) & np.any(offsets != 0, axis=1)
    blocks = np.zeros((len(S), 9, 9), dtype=complex)
    covered = np.zeros(len(S), dtype=bool)
    rows = np.flatnonzero(on_lattice)
    if not len(rows):
        return blocks, covered
    canonical, inverse, pinv, sign = cube_maps(offsets[rows])
    G, Gd, Gdd, ok = canonical_greens_far(canonical, omega, ref, d, tol=tol, max_order=max_order)
    use = ok[inverse]
    if not use.any():
        return blocks, covered
    blocks_c = np.zeros((len(canonical), 9, 9), dtype=complex)
    blocks_c[ok] = assemble_from_greens(G[ok], Gd[ok], Gdd[ok])
    target = rows[use]
    blocks[target] = cube_image_blocks(blocks_c, inverse[use], pinv[use], sign[use])
    covered[target] = True
    return blocks, covered
