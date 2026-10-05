#!/usr/bin/env python3
"""Does the two-centre multipole series converge for cells that touch? The gap condition, measured.

Superposition T-matrix methods couple scatterers through the translation addition theorem, which converges
when the circumscribing spheres of the scatterers do not overlap; a space-filling tessellation violates that
condition for every pair of neighbours. The Cartesian counterpart here is the two-centre multipole series of
the coupling moment of two cells of half-width h = 1 whose centres are R = 2 o apart,

    U[a, c](o) = int int xi^e_a (1 / |x - x'|) xi'^e_c
               = sum_gamma mu^gamma[a, c] / gamma!  d^gamma (1/r) (R),

with the cells' exact pair moments mu^gamma = prod_i int int u^(e_a,i) (u - v)^gamma_i v^(e_c,i) du dv
(rational numbers) and the derivatives of 1/r at R. The kernel 1/r is the static singular part that every
family of the elastodynamic propagator shares. The circumscribing spheres have radius sqrt 3 h each, so the
condition 2 sqrt 3 h < |R| fails at face, edge and corner contact (ratio sqrt 3, 1.22 and 1.0).

For each offset, the partial sums S_N (total order |gamma| <= N) are computed in 60-digit arithmetic, so
that divergence is not mistaken for rounding, and compared with the exact value from the stable closed form
(``blocks.static_term_integral_closed``, validated against the 40-digit reference). Reported per order:
the error of S_N (relative to the largest entry), the size of the terms of order N, and the condition
number kappa_N = sum |terms| / |S_N| of the series in double precision.

Run:  python -u scripts/measure_touching_multipole_series.py [N_max]
"""

import itertools
import math
import sys
from fractions import Fraction
from pathlib import Path

import mpmath as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cubic_scattering.graded_voxel.basis import SOURCE_EXPONENTS, source_exponents  # noqa: E402
from cubic_scattering.graded_voxel.blocks import _sform, static_term_integral_closed  # noqa: E402

mp.mp.dps = 60
OFFSETS = [(1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 0, 0), (2, 1, 1), (3, 0, 0)]
N_TEST, N_SOURCE = 4, 10  # the linear field and its source monomials


def moment_1d(n: int) -> Fraction:
    return Fraction(2, n + 1) if n % 2 == 0 else Fraction(0)


def pair_moment_1d(e_t: int, e_s: int, g: int) -> Fraction:
    """int int u^e_t (u - v)^g v^e_s du dv over [-1, 1]^2, exactly."""
    return sum(
        (
            math.comb(g, j) * (-1) ** (g - j) * moment_1d(e_t + j) * moment_1d(e_s + g - j)
            for j in range(g + 1)
        ),
        Fraction(0),
    )


def inverse_r_derivative(gamma: tuple[int, int, int], x: list) -> mp.mpf:
    """d^gamma (1 / r) at x in 60 digits: the Hermite-type sum with integer coefficients."""
    r2 = x[0] ** 2 + x[1] ** 2 + x[2] ** 2
    total = mp.mpf(0)
    for k in itertools.product(*(range(g // 2 + 1) for g in gamma)):
        coef = 1
        for g, kk in zip(gamma, k, strict=True):
            coef *= math.factorial(g) // (math.factorial(kk) * math.factorial(g - 2 * kk) * 2**kk)
        q = sum(gamma) - sum(k)
        ladder = 1
        for j in range(q):  # F_q of r^-1: (-1)(-3)...(-(2q - 1)) r^(-1 - 2q)
            ladder *= -(2 * j + 1)
        mono = mp.mpf(1)
        for xi, g, kk in zip(x, gamma, k, strict=True):
            mono *= xi ** (g - 2 * kk)
        total += coef * ladder * mono * r2 ** (mp.mpf(-1 - 2 * q) / 2)
    return total


def main(n_max: int) -> int:
    tst, src = SOURCE_EXPONENTS[:N_TEST], source_exponents(N_SOURCE)
    for off in OFFSETS:
        if max(abs(o) for o in off) <= 1:  # touching: the stable closed form
            exact = static_term_integral_closed(-1, (), off, 1.0, N_SOURCE, N_TEST)
        else:  # apart: the integrand is smooth, and Gauss rules of 32 points on each piece reach round-off
            inv_r = lambda x: (1.0 / np.linalg.norm(x, axis=1))[:, None]  # noqa: E731
            exact = _sform(off, 1.0, (0, 0, 0), inv_r, 32, N_SOURCE, N_TEST)[:, :, 0]
        size = float(np.abs(exact).max())
        big_r = [mp.mpf(2 * o) for o in off]
        ratio = 2 * math.sqrt(3) / (2 * math.sqrt(sum(o * o for o in off)))
        s_mp = [[mp.mpf(0)] * N_SOURCE for _ in range(N_TEST)]
        s_dp = np.zeros((N_TEST, N_SOURCE))
        mag_dp = np.zeros((N_TEST, N_SOURCE))
        print(f"offset {off}: (circumscribing-sphere ratio 2 sqrt3 h / R = {ratio:.3f})", flush=True)
        for n in range(n_max + 1):
            order_terms = np.zeros((N_TEST, N_SOURCE))
            for g0 in range(n + 1):
                for g1 in range(n + 1 - g0):
                    gam = (g0, g1, n - g0 - g1)
                    d = inverse_r_derivative(gam, big_r)
                    fact = math.prod(math.factorial(g) for g in gam)
                    for ia, ta in enumerate(tst):
                        for ic, sc in enumerate(src):
                            mu = math.prod(pair_moment_1d(ta[i], sc[i], gam[i]) for i in range(3))
                            if mu == 0:
                                continue
                            term = mp.mpf(mu.numerator) / mu.denominator / fact * d
                            s_mp[ia][ic] += term
                            t_dp = float(term)
                            s_dp[ia, ic] += t_dp
                            mag_dp[ia, ic] += abs(t_dp)
                            order_terms[ia, ic] += abs(t_dp)
            if n % 4 == 0 or n == n_max:
                s = np.array([[float(v) for v in row] for row in s_mp])
                err_mp = np.abs(s - exact).max() / size
                err_dp = np.abs(s_dp - exact).max() / size
                big = np.abs(exact) > 1e-6 * size  # kappa of entries that are not negligible
                kappa = (mag_dp[big] / np.abs(exact[big])).max()
                print(
                    f"   N = {n:3d}: error of S_N (60 digits) {err_mp:.1e}, in double {err_dp:.1e}; "
                    f"terms of order N {order_terms.max() / size:.1e}; kappa_N {kappa:.1e}",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 48))
