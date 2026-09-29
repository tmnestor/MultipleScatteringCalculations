"""The subdivision identity for every (a, c): a block between cells of half-width h equals the sum of the
blocks between their 2^3 sub-cells (half-width h/2), with the polynomials re-expressed on the sub-cells.

Big-cell local xi = (xi_s + sigma) / 2 on the sub-cell with centre offset sigma in {-1, 1}^3 (units of h/2).
L_a(xi) = sum_b C[a, b](sigma) L_b(xi_s); m_c(xi) = sum_d D[c, d](sigma) m_d(xi_s).
K_big[a, c](R) = sum_{sigma, sigma'} sum_{b, d} C[a, b](sigma) K_sub[b, d](R_sub) D[c, d](sigma'),
R_sub = 2 * offset + (sigma - sigma') / 2 in sub-cell grid units.

Run:  conda run -n seismic python -u scripts/check_graded_voxel_subdivision.py [omega] [h]  (300, 2.5)
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cubic_scattering import ReferenceMedium  # noqa: E402
from cubic_scattering.graded_voxel.basis import SOURCE_EXPONENTS, TEST_EXPONENTS  # noqa: E402
from cubic_scattering.graded_voxel.blocks import coupling_block  # noqa: E402

REF = ReferenceMedium(5000.0, 3000.0, 2500.0)
omega = float(sys.argv[1]) if len(sys.argv) > 1 else 300.0
h = float(sys.argv[2]) if len(sys.argv) > 2 else 2.5
xs = sp.symbols("s0 s1 s2")


def expand_on_sub(exps_big, exps_sub, sigma):
    """Coefficients of big-cell monomials in sub-cell monomials: xi = (xi_s + sigma)/2."""
    out = np.zeros((len(exps_big), len(exps_sub)))
    for i, e in enumerate(exps_big):
        expr = sp.expand(sp.Mul(*[((xs[k] + sigma[k]) / 2) ** e[k] for k in range(3)]))
        poly = sp.Poly(expr, *xs)
        for mono, coef in zip(poly.monoms(), poly.coeffs(), strict=True):
            out[i, exps_sub.index(tuple(mono))] = float(coef)
    return out


sigmas = list(itertools.product((-1, 1), repeat=3))
C = {s: expand_on_sub(TEST_EXPONENTS, TEST_EXPONENTS, s) for s in sigmas}
D = {s: expand_on_sub(SOURCE_EXPONENTS, SOURCE_EXPONENTS, s) for s in sigmas}
cache: dict = {}
for off in ((0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 0, 0)):
    big = coupling_block(off, h, omega, REF)
    acc = np.zeros_like(big)
    for s, sp_ in itertools.product(sigmas, sigmas):
        rsub = tuple(int(2 * o + (a - b) // 2) for o, a, b in zip(off, s, sp_, strict=True))
        if rsub not in cache:
            cache[rsub] = coupling_block(rsub, h / 2, omega, REF)
        acc += np.einsum("ab,bdij,cd->acij", C[s], cache[rsub], D[sp_])
    worst = max(
        np.linalg.norm(acc[a, c] - big[a, c]) / np.linalg.norm(big[0, 0])
        for a in range(4)
        for c in range(10)
    )
    ac = max(((a, c) for a in range(4) for c in range(10)), key=lambda p: np.linalg.norm(acc[p] - big[p]))
    print(
        f"offset {off}: worst |sum of sub-blocks - block| / |K00| = {worst:.2e} at (a, c) = {ac}",
        flush=True,
    )
