"""Local polynomial bases of the graded voxel.

A cell with centre c and half-width h has local coordinates xi = (x - c) / h in [-1, 1]^3, in the
package's axis order (z = 0, x = 1, y = 2).  The field is expanded in the TEST basis {1, xi_0, xi_1,
xi_2} (the Legendre polynomials of degree <= 1).  A contrast linear in the cell times a test function is
at most quadratic, so sources live in the SOURCE basis of the ten monomials of degree <= 2.
"""

import numpy as np
from numpy.typing import NDArray

Exponent = tuple[int, int, int]

TEST_EXPONENTS: tuple[Exponent, ...] = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
SOURCE_EXPONENTS: tuple[Exponent, ...] = (
    (0, 0, 0),
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (2, 0, 0),
    (0, 2, 0),
    (0, 0, 2),
    (1, 1, 0),
    (1, 0, 1),
    (0, 1, 1),
)
N_TEST = len(TEST_EXPONENTS)
N_SOURCE = len(SOURCE_EXPONENTS)


def _add(a: Exponent, b: Exponent) -> Exponent:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def moment_1d(e: int) -> float:
    """Integral of xi^e over [-1, 1]."""
    return 0.0 if e % 2 else 2.0 / (e + 1)


def _cell_moment(exps: Exponent, h: float) -> float:
    return h**3 * moment_1d(exps[0]) * moment_1d(exps[1]) * moment_1d(exps[2])


def monomials(exps: tuple[Exponent, ...], xi: NDArray) -> NDArray:
    """Values of the monomials xi^e at points xi, shape (len(exps), N)."""
    xi = np.asarray(xi, dtype=float)
    return np.array([xi[:, 0] ** e[0] * xi[:, 1] ** e[1] * xi[:, 2] ** e[2] for e in exps])


def gram_test(h: float) -> NDArray:
    """<L_a, L_b> over a cell of half-width h, shape (4, 4)."""
    return np.array([[_cell_moment(_add(a, b), h) for b in TEST_EXPONENTS] for a in TEST_EXPONENTS])


def gram_test_source(h: float) -> NDArray:
    """<L_a, m_c> over a cell of half-width h, shape (4, 10)."""
    return np.array([[_cell_moment(_add(a, c), h) for c in SOURCE_EXPONENTS] for a in TEST_EXPONENTS])


def product_table() -> NDArray:
    """P[c, a, b] with L_a L_b = sum_c P[c, a, b] m_c, shape (10, 4, 4)."""
    table = np.zeros((N_SOURCE, N_TEST, N_TEST))
    for a, ea in enumerate(TEST_EXPONENTS):
        for b, eb in enumerate(TEST_EXPONENTS):
            table[SOURCE_EXPONENTS.index(_add(ea, eb)), a, b] = 1.0
    return table


def source_expansion(delta: NDArray) -> NDArray:
    """E[c, b]: the 9 x 9 coefficient of m_c in Delta(xi) L_b(xi).

    Args:
        delta: Shape (4, 9, 9): Delta(xi) = sum_a delta[a] L_a(xi), i.e. the cell's mean contrast
            operator and its three gradient coefficients per unit xi (h times the physical gradient).

    Returns:
        Shape (10, 4, 9, 9).
    """
    return np.einsum("cab,aij->cbij", product_table(), np.asarray(delta, dtype=complex))
