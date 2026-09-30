"""Local polynomial bases of the graded voxel.

A cell with centre c and half-width h has local coordinates xi = (x - c) / h in [-1, 1]^3, in the
package's axis order (z = 0, x = 1, y = 2).  The field is expanded in the TEST basis {1, xi_0, xi_1,
xi_2} (the Legendre polynomials of degree <= 1).  A contrast linear in the cell times a test function is
at most quadratic, so sources live in the SOURCE basis of the ten monomials of degree <= 2.

A contrast QUADRATIC in the cell is expanded in the ten orthogonal polynomials of CONTRAST_BASIS (the test
basis, then P_2(xi_i) and xi_i xi_j); its product with a test function is at most cubic, and the sources
then live in the twenty monomials of degree <= 3, SOURCE_EXPONENTS_CUBIC, whose first ten are
SOURCE_EXPONENTS.  Every function that depends on the source set takes its size (10 or 20).
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
SOURCE_EXPONENTS_CUBIC: tuple[Exponent, ...] = SOURCE_EXPONENTS + (
    (3, 0, 0),
    (0, 3, 0),
    (0, 0, 3),
    (2, 1, 0),
    (2, 0, 1),
    (1, 2, 0),
    (0, 2, 1),
    (1, 0, 2),
    (0, 1, 2),
    (1, 1, 1),
)
#: The contrast basis as {exponent: coefficient}: 1, xi_i, P_2(xi_i) = (3 xi_i^2 - 1)/2, xi_i xi_j.
CONTRAST_BASIS: tuple[dict[Exponent, float], ...] = (
    {(0, 0, 0): 1.0},
    {(1, 0, 0): 1.0},
    {(0, 1, 0): 1.0},
    {(0, 0, 1): 1.0},
    {(2, 0, 0): 1.5, (0, 0, 0): -0.5},
    {(0, 2, 0): 1.5, (0, 0, 0): -0.5},
    {(0, 0, 2): 1.5, (0, 0, 0): -0.5},
    {(1, 1, 0): 1.0},
    {(1, 0, 1): 1.0},
    {(0, 1, 1): 1.0},
)
#: int over [-1, 1]^3 of each contrast basis function squared.
CONTRAST_NORMS: tuple[float, ...] = (8.0, 8 / 3, 8 / 3, 8 / 3, 8 / 5, 8 / 5, 8 / 5, 8 / 9, 8 / 9, 8 / 9)
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


def source_exponents(n_source: int) -> tuple[Exponent, ...]:
    """The source monomials: 10 (degree <= 2) or 20 (degree <= 3)."""
    if n_source not in (10, 20):
        raise ValueError(f"source_exponents: n_source must be 10 or 20, got {n_source}")
    return SOURCE_EXPONENTS_CUBIC[:n_source]


def contrast_values(xi: NDArray) -> NDArray:
    """Values of the ten contrast basis functions at points xi, shape (10, N)."""
    xi = np.asarray(xi, dtype=float)
    return np.array(
        [
            sum(c * xi[:, 0] ** e[0] * xi[:, 1] ** e[1] * xi[:, 2] ** e[2] for e, c in poly.items())
            for poly in CONTRAST_BASIS
        ]
    )


def gram_test(h: float) -> NDArray:
    """<L_a, L_b> over a cell of half-width h, shape (4, 4)."""
    return np.array([[_cell_moment(_add(a, b), h) for b in TEST_EXPONENTS] for a in TEST_EXPONENTS])


def gram_test_source(h: float, n_source: int = 10) -> NDArray:
    """<L_a, m_c> over a cell of half-width h, shape (4, n_source)."""
    return np.array(
        [[_cell_moment(_add(a, c), h) for c in source_exponents(n_source)] for a in TEST_EXPONENTS]
    )


def product_table(n_contrast: int = 4) -> NDArray:
    """P[c, a, b] with Q_a L_b = sum_c P[c, a, b] m_c, Q the first n_contrast contrast basis functions.

    Shape (10, 4, 4) for a linear contrast (n_contrast = 4, Q = L) and (20, 10, 4) for a quadratic one.
    """
    if n_contrast not in (4, 10):
        raise ValueError(f"product_table: n_contrast must be 4 or 10, got {n_contrast}")
    exps = source_exponents(10 if n_contrast == 4 else 20)
    table = np.zeros((len(exps), n_contrast, N_TEST))
    for a, poly in enumerate(CONTRAST_BASIS[:n_contrast]):
        for b, eb in enumerate(TEST_EXPONENTS):
            for ea, coef in poly.items():
                table[exps.index(_add(ea, eb)), a, b] += coef
    return table


def source_expansion(delta: NDArray) -> NDArray:
    """E[c, b]: the 9 x 9 coefficient of m_c in Delta(xi) L_b(xi).

    Args:
        delta: Shape (4, 9, 9): Delta(xi) = sum_a delta[a] L_a(xi), i.e. the cell's mean contrast
            operator and its three gradient coefficients per unit xi (h times the physical gradient); or
            shape (10, 9, 9), the coefficients of the ten functions of CONTRAST_BASIS.

    Returns:
        Shape (10, 4, 9, 9) for a linear contrast, (20, 4, 9, 9) for a quadratic one.
    """
    delta = np.asarray(delta, dtype=complex)
    return np.einsum("cab,aij->cbij", product_table(delta.shape[0]), delta)
