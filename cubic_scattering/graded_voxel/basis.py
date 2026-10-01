"""Local polynomial bases of the graded voxel.

A cell with centre c and half-width h has local coordinates xi = (x - c) / h in [-1, 1]^3, in the
package's axis order (z = 0, x = 1, y = 2).  The field is expanded in the TEST basis {1, xi_0, xi_1,
xi_2} (the Legendre polynomials of degree <= 1).  A contrast linear in the cell times a test function is
at most quadratic, so sources live in the SOURCE basis of the ten monomials of degree <= 2.

A contrast QUADRATIC in the cell is expanded in the ten orthogonal polynomials of CONTRAST_BASIS (the test
basis, then P_2(xi_i) and xi_i xi_j); its product with a test function is at most cubic, and the sources
then live in the twenty monomials of degree <= 3, SOURCE_EXPONENTS_CUBIC, whose first ten are
SOURCE_EXPONENTS.

A FIELD quadratic in the cell is expanded in the same ten orthogonal polynomials (its first four are the
test basis).  With a quadratic contrast its sources are quartic: the 35 monomials of degree <= 4,
SOURCE_EXPONENTS_QUARTIC, whose first twenty are the cubic set.  The source set is fixed by the degree of
the product (``source_size``): 10, 20 or 35 monomials for degree 2, 3 or 4.
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
SOURCE_EXPONENTS_QUARTIC: tuple[Exponent, ...] = SOURCE_EXPONENTS_CUBIC + tuple(
    sorted(
        ((i, j, 4 - i - j) for i in range(5) for j in range(5 - i)),
        key=lambda e: (-max(e), e),
    )
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
    """The source monomials: 10 (degree <= 2), 20 (degree <= 3) or 35 (degree <= 4)."""
    if n_source not in (10, 20, 35):
        raise ValueError(f"source_exponents: n_source must be 10, 20 or 35, got {n_source}")
    return SOURCE_EXPONENTS_QUARTIC[:n_source]


def source_size(n_contrast: int, n_field: int) -> int:
    """Source monomials needed by the product of a contrast (4 or 10 functions) and a field (4 or 10)."""
    for name, n in (("n_contrast", n_contrast), ("n_field", n_field)):
        if n not in (4, 10):
            raise ValueError(f"source_size: {name} must be 4 or 10, got {n}")
    return {2: 10, 3: 20, 4: 35}[(1 if n_contrast == 4 else 2) + (1 if n_field == 4 else 2)]


def field_in_monomials(n_field: int = 10) -> NDArray:
    """C with Q_a = sum_m C[a, m] xi^SOURCE_EXPONENTS[m], shape (n_field, n_field); the identity for 4."""
    c = np.zeros((n_field, n_field))
    for a, poly in enumerate(CONTRAST_BASIS[:n_field]):
        for e, coef in poly.items():
            c[a, SOURCE_EXPONENTS.index(e)] = coef
    return c


def contrast_values(xi: NDArray) -> NDArray:
    """Values of the ten contrast basis functions at points xi, shape (10, N)."""
    xi = np.asarray(xi, dtype=float)
    return np.array(
        [
            sum(c * xi[:, 0] ** e[0] * xi[:, 1] ** e[1] * xi[:, 2] ** e[2] for e, c in poly.items())
            for poly in CONTRAST_BASIS
        ]
    )


def gram_test(h: float, n_field: int = 4) -> NDArray:
    """<Q_a, Q_b> over a cell of half-width h, shape (n_field, n_field): diagonal."""
    return np.diag(h**3 * np.array(CONTRAST_NORMS[:n_field]))


def gram_test_source(h: float, n_source: int = 10, n_field: int = 4) -> NDArray:
    """<Q_a, m_c> over a cell of half-width h, shape (n_field, n_source)."""
    mono = np.array(
        [[_cell_moment(_add(a, c), h) for c in source_exponents(n_source)] for a in SOURCE_EXPONENTS]
    )
    return (field_in_monomials(10) @ mono)[:n_field]


def product_table(n_contrast: int = 4, n_field: int = 4) -> NDArray:
    """P[c, a, b] with Q_a Q_b = sum_c P[c, a, b] m_c: a over the contrast functions, b over the field's.

    Shape (source_size(n_contrast, n_field), n_contrast, n_field): (10, 4, 4) for a linear contrast and
    field, (20, 10, 4) for a quadratic contrast, (35, 10, 10) when both are quadratic.
    """
    exps = source_exponents(source_size(n_contrast, n_field))
    table = np.zeros((len(exps), n_contrast, n_field))
    for a, pa in enumerate(CONTRAST_BASIS[:n_contrast]):
        for b, pb in enumerate(CONTRAST_BASIS[:n_field]):
            for ea, ca in pa.items():
                for eb, cb in pb.items():
                    table[exps.index(_add(ea, eb)), a, b] += ca * cb
    return table


def source_expansion(delta: NDArray, n_field: int = 4) -> NDArray:
    """E[c, b]: the 9 x 9 coefficient of m_c in Delta(xi) Q_b(xi), b over the n_field field functions.

    Args:
        delta: Shape (4, 9, 9): Delta(xi) = sum_a delta[a] L_a(xi), i.e. the cell's mean contrast
            operator and its three gradient coefficients per unit xi (h times the physical gradient); or
            shape (10, 9, 9), the coefficients of the ten functions of CONTRAST_BASIS.

        n_field: 4 for a field linear in the cell, 10 for a quadratic one.

    Returns:
        Shape (n_source, n_field, 9, 9), n_source = source_size(len(delta), n_field).
    """
    delta = np.asarray(delta, dtype=complex)
    return np.einsum("cab,aij->cbij", product_table(delta.shape[0], n_field), delta)
