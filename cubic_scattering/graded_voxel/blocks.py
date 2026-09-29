"""Galerkin coupling blocks K_ac(R) of the graded voxel.

K[a, c](R) = int_{V_m} int_{V_n} L_a((x - x_m)/h) P(x - x') m_c((x' - x_n)/h) dx dx',  R = x_m - x_n,

a 9 x 9 block for each test function a (4) and source monomial c (10).  Distant cells: tensor Gauss in
both cells.  The self cell and its 26 touching neighbours: ``near_block`` (the s-form).
"""

from functools import lru_cache

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray

from ..effective_contrasts import ReferenceMedium
from .basis import SOURCE_EXPONENTS, TEST_EXPONENTS, monomials
from .kernel import kernel_9x9


@lru_cache(maxsize=8)
def _cell_rule(n: int) -> tuple[NDArray, NDArray]:
    x, w = leggauss(n)
    xi = np.stack(np.meshgrid(x, x, x, indexing="ij"), -1).reshape(-1, 3)
    return xi, np.einsum("i,j,k->ijk", w, w, w).ravel()


def gauss_order(offset: tuple[int, int, int]) -> int:
    """Gauss points per axis per cell for a non-touching offset."""
    d = max(abs(o) for o in offset)
    if d <= 1:
        raise ValueError(f"gauss_order: offset {offset} touches; use near_block")
    return 10 if d == 2 else 6 if d <= 4 else 4


def far_block(
    offset: tuple[int, int, int], h: float, omega: float, ref: ReferenceMedium, n_gauss: int
) -> NDArray:
    """K[a, c] for a non-touching offset, shape (4, 10, 9, 9)."""
    xi, w = _cell_rule(n_gauss)
    R = 2.0 * h * np.asarray(offset, dtype=float)
    X = (R[None, None, :] + h * (xi[:, None, :] - xi[None, :, :])).reshape(-1, 3)
    P = kernel_9x9(X, omega, ref).reshape(len(xi), len(xi), 81)
    lt = monomials(TEST_EXPONENTS, xi) * w
    ls = monomials(SOURCE_EXPONENTS, xi) * w
    tmp = np.einsum("ap,pqz->aqz", lt, P)
    return (h**6 * np.einsum("cq,aqz->acz", ls, tmp)).reshape(4, 10, 9, 9)
