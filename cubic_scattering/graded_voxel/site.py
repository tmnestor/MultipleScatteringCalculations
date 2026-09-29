"""The graded single site: contrast operators and the 36 x 36 T-matrix T36.

Unknowns psi(x) = sum_b psi_b L_b(xi); the source density Delta(xi) psi(x) = sum_b sum_c E_cb psi_b m_c(xi).
The tested Lippmann-Schwinger equation of one isolated cell is (M - K(0) E) psi = <L, psi0>, with
M = gram_test (x) I_9 and (K E)_ab = sum_c K_ac(0) E_cb.  The source moments are sigma_a = <L_a, Delta psi>
= F psi, F_ab = sum_c <L_a, m_c> E_cb, so

    T36 = F (M - K(0) E)^{-1} M

maps the incident field's Legendre coefficients to the source moments.  Index order: 9 * a + i.
"""

from collections.abc import Callable

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray

from ..effective_contrasts import MaterialContrast, ReferenceMedium
from ..voigt_tmatrix import effective_stiffness_voigt
from .basis import gram_test, gram_test_source, source_expansion

PHYSICAL_LIMIT = 0.52


def contrast_operator(dlam: float, dmu: float, drho: float, omega: float) -> NDArray:
    """Born contrast operator per unit volume, in the package's T_loc convention (9 x 9)."""
    op = np.zeros((9, 9), dtype=complex)
    op[:3, :3] = omega**2 * drho * np.eye(3)
    op[3:, 3:] = effective_stiffness_voigt(dlam, dmu, dmu)
    return op


def cell_contrast_coefficients(
    profile: Callable[[NDArray], float],
    centre: NDArray,
    h: float,
    contrast: MaterialContrast,
    ref: ReferenceMedium,
    omega: float,
    degree: int,
) -> NDArray:
    """L2 projection of profile(x) * contrast onto {1, xi_i} over one cell, as (4, 9, 9) operators.

    Args:
        profile: Scalar factor multiplying all three contrasts, as a function of position.
        centre: Cell centre (m).
        h: Cell half-width (m).
        contrast: The contrast the profile scales.
        ref: Background medium (for the physical-range check).
        omega: Angular frequency (rad/s).
        degree: 0 keeps the cell mean only; 1 adds the linear moments.

    Raises:
        ValueError: when the fitted linear contrast leaves the physical range at a cell corner.
    """
    x, w = leggauss(6)
    xi = np.stack(np.meshgrid(x, x, x, indexing="ij"), -1).reshape(-1, 3)
    ww = np.einsum("i,j,k->ijk", w, w, w).ravel()
    f = np.array([profile(np.asarray(centre) + h * p) for p in xi])
    coeff = np.zeros(4)
    coeff[0] = (ww @ f) / 8.0
    if degree >= 1:
        coeff[1:] = (ww * f) @ xi / (8.0 / 3.0)
    signs = np.array([[s0, s1, s2] for s0 in (-1, 1) for s1 in (-1, 1) for s2 in (-1, 1)])
    corners = coeff[0] + signs @ coeff[1:]
    for name, value, background in (
        ("Dlambda", contrast.Dlambda, ref.lam),
        ("Dmu", contrast.Dmu, ref.mu),
        ("Drho", contrast.Drho, ref.rho),
    ):
        worst = float(np.max(np.abs(corners * value)) / background)
        if worst >= PHYSICAL_LIMIT:
            raise ValueError(
                f"cell_contrast_coefficients: the linear fit of the profile reaches {worst:.0%} of "
                f"the background {name} at a corner of the cell (cell centre {np.asarray(centre)} m, "
                f"half-width {h} m); the allowed range is below 52% of the background. Fix: refine the "
                "grid (smaller h) or pass a gentler profile or contrast."
            )
    base = contrast_operator(contrast.Dlambda, contrast.Dmu, contrast.Drho, omega)
    return np.array([c * base for c in coeff])


def single_site_t36(h: float, delta: NDArray, self_block: NDArray) -> NDArray:
    """T36 = F (M - K(0) E)^{-1} M, index 9 * a + i.

    Args:
        h: Cell half-width (m).
        delta: Contrast operator coefficients (4, 9, 9), see ``basis.source_expansion``.
        self_block: K(0), shape (4, 10, 9, 9), from ``blocks.near_block((0, 0, 0), ...)``.
    """
    e = source_expansion(delta)  # (10, 4, 9, 9)
    m = np.kron(gram_test(h), np.eye(9))
    ke = np.einsum("acij,cbjk->aibk", self_block, e).reshape(36, 36)
    f = np.einsum("ac,cbij->aibj", gram_test_source(h), e).reshape(36, 36)
    return f @ np.linalg.solve(m - ke, m)
