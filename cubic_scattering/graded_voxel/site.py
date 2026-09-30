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
from .basis import CONTRAST_NORMS, contrast_values, gram_test, gram_test_source, source_expansion

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
    """L2 projection of profile(x) * contrast onto the cell's contrast basis, as (4 or 10, 9, 9) operators.

    Degrees 0 and 1 return the four coefficients of {1, xi_i}; degree 2 returns the ten of
    basis.CONTRAST_BASIS. The basis is orthogonal, so the first four are the same in both.

    Args:
        profile: Scalar factor multiplying all three contrasts, as a function of position.
        centre: Cell centre (m).
        h: Cell half-width (m).
        contrast: The contrast the profile scales.
        ref: Background medium (for the physical-range check).
        omega: Angular frequency (rad/s).
        degree: 0 keeps the cell mean only; 1 adds the linear moments; 2 the quadratic ones.

    Raises:
        ValueError: when the degree is not 0, 1 or 2, or when the fitted contrast leaves the physical
            range at a corner, edge midpoint, face centre or the centre of the cell.
    """
    x, w = leggauss(6)
    xi = np.stack(np.meshgrid(x, x, x, indexing="ij"), -1).reshape(-1, 3)
    ww = np.einsum("i,j,k->ijk", w, w, w).ravel()
    f = np.array([profile(np.asarray(centre) + h * p) for p in xi])
    if degree not in (0, 1, 2):
        raise ValueError(f"cell_contrast_coefficients: degree must be 0, 1 or 2, got {degree}")
    n_keep = (1, 4, 10)[degree]
    coeff = np.zeros(10 if degree == 2 else 4)
    coeff[:n_keep] = (contrast_values(xi)[:n_keep] * ww) @ f / np.array(CONTRAST_NORMS[:n_keep])
    # a linear fit is extreme at the corners; a quadratic one is sampled on the 27 points {-1, 0, 1}^3
    probes = np.array([[s0, s1, s2] for s0 in (-1, 0, 1) for s1 in (-1, 0, 1) for s2 in (-1, 0, 1)])
    corners = coeff @ contrast_values(probes)[: len(coeff)]
    for name, value, background in (
        ("Dlambda", contrast.Dlambda, ref.lam),
        ("Dmu", contrast.Dmu, ref.mu),
        ("Drho", contrast.Drho, ref.rho),
    ):
        worst = float(np.max(np.abs(corners * value)) / background)
        if worst >= PHYSICAL_LIMIT:
            raise ValueError(
                f"cell_contrast_coefficients: the degree-{degree} fit of the profile reaches {worst:.0%} "
                f"of the background {name} inside the cell (cell centre {np.asarray(centre)} m, "
                f"half-width {h} m); the allowed range is below 52% of the background. Fix: refine the "
                "grid (smaller h) or pass a gentler profile or contrast."
            )
    base = contrast_operator(contrast.Dlambda, contrast.Dmu, contrast.Drho, omega)
    return np.array([c * base for c in coeff])


def single_site_t36(h: float, delta: NDArray, self_block: NDArray) -> NDArray:
    """T36 = F (M - K(0) E)^{-1} M, index 9 * a + i.

    Args:
        h: Cell half-width (m).
        delta: Contrast operator coefficients (4, 9, 9) or (10, 9, 9), see ``basis.source_expansion``.
        self_block: K(0), shape (4, 10, 9, 9) or (4, 20, 9, 9) to match, from
            ``blocks.near_block((0, 0, 0), ...)``.
    """
    e = source_expansion(delta)  # (10 or 20, 4, 9, 9)
    m = np.kron(gram_test(h), np.eye(9))
    ke = np.einsum("acij,cbjk->aibk", self_block[:, : e.shape[0]], e).reshape(36, 36)
    f = np.einsum("ac,cbij->aibj", gram_test_source(h, e.shape[0]), e).reshape(36, 36)
    return f @ np.linalg.solve(m - ke, m)
