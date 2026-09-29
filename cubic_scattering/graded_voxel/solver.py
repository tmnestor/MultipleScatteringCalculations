"""Dense Galerkin solve of the graded voxel on the FFT sphere solver's grid.

Global system, cells m, n, test functions a, b (a, b < na; na = 1 for p = 0, 4 for p = 1):

    A[(m, a, i), (n, b, j)] = delta_mn M_ab delta_ij - sum_c K_ac(o_mn)[i, :] E^(n)_cb[:, j],

o_mn = grid_idx[m] - grid_idx[n], K from ``blocks.coupling_block``, E^(n) the source expansion of cell n's
contrast (degree r).  The right-hand side is the plane wave's cell moments <L_a, amp9 e^{i k.x}> in closed
form: J_0(q) = 2 sin(q)/q, J_1(q) = 2i (sin q - q cos q)/q^2, moment = h^3 e^{i k.c} prod_i J_{e_i}(k_i h).
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy.linalg
from numpy.typing import NDArray
from scipy.special import spherical_jn

from ..effective_contrasts import MaterialContrast, ReferenceMedium
from ..sphere_scattering import _plane_wave_strain_voigt
from ..sphere_scattering_fft import _build_grid_index_map
from .basis import TEST_EXPONENTS, gram_test, source_expansion
from .blocks import coupling_block
from .site import cell_contrast_coefficients


def _j(e: int, q: float) -> complex:
    """int_{-1}^{1} xi^e exp(i q xi) d xi for e = 0, 1: 2 j_0(q) and 2i j_1(q).

    The spherical Bessel functions, not (sin q - q cos q)/q^2, which cancels as q -> 0 (relative error
    eps/q^2: 2e-13 at q = 0.026).
    """
    return 2.0 * spherical_jn(0, q) if e == 0 else 2j * spherical_jn(1, q)


def plane_wave_moments(centres: NDArray, h: float, k_vec: NDArray, amp9: NDArray) -> NDArray:
    """<L_a, amp9 exp(i k.x)> over each cell, shape (N, 4, 9)."""
    k_vec = np.asarray(k_vec, dtype=float)
    ph = np.exp(1j * np.asarray(centres) @ k_vec)
    mom = np.array([h**3 * np.prod([_j(e[i], k_vec[i] * h) for i in range(3)]) for e in TEST_EXPONENTS])
    return ph[:, None, None] * mom[None, :, None] * np.asarray(amp9)[None, None, :]


@dataclass
class GradedVoxelResult:
    """Solution of the graded-voxel Galerkin system for one plane wave."""

    centres: NDArray
    grid_idx: NDArray
    h: float
    omega: float
    ref: ReferenceMedium
    delta: NDArray  # (N, 4, 9, 9) contrast operator coefficients per cell
    psi: NDArray  # (N, 4, 9) Legendre coefficients of the state
    p: int
    r: int


def solve_graded_sphere(
    omega: float,
    radius: float,
    ref: ReferenceMedium,
    contrast: MaterialContrast,
    n_sub: int,
    profile: Callable[[NDArray], float],
    k_hat: NDArray,
    pol: NDArray,
    wave_type: str,
    p: int = 1,
    r: int = 1,
    blocks: dict | None = None,
    _order: NDArray | None = None,
) -> GradedVoxelResult:
    """Solve (M - K E) psi = <L, psi0> densely for a plane wave of direction k_hat and polarisation pol.

    Args:
        omega: Angular frequency (rad/s).
        radius: Sphere radius (m); cells are kept by centre, as the FFT sphere solver keeps them.
        ref: Background medium.
        contrast: The contrast that ``profile`` scales.
        n_sub: Cells per edge of the bounding cube.
        profile: Scalar factor of the contrast as a function of position.
        k_hat: Incident direction.
        pol: Incident polarisation.
        wave_type: 'P' or 'S' (the incident speed).
        p: Field degree (0 or 1).
        r: Contrast degree (0 or 1).
        blocks: Optional cache {offset: K} reused across calls with the same h, omega and ref.
        _order: Test-only permutation of the cells.
    """
    grid_idx, centres, h = _build_grid_index_map(radius, n_sub)
    if _order is not None:
        grid_idx, centres = grid_idx[_order], centres[_order]
    n = len(centres)
    na = 1 if p == 0 else 4
    delta = np.array(
        [cell_contrast_coefficients(profile, c, h, contrast, ref, omega, degree=r) for c in centres]
    )
    blocks = {} if blocks is None else blocks
    dim = n * na * 9
    a = np.zeros((dim, dim), dtype=complex)
    m9 = np.kron(gram_test(h)[:na, :na], np.eye(9))
    for col in range(n):
        en = source_expansion(delta[col])[:, :na].transpose(0, 2, 1, 3).reshape(90, na * 9)  # (c j, b k)
        for row in range(n):
            off = tuple(int(v) for v in grid_idx[row] - grid_idx[col])
            if off not in blocks:
                blocks[off] = coupling_block(off, h, omega, ref)  # type: ignore[arg-type]
            k = blocks[off][:na].transpose(0, 2, 1, 3).reshape(na * 9, 90)  # (a i, c j)
            blk = -(k @ en)
            if row == col:
                blk = blk + m9
            a[row * na * 9 : (row + 1) * na * 9, col * na * 9 : (col + 1) * na * 9] = blk
    k_mag = omega / (ref.alpha if wave_type == "P" else ref.beta)
    k_hat = np.asarray(k_hat, dtype=float) / np.linalg.norm(k_hat)
    amp = np.concatenate([np.asarray(pol, dtype=complex), _plane_wave_strain_voigt(k_hat, pol, k_mag)])
    rhs = plane_wave_moments(centres, h, k_mag * k_hat, amp)[:, :na].ravel()
    psi = scipy.linalg.solve(a, rhs, overwrite_a=True, check_finite=False).reshape(n, na, 9)
    full = np.zeros((n, 4, 9), dtype=complex)
    full[:, :na] = psi
    return GradedVoxelResult(centres, grid_idx, h, omega, ref, delta, full, p, r)
