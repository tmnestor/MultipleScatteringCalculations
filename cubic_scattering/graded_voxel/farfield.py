"""Far field of the graded voxel: each cell's polynomial source radiated from Gauss nodes.

A cell's source density is sum_c (sum_b E_cb psi_b) m_c(xi), a polynomial of degree <= 2; it is radiated
as point sources at a tensor Gauss rule's nodes (exact for the polynomial, and the outgoing phase varies
by k h <= 0.25 across a cell).  The point-source formula is that of ``foldy_lax_far_field``: force and
Voigt stress with the same sign, u_P = G_P r (r.F + i k_P r.sigma.r), u_S = G_S (F + i k_S sigma.r)_perp.
"""

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray

from ..effective_contrasts import ReferenceMedium
from ..sphere_scattering import _voigt_to_tensor
from .basis import SOURCE_EXPONENTS, monomials, source_expansion
from .solver import GradedVoxelResult


def radiate(
    points: NDArray,
    sources: NDArray,
    omega: float,
    ref: ReferenceMedium,
    directions: NDArray,
    r_distance: float,
) -> tuple[NDArray, NDArray]:
    """Far-field displacement of point sources (force, Voigt stress) at `points`, shape (M, 3) twice."""
    k_p, k_s = omega / ref.alpha, omega / ref.beta
    forces = sources[:, :3]
    # The shear entries are ENGINEERING stress 2 sigma_pq (the contrast operator maps the engineering strain
    # gamma to 2 dmu gamma; the kernel's shear columns halve it), so the tensor is sigma_pq = entry / 2:
    # sphere_scattering._voigt_to_tensor, the helper foldy_lax_far_field uses. scattered_field has a
    # homonym that does NOT halve; importing it radiated shear stress twice (a full-contrast floor, 2026-09-30).
    sig = np.array([_voigt_to_tensor(s[3:]) for s in sources])
    u_p = np.zeros((len(directions), 3), dtype=complex)
    u_s = np.zeros((len(directions), 3), dtype=complex)
    for o, direction in enumerate(np.asarray(directions, dtype=float)):
        rh = direction / np.linalg.norm(direction)
        proj = points @ rh
        gp = np.exp(1j * k_p * (r_distance - proj)) / (4 * np.pi * ref.rho * ref.alpha**2 * r_distance)
        gs = np.exp(1j * k_s * (r_distance - proj)) / (4 * np.pi * ref.rho * ref.beta**2 * r_distance)
        sr = sig @ rh
        qp = forces @ rh + 1j * k_p * (sr @ rh)
        u_p[o] = (gp * qp).sum() * rh
        qs = forces + 1j * k_s * sr
        qs = qs - np.outer(qs @ rh, rh)
        u_s[o] = (gs[:, None] * qs).sum(axis=0)
    return u_p, u_s


def graded_far_field(
    res: GradedVoxelResult,
    directions: NDArray,
    r_distance: float,
    k_hat: NDArray,
    pol: NDArray,
    wave_type: str,
    n_gauss: int = 4,
) -> tuple[NDArray, NDArray]:
    """Far field of the solved graded voxels; the incident wave (k_hat, pol, wave_type) is already in
    res.psi, and the arguments are kept for the same call shape as ``foldy_lax_far_field``."""
    x, w = leggauss(n_gauss)
    xi = np.stack(np.meshgrid(x, x, x, indexing="ij"), -1).reshape(-1, 3)
    ww = np.einsum("i,j,k->ijk", w, w, w).ravel()
    ms = monomials(SOURCE_EXPONENTS, xi)  # (10, G)
    pts, srcs = [], []
    for c, d, psi in zip(res.centres, res.delta, res.psi, strict=True):
        coef = np.einsum("cbij,bj->ci", source_expansion(d), psi)  # (10, 9)
        srcs.append((ms.T @ coef) * (ww * res.h**3)[:, None])
        pts.append(c + res.h * xi)
    return radiate(np.concatenate(pts), np.concatenate(srcs), res.omega, res.ref, directions, r_distance)
