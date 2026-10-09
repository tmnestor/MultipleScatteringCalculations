"""Radiation of the graded voxel: each cell's polynomial source radiated from Gauss nodes.

A cell's source density is sum_c (sum_b E_cb psi_b) m_c(xi), a polynomial of degree <= 4; it is radiated
as point sources at a tensor Gauss rule's nodes (exact for the polynomial, and the outgoing phase varies
by k h <= 0.25 across a cell).  Two readouts share those sources:

* ``graded_far_field`` keeps the 1/r term only.  Its point-source formula is that of
  ``foldy_lax_far_field``: force and Voigt stress with the same sign, u_P = G_P r (r.F + i k_P r.sigma.r),
  u_S = G_S (F + i k_S sigma.r)_perp.
* ``source_moments`` integrates the sources against exp(-i k.x) in closed form, for any complex k: the
  amplitudes of the plane-wave (Weyl) spectrum, evanescent waves included, need the radiation pattern at
  complex directions.
* ``graded_field`` evaluates the field at a finite distance with the propagator the solve itself uses
  (``kernel.kernel_9x9``), near-field terms included.
"""

from functools import cache

import numpy as np
from numpy.polynomial.legendre import leggauss, poly2leg
from numpy.typing import NDArray
from scipy.special import spherical_jn

from ..effective_contrasts import ReferenceMedium
from ..sphere_scattering import _voigt_to_tensor
from .basis import SOURCE_EXPONENTS_QUARTIC, monomials, source_expansion
from .kernel import kernel_9x9
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
    # homonym that does NOT halve; importing it radiated shear stress twice (a full-contrast floor).
    sig = np.array([_voigt_to_tensor(s[3:]) for s in sources])
    u_p = np.zeros((len(directions), 3), dtype=complex)
    u_s = np.zeros((len(directions), 3), dtype=complex)
    for o, direction in enumerate(np.asarray(directions, dtype=float)):
        rh = direction / np.linalg.norm(direction)
        proj = points @ rh
        gp = np.exp(1j * k_p * (r_distance - proj)) / (
            4 * np.pi * ref.rho * ref.alpha**2 * r_distance
        )
        gs = np.exp(1j * k_s * (r_distance - proj)) / (
            4 * np.pi * ref.rho * ref.beta**2 * r_distance
        )
        sr = sig @ rh
        qp = forces @ rh + 1j * k_p * (sr @ rh)
        u_p[o] = (gp * qp).sum() * rh
        qs = forces + 1j * k_s * sr
        qs = qs - np.outer(qs @ rh, rh)
        u_s[o] = (gs[:, None] * qs).sum(axis=0)
    return u_p, u_s


def radiate_exact(
    points: NDArray,
    sources: NDArray,
    omega: float,
    ref: ReferenceMedium,
    obs_points: NDArray,
) -> tuple[NDArray, NDArray]:
    """Displacement (M, 3) and engineering strain (M, 6) at `obs_points` of point sources at `points`.

    The field is the propagator applied to each source, summed: nothing is dropped, so it holds at any
    distance from the sources.

    Raises:
        ValueError: when an observation point coincides with a source point, where the propagator is a
            distribution.
    """
    obs = np.atleast_2d(np.asarray(obs_points, dtype=float))
    out = np.zeros((len(obs), 9), dtype=complex)
    for o, x in enumerate(obs):
        out[o] = np.einsum("nab,nb->a", kernel_9x9(x - points, omega, ref), sources)
    return out[:, :3], out[:, 3:]


@cache
def _monomial_legendre(n: int) -> tuple[float, ...]:
    """Coefficients of xi^n in the Legendre polynomials P_0 .. P_n."""
    return tuple(poly2leg([0.0] * n + [1.0]))


def monomial_fourier(n: int, q: NDArray) -> NDArray:
    """int_{-1}^{1} xi^n exp(i q xi) d xi for complex q, in closed form.

    xi^n is a combination of P_0 .. P_n, and int P_l exp(i q xi) = 2 i^l j_l(q): the sinc of n = 0
    (j_0(q) = sin q / q) and its derivatives. Through the spherical Bessel functions, not their elementary
    forms, which cancel as q -> 0; ``scipy.special.spherical_jn`` takes complex arguments.
    """
    q = np.asarray(q, dtype=complex)
    return sum(
        c * 2.0 * 1j**l * spherical_jn(l, q)
        for l, c in enumerate(_monomial_legendre(n))
        if c != 0.0
    )


def cell_source_coefficients(res: GradedVoxelResult) -> NDArray:
    """S[cell, c, :]: each cell's source as sum_c S_c m_c(xi), shape (N, n_source, 9)."""
    return np.array(
        [
            np.einsum("cbij,bj->ci", source_expansion(d, psi.shape[0]), psi)
            for d, psi in zip(res.delta, res.psi, strict=True)
        ]
    )


def source_moments(
    res: GradedVoxelResult, k_vec: NDArray, coef: NDArray | None = None
) -> NDArray:
    """Int s(x) exp(-i k.x) d^3x over every cell, summed: the 9 entries (force, Voigt stress) for a complex
    wave vector k_vec. Exact for the cells' polynomial sources, with no quadrature.

    A cell's source is sum_c S_c m_c(xi), x = centre + h xi, so its moment is
    exp(-i k.centre) h^3 sum_c S_c prod_i F_{e_ci}(-k_i h), F = ``monomial_fourier``. ``coef`` is
    ``cell_source_coefficients(res)``, passed when many wave vectors share one solution.
    """
    k_vec = np.asarray(k_vec, dtype=complex)
    coef = cell_source_coefficients(res) if coef is None else coef
    exps = SOURCE_EXPONENTS_QUARTIC[: coef.shape[1]]
    f1 = {
        (i, e): monomial_fourier(e, -k_vec[i] * res.h)
        for i in range(3)
        for e in range(5)
    }
    mono = (
        np.array([f1[0, e[0]] * f1[1, e[1]] * f1[2, e[2]] for e in exps]) * res.h**3
    )  # (n_source,)
    phase = np.exp(-1j * (np.asarray(res.centres) @ k_vec))  # (N,)
    return mono @ np.einsum("n,nci->ci", phase, coef)


def _node_sources(res: GradedVoxelResult, n_gauss: int) -> tuple[NDArray, NDArray]:
    """The cells' polynomial sources as point sources at Gauss nodes: positions (N, 3), sources (N, 9)."""
    x, w = leggauss(n_gauss)
    xi = np.stack(np.meshgrid(x, x, x, indexing="ij"), -1).reshape(-1, 3)
    ww = np.einsum("i,j,k->ijk", w, w, w).ravel()
    ms = monomials(SOURCE_EXPONENTS_QUARTIC, xi)  # (35, G)
    pts, srcs = [], []
    for c, d, psi in zip(res.centres, res.delta, res.psi, strict=True):
        coef = np.einsum(
            "cbij,bj->ci", source_expansion(d, psi.shape[0]), psi
        )  # (n_source, 9)
        srcs.append((ms[: len(coef)].T @ coef) * (ww * res.h**3)[:, None])
        pts.append(c + res.h * xi)
    return np.concatenate(pts), np.concatenate(srcs)


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
    pts, srcs = _node_sources(res, n_gauss)
    return radiate(pts, srcs, res.omega, res.ref, directions, r_distance)


def graded_field(
    res: GradedVoxelResult, obs_points: NDArray, n_gauss: int = 4
) -> tuple[NDArray, NDArray]:
    """Scattered displacement (M, 3) and engineering strain (M, 6) of the solved graded voxels at
    `obs_points`, at any distance outside the cells.

    The Gauss rule integrates the propagator over each cell.  It is exact for the cell's polynomial source
    but not for the propagator, which is singular at the observer: for an observer within about a cell's
    width of a cell that carries contrast, raise `n_gauss` until the value settles.
    """
    pts, srcs = _node_sources(res, n_gauss)
    return radiate_exact(pts, srcs, res.omega, res.ref, obs_points)
