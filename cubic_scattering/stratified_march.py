"""Plane-wave modes of Legendre cells: the projections the stratified vertical march is built on.

Plan: ``docs/2026-10-10-stratified-reference-legendre-cells-3d.md``.

THE FACTORISATION.  Between planes with z - z' of one sign, the whole-space 9 x 9 propagator (the one
Paper 2's blocks integrate, ``graded_voxel.kernel.kernel_9x9``) is, per lateral wavenumber (k_x, k_y),

    P(x - x') = (1 / 4 pi^2) int d^2 k_h  sum_m  w_m  d_m d_m^T M  exp(i k_m . (x - x')),

with one term per mode m of the half propagating from x' to x (P, SV, SH): k_m = (+-k_z,m, k_x, k_y),
d_m = ``sweep_modes.mode_state(k_m, e_m)`` the mode's (u, engineering strain) 9-vector, its polarisation
e_m unit in the bilinear sense, w_m = i / (2 rho c_m^2 k_z,m), and M = diag(1, 1, 1, 1, 1, 1, 1/2, 1/2, 1/2)
the pairing of the engineering-stress source entries with the engineering-strain state.  The source map is
the receiver map transposed, as reciprocity says, with no other factor (gate G1 of
``scripts/gate_march_modes.py``).

THE CELLS.  A Legendre cell's Galerkin coupling then factorises too: the receiver cell enters through
<Q_a, exp(i k.x)> (Legendre field functions, closed form 2 i^l j_l) and the source cell through
int m_c exp(-i k.x) (monomials, ``graded_voxel.farfield.monomial_fourier``).  For cells in different,
non-touching planes this reproduces Paper 2's exact blocks (gate G2).  For cells in TOUCHING planes it does
not converge: there is no exponential decay left once the cell moments are taken, and those blocks must
come from Paper 2's closed forms.

Conventions: seismic units, time e^{-i omega t}, every k_z on Im >= 0, axes z = 0 (down), x = 1, y = 2.
"""

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray
from scipy.special import spherical_jn

from .effective_contrasts import ReferenceMedium
from .graded_voxel.basis import SOURCE_EXPONENTS_QUARTIC
from .graded_voxel.farfield import monomial_fourier
from .resonance_tmatrix import VOIGT_PAIRS
from .sweep_kernels import _branch

#: Pairing of the engineering-stress source entries with the engineering-strain state.
SOURCE_PAIRING = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5])


def mode_states(k_vec: NDArray, pol: NDArray) -> NDArray:
    """Batched ``sweep_modes.mode_state``: (u, engineering strain) of plane waves, shape (N, 9).

    Args:
        k_vec: Wave vectors in (z, x, y) order, shape (N, 3), complex.
        pol: Polarisations, shape (N, 3).

    Returns:
        The 9-vectors, shape (N, 9).
    """
    out = np.zeros((len(k_vec), 9), dtype=complex)
    out[:, :3] = pol
    for a, (p, q) in enumerate(VOIGT_PAIRS):
        val = 0.5j * (k_vec[:, p] * pol[:, q] + k_vec[:, q] * pol[:, p])
        out[:, 3 + a] = val if p == q else 2.0 * val
    return out


def mode_family(
    kx: NDArray,
    ky: NDArray,
    omega: complex,
    ref: ReferenceMedium,
    wave: str,
    sign: float,
) -> list[tuple[NDArray, NDArray, NDArray]]:
    """The modes of one wave type at many lateral wavenumbers.

    Args:
        kx: Lateral wavenumbers along x, shape (N,), 1/km.
        ky: Lateral wavenumbers along y, shape (N,), 1/km.
        omega: Complex angular frequency.
        ref: The medium.
        wave: "P" (one mode) or "S" (SV then SH).
        sign: +1 for down-going, -1 for up-going.

    Returns:
        One (k, d, w) per mode: wave vectors (N, 3), mode states (N, 9), weights (N,).

    Raises:
        ValueError: on an unknown wave type.
    """
    if wave not in ("P", "S"):
        raise ValueError(f"mode_family: wave must be 'P' or 'S', got {wave!r}")
    kx = np.asarray(kx, dtype=float)
    ky = np.asarray(ky, dtype=float)
    c = ref.alpha if wave == "P" else ref.beta
    kz = _branch((omega / c) ** 2 - kx**2 - ky**2)
    k = np.stack([sign * kz, kx + 0j, ky + 0j], axis=-1)
    w = 1j / (2.0 * ref.rho * c**2 * kz)
    if wave == "P":
        # k.k = (omega/c)^2 exactly, so k c / omega is unit in the bilinear sense.
        return [(k, mode_states(k, k * (c / omega)), w)]
    kh = np.hypot(kx, ky)
    safe = np.where(kh < 1e-14, 1.0, kh)
    # SH horizontal, perpendicular to the azimuth; x at normal incidence, as sweep_modes.mode_basis.
    sh = (
        np.stack(
            [
                0.0 * kx,
                np.where(kh < 1e-14, 1.0, -ky / safe),
                np.where(kh < 1e-14, 0.0, kx / safe),
            ],
            -1,
        )
        + 0j
    )
    sv = np.cross(k, sh) * (c / omega)  # |k x sh|^2 = k.k (sh.sh) = (omega/c)^2
    return [(k, mode_states(k, sv), w), (k, mode_states(k, sh), w)]


def receiver_moments(k_vec: NDArray, h: float, n_field: int) -> NDArray:
    """<Q_a, exp(i k.(x - c))> over a cell of half-width h centred at c, shape (N, n_field).

    The field functions are the products of Legendre polynomials of degrees SOURCE_EXPONENTS[a], as in
    ``graded_voxel.solver.plane_wave_moments``; here for complex k.
    """
    j = {
        (i, e): 2.0 * 1j**e * spherical_jn(e, k_vec[:, i] * h)
        for i in range(3)
        for e in range(3)
    }
    return np.stack(
        [
            h**3 * j[0, e[0]] * j[1, e[1]] * j[2, e[2]]
            for e in SOURCE_EXPONENTS_QUARTIC[:n_field]
        ],
        -1,
    )


def source_moments_local(k_vec: NDArray, h: float, n_source: int) -> NDArray:
    """int m_c(xi) exp(-i k.(x - c)) d^3x over a cell of half-width h, shape (N, n_source)."""
    f = {
        (i, e): monomial_fourier(e, -k_vec[:, i] * h)
        for i in range(3)
        for e in range(5)
    }
    return np.stack(
        [
            h**3 * f[0, e[0]] * f[1, e[1]] * f[2, e[2]]
            for e in SOURCE_EXPONENTS_QUARTIC[:n_source]
        ],
        -1,
    )


def mode_integral_block(
    offset: tuple[int, int, int],
    h: float,
    omega: complex,
    ref: ReferenceMedium,
    n_source: int = 10,
    n_test: int = 4,
    n_theta: int = 72,
    n_t: int = 96,
    n_phi: int = 128,
    decay: float = 40.0,
) -> NDArray:
    """Paper 2's Galerkin block K[a, c] between cells in DIFFERENT planes, by the mode integral.

    The arbiter of the mode factorisation (gate G2), not a production route: a polar (k_h, phi) rule,
    with k_h = k sin(theta) on the propagating branch and k_h = k cosh(t) on the evanescent one, each
    wave type on its own branch point, so the 1/k_z of the weight is removed.

    Args:
        offset: Grid offset of the receiver cell from the source cell, in cells (pitch 2h); offset[0] != 0.
        h: Cell half-width, km.
        omega: Angular frequency.
        ref: The medium.
        n_source: Source monomials (10, 20 or 35).
        n_test: Field functions (4 or 10).
        n_theta: Gauss nodes on the propagating branch, per wave type.
        n_t: Gauss nodes on the evanescent branch, per wave type.
        n_phi: Trapezoid nodes in azimuth.
        decay: The evanescent branch is cut where exp(-kappa gap) = exp(-decay).

    Returns:
        K, shape (n_test, n_source, 9, 9), as ``graded_voxel.blocks.coupling_block``.

    Raises:
        ValueError: when the cells are in the same plane.
    """
    if offset[0] == 0:
        raise ValueError(
            "mode_integral_block: offset[0] == 0 puts both cells in one plane, where the z-separation has no\n"
            "  sign and the mode integral does not exist. Same-plane blocks come from Paper 2's closed forms."
        )
    r_vec = 2.0 * h * np.asarray(offset, dtype=float)
    sign = float(np.sign(r_vec[0]))
    gap = max(abs(r_vec[0]) - 2.0 * h, 0.4 * h)
    out = np.zeros((n_test, n_source, 9, 9), dtype=complex)
    phi = 2.0 * np.pi * np.arange(n_phi) / n_phi
    x_th, w_th = leggauss(n_theta)
    x_t, w_t = leggauss(n_t)
    for wave, c in (("P", ref.alpha), ("S", ref.beta)):
        kc = float(np.real(omega)) / c
        th = (x_th + 1.0) * np.pi / 4.0
        q1, wq1 = (
            kc * np.sin(th),
            w_th * np.pi / 4.0 * kc * np.cos(th) * kc * np.sin(th),
        )
        t_max = np.arccosh(1.0 + decay / (kc * gap))
        t = (x_t + 1.0) * t_max / 2.0
        q2, wq2 = kc * np.cosh(t), w_t * t_max / 2.0 * kc * np.sinh(t) * kc * np.cosh(t)
        q = np.concatenate([q1, q2])
        wq = np.concatenate([wq1, wq2])
        qq, pp = np.meshgrid(q, phi, indexing="ij")
        weight = (np.repeat(wq[:, None], n_phi, axis=1) / (2.0 * np.pi * n_phi)).ravel()
        kx, ky = (qq * np.cos(pp)).ravel(), (qq * np.sin(pp)).ravel()
        for k, d, w in mode_family(kx, ky, omega, ref, wave, sign):
            coef = weight * w * np.exp(1j * (k @ r_vec))
            ra = receiver_moments(k, h, n_test)
            sc = source_moments_local(k, h, n_source)
            out += np.einsum("n,na,nc,ni,nj->acij", coef, ra, sc, d, d * SOURCE_PAIRING)
    return out
