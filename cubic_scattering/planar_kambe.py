"""Planar lattice structure constants D[q, s] to high multipole order.

WHAT THEY ARE.  For a square lattice of pitch ``a_L`` in the x-y plane and a
Bloch vector ``k_par``, the scalar field of every site but the origin,

    G(r) = sum_{R != 0} g(r - R) e^{i k_par . R},   g(s) = e^{i kappa |s|}/(4 pi |s|),

is regular near the origin, and its regular-multipole coefficients are the
Kambe / layer-KKR structure constants:

    G(r) = i kappa sum_{q,p} Dbar[q, p] j_q(kappa r) Y_q^p(r^),
    D[q, s] = (-1)^s Dbar[q, -s] ,

the convention of ``Mathematica/IntraPlaneKambe.wl`` (``Dproj``), which the
lattice couplings of the multipole solve contract with Gaunt coefficients.

HOW, AND WHY THIS WAY.  G is evaluated by the planar Ewald split of
``planar_ewald`` (vectorised here over points) and projected onto spherical
harmonics on a sphere of radius rho0 about the origin.  The Phase-3b chain did
exactly this on a SMALL sphere, rho0 = a_L/4, which is adequate for q <= 4 and
limits its energy balance to ~1e-5.  The field is regular for r < a_L -- the
nearest images sit at a_L -- so the sphere can be taken at rho0 = 0.7 a_L.
Then D_q j_q(kappa rho0) behaves like (rho0/a_L)^q: the division by j_q that
destroys the small-sphere route at high q costs a few digits at q = 20, and
the quadrature's aliasing error falls like (rho0/a_L)^(2 n_theta).

Nothing here is newly derived; the Ewald halves and their corrections are
``planar_ewald``'s, ported from the validated Mathematica.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.special import sph_harm_y, spherical_jn, wofz


def _erfc_scaled(u: NDArray) -> NDArray:
    """exp(u^2) erfc(u) for complex u, via the Faddeeva function.

    Args:
        u: Complex argument.

    Returns:
        w(i u).
    """
    return wofz(1j * u)


def lattice_field(
    kappa: complex,
    pts: NDArray,
    a_l: float,
    k_par: NDArray,
    eta: float,
    n_real: int | None = None,
    n_recip: int | None = None,
) -> NDArray[np.complexfloating]:
    """The R != 0 lattice field at many points: ``planar_ewald.ewald_total`` vectorised.

    Args:
        kappa: Wavenumber, Im >= 0.
        pts: Points, shape (N, 3), (x, y, z) with the lattice in the x-y plane.
        a_l: Lattice pitch.
        k_par: Bloch vector, shape (2,).
        eta: Ewald splitting parameter.  The halves carry e^{kappa^2/(4 eta^2)} and
            cancel to the smaller field, so it must grow with kappa (see
            ``structure_constants``).
        n_real: Real-space shells; default makes d eta >= 6.5 at the last shell.
        n_recip: Reciprocal-space shells; default makes |g|/(2 eta) >= 6.5, which
            a fixed count cannot do once eta a_L grows with kappa a_L.

    Returns:
        Shape (N,).
    """
    if n_real is None:
        n_real = max(4, int(np.ceil(6.5 / (eta * a_l))) + 1)
    if n_recip is None:
        n_recip = max(6, int(np.ceil(13.0 * eta * a_l / (2.0 * np.pi))) + 1)
    pts = np.asarray(pts, dtype=float)
    x, y, z = pts[:, 0:1], pts[:, 1:2], pts[:, 2:3]

    idx = np.arange(-n_real, n_real + 1)
    i_g, j_g = (a.ravel()[None, :] for a in np.meshgrid(idx, idx, indexing="ij"))
    d = np.sqrt((x - a_l * i_g) ** 2 + (y - a_l * j_g) ** 2 + z**2)
    phase = np.exp(1j * a_l * (k_par[0] * i_g + k_par[1] * j_g))
    pref = np.exp(-(d**2) * eta**2 + kappa**2 / (4.0 * eta**2))
    acc = _erfc_scaled(d * eta - 1j * kappa / (2.0 * eta)) + _erfc_scaled(
        d * eta + 1j * kappa / (2.0 * eta)
    )
    real = np.sum(phase / d * pref * acc, axis=1) / (8.0 * np.pi)

    idx = np.arange(-n_recip, n_recip + 1)
    m_g, n_g = (a.ravel()[None, :] for a in np.meshgrid(idx, idx, indexing="ij"))
    b = 2.0 * np.pi / a_l
    kgx, kgy = k_par[0] + b * m_g, k_par[1] + b * n_g
    kz = np.sqrt(np.asarray(kappa**2 - (kgx**2 + kgy**2), dtype=complex))
    kz = np.where(kz.imag < 0, -kz, kz)
    az = np.abs(z)
    pref_r = np.exp(-(az**2) * eta**2 + kz**2 / (4.0 * eta**2))
    # THE PAIRING: +|z| eta goes with the decaying term (see planar_ewald).
    term = _erfc_scaled(az * eta + kz / (2j * eta)) + _erfc_scaled(-az * eta + kz / (2j * eta))
    recip = np.sum(np.exp(1j * (kgx * x + kgy * y)) / kz * pref_r * term, axis=1) * 1j / (4.0 * a_l**2)

    rn = np.sqrt(np.sum(pts**2, axis=1))
    return real + recip - np.exp(1j * kappa * rn) / (4.0 * np.pi * rn)


def default_eta(kappa: complex, a_l: float) -> float:
    """The Ewald parameter: max(3/a_L, |kappa|/4).

    The two halves each carry e^{kappa^2/(4 eta^2)} and cancel to the smaller
    lattice field, losing that many digits.  eta a_L = 3 keeps it at e^4 for
    kappa a_L = 12 but lets it reach e^9 at kappa a_L = 18 and e^14 at 22.5;
    eta = |kappa|/4 caps it at e^4 for every kappa.

    Args:
        kappa: Wavenumber.
        a_l: Lattice pitch.

    Returns:
        eta.
    """
    return max(3.0 / a_l, abs(kappa) / 4.0)


def structure_constants(
    kappa: complex,
    qmax: int,
    a_l: float,
    k_par: NDArray,
    rho_frac: float = 0.7,
    eta: float | None = None,
    n_theta: int = 64,
    field: Callable[[NDArray], NDArray] | None = None,
) -> dict[tuple[int, int], complex]:
    """D[q, s] for 0 <= q <= qmax, by projection on a sphere of radius rho_frac a_L.

    Args:
        kappa: Wavenumber, Im >= 0.
        qmax: Highest order.
        a_l: Lattice pitch.
        k_par: Bloch vector, shape (2,).
        rho_frac: Projection radius over the pitch; must be below 1.
        eta: Ewald parameter; defaults to default_eta(kappa, a_l).
        n_theta: Gauss-Legendre nodes in cos(theta); 2 n_theta uniform in phi.
        field: Override for the lattice field (the damped direct sum, in tests).

    Returns:
        {(q, s): D[q, s]}.
    """
    if not 0.0 < rho_frac < 1.0:
        raise ValueError(
            f"rho_frac = {rho_frac}: the projection sphere must lie inside the nearest "
            "image, 0 < rho_frac < 1; use e.g. rho_frac=0.7"
        )
    rho0 = rho_frac * a_l
    u, wu = np.polynomial.legendre.leggauss(n_theta)
    n_phi = 2 * n_theta
    phi = 2.0 * np.pi * np.arange(n_phi) / n_phi
    theta = np.arccos(u)
    th_grid, ph_grid = np.meshgrid(theta, phi, indexing="ij")
    wts = (wu[:, None] * np.full(n_phi, 2.0 * np.pi / n_phi)[None, :]).ravel()
    th, ph = th_grid.ravel(), ph_grid.ravel()
    pts = rho0 * np.column_stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    f = (
        field(pts)
        if field is not None
        else lattice_field(kappa, pts, a_l, k_par, default_eta(kappa, a_l) if eta is None else eta)
    )
    out: dict[tuple[int, int], complex] = {}
    for q in range(qmax + 1):
        denom = 1j * kappa * spherical_jn(q, kappa * rho0)
        for s in range(-q, q + 1):
            integ = np.sum(wts * f * np.conj(sph_harm_y(q, -s, th, ph)))
            out[q, s] = complex((-1) ** s * integ / denom)
    return out
