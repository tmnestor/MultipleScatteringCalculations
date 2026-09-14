"""Ewald summation for a planar (2-D) lattice of scalar Helmholtz sources.

Ported from the project's validated `Mathematica/IntraPlaneKambe.wl` (Phase 3b
cycle 1), which built the standard Kambe / layer-KKR object. Nothing here is
newly derived; the formulas and, importantly, the two hard-won corrections are
inherited.

WHAT IT COMPUTES. For a square lattice of pitch ``a_L`` in the x-y plane and a
Bloch vector ``k_par``, the lattice-summed scalar Green's function

    G(r) = sum_{R} g(r - R) e^{i k_par . R},   g(s) = e^{i kappa |s|} / (4 pi |s|)

split by Ewald into a real-space half and a reciprocal-space half, each
absolutely and rapidly convergent. ``ewald_total`` subtracts the R = 0 self-term
to give the sum over R != 0, which is what a lattice propagator needs.

WHY IT IS NEEDED HERE. `slab_scattering`'s periodic kernel summed only one
(2M-1)^2 patch and wrapped it, which is not a lattice sum: it carries an O(1/M)
artifact that a laterally uniform medium provably cannot have
(`scripts/measure_single_plane_core.py`). Extending it to a direct image sum
converges only as 1/n_images -- measured, `scripts/gate_lateral_sum_invariance.py`
-- because the 1/r block is conditionally convergent in two dimensions. Ewald is
the standard cure and this is its foundation.

TWO INHERITED CORRECTIONS, both recorded as root causes in the Mathematica work:

  * the RECIPROCAL ERFC PAIRING. The growing evanescent term e^{+|z| gamma} must
    pair with the faster-decaying erfc(+|z| eta + .), not the other way round.
    At z = 0 both erfc arguments coincide, so THE z = 0 CASE CANNOT DETECT A
    WRONG PAIRING -- only eta-independence at z != 0 does.
  * the R = 0 SELF-TERM must be excluded. It is the singular h_0 self-field and
    is not regular-multipole expandable.

NUMERICAL NOTE (not in the source, needed for a floating-point port). The
products exp(s i kappa d) erfc(d eta + s i kappa / (2 eta)) have cancelling
exponents: writing erfc(u) = exp(-u^2) w(i u) with w the Faddeeva function, the
s-dependent pieces cancel exactly and both terms share the prefactor
exp(-d^2 eta^2 + kappa^2 / (4 eta^2)). The same holds in the reciprocal half with
exp(-|z|^2 eta^2 + kz^2 / (4 eta^2)). Forming it that way keeps the halves in
range; forming it literally overflows.

Conventions inherited: time e^{-i w t}, outgoing h^(1), lattice in x-y with z the
polar axis. SI-agnostic -- kappa, r and a_L need only share units.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.special import wofz


def _erfc_scaled(u: NDArray) -> NDArray:
    """exp(u^2) erfc(u) for complex u, via the Faddeeva function.

    erfc(u) = exp(-u^2) w(i u), so this returns w(i u) -- the part that stays in
    range once the cancelling exponent has been factored out by the caller.
    """
    return wofz(1j * u)


def ewald_real(
    kappa: complex,
    r: NDArray,
    eta: float,
    n_real: int,
    a_l: float,
    k_par: NDArray,
) -> complex:
    """Real-space half of the planar Ewald sum, at general z."""
    idx = np.arange(-n_real, n_real + 1)
    i_g, j_g = np.meshgrid(idx, idx, indexing="ij")
    rx = r[0] - a_l * i_g
    ry = r[1] - a_l * j_g
    d = np.sqrt(rx**2 + ry**2 + r[2] ** 2)
    phase = np.exp(1j * a_l * (k_par[0] * i_g + k_par[1] * j_g))
    # Shared prefactor after the s-dependent exponents cancel; see module docstring.
    pref = np.exp(-(d**2) * eta**2 + kappa**2 / (4.0 * eta**2))
    acc = np.zeros_like(d, dtype=complex)
    for s in (-1.0, 1.0):
        acc += _erfc_scaled(d * eta + s * 1j * kappa / (2.0 * eta))
    return complex(np.sum(phase / d * pref * acc) / (8.0 * np.pi))


def ewald_recip(
    kappa: complex,
    r: NDArray,
    eta: float,
    n_recip: int,
    a_l: float,
    k_par: NDArray,
) -> complex:
    """Reciprocal-space half of the planar Ewald sum, at general z."""
    idx = np.arange(-n_recip, n_recip + 1)
    m_g, n_g = np.meshgrid(idx, idx, indexing="ij")
    b = 2.0 * np.pi / a_l
    kgx = k_par[0] + b * m_g
    kgy = k_par[1] + b * n_g
    kz = np.sqrt(np.asarray(kappa**2 - (kgx**2 + kgy**2), dtype=complex))
    # Branch: Im(kz) >= 0, so evanescent orders decay with |z|.
    kz = np.where(kz.imag < 0, -kz, kz)
    z = abs(float(r[2]))
    area = a_l**2
    phase = np.exp(1j * (kgx * r[0] + kgy * r[1]))
    pref = np.exp(-(z**2) * eta**2 + kz**2 / (4.0 * eta**2))
    # THE PAIRING. +|z| eta goes with the decaying term. A wrong pairing is
    # invisible at z = 0, where both arguments coincide.
    term = _erfc_scaled(z * eta + kz / (2j * eta)) + _erfc_scaled(-z * eta + kz / (2j * eta))
    return complex(np.sum(phase / kz * pref * term) * 1j / (4.0 * area))


def ewald_total(
    kappa: complex,
    r: NDArray,
    eta: float,
    n_real: int,
    n_recip: int,
    a_l: float,
    k_par: NDArray,
) -> complex:
    """Lattice sum over R != 0: both halves, with the self-term removed."""
    r = np.asarray(r, dtype=float)
    rn = float(np.sqrt(r @ r))
    self_term = np.exp(1j * kappa * rn) / (4.0 * np.pi * rn)
    return (
        ewald_real(kappa, r, eta, n_real, a_l, k_par)
        + ewald_recip(kappa, r, eta, n_recip, a_l, k_par)
        - self_term
    )


def direct_sum(
    kappa: complex,
    r: NDArray,
    n_big: int,
    a_l: float,
    k_par: NDArray,
) -> complex:
    """Plain damped lattice sum over R != 0 -- the ground truth for gating.

    Converges only when Im(kappa) > 0, and then only slowly. It exists to check
    the Ewald split, never to be used in anger.
    """
    idx = np.arange(-n_big, n_big + 1)
    i_g, j_g = np.meshgrid(idx, idx, indexing="ij")
    mask = ~((i_g == 0) & (j_g == 0))
    rx = r[0] - a_l * i_g[mask]
    ry = r[1] - a_l * j_g[mask]
    d = np.sqrt(rx**2 + ry**2 + r[2] ** 2)
    phase = np.exp(1j * a_l * (k_par[0] * i_g[mask] + k_par[1] * j_g[mask]))
    return complex(np.sum(np.exp(1j * kappa * d) / (4.0 * np.pi * d) * phase))
