"""Bloch-summed scalar derivative tensors, and the same-plane 9x9 from them.

THE OBJECT. For a square lattice of pitch ``a_L`` lying in the horizontal plane
and a Bloch vector ``k_par``, this builds

    D^{(n)}_{i1..in}(r) = sum_{R != 0}  [d_{i1..in} g](r - R)  e^{i k_par . R},
    g(s) = exp(i kappa |s|) / (4 pi |s|),

for n = 0..4, and then hands (G, Gd, Gdd) to the same Kupradze assembly the point
propagator uses. Summation is linear and the Kupradze operator has constant
coefficients, so the two commute -- that is the whole reason this works, and
`test_greens_from_scalars_is_linear_in_the_scalar_tensors` gates it.

WHY FOURTH ORDER. The 9-component basis is (u_z, u_x, u_y, e_zz, e_xx, e_yy,
2e_xy, 2e_zy, 2e_zx). Its [[G, C], [H, S]] blocks need G_ij (2 derivatives of the
scalars), G_ij,k (3) and the strain-strain block G_ij,kl (4). The strain block is
also the one that makes the UNSPLIT reciprocal sum diverge -- two more powers of
q turn a 1/(2|q|) summand into ~|q|/2 -- which is why Ewald is not optional here.

═══ COORDINATE CONVENTIONS -- THE TRAP IN THIS FILE ═══════════════════════════
Two orderings meet here and they are not the same:

  * `planar_ewald` (ported from the Mathematica) is (x, y, z): the lattice lies
    in components 0 and 1, and the out-of-plane coordinate is component 2.
  * everything that feeds the 9x9 -- `_voigt_contract`, `_propagator_block_9x9`,
    `slab_scattering` -- is the project's seismological (z, x, y): depth FIRST.

This module works ENTIRELY in (z, x, y), so its output drops straight into the
9x9 assembly, and it permutes at the boundary when it calls `planar_ewald`. The
project's record already carries one defect of exactly this shape (two index
conventions in `horizontal_greens`, differing by a fixed 3.6e-1 that refinement
does not remove), so the ordering is asserted by a gate rather than assumed.
═══════════════════════════════════════════════════════════════════════════════

WHAT IS EXACT AND WHAT IS NOT. The Ewald split is exact for any eta; eta only
moves work between the two halves. That makes ETA-INDEPENDENCE the sharpest
available test, and it is applied here to every derivative order, not just to
the scalar -- a wrong fourth-derivative recursion is eta-dependent even when the
scalar is perfect.

A z-reflection invariant holds at the same-plane point and is worth stating
because it is free evidence: every lattice vector lies in the plane, so the
summed scalar depends on z only through z^2. Hence at z = 0 every derivative
tensor with an ODD number of z indices vanishes identically. Nothing enforces
this in the construction, so it is a genuine check.

Conventions inherited: time e^{-i w t}, outgoing h^(1).
"""

import math

import numpy as np
from numpy.typing import NDArray
from scipy.special import wofz

from .effective_contrasts import ReferenceMedium
from .kupradze_derivatives import (
    MAX_ORDER,
    _delta_x_structure,
    greens_from_scalars,
    scalar_derivative_tensors,
)
from .resonance_tmatrix import _voigt_contract

_SQRT_PI = math.sqrt(math.pi)


# ---------------------------------------------------------------------------
# Derivative recursions -- each one an integer recursion, none transcribed
# ---------------------------------------------------------------------------


def _wofz_derivatives(zeta: complex, order: int) -> list[complex]:
    """w(zeta) and its derivatives, from w' = -2 zeta w + 2i/sqrt(pi).

    Differentiating that identity m times by Leibniz gives
    w^(m+1) = -2 (zeta w^(m) + m w^(m-1)), the constant dropping out for m >= 1.
    """
    out = [complex(wofz(zeta))]
    if order >= 1:
        out.append(-2.0 * zeta * out[0] + 2.0j / _SQRT_PI)
    for m in range(1, order):
        out.append(-2.0 * (zeta * out[m] + m * out[m - 1]))
    return out


def _gaussian_derivatives(t: float, eta: float, scale: complex, order: int) -> list[complex]:
    """Derivatives of ``scale * exp(-t^2 eta^2)``, via E' = -2 t eta^2 E."""
    out = [complex(scale * np.exp(-(t**2) * eta**2))]
    for m in range(order):
        prev = out[m - 1] if m >= 1 else 0.0
        out.append(-2.0 * eta**2 * (t * out[m] + m * prev))
    return out


def _leibniz(f: list[complex], g: list[complex], order: int) -> list[complex]:
    """Plain derivatives of a product from those of its factors."""
    return [complex(sum(math.comb(n, m) * f[m] * g[n - m] for m in range(n + 1))) for n in range(order + 1)]


def ladder_from_plain(plain: list[complex], d: float) -> list[complex]:
    """Convert plain radial derivatives f^(j) into the ladder (1/d d/dd)^k f.

    Writing f_k = sum_j alpha[k][j] d^(j-2k) f^(j), the operator gives the
    integer recursion

        alpha[k+1][j] = alpha[k][j] (j - 2k) + alpha[k][j-1],   alpha[0][0] = 1,

    so the conversion carries no hand-derived coefficients. This is the bridge
    that lets ANY radial function -- not just the Helmholtz kernel with its
    Hankel ladder -- be fed to the Cartesian derivative structures.
    """
    order = len(plain) - 1
    alpha = [[0] * (order + 1) for _ in range(order + 1)]
    alpha[0][0] = 1
    out = [plain[0]]
    for k in range(order):
        for j in range(order + 1):
            prev_j = alpha[k][j - 1] if j >= 1 else 0
            alpha[k + 1][j] = alpha[k][j] * (j - 2 * k) + prev_j
        out.append(
            complex(
                sum(
                    alpha[k + 1][j] * d ** (j - 2 * (k + 1)) * plain[j]
                    for j in range(order + 1)
                    if alpha[k + 1][j]
                )
            )
        )
    return out


def screened_radial_ladder(d: float, kappa: complex, eta: float, order: int = MAX_ORDER) -> list[complex]:
    """Ladder of the real-space Ewald summand, a radial function of d.

    The summand is (1/(8 pi d)) sum_{s=+-1} exp(s i kappa d) erfc(d eta + c_s)
    with c_s = s i kappa / (2 eta). Written with erfc(u) = exp(-u^2) w(i u), the
    s-dependent exponents CANCEL -- exp(s i kappa d) exp(-(d eta + c_s)^2)
    reduces to exp(-d^2 eta^2 + kappa^2/(4 eta^2)) for both s -- so the summand
    is a single Gaussian times the sum of two Faddeeva values. That cancellation
    is what keeps the halves in range (see `planar_ewald`'s numerical note) and
    it also makes the derivatives clean: two textbook recursions and Leibniz.
    """
    gauss = _gaussian_derivatives(d, eta, np.exp(kappa**2 / (4.0 * eta**2)), order)

    v = [0.0 + 0.0j] * (order + 1)
    for s in (-1.0, 1.0):
        zeta = 1j * (d * eta + s * 1j * kappa / (2.0 * eta))
        w_der = _wofz_derivatives(zeta, order)
        for n in range(order + 1):
            v[n] += (1j * eta) ** n * w_der[n]

    b = _leibniz(gauss, v, order)
    inv_d = [complex((-1) ** m * math.factorial(m) / d ** (m + 1)) for m in range(order + 1)]
    plain = _leibniz(b, inv_d, order)
    return ladder_from_plain([p / (8.0 * np.pi) for p in plain], d)


# ---------------------------------------------------------------------------
# The two Ewald halves, as derivative tensors in (z, x, y)
# ---------------------------------------------------------------------------


def _tensors_from_ladder(ladder: list[complex], s_vec: NDArray, order: int) -> list[NDArray]:
    """Cartesian derivative tensors of a radial function from its ladder."""
    x = np.asarray(s_vec, dtype=complex)
    out: list[NDArray] = [np.asarray(ladder[0], dtype=complex)]
    for n in range(1, order + 1):
        acc = np.zeros((3,) * n, dtype=complex)
        for m in range(n // 2 + 1):
            acc = acc + ladder[n - m] * _delta_x_structure(n, m, x)
        out.append(acc)
    return out


def ewald_real_tensors(
    r_vec: NDArray,
    kappa: complex,
    eta: float,
    n_real: int,
    a_l: float,
    k_par: NDArray,
    order: int = MAX_ORDER,
) -> list[NDArray]:
    """Real-space half, differentiated. r_vec and the output are (z, x, y)."""
    out: list[NDArray] = [np.zeros((3,) * n, dtype=complex) for n in range(order + 1)]
    for i in range(-n_real, n_real + 1):
        for j in range(-n_real, n_real + 1):
            s_vec = np.array([r_vec[0], r_vec[1] - a_l * i, r_vec[2] - a_l * j])
            d = float(np.linalg.norm(s_vec))
            phase = np.exp(1j * a_l * (k_par[0] * i + k_par[1] * j))
            ladder = screened_radial_ladder(d, kappa, eta, order)
            for n, t in enumerate(_tensors_from_ladder(ladder, s_vec, order)):
                out[n] = out[n] + phase * t
    return out


def ewald_recip_tensors(
    r_vec: NDArray,
    kappa: complex,
    eta: float,
    n_recip: int,
    a_l: float,
    k_par: NDArray,
    order: int = MAX_ORDER,
) -> list[NDArray]:
    """Reciprocal half, differentiated. r_vec and the output are (z, x, y).

    Lateral derivatives are algebraic -- each one brings down i q. Only the
    out-of-plane derivatives need work, and there the z-dependence is a Gaussian
    times w(zeta_+) + w(zeta_-), which is manifestly EVEN in z: the pair is
    W(z eta) + W(-z eta). So the |z| that appears in the published formula
    introduces no kink, and z can be used directly.
    """
    z = float(r_vec[0])
    b = 2.0 * np.pi / a_l
    area = a_l**2
    out: list[NDArray] = [np.zeros((3,) * n, dtype=complex) for n in range(order + 1)]

    for m in range(-n_recip, n_recip + 1):
        for n_idx in range(-n_recip, n_recip + 1):
            qx = k_par[0] + b * m
            qy = k_par[1] + b * n_idx
            kz = np.sqrt(complex(kappa**2 - (qx**2 + qy**2)))
            if kz.imag < 0:
                kz = -kz  # Im(kz) >= 0, so evanescent orders decay with |z|.

            gauss = _gaussian_derivatives(z, eta, np.exp(kz**2 / (4.0 * eta**2)), order)
            v = [0.0 + 0.0j] * (order + 1)
            for sgn in (1.0, -1.0):
                zeta = 1j * (sgn * z * eta + kz / (2j * eta))
                w_der = _wofz_derivatives(zeta, order)
                for p in range(order + 1):
                    v[p] += (sgn * 1j * eta) ** p * w_der[p]
            p_der = _leibniz(gauss, v, order)

            amp = 1j / (4.0 * area) * np.exp(1j * (qx * r_vec[1] + qy * r_vec[2])) / kz
            # Index 0 is z; indices 1 and 2 are the lattice plane.
            lateral = (0.0, 1j * qx, 1j * qy)
            for deg in range(order + 1):
                if deg == 0:
                    out[0] = out[0] + amp * p_der[0]
                    continue
                acc = np.zeros((3,) * deg, dtype=complex)
                for idx in np.ndindex(*((3,) * deg)):
                    c = idx.count(0)
                    factor = 1.0 + 0.0j
                    for axis in idx:
                        if axis != 0:
                            factor *= lateral[axis]
                    acc[idx] = factor * p_der[c]
                out[deg] = out[deg] + amp * acc
    return out


def lattice_scalar_tensors(
    r_vec: NDArray,
    kappa: complex,
    eta: float,
    n_real: int,
    n_recip: int,
    a_l: float,
    k_par: NDArray,
    order: int = MAX_ORDER,
) -> list[NDArray]:
    """Derivative tensors of the Bloch lattice sum over R != 0, in (z, x, y).

    Both halves plus the removal of the R = 0 self-term, which is the singular
    self-field and belongs in the local T-matrix, not in the propagator.
    """
    real_half = ewald_real_tensors(r_vec, kappa, eta, n_real, a_l, k_par, order)
    recip_half = ewald_recip_tensors(r_vec, kappa, eta, n_recip, a_l, k_par, order)
    self_term = scalar_derivative_tensors(r_vec, kappa, order)
    return [a + b - c for a, b, c in zip(real_half, recip_half, self_term, strict=True)]


def _regularised_self_ladder(kappa: complex, eta: float, order: int) -> list[complex]:
    """Ladder at d = 0 of [screened R = 0 term] - [free-space self-term].

    WHY THIS IS NEEDED AND WHY IT IS DELICATE. The lattice sum a periodic solver
    wants at the DIAGONAL is the field at a cube from all of its own in-plane
    images: the sum over R != 0 evaluated at r = 0. But r = 0 is a lattice point,
    so the real-space half's R = 0 term and the self-term subtraction are each
    singular. Their difference is finite -- and evaluating it by subtracting two
    large numbers near r = 0 is catastrophic, worsening with every derivative.

    THE CANCELLATION, exactly. Both terms are (1/(4 pi d)) times a function
    analytic at d = 0:

        screened R=0 :  A(d) / (4 pi d),   A = E(d) V(d) / 2
        free self    :  B(d) / (4 pi d),   B = exp(i kappa d)

    with A(0) = 1 because V(0) = 2 exp(-kappa^2/(4 eta^2)) -- the two erfc values
    at +-i kappa/(2 eta) sum to 2 -- and B(0) = 1. So the poles cancel exactly and
    the difference is the analytic function sum_j b_j d^j with
    b_j = (A - B)^(j+1)(0) / ((j+1)! 4 pi).

    It must moreover be EVEN in d. The total lattice sum is smooth at r = 0 (every
    other term is regular there), and every other piece of the split is smooth, so
    this one has to be; an odd power of d = |r| is not smooth in r. The odd
    coefficients therefore vanish, and `origin_scalar_tensors` checks that they do
    rather than assuming it -- it is free evidence that the whole split is
    consistent.

    Being even, the function is analytic in d^2, and (1/d d/dd)^k d^(2k) = 2^k k!
    gives the ladder at the origin directly: f_k(0) = b_{2k} 2^k k!.
    """
    n_taylor = order + 1
    gauss = _gaussian_derivatives(0.0, eta, np.exp(kappa**2 / (4.0 * eta**2)), n_taylor)
    v = [0.0 + 0.0j] * (n_taylor + 1)
    for s in (-1.0, 1.0):
        zeta = 1j * (s * 1j * kappa / (2.0 * eta))
        w_der = _wofz_derivatives(zeta, n_taylor)
        for n in range(n_taylor + 1):
            v[n] += (1j * eta) ** n * w_der[n]
    a_plain = [x / 2.0 for x in _leibniz(gauss, v, n_taylor)]
    b_plain = [complex((1j * kappa) ** m) for m in range(n_taylor + 1)]

    # b_j, the Taylor coefficients of the difference divided by d.
    coeff = [
        (a_plain[j + 1] - b_plain[j + 1]) / (math.factorial(j + 1) * 4.0 * np.pi) for j in range(n_taylor)
    ]
    scale = max(abs(c) for c in coeff)
    odd = max((abs(coeff[j]) for j in range(1, n_taylor, 2)), default=0.0)
    if scale > 0.0 and odd / scale > 1e-8:
        raise ValueError(
            "lattice_kupradze: the regularised self-term has a non-vanishing ODD "
            f"Taylor coefficient (worst {odd / scale:.2e} relative). It must be even "
            "in d, because every other piece of the Ewald split is smooth at r = 0 "
            "and the total lattice sum is too. A non-zero odd coefficient means the "
            "screened kernel and the subtracted self-term do not match -- check that "
            "both use the same kappa branch and the same 1/(4 pi r) normalisation."
        )
    return [
        complex(coeff[2 * k] * 2**k * math.factorial(k)) if 2 * k < n_taylor else 0.0j
        for k in range(order + 1)
    ]


def origin_scalar_tensors(
    kappa: complex,
    eta: float,
    n_real: int,
    n_recip: int,
    a_l: float,
    k_par: NDArray,
    order: int = MAX_ORDER,
) -> list[NDArray]:
    """The lattice sum over R != 0 evaluated AT r = 0, with derivatives.

    This is the diagonal term a periodic solver needs: what a cube sees from all
    of its own in-plane images. At r = 0 the Cartesian vector x vanishes, so in
    ``sum_m f_{n-m} S_{n,m}(x)`` only the all-delta structure survives -- every
    ODD-order tensor is identically zero and the even ones reduce to a single
    coefficient times the sum of delta pairings.
    """
    zero = np.zeros(3)
    out: list[NDArray] = [np.zeros((3,) * n, dtype=complex) for n in range(order + 1)]

    # Real-space half over R != 0 only; the R = 0 term is handled analytically.
    for i in range(-n_real, n_real + 1):
        for j in range(-n_real, n_real + 1):
            if i == 0 and j == 0:
                continue
            s_vec = np.array([0.0, -a_l * i, -a_l * j])
            d = float(np.linalg.norm(s_vec))
            phase = np.exp(1j * a_l * (k_par[0] * i + k_par[1] * j))
            ladder = screened_radial_ladder(d, kappa, eta, order)
            for n, t in enumerate(_tensors_from_ladder(ladder, s_vec, order)):
                out[n] = out[n] + phase * t

    # Reciprocal half is regular at r = 0 and needs no special treatment.
    for n, t in enumerate(ewald_recip_tensors(zero, kappa, eta, n_recip, a_l, k_par, order)):
        out[n] = out[n] + t

    # The regularised R = 0 term: only all-delta structures survive at x = 0.
    self_ladder = _regularised_self_ladder(kappa, eta, order)
    out[0] = out[0] + self_ladder[0]
    for n in range(2, order + 1, 2):
        out[n] = out[n] + self_ladder[n // 2] * _delta_x_structure(n, n // 2, zero.astype(complex))
    return out


def lattice_block_9x9(
    r_vec: NDArray,
    omega: complex,
    ref: ReferenceMedium,
    eta: float,
    n_real: int,
    n_recip: int,
    a_l: float,
    k_par: NDArray,
) -> NDArray:
    """The Bloch-summed 9x9 [[G, C], [H, S]] at separation r_vec, in (z, x, y).

    This is the object `slab_scattering`'s periodic kernel needs and does not
    currently have: a TRUE lattice sum rather than a truncated patch. At
    r_vec = (0, .., ..) it is the same-plane term, the hard case.
    """
    r_vec = np.asarray(r_vec, dtype=float)
    if float(np.linalg.norm(r_vec)) < 1.0e-14:
        # The diagonal: a cube and its own images. Needs the analytic origin
        # limit, because the R = 0 screened term and the self-term are each
        # singular there and cancel.
        d_p = origin_scalar_tensors(omega / ref.alpha, eta, n_real, n_recip, a_l, k_par)
        d_s = origin_scalar_tensors(omega / ref.beta, eta, n_real, n_recip, a_l, k_par)
    else:
        d_p = lattice_scalar_tensors(r_vec, omega / ref.alpha, eta, n_real, n_recip, a_l, k_par)
        d_s = lattice_scalar_tensors(r_vec, omega / ref.beta, eta, n_real, n_recip, a_l, k_par)
    G, Gd, Gdd = greens_from_scalars(d_p, d_s, omega, ref)
    C, H, S = _voigt_contract(Gd, Gdd)
    P = np.zeros((9, 9), dtype=complex)
    P[:3, :3] = G
    P[:3, 3:] = C
    P[3:, :3] = H
    P[3:, 3:] = S
    return P


def direct_scalar_tensors(
    r_vec: NDArray,
    kappa: complex,
    n_big: int,
    a_l: float,
    k_par: NDArray,
    order: int = MAX_ORDER,
) -> list[NDArray]:
    """Plain damped lattice sum of the derivative tensors -- the gating arbiter.

    Converges only for Im(kappa) > 0, and slowly even then. It shares no code
    path with the Ewald halves, which is the point; it must be shown converged
    in its own radius before it is allowed to convict anything.
    """
    out: list[NDArray] = [np.zeros((3,) * n, dtype=complex) for n in range(order + 1)]
    for i in range(-n_big, n_big + 1):
        for j in range(-n_big, n_big + 1):
            if i == 0 and j == 0:
                continue
            s_vec = np.array([r_vec[0], r_vec[1] - a_l * i, r_vec[2] - a_l * j])
            phase = np.exp(1j * a_l * (k_par[0] * i + k_par[1] * j))
            for n, t in enumerate(scalar_derivative_tensors(s_vec, kappa, order)):
                out[n] = out[n] + phase * t
    return out


__all__ = [
    "direct_scalar_tensors",
    "ewald_real_tensors",
    "ewald_recip_tensors",
    "ladder_from_plain",
    "lattice_block_9x9",
    "lattice_scalar_tensors",
    "origin_scalar_tensors",
    "screened_radial_ladder",
]
