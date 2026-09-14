"""Cartesian derivative tensors of a scalar Helmholtz field, to fourth order.

WHY THIS EXISTS. The 9x9 elastic propagator is built from the Kupradze
representation

    G_ij = (1/(rho w^2)) [ delta_ij kS^2 g_S  +  d_i d_j (g_S - g_P) ],
    g(r) = exp(i kappa r) / (4 pi r),

and the 9-component basis needs two MORE derivatives than that: the blocks
[[G, C], [H, S]] require G_ij (2 derivatives of the scalars), G_ij,k (3) and
G_ij,kl (4). `resonance_tmatrix.elastodynamic_greens_deriv` already does this for
a POINT source, by hand, as radial phi/psi functions with 3 and 7 tensor
structures. That hand form cannot be reused for a LATTICE SUM: a Bloch sum
sum_R g(r - R) e^{i k.R} is not a function of |r| alone, so there is no radial
decomposition to differentiate.

WHAT IS DIFFERENT HERE. This module keeps the derivative order as data rather
than as hand-derived tensor structures:

    d_{i1..in} f(r) = sum_{m=0}^{n/2}  f_{n-m}(r) * S_{n,m}(x),
    f_k = (1/r d/dr)^k f,

where S_{n,m} is the sum over all distinct ways of pairing 2m of the n indices
into Kronecker deltas and assigning the remaining n-2m to the CARTESIAN vector
x_i (not the unit vector). S_{n,m} is generated combinatorially, so the fourth-
order structures are enumerated rather than transcribed -- which is the step the
project's record flags as unforgiving.

The radial ladder f_k for the outgoing Helmholtz kernel is closed form. By
Rayleigh's formula (1/x d/dx)^n h_0(x) = (-1)^n h_n(x)/x^n, so with x = kappa r

    f_k = (i kappa^{k+1} / (4 pi)) (-1)^k h_k(kappa r) / r^k,

and h_k is evaluated from its terminating series so that COMPLEX kappa works --
attenuative media are the physical case here, and scipy's spherical_jn/yn are
real-argument only.

WHAT IT DELIBERATELY DOES NOT DO. Nothing here is lattice-summed. The derivative
ladder is the reusable piece; substituting a lattice-summed f_k (from the Ewald
split in `planar_ewald`) is the next step and belongs in its own module. Keeping
them apart is what lets the free-space case be checked against the validated
`_propagator_block_9x9` first -- see `scripts/gate_kupradze_derivatives.py`.

Conventions inherited: time e^{-i w t}, outgoing h^(1); derivatives are with
respect to the separation vector r_vec exactly as `elastodynamic_greens_deriv`
takes them, so the output feeds `_voigt_contract` unchanged.
"""

import math
from functools import cache, reduce

import numpy as np
from numpy.typing import NDArray

from .effective_contrasts import ReferenceMedium
from .resonance_tmatrix import _voigt_contract

MAX_ORDER = 4


def _pairings(items: list[int], m: int) -> list[tuple[list[tuple[int, int]], list[int]]]:
    """All ways to choose *m* disjoint unordered pairs from *items*.

    Returns a list of (pairs, leftover). Pairs are unordered and the collection
    of pairs is unordered, so each distinct tensor structure appears exactly
    once -- the count is n! / (2^m m! (n-2m)!), which is the standard
    multiplicity of the delta-delta-x-x structures.
    """
    if m == 0:
        return [([], list(items))]
    out: list[tuple[list[tuple[int, int]], list[int]]] = []
    first = items[0]
    rest = items[1:]
    for partner in rest:
        remaining = [i for i in rest if i != partner]
        for sub_pairs, leftover in _pairings(remaining, m - 1):
            out.append(([(first, partner), *sub_pairs], leftover))
    # The case where `first` is NOT in any pair: it becomes a leftover index.
    if len(rest) >= 2 * m:
        for sub_pairs, leftover in _pairings(rest, m):
            out.append((sub_pairs, [first, *leftover]))
    return out


@cache
def _pairing_positions(n: int, m: int) -> tuple[tuple[int, ...], ...]:
    """Axis destinations for each pairing of S_{n,m}, cached.

    The enumeration is combinatorial and depends only on (n, m), while the
    lattice sum calls it once per term per order -- tens of thousands of times
    for one kernel. Caching the index bookkeeping keeps the sum practical
    without changing what is computed.
    """
    return tuple(
        tuple([a for pair in pairs for a in pair] + singles)
        for pairs, singles in _pairings(list(range(n)), m)
    )


def _delta_x_structure(n: int, m: int, x: NDArray) -> NDArray:
    """S_{n,m}: sum over distinct pairings of m deltas and n-2m copies of x.

    Built by outer-producting the factors in a canonical axis order and then
    moving each axis to the index position it belongs to, so no index algebra is
    written out by hand.
    """
    delta = np.eye(3)
    factors = [delta] * m + [x] * (n - 2 * m)
    core = reduce(np.multiply.outer, factors)
    axes = list(range(n))
    total = np.zeros((3,) * n, dtype=x.dtype)
    # Canonical axis j of `core` belongs at output index `positions[j]`.
    for positions in _pairing_positions(n, m):
        total = total + np.moveaxis(core, axes, list(positions))
    return total


def _hankel1(n: int, x: complex) -> complex:
    """Spherical Hankel h_n^(1) from its terminating series -- complex x allowed.

    h_n(x) = (-i)^{n+1} (e^{ix}/x) sum_{s=0}^{n} (i^s / (s! (2x)^s)) (n+s)!/(n-s)!

    scipy's spherical_jn/spherical_yn take real arguments only, and an
    attenuative medium (complex omega, hence complex kappa) is the physical case
    this project runs in, so the series is used rather than a special-function
    call.
    """
    acc = 0.0 + 0.0j
    for s in range(n + 1):
        acc += (1j**s / (math.factorial(s) * (2.0 * x) ** s)) * (
            math.factorial(n + s) / math.factorial(n - s)
        )
    return complex((-1j) ** (n + 1) * (np.exp(1j * x) / x) * acc)


def radial_ladder(r: float, kappa: complex, order: int = MAX_ORDER) -> list[complex]:
    """f_k = (1/r d/dr)^k [exp(i kappa r)/(4 pi r)] for k = 0..order.

    Args:
        r: Distance, > 0.
        kappa: Wavenumber (may be complex for an attenuative medium).
        order: Highest k required. 4 for the strain-strain block.

    Returns:
        [f_0, ..., f_order].
    """
    x = kappa * r
    return [
        complex(1j * kappa ** (k + 1) / (4.0 * np.pi) * (-1) ** k * _hankel1(k, x) / r**k)
        for k in range(order + 1)
    ]


def scalar_derivative_tensors(
    r_vec: NDArray,
    kappa: complex,
    order: int = MAX_ORDER,
) -> list[NDArray]:
    """Cartesian derivative tensors d_{i1..in} g for n = 0..order.

    Args:
        r_vec: Separation vector (3,). Must be non-zero.
        kappa: Scalar wavenumber.
        order: Highest derivative order.

    Returns:
        A list whose n-th entry has shape (3,)*n, complex.
    """
    r_vec = np.asarray(r_vec, dtype=float)
    r = float(np.linalg.norm(r_vec))
    if r < 1.0e-14:
        raise ValueError(
            "scalar_derivative_tensors: the separation is zero, where the kernel is "
            "singular. Self-interaction belongs in the local T-matrix, and a lattice "
            "sum must exclude R = 0 explicitly (see planar_ewald.ewald_total)."
        )
    ladder = radial_ladder(r, kappa, order)
    x = r_vec.astype(complex)

    out: list[NDArray] = [np.asarray(ladder[0], dtype=complex)]
    for n in range(1, order + 1):
        acc = np.zeros((3,) * n, dtype=complex)
        for m in range(n // 2 + 1):
            acc = acc + ladder[n - m] * _delta_x_structure(n, m, x)
        out.append(acc)
    return out


def greens_from_scalars(
    d_p: list[NDArray],
    d_s: list[NDArray],
    omega: complex,
    ref: ReferenceMedium,
) -> tuple[NDArray, NDArray, NDArray]:
    """Assemble (G, Gd, Gdd) from scalar derivative tensors via Kupradze.

    The SAME two-index Kupradze operator is applied at three derivative levels,
    which is the whole content of the 9x9: differentiating the representation

        G_ij = (1/(rho w^2)) [ delta_ij kS^2 g_S + d_i d_j (g_S - g_P) ]

    with respect to r_k and r_l just raises the order of the scalar tensors,
    because the operator's coefficients are constants.

    Args:
        d_p: Derivative tensors of g_P, orders 0..4 (from the same routine that
            produced d_s, so the two share whatever summation was applied).
        d_s: Derivative tensors of g_S.
        omega: Angular frequency.
        ref: Background medium.

    Returns:
        (G, Gd, Gdd) with shapes (3,3), (3,3,3), (3,3,3,3) -- exactly the triple
        `resonance_tmatrix._voigt_contract` consumes.
    """
    k_s2 = (omega / ref.beta) ** 2
    pref = 1.0 / (ref.rho * omega**2)
    delta = np.eye(3)

    diff2 = d_s[2] - d_p[2]
    diff3 = d_s[3] - d_p[3]
    diff4 = d_s[4] - d_p[4]

    G = pref * (k_s2 * d_s[0] * delta + diff2)
    Gd = pref * (k_s2 * np.einsum("ij,k->ijk", delta, d_s[1]) + diff3)
    Gdd = pref * (k_s2 * np.einsum("ij,kl->ijkl", delta, d_s[2]) + diff4)
    return G, Gd, Gdd


def propagator_block_9x9_kupradze(
    r_vec: NDArray,
    omega: complex,
    ref: ReferenceMedium,
) -> NDArray:
    """Free-space 9x9 [[G, C], [H, S]] built through the derivative ladder.

    Numerically identical to `resonance_tmatrix._propagator_block_9x9`; it exists
    so that the derivative ladder -- the piece the lattice sum will reuse -- can
    be checked against the validated hand-derived path before any lattice
    summation is introduced.
    """
    d_p = scalar_derivative_tensors(r_vec, omega / ref.alpha)
    d_s = scalar_derivative_tensors(r_vec, omega / ref.beta)
    G, Gd, Gdd = greens_from_scalars(d_p, d_s, omega, ref)
    C, H, S = _voigt_contract(Gd, Gdd)
    P = np.zeros((9, 9), dtype=complex)
    P[:3, :3] = G
    P[:3, 3:] = C
    P[3:, :3] = H
    P[3:, 3:] = S
    return P


__all__ = [
    "MAX_ORDER",
    "greens_from_scalars",
    "propagator_block_9x9_kupradze",
    "radial_ladder",
    "scalar_derivative_tensors",
]
