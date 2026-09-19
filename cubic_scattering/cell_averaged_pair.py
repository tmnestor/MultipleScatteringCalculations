"""The single (sinc^1) cell-averaged propagator for a NON-PERIODIC voxel set.

WHY THIS FILE EXISTS
--------------------
``cell_averaged_lattice`` solves the same problem for a Bloch lattice, where the
``dz = 0`` sum diverges and Ewald surgery is unavoidable.  An isolated
scatterer -- a voxelised sphere, say -- has no lattice sum at all, so none of
that machinery applies and none of it is needed.  What is needed is the same
SINGLE average, pairwise.

THE PAIRING, WHICH IS THE WHOLE POINT
-------------------------------------
``scripts/settle_single_site_formulation.py`` settles that the single-site
closure here is COLLOCATION: T9 closes on the cube-centre value via a SINGLE
volume integral, T27 on the volume average via a DOUBLE one.  A T-matrix and a
propagator must close on the same quantity.  The sphere route has been pairing a
single-average T9 with an UNAVERAGED point propagator -- not the double average,
but no average at all -- which is an unmatched pair of exactly the kind that
made the shear correction K look like physics until it measured 1.0000 once the
propagator defects were removed (see the K-is-dead and contact-fix records).

This module supplies the missing half: the propagator averaged over the
RECEIVER cell only, one power of the form factor, which is what a collocation
source requires.

WHY THE SINGLE AVERAGE IS EASY WHERE THE DOUBLE ONE IS NOT
----------------------------------------------------------
With a point source and a cubic receiver cell of half-width ``h = d/2``, the
integrand is regular at every non-zero lattice separation: the closest approach
of the cell to the source is the near face centre, so ``|s - u| >= d/2``.  The
double average has no such protection -- its tent reaches ``r = 0`` whenever the
cells touch, which is the face-contact quadrature bias that has already cost
this project one wrong turn.  So here a product Gauss rule is not a compromise;
it is accurate everywhere, and ``n_gauss`` independence is checkable.

TWO ROUTES, ON PURPOSE
----------------------
``averaged_pair_block_9x9`` integrates directly.  ``tail_pair_block_9x9``
applies the closed-form ``d^2`` tail instead, using

    <g> - g = (d^2/24) grad^2 g + O(d^4),   grad^2 g_c = -kappa_c^2 g_c,

so that ``<D> = (1 - kappa_c^2 d^2 / 24) D + O(d^4)`` exactly, per mode, away
from the origin.  The two must agree to ``O(d^4)`` and diverge from each other
near contact, where the expansion stops being good.  That disagreement is a
measurement, not a defect, and the gate reports where it sets in.

THE MODES MUST NOT BE MIXED BEFORE AVERAGING.  The tail factor carries
``kappa^2`` and P and S do not share it, so both routes average the scalar
derivative tensors per mode and assemble only afterwards -- the same discipline
``cell_averaged_lattice.averaged_same_plane_9x9`` follows, for the same reason.

Conventions inherited: (z, x, y) ordering, time ``e^{-i omega t}``, outgoing
``h^(1)``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .cell_averaged_lattice import _cell_nodes
from .effective_contrasts import ReferenceMedium
from .kupradze_derivatives import (
    MAX_ORDER,
    greens_from_scalars,
    scalar_derivative_tensors,
)

__all__ = [
    "auto_n_gauss",
    "averaged_pair_block_9x9",
    "averaged_pair_scalar_tensors",
    "tail_pair_block_9x9",
]


def auto_n_gauss(s_vec: NDArray, d: float) -> int:
    """Gauss order chosen from how close the receiver cell comes to the source.

    The integrand is regular everywhere, but it is PEAKED near the face of the
    cell closest to the source, and a Gauss rule converges slowly against a
    nearby peak.  How close that face comes is therefore the difficulty, and it
    is a property of the separation alone::

        q = |clamp(|s| - h, 0)| / d

    which is 0.5 for a face neighbour, 0.707 for an edge one and 0.866 for a
    corner one.  Measured convergence to a 24-point reference, at d = 0.25,
    omega = 300:

        face   (q=0.50):  4 -> 1.8e-2   8 -> 2.9e-5   12 -> 3.6e-8   16 -> 1.4e-10
        edge   (q=0.71):  4 -> 6.5e-4   8 -> 9.0e-8   12 -> 1.6e-10
        corner (q=0.87):  4 -> 6.1e-5   8 -> 2.3e-10

    The thresholds below are read off that table, not guessed.  Only the six
    face and twelve edge neighbours are expensive, so the adaptive rule costs
    almost nothing: every other separation is smooth and takes the base order.

    Args:
        s_vec: Separation, shape (3,).
        d: Cell SIDE.

    Returns:
        Gauss points per axis.
    """
    h = 0.5 * d
    closest = np.maximum(np.abs(np.asarray(s_vec, dtype=float)) - h, 0.0)
    q = float(np.linalg.norm(closest)) / d
    if q < 0.6:
        return 16
    if q < 0.8:
        return 12
    if q < 1.5:
        return 8
    return 6


def averaged_pair_scalar_tensors(
    s_vec: NDArray,
    kappa: complex,
    d: float,
    *,
    n_gauss: int | None = None,
    order: int = MAX_ORDER,
) -> list[NDArray]:
    """Scalar derivative tensors averaged over the receiver cell.

    Args:
        s_vec: Separation from source to receiver centre, shape (3,), non-zero.
        kappa: Scalar wavenumber for ONE mode -- P and S must not be mixed
            before this point.
        d: Cell SIDE (not the half-width).
        n_gauss: Gauss points per axis.  ``None`` picks it from the separation
            via ``auto_n_gauss``, which is what the measured convergence above
            requires; an explicit value overrides that and is what the gate
            uses to demonstrate convergence.
        order: Highest derivative order.

    Returns:
        A list whose n-th entry has shape (3,)*n, complex.

    Raises:
        ValueError: If the cell reaches the source, where the single average
            would be singular.
    """
    s = np.asarray(s_vec, dtype=float)
    h = 0.5 * d
    # The receiver cell must not reach the source. The closest approach of a
    # cube of half-width h centred at s is |s| measured face-wise, so the
    # condition is that no axis is within h of the origin simultaneously.
    if float(np.max(np.abs(s))) <= h:
        msg = (
            f"the receiver cell reaches the source: max|s| = {float(np.max(np.abs(s))):.6g} "
            f"<= h = {h:.6g}.\n"
            "  Where: cubic_scattering/cell_averaged_pair.py,\n"
            "         averaged_pair_scalar_tensors()\n"
            "  Valid: a separation of at least one cell pitch. The self term is\n"
            "         the local T-matrix's business and is excluded here.\n"
            "  Fix:   skip the m == n pair, as the Foldy-Lax assembly does."
        )
        raise ValueError(msg)

    ng = auto_n_gauss(s, d) if n_gauss is None else n_gauss
    nodes, wts = _cell_nodes(h, ng)
    acc: list[NDArray] = [np.zeros((3,) * n, dtype=complex) for n in range(order + 1)]
    for uz, wz in zip(nodes, wts, strict=True):
        for ux, wx in zip(nodes, wts, strict=True):
            for uy, wy in zip(nodes, wts, strict=True):
                w = wz * wx * wy
                shifted = s - np.array([uz, ux, uy])
                for n, t in enumerate(scalar_derivative_tensors(shifted, kappa, order)):
                    acc[n] = acc[n] + w * t
    return acc


def _assemble(d_p: list[NDArray], d_s: list[NDArray], omega: complex, ref: ReferenceMedium) -> NDArray:
    """Assemble the 9x9 [[G, C], [H, S]] from per-mode scalar tensors.

    Args:
        d_p: P-mode scalar derivative tensors.
        d_s: S-mode scalar derivative tensors.
        omega: Angular frequency.
        ref: Background medium.

    Returns:
        Shape (9, 9), complex.
    """
    from .resonance_tmatrix import _voigt_contract

    g, gd, gdd = greens_from_scalars(d_p, d_s, omega, ref)
    c, h_blk, s_blk = _voigt_contract(gd, gdd)
    out = np.zeros((9, 9), dtype=complex)
    out[:3, :3] = g
    out[:3, 3:] = c
    out[3:, :3] = h_blk
    out[3:, 3:] = s_blk
    return out


#: Memoised blocks, keyed on the LATTICE OFFSET rather than on the float
#: separation.  The average costs a cubed Gauss rule per mode, some six times
#: the point propagator, and a solver rebuilds the same handful of distinct
#: separations on every call.  Within one assembly a local cache suffices; this
#: one spans calls, which is what a test suite or a frequency sweep needs.
#:
#: Measured, so the limit is not left to inference.  Repeating one solve at the
#: SAME (omega, pitch) costs 55.0 s then 1.1 s, with zero new entries -- the
#: reuse within a configuration is captured in full.  A different omega
#: repopulates all 870 blocks and pays again, because those are genuinely
#: different propagators.  So the sphere tests run in 21 min against 32 without
#: the cache, and NOT the 5 min they took before the cell average became the
#: default: the residual is the average itself, some 63 ms per block for a cubed
#: Gauss rule over two modes.  Reducing it further means using
#: ``tail_pair_block_9x9`` beyond the shells where the two routes part company,
#: which is a separate change and is not made here.
_BLOCK_CACHE: dict[tuple, NDArray] = {}

#: A bound, so a long sweep over many pitches or frequencies cannot grow it
#: without limit.  Each entry is 9x9 complex, about 1.3 kB.
_CACHE_MAX = 200_000


def clear_block_cache() -> None:
    """Empty the memoised propagator blocks."""
    _BLOCK_CACHE.clear()


def averaged_pair_block_9x9(
    s_vec: NDArray,
    omega: float,
    ref: ReferenceMedium,
    d: float,
    *,
    n_gauss: int | None = None,
) -> NDArray:
    """The 9x9 propagator with the receiver cell averaged, by direct quadrature.

    Memoised across calls on the lattice offset.  The key is checked rather
    than assumed: a separation that is not a lattice multiple of ``d`` is
    computed directly and not cached, because rounding it to an offset would
    silently return the propagator for a different separation -- and every
    symmetry and reciprocity test would still pass.

    Args:
        s_vec: Separation between cell centres, shape (3,), non-zero.
        omega: Angular frequency.
        ref: Background medium.
        d: Cell SIDE.
        n_gauss: Gauss points per axis; ``None`` selects it from the separation.

    Returns:
        Shape (9, 9), complex.  A copy, so a caller may write into it without
        corrupting the cache.
    """
    s = np.asarray(s_vec, dtype=float)
    offset = (
        int(round(float(s[0]) / d)),
        int(round(float(s[1]) / d)),
        int(round(float(s[2]) / d)),
    )
    on_lattice = float(np.max(np.abs(s - d * np.array(offset)))) <= 1.0e-9 * d

    if on_lattice:
        key = (offset, float(d), complex(omega), ref.alpha, ref.beta, ref.rho, n_gauss)
        hit = _BLOCK_CACHE.get(key)
        if hit is not None:
            return hit.copy()

    d_p = averaged_pair_scalar_tensors(s, omega / ref.alpha, d, n_gauss=n_gauss)
    d_s = averaged_pair_scalar_tensors(s, omega / ref.beta, d, n_gauss=n_gauss)
    block = _assemble(d_p, d_s, omega, ref)

    if on_lattice and len(_BLOCK_CACHE) < _CACHE_MAX:
        _BLOCK_CACHE[key] = block
    return block.copy()


def tail_pair_block_9x9(
    s_vec: NDArray,
    omega: float,
    ref: ReferenceMedium,
    d: float,
) -> NDArray:
    """The same object through the closed-form d^2 tail, for cross-checking.

    Exact through ``d^2`` away from the origin, and cheap: one scalar factor per
    mode rather than a cubed Gauss rule.  It is NOT a substitute near contact,
    where the expansion's neglected fourth moments matter; the gate measures
    where the two part company.

    Args:
        s_vec: Separation between cell centres, shape (3,), non-zero.
        omega: Angular frequency.
        ref: Background medium.
        d: Cell SIDE.

    Returns:
        Shape (9, 9), complex.
    """
    kp, ks = omega / ref.alpha, omega / ref.beta
    fp = 1.0 - (kp**2) * (d**2) / 24.0
    fs = 1.0 - (ks**2) * (d**2) / 24.0
    d_p = [fp * t for t in scalar_derivative_tensors(s_vec, kp)]
    d_s = [fs * t for t in scalar_derivative_tensors(s_vec, ks)]
    return _assemble(d_p, d_s, omega, ref)
