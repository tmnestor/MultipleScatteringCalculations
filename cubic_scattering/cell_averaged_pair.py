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

THE MODES MUST NOT BE MIXED BEFORE THE TAIL FACTOR.  The tail factor carries
``kappa^2`` and P and S do not share it, so the tail route scales each mode by
its own factor before combining them.  The mode DIFFERENCE D = g_S - g_P is
nevertheless formed as one function (``kupradze_derivatives.
difference_derivative_tensors``), never by subtracting two averaged tensors:
the subtraction cancels as eps / (kappa r)^2 at the nearest quadrature nodes.
The direct quadrature is linear, so it averages D itself; the tail route writes
fs g_S - fp g_P = fs D + (fs - fp) g_P, with fs - fp an explicit small factor.

Conventions inherited: (z, x, y) ordering, time ``e^{-i omega t}``, outgoing
``h^(1)``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml
from numpy.typing import NDArray

from .cell_averaged_lattice import _cell_nodes
from .effective_contrasts import ReferenceMedium
from .kupradze_derivatives import (
    MAX_ORDER,
    difference_derivative_tensors,
    greens_from_difference,
    scalar_derivative_tensors,
)

__all__ = [
    "auto_n_gauss",
    "averaged_pair_block_9x9",
    "averaged_pair_blocks_9x9",
    "averaged_pair_difference_tensors",
    "averaged_pair_scalar_tensors",
    "CellAverageConfig",
    "cell_average_backend",
    "cell_average_config",
    "tail_pair_block_9x9",
]

#: The numerics configuration: how the receiver-cell average is evaluated.
NUMERICS_YAML = Path(__file__).resolve().parent / "numerics.yml"
_BACKENDS = ("python", "fortran")
_ROUTES = ("kseries", "quadrature")
_EXAMPLE = (
    "cell_average:\n"
    "  backend: fortran      # the quadrature; one of: python, fortran\n"
    "  route: kseries        # one of: kseries, quadrature\n"
    "  kseries_tol: 1.0e-15  # a positive number\n"
    "  kseries_max_reach: 3.0  # a positive number: the largest |k_S| r_max the series takes\n"
    "  kseries_store: ~/.cache/cubic_scattering/receiver_kseries.npy\n"
    "  far_expansion: true   # true or false\n"
    "  far_tol: 1.0e-15      # a positive number\n"
    "  far_max_order: 24     # an integer of at least 2\n"
    "  fallback_n_gauss: 24  # an integer of at least 4"
)


@dataclass(frozen=True)
class CellAverageConfig:
    """The ``cell_average`` block of ``numerics.yml``; every key is required.

    Attributes:
        backend: The quadrature's implementation, 'python' or 'fortran'.
        route: 'kseries' (the frequency-amortised series, with the quadrature for the offsets it cannot
            reach) or 'quadrature' (the quadrature for every offset).
        kseries_tol: The series' truncation tolerance.
        kseries_max_reach: The largest |k_S| r_max the series takes. Its error grows like e^(k r) as it
            cancels (measured 7.7e-14 at 2 to 3, 1.4e-11 at 8 to 9), so offsets beyond go to the
            expansion about the cell centre or the quadrature.
        kseries_store: The file holding the series' universal coefficients (``~`` allowed).
        far_expansion: With route 'kseries', whether the offsets the series cannot reach are taken by the
            expansion about the cell centre (``cell_average_far``) before the quadrature.
        far_tol: The expansion's truncation tolerance.
        far_max_order: The highest truncation order of the expansion; offsets needing more go to the
            quadrature.
        fallback_n_gauss: With route 'kseries', the Gauss order of the quadrature for the few offsets
            neither the series nor the expansion takes (close cells at high frequency). A converged rule,
            not the automatic orders, which there give only about 1e-9.
    """

    backend: str
    route: str
    kseries_tol: float
    kseries_max_reach: float
    kseries_store: str
    far_expansion: bool
    far_tol: float
    far_max_order: int
    fallback_n_gauss: int


def _config_error(key: str, problem: str) -> ValueError:
    return ValueError(
        f"cell_average.{key} {problem}\n"
        f"  Where: {NUMERICS_YAML}, key cell_average.{key}\n"
        f"  Valid:\n{_EXAMPLE}\n"
        "  Fix:   set the key as in the block above, at the top level of that file (backend fortran needs\n"
        "         conda run -n seismic python -m cubic_scattering.fortran.build)."
    )


@lru_cache(maxsize=1)
def cell_average_config() -> CellAverageConfig:
    """The ``cell_average`` block of ``cubic_scattering/numerics.yml``, validated.

    Read once per process. There is no default: a missing file, key or invalid value is an error.

    Raises:
        ValueError: with what is wrong, where to fix it, a valid example, and how to recover.
    """
    if not NUMERICS_YAML.is_file():
        raise ValueError(
            "the numerics configuration file is missing.\n"
            f"  Where: {NUMERICS_YAML}\n"
            f"  Valid:\n{_EXAMPLE}\n"
            "  Fix:   create that file with the block above."
        )
    cfg = yaml.safe_load(NUMERICS_YAML.read_text()) or {}
    section = cfg.get("cell_average") if isinstance(cfg, dict) else None
    section = section if isinstance(section, dict) else {}

    def found(key: str) -> str:
        return "is missing" if section.get(key) is None else f"is {section.get(key)!r}"

    def positive(key: str) -> float:
        value = section.get(key)
        if not isinstance(value, bool) and isinstance(value, int | float) and value > 0:
            return float(value)
        hint = ""
        if isinstance(value, str):
            try:
                float(value)
                hint = (
                    f" It was read as TEXT: YAML 1.1 (PyYAML) needs a decimal point in a float, so write"
                    f" {float(value):.1e} rather than {value}."
                )
            except ValueError:
                pass
        raise _config_error(key, f"{found(key)}; it must be a positive number.{hint}")

    backend = section.get("backend")
    if backend not in _BACKENDS:
        raise _config_error("backend", f"{found('backend')}; it must be one of {', '.join(_BACKENDS)}.")
    route = section.get("route")
    if route not in _ROUTES:
        raise _config_error("route", f"{found('route')}; it must be one of {', '.join(_ROUTES)}.")
    tol = positive("kseries_tol")
    reach = positive("kseries_max_reach")
    store = section.get("kseries_store")
    if not isinstance(store, str) or not store:
        raise _config_error("kseries_store", f"{found('kseries_store')}; it must be a file path.")
    far = section.get("far_expansion")
    if not isinstance(far, bool):
        raise _config_error("far_expansion", f"{found('far_expansion')}; it must be true or false.")
    far_tol = positive("far_tol")
    far_order = section.get("far_max_order")
    if isinstance(far_order, bool) or not isinstance(far_order, int) or far_order < 2:
        raise _config_error(
            "far_max_order", f"{found('far_max_order')}; it must be an integer of at least 2."
        )
    fallback = section.get("fallback_n_gauss")
    if isinstance(fallback, bool) or not isinstance(fallback, int) or fallback < 4:
        raise _config_error(
            "fallback_n_gauss", f"{found('fallback_n_gauss')}; it must be an integer of at least 4."
        )
    return CellAverageConfig(str(backend), str(route), tol, reach, store, far, far_tol, far_order, fallback)


def cell_average_backend() -> str:
    """``cell_average.backend`` from ``numerics.yml``: the quadrature's implementation."""
    return cell_average_config().backend


cell_average_backend.cache_clear = cell_average_config.cache_clear  # type: ignore[attr-defined]


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

    ⚠ ACCURACY: THESE ORDERS GIVE ABOUT 1e-8, NOT ROUND-OFF.  The table above
    calibrates the nearest shells only.  Measured on 21 October 2026, the whole
    9 x 9 block at each order against a 32-point rule, worst over |k_S| d = 1e-3,
    0.3, 1, 2 and damping 0 and 0.05 (d = 0.25):

        offset     (1,0,0) (1,1,0) (1,1,1) (2,0,0) (2,1,0) (2,2,1) (3,2,1) (5,3,2)
        n_gauss       16      12       8       6       6       6       6       6
        rel error  4.1e-11 1.2e-11 1.6e-10 1.9e-8  4.3e-9  5.8e-11 4.1e-12 4.0e-13

    The SECOND shell is the weak one: (2, 0, 0) sits exactly at q = 1.5, the
    threshold, so it falls to the base order, while its integrand is still
    sharply peaked.  The default route (``cell_average.route: kseries``: the
    wavenumber series within its reach, the expansion about the cell centre
    beyond, a 24-point rule for what neither takes) agrees with a 32-point
    rule to about 1.2e-13 at every offset, from k_S R = 1 to 12 at n_sub = 25,
    so it is the accurate one.  The
    compiled quadrature (``cell_average_fortran``) reproduces THIS function's
    orders, and its tests compare it with the Python quadrature at the same
    orders: they certify the transcription, not the accuracy.

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
    _check_receiver_clear(s, d)
    return _receiver_average(s, d, n_gauss, order, lambda r: scalar_derivative_tensors(r, kappa, order))


def _check_receiver_clear(s: NDArray, d: float) -> None:
    h = 0.5 * d
    # The receiver cell must not reach the source. The closest approach of a
    # cube of half-width h centred at s is |s| measured face-wise, so the
    # condition is that no axis is within h of the origin simultaneously.
    if float(np.max(np.abs(s))) <= h:
        msg = (
            f"the receiver cell reaches the source: max|s| = {float(np.max(np.abs(s))):.6g} "
            f"<= h = {h:.6g}.\n"
            "  Where: cubic_scattering/cell_averaged_pair.py,\n"
            "         averaged_pair_scalar_tensors() / averaged_pair_difference_tensors()\n"
            "  Valid: a separation of at least one cell pitch. The self term is\n"
            "         the local T-matrix's business and is excluded here.\n"
            "  Fix:   skip the m == n pair, as the Foldy-Lax assembly does."
        )
        raise ValueError(msg)


def averaged_pair_difference_tensors(
    s_vec: NDArray,
    kappa_p: complex,
    kappa_s: complex,
    d: float,
    *,
    n_gauss: int | None = None,
    order: int = MAX_ORDER,
) -> list[NDArray]:
    """Derivative tensors of D = g_S - g_P averaged over the receiver cell.

    The average is linear, so averaging D equals differencing the per-mode averages, exactly; formed
    this way it does not cancel between the modes (``kupradze_derivatives.difference_radial_ladder``),
    which the per-mode route does as eps / (kappa r)^2 at the nearest quadrature nodes.

    Args and Raises: as ``averaged_pair_scalar_tensors``, with the two wavenumbers.
    """
    s = np.asarray(s_vec, dtype=float)
    _check_receiver_clear(s, d)
    return _receiver_average(
        s, d, n_gauss, order, lambda r: difference_derivative_tensors(r, kappa_p, kappa_s, order)
    )


def _receiver_average(
    s: NDArray,
    d: float,
    n_gauss: int | None,
    order: int,
    tensors_at: Callable[[NDArray], list[NDArray]],
) -> list[NDArray]:
    ng = auto_n_gauss(s, d) if n_gauss is None else n_gauss
    nodes, wts = _cell_nodes(0.5 * d, ng)
    acc: list[NDArray] = [np.zeros((3,) * n, dtype=complex) for n in range(order + 1)]
    for uz, wz in zip(nodes, wts, strict=True):
        for ux, wx in zip(nodes, wts, strict=True):
            for uy, wy in zip(nodes, wts, strict=True):
                w = wz * wx * wy
                shifted = s - np.array([uz, ux, uy])
                for n, t in enumerate(tensors_at(shifted)):
                    acc[n] = acc[n] + w * t
    return acc


def _assemble(d_diff: list[NDArray], d_s: list[NDArray], omega: complex, ref: ReferenceMedium) -> NDArray:
    """Assemble the 9x9 [[G, C], [H, S]] from the tensors of D = g_S - g_P and of g_S.

    Args:
        d_diff: Derivative tensors of D.
        d_s: S-mode scalar derivative tensors.
        omega: Angular frequency.
        ref: Background medium.

    Returns:
        Shape (9, 9), complex.
    """
    from .resonance_tmatrix import _voigt_contract

    g, gd, gdd = greens_from_difference(d_diff, d_s, omega, ref)
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
    omega: complex,
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
    key = _lattice_key(s, omega, ref, d, n_gauss)
    if key is not None:
        hit = _BLOCK_CACHE.get(key)
        if hit is not None:
            return hit.copy()

    d_diff = averaged_pair_difference_tensors(s, omega / ref.alpha, omega / ref.beta, d, n_gauss=n_gauss)
    d_s = averaged_pair_scalar_tensors(s, omega / ref.beta, d, n_gauss=n_gauss)
    block = _assemble(d_diff, d_s, omega, ref)

    if key is not None and len(_BLOCK_CACHE) < _CACHE_MAX:
        _BLOCK_CACHE[key] = block
    return block.copy()


def _lattice_key(
    s: NDArray, omega: complex, ref: ReferenceMedium, d: float, n_gauss: int | None
) -> tuple | None:
    """The memo key of an on-lattice separation, or None for one that is not a lattice multiple of d."""
    offset = (
        int(round(float(s[0]) / d)),
        int(round(float(s[1]) / d)),
        int(round(float(s[2]) / d)),
    )
    if float(np.max(np.abs(s - d * np.array(offset)))) > 1.0e-9 * d:
        return None
    return (offset, float(d), complex(omega), ref.alpha, ref.beta, ref.rho, n_gauss)


def averaged_pair_blocks_9x9(
    S: NDArray,
    omega: complex,
    ref: ReferenceMedium,
    d: float,
    *,
    n_gauss: int | None = None,
) -> NDArray:
    """The receiver-cell averaged block at every separation of S (N, 3), shape (N, 9, 9).

    The route is ``cell_average.route`` in ``cubic_scattering/numerics.yml``:

    * ``kseries``: the frequency-amortised wavenumber series (``cell_average_kseries``) for every lattice
      offset within ``kseries_max_reach``; then, with ``far_expansion: true``, the expansion about the
      cell centre (``cell_average_far``) for the distant offsets beyond it; then the quadrature at the
      converged order ``fallback_n_gauss`` for whatever is left (close cells at high frequency,
      separations off the lattice). About 1.2e-13 at every offset, measured from k_S R = 1 to 12.
    * ``quadrature``: the quadrature everywhere, at the automatic Gauss orders (about 1e-8 at worst: see
      ``auto_n_gauss``).

    An explicit ``n_gauss`` always means the quadrature at that order: it is how the convergence gates ask
    for a particular rule. The quadrature itself is ``cell_average.backend``: the per-separation Python
    rule above, or its compiled transcription (``cell_average_fortran``), equal to round-off at the same
    orders.

    Raises:
        ValueError: If a receiver cell reaches its source, or if the configuration is invalid.
        ImportError: If the configuration asks for compiled code that has not been built.
    """
    S = np.atleast_2d(np.asarray(S, dtype=float))
    cfg = cell_average_config()
    if cfg.route == "kseries" and n_gauss is None:
        from .cell_average_kseries import averaged_pair_blocks_9x9_kseries  # local: it imports this module

        blocks, covered = averaged_pair_blocks_9x9_kseries(
            S, omega, ref, d, tol=cfg.kseries_tol, max_reach=cfg.kseries_max_reach
        )
        if cfg.far_expansion and not covered.all():
            from .cell_average_far import averaged_pair_blocks_9x9_far

            rest = np.flatnonzero(~covered)
            far, far_covered = averaged_pair_blocks_9x9_far(
                S[rest], omega, ref, d, tol=cfg.far_tol, max_order=cfg.far_max_order
            )
            blocks[rest[far_covered]] = far[far_covered]
            covered[rest[far_covered]] = True
        if not covered.all():
            blocks[~covered] = _quadrature_blocks(S[~covered], omega, ref, d, cfg.fallback_n_gauss)
        return blocks
    return _quadrature_blocks(S, omega, ref, d, n_gauss)


def _quadrature_blocks(
    S: NDArray, omega: complex, ref: ReferenceMedium, d: float, n_gauss: int | None
) -> NDArray:
    """The blocks by the quadrature of ``cell_average.backend``."""
    if cell_average_backend() == "fortran":
        from .cell_average_fortran import averaged_pair_blocks_9x9_fortran  # local: it imports this module

        return averaged_pair_blocks_9x9_fortran(S, omega, ref, d, n_gauss=n_gauss)
    return np.stack([averaged_pair_block_9x9(s, omega, ref, d, n_gauss=n_gauss) for s in S])


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
    # fs g_S - fp g_P = fs D + (fs - fp) g_P: the mode difference D is formed stably, and fs - fp
    # = -(ks^2 - kp^2) d^2 / 24 is an explicit small factor, so nothing cancels.
    t_p = scalar_derivative_tensors(s_vec, kp)
    t_diff = difference_derivative_tensors(s_vec, kp, ks)
    d_diff = [fs * dd + (fs - fp) * p for dd, p in zip(t_diff, t_p, strict=True)]
    d_s = [fs * t for t in scalar_derivative_tensors(s_vec, ks)]
    return _assemble(d_diff, d_s, omega, ref)
