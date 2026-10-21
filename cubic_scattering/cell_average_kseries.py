"""The receiver-cell averaged pair block as a power series in the wavenumber, amortised across frequency.

WHY THIS EXISTS. Production sweeps many frequencies, and the sphere FFT kernel's averaged blocks are
rebuilt at each one. By quadrature (``cell_average_fortran``) every frequency pays a cubed Gauss rule per
offset. But the frequency enters the average only through powers of the wavenumber:

    g = e^{i kappa r} / r = sum_t (i kappa)^t r^(t - 1) / t!,
    < d^a g >(s) = sum_t (i kappa)^t / t!  d^(t - 1 - |a|)  U(t, a)(s / d),

with U(t, a)(o) = < d^a r^(t - 1) >(o) the average over the UNIT receiver cube at the integer offset o:
numbers that depend on nothing else (not the frequency, the cell size or the medium). They are computed
once (compiled: ``fortran/kseries_moments.f90``), stored on disk, and every frequency after that costs a
weighted sum per offset: about 0.1 to 0.2 ms, against a fresh quadrature. Analytic or tabulated integrals,
where possible, are amortised across frequency; this is that principle applied to the receiver average.

The series is the graded-voxel one: ``graded_voxel.derivatives.moment_table_kseries`` does the summation,
with its own truncation and its refusal when 48 powers cannot reach the tolerance, and is given the stored
coefficients in place of its own (``kseries_coefficients``, which this table equals to round-off). With the
weight W = 0 its table is the cube-averaged Green's tensor and its first and second derivatives, which is
what the 9 x 9 block is assembled from.

ITS ACCURACY DEPENDS ON ITS REACH (measured 21 October 2026). Against a 32-point quadrature, by
|k_S| r_max (r_max the distance from the source to the far side of the cell):

    |k_S| r_max   0-1     1-2     2-3     3-4     4-5     5-6     6-7     7-8     8-9
    worst error   2.8e-14 5.3e-14 7.7e-14 1.8e-13 4.7e-13 9.6e-13 3.5e-12 8.4e-12 1.4e-11

The terms grow to about e^(k r) / sqrt(2 pi k r) before they cancel, so the coefficients' own ~1e-14 is
magnified as the reach grows; the truncation bound of ``moment_table_kseries`` does not see this. So the
series takes only offsets with |k_S| r_max <= ``cell_average.kseries_max_reach`` (3 in production, about
1e-13), and the expansion about the cell centre (``cell_average_far``) takes the distant ones, where it
is accurate. Even at its edge the series is far better than the production quadrature's automatic Gauss
orders (``cell_averaged_pair.auto_n_gauss``), good to about 1e-8 only, worst in the second shell.

WHERE IT DOES NOT APPLY. Beyond ``max_reach``; where 48 powers cannot reach the tolerance (about
|k_S| r_max > 9 at tol 1e-15); and at any separation that is not a lattice offset. Those are returned as
not covered, and ``cell_averaged_pair.averaged_pair_blocks_9x9`` takes them by the expansion about the
cell centre or, failing that, a converged quadrature (``cell_average.fallback_n_gauss``).

THE STORE (a quirk worth knowing). The coefficients live in ONE file outside the repository, named by
``cell_average.kseries_store`` in ``numerics.yml`` (by default under ~/.cache). It is created on first use
and grows by the canonical offsets each run needs, so the first run on a new machine, or the first at a
larger grid, spends about a minute of CPU building it (compiled, OpenMP); later runs read it. It holds
universal constants, so it is never stale for a frequency, cell size or medium; it is invalid only if the
coefficients' layout changes (``LAYOUT`` below), and then loading stops with the instruction to delete it.

It is a single ``.npy`` array, one row per canonical offset (the three integers, then the coefficients
flattened), opened MEMORY-MAPPED: the parallel workers of a frequency sweep share one copy through the
operating system's page cache instead of each loading its own, and a run reads only the rows of the
offsets it uses. One file means one atomic step when it grows (a temporary file, then a rename), so an
interrupted run cannot leave it torn and no index can disagree with its data. Two processes growing it at
the same moment are safe but wasteful: the last rename wins, and rows the other added are recomputed by a
later run.
"""

import math
import os
import tempfile
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from . import cell_averaged_pair as cap
from .cell_average_fortran import cube_image_blocks, cube_maps
from .effective_contrasts import ReferenceMedium
from .gauss_rules import gauss_legendre
from .graded_voxel import derivatives as gd
from .graded_voxel.derivatives_fortran import _term_table
from .graded_voxel.kernel import _assemble as assemble_from_greens

try:
    from .fortran._kseries_moments import kseries_moments_mod as _compiled
except ImportError as err:
    raise ImportError(
        "the compiled k-series moments are not built.\n"
        "  What:  cubic_scattering/fortran/_kseries_moments*.so is missing or cannot be loaded.\n"
        "  Where: built from cubic_scattering/fortran/kseries_moments.f90.\n"
        "  Fix:   conda run -n seismic python -m cubic_scattering.fortran.build\n"
        "         (the seismic environment supplies gfortran, Meson, Ninja and the OpenMP runtime)."
    ) from err

__all__ = [
    "A_LIST",
    "LAYOUT",
    "averaged_pair_blocks_9x9_kseries",
    "canonical_greens",
    "clear_loaded_store",
    "coefficients_fortran",
    "store_path",
]

#: the highest power index t kept (r^(t - 1)), as the graded-voxel series
T_MAX = gd.KSERIES_T_MAX
#: every derivative multi-index the block needs: G and its first two derivatives, two more for the mode
#: difference (the order ``kseries_coefficients`` returns for d_list of order 2)
A_LIST = gd.multi_indices(4)
_D_LIST = list(gd.multi_indices(2))
_W_LIST = [(0, 0, 0)]
_W0 = ((0, 0, 0),)
#: what a stored table must have been built with; a file with any other layout is refused
LAYOUT = f"receiver-kseries 1: t = 0..{T_MAX}, a = multi_indices(4), W = 0, kseries_gauss_order rules"
#: one row of the store: the offset's three integers, then its (T_MAX + 1) x len(A_LIST) coefficients
ROW = 3 + (T_MAX + 1) * len(A_LIST)

#: the store as read in this process: the memory-mapped rows, and the row of each canonical offset
_LOADED: tuple[NDArray | None, dict[tuple[int, int, int], int]] | None = None


def store_path() -> Path:
    """The coefficient store, ``cell_average.kseries_store`` in ``numerics.yml`` (``~`` expanded)."""
    return Path(cap.cell_average_config().kseries_store).expanduser()


def clear_loaded_store() -> None:
    """Forget the table read into this process (the file is untouched)."""
    global _LOADED
    _LOADED = None


def coefficients_fortran(offsets: NDArray) -> NDArray:
    """U(t, a) at integer offsets (N, 3), shape (N, T_MAX + 1, len(A_LIST)), by the compiled rule.

    Each offset takes the Gauss order ``kseries_coefficients`` uses for it, so the two are the same sums.
    """
    offsets = np.atleast_2d(np.asarray(offsets, dtype=int))
    orders_each = np.array(
        [gd.kseries_gauss_order((int(o[0]), int(o[1]), int(o[2])), _W0) for o in offsets]
    )
    orders = sorted(set(orders_each.tolist()))
    x1 = np.zeros((max(orders), len(orders)))
    w1 = np.zeros((max(orders), len(orders)))
    for k, ng in enumerate(orders):
        x, w = gauss_legendre(ng)
        x1[:ng, k], w1[:ng, k] = 0.5 * x, 0.5 * w  # as derivatives.cell_rule(1.0, ng)
    rule = (np.searchsorted(orders, orders_each) + 1).astype(np.int32)
    start, coef, exps, qs = _term_table(4)
    u = _compiled.kseries_moments(
        np.asfortranarray(offsets.T.astype(float)),
        np.asfortranarray(x1),
        np.asfortranarray(w1),
        np.array(orders, dtype=np.int32),
        rule,
        int(T_MAX),
        start,
        coef,
        exps,
        qs,
        int(qs.max()),
        int(exps.max()),
    )
    return np.ascontiguousarray(np.moveaxis(u, -1, 0))


def _layout_error(path: Path, found: str) -> ValueError:
    return ValueError(
        f"the k-series coefficient store has the wrong layout ({found}).\n"
        f"  Where: {path} (cell_average.kseries_store in {cap.NUMERICS_YAML})\n"
        f"  Valid: a 2-D float64 array of rows of {ROW} numbers, layout {LAYOUT!r}\n"
        "  Fix:   delete that file; the next run rebuilds it (about a minute of CPU for n_sub = 25)."
    )


def _load(path: Path) -> tuple[NDArray | None, dict[tuple[int, int, int], int]]:
    """The store memory-mapped, and the row of each offset in it; (None, {}) when there is no store."""
    if not path.exists():
        return None, {}
    try:
        rows = np.load(path, mmap_mode="r")
    except ValueError as err:
        raise _layout_error(path, f"not a .npy array: {err}") from None
    if rows.ndim != 2 or rows.shape[1] != ROW or rows.dtype != np.float64:
        raise _layout_error(path, f"shape {rows.shape}, dtype {rows.dtype}")
    index = {(int(r[0]), int(r[1]), int(r[2])): i for i, r in enumerate(np.asarray(rows[:, :3]))}
    return rows, index


def _save(path: Path, rows: NDArray) -> None:
    """Write the rows to the store atomically: a temporary file in the same directory, then a rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".npy")
    os.close(fd)
    tmp = Path(name)
    try:
        np.save(tmp, rows)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def _coefficients_for(canonical: list[tuple[int, int, int]]) -> dict[tuple[int, int, int], NDArray]:
    """The coefficients (T_MAX + 1, len(A_LIST)) of these canonical offsets, from the store.

    Any offset missing from the store is computed (compiled), appended, and the store rewritten and
    mapped again. The arrays returned are views of the mapped file: read-only, paged in on demand.
    """
    global _LOADED
    path = store_path()
    if _LOADED is None:
        _LOADED = _load(path)
    rows, index = _LOADED
    missing = [c for c in canonical if c not in index]
    if missing:
        new = np.empty((len(missing), ROW))
        new[:, :3] = np.array(missing, dtype=float)
        new[:, 3:] = coefficients_fortran(np.array(missing)).reshape(len(missing), -1)
        _save(path, new if rows is None else np.concatenate([np.asarray(rows), new]))
        _LOADED = _load(path)
        rows, index = _LOADED
    assert rows is not None
    shape = (T_MAX + 1, len(A_LIST))
    return {c: rows[index[c], 3:].reshape(shape) for c in canonical}


def averaged_pair_blocks_9x9_kseries(
    S: NDArray, omega: complex, ref: ReferenceMedium, d: float, *, tol: float, max_reach: float
) -> tuple[NDArray, NDArray]:
    """The averaged 9 x 9 blocks at separations S (N, 3) by the wavenumber series, where it applies.

    Args:
        S: Separations (N, 3).
        omega: Angular frequency (complex when attenuated).
        ref: Background medium.
        d: Cell side.
        tol: Truncation tolerance of the series (``cell_average.kseries_tol``).
        max_reach: The largest |k_S| r_max the series takes (``cell_average.kseries_max_reach``): beyond it
            the series loses digits to cancellation (see the module docstring), and the offset is left
            to the caller.

    Returns:
        (blocks (N, 9, 9), covered (N,)): covered is False for a separation that is not a non-zero lattice
        offset, beyond max_reach, or whose series needs more than T_MAX powers; its block is zero, for
        the caller to fill.
    """
    S = np.atleast_2d(np.asarray(S, dtype=float))
    offsets = np.rint(S / d).astype(int)
    on_lattice = (np.max(np.abs(S - d * offsets), axis=1) <= 1.0e-9 * d) & np.any(offsets != 0, axis=1)
    blocks = np.zeros((len(S), 9, 9), dtype=complex)
    covered = np.zeros(len(S), dtype=bool)
    rows = np.flatnonzero(on_lattice)
    if not len(rows):
        return blocks, covered

    canonical, inverse, pinv, sign = cube_maps(offsets[rows])
    # beyond max_reach the cancellation costs digits: those offsets are neither computed nor stored
    r_max = (np.linalg.norm(canonical.astype(float), axis=1) + np.sqrt(3.0) / 2.0) * d
    near = np.flatnonzero(abs(omega / ref.beta) * r_max <= max_reach)
    ok = np.zeros(len(canonical), dtype=bool)
    blocks_c = np.zeros((len(canonical), 9, 9), dtype=complex)
    if len(near):
        keys = [(int(c[0]), int(c[1]), int(c[2])) for c in canonical[near]]
        table = _coefficients_for(keys)
        U = np.stack([table[k] for k in keys])  # (K, T + 1, n_a), gathered from the mapped store
        G, Gd, Gdd, ok_near = canonical_greens(canonical[near], U, omega, ref, d, tol)
        ok[near] = ok_near
        done = near[ok_near]
        blocks_c[done] = assemble_from_greens(G[ok_near], Gd[ok_near], Gdd[ok_near])  # once per canonical
    use = ok[inverse]
    if not use.any():
        return blocks, covered
    target = rows[use]
    blocks[target] = cube_image_blocks(blocks_c, inverse[use], pinv[use], sign[use])
    covered[target] = True
    return blocks, covered


def canonical_greens(
    canonical: NDArray, U: NDArray, omega: complex, ref: ReferenceMedium, d: float, tol: float
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """The cell-averaged G, dG, ddG at canonical offsets from their coefficients, all offsets at once.

    The sum of ``graded_voxel.derivatives.moment_table_kseries`` (weight W = 0), vectorised over the
    offsets: the same weights, the same truncation for each offset (the first t > 4 whose bound
    (|k_S| r_max)^(t - 2) / (t - 2)! is below tol), the same refusal when no t <= T_MAX reaches it. Equal to
    that function, called offset by offset, to round-off (``tests/test_cell_average_kseries.py``).

    Args:
        canonical: Integer offsets (K, 3).
        U: Their coefficients (K, T_MAX + 1, len(A_LIST)).
        omega: Angular frequency.
        ref: Background medium.
        d: Cell side.
        tol: Truncation tolerance.

    Returns:
        (G (K, 3, 3), Gd (K, 3, 3, 3), Gdd (K, 3, 3, 3, 3), ok (K,)): the averages, and whether the series
        reached the tolerance; where it did not, the entries are zero.
    """
    k_s, k_p = omega / ref.beta, omega / ref.alpha
    r_max = (np.linalg.norm(canonical.astype(float), axis=1) + np.sqrt(3.0) / 2.0) * d
    t_all = np.arange(T_MAX + 1)
    fact = np.array([float(math.factorial(t)) for t in t_all])
    # the first t > 4 with (|k_S| r_max)^(t - 2) / (t - 2)! < tol, in moment_table_kseries' arithmetic
    t_try = t_all[5:]
    small = (abs(k_s) * r_max[:, None]) ** (t_try - 2) / fact[t_try - 2] < tol
    ok = small.any(axis=1)
    n_terms = np.where(ok, t_try[np.argmax(small, axis=1)], 0)
    keep = t_all[None, :] < n_terms[:, None]  # (K, T + 1): the terms each offset sums

    s_t = d**t_all
    c1 = (1j * k_s) ** t_all / fact * s_t
    c2 = np.where(t_all >= 1, ((1j * k_s) ** t_all - (1j * k_p) ** t_all) / (fact * k_s**2), 0.0) * s_t
    a_deg = np.array([sum(a) for a in A_LIST], dtype=float)
    scale = d ** (2.0 - a_deg)  # side^((t - 1) - |a| + 3) = side^t side^(2 - |a|)
    v1 = np.einsum("kt,t,kta->ka", keep, c1, U) * scale
    v2 = np.einsum("kt,t,kta->ka", keep, c2, U) * scale

    rows_d, rows_b = gd._kseries_rows(tuple(_D_LIST), A_LIST)
    T = np.zeros((len(canonical), 3, 3, len(_D_LIST)), dtype=complex)
    for i in range(3):
        for n in range(i, 3):
            val = v2[:, rows_b[i][n]]
            if i == n:
                val = val + v1[:, rows_d]
            T[:, i, n] = val
            T[:, n, i] = val
    T /= 4.0 * np.pi * ref.mu * d**3  # the integral over the cell, as an average

    pos = {a: k for k, a in enumerate(_D_LIST)}

    def column(*axes: int) -> int:
        """Where the derivative d_{axes} sits in T's last index."""
        e = [0, 0, 0]
        for ax in axes:
            e[ax] += 1
        return pos[(e[0], e[1], e[2])]

    G = T[:, :, :, column()]
    Gd = np.stack([T[:, :, :, column(k)] for k in range(3)], axis=-1)
    Gdd = np.stack([np.stack([T[:, :, :, column(k, m)] for m in range(3)], -1) for k in range(3)], axis=-2)
    G[~ok], Gd[~ok], Gdd[~ok] = 0.0, 0.0, 0.0
    return G, Gd, Gdd, ok
