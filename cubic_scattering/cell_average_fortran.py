"""The cell-averaged pair block by the compiled quadrature (Fortran 2008, OpenMP over offsets).

The same object as ``cell_averaged_pair.averaged_pair_block_9x9``, for many separations in one call: the
receiver-cell Gauss rule, transcribed term for term in ``cubic_scattering/fortran/cell_average.f90``, with
the same nodes (``_cell_nodes``), the same Gauss orders (``auto_n_gauss``), the same series switch and
length for the mode difference, and the same closed forms. The 9 x 9 assembly from the averaged tensors
is the package's own (``cell_averaged_pair._assemble``), so the convention cannot differ.

The memo is shared with the Python route: an on-lattice separation found there is returned from it, and
every on-lattice separation computed here is stored in it, under the same key.

There is no fallback: if the extension has not been built, the import fails with the instruction to
build it.
"""

import numpy as np
from numpy.typing import NDArray

from . import cell_averaged_pair as cap
from .cell_averaged_lattice import _cell_nodes
from .effective_contrasts import ReferenceMedium
from .kupradze_derivatives import DIFFERENCE_N_SERIES, DIFFERENCE_SERIES_LIMIT

try:
    from .fortran._cell_average import cell_average as _compiled
except ImportError as err:
    raise ImportError(
        "the compiled cell average is not built.\n"
        "  What:  cubic_scattering/fortran/_cell_average*.so is missing or cannot be loaded.\n"
        "  Where: built from cubic_scattering/fortran/cell_average.f90.\n"
        "  Fix:   conda run -n seismic python -m cubic_scattering.fortran.build\n"
        "         (the seismic environment supplies gfortran, Meson, Ninja and the OpenMP runtime)."
    ) from err

__all__ = ["averaged_pair_blocks_9x9_fortran", "averaged_tensors_fortran"]


def averaged_tensors_fortran(
    S: NDArray, kappa_p: complex, kappa_s: complex, d: float, n_gauss: NDArray
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
    """Receiver-cell averages of d^2, d^3, d^4 of D = g_S - g_P and of d^0, d^1, d^2 of g_S.

    Args:
        S: Separations (N, 3), each with its receiver cell clear of the source.
        kappa_p: P wavenumber omega / alpha.
        kappa_s: S wavenumber omega / beta.
        d: Cell SIDE.
        n_gauss: Gauss points per axis for each separation, shape (N,).

    Returns:
        (dd2, dd3, dd4, ds0, ds1, ds2) with shapes (N, 3, 3), (N, 3, 3, 3), (N, 3, 3, 3, 3), (N,),
        (N, 3), (N, 3, 3): the entries 2..4 of ``averaged_pair_difference_tensors`` and 0..2 of
        ``averaged_pair_scalar_tensors``, for every separation.

    Raises:
        ValueError: If a receiver cell reaches its source.
    """
    S = np.atleast_2d(np.asarray(S, dtype=float))
    for s in S:
        cap._check_receiver_clear(s, d)
    orders = sorted({int(g) for g in n_gauss})
    nodes = np.zeros((max(orders), len(orders)))
    weights = np.zeros((max(orders), len(orders)))
    for k, ng in enumerate(orders):
        nodes[:ng, k], weights[:ng, k] = _cell_nodes(0.5 * d, ng)
    rule = np.array([orders.index(int(g)) + 1 for g in n_gauss], dtype=np.int32)
    dd2, dd3, dd4, ds0, ds1, ds2 = _compiled.averaged_tensors(
        np.asfortranarray(S.T),
        complex(kappa_p),
        complex(kappa_s),
        np.asfortranarray(nodes),
        np.asfortranarray(weights),
        np.array(orders, dtype=np.int32),
        rule,
        float(DIFFERENCE_SERIES_LIMIT),
        int(DIFFERENCE_N_SERIES),
    )
    return (
        np.ascontiguousarray(np.moveaxis(dd2, -1, 0)),
        np.ascontiguousarray(np.moveaxis(dd3, -1, 0)),
        np.ascontiguousarray(np.moveaxis(dd4, -1, 0)),
        np.ascontiguousarray(ds0),
        np.ascontiguousarray(np.moveaxis(ds1, -1, 0)),
        np.ascontiguousarray(np.moveaxis(ds2, -1, 0)),
    )


def averaged_pair_blocks_9x9_fortran(
    S: NDArray, omega: complex, ref: ReferenceMedium, d: float, *, n_gauss: int | None = None
) -> NDArray:
    """``averaged_pair_block_9x9`` at every separation of S (N, 3), shape (N, 9, 9).

    Memoised in the same table, under the same key, as the Python route. Returns fresh arrays, so a
    caller may write into them without corrupting the memo.

    Raises:
        ValueError: If a receiver cell reaches its source.
    """
    S = np.atleast_2d(np.asarray(S, dtype=float))
    out = np.zeros((len(S), 9, 9), dtype=complex)
    keys: list[tuple | None] = []
    todo: list[int] = []
    for i, s in enumerate(S):
        key = cap._lattice_key(s, omega, ref, d, n_gauss)
        hit = cap._BLOCK_CACHE.get(key) if key is not None else None
        if hit is not None:
            out[i] = hit
        else:
            todo.append(i)
        keys.append(key)
    if not todo:
        return out

    St = S[todo]
    ng = np.array([cap.auto_n_gauss(s, d) if n_gauss is None else n_gauss for s in St])
    dd2, dd3, dd4, ds0, ds1, ds2 = averaged_tensors_fortran(St, omega / ref.alpha, omega / ref.beta, d, ng)
    zero = np.zeros(())
    for j, i in enumerate(todo):
        d_diff = [zero, zero, dd2[j], dd3[j], dd4[j]]  # orders 0 and 1 of D do not enter the block
        d_s = [ds0[j], ds1[j], ds2[j]]
        out[i] = cap._assemble(d_diff, d_s, omega, ref)
        key = keys[i]
        if key is not None and len(cap._BLOCK_CACHE) < cap._CACHE_MAX:
            cap._BLOCK_CACHE[key] = out[i].copy()
    return out
