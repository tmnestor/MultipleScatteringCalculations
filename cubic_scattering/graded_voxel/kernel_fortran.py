"""The 9 x 9 point propagator by the compiled kernel (Fortran 2008, OpenMP over points).

The same object as ``kernel.kernel_9x9``, transcribed term for term in
``cubic_scattering/fortran/point_kernel.f90``: the same ladder, the same series below
``kernel.SERIES_LIMIT`` with ``kernel.N_SERIES`` terms, the same closed forms above it, the same static /
dynamic split. The Voigt contraction is the package's own maps (``kernel.voigt_maps``), passed in, so the
convention cannot differ between the two. Each point is computed independently, so the result does not
depend on the number of OpenMP threads.

Nothing calls this automatically. It is a separate function, and there is no fallback: if the extension
has not been built, the import fails with the instruction to build it.
"""

import numpy as np
from numpy.typing import NDArray

from ..effective_contrasts import ReferenceMedium
from .kernel import N_SERIES, SERIES_LIMIT, voigt_maps

try:
    from ..fortran._point_kernel import point_kernel as _compiled
except ImportError as err:
    raise ImportError(
        "the compiled point kernel is not built.\n"
        "  What:  cubic_scattering/fortran/_point_kernel*.so is missing or cannot be loaded.\n"
        "  Where: built from cubic_scattering/fortran/point_kernel.f90.\n"
        "  Fix:   conda run -n seismic python -m cubic_scattering.fortran.build\n"
        "         (the seismic environment supplies gfortran, Meson, Ninja and the OpenMP runtime)."
    ) from err


def kernel_9x9_fortran(
    X: NDArray, omega: complex, ref: ReferenceMedium, *, static: bool = True, dynamic: bool = True
) -> NDArray:
    """The propagator at separations X = x - x' (N, 3), shape (N, 9, 9), as ``kernel.kernel_9x9``.

    Raises:
        ValueError: at r = 0, where the propagator is a distribution.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if np.any(np.linalg.norm(X, axis=1) == 0.0):
        raise ValueError(
            "kernel_9x9_fortran: r = 0 requested. The propagator is a distribution at the origin; its "
            "cell integrals come from graded_voxel.blocks.near_block, never from a point value."
        )
    mc, mh, ms = voigt_maps()
    p = _compiled.kernel_9x9(
        np.asfortranarray(X.T),
        complex(omega),
        float(ref.alpha),
        float(ref.beta),
        float(ref.mu),
        int(static),
        int(dynamic),
        float(SERIES_LIMIT),
        int(N_SERIES),
        np.asfortranarray(mc),
        np.asfortranarray(mh),
        np.asfortranarray(ms),
    )
    return np.ascontiguousarray(np.moveaxis(p, 2, 0))


def apply_9x9_fortran(
    obs: NDArray, points: NDArray, sources: NDArray, omega: complex, ref: ReferenceMedium, *, rows: int
) -> NDArray:
    """sum_p P(obs - x_p) s_p at every observer, shape (M, rows): the propagator applied as it is evaluated.

    The same propagator as ``kernel_9x9_fortran`` (full, static and dynamic), but never stored: the Python
    route built every point's 9 x 9 block for each observer at once (about 5.5 GB per observer for the
    graded sphere of the thesis's Chapter 7). rows = 3 gives the displacement only and skips the second
    derivatives that the strain rows need; rows = 9 gives displacement and engineering strain.

    Args:
        obs: Observers (M, 3).
        points: Source points (N, 3).
        sources: Their sources (N, 9): force, then Voigt stress, as ``kernel_9x9`` columns.
        omega: Angular frequency.
        ref: Background medium.
        rows: 3 (displacement) or 9 (displacement and strain).

    Raises:
        ValueError: if a source point coincides with an observer (r = 0, where the propagator is a
            distribution), or rows is neither 3 nor 9.
    """
    if rows not in (3, 9):
        raise ValueError(
            f"apply_9x9_fortran: rows is {rows}; it must be 3 (displacement) or 9 (with strain)."
        )
    obs = np.atleast_2d(np.asarray(obs, dtype=float))
    points = np.atleast_2d(np.asarray(points, dtype=float))
    mc, mh, ms = voigt_maps()
    out, n_zero = _compiled.apply_9x9(
        np.asfortranarray(obs.T),
        np.asfortranarray(points.T),
        np.asfortranarray(np.asarray(sources, dtype=complex).T),
        complex(omega),
        float(ref.alpha),
        float(ref.beta),
        float(ref.mu),
        float(SERIES_LIMIT),
        int(N_SERIES),
        np.asfortranarray(mc),
        np.asfortranarray(mh),
        np.asfortranarray(ms),
        int(rows),
    )
    if np.any(n_zero):
        which = np.flatnonzero(n_zero)
        raise ValueError(
            f"apply_9x9_fortran: r = 0 between observer(s) {which.tolist()} and a source point. The "
            "propagator is a distribution at the origin: an observer must lie outside the sources (for a "
            "cell's Gauss nodes, outside the cell)."
        )
    return np.ascontiguousarray(out.T)
