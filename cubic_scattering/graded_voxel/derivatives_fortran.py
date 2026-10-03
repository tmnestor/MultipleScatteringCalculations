"""Derivatives of g_S and B to any order by the compiled routine (Fortran 2008, OpenMP over points).

The same object as ``derivatives.scalar_derivative_fields_python``, transcribed term for term in
``cubic_scattering/fortran/green_derivatives.f90``: the same ladder, the same series below
``kernel.SERIES_LIMIT`` with ``kernel.N_SERIES`` terms, the same closed forms above it. The term table and
the polynomials p_q are computed in Python and passed in, so the two cannot differ in them.

There is no fallback: if the extension has not been built, the import fails with the instruction to
build it.
"""

from functools import cache

import numpy as np
from numpy.typing import NDArray

from ..effective_contrasts import ReferenceMedium
from .derivatives import _order_for, bessel_polynomials, derivative_terms, multi_indices
from .kernel import N_SERIES, SERIES_LIMIT

try:
    from ..fortran._green_derivatives import green_derivatives as _compiled
except ImportError as err:
    raise ImportError(
        "the compiled Green's-tensor derivatives are not built.\n"
        "  What:  cubic_scattering/fortran/_green_derivatives*.so is missing or cannot be loaded.\n"
        "  Where: built from cubic_scattering/fortran/green_derivatives.f90.\n"
        "  Fix:   conda run -n seismic python -m cubic_scattering.fortran.build\n"
        "         (the seismic environment supplies gfortran, Meson, Ninja and the OpenMP runtime)."
    ) from err


@cache
def _term_table(order: int) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """The terms of every multi-index to this order, flattened for Fortran (1-based starts)."""
    start, coef, exps, qs = [1], [], [], []
    for a in multi_indices(order):
        for c, e, q in derivative_terms(a):
            coef.append(c)
            exps.append(e)
            qs.append(q)
        start.append(len(coef) + 1)
    return (
        np.array(start, dtype=np.int32),
        np.array(coef, dtype=float),
        np.asfortranarray(np.array(exps, dtype=np.int32).T),
        np.array(qs, dtype=np.int32),
    )


def scalar_derivative_fields_fortran(
    X: NDArray, omega: complex, ref: ReferenceMedium, n_s: int, n_b: int
) -> tuple[NDArray, NDArray]:
    """As ``derivatives.scalar_derivative_fields``: shapes (n_s, N) and (n_b, N)."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    order = _order_for(max(n_s, n_b))
    start, coef, exps, qs = _term_table(order)
    n_idx = len(multi_indices(order))
    out_s, out_b = _compiled.scalar_derivative_fields(
        np.asfortranarray(X.T),
        complex(omega),
        float(ref.alpha),
        float(ref.beta),
        float(SERIES_LIMIT),
        int(N_SERIES),
        np.asfortranarray(bessel_polynomials(order)),
        start[: n_idx + 1],
        coef,
        exps,
        qs,
        int(n_s),
        int(n_b),
    )
    return out_s, out_b
