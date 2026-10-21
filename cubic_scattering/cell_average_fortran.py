"""The cell-averaged pair block by the compiled quadrature (Fortran 2008, OpenMP over offsets).

The same object as ``cell_averaged_pair.averaged_pair_block_9x9``, for many separations in one call: the
receiver-cell Gauss rule, transcribed term for term in ``cubic_scattering/fortran/cell_average.f90``, with
the same nodes (``_cell_nodes``), the same Gauss orders (``auto_n_gauss``), the same series switch and
length for the mode difference, and the same closed forms. The 9 x 9 assembly from the averaged tensors
is the package's own (``cell_averaged_pair._assemble``), so the convention cannot differ.

The memo is shared with the Python route: an on-lattice separation found there is returned from it, and
every on-lattice separation computed here is stored in it, under the same key.

⚠ ACCURACY IS THAT OF THE GAUSS ORDERS, ABOUT 1e-8 AT WORST. Equal to the Python quadrature to 1e-12 at
the same orders (``tests/test_cell_average_fortran.py``), which certifies the transcription only. The
automatic orders themselves (``auto_n_gauss``) leave the second shell wrong by up to 1.9e-8 (offset
(2, 0, 0)) and fall to 4e-13 only from about (5, 3, 2) outward: see ``auto_n_gauss`` for the measured
table. The default route (``cell_average.route: kseries``, series then expansion about the cell centre
then a converged quadrature) is accurate to about 1.2e-13 at every offset.

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

__all__ = ["auto_n_gauss_many", "averaged_pair_blocks_9x9_fortran", "averaged_tensors_fortran"]


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

    THE CUBE'S SYMMETRY. Every separation is s = M c with c = (|s|, sorted descending) and M a signed
    permutation, one of the 48 symmetries of the cube. The receiver cell and its Gauss rule are invariant
    under M and the integrand is radial, so each averaged tensor at s is the one at c with its indices
    permuted and signed: T_s[i, j, ...] = sigma_i sigma_j ... T_c[pi(i), pi(j), ...]. The quadrature
    therefore runs once per distinct (c, Gauss order), about 1/48 of a lattice's offsets, and the rest is
    index bookkeeping, exact up to the round-off of summing the nodes in another order.

    Raises:
        ValueError: If a receiver cell reaches its source.
    """
    S = np.atleast_2d(np.asarray(S, dtype=float))
    blocked = np.max(np.abs(S), axis=1) <= 0.5 * d
    if np.any(blocked):
        cap._check_receiver_clear(S[np.argmax(blocked)], d)  # raises, with the full diagnostic
    n_gauss = np.asarray(n_gauss, dtype=int)
    absolute = np.abs(S)
    perm = np.argsort(-absolute, axis=1, kind="stable")  # c_k = |s|_{perm[k]}
    canonical = np.take_along_axis(absolute, perm, axis=1)
    keys, first, inverse = np.unique(
        np.column_stack([canonical, n_gauss]), axis=0, return_index=True, return_inverse=True
    )
    tensors = _canonical_tensors(keys[:, :3], kappa_p, kappa_s, d, n_gauss[first])
    pinv = np.argsort(perm, axis=1)  # |s|_i = c_{pinv[i]}
    sign = np.where(S < 0.0, -1.0, 1.0)
    return tuple(_signed_permutation(t[inverse.ravel()], pinv, sign) for t in tensors)  # type: ignore[return-value]


def cube_maps(offsets: NDArray) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Each offset as a signed permutation of its canonical offset (|o| sorted descending).

    Args:
        offsets: Offsets or separations (N, 3).

    Returns:
        (canonical (K, 3), inverse (N,), pinv (N, 3), sign (N, 3)): the distinct canonical offsets; the one
        each offset maps to; and the map itself, |o|_i = c_{pinv[i]} and o_i = sign[i] |o|_i.
    """
    absolute = np.abs(offsets)
    perm = np.argsort(-absolute, axis=1, kind="stable")
    canonical, inverse = np.unique(np.take_along_axis(absolute, perm, axis=1), axis=0, return_inverse=True)
    return canonical, inverse.ravel(), np.argsort(perm, axis=1), np.where(offsets < 0, -1.0, 1.0)


def cube_image_blocks(blocks_c: NDArray, inverse: NDArray, pinv: NDArray, sign: NDArray) -> NDArray:
    """The 9 x 9 blocks at every offset from those at the canonical offsets, by the cube's symmetry.

    A signed permutation of the axes (o_i = sign_i c_{pinv(i)}) is a signed permutation of the block's 9
    components: displacement i takes component pinv(i) with sign sign_i, and the Voigt pair (p, q) takes
    the pair (pinv(p), pinv(q)) with sign sign_p sign_q. A diagonal pair stays diagonal and an off-diagonal
    one off-diagonal, so the engineering factors of the contraction are untouched. Exact up to round-off,
    and cheaper than assembling each image: one gather of 81 entries per offset.
    """
    from .resonance_tmatrix import VOIGT_PAIRS

    voigt_of = np.zeros((3, 3), dtype=int)
    for v, (p, q) in enumerate(VOIGT_PAIRS):
        voigt_of[p, q] = voigt_of[q, p] = 3 + v
    first = np.array([p for p, _ in VOIGT_PAIRS])
    second = np.array([q for _, q in VOIGT_PAIRS])
    comp = np.concatenate([pinv, voigt_of[pinv[:, first], pinv[:, second]]], axis=1)  # (N, 9)
    sgn = np.concatenate([sign, sign[:, first] * sign[:, second]], axis=1)  # (N, 9)
    rows = np.arange(len(inverse))[:, None, None]
    gathered = blocks_c[inverse[rows], comp[:, :, None], comp[:, None, :]]
    return sgn[:, :, None] * sgn[:, None, :] * gathered


def _signed_permutation(T: NDArray, pinv: NDArray, sign: NDArray) -> NDArray:
    """T_s[p, i, j, ...] = sign[p, i] sign[p, j] ... T[p, pinv[p, i], pinv[p, j], ...], for any order."""
    n_sep, order = len(T), T.ndim - 1
    if order == 0:
        return T.copy()
    index: list[NDArray] = [np.arange(n_sep).reshape((n_sep,) + (1,) * order)]
    factor = np.ones((n_sep,) + (1,) * order)
    for axis in range(order):
        shape = [n_sep] + [1] * order
        shape[axis + 1] = 3
        index.append(pinv.reshape(shape))
        factor = factor * sign.reshape(shape)
    return factor * T[tuple(index)]


def _canonical_tensors(
    C: NDArray, kappa_p: complex, kappa_s: complex, d: float, n_gauss: NDArray
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
    """The compiled quadrature at the separations C (N, 3), one call, with the leading axis first."""
    orders = sorted({int(g) for g in n_gauss})
    nodes = np.zeros((max(orders), len(orders)))
    weights = np.zeros((max(orders), len(orders)))
    for k, ng in enumerate(orders):
        nodes[:ng, k], weights[:ng, k] = _cell_nodes(0.5 * d, ng)
    rule = (np.searchsorted(orders, np.asarray(n_gauss, dtype=int)) + 1).astype(np.int32)
    dd2, dd3, dd4, ds0, ds1, ds2 = _compiled.averaged_tensors(
        np.asfortranarray(C.T),
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
    keys = _lattice_keys(S, omega, ref, d, n_gauss)
    todo: list[int] = []
    for i, key in enumerate(keys):
        hit = cap._BLOCK_CACHE.get(key) if key is not None else None
        if hit is not None:
            out[i] = hit
        else:
            todo.append(i)
    if not todo:
        return out

    St = S[todo]
    ng = auto_n_gauss_many(St, d) if n_gauss is None else np.full(len(St), n_gauss)
    tensors = averaged_tensors_fortran(St, omega / ref.alpha, omega / ref.beta, d, ng)
    blocks = _assemble_many(*tensors, omega, ref)
    out[todo] = blocks
    for j, i in enumerate(todo):
        key = keys[i]
        if key is not None and len(cap._BLOCK_CACHE) < cap._CACHE_MAX:
            cap._BLOCK_CACHE[key] = blocks[j].copy()
    return out


def auto_n_gauss_many(S: NDArray, d: float) -> NDArray:
    """``cell_averaged_pair.auto_n_gauss`` for every separation of S (N, 3) at once, shape (N,)."""
    closest = np.maximum(np.abs(S) - 0.5 * d, 0.0)
    q = np.sqrt(np.sum(closest * closest, axis=1)) / d
    return np.select([q < 0.6, q < 0.8, q < 1.5], [16, 12, 8], default=6)


def _lattice_keys(
    S: NDArray, omega: complex, ref: ReferenceMedium, d: float, n_gauss: int | None
) -> list[tuple | None]:
    """``cell_averaged_pair._lattice_key`` for every separation of S (N, 3); None off the lattice."""
    offsets = np.rint(S / d).astype(int)
    on_lattice = np.max(np.abs(S - d * offsets), axis=1) <= 1.0e-9 * d
    tail = (float(d), complex(omega), ref.alpha, ref.beta, ref.rho, n_gauss)
    return [
        ((int(o[0]), int(o[1]), int(o[2])), *tail) if ok else None
        for o, ok in zip(offsets.tolist(), on_lattice.tolist(), strict=True)
    ]


def _assemble_many(
    dd2: NDArray,
    dd3: NDArray,
    dd4: NDArray,
    ds0: NDArray,
    ds1: NDArray,
    ds2: NDArray,
    omega: complex,
    ref: ReferenceMedium,
) -> NDArray:
    """The 9 x 9 blocks from the averaged tensors, all at once: ``cell_averaged_pair._assemble`` batched.

    G_ij = (1 / rho w^2) [ delta_ij k_S^2 g_S + d_i d_j D ] and its first and second derivatives, as
    ``kupradze_derivatives.greens_from_difference``, then the package's Voigt contraction as linear maps
    (``graded_voxel.kernel._assemble``).
    """
    from .graded_voxel.kernel import _assemble as assemble_from_greens

    k_s2 = (omega / ref.beta) ** 2
    pref = 1.0 / (ref.rho * omega**2)
    delta = np.eye(3)
    G = pref * (k_s2 * ds0[:, None, None] * delta + dd2)
    Gd = pref * (k_s2 * np.einsum("ij,pk->pijk", delta, ds1) + dd3)
    Gdd = pref * (k_s2 * np.einsum("ij,pkl->pijkl", delta, ds2) + dd4)
    return assemble_from_greens(G, Gd, Gdd)
