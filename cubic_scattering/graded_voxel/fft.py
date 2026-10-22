"""FFT matrix-vector solver for the graded voxel on the sphere grid.

The coupling blocks K(o) depend only on the grid offset o, and each cell's contrast E_n acts on its own
cell, so the Galerkin operator (M - K E) is a block-Toeplitz convolution followed by a cell-local map:

    y_m = M psi_m - sum_n K(g_m - g_n) s_n,     s_n = E_n psi_n  (the source monomial coefficients),

which is evaluated with FFTs on the padded grid of (2n - 1)^3 points (circular embedding of the offsets).

THE BLOCKS BY THE CUBE'S SYMMETRY.  The background is isotropic, so P(Q r) = S P(r) S^T for every signed
permutation Q (48 of them), S = diag(Q, R6(Q)); for signed permutations the tensor and engineering Voigt
representations coincide.  The cube maps to itself, so with xi = Q eta, L_a(Q eta) = sum_b T_ab L_b(eta),
m_c(Q eta) = sum_d U_cd m_d(eta):

    K(Q o)_ac = sum_bd T_ab U_cd  S K_bd(o) S^T.

Only one offset per orbit is computed (``coupling_block``); the rest follow exactly.

THE SOLVE.  GMRES with a block-Jacobi preconditioner: each cell's own (M - K(0) E_n)^-1, which carries the
self-interaction that dominates the second-order term.  scipy's ``maxiter`` counts RESTART CYCLES, so the
total number of iterations is bounded explicitly as restart * max_cycles.
"""

import itertools
import time
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator, gmres

from ..effective_contrasts import MaterialContrast, ReferenceMedium
from ..sphere_scattering import _plane_wave_strain_voigt
from ..sphere_scattering_fft import _build_grid_index_map, fast_odd_size
from .basis import (
    contrast_values,
    gram_test,
    monomials,
    source_expansion,
    source_exponents,
)
from .blocks import NEAR_STORE_MAX_KH, coupling_block, coupling_blocks_far_lattice, near_block_series
from .kernel import point_kernel_backend
from .site import cell_contrast_coefficients_all
from .solver import GradedVoxelResult, field_sizes, plane_wave_moments

VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


@lru_cache(maxsize=1)
def signed_permutations() -> tuple[NDArray, ...]:
    """The 48 signed permutation matrices of the cube group."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            q = np.zeros((3, 3))
            for i, j in enumerate(perm):
                q[i, j] = signs[i]
            out.append(q)
    return tuple(out)


def _voigt_rep(q: NDArray) -> NDArray:
    """R6 with (Q e Q^T) in Voigt = R6 e in Voigt, for a signed permutation Q."""
    rep = np.zeros((6, 6))
    for b, (m, n) in enumerate(VOIGT_PAIRS):
        e = np.zeros((3, 3))
        e[m, n] = e[n, m] = 1.0
        er = q @ e @ q.T
        for a, (i, j) in enumerate(VOIGT_PAIRS):
            rep[a, b] = er[i, j]
    return rep


def symmetry_reps(q: NDArray, n_source: int = 10, n_test: int = 4) -> tuple[NDArray, NDArray, NDArray]:
    """(T, U, S): the field-function (n_test square), source-monomial (n_source square) and 9-component
    (9 x 9) representations of Q."""
    key = (q.tobytes(), n_source, n_test)
    if key not in _REPS:
        _REPS[key] = _symmetry_reps(q, n_source, n_test)
    return _REPS[key]


_REPS: dict[tuple, tuple[NDArray, NDArray, NDArray]] = {}


def _symmetry_reps(q: NDArray, n_source: int, n_test: int) -> tuple[NDArray, NDArray, NDArray]:
    rng = np.random.default_rng(12345)
    eta = rng.uniform(-1.0, 1.0, (96, 3))
    fa = contrast_values(eta @ q.T)[:n_test]  # Q_a(Q eta_k); the first four are L_a
    fb = contrast_values(eta)[:n_test]
    t = np.round(np.linalg.lstsq(fb.T, fa.T, rcond=None)[0].T)  # exactly 0 or +-1
    exps = source_exponents(n_source)
    a = monomials(exps, eta @ q.T)  # m_c(Q eta_k)
    b = monomials(exps, eta)  # m_d(eta_k)
    u = np.linalg.lstsq(b.T, a.T, rcond=None)[0].T
    u = np.round(u)  # exactly 0 or +-1 for a signed permutation
    s = np.zeros((9, 9))
    s[:3, :3] = q
    s[3:, 3:] = _voigt_rep(q)
    return t, u, s


def _signed_permutation(m: NDArray) -> tuple[NDArray, NDArray] | None:
    """(perm, sign) with m[i, perm[i]] = sign[i] the only nonzero of row i, or None if m is not one."""
    perm = np.argmax(np.abs(m), axis=1)
    sign = m[np.arange(len(m)), perm]
    if not (np.all(np.abs(sign) == 1.0) and np.count_nonzero(m) == len(m) and len(set(perm)) == len(m)):
        return None
    return perm, sign


def _transform(q: NDArray, block: NDArray) -> NDArray:
    """T K S^T under the cube symmetry q, with U and S on the source and 9-component indices.

    Every representation of a signed permutation is itself one, so the product is a reindexing with signs:
    no arithmetic beyond the signs (the dense einsum it replaces cost 0.05 s per offset at degree two).
    """
    t, u, s = symmetry_reps(q, block.shape[1], block.shape[0])
    sp = [_signed_permutation(m) for m in (t, u, s)]
    if any(x is None for x in sp):
        tmp = np.einsum("ij,bdjk,lk->bdil", s, block, s)
        return np.einsum("ab,cd,bdil->acil", t, u, tmp)
    (pt, st), (pu, su), (ps, ss) = sp
    out = block[np.ix_(pt, pu, ps, ps)]
    return out * (
        st[:, None, None, None]
        * su[None, :, None, None]
        * ss[None, None, :, None]
        * ss[None, None, None, :]
    )


def offset_blocks(
    n_sub: int,
    h: float,
    omega: float,
    ref: ReferenceMedium,
    n_source: int = 10,
    n_test: int = 4,
) -> dict[tuple[int, int, int], NDArray]:
    """K(o) for every offset o in [-(n-1), n-1]^3: one ``coupling_block`` per cube-group orbit."""
    canonical = canonical_blocks(n_sub, h, omega, ref, n_source, n_test)
    return {off: _transform(q, canonical[c]) for off, c, q in _orbit_map(n_sub)}


def canonical_blocks(
    n_sub: int,
    h: float,
    omega: float,
    ref: ReferenceMedium,
    n_source: int = 10,
    n_test: int = 4,
) -> dict[tuple[int, int, int], NDArray]:
    """One coupling block per cube-group orbit of the offsets in [-(n-1), n-1]^3, keyed by the orbit's
    canonical offset (a >= b >= d >= 0): about 1/48 of the offsets, so the kernel need not be held per
    offset (``transformed_kernel``).

    The same Galerkin blocks as ``coupling_block``, by faster routes: the touching offsets (the self cell
    and its neighbours) by ``near_block_series`` within its stored moments (k_S h <= 1/2; ``near_block``
    beyond), the wavenumber series over universal moments, equal to
    ``near_block`` to about 1e-15 and 2 to 6 ms against 350 to 740 ms a block; the rest by
    ``coupling_blocks_far_lattice``, ``coupling_block``'s own s-form Gauss rule with the kernel evaluated
    once on a lattice shared by neighbouring offsets and contracted axis by axis (``coupling_blocks_far``,
    the same rule offset by offset, took 17.5 s at N = 32), equal to round-off.
    """
    rng = range(n_sub)
    canon = [c for c in itertools.product(rng, rng, rng) if c[0] >= c[1] >= c[2]]
    touching = [c for c in canon if c[0] <= 1]
    if abs(omega) / ref.beta * h <= NEAR_STORE_MAX_KH:  # the stored moments: milliseconds a block
        out = {c: near_block_series(c, h, omega, ref, n_source, n_test) for c in touching}
    else:  # beyond the store the series would compute its moments, minutes each: Gauss, as before
        out = {c: coupling_block(c, h, omega, ref, n_source, n_test) for c in touching}
    if n_sub > 2:
        out.update(coupling_blocks_far_lattice(n_sub, h, omega, ref, n_source, n_test))
    return dict(sorted(out.items()))


@lru_cache(maxsize=4)
def _orbit_map(n_sub: int) -> tuple[tuple, ...]:
    """(offset, canonical offset, the signed permutation taking the one to the other) for every offset."""
    qs = signed_permutations()
    out = []
    rng = range(-(n_sub - 1), n_sub)
    for off in itertools.product(rng, rng, rng):
        c = tuple(sorted((abs(o) for o in off), reverse=True))
        cv, target = np.array(c, dtype=float), np.array(off, dtype=float)
        out.append((off, c, next(q for q in qs if np.array_equal(q @ cv, target))))
    return tuple(out)


def transformed_kernel(
    canonical: dict[tuple[int, int, int], NDArray],
    n_sub: int,
    na: int,
    nc: int,
    dtype: type = np.complex128,
    path: Path | None = None,
    budget_bytes: float = 2e9,
) -> NDArray:
    """The coupling's kernel on the (2n - 1)^3 circular embedding, transformed: kh[row, col, i, j, k] with
    row = (a, i9) of the test functions and col = (c, j9) of the source monomials, as the FFT solve uses it.

    Built a group of rows at a time from the canonical blocks (``canonical_blocks``), each group's
    real-space kernel assembled by the cube's symmetry, transformed in double precision and stored in
    ``dtype``; the group's size keeps its working arrays within ``budget_bytes``.  In memory, or
    memory-mapped at ``path`` (created, overwritten), so that the model's size is bounded by the disk, not
    the memory.
    """
    npad = 2 * n_sub - 1
    rows, cols = na * 9, nc * 9
    shape = (rows, cols, npad, npad, npad)
    kh: NDArray
    if path is None:
        kh = np.empty(shape, dtype=dtype)
    else:
        kh = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    per_row = 2 * cols * npad**3 * 16  # the group's real-space array and its transform, complex128
    group = max(1, min(rows, int(budget_bytes // per_row)))
    offsets, inverse, q_index, keys = _orbit_arrays(n_sub)
    stack = np.stack([canonical[k] for k in keys])  # (K, n_test, n_source, 9, 9)
    maps = _signed_maps(stack.shape[2], stack.shape[1])
    wrapped = (offsets[:, 0] % npad, offsets[:, 1] % npad, offsets[:, 2] % npad)
    for r0 in range(0, rows, group):
        sel = slice(r0, min(rows, r0 + group))
        real = np.zeros((sel.stop - sel.start, cols, npad, npad, npad), dtype=complex)
        _fill_rows(real, sel, stack, inverse, q_index, maps, na, nc, wrapped)
        kh[sel] = np.fft.fftn(real, axes=(2, 3, 4))
        del real
    if isinstance(kh, np.memmap):
        kh.flush()
    return kh


def canonical_transformed_kernel(
    canonical: dict[tuple[int, int, int], NDArray],
    n_sub: int,
    na: int,
    nc: int,
    dtype: type = np.complex128,
    path: Path | None = None,
    budget_bytes: float = 2e9,
) -> NDArray:
    """The transformed kernel at the canonical k only: khc[k, col, row], k over the canonical offsets
    a >= b >= d >= 0 in sorted order (``canonical``'s keys), equal to ``transformed_kernel``'s
    kh[row, col, k] there to round-off.

    THE CUBE GROUP IN FOURIER SPACE.  K(Q o) = R K(o) R^T, R the signed permutation of Q on the rows and on
    the columns, and Q is orthogonal and maps the circular grid to itself (2n - 1 is odd, so the centred
    k in [-(n-1), n-1]^3 are unique), so K^(Q k) = R K^(k) R^T: the k-orbits are the offsets' orbits, and
    the kernel at one k per orbit (1/42 of the points at n = 32: 0.3 GB in double precision, against 13 GB)
    determines it everywhere (``fourier_symmetry_tables``, ``kernel_fortran.apply_canonical_fortran``).

    BUILT BY THE GROUP AGAIN, WITHOUT AN FFT.  Element by element the identity reads
    K^(Q k)[r, c] = sr_Q[r] sc_Q[c] K^(k)[pr_Q[r], pc_Q[c]], and the group does two things to the 36 x 90
    scalar arrays K[r, c](o):

    * an axis permutation Q moves array (r, c) to array (pr_Q[r], pc_Q[c]), so with Q chosen to take
      (r, c) to the representative (r0, c0) of its orbit,
      K^(k_c)[r, c] = sr_Q[r] sc_Q[c] K^(Q^T k_c)[r0, c0]: one array per orbit of index pairs is
      transformed, 576 of the 3,240 for p = r = 1 (orbits of 3 and 6); a pure permutation keeps Q^T k_c
      in the non-negative octant;
    * an axis flip maps every index to itself with a sign, so each array is even or odd along each axis,
      and its transform at the non-negative wavenumbers is a cosine or a sine sum over the non-negative
      octant of offsets, sum_o w_o f(o) cos(2 pi k o / n) (w_0 = 1, w_o = 2) or -2i sum_o f(o)
      sin(2 pi k o / n): N^3 values, not (2N - 1)^3, transformed by three N x N matrix products.

    So the real-space kernel is filled only for the representatives on the octant, from the canonical
    blocks (``_fill_rows``'s re-indexing), transformed by those products, and every entry of the canonical
    kernel gathered from them. At n = 32 this replaces 3,240 FFTs of 250,047 points by 576 octant
    transforms of 32,768 (24 s to a few). The representatives are transformed a batch at a time and the
    result gathered a batch of k at a time, each batch's working arrays within ``budget_bytes``. In memory,
    or memory-mapped at ``path`` (created, overwritten).

    Raises:
        ValueError: if a representation of the cube group is not a signed permutation
            (``fourier_symmetry_tables``).
    """
    rows, cols = na * 9, nc * 9
    keys = _orbit_arrays(n_sub)[3]
    stack = np.stack([canonical[k] for k in keys])  # (K, n_test, n_source, 9, 9): [a, c, i, l]
    blocks = (
        stack[:, :na, :nc].transpose(0, 1, 3, 2, 4).reshape(len(keys), rows, cols)
    )  # [k, (a, i), (c, l)]
    maps = fourier_symmetry_tables(n_sub, na, nc, stack.shape[2], stack.shape[1])[3:]
    return _canonical_hat_from_blocks(blocks, n_sub, maps, dtype, path, budget_bytes)


def _canonical_hat_from_blocks(
    blocks: NDArray,
    n_sub: int,
    maps: tuple[NDArray, ...],
    dtype: type,
    path: Path | None,
    budget_bytes: float,
) -> NDArray:
    """``canonical_transformed_kernel``'s construction from the canonical blocks as matrices: blocks
    (K, rows, cols) at the canonical offsets of ``_orbit_arrays(n_sub)``, and the symmetries' row and
    column maps (pr, sr, pc, sc) of ``fourier_symmetry_tables``. Returns khc (K, cols, rows). Shared by
    the Galerkin kernel (36 x 90 blocks) and the point kernel (9 x 9: one test function and one source
    monomial, the constant that every symmetry fixes).
    """
    rows, cols = blocks.shape[1], blocks.shape[2]
    pr, sr, pc, sc = maps
    npad = 2 * n_sub - 1
    offsets, inverse, q_index, keys = _orbit_arrays(n_sub)
    qs = np.stack(signed_permutations())
    # the orbits of index pairs under the axis permutations, and the permutation taking each pair to its
    # orbit's representative (the smallest flat index (r0 * cols + c0) of the orbit)
    perm_q = np.array([q for q in range(len(qs)) if np.all(qs[q] >= 0)])
    pair_key = pr[perm_q][:, :, None] * cols + pc[perm_q][:, None, :]  # (6, rows, cols)
    best = np.argmin(pair_key, axis=0)  # (rows, cols): which permutation
    reps, rep_of = np.unique(np.take_along_axis(pair_key, best[None], 0)[0], return_inverse=True)
    rep_of = rep_of.reshape(rows, cols)
    r0, c0 = np.divmod(reps, cols)
    q_star = perm_q[best]
    sign = sr[q_star, np.arange(rows)[:, None]] * sc[q_star, np.arange(cols)[None, :]]
    # each representative's parity along each axis, from the flip of that axis
    parity = np.empty((len(reps), 3))
    for axis in range(3):
        flip = int(
            next(
                q
                for q in range(len(qs))
                if np.array_equal(qs[q], np.diag(np.where(np.arange(3) == axis, -1.0, 1.0)))
            )
        )
        assert np.array_equal(pr[flip], np.arange(rows)) and np.array_equal(pc[flip], np.arange(cols))
        parity[:, axis] = sr[flip, r0] * sc[flip, c0]
    # the octant transforms: the even and the odd sum along one axis, k and o in 0 .. N - 1
    o = np.arange(n_sub)
    phase = 2.0 * np.pi * np.outer(o, o) / npad
    even = np.cos(phase) * np.where(o == 0, 1.0, 2.0)[None, :]
    odd = -2j * np.sin(phase)
    octant = np.flatnonzero(np.all(offsets >= 0, axis=1))  # (N^3,), lexicographic: reshapes to (N, N, N)
    q_o, i_o = q_index[octant], inverse[octant]
    hat = np.empty((len(reps), n_sub**3), dtype=complex)
    batch = max(1, int(budget_bytes // (4 * 16 * n_sub**3)))
    for pattern in {tuple(p) for p in parity}:
        members = np.flatnonzero(np.all(parity == pattern, axis=1))
        m0, m1, m2 = (even if p > 0 else odd for p in pattern)
        for start in range(0, len(members), batch):
            sel = members[start : start + batch]
            rr, cc = r0[sel][:, None], c0[sel][:, None]
            f = (
                sr[q_o[None, :], rr]
                * sc[q_o[None, :], cc]
                * blocks[i_o[None, :], pr[q_o[None, :], rr], pc[q_o[None, :], cc]]
            )
            # each axis as ONE matrix product, the axis brought last by a contiguous copy: a stacked
            # product splits into thousands of N x N products, which the BLAS threads past N = 40 at a
            # loss (2.3 s against 0.3 s at N = 41, the same numbers)
            nr, n = len(sel), n_sub
            f = (f.reshape(-1, n) @ m2.T).reshape(nr, n, n, n)  # (r, a, b, m)
            f = (np.ascontiguousarray(f.transpose(0, 1, 3, 2)).reshape(-1, n) @ m1.T).reshape(nr, n, n, n)
            f = f.transpose(0, 1, 3, 2)  # (r, a, l, m)
            f = (np.ascontiguousarray(f.transpose(0, 2, 3, 1)).reshape(-1, n) @ m0.T).reshape(nr, n, n, n)
            f = f.transpose(0, 3, 1, 2)  # (r, k, l, m)
            hat[sel] = f.reshape(nr, -1)
    # gather: khc[k, c, r] = sign[r, c] hat[rep_of[r, c], Q^T k], Q = perm_q[best[r, c]]
    key_arr = np.array(keys)
    moved = np.stack([key_arr @ qs[q] for q in perm_q]).astype(np.int64)  # (6, K, 3): Q^T k as rows
    at = (moved[..., 0] * n_sub + moved[..., 1]) * n_sub + moved[..., 2]  # (6, K) into the octant
    shape = (len(keys), cols, rows)
    khc: NDArray
    if path is None:
        khc = np.empty(shape, dtype=dtype)
    else:
        khc = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    k_batch = max(1, int(budget_bytes // (32 * rows * cols)))
    for k0 in range(0, len(keys), k_batch):
        ks = slice(k0, min(len(keys), k0 + k_batch))
        vals = sign[..., None] * hat[rep_of[..., None], at[best][..., ks]]  # (rows, cols, k)
        khc[ks] = vals.transpose(2, 1, 0)
    if isinstance(khc, np.memmap):
        khc.flush()
    return khc


@lru_cache(maxsize=8)
def fourier_symmetry_tables(
    n_sub: int, na: int, nc: int, n_source: int, n_test: int
) -> tuple[NDArray, ...]:
    """The index tables of the product through the canonical kernel, 0-based, int32 and float64.

    Returns (q_f, k_f, order, pr, sr, pc, sc): for every Fourier point f (C order of the (2n - 1)^3 grid)
    the symmetry q_f[f] taking its canonical k (index k_f[f] into the canonical kernel) to it; ``order``,
    the points grouped by canonical k; and for each of the 48 symmetries the row map (pr, sr) (48, na * 9)
    and the column map (pc, sc) (48, nc * 9):

        K^(f)[r, c] = sr[q, r] sc[q, c] K^(k_f)[pr[q, r], pc[q, c]],

    ``_fill_rows``'s re-indexing, row r = (a, i) = divmod(r, 9), column (c, l) likewise.

    Raises:
        ValueError: if a representation is not a signed permutation (then only the full kernel applies).
    """
    npad = 2 * n_sub - 1
    offsets, inverse, q_index, _ = _orbit_arrays(n_sub)
    maps = _signed_maps(n_source, n_test)  # the canonical blocks' own bases, as _fill_rows takes them
    if maps is None:
        raise ValueError(
            "fourier_symmetry_tables: a representation of the cube group is not a signed permutation, so "
            "the kernel cannot be stored at the canonical k. Fix: use the full kernel "
            "(transformed_kernel) with point_kernel.backend: python."
        )
    pt, st, pu, su, ps, ss = maps
    flat = ((offsets[:, 0] % npad) * npad + offsets[:, 1] % npad) * npad + offsets[:, 2] % npad
    q_f = np.empty(npad**3, dtype=np.int32)
    k_f = np.empty(npad**3, dtype=np.int32)
    q_f[flat], k_f[flat] = q_index, inverse
    order = np.argsort(k_f, kind="stable").astype(np.int32)
    a, i = np.divmod(np.arange(na * 9), 9)
    c, l9 = np.divmod(np.arange(nc * 9), 9)
    pr = (pt[:, a] * 9 + ps[:, i]).astype(np.int32)
    sr = (st[:, a] * ss[:, i]).astype(float)
    pc = (pu[:, c] * 9 + ps[:, l9]).astype(np.int32)
    sc = (su[:, c] * ss[:, l9]).astype(float)
    return q_f, k_f, order, pr, sr, pc, sc


#: offsets placed into a row group per gather, bounding the gather's temporary arrays
_FILL_CHUNK = 16384


@lru_cache(maxsize=8)
def _orbit_arrays(n_sub: int) -> tuple[NDArray, NDArray, NDArray, tuple[tuple[int, int, int], ...]]:
    """``_orbit_map`` as arrays, built without a Python loop over the offsets.

    Returns (offsets (N, 3), inverse (N,), q_index (N,), keys): every offset of [-(n-1), n-1]^3 in
    ``_orbit_map``'s order, the canonical offset each belongs to (an index into ``keys``, the canonical
    offsets a >= b >= d >= 0 in sorted order), and the signed permutation taking that canonical offset to it
    (an index into ``signed_permutations``). The permutation is the FIRST of the 48 that does so, as
    ``_orbit_map`` chooses it, so the two give identical kernels.

    Computed directly rather than by comparing every offset with its 48 images (5.5 s at n = 68):
    ``signed_permutations`` lists the axis permutations pi in ``itertools.permutations`` order, each with
    its eight sign patterns in ``itertools.product((1, -1), ...)`` order, and Q c = o needs
    c[pi(i)] = |o_i| and the signs of o, a zero component taking +1, the first. So the first match is the
    first pi that fits, with the signs of o, at index 8 pi + (4, 2, 1).(o < 0).
    """
    span = np.arange(-(n_sub - 1), n_sub)
    offsets = np.stack(np.meshgrid(span, span, span, indexing="ij"), -1).reshape(-1, 3)
    absval = np.abs(offsets)
    canonical = -np.sort(-absval, axis=1)
    rng = range(n_sub)
    keys_arr = np.array([c for c in itertools.product(rng, rng, rng) if c[0] >= c[1] >= c[2]])  # sorted
    code = (canonical[:, 0] * n_sub + canonical[:, 1]) * n_sub + canonical[:, 2]
    key_code = (keys_arr[:, 0] * n_sub + keys_arr[:, 1]) * n_sub + keys_arr[:, 2]  # increasing
    inverse = np.searchsorted(key_code, code)
    sign_index = (offsets < 0) @ np.array([4, 2, 1])
    q_index = np.full(len(offsets), -1)
    for p, perm in enumerate(itertools.permutations(range(3))):
        fits = np.all(canonical[:, list(perm)] == absval, axis=1) & (q_index < 0)
        q_index[fits] = 8 * p + sign_index[fits]
    keys = tuple((int(k[0]), int(k[1]), int(k[2])) for k in keys_arr)
    return offsets, inverse, q_index, keys


@lru_cache(maxsize=8)
def _signed_maps(n_source: int, n_test: int) -> tuple[NDArray, ...] | None:
    """The 48 symmetries' representations as (perm, sign) arrays: PT, ST (48, n_test), PU, SU
    (48, n_source), PS, SS (48, 9); None if any representation is not a signed permutation (then
    ``_fill_rows`` uses the dense transform)."""
    pt: list[NDArray] = []
    st: list[NDArray] = []
    pu: list[NDArray] = []
    su: list[NDArray] = []
    ps: list[NDArray] = []
    ss: list[NDArray] = []
    for q in signed_permutations():
        t, u, s = symmetry_reps(q, n_source, n_test)
        sp = [_signed_permutation(m) for m in (t, u, s)]
        if any(x is None for x in sp):
            return None
        (a, b), (c, e), (f, g) = sp  # type: ignore[misc]
        for store, value in zip((pt, st, pu, su, ps, ss), (a, b, c, e, f, g), strict=True):
            store.append(value)
    return tuple(np.array(x) for x in (pt, st, pu, su, ps, ss))


def _fill_rows(
    real: NDArray,
    sel: slice,
    stack: NDArray,
    inverse: NDArray,
    q_index: NDArray,
    maps: tuple[NDArray, ...] | None,
    na: int,
    nc: int,
    wrapped: tuple[NDArray, NDArray, NDArray],
) -> None:
    """Place rows ``sel`` of every offset's block into ``real`` (rows, cols, npad, npad, npad), vectorised.

    Row r of a block is (test function a, component i) = divmod(r, 9), column (source c, component l), and
    the block at an offset is its canonical block re-indexed with signs:

        K_o[a, c, i, l] = st[a] su[c] ss[i] ss[l] K_canonical[pt[a], pu[c], ps[i], ps[l]],

    (pt, st), (pu, su), (ps, ss) the signed permutations of the offset's symmetry (``_transform``'s fast
    path, the same products in a gather over all offsets at once).
    """
    rows = np.arange(sel.start, sel.stop)
    a_g, i_g = rows // 9, rows % 9
    if maps is None:  # a representation that is not a signed permutation: the dense transform
        qs = signed_permutations()
        for n in range(len(inverse)):
            blk = _transform(qs[q_index[n]], stack[inverse[n]])[:na, :nc].transpose(0, 2, 1, 3)
            blk = blk.reshape(na * 9, nc * 9)
            real[:, :, wrapped[0][n], wrapped[1][n], wrapped[2][n]] = blk[sel]
        return
    pt, st, pu, su, ps, ss = maps
    for n0 in range(0, len(inverse), _FILL_CHUNK):
        sl = slice(n0, min(len(inverse), n0 + _FILL_CHUNK))
        q, inv = q_index[sl], inverse[sl]
        # index arrays broadcast to (g, nc, 9, n): rows, source monomials, column components, offsets
        i0 = inv[None, None, None, :]
        i1 = pt[q][:, a_g].T[:, None, None, :]
        i2 = pu[q][:, :nc].T[None, :, None, :]
        i3 = ps[q][:, i_g].T[:, None, None, :]
        i4 = ps[q].T[None, None, :, :]
        sign = (
            st[q][:, a_g].T[:, None, None, :]
            * su[q][:, :nc].T[None, :, None, :]
            * ss[q][:, i_g].T[:, None, None, :]
            * ss[q].T[None, None, :, :]
        )
        val = sign * stack[i0, i1, i2, i3, i4]  # (g, nc, 9, n)
        real[:, :, wrapped[0][sl], wrapped[1][sl], wrapped[2][sl]] = val.reshape(len(rows), nc * 9, -1)


def _block_jacobi(m9: NDArray, k0: NDArray, e_cells: NDArray) -> NDArray:
    """Each cell's own (M - K(0) E_n)^-1, (N, rows, rows): the block-Jacobi preconditioner of the solve.

    All cells at once, one batched product and one stacked inverse, where a loop over the cells formed and
    inverted each 36 x 36 matrix in turn (8.9 s at n = 32, 17.6 s at n = 40).
    """
    na, rows = m9.shape[0], m9.shape[0] * 9
    ke = np.einsum("acij,ncbjk->naibk", k0, e_cells, optimize=True).reshape(len(e_cells), rows, rows)
    return np.linalg.inv(np.kron(m9, np.eye(9))[None] - ke)


def solve_graded_sphere_fft(
    omega: float,
    radius: float,
    ref: ReferenceMedium,
    contrast: MaterialContrast,
    n_sub: int,
    profile: Callable[[NDArray], float],
    k_hat: NDArray,
    pol: NDArray,
    wave_type: str,
    p: int = 1,
    r: int = 1,
    gmres_tol: float = 1e-11,
    restart: int = 100,
    max_cycles: int = 20,
    blocks: dict | None = None,
    kernel_dtype: type = np.complex128,
    kernel_path: Path | None = None,
    timings: dict[str, float] | None = None,
) -> GradedVoxelResult:
    """The graded-voxel solve of ``solver.solve_graded_sphere``, with an FFT matvec and GMRES.

    Same cells, contrast projection and right-hand side as the dense solver (every cell overlapping the
    sphere is kept).  The transformed kernel is built from the canonical blocks (``transformed_kernel``),
    stored in ``kernel_dtype`` (complex64 halves its memory; the product is then that of a fixed operator
    perturbed by ~1e-7, so GMRES converges as before and the solution moves by about that) and, with
    ``kernel_path``, memory-mapped to that file rather than held in memory.  ``blocks`` (every offset's
    block, ``offset_blocks``) is the former path, kept.  ``timings``, when given, is filled with kernel_s
    (the contrast projection, the blocks, the kernel and the preconditioner), solve_s (GMRES) and products
    (the number of matrix-vector products).

    Raises:
        RuntimeError: when GMRES does not reach ``gmres_tol`` within restart * max_cycles iterations.
    """
    t_start = time.perf_counter()
    h_cell = radius / n_sub
    grid_idx, centres, h = _build_grid_index_map(
        radius,
        n_sub,
        lambda q: bool(np.linalg.norm(q) < radius + np.sqrt(3.0) * h_cell),
    )
    n = len(centres)
    na, n_field, _, n_source = field_sizes(p, r)
    nc = 1 if (p == 0 and r == 0) else n_source
    delta = cell_contrast_coefficients_all(
        profile,
        centres,
        h,
        contrast,
        ref,
        omega,
        degree=r,  # the projection once per sweep
    )
    e_cells = np.array([source_expansion(d, n_field)[:nc, :na] for d in delta])  # (N, nc, na, 9, 9)
    npad = 2 * n_sub - 1
    rows, cols = na * 9, nc * 9
    # the compiled backend stores the kernel at the canonical k only (the cube group in Fourier space);
    # the reference backend, and the every-offset path, the full kernel
    canonical_k = blocks is None and point_kernel_backend() == "fortran"
    if canonical_k:
        # embedded in the next fast odd size (``fast_odd_size``: 81 rather than the prime 79 at n_sub = 40),
        # as a grid of n_eff cells whose blocks beyond the real offsets are zero
        npad = fast_odd_size(2 * n_sub - 1)
        n_eff = (npad + 1) // 2
        canonical = canonical_blocks(n_sub, h, omega, ref, n_source, n_field)
        k0 = canonical[(0, 0, 0)][:na, :nc]
        zero = np.zeros_like(canonical[(0, 0, 0)])
        for key in _orbit_arrays(n_eff)[3]:
            canonical.setdefault(key, zero)
        kh = canonical_transformed_kernel(canonical, n_eff, na, nc, kernel_dtype, kernel_path)
        tables = fourier_symmetry_tables(n_eff, na, nc, n_source, n_field)
        del canonical
    elif blocks is None:
        # the canonical blocks alone; the kernel assembled from them a group of rows at a time
        canonical = canonical_blocks(n_sub, h, omega, ref, n_source, n_field)
        kh = transformed_kernel(canonical, n_sub, na, nc, kernel_dtype, kernel_path)
        k0 = canonical[(0, 0, 0)][:na, :nc]
        del canonical
    else:
        kh = np.zeros((rows, cols, npad, npad, npad), dtype=complex)
        for off, blk in blocks.items():
            kh[:, :, off[0] % npad, off[1] % npad, off[2] % npad] = (
                blk[:na, :nc].transpose(0, 2, 1, 3).reshape(rows, cols)
            )
        for row in range(rows):
            kh[row] = np.fft.fftn(kh[row], axes=(1, 2, 3))
        kh = kh.astype(kernel_dtype, copy=False)
        k0 = blocks[(0, 0, 0)][:na, :nc]
    g0, g1, g2 = grid_idx[:, 0], grid_idx[:, 1], grid_idx[:, 2]
    m9 = gram_test(h, n_field)[:na, :na]
    # the kernel product follows point_kernel.backend: the compiled one, or the NumPy reference
    compiled_product = point_kernel_backend() == "fortran" and kh.flags["C_CONTIGUOUS"]
    if compiled_product:
        # local: kernel_fortran imports kernel
        from .kernel_fortran import apply_canonical_fortran, apply_hat_fortran

    products = [0]

    def matvec(x: NDArray) -> NDArray:
        products[0] += 1
        psi = x.reshape(n, na, 9)
        src = np.einsum("ncbij,nbj->nci", e_cells, psi).reshape(n, cols)
        m9psi = np.einsum("ab,nbi->nai", m9, psi).reshape(n, rows)
        if canonical_k:  # point-major, so that each Fourier point's sources and results are contiguous
            grid = np.zeros((npad, npad, npad, cols), dtype=complex)
            grid[g0, g1, g2] = src
            sh = np.fft.fftn(grid, axes=(0, 1, 2)).reshape(-1, cols)
            yh = apply_canonical_fortran(kh, sh, tables).reshape(npad, npad, npad, rows)
            return (m9psi - np.fft.ifftn(yh, axes=(0, 1, 2))[g0, g1, g2]).ravel()
        grid = np.zeros((cols, npad, npad, npad), dtype=complex)
        grid[:, g0, g1, g2] = src.T
        sh = np.fft.fftn(grid, axes=(1, 2, 3)).reshape(cols, -1)
        if compiled_product:  # one pass over the kernel by all threads, as it lies (kernel_apply.f90)
            yh = apply_hat_fortran(kh, sh)
        else:
            yh = np.empty((rows, sh.shape[1]), dtype=complex)
            for row in range(rows):
                yh[row] = np.einsum("cf,cf->f", np.asarray(kh[row]).reshape(cols, -1), sh)
        y = np.fft.ifftn(yh.reshape(rows, npad, npad, npad), axes=(1, 2, 3))[:, g0, g1, g2].T
        return (m9psi - y).ravel()

    pre = _block_jacobi(m9, k0, e_cells)

    def precondition(x: NDArray) -> NDArray:
        return np.einsum("nij,nj->ni", pre, x.reshape(n, rows)).ravel()

    dim = n * rows
    a_op = LinearOperator((dim, dim), matvec=matvec, dtype=complex)
    m_op = LinearOperator((dim, dim), matvec=precondition, dtype=complex)
    k_mag = omega / (ref.alpha if wave_type == "P" else ref.beta)
    k_hat = np.asarray(k_hat, dtype=float) / np.linalg.norm(k_hat)
    amp = np.concatenate([np.asarray(pol, dtype=complex), _plane_wave_strain_voigt(k_hat, pol, k_mag)])
    rhs = plane_wave_moments(centres, h, k_mag * k_hat, amp, n_field)[:, :na].ravel()
    t_solve = time.perf_counter()
    sol, info = gmres(
        a_op,
        rhs,
        x0=precondition(rhs),
        M=m_op,
        rtol=gmres_tol,
        atol=0.0,
        restart=restart,
        maxiter=max_cycles,
    )
    if info != 0:
        raise RuntimeError(
            f"solve_graded_sphere_fft: GMRES did not reach rtol {gmres_tol:g} "
            f"within {restart * max_cycles} iterations (restart {restart} x {max_cycles} cycles; "
            f"info {info}) at n_sub = {n_sub}. Fix: raise "
            "max_cycles or restart, or loosen gmres_tol."
        )
    if timings is not None:
        timings.update(
            kernel_s=t_solve - t_start, solve_s=time.perf_counter() - t_solve, products=float(products[0])
        )
    full = np.zeros((n, n_field, 9), dtype=complex)
    full[:, :na] = sol.reshape(n, na, 9)
    return GradedVoxelResult(centres, grid_idx, h, omega, ref, delta, full, p, r)
