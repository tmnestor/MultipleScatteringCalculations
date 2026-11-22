"""A matrix-free product for cells of several sizes: the octree's Galerkin operator through one FFT.

``octree.solve_graded_octree`` assembles the system of a tree densely, because the convolution structure
of a uniform grid is lost when the cells differ in size. It is not lost one level down. A leaf of
half-width H = 2^j h divides into 8^j descendants of the smallest half-width h of the tree, and on each
descendant d the leaf's polynomials are polynomials of the same degree in the descendant's coordinate
(``octree.field_reexpansion``, ``octree.source_reexpansion``):

    Q_a(leaf) = sum_b C_d[a, b] Q_b(d),        m_c(leaf) = sum_f D_d[c, f] m_f(d).

The block between any two leaves m and n is therefore a finite sum of blocks between equal cells of the
smallest size,

    K_ac(m, n) = sum over d in m, e in n of  sum_bf C_d[a, b] K_bf(d - e) D_e[c, f],

and K(d - e) depends on the offset alone. So the product of the tree's operator with a vector is

    spread:   the source coefficients s_n = E_n psi_n of each leaf go to its descendants, s_e = D_e^T s_n;
    convolve: y_d = sum_e K(d - e) s_e on the uniform lattice of the smallest cells, by FFT, with the
              kernel and the compiled products of ``fft``;
    gather:   y_m = sum over d in m of C_d y_d;

and the result is M psi_m - y_m. Nothing is approximated: where
the dense solver takes the block between two large leaves from the closed forms at their own size, this
product sums the same integral over their descendants, and the two agree to round-off (4e-14 of the
coupling on a tree of three sizes).

WHAT IT COSTS, AND WHAT IT DOES NOT SAVE. The unknowns are the tree's, but each product is an FFT on the
uniform lattice of the smallest cells over the bounding cube of the tree, so its time and the memory of
the kernel are those of the uniform grid at the smallest size. What is gained over the dense solver is
reach: the dense matrix grows as the square of the number of leaves and its solve as the cube, and stops
near 2200 constant leaves. What is not gained is a product cheaper than the fine uniform grid's.

THE SOLVE (``solve_graded_octree_fft``). GMRES with a block-Jacobi preconditioner, each leaf's own
(M - K(0; H) E_n)^-1 with the self block at the leaf's own size. ``born_series_octree_fft`` gives the
terms of the scheme's Born series with no solve, one product each, as ``octree.born_series_octree``.
One field degree and one contrast degree for the whole tree.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator, gmres

from ..effective_contrasts import MaterialContrast, ReferenceMedium
from ..sphere_scattering import _plane_wave_strain_voigt
from ..sphere_scattering_fft import fast_odd_size
from .basis import gram_test, source_expansion
from .fft import (
    canonical_blocks,
    canonical_transformed_kernel,
    fourier_symmetry_tables,
    transformed_kernel,
)
from .kernel import point_kernel_backend
from .octree import (
    EqualBlocks,
    OctreeResult,
    _descendants,
    _levels_apart,
    field_reexpansion,
    source_reexpansion,
)
from .site import cell_contrast_coefficients
from .solver import field_sizes, plane_wave_moments


@dataclass
class OctreeOperator:
    """The Galerkin operator (M - K E) of a tree, matrix-free, with what a solve or a series needs."""

    matvec: Callable[[NDArray], NDArray]  # (M - K E) x
    precondition: Callable[[NDArray], NDArray]  # block-Jacobi, each leaf's own inverse
    mass: Callable[[NDArray, bool], NDArray]  # M x, or M^-1 x when the flag is set
    rhs: NDArray  # the moments of the incident state
    delta: NDArray  # (N, 4 or 10, 9, 9): contrast coefficients of the leaves
    na: int
    n_field: int
    n_fine: int  # cells of the smallest size across the bounding cube
    products: list  # [number of products so far]


def _fine_product(
    n_sub: int, h: float, omega: float, ref: ReferenceMedium, p: int, r: int, kernel_dtype: type
) -> tuple[Callable[[NDArray], NDArray], int]:
    """The convolution with K on the uniform lattice of n_sub^3 cells of half-width h.

    Returns (apply, npad): apply takes the sources on the padded grid, (npad, npad, npad, cols), and
    returns the fields, (npad, npad, npad, rows). The kernel is built as ``fft.solve_graded_sphere_fft``
    builds it: at the canonical wavenumbers with the compiled product where that backend is present, in
    full otherwise.
    """
    na, n_field, _, n_source = field_sizes(p, r)
    nc = 1 if (p == 0 and r == 0) else n_source
    rows, cols = na * 9, nc * 9
    canonical = canonical_blocks(n_sub, h, omega, ref, n_source, n_field)
    if point_kernel_backend() == "fortran":
        from .kernel_fortran import apply_canonical_fortran  # local: kernel_fortran imports kernel

        npad = fast_odd_size(2 * n_sub - 1)
        n_eff = (npad + 1) // 2
        zero = np.zeros_like(canonical[(0, 0, 0)])
        for a in range(n_eff):
            for b in range(a + 1):
                for d in range(b + 1):
                    canonical.setdefault((a, b, d), zero)
        canonical = dict(sorted(canonical.items()))
        kh = canonical_transformed_kernel(canonical, n_eff, na, nc, kernel_dtype)
        tables = fourier_symmetry_tables(n_eff, na, nc, n_source, n_field)

        def apply(grid: NDArray) -> NDArray:
            sh = np.fft.fftn(grid, axes=(0, 1, 2)).reshape(-1, cols)
            yh = apply_canonical_fortran(kh, sh, tables).reshape(npad, npad, npad, rows)
            return np.fft.ifftn(yh, axes=(0, 1, 2))

        return apply, npad
    npad = 2 * n_sub - 1
    kh = transformed_kernel(canonical, n_sub, na, nc, kernel_dtype)

    def apply_full(grid: NDArray) -> NDArray:
        sh = np.fft.fftn(grid, axes=(0, 1, 2)).reshape(-1, cols)
        yh = np.einsum("rcf,fc->fr", np.asarray(kh).reshape(rows, cols, -1), sh)
        return np.fft.ifftn(yh.reshape(npad, npad, npad, rows), axes=(0, 1, 2))

    return apply_full, npad


def octree_operator_fft(
    omega: float,
    ref: ReferenceMedium,
    contrast: MaterialContrast,
    centres: NDArray,
    half_widths: NDArray,
    profile: Callable[[NDArray], float],
    k_hat: NDArray,
    pol: NDArray,
    wave_type: str,
    p: int = 1,
    r: int = 1,
    kernel_dtype: type = np.complex128,
) -> OctreeOperator:
    """Build the matrix-free operator of a tree; arguments as ``octree.solve_graded_octree``, but with one
    field degree p and one contrast degree r for every leaf.

    Raises:
        ValueError: when the half-widths are not in ratios 2^j, or the leaves are not on the lattice of
            the smallest cells.
    """
    centres = np.asarray(centres, dtype=float)
    half_widths = np.asarray(half_widths, dtype=float)
    n = len(centres)
    na, n_field, _, n_source = field_sizes(p, r)
    nc = 1 if (p == 0 and r == 0) else n_source
    rows, cols = na * 9, nc * 9
    h = float(half_widths.min())
    lo = (centres - half_widths[:, None]).min(axis=0)
    extent = float((centres + half_widths[:, None] - lo).max())
    n_fine = round(extent / (2.0 * h))
    apply, npad = _fine_product(n_fine, h, omega, ref, p, r, kernel_dtype)

    delta = np.array(
        [
            cell_contrast_coefficients(profile, c, float(hw), contrast, ref, omega, degree=r)
            for c, hw in zip(centres, half_widths, strict=True)
        ]
    )
    e_cells = np.array([source_expansion(d, n_field)[:nc, :na] for d in delta])  # (N, nc, na, 9, 9)

    # the leaves by size: their descendants on the fine lattice, and the two re-expansions
    equal = EqualBlocks(omega, ref, n_source, n_field)
    levels = []
    gram_of = np.empty((n, na, na))
    pre = np.empty((n, rows, rows), dtype=complex)
    rhs = np.empty((n, na, 9), dtype=complex)
    k_mag = omega / (ref.alpha if wave_type == "P" else ref.beta)
    k_unit = np.asarray(k_hat, dtype=float) / np.linalg.norm(k_hat)
    amp = np.concatenate([np.asarray(pol, dtype=complex), _plane_wave_strain_voigt(k_unit, pol, k_mag)])
    for size in np.unique(half_widths):
        members = np.flatnonzero(half_widths == size)
        j = _levels_apart(float(size), h)
        kids0, _ = _descendants(np.zeros(3), float(size), j)  # (8^j, 3), relative to the leaf's centre
        shifts = kids0 / float(size)
        c_mats = np.array(
            [field_reexpansion(n_field, h / float(size), tuple(map(float, s)))[:na, :na] for s in shifts]
        )
        d_mats = np.array(
            [source_reexpansion(n_source, h / float(size), tuple(map(float, s)))[:nc, :nc] for s in shifts]
        )
        pos = (centres[members][:, None, :] + kids0[None, :, :] - h - lo) / (2.0 * h)
        idx = np.round(pos).astype(int)
        if np.abs(pos - idx).max() > 1e-6 or idx.min() < 0 or idx.max() >= n_fine:
            raise ValueError(
                f"octree_operator_fft: leaves of half-width {size:g} do not lie on the lattice of the "
                f"smallest cells (half-width {h:g}); the worst is {np.abs(pos - idx).max():.3g} of a cell "
                "off it. Fix: pass the leaves of an octree, each cell's centre an odd number of its own "
                "half-widths from the faces of the root cube."
            )
        levels.append((members, idx, c_mats, d_mats))
        m9 = gram_test(float(size), n_field)[:na, :na]
        gram_of[members] = m9
        k0 = equal((0, 0, 0), float(size))[:na, :nc]
        ke = np.einsum("acij,ncbjk->naibk", k0, e_cells[members], optimize=True).reshape(-1, rows, rows)
        pre[members] = np.linalg.inv(np.kron(m9, np.eye(9))[None] - ke)
        rhs[members] = plane_wave_moments(centres[members], float(size), k_mag * k_unit, amp, n_field)[
            :, :na
        ]

    products = [0]

    def matvec(x: NDArray) -> NDArray:
        products[0] += 1
        psi = x.reshape(n, na, 9)
        src = np.einsum("ncbij,nbj->nci", e_cells, psi)  # (N, nc, 9)
        grid = np.zeros((npad, npad, npad, cols), dtype=complex)
        for members, idx, _, d_mats in levels:
            fine = np.einsum("kcf,nci->nkfi", d_mats, src[members]).reshape(len(members), -1, cols)
            grid[idx[..., 0], idx[..., 1], idx[..., 2]] = fine
        field = apply(grid)
        out = np.einsum("nab,nbi->nai", gram_of, psi)
        for members, idx, c_mats, _ in levels:
            fine = field[idx[..., 0], idx[..., 1], idx[..., 2]].reshape(len(members), -1, na, 9)
            out[members] -= np.einsum("kab,nkbi->nai", c_mats, fine)
        return out.ravel()

    def precondition(x: NDArray) -> NDArray:
        return np.einsum("nij,nj->ni", pre, x.reshape(n, rows)).ravel()

    gram_inv = np.linalg.inv(gram_of)

    def mass(x: NDArray, inverse: bool) -> NDArray:
        return np.einsum("nab,nbi->nai", gram_inv if inverse else gram_of, x.reshape(n, na, 9)).ravel()

    return OctreeOperator(matvec, precondition, mass, rhs.ravel(), delta, na, n_field, n_fine, products)


def _result(
    op: OctreeOperator,
    coeffs: NDArray,
    centres: NDArray,
    half_widths: NDArray,
    omega: float,
    ref: ReferenceMedium,
    p: int,
    r: int,
) -> OctreeResult:
    n = len(centres)
    full = np.zeros((n, op.n_field, 9), dtype=complex)
    full[:, : op.na] = coeffs.reshape(n, op.na, 9)
    return OctreeResult(
        np.asarray(centres, dtype=float),
        np.asarray(half_widths, dtype=float),
        omega,
        ref,
        op.delta,
        full,
        p,
        r,
    )


def solve_graded_octree_fft(
    omega: float,
    ref: ReferenceMedium,
    contrast: MaterialContrast,
    centres: NDArray,
    half_widths: NDArray,
    profile: Callable[[NDArray], float],
    k_hat: NDArray,
    pol: NDArray,
    wave_type: str,
    p: int = 1,
    r: int = 1,
    gmres_tol: float = 1e-11,
    restart: int = 100,
    max_cycles: int = 20,
    kernel_dtype: type = np.complex128,
    timings: dict[str, float] | None = None,
    operator: OctreeOperator | None = None,
) -> OctreeResult:
    """The solve of ``octree.solve_graded_octree`` without the dense matrix: an FFT product and GMRES.

    Args:
        omega: Angular frequency (rad/s).
        ref: Background medium.
        contrast: The contrast that ``profile`` scales.
        centres: Leaf centres, shape (N, 3) (m).
        half_widths: Leaf half-widths, shape (N,) (m), in ratios 2^j.
        profile: Scalar factor of the contrast as a function of position.
        k_hat: Incident direction.
        pol: Incident polarisation.
        wave_type: 'P' or 'S' (the incident speed).
        p: Field degree (0, 1 or 2), the same for every leaf.
        r: Contrast degree (0, 1 or 2), the same for every leaf.
        gmres_tol: Relative residual at which GMRES stops.
        restart: GMRES restart length.
        max_cycles: Restart cycles allowed (scipy's ``maxiter`` counts cycles, not iterations).
        kernel_dtype: complex128, or complex64 to halve the kernel's memory.
        timings: When given, filled with kernel_s, solve_s, products and n_fine.
        operator: The operator of this same tree, wave and contrast, already built
            (``octree_operator_fft``), so that a series and a solve share one kernel.

    Raises:
        ValueError: as ``octree_operator_fft``.
        RuntimeError: when GMRES does not reach ``gmres_tol`` within restart * max_cycles iterations.
    """
    t_start = time.perf_counter()
    op = operator or octree_operator_fft(
        omega, ref, contrast, centres, half_widths, profile, k_hat, pol, wave_type, p, r, kernel_dtype
    )
    dim = len(op.rhs)
    t_solve = time.perf_counter()
    sol, info = gmres(
        LinearOperator((dim, dim), matvec=op.matvec, dtype=complex),
        op.rhs,
        x0=op.precondition(op.rhs),
        M=LinearOperator((dim, dim), matvec=op.precondition, dtype=complex),
        rtol=gmres_tol,
        atol=0.0,
        restart=restart,
        maxiter=max_cycles,
    )
    if info != 0:
        raise RuntimeError(
            f"solve_graded_octree_fft: GMRES did not reach rtol {gmres_tol:g} within "
            f"{restart * max_cycles} iterations (restart {restart} x {max_cycles} cycles; info {info}) on "
            f"{len(centres)} leaves. Fix: raise max_cycles or restart, or loosen gmres_tol."
        )
    if timings is not None:
        timings.update(
            kernel_s=t_solve - t_start,
            solve_s=time.perf_counter() - t_solve,
            products=float(op.products[0]),
            n_fine=float(op.n_fine),
        )
    return _result(op, sol, centres, half_widths, omega, ref, p, r)


def born_series_octree_fft(
    omega: float,
    ref: ReferenceMedium,
    contrast: MaterialContrast,
    centres: NDArray,
    half_widths: NDArray,
    profile: Callable[[NDArray], float],
    k_hat: NDArray,
    pol: NDArray,
    wave_type: str,
    orders: int,
    p: int = 1,
    r: int = 1,
    kernel_dtype: type = np.complex128,
    operator: OctreeOperator | None = None,
) -> list[OctreeResult]:
    """The scheme's Born series on a tree, term by term, with no solve and no dense matrix.

    psi_1 = M^-1 b and psi_{n+1} = M^-1 (K E) psi_n with K E psi = M psi - (M - K E) psi, one product
    each, as ``octree.born_series_octree``. Arguments as ``solve_graded_octree_fft``; ``orders`` is the
    number of terms returned, and ``operator`` an operator of this tree already built.

    Raises:
        ValueError: when orders < 1, or as ``octree_operator_fft``.
    """
    if orders < 1:
        raise ValueError(
            f"born_series_octree_fft: orders = {orders}; at least one term is needed.\n"
            "  Fix: pass orders >= 1 (orders = 2 gives the Born term and the term of second order)."
        )
    op = operator or octree_operator_fft(
        omega, ref, contrast, centres, half_widths, profile, k_hat, pol, wave_type, p, r, kernel_dtype
    )
    terms = [op.mass(op.rhs, True)]
    for _ in range(orders - 1):
        prev = terms[-1]
        terms.append(op.mass(op.mass(prev, False) - op.matvec(prev), True))
    return [_result(op, t, centres, half_widths, omega, ref, p, r) for t in terms]
