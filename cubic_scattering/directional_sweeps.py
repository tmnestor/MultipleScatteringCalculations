#!/usr/bin/env python3
"""Direction-pure partial-wave sweeps for the Cartesian G0 matvec.

Coupling WITHIN a depth plane is summed on the k_x pole by a running
accumulation along x; coupling BETWEEN depth planes is summed on the k_z pole in
the lateral wavenumber domain. Every sweep marches along an axis of nonzero
separation, so the same-depth sum that diverges in the conventional k_z-pole
construction is never the one evaluated.

The state is the 9-component (u_z, u_x, u_y, e_zz, e_xx, e_yy, 2e_xy, 2e_zy,
2e_zx) vector throughout. There is no representation conversion at any sweep
boundary.

Coordinates: z = axis 0 (down), x = axis 1 (right), y = axis 2 (out).
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .effective_contrasts import ReferenceMedium
from .sweep_kernels import LateralSplit, lateral_split_9x9, vertical_kernel_9x9


@dataclass(frozen=True)
class SweepGrid:
    """Lattice and quadrature for a 2.5-D sweep at one k_y.

    Attributes:
        n_z: Number of depth planes.
        n_x: Number of sites along x per plane. Must be >= 2.
        pitch: Voxel pitch, km.
        ky: The 2.5-D lateral parameter, 1/km.
        kz_nodes: k_z quadrature nodes, shape (n_kz,).
        kz_weights: k_z quadrature weights INCLUDING the 1/(2 pi), shape (n_kz,).
        kx_nodes: k_x quadrature nodes for the inter-plane sweep, shape (n_kx,).
        kx_weights: k_x quadrature weights INCLUDING the 1/(2 pi), shape (n_kx,).
    """

    n_z: int
    n_x: int
    pitch: float
    ky: float
    kz_nodes: NDArray
    kz_weights: NDArray
    kx_nodes: NDArray
    kx_weights: NDArray


def make_sweep_grid(
    n_z: int,
    n_x: int,
    pitch: float,
    ky: float,
    *,
    kz_max: float | None = None,
    n_kz: int = 2048,
    kx_max: float | None = None,
    n_kx: int = 2048,
) -> SweepGrid:
    """Build the lattice and both quadratures.

    The k_z integrand decays as e^{-|kz| pitch} at the nearest-neighbour
    separation, so the default cutoff is 30/pitch -- about thirteen e-foldings.
    The k_x integrand for the inter-plane sweep decays as e^{-|kx| |dz|} with
    |dz| >= pitch, so the same cutoff serves.

    Always confirm by refinement: an undersized k-grid is this project's most
    expensive recurring numerical error.

    The k_x quadrature is a plain wide grid, NOT an FFT grid, and that is
    deliberate. An FFT along x would sample k_x only on the Nyquist window
    +-pi/pitch -- where the integrand is still ~4% of its peak at |dz| = pitch,
    so the tail is not negligible -- and would periodize the real-space kernel
    at the domain width, which is precisely the horizontal periodicity the real
    Earth does not have. A direct quadrature has neither defect. An aliased-FFT
    accelerator can be added later and gated against this; correctness first.

    Args:
        n_z: Number of depth planes (>= 1).
        n_x: Sites along x (>= 2).
        pitch: Voxel pitch in km (> 0).
        ky: The 2.5-D lateral wavenumber, 1/km.
        kz_max: k_z quadrature cutoff. Defaults to 30/pitch.
        n_kz: Number of k_z nodes.
        kx_max: k_x quadrature cutoff. Defaults to 30/pitch.
        n_kx: Number of k_x nodes.

    Returns:
        A SweepGrid.

    Raises:
        ValueError: on a lattice too small to sweep, or a non-positive pitch.
    """
    if n_x < 2:
        msg = (
            f"n_x must be >= 2 for a lateral sweep, got {n_x}.\n"
            "  Where: cubic_scattering/directional_sweeps.py, make_sweep_grid(n_x=...)\n"
            "  Valid: n_x=2 or more, e.g. n_x=32\n"
            "  Fix:   a single-site row has no lateral coupling at all -- drop the\n"
            "         lateral sweep for that model rather than running it on one site."
        )
        raise ValueError(msg) from None
    if n_z < 1:
        msg = (
            f"n_z must be >= 1, got {n_z}.\n"
            "  Where: cubic_scattering/directional_sweeps.py, make_sweep_grid(n_z=...)\n"
            "  Valid: n_z=1 or more, e.g. n_z=4\n"
            "  Fix:   pass the number of depth planes in the model."
        )
        raise ValueError(msg) from None
    if pitch <= 0.0:
        msg = (
            f"pitch must be > 0, got {pitch!r}.\n"
            "  Where: cubic_scattering/directional_sweeps.py, make_sweep_grid(pitch=...)\n"
            "  Valid: a positive voxel pitch in km, e.g. pitch=0.25\n"
            "  Fix:   use the cube side length d = 2a, not the half-width a."
        )
        raise ValueError(msg) from None

    def _trapezoid(cutoff: float, count: int) -> tuple[NDArray, NDArray]:
        nodes = np.linspace(-cutoff, cutoff, count)
        dk = nodes[1] - nodes[0]
        weights = np.full(count, dk / (2.0 * np.pi))
        weights[0] *= 0.5
        weights[-1] *= 0.5
        return nodes, weights

    kz_cut = 30.0 / pitch if kz_max is None else float(kz_max)
    kx_cut = 30.0 / pitch if kx_max is None else float(kx_max)
    kz_nodes, kz_weights = _trapezoid(kz_cut, n_kz)
    kx_nodes, kx_weights = _trapezoid(kx_cut, n_kx)

    return SweepGrid(
        n_z=n_z,
        n_x=n_x,
        pitch=float(pitch),
        ky=float(ky),
        kz_nodes=kz_nodes,
        kz_weights=kz_weights,
        kx_nodes=kx_nodes,
        kx_weights=kx_weights,
    )


def _check_split(grid: SweepGrid, split: LateralSplit, name: str) -> None:
    """Reject a split built for a different grid than the one it will sweep."""
    if abs(split.pitch - grid.pitch) > 1e-12 * max(1.0, grid.pitch):
        msg = (
            f"{name} was built for pitch={split.pitch!r} but the grid has "
            f"pitch={grid.pitch!r}.\n"
            "  Where: cubic_scattering/directional_sweeps.py, sweep_x\n"
            "  Valid: the split and the grid must share one pitch, e.g. both 0.25\n"
            "  Fix:   build the split with lateral_split_9x9(..., grid.pitch, ...)."
        )
        raise ValueError(msg) from None
    if split.amp_p.shape[2] != grid.kz_nodes.size:
        msg = (
            f"{name} has {split.amp_p.shape[2]} k_z nodes, the grid has "
            f"{grid.kz_nodes.size}.\n"
            "  Where: cubic_scattering/directional_sweeps.py, sweep_x\n"
            "  Valid: identical node counts, e.g. both 2048\n"
            "  Fix:   build the split with lateral_split_9x9(grid.ky, grid.kz_nodes, ...)."
        )
        raise ValueError(msg) from None


def _check_state(sources: NDArray, grid: SweepGrid, where: str) -> None:
    """Reject a state array that is not (n_z, n_x, 9)."""
    expect = (grid.n_z, grid.n_x, 9)
    if sources.shape != expect:
        msg = (
            f"sources has shape {sources.shape}, expected {expect}.\n"
            f"  Where: cubic_scattering/directional_sweeps.py, {where}(sources=...)\n"
            "  Valid: a complex array of shape (n_z, n_x, 9)\n"
            "  Fix:   reshape the solver state before the matvec; the trailing axis is\n"
            "         the 9-component (u, e_Voigt) state, not 3 or 6."
        )
        raise ValueError(msg) from None


def sweep_x(
    sources: NDArray,
    grid: SweepGrid,
    split_right: LateralSplit,
    split_left: LateralSplit,
) -> NDArray:
    """Accumulate intra-plane lateral coupling by two running sweeps.

    Reads the accumulator BEFORE adding the local source, so the minimum
    separation is one pitch and the self-term is never formed. Unrolled, the
    right pass gives out[i] = sum_{j<i} Phi^{i-j} s[j].

    Args:
        sources: Source 9-vectors, shape (n_z, n_x, 9).
        grid: The lattice and k_z quadrature.
        split_right: Amplitude/phase split for +x, direction='right'.
        split_left: Amplitude/phase split for -x, direction='left'.

    Returns:
        The accumulated field, shape (n_z, n_x, 9).

    Raises:
        ValueError: on a grid/split mismatch, two splits of the same direction,
            or a wrong source shape.
    """
    _check_split(grid, split_right, "split_right")
    _check_split(grid, split_left, "split_left")
    if split_right.direction != "right" or split_left.direction != "left":
        msg = (
            f"split directions are ({split_right.direction!r}, {split_left.direction!r}), "
            "expected ('right', 'left').\n"
            "  Where: cubic_scattering/directional_sweeps.py, sweep_x\n"
            "  Valid: lateral_split_9x9(..., direction='right') and '...left'\n"
            "  Fix:   passing the same split twice silently symmetrises the field."
        )
        raise ValueError(msg) from None
    _check_state(sources, grid, "sweep_x")

    out = np.zeros((grid.n_z, grid.n_x, 9), dtype=complex)
    w = grid.kz_weights
    n_kz = grid.kz_nodes.size

    passes = (
        (split_right, range(grid.n_x)),
        (split_left, reversed(range(grid.n_x))),
    )
    for split, order in passes:
        acc_p = np.zeros((grid.n_z, n_kz, 9), dtype=complex)
        acc_s = np.zeros_like(acc_p)
        for i in order:
            out[:, i, :] += np.einsum("k,abk,zkb->za", w, split.amp_p, acc_p)
            out[:, i, :] += np.einsum("k,abk,zkb->za", w, split.amp_s, acc_s)
            acc_p = (acc_p + sources[:, i, None, :]) * split.phase_p[None, :, None]
            acc_s = (acc_s + sources[:, i, None, :]) * split.phase_s[None, :, None]

    return out


@dataclass(frozen=True)
class LayeredBackground:
    """A stratified background for the vertical sweep.

    Attributes:
        model: A ``LayerModel`` whose layer boundaries include one interface per
            depth plane. Layers between consecutive planes are one pitch thick.
        plane_ifaces: Interface index of each depth plane, in order, length n_z.
            Planes must lie in layer INTERIORS (a pseudo-interface with no
            material contrast counts as interior); ``corrected_layered_9x9``
            enforces that, because the correction operator K is built from the
            local vertical S slowness and is two-valued on a material jump.
    """

    model: object
    plane_ifaces: tuple[int, ...]


def build_vertical_stack_layered(grid: SweepGrid, background: LayeredBackground, omega: complex) -> NDArray:
    """Plane-to-plane kernels for a STRATIFIED background.

    This is the production vertical operator. It is the thesis Chapter 5
    stratified propagator ``Q^d = (I - S_int E)^-1 S_int`` (Eq. PstratDef),
    reached through the already-validated route: the Riccati layered Green's
    function, plus the three wrapper corrections D1/D2/D3.

    ``build_vertical_stack`` is the whole-space special case, retained as the
    homogeneous-limit arbiter: with a uniform model and the plane pair taken
    deep enough that the free surface is attenuated, the two agree to ~1e-15.

    Args:
        grid: The lattice and both quadratures.
        background: The stratified model and the plane-to-interface map.
        omega: Angular frequency (rad/s).

    Returns:
        Array of shape (n_z, n_z, 9, 9, n_kx); the diagonal is zero.

    Raises:
        ValueError: if the plane map does not have one entry per depth plane.
    """
    if len(background.plane_ifaces) != grid.n_z:
        msg = (
            f"plane_ifaces has {len(background.plane_ifaces)} entries but the grid has "
            f"n_z={grid.n_z} depth planes.\n"
            "  Where: cubic_scattering/directional_sweeps.py, build_vertical_stack_layered\n"
            "  Valid: one interface index per plane, e.g. plane_ifaces=(1, 2, 3, 4)\n"
            "  Fix:   build the LayerModel with one layer per inter-plane gap, each of\n"
            "         thickness grid.pitch, and list the interface at each plane."
        )
        raise ValueError(msg) from None

    from .layered_correction import corrected_layered_9x9

    n_kx = grid.kx_nodes.size
    ky_arr = np.full(n_kx, grid.ky)
    out = np.zeros((grid.n_z, grid.n_z, 9, 9, n_kx), dtype=complex)
    for lz in range(grid.n_z):
        for mz in range(grid.n_z):
            if lz == mz:
                continue
            g9 = corrected_layered_9x9(
                background.model,
                omega,
                grid.kx_nodes,
                ky_arr,
                source_iface=background.plane_ifaces[mz],
                receiver_iface=background.plane_ifaces[lz],
            )
            out[lz, mz] = np.moveaxis(g9, 0, -1)
    return out


def build_vertical_stack(grid: SweepGrid, ref: ReferenceMedium, omega: complex) -> NDArray:
    """Whole-space plane-to-plane kernels -- the HOMOGENEOUS-LIMIT case.

    For a stratified background use ``build_vertical_stack_layered``. This
    function is retained because it is an independent construction (k_z residue
    of the whole-space Green's tensor, gated against the closed-form Kupradze
    propagator) and therefore arbitrates the stratified one in the limit.

    Args:
        grid: The lattice and both quadratures.
        ref: Background medium.
        omega: Complex angular frequency.

    Returns:
        Array of shape (n_z, n_z, 9, 9, n_kx). The diagonal [lz, lz] is left
        zero: same-plane coupling belongs to sweep_x, and the kernel is not even
        defined at dz = 0.
    """
    n_kx = grid.kx_nodes.size
    out = np.zeros((grid.n_z, grid.n_z, 9, 9, n_kx), dtype=complex)
    for lz in range(grid.n_z):
        for mz in range(grid.n_z):
            if lz == mz:
                continue
            out[lz, mz] = vertical_kernel_9x9(grid.kx_nodes, grid.ky, (lz - mz) * grid.pitch, omega, ref)
    return out


def sweep_z(sources: NDArray, grid: SweepGrid, vertical: NDArray) -> NDArray:
    """Accumulate inter-plane coupling through the lateral wavenumber domain.

    For every ordered pair of DISTINCT planes, sums

        out[lz, i] = sum_{mz != lz} sum_j K_{lz,mz}((i - j) pitch) s[mz, j]

    with K the inverse k_x transform of the plane-to-plane kernel. The sum over
    j is carried in the k_x domain, so the cost is O(n_z^2 n_kx + n_z n_x n_kx)
    rather than O(n_z^2 n_x^2), while remaining an exact quadrature of the
    continuum integral -- there is no transform of finite period anywhere, hence
    no lateral periodicity.

    Args:
        sources: Source 9-vectors, shape (n_z, n_x, 9).
        grid: The lattice and both quadratures.
        vertical: Kernels from build_vertical_stack, (n_z, n_z, 9, 9, n_kx).

    Returns:
        The accumulated field, shape (n_z, n_x, 9).

    Raises:
        ValueError: on a wrong source shape or a grid/kernel mismatch.
    """
    _check_state(sources, grid, "sweep_z")
    n_kx = grid.kx_nodes.size
    if vertical.shape != (grid.n_z, grid.n_z, 9, 9, n_kx):
        msg = (
            f"vertical has shape {vertical.shape}, expected "
            f"{(grid.n_z, grid.n_z, 9, 9, n_kx)}.\n"
            "  Where: cubic_scattering/directional_sweeps.py, sweep_z(vertical=...)\n"
            "  Valid: the array returned by build_vertical_stack(grid, ref, omega)\n"
            "  Fix:   rebuild the stack from the SAME grid you are sweeping."
        )
        raise ValueError(msg) from None

    # Forward phase e^{-i kx x_j} and its conjugate for the readout.
    x = np.arange(grid.n_x) * grid.pitch
    phase = np.exp(-1j * np.outer(grid.kx_nodes, x))  # (n_kx, n_x)

    # Source spectra, one per plane: (n_z, n_kx, 9)
    spec = np.einsum("kj,zjb->zkb", phase, sources)

    acc = np.zeros((grid.n_z, n_kx, 9), dtype=complex)
    for lz in range(grid.n_z):
        for mz in range(grid.n_z):
            if lz == mz:
                continue
            acc[lz] += np.einsum("abk,kb->ka", vertical[lz, mz], spec[mz])

    return np.einsum("k,kj,zka->zja", grid.kx_weights, np.conj(phase), acc)


@dataclass(frozen=True)
class G0Cache:
    """Everything G0 needs that does not change across Krylov iterations.

    Attributes:
        grid: The lattice and both quadratures.
        split_right: Lateral amplitude/phase split, +x.
        split_left: Lateral amplitude/phase split, -x.
        vertical: Plane-to-plane kernels, shape (n_z, n_z, 9, 9, n_kx).
    """

    grid: SweepGrid
    split_right: LateralSplit
    split_left: LateralSplit
    vertical: NDArray


def build_g0_cache(
    grid: SweepGrid,
    ref: ReferenceMedium,
    omega: complex,
    *,
    background: LayeredBackground | None = None,
) -> G0Cache:
    """Build every direction's kernels once, outside the Krylov loop.

    Args:
        grid: The lattice and both quadratures.
        ref: Background medium for the LATERAL sweep.
        omega: Complex angular frequency.
        background: Stratified background for the vertical sweep. When omitted,
            the vertical sweep uses the whole-space kernel built from ``ref``.

    Returns:
        A G0Cache.

    Note:
        Stage 1 dresses only the VERTICAL sweep with the layering. Same-depth
        lateral coupling still goes through the whole-space kernel, so layer
        reverberations between two voxels in one plane are missing. That is
        small for a plane well inside a layer and NOT small next to a strong
        interface. ``layered_correction`` holds the machinery to close it.
    """
    vertical = (
        build_vertical_stack(grid, ref, omega)
        if background is None
        else build_vertical_stack_layered(grid, background, omega)
    )
    return G0Cache(
        grid=grid,
        split_right=lateral_split_9x9(grid.ky, grid.kz_nodes, grid.pitch, omega, ref, direction="right"),
        split_left=lateral_split_9x9(grid.ky, grid.kz_nodes, grid.pitch, omega, ref, direction="left"),
        vertical=vertical,
    )


def apply_g0(sources: NDArray, cache: G0Cache) -> NDArray:
    """Apply the full G0: intra-plane lateral plus inter-plane vertical.

    A pure forward summation -- no inversion, no embedding, no reverberation.
    Every order of multiple scattering is built by the Krylov iterations, not by
    the propagator.

    The two sweeps partition the off-diagonal pairs exactly: sweep_x reaches
    same-plane pairs only, sweep_z different-plane pairs only, and neither
    reaches the site itself (its self-term is closed inside T0).

    Args:
        sources: Source 9-vectors, shape (n_z, n_x, 9).
        cache: Precomputed kernels from build_g0_cache.

    Returns:
        The field at every site, shape (n_z, n_x, 9), excluding the self-term.
    """
    return sweep_x(sources, cache.grid, cache.split_right, cache.split_left) + sweep_z(
        sources, cache.grid, cache.vertical
    )


def sweep_y(sources: NDArray, grid: SweepGrid, cache: G0Cache) -> NDArray:
    """In-out (k_y) sweep. Stage 2 -- not implemented in stage 1.

    Raises:
        NotImplementedError: always, naming the stage. Silently omitting the
            in-out coupling would discard every out-of-plane path while still
            returning a plausible field, which is far worse than failing.
    """
    msg = (
        "sweep_y is stage 2 of the directional-sweep design and is not implemented.\n"
        "  Where: cubic_scattering/directional_sweeps.py, sweep_y\n"
        "  Valid: stage 1 is 2.5-D -- heterogeneity in (z, x), y invariant, with k_y\n"
        "         held fixed on the SweepGrid and integrated over afterwards.\n"
        "  Fix:   solve once per k_y with solve_sweep_foldy_lax and integrate, or\n"
        "         implement the in-out pair (spec section 6, stage 2). Do NOT drop\n"
        "         the term: omitting it silently discards all out-of-plane coupling."
    )
    raise NotImplementedError(msg) from None
