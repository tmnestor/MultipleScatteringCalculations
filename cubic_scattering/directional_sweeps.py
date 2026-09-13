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

from .sweep_kernels import LateralSplit


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
    """

    n_z: int
    n_x: int
    pitch: float
    ky: float
    kz_nodes: NDArray
    kz_weights: NDArray


def make_sweep_grid(
    n_z: int,
    n_x: int,
    pitch: float,
    ky: float,
    *,
    kz_max: float | None = None,
    n_kz: int = 2048,
) -> SweepGrid:
    """Build the lattice and the k_z quadrature.

    The k_z integrand decays as e^{-|kz| pitch} at the nearest-neighbour
    separation, so the default cutoff is 30/pitch -- about thirteen e-foldings.
    Always confirm by refinement: an undersized k-grid is this project's most
    expensive recurring numerical error.

    Args:
        n_z: Number of depth planes (>= 1).
        n_x: Sites along x (>= 2).
        pitch: Voxel pitch in km (> 0).
        ky: The 2.5-D lateral wavenumber, 1/km.
        kz_max: Quadrature cutoff. Defaults to 30/pitch.
        n_kz: Number of nodes.

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

    cutoff = 30.0 / pitch if kz_max is None else float(kz_max)
    nodes = np.linspace(-cutoff, cutoff, n_kz)
    dk = nodes[1] - nodes[0]
    weights = np.full(n_kz, dk / (2.0 * np.pi))
    weights[0] *= 0.5
    weights[-1] *= 0.5

    return SweepGrid(
        n_z=n_z,
        n_x=n_x,
        pitch=float(pitch),
        ky=float(ky),
        kz_nodes=nodes,
        kz_weights=weights,
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
