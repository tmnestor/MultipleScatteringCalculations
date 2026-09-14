#!/usr/bin/env python3
"""Real-space pair-propagator tables for the three-dimensional solver.

WHY THESE EXIST, since they replace something that looked more elegant. Stage 1
carries in-plane coupling as a running spectral sweep, which works because the
2.5-D accumulator is (n_z, n_kz, 9) -- a few megabytes. In three dimensions the
same construction needs a 2-D transverse quadrature and the accumulator becomes
(n_z, n_x, n_k, 9): 68 GB on a 16^3 lattice at the converged rule, with two live
at once, and 615 GB for the inter-plane stack. Measured, not estimated --
`scripts/measure_sweep3d_cost.py`.

The same operators tabulated in REAL SPACE are 1.2 MB and 149 MB. So the
architecture in 3-D is:

  * Dz = 0 coupling -- the closed-form whole-space propagator, tabulated by
    separation. Exact, no quadrature at all. That is this module.
  * Dz != 0 coupling -- the stratified propagator, which exists only in
    (k_x, k_y), transformed to real space ONCE at build time. Also this module,
    in `inter_plane_table`.

A table is indexed by SEPARATION, so it holds (2 n_x - 1) x (2 n_y - 1) entries
for a lattice of n_x by n_y, with the zero-separation entry poisoned to NaN. The
poisoning is deliberate: the self-term belongs inside T0 and must be reached by
no propagator, and a zero there would let a partition bug contribute silently
instead of failing where it happens.

Conventions. Seismic units (km/s, g/cm3, GPa, km); time e^{-i omega t}; state
ordered (u_z, u_x, u_y, e_zz, e_xx, e_yy, 2e_xy, 2e_zy, 2e_zx) with z = axis 0
(down), x = 1, y = 2. `omega` is COMPLEX: a real one puts the 1/k branch points
on the contour for anything spectral these tables compose with.

NOTE ON ARGUMENT ORDER, which has cost time here before.
`horizontal_greens.exact_propagator_9x9` takes CARTESIAN `(x, y, z)` in that
order, while the state vector is ordered `(z, x, y)`. A separation of one pitch
along y is `(0.0, pitch, 0.0)`. Reversing it produces a plausible wrong answer,
not an error.
"""

import numpy as np
from numpy.typing import NDArray

from .effective_contrasts import ReferenceMedium
from .horizontal_greens import exact_propagator_9x9

__all__ = ["same_depth_table", "separation_index"]


def separation_index(d: int, n: int) -> int:
    """Index into a separation-tabulated axis of length 2n-1.

    Args:
        d: Separation in voxels, in [-(n-1), n-1].
        n: Number of sites along the axis.

    Returns:
        The array index, d + n - 1.
    """
    return d + n - 1


def _check_lattice(n_x: int, n_y: int) -> None:
    if n_x < 2 or n_y < 2:
        msg = (
            f"need n_x >= 2 and n_y >= 2 for a same-depth table, got ({n_x}, {n_y}).\n"
            "  Where: cubic_scattering/pair_propagators.py, same_depth_table\n"
            "  Valid: e.g. n_x=16, n_y=16\n"
            "  Fix:   a lattice with a single site along an axis has no coupling\n"
            "         along it -- drop that axis from the model rather than\n"
            "         tabulating a propagator that would have only its own\n"
            "         poisoned self-term."
        )
        raise ValueError(msg) from None


def same_depth_table(
    n_x: int,
    n_y: int,
    pitch: float,
    omega: complex,
    ref: ReferenceMedium,
) -> NDArray:
    """Closed-form whole-space propagator at every same-depth separation.

    Covers pair classes B and C of the 3-D partition together -- every pair with
    Dz = 0, whatever its Dx and Dy. There is no quadrature and therefore no
    quadrature error: each entry is the closed form evaluated at that
    separation.

    Args:
        n_x: Sites along x (>= 2).
        n_y: Sites along y (>= 2).
        pitch: Voxel pitch, km. Must be > 0.
        omega: Complex angular frequency, rad/s.
        ref: Background medium (seismic units).

    Returns:
        Shape (2 n_x - 1, 2 n_y - 1, 9, 9), indexed by
        ``(separation_index(dx, n_x), separation_index(dy, n_y))``. The
        zero-separation entry is NaN throughout -- see the module docstring.

    Raises:
        ValueError: on a lattice too small along either axis, or a non-positive
            pitch.
    """
    _check_lattice(n_x, n_y)
    if pitch <= 0.0:
        msg = (
            f"pitch must be > 0, got {pitch!r}.\n"
            "  Where: cubic_scattering/pair_propagators.py, same_depth_table(pitch=...)\n"
            "  Valid: a positive voxel pitch in km, e.g. pitch=0.25\n"
            "  Fix:   use the cube side length d = 2a, not the half-width a."
        )
        raise ValueError(msg) from None

    tab = np.empty((2 * n_x - 1, 2 * n_y - 1, 9, 9), dtype=complex)
    for dx in range(-(n_x - 1), n_x):
        for dy in range(-(n_y - 1), n_y):
            i, j = separation_index(dx, n_x), separation_index(dy, n_y)
            if dx == 0 and dy == 0:
                tab[i, j] = np.nan
                continue
            # Cartesian (x, y, z) -- see the module docstring.
            tab[i, j] = exact_propagator_9x9(dx * pitch, dy * pitch, 0.0, omega, ref)
    return tab
