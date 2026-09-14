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

  * `same_depth_table` -- Dz = 0 coupling as the closed-form whole-space
    propagator, tabulated by separation. Exact, no quadrature at all.
  * `layered_stack_table` -- the plane-to-plane propagator, with the whole-space
    part subtracted in spectral space and added back in real space from the
    closed form, so that ONLY the layer reverberation is ever transformed. That
    subtraction is the economy of the whole design: the full layered kernel
    needs some 2.4M transverse nodes, the reverberation about 65k.

A table is indexed by SEPARATION, so it holds (2 n_x - 1) x (2 n_y - 1) entries
for a lattice of n_x by n_y.

THE TWO TABLES TREAT THE SELF-SITE DIFFERENTLY, AND THAT IS DELIBERATE.
`same_depth_table` poisons dx = dy = 0 with NaN: the whole-space self-term
belongs inside T0, must be reached by no propagator, and a zero there would let
a partition bug contribute silently instead of failing where it happens.
`layered_stack_table`'s diagonal block DOES carry dx = dy = 0, because the
self-RETURN -- out of a voxel, off a layer boundary, back to the same voxel --
is not in T0 either, T0 being the whole-space single-site T-matrix. Stage 1
established this and `scripts/gate_dg0_absolute_magnitude.py` validated its
magnitude absolutely. Dropping it would silently lose every layer-reflected
path.

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

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .effective_contrasts import ReferenceMedium
from .horizontal_greens import exact_propagator_9x9

__all__ = ["TransverseRule", "layered_stack_table", "same_depth_table", "separation_index"]


@dataclass(frozen=True)
class TransverseRule:
    """Tensor-product (k_x, k_y) quadrature for the layered transform.

    A TENSOR product, deliberately, not the radially masked set that pays at
    Dz = 0: separability is what keeps the transform cheap, and the mask
    destroys it. At Dz != 0 the e^{-kappa |dz|} factor already suppresses the
    corners the mask would drop.

    The measured rule for the layer reverberation is kr_max = 10/pitch with
    n_axis = 256 -- `scripts/measure_dg0_transverse_rule.py`. The reverberation
    needs a far smaller cutoff than the singular whole-space kernel (10 against
    30), which is what makes this affordable.

    Attributes:
        kr_max: Cutoff on each axis, 1/km.
        n_axis: Nodes per axis.
    """

    kr_max: float
    n_axis: int

    def nodes(self) -> tuple[NDArray, float]:
        """Midpoint nodes on one axis, and the spacing.

        Returns:
            (k, dk) with k of shape (n_axis,).
        """
        edge = np.linspace(-self.kr_max, self.kr_max, self.n_axis + 1)
        return 0.5 * (edge[:-1] + edge[1:]), float(edge[1] - edge[0])


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


def _transform_separable(spec: NDArray, k: NDArray, dk: float, n_x: int, n_y: int, pitch: float) -> NDArray:
    """(k_x, k_y) -> (dx, dy) by two 1-D sums instead of one 2-D one.

    The Fourier sum over a tensor grid factorises, so transforming along k_x and
    then k_y costs n_k * sqrt(n_out) rather than n_k * n_out. Measured, the
    transform is not the bottleneck either way -- the stratified propagator
    evaluation dominates at ~0.88 ms/node -- but the separable form costs
    nothing to write and keeps the scaling sane as the lattice grows.

    Args:
        spec: Spectral values, shape (n_k, n_k, 9, 9), indexed (k_x, k_y).
        k: Node positions on one axis, shape (n_k,).
        dk: Node spacing.
        n_x: Sites along x.
        n_y: Sites along y.
        pitch: Voxel pitch, km.

    Returns:
        Shape (2 n_x - 1, 2 n_y - 1, 9, 9).
    """
    dx = np.arange(-(n_x - 1), n_x) * pitch
    dy = np.arange(-(n_y - 1), n_y) * pitch
    e_x = np.exp(1j * np.outer(dx, k))  # (n_dx, n_k)
    e_y = np.exp(1j * np.outer(dy, k))  # (n_dy, n_k)
    w = dk * dk / (2 * np.pi) ** 2
    part = np.einsum("qb,abij->aqij", e_y, spec)
    return np.einsum("pa,aqij->pqij", e_x, part) * w


def layered_stack_table(
    n_z: int,
    n_x: int,
    n_y: int,
    pitch: float,
    omega: complex,
    ref: ReferenceMedium,
    *,
    model: object | None = None,
    plane_ifaces: tuple[int, ...] | None = None,
    transverse: TransverseRule | None = None,
    free_surface: bool = False,
) -> NDArray:
    """Plane-to-plane propagator at every separation, in real space.

    The 3-D counterpart of ``directional_sweeps.build_vertical_stack_layered``,
    with the SAME semantics:

      * off-diagonal ``[lz, mz]`` -- the full propagator from plane mz to plane
        lz, so ``dz = (lz - mz) * pitch``;
      * diagonal ``[lz, lz]`` -- the layer REVERBERATION only, the full layered
        kernel minus the same-depth whole-space term. Zero without a model.

    THE DIAGONAL CARRIES dx = dy = 0, where ``same_depth_table`` poisons it, and
    that is not an inconsistency. The whole-space self-term belongs to T0; the
    self-RETURN -- out of a voxel, off a layer boundary, back to the same voxel
    -- does not, because T0 is the whole-space single-site T-matrix. Dropping it
    would silently lose every layer-reflected path.

    Args:
        n_z: Number of depth planes (>= 1).
        n_x: Sites along x (>= 2).
        n_y: Sites along y (>= 2).
        pitch: Voxel pitch, km (> 0).
        omega: Complex angular frequency, rad/s.
        ref: Background medium, used for the whole-space case and for the
            same-depth subtraction on the diagonal.
        model: A ``LayerModel`` for the stratified background. Omit for the
            whole-space case.
        plane_ifaces: Interface index of each depth plane, length n_z. Required
            with ``model``.
        transverse: Quadrature for the spectral-to-real transform. Required
            with ``model``.
        free_surface: Close the ocean above with a pressure-release surface, so
            the water column is a FINITE layer that reverberates. Default False
            leaves the ocean as a half-space above the seabed: the seabed
            fluid-solid interface is still present and still reflects, but the
            water thickness does not reach the answer at all. Turn it on for any
            comparison against a solver that carries a free surface -- ``FFTProp``
            does. See ``scripts/gate_free_surface_reverberation.py``.

    Returns:
        Shape (n_z, n_z, 2 n_x - 1, 2 n_y - 1, 9, 9).

    Raises:
        ValueError: on a bad lattice or pitch, or on a model supplied without a
            matching plane map and transverse rule.
    """
    _check_lattice(n_x, n_y)
    if pitch <= 0.0:
        msg = (
            f"pitch must be > 0, got {pitch!r}.\n"
            "  Where: cubic_scattering/pair_propagators.py, layered_stack_table(pitch=...)\n"
            "  Valid: a positive voxel pitch in km, e.g. pitch=0.25\n"
            "  Fix:   use the cube side length d = 2a, not the half-width a."
        )
        raise ValueError(msg) from None

    tab = np.zeros((n_z, n_z, 2 * n_x - 1, 2 * n_y - 1, 9, 9), dtype=complex)

    if model is None:
        for lz in range(n_z):
            for mz in range(n_z):
                if lz == mz:
                    continue  # no reverberation without layering
                for dx in range(-(n_x - 1), n_x):
                    for dy in range(-(n_y - 1), n_y):
                        tab[lz, mz, separation_index(dx, n_x), separation_index(dy, n_y)] = (
                            exact_propagator_9x9(dx * pitch, dy * pitch, (lz - mz) * pitch, omega, ref)
                        )
        return tab

    if plane_ifaces is None or len(plane_ifaces) != n_z or transverse is None:
        got = "None" if plane_ifaces is None else str(len(plane_ifaces))
        msg = (
            f"a stratified model needs a plane map of length n_z={n_z} and a transverse "
            f"rule; got plane_ifaces length {got} and "
            f"transverse={'None' if transverse is None else 'set'}.\n"
            "  Where: cubic_scattering/pair_propagators.py, layered_stack_table\n"
            "  Valid: plane_ifaces=(8, 9) with n_z=2, and "
            "transverse=TransverseRule(kr_max=10.0/pitch, n_axis=256)\n"
            "  Fix:   build the LayerModel with one layer per inter-plane gap, each of\n"
            "         thickness pitch, list the interface at each plane, and pass the\n"
            "         rule measured by scripts/measure_dg0_transverse_rule.py."
        )
        raise ValueError(msg) from None

    from .layered_correction import corrected_layered_9x9
    from .sweep_kernels import same_depth_kernel_9x9, vertical_kernel_9x9

    k, dk = transverse.nodes()
    kxg, kyg = np.meshgrid(k, k, indexing="ij")
    kx, ky = kxg.ravel(), kyg.ravel()
    n_k = k.size
    s_p = model.complex_slowness_p()  # type: ignore[attr-defined]
    s_s = model.complex_slowness_s()  # type: ignore[attr-defined]
    rho_m = model.rho  # type: ignore[attr-defined]

    for lz in range(n_z):
        for mz in range(n_z):
            g9 = corrected_layered_9x9(
                model,
                omega,
                kx,
                ky,
                source_iface=plane_ifaces[mz],
                receiver_iface=plane_ifaces[lz],
                free_surface=free_surface,
            )
            # SUBTRACT THE WHOLE-SPACE PART IN SPECTRAL SPACE, ALWAYS -- not
            # only on the diagonal. Only the REVERBERATION is ever transformed.
            #
            # This is the entire economy of the design. The full layered kernel
            # inherits the whole-space kernel's slow spectral decay and needs
            # kr*pitch = 30 at dk = 0.156, some 2.4M nodes; the reverberation
            # converges at kr*pitch = 10, some 65k. Transforming the full kernel
            # puts the expensive rule back and throws away the 36x saving that
            # makes this architecture affordable at all. Measured in
            # scripts/measure_sweep3d_cost.py and measure_dg0_transverse_rule.py.
            #
            # The local medium AT THE PLANE, not the caller's reference --
            # mirroring build_vertical_stack_layered. With a contrast between
            # planes the two differ, and the caller's would leave a residue.
            j_lay = max(int(plane_ifaces[lz]), 1)
            ref_local = ReferenceMedium(1.0 / s_p[j_lay], 1.0 / s_s[j_lay], rho_m[j_lay])
            d_z = (lz - mz) * pitch

            # meshgrid(indexing="ij") ravels as index = i * n_k + j with
            # kx = k[i] and ky = k[j], so a STRIDED slice is the set with ky
            # fixed and kx running -- the orientation both whole-space kernels
            # take (array kx, scalar ky). A contiguous slice is the transpose.
            for j in range(n_k):
                ws = (
                    same_depth_kernel_9x9(k, float(k[j]), omega, ref_local)
                    if lz == mz
                    else vertical_kernel_9x9(k, float(k[j]), d_z, omega, ref_local)
                )
                g9[j::n_k] -= np.moveaxis(ws, -1, 0)

            tab[lz, mz] = _transform_separable(g9.reshape(n_k, n_k, 9, 9), k, dk, n_x, n_y, pitch)

            if lz != mz:
                # Add the whole-space part back in REAL space, where it is the
                # closed form and therefore exact. The diagonal gets no such
                # term: its whole-space half belongs to same_depth_table.
                for dx in range(-(n_x - 1), n_x):
                    for dy in range(-(n_y - 1), n_y):
                        tab[lz, mz, separation_index(dx, n_x), separation_index(dy, n_y)] += (
                            exact_propagator_9x9(dx * pitch, dy * pitch, d_z, omega, ref_local)
                        )
    return tab
