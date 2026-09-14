"""3D Slab Foldy-Lax Multiple Scattering Solver.

Solves the Foldy-Lax system ``(I - G·T)ψ = ψ⁰`` for an M×M×N_z lattice
of cubes with individual elastic properties.  Uses FFT convolution in the
horizontal plane for O(N_z² × M² log M) matvec cost.

Coordinate system: z=0 (down), x=1 (right), y=2 (out) — seismological.
Voigt ordering: (zz, xx, yy, xy, zy, zx) with engineering halving.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator, gmres

from .effective_contrasts import (
    MaterialContrast,
    ReferenceMedium,
    compute_cube_tmatrix,
)
from .inter_voxel_propagator import inter_voxel_propagator_9x9
from .kennett_layers import (
    IsotropicLayer,
    LayerStack,
    _complex_slowness,
    _vertical_slowness,
    kennett_layers,
)
from .lattice_greens import _apply_refl_x, _apply_refl_y, _apply_rot90
from .lattice_kupradze import bloch_kernel_hat_9x9
from .resonance_tmatrix import _propagator_block_9x9, _sub_cell_tmatrix_9x9
from .sphere_scattering import _plane_wave_strain_voigt

# ═══════════════════════════════════════════════════════════════
#  Data Structures
# ═══════════════════════════════════════════════════════════════


@dataclass
class SlabGeometry:
    """Geometry of the M×M×N_z slab lattice.

    Args:
        M: Horizontal grid size (M×M cubes per layer).
        N_z: Number of vertical layers.
        a: Cube half-width (m).  Cube side length d = 2a.
    """

    M: int
    N_z: int
    a: float

    def __post_init__(self) -> None:
        if self.M < 1:
            msg = f"M must be >= 1, got {self.M}"
            raise ValueError(msg)
        if self.N_z < 1:
            msg = f"N_z must be >= 1, got {self.N_z}"
            raise ValueError(msg)
        if self.a <= 0:
            msg = f"a must be > 0, got {self.a}"
            raise ValueError(msg)

    @property
    def d(self) -> float:
        """Cube side length (= 2a)."""
        return 2.0 * self.a

    @property
    def n_cubes(self) -> int:
        """Total number of cubes."""
        return self.M * self.M * self.N_z

    def cube_centre(self, lz: int, i: int, j: int) -> NDArray[np.floating]:
        """Centre of cube (lz, i, j) in seismological (z, x, y) ordering.

        Args:
            lz: Vertical layer index (0 = shallowest).
            i: Horizontal x-index.
            j: Horizontal y-index.

        Returns:
            Coordinates [z, x, y] in metres.
        """
        d = self.d
        z = (lz + 0.5) * d
        x = (i - (self.M - 1) / 2.0) * d
        y = (j - (self.M - 1) / 2.0) * d
        return np.array([z, x, y])

    def all_centres(self) -> NDArray[np.floating]:
        """Centres of all cubes, shape (N_z, M, M, 3)."""
        d = self.d
        z = (np.arange(self.N_z) + 0.5) * d
        x = (np.arange(self.M) - (self.M - 1) / 2.0) * d
        y = (np.arange(self.M) - (self.M - 1) / 2.0) * d
        zz, xx, yy = np.meshgrid(z, x, y, indexing="ij")
        return np.stack([zz, xx, yy], axis=-1)


@dataclass
class SlabMaterial:
    """Material contrasts for each cube in the slab.

    Args:
        Dlambda: Δλ contrast array, shape (N_z, M, M) in Pa.
        Dmu: Δμ contrast array, shape (N_z, M, M) in Pa.
        Drho: Δρ contrast array, shape (N_z, M, M) in kg/m³.
        ref: Background elastic medium.
    """

    Dlambda: NDArray
    Dmu: NDArray
    Drho: NDArray
    ref: ReferenceMedium

    def __post_init__(self) -> None:
        if self.Dlambda.shape != self.Dmu.shape or self.Dlambda.shape != self.Drho.shape:
            msg = (
                f"Contrast array shapes must match: "
                f"Dlambda={self.Dlambda.shape}, Dmu={self.Dmu.shape}, "
                f"Drho={self.Drho.shape}"
            )
            raise ValueError(msg)
        if self.Dlambda.ndim != 3:
            msg = f"Contrast arrays must be 3D (N_z, M, M), got ndim={self.Dlambda.ndim}"
            raise ValueError(msg)


@dataclass
class SlabResult:
    """Result of the slab multiple scattering computation.

    Attributes:
        psi: Exciting field, shape (N_z, M, M, 9).
        psi0: Incident field, shape (N_z, M, M, 9).
        geometry: Slab lattice geometry.
        material: Per-cube material contrasts.
        omega: Angular frequency (rad/s).
        k_hat: Unit incident propagation direction.
        wave_type: 'P' or 'S'.
        n_gmres_iter: Number of GMRES matvec evaluations.
        gmres_residual: Final relative residual norm.
    """

    psi: NDArray
    psi0: NDArray
    geometry: SlabGeometry
    material: SlabMaterial
    omega: float
    k_hat: NDArray
    wave_type: str
    n_gmres_iter: int
    gmres_residual: float
    periodic: bool = False


# ═══════════════════════════════════════════════════════════════
#  1. T-matrix construction
# ═══════════════════════════════════════════════════════════════


def compute_slab_tmatrices(
    geometry: SlabGeometry,
    material: SlabMaterial,
    omega: float,
    k_hat: NDArray | None = None,
) -> NDArray:
    """Build 9×9 T-matrix for each cube in the slab.

    Caches by unique (Δλ, Δμ, Δρ) triples for efficiency with
    binary random media.

    Args:
        geometry: Slab lattice geometry.
        material: Per-cube material contrasts.
        omega: Angular frequency (rad/s).
        k_hat: Optional unit incidence direction for the O((ka)⁴) cubic
            anisotropy of the single-site form factor.  Defaults to ``None``
            (ISOTROPIC) — the slab specular response orientation-averages the
            cubic anisotropy out, so the isotropic c₄ is the correct choice and
            the per-contrast cache stays direction-independent.  Threaded to
            :func:`compute_cube_tmatrix` for completeness; leave ``None``.

    Returns:
        T-matrices, shape (N_z, M, M, 9, 9), complex.
    """
    M, N_z, a = geometry.M, geometry.N_z, geometry.a
    ref = material.ref
    T_all = np.zeros((N_z, M, M, 9, 9), dtype=complex)
    # Function-local cache, rebuilt per call. The key omits k_hat because k_hat
    # is constant across all cubes within one call. If this cache is ever lifted
    # to cross-call memoization, k_hat MUST be added to the key.
    cache: dict[tuple[float, float, float], NDArray] = {}

    for lz in range(N_z):
        for i in range(M):
            for j in range(M):
                dl = float(material.Dlambda[lz, i, j])
                dm = float(material.Dmu[lz, i, j])
                dr = float(material.Drho[lz, i, j])
                key = (dl, dm, dr)
                if key not in cache:
                    contrast = MaterialContrast(dl, dm, dr)
                    result = compute_cube_tmatrix(omega, a, ref, contrast, k_hat=k_hat)
                    cache[key] = _sub_cell_tmatrix_9x9(result, omega, a)
                T_all[lz, i, j] = cache[key]

    return T_all


# ═══════════════════════════════════════════════════════════════
#  2. D4h symmetry helpers
# ═══════════════════════════════════════════════════════════════


def _add_supercell_images(
    kernel_circ: NDArray,
    M: int,
    d: float,
    dz: float,
    omega: float,
    ref: ReferenceMedium,
    n_img: int,
) -> None:
    """Add the periodic images the (2M-1)^2 patch leaves out, in place.

    For a medium periodic with supercell L = M d, the kernel entry at in-cell
    offset (cx, cy) must be the sum over EVERY image,

        K(cx, cy) = sum_{nx, ny} G(dz, (cx + nx M) d, (cy + ny M) d)

    The existing fold supplies the terms with (cx + nx M) and (cy + ny M) inside
    [-(M-1), M-1]; this adds the rest out to |nx|, |ny| <= n_img.

    Convergence is the point of the exercise and is NOT assumed: the 1/r^3
    strain block falls off fast enough for a direct sum, while the 1/r
    displacement block converges only conditionally in two dimensions, so the
    residual after truncation is what decides whether an Ewald split is needed.
    `scripts/gate_lateral_sum_invariance.py` measures it against the exact
    supercell-invariance requirement rather than inspecting the terms.
    """
    for cx in range(M):
        for cy in range(M):
            acc = np.zeros((9, 9), dtype=complex)
            for nx in range(-n_img, n_img + 1):
                for ny in range(-n_img, n_img + 1):
                    gx, gy = cx + nx * M, cy + ny * M
                    # Skip what the (2M-1)^2 patch already contributed, and the
                    # self term, which belongs to T0 and not to the propagator.
                    if abs(gx) <= M - 1 and abs(gy) <= M - 1:
                        continue
                    if gx == 0 and gy == 0 and abs(dz) < 1e-15 * max(d, 1.0):
                        continue
                    acc += _propagator_block_9x9(np.array([dz, gx * d, gy * d]), omega, ref)
            kernel_circ[cx, cy] += acc


def _bloch_contact_correction(
    M: int,
    d: float,
    dz_vox: int,
    omega: float,
    ref: ReferenceMedium,
    n_orders: int,
    va_radius: int,
) -> NDArray:
    """Bloch transform of [Galerkin - point] over the contact shell.

    The exact lattice sum is assembled from point propagators, which are a
    MIDPOINT rule for what should be the doubly volume-averaged operator between
    cells. That approximation is scale-invariant at contact -- (cell size) /
    (separation) is 1 at every refinement -- and negligible beyond it, where the
    ratio falls. So the correction is short-ranged by construction and its Bloch
    transform is a finite sum over a handful of cells, exact and cheap.

    Args:
        M: Supercell size.
        d: Lattice pitch.
        dz_vox: Vertical separation in CELLS (not metres).
        omega: Angular frequency.
        ref: Background medium.
        n_orders: Dynamic correction orders for the volume-averaged propagator.
        va_radius: Chebyshev cell radius of the correction shell.

    Returns:
        Shape (M, M, 9, 9), to be added to the point-propagator Bloch kernel.
    """
    offsets: list[tuple[int, int]] = []
    deltas: list[NDArray] = []
    for dx in range(-va_radius, va_radius + 1):
        for dy in range(-va_radius, va_radius + 1):
            if max(abs(dz_vox), abs(dx), abs(dy)) > va_radius:
                continue
            if dz_vox == 0 and dx == 0 and dy == 0:
                continue  # self-term lives in the local T-matrix
            g_avg = inter_voxel_propagator_9x9(
                (dz_vox, dx, dy), ref.alpha, ref.beta, ref.rho, omega, n_orders, d=d
            )
            g_pt = _propagator_block_9x9(np.array([dz_vox * d, dx * d, dy * d]), omega, ref)
            offsets.append((dx, dy))
            deltas.append(g_avg - g_pt)

    out = np.zeros((M, M, 9, 9), dtype=complex)
    if not offsets:
        return out
    for n1 in range(M):
        for n2 in range(M):
            # Same convention as the kernel it corrects: sum_R f(R) exp(-i k.R).
            k_par = 2.0 * np.pi * np.array([n1, n2], dtype=float) / (M * d)
            for (dx, dy), delta in zip(offsets, deltas, strict=True):
                phase = np.exp(-1j * (k_par[0] * dx * d + k_par[1] * dy * d))
                out[n1, n2] += delta * phase
    return out


def _cell_averaged_propagator(
    r_vec: NDArray,
    d: float,
    omega: float,
    ref: ReferenceMedium,
    n_gauss: int = 4,
    *,
    double: bool = True,
) -> NDArray:
    """<G> over the SOURCE cell, and the FIELD cell too when ``double``.

    (1/V) ∫_cell G(r_vec - u) du for a cube of side d.

    The Foldy-Lax sum approximates the continuum integral over each source cell
    by its midpoint value, V·G(r). The correct object is V·<G>. The difference
    is O((d/r)^2) relative, which for a space-filling lattice is scale-invariant
    -- refining the cells shrinks d and r together -- so it is a bias that
    refinement cannot remove.

    Valid only where G is SMOOTH over the cell, i.e. beyond the contact shell.
    At contact the kernel is singular and the analytic tables in
    inter_voxel_propagator must be used instead; this routine is never called
    there.
    """
    x, w = np.polynomial.legendre.leggauss(n_gauss)
    if double:
        # THE DOUBLE (GALERKIN) AVERAGE, over the source cell AND the field cell.
        # Averaging only the source still collocates the field at the receiving
        # cell centre, but the T-matrix responds to the field over its OWN
        # volume. The double average over two cubes is a single integral against
        # the convolution of their indicators -- a product of tent functions of
        # half-width d -- so it costs no more than the single average. The tent
        # has a kink at 0, so each axis is integrated on [-d,0] and [0,d]
        # separately rather than straddling it.
        nodes, wts = [], []
        for lo, hi in ((-d, 0.0), (0.0, d)):
            mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
            for xi, wi in zip(x, w, strict=True):
                t = mid + half * xi
                nodes.append(t)
                wts.append(wi * half * (1.0 - abs(t) / d) / d)
    else:
        nodes = list(0.5 * d * x)
        wts = list(0.5 * w)
    acc = np.zeros((9, 9), dtype=complex)
    for i, ui in enumerate(nodes):
        for j, uj in enumerate(nodes):
            for k, uk in enumerate(nodes):
                off = np.array([ui, uj, uk])
                acc += (wts[i] * wts[j] * wts[k]) * _propagator_block_9x9(r_vec - off, omega, ref)
    return acc


def _identity(G: NDArray) -> NDArray:
    return G.copy()


def _refl_xy(G: NDArray) -> NDArray:
    return _apply_refl_x(_apply_refl_y(G))


def _rot90_rx(G: NDArray) -> NDArray:
    return _apply_refl_x(_apply_rot90(G))


def _rot90_ry(G: NDArray) -> NDArray:
    return _apply_refl_y(_apply_rot90(G))


def _rot90_rxy(G: NDArray) -> NDArray:
    return _apply_refl_x(_apply_refl_y(_apply_rot90(G)))


def _d4h_orbit(dx: int, dy: int) -> list[tuple[int, int, Callable[[NDArray], NDArray]]]:
    """D4h orbit of horizontal offset (dx, dy).

    Returns all symmetry-related offsets and their 9×9 transformations.
    Exploits the D4h point symmetry of the square lattice in the (x,y) plane.
    """
    if dx == 0 and dy == 0:
        return [(0, 0, _identity)]

    if dx == dy:
        return [
            (dx, dy, _identity),
            (-dx, dy, _apply_refl_x),
            (dx, -dy, _apply_refl_y),
            (-dx, -dy, _refl_xy),
        ]

    return [
        (dx, dy, _identity),
        (-dx, dy, _apply_refl_x),
        (dx, -dy, _apply_refl_y),
        (-dx, -dy, _refl_xy),
        (dy, dx, _apply_rot90),
        (-dy, dx, _rot90_rx),
        (dy, -dx, _rot90_ry),
        (-dy, -dx, _rot90_rxy),
    ]


# ═══════════════════════════════════════════════════════════════
#  3. Kernel building
# ═══════════════════════════════════════════════════════════════


def _build_slab_kernels(
    geometry: SlabGeometry,
    omega: float,
    ref: ReferenceMedium,
    *,
    volume_averaged: bool = False,
    n_orders: int = 2,
    periodic: bool = False,
    va_radius: int = 1,
    va_all: bool = False,
    va_gauss: int = 4,
    lattice_images: int = 0,
    lattice_ewald: bool = False,
    ewald_eta: float | None = None,
    ewald_cutoff: int = 4,
) -> NDArray:
    """Build FFT kernels for all vertical separations.

    For each Δz, builds the (2M-1)×(2M-1) horizontal propagator kernel
    using D4h symmetry (~8× speedup), then FFTs.

    Args:
        geometry: Slab lattice geometry.
        omega: Angular frequency (rad/s).
        ref: Background elastic medium.
        volume_averaged: If True, use volume-averaged inter-voxel propagator
            for nearest-neighbour separations (26 cubes with max offset ≤ 1).
        n_orders: Dynamic correction orders for volume-averaged propagator
            (0=static, 1=+ω², 2=+ω⁴). Only used when volume_averaged=True.
        periodic: If True, fold kernel to M×M for circular convolution
            (infinite periodic slab). Default False gives (2M-1)×(2M-1)
            linear convolution (finite slab).

    Returns:
        FFT'd kernels, shape (2*N_z-1, H_xy, H_xy, 9, 9), complex,
        where H_xy = M if periodic else 2*M-1.
    """
    M, N_z, d = geometry.M, geometry.N_z, geometry.d
    S = 2 * M - 1
    n_dz = 2 * N_z - 1
    H_xy = M if periodic else S
    kernel_hat = np.zeros((n_dz, H_xy, H_xy, 9, 9), dtype=complex)

    if lattice_ewald:
        if not periodic:
            raise ValueError(
                "build_slab_kernels: lattice_ewald=True requires periodic=True. "
                "The Ewald route constructs the kernel at the M^2 Bloch points of a "
                "PERIODIC supercell; a finite slab has no lattice to sum over. "
                "Either set periodic=True, or drop lattice_ewald for the finite case. "
                "Example: build_slab_kernels(geom, omega, ref, periodic=True, "
                "lattice_ewald=True)."
            )
        # THE TRUNCATION IS BYPASSED ENTIRELY, not patched. fft2(kernel_circ) IS
        # the Bloch sum at k_n = 2 pi n / (M d); the defect was that only one
        # (2M-1)^2 patch of that sum was ever formed. So write the answer at
        # those M^2 points directly and never build the real-space array.
        for k in range(n_dz):
            dz_vox = k - (N_z - 1)
            kernel_hat[k] = bloch_kernel_hat_9x9(
                M,
                d,
                dz_vox * d,
                omega,
                ref,
                eta=ewald_eta,
                cutoff=ewald_cutoff,
            )
            if volume_averaged:
                # The lattice sum is built from POINT propagators. The Galerkin
                # (doubly volume-averaged) object differs from the point value
                # only at contact -- beyond it (cell size)/(separation) falls and
                # the midpoint rule is already good -- so the difference is a
                # SHORT-RANGED correction that can be added exactly, as a finite
                # Bloch sum over the contact shell. Without this the Ewald route
                # would silently ignore volume_averaged.
                kernel_hat[k] += _bloch_contact_correction(M, d, dz_vox, omega, ref, n_orders, va_radius)
        return kernel_hat

    for k in range(n_dz):
        dz_vox = k - (N_z - 1)
        dz = dz_vox * d
        kernel_spatial = np.zeros((S, S, 9, 9), dtype=complex)

        # Fundamental domain: 0 ≤ dy ≤ dx, dx ∈ [0, M-1]
        for dx in range(M):
            for dy in range(dx + 1):
                if dx == 0 and dy == 0 and abs(dz) < 1e-15 * max(d, 1.0):
                    continue  # self-term zeroed

                # The volume-averaged (Galerkin) propagator is applied out to
                # va_radius cells in the Chebyshev sense. The default of 1 is
                # the nearest-neighbour shell and reproduces the historical
                # behaviour bit for bit. It is a parameter because the point
                # propagator is a MIDPOINT rule for what should be a doubly
                # volume-averaged operator, and its error does not fall with the
                # cell size at contact -- (cell size)/(separation) is 1 at every
                # scale -- so how far out the averaging must reach before the
                # scheme converges is a question to be measured, not assumed.
                is_nn = max(abs(dz_vox), dx, dy) <= va_radius

                if volume_averaged and is_nn:
                    # Call inter_voxel_propagator_9x9 for each orbit point
                    # (it has its own O_h rotation, so pass signed offsets).
                    # d = geometry.d is the PHYSICAL cube side = lattice
                    # pitch — the propagator tables are unit-pitch and
                    # rescale internally (G/C/H/S by d^-1/-2/-2/-3).
                    for sdx, sdy, _transform in _d4h_orbit(dx, dy):
                        R_lattice = (dz_vox, sdx, sdy)
                        G0 = inter_voxel_propagator_9x9(
                            R_lattice,
                            ref.alpha,
                            ref.beta,
                            ref.rho,
                            omega,
                            n_orders,
                            d=d,
                        )
                        kernel_spatial[sdx + M - 1, sdy + M - 1] = G0
                else:
                    r_vec = np.array([dz, dx * d, dy * d])
                    # THE CONTINUUM INTEGRAL OVER THE SOURCE CELL IS V<G>, NOT
                    # V G(r). Collocating at the cell centre is a midpoint rule,
                    # and its error is scale-invariant -- <G> - G ~ (d/r)^2 G, so
                    # summed over r = d*n it is a pure number that refinement
                    # cannot dilute. That is the measured non-convergence.
                    # Beyond the contact shell the kernel is SMOOTH over the
                    # cell, so the average is plain Gauss quadrature; only
                    # contact needs the analytic tables above.
                    G0 = (
                        _cell_averaged_propagator(r_vec, d, omega, ref, va_gauss)
                        if volume_averaged and va_all
                        else _propagator_block_9x9(r_vec, omega, ref)
                    )

                    for sdx, sdy, transform in _d4h_orbit(dx, dy):
                        kernel_spatial[sdx + M - 1, sdy + M - 1] = transform(G0)

        if periodic:
            # Fold (2M-1)×(2M-1) spatial kernel into M×M for circular convolution
            kernel_circ = np.zeros((M, M, 9, 9), dtype=complex)
            for ix in range(S):
                for iy in range(S):
                    # dx ranges from -(M-1) to +(M-1), stored at ix = dx + M-1
                    dx_val = ix - (M - 1)
                    dy_val = iy - (M - 1)
                    kernel_circ[dx_val % M, dy_val % M] += kernel_spatial[ix, iy]
            if lattice_images:
                # THE SUM ABOVE IS NOT A LATTICE SUM. It covers only
                # dx, dy in [-(M-1), M-1] -- one (2M-1)^2 patch -- and then wraps.
                # A periodic medium needs every image of the supercell. The
                # omitted tail of the 1/r^3 strain block beyond radius ~M d falls
                # as 1/(M d), which shows up as an error ~ A + B/M and, since a
                # laterally uniform medium cannot depend on M at all, is provably
                # an artifact (scripts/measure_single_plane_core.py).
                #
                # Here the missing images are added explicitly. The nearest
                # image is M cells away, far outside the contact shell, so the
                # point propagator is the right object for every term added.
                _add_supercell_images(kernel_circ, M, d, dz, omega, ref, lattice_images)
            kernel_hat[k] = np.fft.fft2(kernel_circ, axes=(0, 1))
        else:
            # FFT over spatial dimensions for all 9×9 components
            kernel_hat[k] = np.fft.fft2(kernel_spatial, axes=(0, 1))

    return kernel_hat


# ═══════════════════════════════════════════════════════════════
#  4. FFT-accelerated matvec: (I − G·T)ψ
# ═══════════════════════════════════════════════════════════════


def _slab_matvec(
    psi_flat: NDArray,
    T_local: NDArray,
    kernel_hat: NDArray,
    geometry: SlabGeometry,
    *,
    periodic: bool = False,
) -> NDArray:
    """Compute (I − G·T)ψ using FFT convolution.

    Args:
        psi_flat: Exciting field, flat array of length N_z × M × M × 9.
        T_local: Per-cube T-matrices, shape (N_z, M, M, 9, 9).
        kernel_hat: FFT'd kernels from ``_build_slab_kernels``.
        geometry: Slab lattice geometry.
        periodic: If True, use circular convolution on M×M grid
            (kernel_hat has shape (n_dz, M, M, 9, 9)). Default False
            uses zero-padded linear convolution on (2M-1)×(2M-1).

    Returns:
        Result of (I − G·T)ψ, flat array of same length.
    """
    M, N_z = geometry.M, geometry.N_z

    psi = psi_flat.reshape(N_z, M, M, 9)

    # T-multiply: τ[l,i,j,:] = T[l,i,j,:,:] @ ψ[l,i,j,:]
    tau = np.einsum("lmnab,lmnb->lmna", T_local, psi)

    if periodic:
        # Circular convolution: FFT tau directly on M×M
        H_xy = M
        tau_hat = np.fft.fft2(tau, axes=(1, 2))
    else:
        # Linear convolution: zero-pad to (2M-1)×(2M-1)
        H_xy = 2 * M - 1
        tau_pad = np.zeros((N_z, H_xy, H_xy, 9), dtype=complex)
        tau_pad[:, :M, :M, :] = tau
        tau_hat = np.fft.fft2(tau_pad, axes=(1, 2))

    # Accumulate in Fourier domain: double loop over layer pairs
    acc_hat = np.zeros((N_z, H_xy, H_xy, 9), dtype=complex)
    for m in range(N_z):
        for n in range(N_z):
            dz_idx = (m - n) + (N_z - 1)
            # 9×9 matrix-vector product at each (kx, ky) point
            acc_hat[m] += np.einsum("xyij,xyj->xyi", kernel_hat[dz_idx], tau_hat[n])

    # IFFT and extract valid region
    acc = np.fft.ifft2(acc_hat, axes=(1, 2))

    if periodic:
        # Full M×M output is valid (circular convolution)
        acc_valid = acc
    else:
        # Extract alias-free region from linear convolution
        S = H_xy
        acc_valid = acc[:, M - 1 : S, M - 1 : S, :]

    return (psi - acc_valid).ravel()


# ═══════════════════════════════════════════════════════════════
#  5. Incident field
# ═══════════════════════════════════════════════════════════════


def _build_slab_incident_field(
    geometry: SlabGeometry,
    omega: float,
    ref: ReferenceMedium,
    k_hat: NDArray[np.floating],
    wave_type: str,
) -> NDArray:
    """Build plane-wave incident field for the slab.

    Args:
        geometry: Slab lattice geometry.
        omega: Angular frequency (rad/s).
        ref: Background elastic medium.
        k_hat: Unit propagation direction (z, x, y).
        wave_type: 'P', 'S' (SV), or 'SH'.

    Returns:
        Incident field, shape (N_z, M, M, 9).
    """
    k_hat = np.asarray(k_hat, dtype=float)
    k_hat = k_hat / np.linalg.norm(k_hat)

    if wave_type == "P":
        k_mag = omega / ref.alpha
        pol = k_hat.copy()
    elif wave_type == "S":
        k_mag = omega / ref.beta
        # SV polarisation: in vertical plane, perpendicular to k_hat
        z_hat = np.array([1.0, 0.0, 0.0])
        cross = np.cross(k_hat, z_hat)
        if np.linalg.norm(cross) < 1e-10:
            # Vertical incidence — continuous p→0⁺ limit of the generic
            # branch below, which is −x̂. The previous +x̂ choice was
            # discontinuous against that limit and broke the R_SS sign
            # (vs Kennett) at exactly p = 0.
            pol = np.array([0.0, -1.0, 0.0])
        else:
            pol = np.cross(cross, k_hat)
            pol = pol / np.linalg.norm(pol)
    elif wave_type == "SH":
        k_mag = omega / ref.beta
        # SH polarisation: horizontal, perpendicular to the sagittal plane
        pol = np.array([0.0, 0.0, 1.0])
    else:
        msg = f"wave_type must be 'P', 'S', or 'SH', got '{wave_type}'"
        raise ValueError(msg)

    eps_voigt = _plane_wave_strain_voigt(k_hat, pol, k_mag)

    # Build 9-component incident vector: [displacement(3), strain(6)]
    inc_vec = np.zeros(9, dtype=complex)
    inc_vec[:3] = pol
    inc_vec[3:] = eps_voigt

    # Phase at each cube centre
    centres = geometry.all_centres()  # (N_z, M, M, 3)
    phase = np.exp(1j * k_mag * np.einsum("k,lmnk->lmn", k_hat, centres))

    # Broadcast: psi0 = inc_vec * phase
    return inc_vec[np.newaxis, np.newaxis, np.newaxis, :] * phase[..., np.newaxis]


def _build_slab_incident_field_slowness(
    geometry: SlabGeometry,
    omega: float,
    s_vec: NDArray[np.complexfloating],
    pol: NDArray[np.complexfloating],
) -> NDArray:
    """Incident plane-wave field from a (possibly complex) slowness vector.

    Supports inhomogeneous (evanescent) incidence: u = pol·exp(iω s⃗·r)
    with complex s⃗ and pol. For Im(s_z) > 0 the field decays with depth,
    which is the physical post-critical incident wave (the
    ``_vertical_slowness`` branch enforces Im η ≥ 0, so e^{iωηz} with
    z positive downward gives e^{−ω|η|z} decay). Unit complex amplitude
    (pol·pol = 1, analytic continuation) at the z = 0 datum;
    sub-critically this equals unit displacement amplitude. The amplitude
    convention is the caller's responsibility — this builder applies
    whatever pol it is given. Kennett's coefficients use the same
    analytic-continuation convention, so comparisons remain
    convention-consistent post-critically.

    The strain uses the general complex-slowness plane-wave formula
    ε_ij = (iω/2)(s_i pol_j + s_j pol_i): passing s⃗ (unnormalized,
    complex) and ω to ``_plane_wave_strain_voigt`` as (k_hat, k_mag)
    gives exactly this — its internals are outer products, with no
    real-input or unit-norm assumption.

    Args:
        geometry: Slab lattice geometry.
        omega: Angular frequency (rad/s).
        s_vec: Slowness vector (s_z, s_x, s_y), possibly complex (s/m).
        pol: Polarisation vector, possibly complex (complex-unit
            normalization: pol·pol = 1, no conjugation).

    Returns:
        Incident field, shape (N_z, M, M, 9).
    """
    eps_voigt = _plane_wave_strain_voigt(s_vec, pol, omega)
    inc_vec = np.zeros(9, dtype=complex)
    inc_vec[:3] = pol
    inc_vec[3:] = eps_voigt
    centres = geometry.all_centres()
    phase = np.exp(1j * omega * np.einsum("k,lmnk->lmn", s_vec, centres))
    return inc_vec[np.newaxis, np.newaxis, np.newaxis, :] * phase[..., np.newaxis]


# ═══════════════════════════════════════════════════════════════
#  6. Main solver
# ═══════════════════════════════════════════════════════════════


def build_slab_kernels(
    geometry: SlabGeometry,
    omega: float,
    ref: ReferenceMedium,
    *,
    volume_averaged: bool = False,
    n_orders: int = 2,
    periodic: bool = False,
    va_radius: int = 1,
    va_all: bool = False,
    va_gauss: int = 4,
    lattice_images: int = 0,
    lattice_ewald: bool = False,
    ewald_eta: float | None = None,
    ewald_cutoff: int = 4,
) -> NDArray:
    """Build the FFT propagator kernel, for reuse across right-hand sides.

    The kernel depends only on the lattice, the frequency and the background
    medium — not on the incident field — so one build serves every incidence
    angle and wave type. Pass the result to ``compute_slab_scattering`` as
    ``kernel_hat``.

    Args:
        geometry: Slab lattice geometry.
        omega: Angular frequency (rad/s).
        ref: Background elastic medium.
        volume_averaged: Use the volume-averaged inter-voxel propagator for
            nearest-neighbour separations.
        n_orders: Dynamic correction orders when volume_averaged is True.
        periodic: Fold to M x M for circular convolution.
        va_radius: Chebyshev cell radius out to which the volume-averaged
            propagator is used. 1 (default) is the nearest-neighbour shell.
        va_all: Average the propagator over the source cell at EVERY separation,
            by Gauss quadrature beyond the contact shell. The Foldy-Lax sum
            otherwise uses the midpoint value G(r) in place of the continuum
            integral V<G>, a scale-invariant bias that refinement cannot remove.
        va_gauss: Gauss points per axis for that quadrature (default 4).
        lattice_images: Supercell images added to the periodic lateral sum.
            0 (default) reproduces the historical truncated-and-wrapped sum,
            which is NOT a lattice sum and carries an O(1/M) artifact.
        lattice_ewald: Build the periodic kernel as an EXACT Bloch lattice sum by
            Ewald summation, bypassing the real-space assembly and its
            truncation entirely. Requires periodic=True. This is the correct
            route; ``lattice_images`` is the partial fix it supersedes, kept
            because its 1/n_img stall is what showed Ewald to be necessary.
        ewald_eta: Ewald splitting parameter; None uses sqrt(pi)/d. The result is
            independent of it, which is how the sum is gated.
        ewald_cutoff: Half-width of both Ewald sums (default 4, ~1e-12).

    Returns:
        The FFT kernel; pass it straight back as ``kernel_hat``.
    """
    return _build_slab_kernels(
        geometry,
        omega,
        ref,
        volume_averaged=volume_averaged,
        n_orders=n_orders,
        periodic=periodic,
        va_radius=va_radius,
        va_all=va_all,
        va_gauss=va_gauss,
        lattice_images=lattice_images,
        lattice_ewald=lattice_ewald,
        ewald_eta=ewald_eta,
        ewald_cutoff=ewald_cutoff,
    )


def compute_slab_scattering(
    geometry: SlabGeometry,
    material: SlabMaterial,
    omega: float,
    k_hat: NDArray[np.floating],
    wave_type: str = "P",
    gmres_tol: float = 1e-6,
    max_iter: int = 500,
    *,
    volume_averaged: bool = False,
    n_orders: int = 2,
    periodic: bool = False,
    psi0: NDArray | None = None,
    kernel_hat: NDArray | None = None,
    T_local: NDArray | None = None,
) -> SlabResult:
    """Solve the Foldy-Lax slab scattering problem via GMRES.

    Solves ``(I − G·T)ψ = ψ⁰`` for the exciting field ψ using
    FFT-accelerated matvec.

    Args:
        geometry: Slab lattice geometry.
        material: Per-cube material contrasts.
        omega: Angular frequency (rad/s).
        k_hat: Unit incident propagation direction (z, x, y). When
            ``psi0`` is given this is metadata only (stored on the
            result, not used to build the incident field).
        wave_type: 'P', 'S' (SV), or 'SH'.
        gmres_tol: GMRES relative tolerance.
        max_iter: Maximum GMRES iterations.
        volume_averaged: If True, use volume-averaged inter-voxel propagator
            for nearest-neighbour separations.
        n_orders: Dynamic correction orders for volume-averaged propagator.
        periodic: If True, use circular convolution for an infinite periodic
            slab. Default False gives linear convolution (finite slab).
        psi0: Optional prebuilt incident field, shape (N_z, M, M, 9).
            When provided, the homogeneous-plane-wave construction from
            (k_hat, wave_type) is skipped — use
            ``_build_slab_incident_field_slowness`` to build evanescent
            (complex-slowness) incident fields. Note: result.k_hat is
            consumed by slab_reflected_field (r_hat = -k_hat); for a
            post-critical P solve built via psi0, k_hat is degenerate
            (horizontal) and slab_reflected_field output would be
            meaningless — use slab_weyl_amplitudes for extraction.
        kernel_hat: Optional prebuilt FFT kernel from ``build_slab_kernels``.
            The kernel depends only on (geometry, omega, ref) and the
            volume_averaged/n_orders/periodic flags — NOT on the incident
            field — so one build serves every incidence angle and wave type
            at that geometry and frequency. Building it costs 87–215 matvecs
            (measured 2026-09-13), so reusing it across an angle sweep is
            worth roughly a hundredfold. When given, the flags above must
            match those the kernel was built with; they are not re-checked.

    Returns:
        SlabResult with exciting and incident fields.
    """
    # A supplied T_local lets a caller probe a MODIFIED single-site response --
    # a lattice renormalisation, say -- without having to fake it through the
    # contrast, which T depends on nonlinearly. None rebuilds from the material,
    # which is the existing behaviour.
    if T_local is None:
        T_local = compute_slab_tmatrices(geometry, material, omega)
    if kernel_hat is None:
        kernel_hat = _build_slab_kernels(
            geometry,
            omega,
            material.ref,
            volume_averaged=volume_averaged,
            n_orders=n_orders,
            periodic=periodic,
        )
    if psi0 is None:
        psi0 = _build_slab_incident_field(geometry, omega, material.ref, k_hat, wave_type)

    n = geometry.N_z * geometry.M * geometry.M * 9
    n_matvec = [0]

    def matvec(x: NDArray) -> NDArray:
        n_matvec[0] += 1
        return _slab_matvec(x, T_local, kernel_hat, geometry, periodic=periodic)

    A = LinearOperator(shape=(n, n), matvec=matvec, dtype=complex)

    b = psi0.ravel()
    x, info = gmres(A, b, x0=b.copy(), rtol=gmres_tol, maxiter=max_iter)

    if info != 0:
        import warnings

        warnings.warn(
            f"GMRES did not converge: info={info} after {max_iter} iterations",
            stacklevel=2,
        )

    residual = np.linalg.norm(matvec(x) - b) / np.linalg.norm(b)

    return SlabResult(
        psi=x.reshape(geometry.N_z, geometry.M, geometry.M, 9),
        psi0=psi0,
        geometry=geometry,
        material=material,
        omega=omega,
        k_hat=np.asarray(k_hat, dtype=float),
        wave_type=wave_type,
        n_gmres_iter=n_matvec[0],
        gmres_residual=float(residual),
        periodic=periodic,
    )


# ═══════════════════════════════════════════════════════════════
#  7. Reflected field extraction
# ═══════════════════════════════════════════════════════════════


def _voigt_to_tensor(voigt_6: NDArray) -> NDArray:
    """Convert Voigt stress vector to 3×3 symmetric tensor.

    Voigt stores doubled off-diagonal:
    [σ_zz, σ_xx, σ_yy, 2σ_xy, 2σ_zy, 2σ_zx].
    """
    T = np.zeros((3, 3), dtype=complex)
    T[0, 0] = voigt_6[0]
    T[1, 1] = voigt_6[1]
    T[2, 2] = voigt_6[2]
    T[1, 2] = T[2, 1] = voigt_6[3] / 2.0
    T[0, 2] = T[2, 0] = voigt_6[4] / 2.0
    T[0, 1] = T[1, 0] = voigt_6[5] / 2.0
    return T


def slab_reflected_field(
    result: SlabResult,
    T_local: NDArray,
) -> tuple[complex, complex, complex]:
    """Extract reflected-wave amplitudes from the slab solution.

    Sums far-field contributions from all cubes in the upgoing direction
    (opposite to incident k_hat), normalised per unit area.

    Sign convention: global minus from T-matrix force convention,
    matching ``sphere_scattering.foldy_lax_far_field``.

    Args:
        result: Solved slab scattering result.
        T_local: Per-cube T-matrices, shape (N_z, M, M, 9, 9).

    Returns:
        (R_PP, R_PS, R_SP) complex reflection amplitudes.
    """
    geom = result.geometry
    ref = result.material.ref
    omega = result.omega
    k_hat = result.k_hat / np.linalg.norm(result.k_hat)

    kP = omega / ref.alpha
    kS = omega / ref.beta
    r_hat = -k_hat  # reflected (upgoing) direction

    # Far-field Green's prefactors (no 1/r — far-field amplitude)
    G_far_P = 1.0 / (4.0 * np.pi * ref.rho * ref.alpha**2)
    G_far_S = 1.0 / (4.0 * np.pi * ref.rho * ref.beta**2)

    # Compute sources: T @ ψ at each cube
    source = np.einsum("lmnab,lmnb->lmna", T_local, result.psi)
    centres = geom.all_centres()

    u_P = np.zeros(3, dtype=complex)
    u_S = np.zeros(3, dtype=complex)

    for lz in range(geom.N_z):
        for i in range(geom.M):
            for j in range(geom.M):
                force = source[lz, i, j, :3]
                sigma = _voigt_to_tensor(source[lz, i, j, 3:])
                r_cube = centres[lz, i, j]

                phase_P = np.exp(-1j * kP * np.dot(r_hat, r_cube))
                phase_S = np.exp(-1j * kS * np.dot(r_hat, r_cube))

                # P-wave (global minus from T-matrix sign convention)
                sigma_r = sigma @ r_hat
                sigma_rr = np.dot(r_hat, sigma_r)
                Q_P = np.dot(r_hat, force) - 1j * kP * sigma_rr
                u_P -= G_far_P * phase_P * Q_P * r_hat

                # S-wave
                Q_S = force - 1j * kS * sigma_r
                Q_S_perp = Q_S - np.dot(r_hat, Q_S) * r_hat
                u_S -= G_far_S * phase_S * Q_S_perp

    # Normalise per unit area
    area = (geom.M * geom.d) ** 2
    u_P /= area
    u_S /= area

    # Scalar amplitudes
    R_PP = complex(np.dot(r_hat, u_P))

    # SV polarisation in reflected direction
    z_hat = np.array([1.0, 0.0, 0.0])
    cross = np.cross(r_hat, z_hat)
    if np.linalg.norm(cross) < 1e-10:
        sv_hat = np.array([0.0, 1.0, 0.0])
    else:
        sv_hat = np.cross(cross, r_hat)
        sv_hat = sv_hat / np.linalg.norm(sv_hat)

    R_PS = complex(np.dot(sv_hat, u_S))
    R_SP = R_PP  # same P-projection, meaningful for S-wave incidence

    return R_PP, R_PS, R_SP


@dataclass
class WeylAmplitudes:
    """Specular Weyl amplitudes from one periodic-slab solve.

    Displacement-amplitude convention (unit-displacement incident wave).
    Convert to the Kennett modified convention via SlabReflectionMatrix.to_modified()
    (a measured diagonal similarity — see its docstring) before comparing off-diagonal
    channels.

    Attributes:
        R_P: Outgoing specular P amplitude.
        R_SV: Outgoing specular SV amplitude (sagittal polarisation).
        R_SH: Outgoing specular SH amplitude (y polarisation).
        p: Horizontal slowness used (s/m).
        eta_P: Vertical P slowness (complex past critical).
        eta_S: Vertical S slowness (complex past critical).
    """

    R_P: complex
    R_SV: complex
    R_SH: complex
    p: float
    eta_P: complex
    eta_S: complex


def slab_weyl_amplitudes(result: SlabResult, T_local: NDArray, *, p: float = 0.0) -> WeylAmplitudes:
    """Extract all specular outgoing amplitudes (P, SV, SH) via Weyl sums.

    The 2D lattice sum replaces exp(ikr)/(4πr) with i/(2k_z d²)·exp(ik_z|z|)
    per mode. The source coupling uses the full reflected wave vector
    ω·s⃗_m = k_m·d̂_m (NOT the vertical wavenumber — they differ at p>0):

        Q_P  = −d̂_P·f − iω (s⃗_P·σ·d̂_P)            (scalar)
        Q⃗_S  = −f − iω (σ·s⃗_S)                      (vector)
        R_m  = −i/(2 ω η_m d² ρ c_m²) Σ_l Q_m,l exp(iω η_m z_l)

    with the SV/SH amplitudes the ŝv/ŝh projections of Q⃗_S. The force term
    is negated (T-matrix +ω²Δρ V u convention, opposite to the
    Lippmann-Schwinger body force). Sources are averaged over the M²
    horizontal cubes per layer (specular/coherent response).

    Undefined at grazing incidence p = 1/α (P) or p = 1/β (S), where the
    corresponding η_m = 0 makes the Weyl prefactor singular.

    Args:
        result: Solved slab scattering result (use periodic=True).
        T_local: Per-cube T-matrices, shape (N_z, M, M, 9, 9).
        p: Horizontal slowness (s/m).

    Returns:
        WeylAmplitudes in the displacement convention.
    """
    geom = result.geometry
    ref = result.material.ref
    omega = result.omega
    d = geom.d

    eta_P = _vertical_slowness(_complex_slowness(ref.alpha, np.inf), p)
    eta_S = _vertical_slowness(_complex_slowness(ref.beta, np.inf), p)
    kz_P = omega * eta_P
    kz_S = omega * eta_S

    # Upgoing slowness vectors, complex-unit directions, S polarisations
    s_vec_P = np.array([-eta_P, p, 0.0], dtype=complex)
    s_vec_S = np.array([-eta_S, p, 0.0], dtype=complex)
    d_P = ref.alpha * s_vec_P
    sv_hat = ref.beta * np.array([p, eta_S, 0.0], dtype=complex)
    sh_hat = np.array([0.0, 0.0, 1.0])

    source = np.einsum("lmnab,lmnb->lmna", T_local, result.psi)
    centres = geom.all_centres()

    tot_P = 0.0 + 0.0j
    tot_SV = 0.0 + 0.0j
    tot_SH = 0.0 + 0.0j
    for lz in range(geom.N_z):
        f_avg = np.mean(source[lz, :, :, :3], axis=(0, 1))
        sig_avg = _voigt_to_tensor(np.mean(source[lz, :, :, 3:], axis=(0, 1)))
        z_l = centres[lz, 0, 0, 0]

        Q_P = -np.dot(d_P, f_avg) - 1j * omega * np.dot(s_vec_P, sig_avg @ d_P)
        tot_P += Q_P * np.exp(1j * kz_P * z_l)

        Q_S = -f_avg - 1j * omega * (sig_avg @ s_vec_S)
        phase_S = np.exp(1j * kz_S * z_l)
        tot_SV += np.dot(sv_hat, Q_S) * phase_S
        tot_SH += np.dot(sh_hat, Q_S) * phase_S

    pref_P = -1j / (2.0 * kz_P * d**2 * ref.rho * ref.alpha**2)
    pref_S = -1j / (2.0 * kz_S * d**2 * ref.rho * ref.beta**2)

    return WeylAmplitudes(
        R_P=complex(pref_P * tot_P),
        R_SV=complex(pref_S * tot_SV),
        R_SH=complex(pref_S * tot_SH),
        p=p,
        eta_P=complex(eta_P),
        eta_S=complex(eta_S),
    )


def slab_rpp_periodic(result: SlabResult, T_local: NDArray, *, p: float = 0.0) -> complex:
    """Specular P→P reflection coefficient for a periodic slab.

    Uses the Weyl representation: the 2D lattice sum replaces
    exp(ikr)/(4πr) with i/(2k_z d²)·exp(ik_z|z|), giving:

        R_PP = -(i / (2k_z d² ρα²)) × Σ_l Q_P,l × exp(ik_z z_l)

    where k_z = ω·η_P is the vertical P-wavenumber, η_P = √(1/α² - p²),
    and Q_P,l is the far-field P-source scalar for layer l, averaged over
    the M² horizontal cubes.

    For Kennett comparison, use ``periodic=True`` in ``compute_slab_scattering``
    so that the solver's circular convolution matches the infinite-medium
    assumption of the Kennett reflectivity.

    Args:
        result: Solved slab scattering result.
        T_local: Per-cube T-matrices, shape (N_z, M, M, 9, 9).
        p: Horizontal slowness (s/m). Default 0.0 (normal incidence).

    Returns:
        Complex specular P→P reflection coefficient (dimensionless).

    Note:
        Delegates to slab_weyl_amplitudes. The oblique stress coupling now
        uses the full wave vector (−iω s⃗·σ·d̂); the previous −i k_z σ_rr form
        was correct only at p=0.
    """
    return slab_weyl_amplitudes(result, T_local, p=p).R_P


@dataclass
class SlabReflectionMatrix:
    """Specular reflection matrix of the periodic heterogeneous slab.

    Attributes:
        R_psv: 2×2 P-SV matrix, displacement convention.
            Rows = outgoing mode (0=P, 1=SV); columns = incident mode.
        R_sh: SH→SH coefficient.
        p: Horizontal slowness (s/m).
        omega: Angular frequency (rad/s).
        eta_P: Vertical P slowness in the background.
        eta_S: Vertical S slowness in the background.
        n_gmres_iters: GMRES iterations for the (P, SV, SH) solves.
            Each slot stores SlabResult.n_gmres_iter, the matvec count
            including the final residual-check matvec; the SH slot is 0
            when include_sh=False.
    """

    R_psv: NDArray[np.complexfloating]
    R_sh: complex
    p: float
    omega: float
    eta_P: complex
    eta_S: complex
    n_gmres_iters: tuple[int, int, int] = (0, 0, 0)

    def to_modified(self) -> NDArray[np.complexfloating]:
        """Convert to the Kennett modified convention of kennett_layers.

        Diagonal similarity R̃ = D R D⁻¹ with D = diag(α·√η_P, i·β·√η_S).
        A naive sqrt(η_i/η_j) form would assume pure energy-normalized
        displacement amplitudes; the Kennett implementation in this codebase
        normalizes via its eigenvector convention, which carries velocity
        factors and a factor i on SV (visible as the m2ci = −2i factor in
        psv_solid_solid, kennett_layers.py ~line 358). The conversion was
        pinned by measurement against Kennett, with the reciprocity
        invariant — the product of the two off-diagonal ratios, which is
        invariant under any diagonal similarity — equal to 1 as the
        convention-independent check. Diagonal entries are unchanged; the
        modified matrix is symmetric by reciprocity. Singular at grazing
        incidence (η_P = 0 or η_S = 0 makes D non-invertible).
        """
        # Background velocities recovered from 1/c_m² = η_m² + p²
        # (exact for evanescent modes too, where η_m is imaginary).
        alpha = 1.0 / np.sqrt(self.eta_P**2 + self.p**2)
        beta = 1.0 / np.sqrt(self.eta_S**2 + self.p**2)
        d = np.array([alpha * np.sqrt(self.eta_P), 1j * beta * np.sqrt(self.eta_S)])
        return self.R_psv * np.outer(d, 1.0 / d)


def slab_reflection_matrix(
    geometry: SlabGeometry,
    material: SlabMaterial,
    omega: float,
    *,
    p: float = 0.0,
    gmres_tol: float = 1e-6,
    max_iter: int = 500,
    volume_averaged: bool = False,
    n_orders: int = 2,
    include_sh: bool = True,
) -> SlabReflectionMatrix:
    """Full specular reflection matrix via three periodic Foldy-Lax solves.

    Runs P-, SV-, and SH-incident solves at the same horizontal slowness p
    (downgoing slowness vector per mode: s⃗_m = (η_m, p, 0), possibly
    complex past critical) and assembles the 2×2 P-SV matrix plus the SH
    coefficient from the shared Weyl extractor. SH decouples from P-SV in
    the horizontally averaged (specular) response; the SH-incident solve
    only populates R_sh.

    Incident fields are built from complex slowness vectors
    (``_build_slab_incident_field_slowness``): past the corresponding
    critical slowness (p > 1/α for P, p > 1/β for SV) the incident wave
    is the physical inhomogeneous (evanescent) field
    u = pol·exp(iω(px + η z)) with η purely imaginary (Im η > 0),
    decaying with depth. Sub-critically this reduces to the homogeneous
    plane wave used previously.

    Validity:
        All channels are defined for all p except the exact grazing
        singularities p = 1/α or p = 1/β, where η_m = 0 makes the Weyl
        prefactor (and to_modified) singular.

    Args:
        geometry: Slab lattice geometry.
        material: Per-cube material contrasts.
        omega: Angular frequency (rad/s).
        p: Horizontal slowness (s/m).
        gmres_tol: GMRES relative tolerance.
        max_iter: Maximum GMRES iterations.
        volume_averaged: Use volume-averaged inter-voxel propagator.
        n_orders: Dynamic correction orders for the volume-averaged propagator.
        include_sh: If False, skip the SH-incident solve (R_sh = 0).
            Intended for embeddings where SH cannot couple (e.g. through
            a fluid).

    Returns:
        SlabReflectionMatrix (displacement convention; use .to_modified()
        for Kennett comparison and recursion mixing).
    """
    ref = material.ref
    eta_P = _vertical_slowness(_complex_slowness(ref.alpha, np.inf), p)
    eta_S = _vertical_slowness(_complex_slowness(ref.beta, np.inf), p)

    # Downgoing complex slowness vectors and complex-unit polarisations
    # (analytic continuation of the real-vector conventions; pol·pol = 1,
    # no conjugation). Im η > 0 past critical ⇒ depth-decaying incidence.
    s_vec_P = np.array([eta_P, p, 0.0], dtype=complex)
    s_vec_S = np.array([eta_S, p, 0.0], dtype=complex)
    pol_P = np.asarray(ref.alpha * s_vec_P, dtype=complex)
    pol_SV = np.asarray(ref.beta * np.array([p, -eta_S, 0.0]), dtype=complex)
    pol_SH = np.array([0.0, 0.0, 1.0], dtype=complex)

    # Real parts of the propagation directions — result metadata only
    k_hat_P = np.array([float(np.real(eta_P * ref.alpha)), p * ref.alpha, 0.0])
    k_hat_S = np.array([float(np.real(eta_S * ref.beta)), p * ref.beta, 0.0])

    T_local = compute_slab_tmatrices(geometry, material, omega)

    incidences: list[
        tuple[
            str,
            NDArray[np.floating],
            NDArray[np.complexfloating],
            NDArray[np.complexfloating],
        ]
    ] = [("P", k_hat_P, s_vec_P, pol_P), ("S", k_hat_S, s_vec_S, pol_SV)]
    if include_sh:
        incidences.append(("SH", k_hat_S, s_vec_S, pol_SH))

    amps: dict[str, WeylAmplitudes] = {}
    iters: dict[str, int] = {"P": 0, "S": 0, "SH": 0}
    for wave_type, k_hat, s_vec, pol in incidences:
        psi0 = _build_slab_incident_field_slowness(geometry, omega, s_vec, pol)
        result = compute_slab_scattering(
            geometry,
            material,
            omega,
            k_hat,
            wave_type=wave_type,
            gmres_tol=gmres_tol,
            max_iter=max_iter,
            periodic=True,
            volume_averaged=volume_averaged,
            n_orders=n_orders,
            psi0=psi0,
        )
        amps[wave_type] = slab_weyl_amplitudes(result, T_local, p=p)
        iters[wave_type] = result.n_gmres_iter

    R_psv = np.array(
        [
            [amps["P"].R_P, amps["S"].R_P],
            [amps["P"].R_SV, amps["S"].R_SV],
        ],
        dtype=complex,
    )
    return SlabReflectionMatrix(
        R_psv=R_psv,
        R_sh=amps["SH"].R_SH if include_sh else 0.0j,
        p=p,
        omega=omega,
        eta_P=complex(eta_P),
        eta_S=complex(eta_S),
        n_gmres_iters=(iters["P"], iters["S"], iters["SH"]),
    )


@dataclass
class KennettChannelReference:
    """All five Kennett reflection channels for a uniform 3-layer stack.

    Modified (energy-normalized) convention, as stored by kennett_layers.
    Channel naming follows KennettResult: R_PS = RD_psv[0, 1]. For
    reflection matrices in the modified convention the matrix is symmetric,
    so the in/out index order is immaterial here.
    """

    R_PP: complex
    R_PS: complex
    R_SP: complex
    R_SS: complex
    R_SH: complex


def kennett_reference_matrix(
    ref: ReferenceMedium,
    contrast: MaterialContrast,
    H: float,
    omega: float,
    *,
    p: float = 0.0,
) -> KennettChannelReference:
    """Kennett reference for all five channels of a uniform layer.

    Same 3-layer stack as kennett_reference_rpp:
    background(dummy) | perturbed(H) | background(halfspace), at slowness p.

    Args:
        ref: Background elastic medium.
        contrast: Material contrast defining the perturbed layer.
        H: Layer thickness (m).
        omega: Angular frequency (rad/s).
        p: Horizontal slowness (s/m).

    Returns:
        KennettChannelReference with complex coefficients.
    """
    lam_bg = ref.rho * (ref.alpha**2 - 2.0 * ref.beta**2)
    mu_bg = ref.rho * ref.beta**2
    lam_p = lam_bg + contrast.Dlambda
    mu_p = mu_bg + contrast.Dmu
    rho_p = ref.rho + contrast.Drho
    alpha_p = float(np.sqrt((lam_p + 2.0 * mu_p) / rho_p))
    beta_p = float(np.sqrt(mu_p / rho_p))

    stack = LayerStack(
        layers=[
            IsotropicLayer(alpha=ref.alpha, beta=ref.beta, rho=ref.rho, thickness=100.0),
            IsotropicLayer(alpha=alpha_p, beta=beta_p, rho=rho_p, thickness=H),
            IsotropicLayer(alpha=ref.alpha, beta=ref.beta, rho=ref.rho, thickness=np.inf),
        ]
    )
    result = kennett_layers(stack, p=p, omega=np.array([omega]))
    return KennettChannelReference(
        R_PP=complex(result.RPP[0]),
        R_PS=complex(result.RPS[0]),
        R_SP=complex(result.RSP[0]),
        R_SS=complex(result.RSS[0]),
        R_SH=complex(result.RSH[0]),
    )


def kennett_reference_rpp(
    ref: ReferenceMedium,
    contrast: MaterialContrast,
    H: float,
    omega: float,
) -> complex:
    """Kennett R_PP for a uniform layer of thickness H at normal incidence.

    Builds a 3-layer stack: background(dummy) | perturbed(H) | background(halfspace),
    then runs the Kennett recursion at p=0.

    Args:
        ref: Background elastic medium.
        contrast: Material contrast defining the perturbed layer.
        H: Layer thickness (m).
        omega: Angular frequency (rad/s).

    Returns:
        Complex PP reflection coefficient at normal incidence.
    """
    return kennett_reference_matrix(ref, contrast, H, omega, p=0.0).R_PP


# ═══════════════════════════════════════════════════════════════
#  8. Utility functions
# ═══════════════════════════════════════════════════════════════


def random_slab_material(
    geometry: SlabGeometry,
    ref: ReferenceMedium,
    contrast: MaterialContrast,
    phi: float | Callable[[int], float],
    seed: int | None = None,
) -> SlabMaterial:
    """Generate a binary random slab material.

    Each cube is independently an inclusion (probability φ) or matrix
    (zero contrast, probability 1−φ).

    Args:
        geometry: Slab lattice geometry.
        ref: Background elastic medium.
        contrast: Material contrast for inclusions.
        phi: Volume fraction — scalar or callable ``phi(layer_index)``.
        seed: Random seed for reproducibility.

    Returns:
        SlabMaterial with binary random contrasts.
    """
    rng = np.random.default_rng(seed)
    M, N_z = geometry.M, geometry.N_z
    shape = (N_z, M, M)

    Dlambda = np.zeros(shape)
    Dmu = np.zeros(shape)
    Drho = np.zeros(shape)

    for lz in range(N_z):
        p = phi(lz) if callable(phi) else phi
        mask = rng.random((M, M)) < p
        Dlambda[lz, mask] = contrast.Dlambda
        Dmu[lz, mask] = contrast.Dmu
        Drho[lz, mask] = contrast.Drho

    return SlabMaterial(Dlambda=Dlambda, Dmu=Dmu, Drho=Drho, ref=ref)


def uniform_slab_material(
    geometry: SlabGeometry,
    ref: ReferenceMedium,
    contrast: MaterialContrast,
) -> SlabMaterial:
    """Create a uniform slab where all cubes have the same contrast.

    Args:
        geometry: Slab lattice geometry.
        ref: Background elastic medium.
        contrast: Material contrast for all cubes.

    Returns:
        SlabMaterial with uniform contrasts.
    """
    shape = (geometry.N_z, geometry.M, geometry.M)
    return SlabMaterial(
        Dlambda=np.full(shape, contrast.Dlambda),
        Dmu=np.full(shape, contrast.Dmu),
        Drho=np.full(shape, contrast.Drho),
        ref=ref,
    )
