"""3D Slab Foldy-Lax Multiple Scattering Solver.

Solves the Foldy-Lax system ``(I - G·T)ψ = ψ⁰`` for an M×M×N_z lattice
of cubes with individual elastic properties.  Uses FFT convolution in the
horizontal plane for O(N_z² × M² log M) matvec cost.

Coordinate system: z=0 (down), x=1 (right), y=2 (out) — seismological.
Voigt ordering: (zz, xx, yy, xy, zy, zx) with engineering halving.
"""

from dataclasses import dataclass
from typing import Callable

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
        if (
            self.Dlambda.shape != self.Dmu.shape
            or self.Dlambda.shape != self.Drho.shape
        ):
            msg = (
                f"Contrast array shapes must match: "
                f"Dlambda={self.Dlambda.shape}, Dmu={self.Dmu.shape}, "
                f"Drho={self.Drho.shape}"
            )
            raise ValueError(msg)
        if self.Dlambda.ndim != 3:
            msg = (
                f"Contrast arrays must be 3D (N_z, M, M), got ndim={self.Dlambda.ndim}"
            )
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
) -> NDArray:
    """Build 9×9 T-matrix for each cube in the slab.

    Caches by unique (Δλ, Δμ, Δρ) triples for efficiency with
    binary random media.

    Args:
        geometry: Slab lattice geometry.
        material: Per-cube material contrasts.
        omega: Angular frequency (rad/s).

    Returns:
        T-matrices, shape (N_z, M, M, 9, 9), complex.
    """
    M, N_z, a = geometry.M, geometry.N_z, geometry.a
    ref = material.ref
    T_all = np.zeros((N_z, M, M, 9, 9), dtype=complex)
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
                    result = compute_cube_tmatrix(omega, a, ref, contrast)
                    cache[key] = _sub_cell_tmatrix_9x9(result, omega, a)
                T_all[lz, i, j] = cache[key]

    return T_all


# ═══════════════════════════════════════════════════════════════
#  2. D4h symmetry helpers
# ═══════════════════════════════════════════════════════════════


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

    for k in range(n_dz):
        dz_vox = k - (N_z - 1)
        dz = dz_vox * d
        kernel_spatial = np.zeros((S, S, 9, 9), dtype=complex)

        # Fundamental domain: 0 ≤ dy ≤ dx, dx ∈ [0, M-1]
        for dx in range(M):
            for dy in range(dx + 1):
                if dx == 0 and dy == 0 and abs(dz) < 1e-15 * max(d, 1.0):
                    continue  # self-term zeroed

                is_nn = max(abs(dz_vox), dx, dy) <= 1

                if volume_averaged and is_nn:
                    # Call inter_voxel_propagator_9x9 for each orbit point
                    # (it has its own O_h rotation, so pass signed offsets)
                    for sdx, sdy, _transform in _d4h_orbit(dx, dy):
                        R_lattice = (dz_vox, sdx, sdy)
                        G0 = inter_voxel_propagator_9x9(
                            R_lattice,
                            ref.alpha,
                            ref.beta,
                            ref.rho,
                            omega,
                            n_orders,
                        )
                        kernel_spatial[sdx + M - 1, sdy + M - 1] = G0
                else:
                    r_vec = np.array([dz, dx * d, dy * d])
                    G0 = _propagator_block_9x9(r_vec, omega, ref)

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
        wave_type: 'P' or 'S'.

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
            # Vertical incidence — pick x-direction
            pol = np.array([0.0, 1.0, 0.0])
        else:
            pol = np.cross(cross, k_hat)
            pol = pol / np.linalg.norm(pol)
    else:
        msg = f"wave_type must be 'P' or 'S', got '{wave_type}'"
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


# ═══════════════════════════════════════════════════════════════
#  6. Main solver
# ═══════════════════════════════════════════════════════════════


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
) -> SlabResult:
    """Solve the Foldy-Lax slab scattering problem via GMRES.

    Solves ``(I − G·T)ψ = ψ⁰`` for the exciting field ψ using
    FFT-accelerated matvec.

    Args:
        geometry: Slab lattice geometry.
        material: Per-cube material contrasts.
        omega: Angular frequency (rad/s).
        k_hat: Unit incident propagation direction (z, x, y).
        wave_type: 'P' or 'S'.
        gmres_tol: GMRES relative tolerance.
        max_iter: Maximum GMRES iterations.
        volume_averaged: If True, use volume-averaged inter-voxel propagator
            for nearest-neighbour separations.
        n_orders: Dynamic correction orders for volume-averaged propagator.
        periodic: If True, use circular convolution for an infinite periodic
            slab. Default False gives linear convolution (finite slab).

    Returns:
        SlabResult with exciting and incident fields.
    """
    T_local = compute_slab_tmatrices(geometry, material, omega)
    kernel_hat = _build_slab_kernels(
        geometry,
        omega,
        material.ref,
        volume_averaged=volume_averaged,
        n_orders=n_orders,
        periodic=periodic,
    )
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


def slab_rpp_periodic(
    result: SlabResult, T_local: NDArray, *, p: float = 0.0
) -> complex:
    """Specular P→P reflection coefficient for a periodic slab.

    Uses the Weyl representation: the 2D lattice sum replaces
    exp(ikr)/(4πr) with i/(2k_z d²)·exp(ik_z|z|), giving:

        R_PP = -(i / (2k_z d² ρα²)) × Σ_l Q_P,l × exp(ik_z z_l)

    where k_z = ω·η_P is the vertical P-wavenumber, η_P = √(1/α² - p²),
    and Q_P,l = r̂·f_l − ik_z(r̂·σ_l·r̂) is the far-field P-source scalar
    for layer l, averaged over the M² horizontal cubes.

    For Kennett comparison, use ``periodic=True`` in ``compute_slab_scattering``
    so that the solver's circular convolution matches the infinite-medium
    assumption of the Kennett reflectivity.

    Args:
        result: Solved slab scattering result.
        T_local: Per-cube T-matrices, shape (N_z, M, M, 9, 9).
        p: Horizontal slowness (s/m). Default 0.0 (normal incidence).

    Returns:
        Complex specular P→P reflection coefficient (dimensionless).
    """
    geom = result.geometry
    ref = result.material.ref
    omega = result.omega
    d = geom.d

    # Vertical slowness and wavenumber
    s_P = _complex_slowness(ref.alpha, np.inf)
    eta_P = _vertical_slowness(s_P, p)
    k_z = omega * eta_P

    # Reflected (upgoing) direction: r_hat = [-η_P·α, p·α, 0] in (z,x,y)
    r_hat = np.array([-eta_P * ref.alpha, p * ref.alpha, 0.0])

    # Sources: τ = T·ψ at each cube
    source = np.einsum("lmnab,lmnb->lmna", T_local, result.psi)
    centres = geom.all_centres()

    total = 0.0 + 0j
    for lz in range(geom.N_z):
        # Average source over horizontal cubes in this layer
        force_avg = np.mean(source[lz, :, :, :3], axis=(0, 1))
        sigma_voigt_avg = np.mean(source[lz, :, :, 3:], axis=(0, 1))
        sigma_avg = _voigt_to_tensor(sigma_voigt_avg)

        # Far-field P-source scalar: Q_P = −r̂·f − ik_z(r̂·σ·r̂)
        # The force sign is negated because the T-matrix convention uses
        # +ω²Δρ V u (opposite to the Lippmann-Schwinger body force −ω²δρ u).
        sigma_rr = np.dot(r_hat, sigma_avg @ r_hat)
        Q_P = -np.dot(r_hat, force_avg) - 1j * k_z * sigma_rr

        # Weyl propagation phase to surface
        z_l = centres[lz, 0, 0, 0]
        phase = np.exp(1j * k_z * z_l)

        total += Q_P * phase

    # Weyl prefactor: −i/(2k_z d² ρα²)
    return complex(-1j / (2.0 * k_z * d**2 * ref.rho * ref.alpha**2) * total)


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
    # Background moduli
    lam_bg = ref.rho * (ref.alpha**2 - 2.0 * ref.beta**2)
    mu_bg = ref.rho * ref.beta**2

    # Perturbed layer velocities
    lam_p = lam_bg + contrast.Dlambda
    mu_p = mu_bg + contrast.Dmu
    rho_p = ref.rho + contrast.Drho
    alpha_p = float(np.sqrt((lam_p + 2.0 * mu_p) / rho_p))
    beta_p = float(np.sqrt(mu_p / rho_p))

    # 3-layer stack: background(dummy) | perturbed(H) | background(halfspace)
    stack = LayerStack(
        layers=[
            IsotropicLayer(
                alpha=ref.alpha, beta=ref.beta, rho=ref.rho, thickness=100.0
            ),
            IsotropicLayer(alpha=alpha_p, beta=beta_p, rho=rho_p, thickness=H),
            IsotropicLayer(
                alpha=ref.alpha, beta=ref.beta, rho=ref.rho, thickness=np.inf
            ),
        ]
    )

    result = kennett_layers(stack, p=0.0, omega=np.array([omega]))
    return complex(result.RPP[0])


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
