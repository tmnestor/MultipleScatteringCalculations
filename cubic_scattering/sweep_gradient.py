#!/usr/bin/env python3
"""Adjoint-state misfit gradient for the directional-sweep Foldy-Lax solver.

The derivation and every validation behind it are note (IV),
LatexPDFs/AdjointStateGradients. For one source, frequency and k_y node:

    forward    (I - G0(k_y) T0) psi = psi_inc                    solve_sweep_foldy_lax
    data       d = R T0 psi                                      ReceiverMap
    misfit     M = 1/2 ||d - d_obs||^2
    adjoint    (I - G0^T T0^T) psi_adj = R^T conj(r)             adjoint_solve
    gradient   dM/dm_{i,l} = Re psi_adj_i^T (dT0_i/dm_l) psi_i  local 9x9 contraction

psi_adj is the note's tilde-psi: the thesis pairs the incident field b with
b-tilde radiated from the receiver (VariationalSum.tex, Eq. btdef), and
psi_adj is the exciting field that b-tilde drives.

THE ADJOINT NEEDS NO NEW OPERATOR. Gate A1 measured
G0(k_y)^T = W_pm G0(-k_y) W_pm^-1 on the assembled sweep operator, with
W_pm = I_pm W = diag(1,1,1, -1,-1,-1, -1/2,-1/2,-1/2), and gate_t0_reciprocity
T0^T = W_pm T0 W_pm^-1 for the Rayleigh cube. Together

    (I - G0^T T0^T) = W_pm (I - G0(-k_y) T0) W_pm^-1,

so the adjoint is the FORWARD solve at -k_y, on the right-hand side W_pm^-1 b_tilde,
followed by psi_adj = W_pm psi_minus. The resonance composite T0 fails that law (1.6-17%) and
must not be used here; only the Rayleigh cube is supported.

THE RECEIVER MAP IS DENSE. R has one row per recorded component and one column
per voxel state entry -- small -- so R^T is exact by construction. Gate A1b
showed R^T r equals W_pm times the field of a point force at the receiver; that is
the route to take if R ever becomes too large to store, not before.

dT0/dm IS A RICHARDSON DIFFERENCE, not complex step. T0 is not holomorphic in
the contrast (compute_cube_tmatrix applies its real form factors to the real
parts only), so complex step is wrong by 4e-4 to 2e-2; gate A2 measured the
Richardson derivative to <= 2.5e-11.

A 2.5-D gradient is a sum over k_y nodes with the quadrature weight and the
phase e^{i k_y y_r} on each node's residual; this module works at ONE node and
leaves that sum to the caller.

Coordinates: z = axis 0 (down), x = axis 1 (right), y = axis 2 (out).
State: (u_z, u_x, u_y, e_zz, e_xx, e_yy, 2e_xy, 2e_zy, 2e_zx).
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .directional_sweeps import G0Cache, SweepGrid, apply_g0
from .effective_contrasts import MaterialContrast, ReferenceMedium, compute_cube_tmatrix
from .resonance_tmatrix import _sub_cell_tmatrix_9x9
from .sweep_kernels import vertical_kernel_9x9
from .sweep_solver import solve_sweep_foldy_lax

# W_pm = I_pm W: the Voigt engineering weight W times the displacement-strain parity I_pm
# (gate_a1_sweep_reciprocity, gate_t0_reciprocity).
W_SIGNED = np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -0.5, -0.5, -0.5])


@dataclass(frozen=True)
class ReceiverMap:
    """A dense linear map from voxel contrast sources to recorded data.

    Attributes:
        matrix: Shape (n_data, n_z * n_x * 9); column order is the C-order ravel
            of an (n_z, n_x, 9) state.
        state_shape: (n_z, n_x, 9).
    """

    matrix: NDArray
    state_shape: tuple[int, int, int]

    @property
    def n_data(self) -> int:
        """Number of recorded data values."""
        return int(self.matrix.shape[0])

    def apply(self, sources: NDArray) -> NDArray:
        """Data recorded from contrast sources of shape state_shape."""
        return self.matrix @ sources.ravel()

    def apply_transpose(self, data: NDArray) -> NDArray:
        """R^T applied to a data vector, returned in state_shape."""
        return (self.matrix.T @ data).reshape(self.state_shape)


@dataclass(frozen=True)
class GradientResult:
    """Misfit, gradient and the fields that produced them.

    Attributes:
        misfit: Misfit 1/2 ||d - d_obs||^2.
        grad: dM/dm, shape (n_z, n_x, 3), parameters (Dlambda, Dmu, Drho).
        data: Predicted data d = R T0 psi, shape (n_data,).
        psi: Forward exciting field, shape (n_z, n_x, 9).
        psi_adj: Adjoint field, shape (n_z, n_x, 9).
    """

    misfit: float
    grad: NDArray
    data: NDArray
    psi: NDArray
    psi_adj: NDArray


def whole_space_receiver_map(
    grid: SweepGrid,
    ref: ReferenceMedium,
    omega: complex,
    *,
    receiver_x: NDArray,
    receiver_dz: float,
    rows: Sequence[int],
) -> ReceiverMap:
    """Receiver map for a whole-space background, at the grid's k_y.

    Each receiver sits at lateral position x_r on a line at depth receiver_dz
    below plane 0, and records the state components ``rows``. The kernel from
    plane z is the whole-space plane-to-plane kernel at dz = receiver_dz - z
    pitch, carried to x_r by the same k_x quadrature the vertical sweep uses:

        R[(r, c), (z, j, b)] = sum_k w_k e^{i kx (x_r - x_j)} K_{c b}(kx; dz).

    Args:
        grid: The sweep grid; its k_x nodes, weights and k_y are used.
        ref: Background medium.
        omega: Complex angular frequency.
        receiver_x: Receiver lateral positions, km, shape (n_rcv,).
        receiver_dz: Receiver depth below plane 0, km. Must not coincide with a
            voxel plane.
        rows: State components recorded, e.g. (0, 1, 2) for displacement.

    Returns:
        A ReceiverMap with n_data = n_rcv * len(rows).

    Raises:
        ValueError: if the receiver line lies on a voxel plane.
    """
    kernels = []
    for z in range(grid.n_z):
        dz = float(receiver_dz) - z * grid.pitch
        _check_off_plane(dz, grid.pitch, z)
        kernels.append(vertical_kernel_9x9(grid.kx_nodes, grid.ky, dz, omega, ref))
    return _assemble(grid, kernels, np.asarray(receiver_x, dtype=float), tuple(rows))


def layered_receiver_map(
    grid: SweepGrid,
    model: object,
    omega: complex,
    *,
    receiver_x: NDArray,
    receiver_iface: int,
    plane_ifaces: Sequence[int],
    rows: Sequence[int],
) -> ReceiverMap:
    """Receiver map for a stratified background, at the grid's k_y.

    The layered counterpart of ``whole_space_receiver_map``: the kernel from
    plane z to the receiver interface is ``corrected_layered_9x9``, the object
    the vertical sweep carries between planes, and it is taken to the receiver
    by the same k_x quadrature.

    Args:
        grid: The sweep grid; its k_x nodes, weights and k_y are used.
        model: A ``LayerModel``.
        omega: Angular frequency.
        receiver_x: Receiver lateral positions, km, shape (n_rcv,).
        receiver_iface: Interface index of the receiver line: interior to a
            layer, and not one of the voxel planes.
        plane_ifaces: Interface index of each voxel plane, length n_z.
        rows: State components recorded, e.g. (0, 1, 2) for displacement.

    Returns:
        A ReceiverMap with n_data = n_rcv * len(rows).

    Raises:
        ValueError: on a plane map of the wrong length, or a receiver on a plane.
    """
    from .layered_correction import corrected_layered_9x9

    if len(plane_ifaces) != grid.n_z:
        msg = (
            f"plane_ifaces has {len(plane_ifaces)} entries but the grid has n_z={grid.n_z}.\n"
            "  Where: cubic_scattering/sweep_gradient.py, layered_receiver_map(plane_ifaces=...)\n"
            "  Valid: one interface index per voxel plane, e.g. plane_ifaces=(7, 11)\n"
            "  Fix:   pass the plane map of the forward cache's LayeredBackground."
        )
        raise ValueError(msg) from None
    if receiver_iface in plane_ifaces:
        msg = (
            f"receiver_iface={receiver_iface} is one of the voxel planes {tuple(plane_ifaces)}.\n"
            "  Where: cubic_scattering/sweep_gradient.py, layered_receiver_map(receiver_iface=...)\n"
            "  Valid: an interface interior to a layer and not in plane_ifaces\n"
            "  Fix:   the plane-to-plane kernel diverges at equal depth; move the\n"
            "         receiver line to a different interface."
        )
        raise ValueError(msg) from None

    ky_arr = np.full(grid.kx_nodes.size, grid.ky)
    kernels = []
    for z in range(grid.n_z):
        g9 = corrected_layered_9x9(
            model, omega, grid.kx_nodes, ky_arr, source_iface=plane_ifaces[z], receiver_iface=receiver_iface
        )
        kernels.append(np.moveaxis(g9, 0, -1))
    return _assemble(grid, kernels, np.asarray(receiver_x, dtype=float), tuple(rows))


def _check_off_plane(dz: float, pitch: float, z: int) -> None:
    """Reject a receiver line at a voxel plane's depth."""
    if abs(dz) < 1e-9 * pitch:
        msg = (
            f"the receiver line lies on voxel plane {z} (dz = {dz:g} km).\n"
            "  Where: cubic_scattering/sweep_gradient.py, whole_space_receiver_map(receiver_dz=...)\n"
            "  Valid: a depth that is not a multiple of the pitch below plane 0, e.g.\n"
            "         receiver_dz=-1.0 for a line 1 km above the grid\n"
            "  Fix:   the plane-to-plane kernel diverges at equal depth; move the\n"
            "         receiver line off the voxel planes."
        )
        raise ValueError(msg) from None


def _assemble(
    grid: SweepGrid, kernels: list[NDArray], receiver_x: NDArray, rows: tuple[int, ...]
) -> ReceiverMap:
    """Dense R from per-plane kernels of shape (9, 9, n_kx)."""
    x_vox = np.arange(grid.n_x) * grid.pitch
    # phase[r, j, k] = w_k e^{i kx (x_r - x_j)}
    phase = grid.kx_weights[None, None, :] * np.exp(
        1j * grid.kx_nodes[None, None, :] * (receiver_x[:, None, None] - x_vox[None, :, None])
    )
    sel = np.asarray(rows)
    n_rcv = receiver_x.size
    mat = np.zeros((n_rcv, sel.size, grid.n_z, grid.n_x, 9), dtype=complex)
    for z, ker in enumerate(kernels):
        mat[:, :, z] = np.einsum("rjk,cbk->rcjb", phase, ker[sel])
    return ReceiverMap(
        matrix=mat.reshape(n_rcv * sel.size, grid.n_z * grid.n_x * 9),
        state_shape=(grid.n_z, grid.n_x, 9),
    )


def rayleigh_t0_and_derivative(
    contrasts: NDArray,
    plane_refs: Sequence[ReferenceMedium],
    omega: complex,
    half_width: float,
    *,
    step: NDArray,
) -> tuple[NDArray, NDArray]:
    """Rayleigh cube T0 per voxel and its derivative in (Dlambda, Dmu, Drho).

    The derivative is one Richardson extrapolation of central differences at
    steps h and h/2, h/2 and h/4 -- the scheme gate A2 measured to <= 2.5e-11.

    Args:
        contrasts: Per-voxel (Dlambda, Dmu, Drho), shape (n_z, n_x, 3), GPa and
            g/cm3, relative to the voxel's own plane medium.
        plane_refs: Reference medium of each plane, length n_z.
        omega: Complex angular frequency.
        half_width: Cube half-width a, km (half the pitch).
        step: Base Richardson step per parameter, shape (3,).

    Returns:
        (t0, dt0) of shapes (n_z, n_x, 9, 9) and (n_z, n_x, 3, 9, 9).

    Raises:
        ValueError: on a wrong contrast shape or plane-medium count.
    """
    contrasts = np.asarray(contrasts, dtype=float)
    if contrasts.ndim != 3 or contrasts.shape[-1] != 3:
        msg = (
            f"contrasts has shape {contrasts.shape}, expected (n_z, n_x, 3).\n"
            "  Where: cubic_scattering/sweep_gradient.py, rayleigh_t0_and_derivative(contrasts=...)\n"
            "  Valid: one (Dlambda, Dmu, Drho) triple per voxel\n"
            "  Fix:   stack the per-voxel contrasts along a last axis of length 3."
        )
        raise ValueError(msg) from None
    n_z, n_x, _ = contrasts.shape
    if len(plane_refs) != n_z:
        msg = (
            f"plane_refs has {len(plane_refs)} entries but contrasts has n_z={n_z}.\n"
            "  Where: cubic_scattering/sweep_gradient.py, rayleigh_t0_and_derivative(plane_refs=...)\n"
            "  Valid: one ReferenceMedium per depth plane\n"
            "  Fix:   pass the local medium of each plane, as the forward cache uses."
        )
        raise ValueError(msg) from None

    def cube(m: NDArray, ref: ReferenceMedium) -> NDArray:
        contrast = MaterialContrast(Dlambda=m[0], Dmu=m[1], Drho=m[2])
        res = compute_cube_tmatrix(omega, half_width, ref, contrast)
        return _sub_cell_tmatrix_9x9(res, omega, half_width)

    t0 = np.zeros((n_z, n_x, 9, 9), dtype=complex)
    dt0 = np.zeros((n_z, n_x, 3, 9, 9), dtype=complex)
    for iz in range(n_z):
        ref = plane_refs[iz]
        for ix in range(n_x):
            m = contrasts[iz, ix]
            t0[iz, ix] = cube(m, ref)
            for a in range(3):
                e = np.zeros(3)
                e[a] = 1.0
                d = []
                for j in range(3):
                    h = float(step[a]) / 2**j
                    d.append((cube(m + h * e, ref) - cube(m - h * e, ref)) / (2.0 * h))
                dt0[iz, ix, a] = (4.0 * d[2] - d[1]) / 3.0
    return t0, dt0


def adjoint_solve(cache_minus: G0Cache, t0: NDArray, rhs: NDArray, *, tol: float, max_iter: int) -> NDArray:
    """Solve (I - G0(k_y)^T T0^T) psi_adj = rhs as the forward solve at -k_y.

    Args:
        cache_minus: Sweep cache built at -k_y.
        t0: Rayleigh T0 blocks, shape (n_z, n_x, 9, 9).
        rhs: Right-hand side, shape (n_z, n_x, 9).
        tol: GMRES relative tolerance.
        max_iter: GMRES iteration cap.

    Returns:
        psi_adj, shape (n_z, n_x, 9).
    """
    psi_minus = solve_sweep_foldy_lax(cache_minus, t0, rhs / W_SIGNED, tol=tol, max_iter=max_iter).psi
    return psi_minus * W_SIGNED


def misfit_and_gradient(
    cache_plus: G0Cache,
    cache_minus: G0Cache,
    t0: NDArray,
    dt0: NDArray,
    psi_inc: NDArray,
    receiver: ReceiverMap,
    d_obs: NDArray,
    *,
    tol: float,
    max_iter: int,
) -> GradientResult:
    """Misfit and its gradient in the voxel contrasts, at one source and k_y node.

    ``d_obs`` is the SCATTERED data: observed minus the reference response,
    which does not depend on the voxel contrasts.

    Args:
        cache_plus: Sweep cache at +k_y (the forward problem).
        cache_minus: Sweep cache at -k_y (the adjoint problem).
        t0: Rayleigh T0 blocks, shape (n_z, n_x, 9, 9).
        dt0: dT0/dm, shape (n_z, n_x, 3, 9, 9).
        psi_inc: Incident field, shape (n_z, n_x, 9).
        receiver: The receiver map at +k_y.
        d_obs: Observed scattered data, shape (n_data,).
        tol: GMRES relative tolerance for both solves.
        max_iter: GMRES iteration cap for both solves.

    Returns:
        A GradientResult.

    Raises:
        ValueError: if the caches are not at opposite k_y, or d_obs has the
            wrong length.
    """
    if not np.isclose(cache_minus.grid.ky, -cache_plus.grid.ky, rtol=0.0, atol=1e-12):
        msg = (
            f"cache_minus is at k_y={cache_minus.grid.ky:g}, not -{cache_plus.grid.ky:g}.\n"
            "  Where: cubic_scattering/sweep_gradient.py, misfit_and_gradient(cache_minus=...)\n"
            "  Valid: two caches from the same grid settings at k_y and -k_y\n"
            "  Fix:   build cache_minus with make_sweep_grid(..., ky=-ky); the adjoint is\n"
            "         the forward solve at the OPPOSITE k_y (gate A1)."
        )
        raise ValueError(msg) from None
    d_obs = np.asarray(d_obs)
    if d_obs.shape != (receiver.n_data,):
        msg = (
            f"d_obs has shape {d_obs.shape}, expected ({receiver.n_data},).\n"
            "  Where: cubic_scattering/sweep_gradient.py, misfit_and_gradient(d_obs=...)\n"
            "  Valid: one value per receiver row, ordered (receiver, component)\n"
            "  Fix:   build d_obs with the same receiver map and component rows."
        )
        raise ValueError(msg) from None

    psi = solve_sweep_foldy_lax(cache_plus, t0, psi_inc, tol=tol, max_iter=max_iter).psi
    tau = np.einsum("zxab,zxb->zxa", t0, psi)
    data = receiver.apply(tau)
    r = data - d_obs
    misfit = 0.5 * float(np.vdot(r, r).real)
    grad, psi_adj = _adjoint_product(cache_minus, t0, dt0, psi, receiver, r, tol=tol, max_iter=max_iter)
    return GradientResult(misfit=misfit, grad=grad, data=data, psi=psi, psi_adj=psi_adj)


def frechet_vector_product(
    cache_plus: G0Cache,
    t0: NDArray,
    dt0: NDArray,
    psi: NDArray,
    receiver: ReceiverMap,
    dm: NDArray,
    *,
    tol: float,
    max_iter: int,
) -> NDArray:
    """Tangent-linear data perturbation F dm, by one further forward solve.

    Differentiating (I - G0 T0) psi = psi_inc and d = R T0 psi:

        s       = dT0 psi,      dT0 = sum_a (dT0/dm_a) dm_a     (per voxel)
        dpsi    = (I - G0 T0)^-1 G0 s
        F dm    = R (s + T0 dpsi).

    Args:
        cache_plus: Sweep cache at +k_y, as for the forward solve.
        t0: Rayleigh T0 blocks, shape (n_z, n_x, 9, 9).
        dt0: dT0/dm, shape (n_z, n_x, 3, 9, 9).
        psi: Forward exciting field at the same model, shape (n_z, n_x, 9).
        receiver: The receiver map at +k_y.
        dm: Contrast perturbation, shape (n_z, n_x, 3).
        tol: GMRES relative tolerance.
        max_iter: GMRES iteration cap.

    Returns:
        F dm, shape (n_data,).
    """
    s = np.einsum("zxaij,zxa,zxj->zxi", dt0, dm, psi)
    dpsi = solve_sweep_foldy_lax(cache_plus, t0, apply_g0(s, cache_plus), tol=tol, max_iter=max_iter).psi
    return receiver.apply(s + np.einsum("zxab,zxb->zxa", t0, dpsi))


def frechet_adjoint_product(
    cache_minus: G0Cache,
    t0: NDArray,
    dt0: NDArray,
    psi: NDArray,
    receiver: ReceiverMap,
    q: NDArray,
    *,
    tol: float,
    max_iter: int,
) -> NDArray:
    """Re(F^H q), the transpose of ``frechet_vector_product`` in the real inner product.

    For any real dm and complex q, Re(q^H F dm) = sum(dm * Re(F^H q)) -- the
    dot-product identity gate A4 checks. With q the data residual this IS the
    misfit gradient, and ``misfit_and_gradient`` computes it through here.

    Args:
        cache_minus: Sweep cache at -k_y (the adjoint problem).
        t0: Rayleigh T0 blocks, shape (n_z, n_x, 9, 9).
        dt0: dT0/dm, shape (n_z, n_x, 3, 9, 9).
        psi: Forward exciting field, shape (n_z, n_x, 9).
        receiver: The receiver map at +k_y.
        q: Data-space vector, shape (n_data,).
        tol: GMRES relative tolerance.
        max_iter: GMRES iteration cap.

    Returns:
        Re(F^H q), shape (n_z, n_x, 3).
    """
    return _adjoint_product(cache_minus, t0, dt0, psi, receiver, q, tol=tol, max_iter=max_iter)[0]


def _adjoint_product(cache_minus, t0, dt0, psi, receiver, q, *, tol, max_iter) -> tuple[NDArray, NDArray]:
    """Re(F^H q) and the adjoint field psi_adj it was contracted with."""
    psi_adj = adjoint_solve(
        cache_minus, t0, receiver.apply_transpose(np.conj(q)), tol=tol, max_iter=max_iter
    )
    return np.einsum("zxi,zxaij,zxj->zxa", psi_adj, dt0, psi).real, psi_adj
