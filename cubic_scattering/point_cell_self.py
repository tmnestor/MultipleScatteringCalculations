"""The point (collocation) cell's self term and its exact single-site T-matrix, at any frequency.

WHAT THE POINT CELL NEEDS
-------------------------
The classical point-matching cell takes the nine-component state e = (u, strain) as uniform in the cube and
imposes the equation at the centre. Its own field there is S_self D e, with

    S_self = integral over the cube of P(y) dy,

P the 9x9 point propagator and D = diag(omega^2 drho I_3, dC_V) the contrast. So e = psi + S_self D e and

    T = d^3 D (I - S_self D)^{-1},

the uniform-strain closure. Statically S_self is the cube's depolarisation (Chiu 1977, Nozaki 2001). At
finite frequency it is new, and this module evaluates it exactly.

WHY THE FACES
-------------
The strain-from-moment block of P is a fourth derivative of D = g_S - g_P (and a second of g_S), which
decays as r^-3 at the centre: not absolutely integrable, and its value depends on the excluded region. The
integral is a distribution, and pairing with the cube's indicator moves one derivative onto the faces,

    integral_V d_{i1} d_{i2..in} f dV = sum over faces of n_{i1} integral_face d_{i2..in} f dA,

valid when d^{n-1} f is locally integrable (here at worst r^-2). The faces lie a distance h from the
centre, so the face integrands are smooth and a Gauss rule on each face converges exponentially. The
delta-function content of the inclusion problem, which a volume quadrature misses, is carried by the
divergence theorem itself. The undifferentiated g_S, weakly singular, is written as the divergence of the
radial field F(r) r_hat / r^2 with F(r) = integral_0^r f(s) s^2 ds, which moves it onto the faces too.

Nothing here is a series in k h, so the self term is exact at every frequency, not to a fixed order. The
same face route at a non-zero offset reproduces the receiver-averaged blocks of ``cell_averaged_pair``,
which is how it is checked.

Conventions inherited from ``kupradze_derivatives`` and ``resonance_tmatrix``: g = exp(i kappa r)/(4 pi r),
(z, x, y) ordering, time e^{-i omega t}, the 9x9 [[G, C], [H, S]] of ``_voigt_contract``.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from .cell_averaged_pair import _assemble
from .effective_contrasts import MaterialContrast, ReferenceMedium
from .kupradze_derivatives import (
    MAX_ORDER,
    difference_derivative_tensors,
    scalar_derivative_tensors,
)
from .voigt_tmatrix import effective_stiffness_voigt

#: Gauss points per axis on each quarter of a face. The integrand peaks at the face centre, a distance h
#: from the cube's centre, so on a quarter face its nearest complex singularity lies about one panel width
#: off the interval; 16 points give about 1e-13 (``test_point_cell_self``).
N_FACE = 16

#: Gauss points for F(r) = r^2 integral_0^1 t exp(i kappa r t) dt / (4 pi), smooth on [0, 1].
N_RADIAL = 24


def _face_nodes(centre: NDArray, h: float, n: int) -> list[tuple[int, float, NDArray, NDArray]]:
    """(axis, sign, points, weights) for the six faces, each split into four panels."""
    x, w = np.polynomial.legendre.leggauss(n)
    half = 0.5 * h
    # two panels [-h, 0] and [0, h] per face axis
    u = np.concatenate([half * x - half, half * x + half])
    wu = np.concatenate([half * w, half * w])
    uu, vv = np.meshgrid(u, u, indexing="ij")
    ww = np.outer(wu, wu).ravel()
    faces = []
    for axis in range(3):
        others = [b for b in range(3) if b != axis]
        for sign in (-1.0, 1.0):
            pts = np.empty((uu.size, 3))
            pts[:, axis] = centre[axis] + sign * h
            pts[:, others[0]] = centre[others[0]] + uu.ravel()
            pts[:, others[1]] = centre[others[1]] + vv.ravel()
            faces.append((axis, sign, pts, ww))
    return faces


def cube_integral_tensors(
    centre: NDArray,
    h: float,
    tensors_at: Callable[[NDArray], list[NDArray]],
    kappa: complex | None,
    order: int = MAX_ORDER,
    n_face: int = N_FACE,
) -> list[NDArray]:
    """Integrals over the cube of the derivative tensors d_{i1..in} f, n = 0..order, taken as distributions.

    Args:
        centre: The cube's centre relative to the source, shape (3,); the origin for the self term.
        h: The cube's half-width.
        tensors_at: The derivative tensors of f at a point, orders 0..order - 1 at least (as
            ``kupradze_derivatives.scalar_derivative_tensors``).
        kappa: The wavenumber when f = exp(i kappa r)/(4 pi r), whose order-0 integral is moved onto the
            faces by the radial field; None when f is regular at the origin (D = g_S - g_P), whose order-0
            integral is then not needed and returned as NaN.
        order: The highest derivative order.
        n_face: Gauss points per axis on each quarter face.

    Returns:
        A list whose n-th entry has shape (3,)*n, complex: integral_V d^n f dV (not the average).
    """
    c = np.asarray(centre, dtype=float)
    out: list[NDArray] = [np.zeros((3,) * n, dtype=complex) for n in range(order + 1)]
    out[0] = np.array(np.nan + 0j) if kappa is None else np.array(0j)
    t, wt = np.polynomial.legendre.leggauss(N_RADIAL)
    t, wt = 0.5 * (t + 1.0), 0.5 * wt
    for axis, sign, pts, wts in _face_nodes(c, h, n_face):
        for p, w in zip(pts, wts, strict=True):
            tens = tensors_at(p)
            for n in range(1, order + 1):
                # n_{i1} times the order n-1 tensor: only the face's own axis in the first index
                out[n][axis] = out[n][axis] + sign * w * tens[n - 1]
            if kappa is not None:
                r = float(np.linalg.norm(p))
                f_r = r**2 * np.sum(wt * t * np.exp(1j * kappa * r * t)) / (4.0 * np.pi)
                out[0] = out[0] + w * f_r * sign * p[axis] / r**3
    return out


def self_block_9x9(omega: complex, ref: ReferenceMedium, d: float, n_face: int = N_FACE) -> NDArray:
    """S_self: the 9x9 point propagator integrated over the cube of side ``d`` about the source, exactly.

    Args:
        omega: Angular frequency (may be complex).
        ref: Background medium.
        d: Cell SIDE.
        n_face: Gauss points per axis on each quarter face.

    Returns:
        Shape (9, 9), complex, in the layout of ``cell_averaged_pair.averaged_pair_block_9x9`` times d^3.
    """
    return offset_block_9x9(np.zeros(3), omega, ref, d, n_face=n_face)


def offset_block_9x9(
    s_vec: NDArray, omega: complex, ref: ReferenceMedium, d: float, n_face: int = N_FACE
) -> NDArray:
    """The 9x9 point propagator integrated over the cube of side ``d`` centred at ``s_vec`` from the source.

    At a non-zero lattice offset it is d^3 times ``averaged_pair_block_9x9``; at the origin it is S_self.
    """
    k_p, k_s = omega / ref.alpha, omega / ref.beta
    h = 0.5 * d
    d_diff = cube_integral_tensors(
        s_vec,
        h,
        lambda r: difference_derivative_tensors(r, k_p, k_s, MAX_ORDER - 1),
        None,
        n_face=n_face,
    )
    d_s = cube_integral_tensors(s_vec, h, lambda r: scalar_derivative_tensors(r, k_s, 1), k_s, 2, n_face)
    return _assemble(d_diff, d_s, omega, ref)


def contrast_9x9(omega: complex, contrast: MaterialContrast) -> NDArray:
    """D = diag(omega^2 drho I_3, dC_V), the Born limit of ``_sub_cell_tmatrix_9x9`` divided by d^3."""
    out = np.zeros((9, 9), dtype=complex)
    out[:3, :3] = omega**2 * contrast.Drho * np.eye(3)
    out[3:, 3:] = effective_stiffness_voigt(contrast.Dlambda, contrast.Dmu, contrast.Dmu)
    return out


def exact_point_tmatrix_9x9(
    omega: complex, d: float, contrast: MaterialContrast, s_self: NDArray
) -> NDArray:
    """The uniform-strain closure T = d^3 D (I - S_self D)^{-1} with the exact self term.

    Args:
        omega: Angular frequency.
        d: Cell SIDE.
        contrast: The cell's contrast.
        s_self: ``self_block_9x9(omega, ref, d)``, which does not depend on the contrast.

    Returns:
        Shape (9, 9), complex, in the layout of ``_sub_cell_tmatrix_9x9``.
    """
    dm = contrast_9x9(omega, contrast)
    return d**3 * dm @ np.linalg.inv(np.eye(9) - s_self @ dm)
