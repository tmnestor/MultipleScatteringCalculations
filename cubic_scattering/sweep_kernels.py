#!/usr/bin/env python3
"""Kernel algebra for the Cartesian directional-sweep propagator.

The lateral (k_x-pole) propagator factorises exactly into a
separation-independent 9x9 amplitude and a scalar one-pitch phase, one pair per
pole:

    P(ky, kz; dx) = M_P(ky, kz) e^{i kx_P dx} + M_S(ky, kz) e^{i kx_S dx}

because the only dx dependence anywhere in the bundled kernel
(horizontal_greens.post_kx_residue_kernel_9x9_vec) is the two exponentials. The
T-isotropic and T-polarisation terms share a pole, hence a phase, so there are
two accumulators per direction rather than three.

A sweep applies the amplitude once, at readout, and accumulates only the phase.
Getting that division wrong produces a field that is wrong by a
distance-dependent factor and that every reciprocity check will still pass,
because those checks are homogeneous of degree one and blind to exactly this.
See the rung-1 test.

Coordinates: z = axis 0 (down), x = axis 1 (right), y = axis 2 (out).
Time convention e^{-i omega t}; every transverse wavenumber pinned to Im >= 0.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .effective_contrasts import ReferenceMedium
from .resonance_tmatrix import VOIGT_PAIRS

_DIRECTIONS = ("right", "left")

# Parity of each of the nine state components under x -> -x, with the state
# ordered (u_z, u_x, u_y, e_zz, e_xx, e_yy, 2e_xy, 2e_zy, 2e_zx). A component is
# odd iff it carries an ODD number of x indices: u_x, e_xy and e_zx. Note e_xx
# carries two and is therefore EVEN -- the easy mistake here is to pattern-match
# on the letter x rather than count it.
R9 = np.array([1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0])


@dataclass(frozen=True)
class LateralSplit:
    """Separation-independent amplitudes and one-pitch phases, per pole.

    Attributes:
        amp_p: P-pole amplitude, shape (9, 9, n_kz).
        amp_s: S-pole amplitude (isotropic plus polarisation), shape (9, 9, n_kz).
        phase_p: e^{i kx_P pitch}, shape (n_kz,).
        phase_s: e^{i kx_S pitch}, shape (n_kz,).
        pitch: The voxel pitch these phases were built for.
        direction: 'right' (+x) or 'left' (-x).
    """

    amp_p: NDArray
    amp_s: NDArray
    phase_p: NDArray
    phase_s: NDArray
    pitch: float
    direction: str


def _branch(k2: NDArray) -> NDArray:
    """Square root with the outgoing branch pinned to Im >= 0."""
    k = np.sqrt(k2 + 0j)
    return np.where(np.imag(k) < 0, -k, k)


def _add_s_block(
    val: NDArray,
    g_pole: NDArray,
    k_vec: list,
    p: int,
    q: int,
    m: int,
    n: int,
) -> None:
    """Accumulate one pole's contribution to S[alpha, beta].

    S[alpha, beta] = sum_poles (-k_a k_b) G^pole_{ij}, Voigt-contracted on both
    pairs. Ported unchanged from horizontal_greens._add_S_block_pole.
    """
    if p == q and m == n:
        val += -k_vec[p] * k_vec[m] * g_pole[p, m, :]
    elif p == q and m != n:
        val += -k_vec[p] * k_vec[n] * g_pole[p, m, :]
        val += -k_vec[p] * k_vec[m] * g_pole[p, n, :]
    elif p != q and m == n:
        val += -k_vec[q] * k_vec[m] * g_pole[p, m, :]
        val += -k_vec[p] * k_vec[m] * g_pole[q, m, :]
    else:
        val += -k_vec[q] * k_vec[n] * g_pole[p, m, :]
        val += -k_vec[q] * k_vec[m] * g_pole[p, n, :]
        val += -k_vec[p] * k_vec[n] * g_pole[q, m, :]
        val += -k_vec[p] * k_vec[m] * g_pole[q, n, :]


def _assemble_9x9(g_total: NDArray, g_parts: list[NDArray], k_parts: list[list[NDArray]]) -> NDArray:
    """Build [[G, C], [H, S]] from per-pole 3x3 blocks and their k-vectors.

    Each part carries its OWN k-vector, because the x-derivative of a P-pole
    term uses kx_P and of an S-pole term uses kx_S. Collapsing them to a single
    k-vector is the classic defect here.

    Args:
        g_total: Summed 3x3 G block for this pole group, shape (3, 3, n).
        g_parts: The individual 3x3 contributions, each shape (3, 3, n).
        k_parts: One k-vector list [kz, kx, ky] per entry of g_parts.

    Returns:
        P of shape (9, 9, n).
    """
    n = g_total.shape[2]
    p = np.zeros((9, 9, n), dtype=complex)
    p[:3, :3, :] = g_total

    for a, (pp, qq) in enumerate(VOIGT_PAIRS):
        for i in range(3):
            c_val = np.zeros(n, dtype=complex)
            h_val = np.zeros(n, dtype=complex)
            for gpart, kvec in zip(g_parts, k_parts, strict=True):
                if pp == qq:
                    c_val += 1j * kvec[pp] * gpart[i, pp, :]
                    h_val += 1j * kvec[pp] * gpart[pp, i, :]
                else:
                    c_val += 1j * kvec[qq] * gpart[i, pp, :]
                    c_val += 1j * kvec[pp] * gpart[i, qq, :]
                    h_val += 1j * kvec[qq] * gpart[pp, i, :]
                    h_val += 1j * kvec[pp] * gpart[qq, i, :]
            p[i, 3 + a, :] = c_val
            p[3 + a, i, :] = h_val

    for a, (pp, qq) in enumerate(VOIGT_PAIRS):
        for b, (mm, nn) in enumerate(VOIGT_PAIRS):
            val = np.zeros(n, dtype=complex)
            for gpart, kvec in zip(g_parts, k_parts, strict=True):
                _add_s_block(val, gpart, kvec, pp, qq, mm, nn)
            p[3 + a, 3 + b, :] = val

    # Engineering doubling: halve the off-diagonal stress columns (Voigt 3, 4, 5
    # -> state columns 6, 7, 8), matching horizontal_greens._voigt_contract.
    p[:, 6:9, :] *= 0.5

    return p


def lateral_split_9x9(
    ky: float,
    kz_arr: NDArray,
    pitch: float,
    omega: complex,
    ref: ReferenceMedium,
    *,
    direction: str = "right",
) -> LateralSplit:
    """Split the post-k_x-residue 9x9 kernel into amplitude and one-pitch phase.

    Vectorised over k_z at fixed k_y -- the transpose of
    ``horizontal_greens.post_kx_residue_kernel_9x9_vec``, because stage 1 holds
    k_y fixed as the 2.5-D parameter and integrates over k_z. The two are NOT
    interchangeable by relabelling: k_z sits at index 0 of the k-vector and k_y
    at index 2.

    Args:
        ky: Lateral wavenumber out of the plane (the 2.5-D parameter), 1/km.
        kz_arr: Quadrature nodes in k_z, shape (n_kz,), 1/km.
        pitch: Voxel pitch along x, km. Must be > 0.
        omega: Angular complex frequency, rad/s.
        ref: Background medium (seismic units).
        direction: 'right' for +x propagation, 'left' for -x.

    Returns:
        A LateralSplit.

    Raises:
        ValueError: On a non-positive pitch, a zero frequency, or an unknown
            direction.
    """
    if pitch <= 0.0:
        msg = (
            f"pitch must be > 0, got {pitch!r}.\n"
            "  Where: cubic_scattering/sweep_kernels.py, lateral_split_9x9(pitch=...)\n"
            "  Valid: a positive voxel pitch in km, e.g. pitch=0.25\n"
            "  Fix:   pass the SweepGrid's pitch (grid.pitch), not a difference of "
            "centres."
        )
        raise ValueError(msg) from None
    if omega == 0:
        msg = (
            "omega must be non-zero: the partial-wave decomposition is undefined at\n"
            "  zero frequency (both poles collapse to k=0).\n"
            "  Where: cubic_scattering/sweep_kernels.py, lateral_split_9x9(omega=...)\n"
            "  Valid: a complex angular frequency, e.g. omega=2*np.pi*(1+0.03j)\n"
            "  Fix:   drop omega=0 from the frequency list; the static limit needs the\n"
            "         Eshelby route (cube_eshelby.py), not this propagator."
        )
        raise ValueError(msg) from None
    if direction not in _DIRECTIONS:
        msg = (
            f"direction must be one of {_DIRECTIONS}, got {direction!r}.\n"
            "  Where: cubic_scattering/sweep_kernels.py, lateral_split_9x9(direction=...)\n"
            "  Valid: direction='right' (+x) or direction='left' (-x)\n"
            "  Fix:   vertical coupling is not a lateral split -- use "
            "vertical_kernel_9x9."
        )
        raise ValueError(msg) from None

    kz = np.asarray(kz_arr, dtype=float)
    n_kz = kz.size
    rho, alpha, beta = ref.rho, ref.alpha, ref.beta
    kp2 = (omega / alpha) ** 2
    ks2 = (omega / beta) ** 2

    kx_p = _branch(kp2 - ky**2 - kz**2)
    kx_s = _branch(ks2 - ky**2 - kz**2)

    sign = 1.0 if direction == "right" else -1.0

    # k-vectors in seismological order (z, x, y). The x component carries the
    # propagation sign: every odd x-derivative in the C and H blocks flips with
    # it, which is exactly what distinguishes the left sweep from the right.
    kvec_p = [kz.astype(complex), sign * kx_p, np.full(n_kz, ky, dtype=complex)]
    kvec_s = [kz.astype(complex), sign * kx_s, np.full(n_kz, ky, dtype=complex)]

    # Scalar coefficients with the exponential REMOVED -- that is the whole point.
    c_s_iso = (1j / (2 * rho)) / (beta**2 * kx_s)
    c_p_pol = (1j / (2 * rho)) / (omega**2 * kx_p)
    c_s_pol = -(1j / (2 * rho)) / (omega**2 * kx_s)

    g_p = np.zeros((3, 3, n_kz), dtype=complex)
    g_s_iso = np.zeros((3, 3, n_kz), dtype=complex)
    g_s_pol = np.zeros((3, 3, n_kz), dtype=complex)
    for i in range(3):
        g_s_iso[i, i, :] = c_s_iso
        for j in range(3):
            g_p[i, j, :] = kvec_p[i] * kvec_p[j] * c_p_pol
            g_s_pol[i, j, :] = kvec_s[i] * kvec_s[j] * c_s_pol

    amp_p = _assemble_9x9(g_p, [g_p], [kvec_p])
    amp_s = _assemble_9x9(g_s_iso + g_s_pol, [g_s_iso, g_s_pol], [kvec_s, kvec_s])

    phase_p = np.exp(1j * kx_p * pitch)
    phase_s = np.exp(1j * kx_s * pitch)

    return LateralSplit(
        amp_p=amp_p,
        amp_s=amp_s,
        phase_p=phase_p,
        phase_s=phase_s,
        pitch=float(pitch),
        direction=direction,
    )


def same_depth_kernel_9x9(kx_arr: NDArray, ky: float, omega: complex, ref: ReferenceMedium) -> NDArray:
    """Whole-space 9x9 at EQUAL depth, as the limit from ABOVE (dz -> 0-).

    Used only to be SUBTRACTED. On its own this kernel is not integrable over
    k_x: its strain-strain block grows like |k_x|, which is the divergence at
    equal depth that the lateral sweep exists to avoid. It becomes useful in the
    difference

        DeltaG(k_x) = G_layered(j <- j) - same_depth_kernel_9x9(k_x)

    where the divergent direct part cancels exactly and only the layer
    reverberations survive -- and those decay like e^{-kappa 2H}, H the distance
    to the nearest interface. Measured: identically zero for a uniform model at
    every k_x, and falling from 4e-4 to machine zero by k_x = 24 for a model
    with a contrast two layers away.

    THE SIDE MATTERS. The field is discontinuous across the source plane, so the
    two one-sided limits differ -- by a factor approaching 2 at large k_x, not by
    a small amount. ``corrected_layered_9x9`` with source_iface == receiver_iface
    returns the limit from ABOVE, so that is what this reproduces; subtracting
    the other side leaves the whole jump behind instead of cancelling it.

    Args:
        kx_arr: Lateral wavenumber nodes along x, shape (n_kx,), 1/km.
        ky: The 2.5-D lateral parameter, 1/km.
        omega: Complex angular frequency.
        ref: Background medium.

    Returns:
        P of shape (9, 9, n_kx).
    """
    kx = np.asarray(kx_arr, dtype=float)
    n = kx.size
    rho, alpha, beta = ref.rho, ref.alpha, ref.beta
    kh2 = kx**2 + ky**2
    kz_p = _branch((omega / alpha) ** 2 - kh2)
    kz_s = _branch((omega / beta) ** 2 - kh2)

    # sgn = -1: the limit from above. No exponential -- dz is zero.
    kvec_p = [-kz_p, kx.astype(complex), np.full(n, ky, dtype=complex)]
    kvec_s = [-kz_s, kx.astype(complex), np.full(n, ky, dtype=complex)]

    c_s_iso = (1j / (2 * rho)) / (beta**2 * kz_s)
    c_p_pol = (1j / (2 * rho)) / (omega**2 * kz_p)
    c_s_pol = -(1j / (2 * rho)) / (omega**2 * kz_s)

    g_p = np.zeros((3, 3, n), dtype=complex)
    g_s_iso = np.zeros((3, 3, n), dtype=complex)
    g_s_pol = np.zeros((3, 3, n), dtype=complex)
    for i in range(3):
        g_s_iso[i, i, :] = c_s_iso
        for j in range(3):
            g_p[i, j, :] = kvec_p[i] * kvec_p[j] * c_p_pol
            g_s_pol[i, j, :] = kvec_s[i] * kvec_s[j] * c_s_pol

    total = g_p + g_s_iso + g_s_pol
    return _assemble_9x9(total, [g_p, g_s_iso, g_s_pol], [kvec_p, kvec_s, kvec_s])


def _cell_sinc(z: NDArray) -> NDArray:
    """sin(z)/z for COMPLEX z, equal to 1 at z = 0.

    `np.sinc` is real-only and carries a pi in its argument; the evanescent
    branch here has z = i kappa h, where sin(z)/z = sinh(kappa h)/(kappa h).
    """
    z = np.asarray(z, dtype=complex)
    out = np.ones_like(z)
    big = np.abs(z) > 1.0e-12
    out[big] = np.sin(z[big]) / z[big]
    return out


def vertical_kernel_9x9(
    kx_arr: NDArray,
    ky: float,
    dz: float,
    omega: complex,
    ref: ReferenceMedium,
    cell_half_width: float | None = None,
) -> NDArray:
    """Whole-space 9x9 plane-to-plane kernel, k_z integral done by residue.

    Between planes dz != 0, so the surviving (kx, ky) integral keeps its
    e^{-kappa |dz|} convergence factor. This is the construction that is
    UNUSABLE at equal depth and perfectly well behaved away from it -- which is
    why same-depth coupling is the lateral sweep's job and this one is used only
    between planes.

    The sign of dz enters the k-vector's z component, not merely the
    exponential: every odd z-derivative in the C and H blocks flips with it,
    exactly as the x component does in lateral_split_9x9.

    Args:
        kx_arr: Lateral wavenumber nodes along x, shape (n_kx,), 1/km.
        ky: The 2.5-D lateral parameter, 1/km.
        dz: Signed depth separation, km. Must be non-zero.
        omega: Complex angular frequency.
        ref: Background medium.
        cell_half_width: If given, return the kernel averaged over a CUBIC
            SOURCE CELL of half-width h, instead of the point kernel. None
            (default) leaves every existing caller bit-for-bit unchanged.

            The source-cell average is a convolution with the cell indicator,
            so on a plane-wave component it is exactly the SINGLE form factor
            sinc(k_x h) sinc(k_y h) sinc(k_z h) -- one power, not the squared
            form factor of `inter_voxel_propagator`, which is the DOUBLE
            (Galerkin) average of source cell against field cell.

            ⚠ APPLIED PER MODE. The P and S poles carry different k_z, so a
            single scalar factor on the assembled 9x9 is WRONG; each pole is
            scaled by the sinc built from its own k_z. This is the same defect
            `_assemble_9x9` warns about for the k-vectors.

            ⚠ AND IT HALVES THE SPECTRAL DECAY RATE. On the evanescent branch
            k_z = i kappa, so sinc(k_z h) = sinh(kappa h)/(kappa h) ~
            e^{kappa h}/(2 kappa h), which GROWS. Against the kernel's own
            e^{-kappa|dz|} the product decays as e^{-kappa(|dz| - h)}: still
            convergent because |dz| >= 2h for distinct planes, but at half the
            rate at dz = d. A reciprocal-sum cutoff tuned for the point kernel
            is therefore NOT sufficient here -- `_spectral_bloch_block` raises
            its floor when averaging is on.

    Returns:
        P of shape (9, 9, n_kx).

    Raises:
        ValueError: if dz is zero.
    """
    if dz == 0.0:
        msg = (
            "dz must be non-zero: at equal depth the (kx, ky) integral loses its\n"
            "  e^{-kappa|dz|} convergence factor and diverges.\n"
            "  Where: cubic_scattering/sweep_kernels.py, vertical_kernel_9x9(dz=...)\n"
            "  Valid: a signed plane separation in km, e.g. dz=0.25 or dz=-0.5\n"
            "  Fix:   same-depth coupling is the LATERAL sweep's job -- call\n"
            "         directional_sweeps.sweep_x for it, not sweep_z."
        )
        raise ValueError(msg) from None

    kx = np.asarray(kx_arr, dtype=float)
    n = kx.size
    rho, alpha, beta = ref.rho, ref.alpha, ref.beta
    kh2 = kx**2 + ky**2
    kz_p = _branch((omega / alpha) ** 2 - kh2)
    kz_s = _branch((omega / beta) ** 2 - kh2)

    sign = 1.0 if dz > 0 else -1.0
    e_p = np.exp(1j * kz_p * abs(dz))
    e_s = np.exp(1j * kz_s * abs(dz))

    kvec_p = [sign * kz_p, kx.astype(complex), np.full(n, ky, dtype=complex)]
    kvec_s = [sign * kz_s, kx.astype(complex), np.full(n, ky, dtype=complex)]

    # Source-cell average, per mode. Scaling these coefficients is sufficient:
    # g_p, g_s_iso and g_s_pol are built linearly from them, and _assemble_9x9
    # is linear in the g-parts, so the C, H and S blocks inherit the factor.
    h_cell = 0.0
    ff_xy = np.ones(n, dtype=complex)
    if cell_half_width is None:
        ff_p = ff_s = np.ones(n, dtype=complex)
    else:
        h_cell = float(cell_half_width)
        ff_xy = _cell_sinc(kx * h_cell) * _cell_sinc(np.full(n, ky) * h_cell)
        ff_p = ff_xy * _cell_sinc(kz_p * h_cell)
        ff_s = ff_xy * _cell_sinc(kz_s * h_cell)

    c_s_iso = ff_s * (1j / (2 * rho)) * e_s / (beta**2 * kz_s)
    g_s_iso = np.zeros((3, 3, n), dtype=complex)
    for i in range(3):
        g_s_iso[i, i, :] = c_s_iso
    iso = _assemble_9x9(g_s_iso, [g_s_iso], [kvec_s])

    # THE POLARISATION PARTS, AS ONE DIFFERENCE. The P pole and the S pole's polarisation term are the
    # same function of u = kappa^2 at the two wavenumbers,
    #     Phi(u) = assembled [ (i / 2 rho) k k e^{i k_z |dz|} ff(k_z) / k_z ],  k_z = sqrt(u - kh^2),
    # and they enter as (Phi(k_P^2) - Phi(k_S^2)) / omega^2. For an evanescent order at small kappa each
    # is ~ kh^2 / (omega^2 |kh|) while the difference is O(1), so summing them loses eps (kh / kappa)^2.
    # The difference is taken by a contour integral in u instead (``_pole_difference``).
    def phi(u: NDArray) -> NDArray:
        kz = _branch(u - kh2)
        ff = np.ones(n, dtype=complex) if cell_half_width is None else ff_xy * _cell_sinc(kz * h_cell)
        coef = ff * (1j / (2 * rho)) * np.exp(1j * kz * abs(dz)) / kz
        kv = [sign * kz, kx.astype(complex), np.full(n, ky, dtype=complex)]
        g = np.zeros((3, 3, n), dtype=complex)
        for i in range(3):
            for j in range(3):
                g[i, j, :] = kv[i] * kv[j] * coef
        return _assemble_9x9(g, [g], [kv])

    u_p, u_s = (omega / alpha) ** 2, (omega / beta) ** 2
    return iso + _pole_difference(phi, u_p, u_s, kh2, abs(dz)) * (1.0 / alpha**2 - 1.0 / beta**2)


_POLE_MAX_RATIO = 0.1


def _pole_difference(
    phi: Callable[[NDArray], NDArray], u_p: complex, u_s: complex, kh2: NDArray, dz_abs: float
) -> NDArray:
    """(Phi(u_p) - Phi(u_s)) / (u_p - u_s), node by node, without subtracting nearly equal values.

    Phi maps an array of u (one per lateral node) to (9, 9, n). At a node with kh^2 > |u| the order is
    evanescent and Phi is analytic in u for |u| < kh^2 (the branch point of k_z), so the divided
    difference is the Cauchy integral (1 / 2 pi i) oint Phi(z) / ((z - u_p)(z - u_s)) dz on a circle
    about the midpoint of radius a quarter of the reach; the trapezoid rule converges
    as max(half-gap / radius, 1/4)^N. Propagating and near-anomaly nodes, where the two poles differ at
    order one, take the plain divided difference.

    The reach is the smaller of the distance to the branch point, kh^2 - |midpoint|, and Phi's scale of
    variation: on the evanescent branch e^{i k_z |dz|} = e^{-|dz| sqrt(kh^2 - u)} changes by a factor e
    when u moves by 2 kh / |dz|, which for high orders is far shorter. Taking the branch-point distance
    alone lets Phi vary by e^{kh |dz| / 8} round the circle and amplifies its round-off accordingly.
    """
    centre = 0.5 * (u_p + u_s)
    half_gap = 0.5 * abs(u_p - u_s)
    reach = np.minimum(kh2 - abs(centre), 2.0 * np.sqrt(kh2) / dz_abs)
    contour = reach > (half_gap / _POLE_MAX_RATIO)
    out = (phi(np.full(kh2.shape, u_p, dtype=complex)) - phi(np.full(kh2.shape, u_s, dtype=complex))) / (
        u_p - u_s
    )
    if not contour.any() or half_gap == 0.0:
        return out
    radius = np.where(contour, 0.25 * reach, 0.25 * half_gap / _POLE_MAX_RATIO)
    ratio = float(np.max(np.maximum(half_gap / radius[contour], 0.25)))
    n_nodes = max(8, math.ceil(math.log(1e-17) / math.log(ratio)) + 2)
    acc = np.zeros_like(out)
    for j in range(n_nodes):
        z = centre + radius * np.exp(2j * np.pi * (j + 0.5) / n_nodes)
        weight = (z - centre) / ((z - u_p) * (z - u_s)) / n_nodes
        acc += phi(z) * weight
    return np.where(contour, acc, out)
