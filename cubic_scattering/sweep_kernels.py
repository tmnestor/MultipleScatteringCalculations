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
