#!/usr/bin/env python3
"""The (u, eps_Voigt) <-> P/SV/SH mode bridge for the vertical sweep.

At fixed (k_x, k_y) an elastic medium supports exactly six plane-wave modes:
P, SV and SH, each going up or down. The whole-space plane-to-plane kernel
therefore has RANK 3 for a given propagation direction -- measured, with the
fourth singular value at 1e-16 -- and factorises exactly as

    P_ws(dz) = D diag(e^{i kz_m |dz|}) S

with D the receiver-side mode embedding and S the source-side radiation. The
layered background enters by replacing the diagonal phase with the layered
mode-space response; nothing else in the sweep changes.

This bridge lives in its own module ON PURPOSE. Representation conversion is
where this project's defects have actually lived: every one of the three
defects resolved in the 9x9 wrapper work was a conversion-convention error, and
one of them survived months because a symmetry gate passed it happily.

Two standing conventions, both easy to get wrong:

  * Seismic units (km/s, g/cm3, GPa). In SI the mode matrix appears
    ill-conditioned at ~1e10; that is a metres-versus-pascals artefact scaling
    as rho*omega*v, NOT a defect, and must not be "fixed" by regularisation.
  * Time convention e^{-i omega t}, every vertical wavenumber pinned to Im >= 0,
    so a down-going wave is e^{+i kz z} with z increasing downward.

Mode order throughout: (P down, SV down, SH down, P up, SV up, SH up).
Coordinates: z = axis 0 (down), x = axis 1 (right), y = axis 2 (out).
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .effective_contrasts import ReferenceMedium
from .resonance_tmatrix import VOIGT_PAIRS
from .sweep_kernels import _branch, vertical_kernel_9x9

MODE_NAMES = ("P down", "SV down", "SH down", "P up", "SV up", "SH up")


@dataclass(frozen=True)
class ModeBasis:
    """The six plane-wave modes at one (k_x, k_y).

    Attributes:
        k_vectors: Wavevectors in (z, x, y) order, shape (6, 3) complex.
        polarisations: Unit polarisation vectors, shape (6, 3) complex. Unit in
            the complex-BILINEAR sense (pol @ pol == 1), not the conjugate one:
            these continue analytically into the evanescent regime, where a
            conjugate normalisation would not.
        kz_p: Vertical wavenumber of the down-going P mode.
        kz_s: Vertical wavenumber of the down-going S modes.
    """

    k_vectors: NDArray
    polarisations: NDArray
    kz_p: complex
    kz_s: complex


@dataclass(frozen=True)
class VerticalFactorisation:
    """D diag(phase) S for the whole-space kernel at one (k_x, k_y).

    Attributes:
        receiver: Mode embedding, shape (9, 6).
        source: Source-side radiation, shape (6, 9). Rows for the half not
            propagating in the fitted direction are exactly zero.
        kz: PROPAGATION wavenumber per mode, shape (6,), always on the Im >= 0
            branch -- so the phase over a distance |dz| is e^{+i kz |dz|} and
            decays, for the up-going half as much as the down-going one. This is
            NOT the signed z-component of the wavevector: propagation direction
            is carried by ModeBasis.k_vectors, which is what sets the
            polarisation and the odd-z signs in the C and H blocks. Conflating
            the two makes the down-going half exact and the up-going half grow
            instead of decay.
        sign: +1 if fitted for dz > 0 (down-going), -1 for dz < 0.
    """

    receiver: NDArray
    source: NDArray
    kz: NDArray
    sign: float

    def evaluate(self, dz: float) -> NDArray:
        """Rebuild the 9x9 kernel at separation dz.

        Args:
            dz: Signed depth separation, km. Must have the same sign as the
                direction this factorisation was fitted for.

        Returns:
            The 9x9 kernel.

        Raises:
            ValueError: if dz has the wrong sign or is zero.
        """
        if dz == 0.0 or np.sign(dz) != self.sign:
            msg = (
                f"dz={dz!r} does not match this factorisation, fitted for "
                f"sign={self.sign:+.0f}.\n"
                "  Where: cubic_scattering/sweep_modes.py, VerticalFactorisation.evaluate\n"
                "  Valid: a non-zero dz of the fitted sign, e.g. "
                f"dz={self.sign * 0.25}\n"
                "  Fix:   fit a separate factorisation per direction -- the up-going and\n"
                "         down-going halves radiate differently and must not be shared."
            )
            raise ValueError(msg) from None
        phase = np.exp(1j * self.kz * abs(dz))
        return self.receiver @ np.diag(phase) @ self.source


def mode_state(k_vec: NDArray, pol: NDArray) -> NDArray:
    """The 9-component state (u, eps_Voigt) of a plane wave.

    eps_ab = (i/2)(k_a u_b + k_b u_a), with engineering doubling on the three
    off-diagonal Voigt components so that the state matches what the kernels and
    T0 already use.

    Args:
        k_vec: Wavevector in (z, x, y) order, shape (3,).
        pol: Polarisation vector, shape (3,).

    Returns:
        The 9-vector.
    """
    out = np.zeros(9, dtype=complex)
    out[:3] = pol
    for a, (p, q) in enumerate(VOIGT_PAIRS):
        val = 0.5j * (k_vec[p] * pol[q] + k_vec[q] * pol[p])
        out[3 + a] = val if p == q else 2.0 * val
    return out


def mode_basis(kx: float, ky: float, omega: complex, ref: ReferenceMedium) -> ModeBasis:
    """Build the six plane-wave modes at one (k_x, k_y).

    Args:
        kx: Lateral wavenumber along x, 1/km.
        ky: Lateral wavenumber along y, 1/km.
        omega: Complex angular frequency.
        ref: Background medium (seismic units).

    Returns:
        A ModeBasis.
    """
    kh2 = kx**2 + ky**2
    kz_p = complex(_branch(np.array([(omega / ref.alpha) ** 2 - kh2]))[0])
    kz_s = complex(_branch(np.array([(omega / ref.beta) ** 2 - kh2]))[0])

    # SH lies in the horizontal plane, perpendicular to the propagation azimuth.
    # At normal incidence the azimuth is undefined and any horizontal direction
    # serves; pick x so the choice is deterministic rather than accidental.
    kh = np.hypot(kx, ky)
    sh = np.array([0.0, 1.0, 0.0]) if kh < 1e-12 else np.array([0.0, -ky, kx]) / kh

    k_vectors = np.zeros((6, 3), dtype=complex)
    polarisations = np.zeros((6, 3), dtype=complex)
    for half, sgn in enumerate((1.0, -1.0)):
        kp = np.array([sgn * kz_p, kx, ky], dtype=complex)
        ks = np.array([sgn * kz_s, kx, ky], dtype=complex)
        # Bilinear normalisation: continues analytically into evanescence.
        pol_p = kp / np.sqrt(kp @ kp)
        sv = np.cross(ks, sh)
        sv = sv / np.sqrt(sv @ sv)

        k_vectors[3 * half + 0] = kp
        k_vectors[3 * half + 1] = ks
        k_vectors[3 * half + 2] = ks
        polarisations[3 * half + 0] = pol_p
        polarisations[3 * half + 1] = sv
        polarisations[3 * half + 2] = sh

    return ModeBasis(k_vectors=k_vectors, polarisations=polarisations, kz_p=kz_p, kz_s=kz_s)


def modes_to_state(kx: float, ky: float, omega: complex, ref: ReferenceMedium) -> NDArray:
    """Embedding of the six mode amplitudes into the 9-component state.

    Returns:
        Array of shape (9, 6), columns in MODE_NAMES order.
    """
    mb = mode_basis(kx, ky, omega, ref)
    return np.column_stack([mode_state(mb.k_vectors[m], mb.polarisations[m]) for m in range(6)])


def state_to_modes(kx: float, ky: float, omega: complex, ref: ReferenceMedium) -> NDArray:
    """Projection of a 9-component state onto the six mode amplitudes.

    The left inverse of modes_to_state. Exact on the mode subspace; on the full
    9-space the round trip is a rank-6 PROJECTOR, because three combinations of
    (u, eps) are fixed by the equations of motion and carry no independent
    amplitude.

    Returns:
        Array of shape (6, 9).
    """
    return np.linalg.pinv(modes_to_state(kx, ky, omega, ref))


def vertical_factorisation(
    kx: float, ky: float, dz: float, omega: complex, ref: ReferenceMedium
) -> VerticalFactorisation:
    """Factorise the whole-space vertical kernel as D diag(phase) S.

    The source side is recovered from the kernel at the given dz by removing the
    known propagation phase. Because the kernel has rank 3 for a given
    direction, this is a determination rather than a fit -- and the test that
    matters is that the SAME source side reproduces the kernel at every OTHER
    separation, which a wrong mode embedding cannot do.

    Args:
        kx: Lateral wavenumber along x, 1/km.
        ky: Lateral wavenumber along y, 1/km.
        dz: Signed separation to fit at, km. Non-zero.
        omega: Complex angular frequency.
        ref: Background medium.

    Returns:
        A VerticalFactorisation.

    Raises:
        ValueError: if dz is zero.
    """
    if dz == 0.0:
        msg = (
            "dz must be non-zero: the whole-space kernel is undefined at equal depth.\n"
            "  Where: cubic_scattering/sweep_modes.py, vertical_factorisation(dz=...)\n"
            "  Valid: a signed separation in km, e.g. dz=0.25\n"
            "  Fix:   same-depth coupling is the lateral sweep's job."
        )
        raise ValueError(msg) from None

    mb = mode_basis(kx, ky, omega, ref)
    receiver = modes_to_state(kx, ky, omega, ref)
    # Propagation wavenumbers, Im >= 0 for every mode -- see the note on
    # VerticalFactorisation.kz. Both halves decay over |dz|.
    kz = np.array([mb.kz_p, mb.kz_s, mb.kz_s, mb.kz_p, mb.kz_s, mb.kz_s])

    sign = float(np.sign(dz))
    kernel = vertical_kernel_9x9(np.array([kx]), ky, dz, omega, ref)[:, :, 0]

    # Only the half propagating in this direction carries amplitude.
    active = slice(0, 3) if sign > 0 else slice(3, 6)
    d_active = receiver[:, active]
    phase = np.exp(1j * kz[active] * abs(dz))

    source = np.zeros((6, 9), dtype=complex)
    source[active] = np.diag(1.0 / phase) @ np.linalg.pinv(d_active) @ kernel

    return VerticalFactorisation(receiver=receiver, source=source, kz=kz, sign=sign)
