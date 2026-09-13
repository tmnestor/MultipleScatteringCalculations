#!/usr/bin/env python3
"""Absolute calibration of the stratified 6x6 against the closed-form whole space.

WHY THIS GATE EXISTS
--------------------
Every gate in ``gate_9x9_source_convention.py`` is a *reciprocity* statement,
and reciprocity is homogeneous of degree one: it certifies structure, never
magnitude, and it compares two different matrices rather than looking inside
one.  This gate is the missing absolute leg.  It uses the one quadrant of the
problem that carries no convention risk at all.

A point force ``F`` makes a jump ONLY in the traction half of the state,
``[T_i] = -F_i``, with no displacement jump.  So the force columns of the
source operator are known exactly and trivially --- no Voigt ordering, no
engineering-strain weight ``W``, no strain operator ``A``.  The force ->
displacement quadrant of the layered Green's function must therefore equal the
whole-space displacement Green's tensor in the homogeneous limit, up to ONE
scalar carrying the basis and normalisation convention:

    Gt_uu(kx, ky, dz)  ==  c * G6[0:3, 3:6]

If the nine entrywise ratios are not a single constant, the defect lies inside
the 6x6 and no choice of source operator can repair it.

THE REFERENCE
-------------
For a homogeneous whole space, time convention ``e^{-iwt}``, transform pair
``exp(+i(kx x + ky y))``, and ``dz != 0``:

    Gt_ij = (1/(rho w^2)) [ qP_i qP_j gP - qS_i qS_j gS + delta_ij kS^2 gS ]
    gM    = i exp(i nuM |dz|) / (2 nuM),   nuM = sqrt(kM^2 - kx^2 - ky^2)
    qM    = (sgn(dz) nuM, kx, ky)          in (z, x, y) index order

``validate_reference`` checks this against the repo's own validated spatial
tensor ``horizontal_greens.exact_greens`` through an independent 2-D inverse
transform, so the reference is not taken on trust.  The k-grid must contain
the S pole (kS = w/beta); a grid that does not makes the check fail by O(1).

Run:  conda run -n seismic python scripts/gate_displacement_only_calibration.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, "/Users/tod/Desktop/SeismicInversion")

from cubic_scattering.horizontal_greens import exact_greens  # isort: skip

from Kennett_Reflectivity.layer_model import LayerModel  # isort: skip
from GlobalMatrix.layered_greens import layered_greens_6x6  # isort: skip

TOL_REFERENCE = 5e-4


def homogeneous_model(
    al: float = 4.0,
    be: float = 2.22,
    rh: float = 2.6,
    q: float = 20.0,
    n_lay: int = 16,
    dz: float = 1.0,
) -> LayerModel:
    """Uniform crust between an ocean and an identical half-space.

    The half-space repeats the crust so nothing reflects from below; the ocean
    and free surface are 8 km away and attenuated, so a deep interface pair
    sees an approximate whole space.

    Args:
        al: Crust P velocity (km/s).
        be: Crust S velocity (km/s).
        rh: Crust density (g/cm^3).
        q: Quality factor applied to every solid layer.
        n_lay: Number of identical crust layers.
        dz: Thickness of each crust layer (km).

    Returns:
        The layered model.
    """
    return LayerModel.from_arrays(
        alpha=[1.5, *([al] * n_lay), al],
        beta=[0.0, *([be] * n_lay), be],
        rho=[1.03, *([rh] * n_lay), rh],
        thickness=[3.0, *([dz] * n_lay), np.inf],
        Q_alpha=[q, *([q] * n_lay), q],
        Q_beta=[1e10, *([q] * n_lay), q],
    )


def _vertical(k2: complex, kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
    """Vertical wavenumber with the decaying branch (Im >= 0)."""
    nu = np.sqrt(k2 - kx**2 - ky**2 + 0j)
    return np.where(np.imag(nu) < 0, -nu, nu)


def whole_space_guu(
    kx: float,
    ky: float,
    dz: float,
    omega: float,
    rho: float,
    s_p: complex,
    s_s: complex,
) -> np.ndarray:
    """Closed-form 3x3 displacement Green's tensor in the (kx, ky, dz) domain.

    Args:
        kx: Horizontal wavenumber, x component (rad/km).
        ky: Horizontal wavenumber, y component (rad/km).
        dz: Receiver depth minus source depth (km), non-zero.
        omega: Angular frequency (rad/s).
        rho: Density (g/cm^3).
        s_p: Complex P slowness (s/km).
        s_s: Complex S slowness (s/km).

    Returns:
        Gt: shape (3, 3), index order (z, x, y).
    """
    k_p, k_s = omega * s_p, omega * s_s
    nu_p = _vertical(k_p**2, np.asarray(kx), np.asarray(ky))
    nu_s = _vertical(k_s**2, np.asarray(kx), np.asarray(ky))

    sgn = 1.0 if dz > 0 else -1.0
    g_p = 1j * np.exp(1j * nu_p * abs(dz)) / (2 * nu_p)
    g_s = 1j * np.exp(1j * nu_s * abs(dz)) / (2 * nu_s)

    q_p = np.array([sgn * nu_p, kx, ky], dtype=complex)
    q_s = np.array([sgn * nu_s, kx, ky], dtype=complex)

    out = np.zeros((3, 3), dtype=complex)
    for i in range(3):
        for j in range(3):
            d = 1.0 if i == j else 0.0
            out[i, j] = (q_p[i] * q_p[j] * g_p - q_s[i] * q_s[j] * g_s + d * k_s**2 * g_s) / (
                rho * omega**2
            )
    return out


def validate_reference(
    omega: float,
    rho: float,
    s_p: complex,
    s_s: complex,
    npt: int = 2048,
    kmax: float = 400.0,
) -> float:
    """Check the reference by inverse-transforming it back to space.

    Args:
        omega: Angular frequency (rad/s).
        rho: Density (g/cm^3).
        s_p: Complex P slowness (s/km).
        s_s: Complex S slowness (s/km).
        npt: Grid points per axis.
        kmax: Half-width of the wavenumber grid (rad/km); must exceed w/beta.

    Returns:
        Worst relative error over 9 components at 4 offsets.
    """
    dz = 0.35
    dk = 2 * kmax / npt
    kk = (np.arange(npt) - npt // 2) * dk
    kx, ky = np.meshgrid(kk, kk, indexing="ij")

    k_p, k_s = omega * s_p, omega * s_s
    nu_p, nu_s = _vertical(k_p**2, kx, ky), _vertical(k_s**2, kx, ky)
    g_p = 1j * np.exp(1j * nu_p * abs(dz)) / (2 * nu_p)
    g_s = 1j * np.exp(1j * nu_s * abs(dz)) / (2 * nu_s)
    q_p, q_s = [nu_p, kx, ky], [nu_s, kx, ky]

    dx = np.pi / kmax
    xx = (np.arange(npt) - npt // 2) * dx
    zxy = [2, 0, 1]  # exact_greens indexes (x, y, z); this module uses (z, x, y)

    print("=" * 74)
    print("REFERENCE VALIDATION  (2-D inverse transform vs exact_greens)")
    print("=" * 74)
    worst = 0.0
    for i in range(3):
        for j in range(3):
            d = 1.0 if i == j else 0.0
            spec = (q_p[i] * q_p[j] * g_p - q_s[i] * q_s[j] * g_s + d * k_s**2 * g_s) / (rho * omega**2)
            grid = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(spec))) * npt**2 * dk**2 / (2 * np.pi) ** 2
            for mx, my in ((8, 0), (0, 8), (14, 14), (30, 6)):
                ix, iy = npt // 2 + mx, npt // 2 + my
                ref = exact_greens(xx[ix], xx[iy], dz, omega, rho=rho, alpha=1 / s_p, beta=1 / s_s)[
                    zxy[i], zxy[j]
                ]
                if abs(ref) > 1e-12:
                    worst = max(worst, abs(grid[ix, iy] - ref) / abs(ref))
    verdict = "VALIDATED" if worst < TOL_REFERENCE else "NOT VALIDATED"
    print(f"  worst relative error: {worst:.3e}   -> reference {verdict}\n")
    return worst


def channel_split(model: LayerModel, f: float, p: float, j: int, i: int, dz: float) -> None:
    """Ratios per channel on the ky = 0 slice.

    With ky = 0 the propagation plane is the (z, x) plane of this project's
    frame, so P-SV lives entirely in (z, x) and SH entirely in y.  Each entry
    below therefore belongs to exactly one channel with no mixing, and the
    P-SV and SH questions separate without introducing any other frame.

    Args:
        model: Layered model.
        f: Frequency (Hz).
        p: Horizontal slowness (s/km).
        j: Source interface index.
        i: Receiver interface index.
        dz: Receiver depth minus source depth (km).
    """
    w = 2.0 * np.pi * f
    kx = w * p
    g6 = layered_greens_6x6(model, w, np.array([kx]), np.array([0.0]), source_iface=j, receiver_iface=i)[0]
    s_p = model.complex_slowness_p()[1]
    s_s = model.complex_slowness_s()[1]
    gt = whole_space_guu(kx, 0.0, dz, w, model.rho[1], s_p, s_s)

    print(f"  f={f:5.1f} Hz  p={p:5.2f}  dz={dz:+.1f} km")
    for (a, b), nm in (
        ((0, 0), "P-SV  u_z <- [T_zz]"),
        ((0, 1), "P-SV  u_z <- [T_xz]"),
        ((1, 0), "P-SV  u_x <- [T_zz]"),
        ((1, 1), "P-SV  u_x <- [T_xz]"),
        ((2, 2), "SH    u_y <- [T_yz]"),
    ):
        r = gt[a, b] / g6[a, 3 + b] * w**2
        print(f"    {nm:22s} ratio*w^2 = {r.real:+11.6f}{r.imag:+11.6f}j")


def sh_factor_sweep(model: LayerModel) -> None:
    """Show that the SH column carries a spurious factor 1/eta_S.

    Args:
        model: Layered model.
    """
    s_p = model.complex_slowness_p()[1]
    s_s = model.complex_slowness_s()[1]
    print("=" * 74)
    print("SH SOURCE NORMALISATION:  (SH ratio*w^2) / (-1/eta_S)")
    print("=" * 74)
    print("   f (Hz)     p     P-SV ratio*w^2        quotient (should be 1)")
    for f in (6.0, 24.0, 48.0):
        for p in (0.02, 0.05, 0.12, 0.20):
            w = 2.0 * np.pi * f
            g6 = layered_greens_6x6(
                model,
                w,
                np.array([w * p]),
                np.array([0.0]),
                source_iface=9,
                receiver_iface=8,
            )[0]
            gt = whole_space_guu(w * p, 0.0, -1.0, w, model.rho[1], s_p, s_s)
            r_psv = gt[0, 0] / g6[0, 3] * w**2
            r_sh = gt[2, 2] / g6[2, 5] * w**2
            eta_s = _vertical(s_s**2, np.asarray(p), np.asarray(0.0))
            quot = (-r_sh) / (1.0 / eta_s)
            print(
                f"  {f:7.1f}  {p:5.2f}   {r_psv.real:+10.6f}        {quot.real:+10.6f}{quot.imag:+10.6f}j"
            )
    print()


def cartesian_correction(model: LayerModel, f: float = 48.0) -> float:
    """Measure the source-side correction K directly in (z, x, y), off-axis.

    Defining ``G6[0:3, 3:6] = (-i w)^2 . G_uu^exact . K`` and solving the 3x3
    system for K makes no assumption about any sagittal/transverse frame.  The
    measured K is, to machine precision, the projector form

        K_zz,zz = 1
        K_ab    = -khat_a khat_b + eta_S (delta_ab - khat_a khat_b)

    on the horizontal traction slots (sigma_xz, sigma_yz), with
    ``khat = (kx, ky)/|k|``.  It is diagonal in (z, x, y) ONLY when kx ky = 0.

    Args:
        model: Layered model.
        f: Frequency (Hz); use one high enough that the homogeneous limit has
            converged (see channel_split).

    Returns:
        Worst relative error of the projector form over the sampled geometries.
    """
    s_p = model.complex_slowness_p()[1]
    s_s = model.complex_slowness_s()[1]
    w = 2.0 * np.pi * f

    print("=" * 74)
    print("SOURCE CORRECTION K IN (z, x, y):  G6[0:3,3:6] = (-i w)^2 Gt K")
    print("  K_ab = -khat_a khat_b + eta_S (delta_ab - khat_a khat_b),  a,b in {x,y}")
    print("=" * 74)
    print("     p    kx/|k|  ky/|k|    rel err vs projector form   off-diag weight")
    worst = 0.0
    for p, c, s in (
        (0.12, 1.0, 0.0),
        (0.12, 0.8, 0.6),
        (0.12, 0.6, 0.8),
        (0.12, 0.0, 1.0),
        (0.20, 0.6, 0.8),
    ):
        kx, ky = w * p * c, w * p * s
        g6 = layered_greens_6x6(model, w, np.array([kx]), np.array([ky]), source_iface=9, receiver_iface=8)[
            0
        ]
        gt = whole_space_guu(kx, ky, -1.0, w, model.rho[1], s_p, s_s)
        k_meas = np.linalg.solve((-1j * w) ** 2 * gt, g6[0:3, 3:6])

        eta_s = complex(_vertical(s_s**2, np.asarray(p), np.asarray(0.0)))
        khat = np.array([c, s])
        par = np.outer(khat, khat)
        pred = np.eye(3, dtype=complex)
        pred[1:, 1:] = -par + eta_s * (np.eye(2) - par)

        err = np.linalg.norm(k_meas - pred) / np.linalg.norm(pred)
        off = np.linalg.norm(k_meas - np.diag(np.diag(k_meas))) / np.linalg.norm(k_meas)
        worst = max(worst, float(err))
        print(f"  {p:5.2f}   {c:5.2f}   {s:5.2f}        {err:.3e}              {off:.3e}")
    print(f"\n  worst: {worst:.3e}   -> projector form {'CONFIRMED' if worst < 1e-10 else 'REJECTED'}")
    print("  off-diagonal weight is ~0.66 whenever kx ky != 0: K is NOT diagonal")
    print("  in (z, x, y), so no diagonal source correction can exist.\n")
    return worst


def main() -> int:
    """Run the reference validation, the channel split and the SH sweep.

    Returns:
        0 if the reference validated, 1 otherwise.
    """
    model = homogeneous_model()
    worst = validate_reference(
        2 * np.pi * 12.0,
        model.rho[1],
        model.complex_slowness_p()[1],
        model.complex_slowness_s()[1],
    )
    if worst >= TOL_REFERENCE:
        print("Reference failed to validate; downstream numbers are meaningless.")
        return 1

    print("=" * 74)
    print("CHANNEL SPLIT at ky = 0   ratio = Gt[i,j] / G6[i,3+j], scaled by w^2")
    print("  expected if the 6x6 were consistent: ONE constant for all five")
    print("=" * 74)
    for f in (6.0, 24.0):
        for dz in (-1.0, +1.0):
            channel_split(model, f, 0.05, 9, 8, dz)
    print()
    sh_factor_sweep(model)
    cartesian_correction(model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
