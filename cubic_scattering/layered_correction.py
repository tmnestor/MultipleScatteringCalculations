#!/usr/bin/env python3
"""Reconcile the stratified layered Green's function with this package's 9x9.

The stratified propagator ``P^z`` and this package's lateral propagator ``P^x``
must speak one 9-component language before they can be composed in a single
resolvent.  They did not.  Three defects separated them, all measured against an
absolute whole-space reference rather than inferred from reciprocity:

D1  ``assemble_greens_6x6`` rescales the stress SOURCE columns on the
    displacement rows only, so the stress-row x stress-column block carries one
    factor of ``(-i w)`` where a consistent change of basis needs two.

D2  The traction half of BOTH indices carries the operator ``K`` below: as ``K``
    on the rows, as ``(-i w)^2 K`` on the source columns, with ``-1`` on the
    displacement source columns.

D3  The source operator must carry the body force and the stress glut with
    OPPOSITE relative sign.  Equivalently, this package's ``Dsigma*`` is minus
    the stress glut ``m`` of the underlying theory.

With all three applied, ``W . A G B`` is symmetric to ~1e-15 (it was 0.40-0.81)
and the corrected 6x6 obeys the bare symplectic reciprocity law
``G(i<-j)(+k) = J6 [G(j<-i)(-k)]^T J6`` with no weight at all.

THE ONE RESTRICTION.  ``K`` is built from the local vertical S slowness
``eta_S``, which is two-valued on a material interface.  Source and receiver
planes must therefore lie in the INTERIOR of a layer; there the correction is
exact even with strong contrasts between them (measured 1.7e-15 with a fast slab
crossed twice).  ``assert_interface_continuous`` enforces this.

Conventions: seismic units (km/s, g/cm^3, GPa, km), time ``e^{-iwt}``, transform
``exp(+i(kx x + ky y))``, index order z = 0 (down), x = 1, y = 2.  State vectors
are ``(u_z, u_x, u_y, T_zz, T_xz, T_yz)`` and
``(u_z, u_x, u_y, e_zz, e_xx, e_yy, 2e_xy, 2e_zy, 2e_zx)``; the engineering
doubling on the shear slots is the origin of ``W9``.
"""

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "J6",
    "W9",
    "assert_interface_continuous",
    "correct_6x6",
    "corrected_layered_6x6",
    "k_operator",
    "source_jump_operator",
    "strain_from_state",
]

J6 = np.zeros((6, 6))
J6[:3, 3:], J6[3:, :3] = np.eye(3), -np.eye(3)

W9 = np.diag(np.array([1, 1, 1, 1, 1, 1, 0.5, 0.5, 0.5], dtype=float))


def strain_from_state(kx: float, ky: float, rho: float, alpha: complex, beta: complex) -> NDArray:
    """Receiver operator A: (u, T) -> (u, eps), 9x6.

    Recovers ``e_zz`` from ``T_zz`` and the in-plane strains, so it never needs
    ``k_z``.

    Args:
        kx: Horizontal wavenumber, x component (rad/km).
        ky: Horizontal wavenumber, y component (rad/km).
        rho: Density (g/cm^3).
        alpha: P velocity (km/s), may be complex.
        beta: S velocity (km/s), may be complex.

    Returns:
        Shape (9, 6) complex.
    """
    mu = rho * beta**2
    m2 = rho * alpha**2
    lam = m2 - 2 * mu
    ikx, iky = 1j * kx, 1j * ky

    a = np.zeros((9, 6), dtype=complex)
    a[0, 0] = a[1, 1] = a[2, 2] = 1.0
    a[3, 1], a[3, 2], a[3, 3] = -lam / m2 * ikx, -lam / m2 * iky, 1.0 / m2
    a[4, 1] = ikx
    a[5, 2] = iky
    a[6, 1], a[6, 2] = iky, ikx
    a[7, 5] = 1.0 / mu
    a[8, 4] = 1.0 / mu
    return a


def k_operator(s_s: complex, kx: float, ky: float, omega: float) -> NDArray:
    """The traction-half normalisation operator (defect D2), in (z, x, y).

    ``K = 1 (+) (-P_par + eta_S P_perp)`` acting on
    ``(sigma_zz, sigma_xz, sigma_yz)``, with ``P_par = khat khat^T``,
    ``P_perp = I2 - P_par`` and ``eta_S = sqrt(s_s^2 - p^2)``.

    It is diagonal in (z, x, y) only when ``kx ky = 0``; elsewhere roughly two
    thirds of its weight is off-diagonal, which is why no diagonal source
    correction exists.  It is even in ``k``.

    Args:
        s_s: Complex S slowness of the LOCAL medium (s/km).
        kx: Horizontal wavenumber, x component (rad/km).
        ky: Horizontal wavenumber, y component (rad/km).
        omega: Angular frequency (rad/s).

    Returns:
        Shape (3, 3) complex.

    Raises:
        ValueError: If ``kx`` and ``ky`` are both zero.
    """
    kpar = float(np.hypot(kx, ky))
    if kpar == 0.0:
        msg = (
            "k_operator is undefined at kx = ky = 0.\n"
            "  What: the projectors P_par and P_perp need a horizontal\n"
            "        propagation direction, and there is none at the origin.\n"
            "  Where: the caller's wavenumber grid includes k = 0.\n"
            "  Expected: hypot(kx, ky) > 0.\n"
            "  Fix: offset the origin sample, as the slab path does with\n"
            "       p = 1e-6 for a normal-incidence ray."
        )
        raise ValueError(msg) from None

    khat = np.array([kx, ky]) / kpar
    par = np.outer(khat, khat)
    eta = np.sqrt(s_s**2 - (kpar / omega) ** 2 + 0j)
    if np.imag(eta) < 0:
        eta = -eta

    out = np.eye(3, dtype=complex)
    out[1:, 1:] = -par + complex(eta) * (np.eye(2) - par)
    return out


def correct_6x6(
    g6: NDArray,
    omega: float,
    s_s_source: complex,
    s_s_receiver: complex,
    kx: float,
    ky: float,
) -> NDArray:
    """Apply defects D1 and D2 to a raw layered 6x6.

    Args:
        g6: Raw 6x6 from the layered solver, basis (u, T).
        omega: Angular frequency (rad/s).
        s_s_source: Complex S slowness at the source plane (s/km).
        s_s_receiver: Complex S slowness at the receiver plane (s/km).
        kx: Horizontal wavenumber, x component (rad/km).
        ky: Horizontal wavenumber, y component (rad/km).

    Returns:
        Shape (6, 6) complex, equal to the exact jump response of the reference
        medium in the limits where that is known.
    """
    miw = -1j * omega
    out = np.array(g6, dtype=complex, copy=True)
    out[3:, 3:] *= miw  # D1

    lrow = np.eye(6, dtype=complex)
    lrow[3:, 3:] = k_operator(s_s_receiver, kx, ky, omega)
    scol = np.eye(6, dtype=complex)
    scol[0:3, 0:3] = -np.eye(3)
    scol[3:, 3:] = -(miw**2) * k_operator(s_s_source, kx, ky, omega)
    return lrow @ out @ np.linalg.inv(scol)  # D2


def source_jump_operator(kx: float, ky: float, rho: float, alpha: complex, beta: complex) -> NDArray:
    """Source operator B: (F_i, Dsigma*_Va) -> state jump [b], 6x9.

    Derived from the governing first-order system, not fitted: integrating
    ``d_z b = A b + F delta + ...`` across the source plane gives the classical
    source discontinuity vector, which equals ``J6 A_src(-k)^T W`` entry for
    entry.  The glut columns then carry the D3 sign, so that with this
    package's ``Dsigma* = -m`` the result reproduces the underlying theory's own
    worked point explosion exactly.

    Args:
        kx: Horizontal wavenumber, x component (rad/km).
        ky: Horizontal wavenumber, y component (rad/km).
        rho: Density at the SOURCE plane (g/cm^3).
        alpha: P velocity at the source plane (km/s).
        beta: S velocity at the source plane (km/s).

    Returns:
        Shape (6, 9) complex.
    """
    b = J6 @ strain_from_state(-kx, -ky, rho, alpha, beta).T @ W9
    b[:, 3:] *= -1.0  # D3
    return b


def assert_interface_continuous(model: object, iface: int, role: str) -> None:
    """Reject a source or receiver plane that sits on a material discontinuity.

    Args:
        model: Anything exposing ``alpha``, ``beta``, ``rho`` sequences and
            ``n_layers``.
        iface: Interface index; interface k lies at the bottom of layer k.
        role: "source" or "receiver", used in the message.

    Raises:
        ValueError: If the layers either side of the interface differ.
    """
    above = max(int(iface), 1)
    below = above + 1
    if below >= int(model.n_layers):  # type: ignore[attr-defined]
        return

    al = model.alpha  # type: ignore[attr-defined]
    be = model.beta  # type: ignore[attr-defined]
    rh = model.rho  # type: ignore[attr-defined]
    if al[above] == al[below] and be[above] == be[below] and rh[above] == rh[below]:
        return

    msg = (
        f"{role} interface {iface} lies on a material discontinuity.\n"
        f"  What: layer {above} has (alpha, beta, rho) = "
        f"({al[above]}, {be[above]}, {rh[above]}) but layer {below} has "
        f"({al[below]}, {be[below]}, {rh[below]}).\n"
        "  Why: the correction operator K is built from the local vertical S\n"
        "       slowness eta_S, which is two-valued at a material interface, so\n"
        "       K is not defined there.  Measured: no choice of side does better\n"
        "       than 3.9e-3, against 1.7e-15 for a plane inside a layer.\n"
        "  Fix: place the plane in the interior of a layer.  If a scatterer must\n"
        "       sit at that depth, subdivide the layer and use an interior\n"
        "       interface -- subdivision of a uniform layer is transparent."
    )
    raise ValueError(msg) from None


def corrected_layered_6x6(
    model: object,
    omega: float,
    kx: float,
    ky: float,
    source_iface: int,
    receiver_iface: int,
) -> NDArray:
    """Corrected stratified 6x6, with the placement restriction enforced.

    Imports the external layered solver lazily so that this module stays usable
    without it.

    Args:
        model: Stratified model accepted by ``layered_greens_6x6``.
        omega: Angular frequency (rad/s).
        kx: Horizontal wavenumber, x component (rad/km).
        ky: Horizontal wavenumber, y component (rad/km).
        source_iface: Source interface index.
        receiver_iface: Receiver interface index.

    Returns:
        Shape (6, 6) complex in basis (u_z, u_x, u_y, T_zz, T_xz, T_yz).
    """
    from GlobalMatrix.layered_greens import layered_greens_6x6

    assert_interface_continuous(model, source_iface, "source")
    assert_interface_continuous(model, receiver_iface, "receiver")

    raw = layered_greens_6x6(
        model,
        omega,
        np.array([kx]),
        np.array([ky]),
        source_iface=source_iface,
        receiver_iface=receiver_iface,
    )[0]
    s_s = model.complex_slowness_s()  # type: ignore[attr-defined]
    return correct_6x6(
        raw,
        omega,
        s_s[max(int(source_iface), 1)],
        s_s[max(int(receiver_iface), 1)],
        kx,
        ky,
    )
