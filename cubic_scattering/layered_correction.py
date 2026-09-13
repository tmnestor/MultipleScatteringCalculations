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
    displacement source columns.  ``K`` is a HOUSEHOLDER REFLECTION in the
    horizontal plane, ``1 (+) (I2 - 2 khat khat^T)`` -- purely geometric.

D3  The source operator must carry the body force and the stress glut with
    OPPOSITE relative sign.  Equivalently, this package's ``Dsigma*`` is minus
    the stress glut ``m`` of the underlying theory.

With all three applied, ``W . A G B`` is symmetric to ~1e-15 (it was 0.40-0.81)
and the corrected 6x6 obeys the bare symplectic reciprocity law
``G(i<-j)(+k) = J6 [G(j<-i)(-k)]^T J6`` with no weight at all.

THE ONE RESTRICTION.  Source and receiver planes must lie in the INTERIOR of a
layer; there the correction is exact even with strong contrasts between them
(measured 1.7e-15 with a fast slab crossed twice).
``assert_interface_continuous`` enforces this.  The restriction originally came
from ``K`` being built on the local ``eta_S``, which is two-valued at a material
jump; ``K`` no longer uses ``eta_S`` (see below), but the rest of the
construction still reads a single local medium per plane, so the restriction
stands.

THE SH IMPEDANCE FIX, 2026-09-14.  ``K``'s perpendicular component previously
carried ``eta_S``.  That was not physics.  It compensated a missing ``eta`` in
``GlobalMatrix.layer_matrix.layer_eigenvectors_sh_batched``, whose SH
eigenvectors had traction/displacement ratio ``mu`` where an SH wave requires
``mu*eta``.  The pair cancelled exactly in UNIFORM media -- which is why GATE D,
GATE F, the symplectic law and the whole-space reduction all passed at 1e-15 and
none of them could see it -- while the LAYERED SH reflection came out
angle-independent, ``(mu1-mu2)/(mu1+mu2)`` instead of the Aki & Richards
``(mu1 eta1 - mu2 eta2)/(mu1 eta1 + mu2 eta2)``.  A reflection coefficient that
does not vary with the ray parameter is the tell.

Both halves are now corrected, and they are LETHAL APART: either alone takes the
uniform-limit reduction from 1e-15 to ~3e-1.  ``assert_sh_impedance_paired``
checks the sibling's declared convention at every layered call, and
``scripts/gate_sh_impedance.py`` holds the uniform limit and the interface
reflection simultaneously.  P and SV were correct throughout and are unchanged
(1.000000 against Kennett at every angle).

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
    "assert_sh_impedance_paired",
    "correct_6x6",
    "corrected_layered_6x6",
    "corrected_layered_9x9",
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


def k_operator(s_s: complex, kx: float, ky: float, omega: complex) -> NDArray:
    """The traction-half basis operator (defect D2), in (z, x, y).

    ``K = 1 (+) (-P_par + P_perp) = 1 (+) (I2 - 2 khat khat^T)`` acting on
    ``(sigma_zz, sigma_xz, sigma_yz)``, with ``P_par = khat khat^T`` and
    ``P_perp = I2 - P_par``.

    The in-plane block is a HOUSEHOLDER REFLECTION about the plane perpendicular
    to the horizontal propagation direction: purely geometric, independent of the
    medium and of frequency.  It is diagonal in (z, x, y) only when ``kx ky = 0``;
    elsewhere roughly two thirds of its weight is off-diagonal, which is why no
    diagonal source correction exists.  It is even in ``k`` and involutive
    (``K K = I``).

    SIMPLIFIED 2026-09-14, AND THE CHANGE IS COUPLED.  The perpendicular
    component previously carried a factor ``eta_S = sqrt(s_s^2 - p^2)``.  That
    factor was not physics: it compensated a missing ``eta`` in the SH impedance
    of ``GlobalMatrix.layer_matrix.layer_eigenvectors_sh_batched``, whose
    eigenvectors had traction/displacement ratio ``mu`` where the SH wave
    requires ``mu*eta``.  Together the two errors cancelled in UNIFORM media --
    which is why GATE D, GATE F and the whole-space reduction all passed at
    1e-15 and none of them could see it -- while leaving the layered SH
    reflection angle-INDEPENDENT and wrong.

    ``s_s`` and ``omega`` are therefore no longer used.  They are retained in the
    signature because every caller passes them and because the *next* reader will
    want to see, right here, that the medium dependence was removed deliberately.

    **Requires the matching change in
    ``GlobalMatrix/layer_matrix.py:layer_eigenvectors_sh_batched``.**  Applying
    either alone takes the uniform-limit reduction from 1e-15 to ~3e-1.  The pair
    is gated by ``scripts/gate_sh_impedance.py`` and checked at import time by
    ``assert_sh_impedance_paired`` below.

    Args:
        s_s: Complex S slowness of the LOCAL medium (s/km).  Unused; see above.
        kx: Horizontal wavenumber, x component (rad/km).
        ky: Horizontal wavenumber, y component (rad/km).
        omega: Angular frequency (rad/s).  Unused; see above.

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

    out = np.eye(3, dtype=complex)
    out[1:, 1:] = np.eye(2) - 2.0 * par
    return out


def assert_sh_impedance_paired() -> None:
    """Fail loudly if the sibling repository's SH convention does not match.

    The SH fix spans two repositories and the halves are lethal apart: either
    alone takes the uniform-limit reduction from 1e-15 to ~3e-1, silently, with
    no exception and no obviously wrong number.  Pulling one repository without
    the other is therefore the realistic failure, and this turns it into a
    diagnostic at the moment of use.

    Raises:
        RuntimeError: if the sibling declares a different SH impedance
            convention from the one this correction assumes.
    """
    try:
        from GlobalMatrix.layer_matrix import SH_IMPEDANCE_CONVENTION
    except ImportError:
        return  # sibling absent: nothing layered can run anyway

    if SH_IMPEDANCE_CONVENTION != "mu*neta":
        msg = (
            f"SH impedance convention mismatch: the sibling GlobalMatrix declares "
            f"{SH_IMPEDANCE_CONVENTION!r}, this correction assumes 'mu*neta'.\n"
            "  What: k_operator's in-plane block is a pure Householder reflection,\n"
            "        which is correct ONLY when the layered SH eigenvector carries\n"
            "        impedance mu*neta. An older sibling carries impedance mu and a\n"
            "        compensating eta_S belongs in K.\n"
            "  Where: GlobalMatrix/layer_matrix.py:layer_eigenvectors_sh_batched and\n"
            "         cubic_scattering/layered_correction.py:k_operator.\n"
            "  Expected: both at the 'mu*neta' convention.\n"
            "  Fix: update the sibling repository to the paired version, then run\n"
            "       scripts/gate_sh_impedance.py, which holds both constraints."
        )
        raise RuntimeError(msg) from None


def correct_6x6(
    g6: NDArray,
    omega: complex,
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

    assert_sh_impedance_paired()

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


def corrected_layered_9x9(
    model: object,
    omega: complex,
    kx: NDArray,
    ky: NDArray,
    source_iface: int,
    receiver_iface: int,
) -> NDArray:
    """Corrected stratified 9x9 in the ``(u, eps)`` basis, over a wavenumber grid.

    ``G9 = A(receiver) . correct_6x6(G6) . B(source)``. This is the stratified
    plane-to-plane propagator the directional-sweep solver uses as its vertical
    operator; ``sweep_kernels.vertical_kernel_9x9`` is its homogeneous limit, and
    the two agree to ~1e-15 (see ``scripts/gate_sweep_rung3_layered.py``).

    ATTENUATION CONSISTENCY -- the reason this exists rather than
    ``GlobalMatrix.layered_greens.layered_greens_9x9`` or
    ``scripts/composed_matvec.resolved_9x9_grid``. Both of those build ``A`` and
    ``B`` from ``_interface_elastic_properties``, which returns
    ``float(model.alpha[j])`` -- the UNDAMPED real velocity -- while ``G6``
    itself is computed from the complex, attenuative slowness. The operators and
    the Green's function then describe different media. At field Q (600-1000)
    the discrepancy is ~0.1% and invisible; at the Q = 2 used to isolate the
    whole-space limit it is 100%, and the homogeneous reduction fails outright
    (measured 1.009 relative). Symmetry gates cannot see it either, being
    homogeneous of degree one. This function takes both media from
    ``complex_slowness_p/s``, and the reduction then holds at 1e-15.

    Args:
        model: Stratified model accepted by ``layered_greens_6x6``.
        omega: Angular frequency (rad/s).
        kx: Horizontal wavenumber x-components, any shape.
        ky: Horizontal wavenumber y-components, same shape as ``kx``.
        source_iface: Source interface index.
        receiver_iface: Receiver interface index.

    Returns:
        Shape ``(*kx.shape, 9, 9)`` complex, basis
        ``(u_z, u_x, u_y, e_zz, e_xx, e_yy, 2e_xy, 2e_zy, 2e_zx)``.
    """
    from GlobalMatrix.layered_greens import layered_greens_6x6

    assert_sh_impedance_paired()
    assert_interface_continuous(model, source_iface, "source")
    assert_interface_continuous(model, receiver_iface, "receiver")

    shape = np.shape(kx)
    kxf = np.ravel(np.asarray(kx, dtype=float))
    kyf = np.ravel(np.asarray(ky, dtype=float))

    raw = layered_greens_6x6(
        model, omega, kxf, kyf, source_iface=source_iface, receiver_iface=receiver_iface
    )

    s_p = model.complex_slowness_p()  # type: ignore[attr-defined]
    s_s = model.complex_slowness_s()  # type: ignore[attr-defined]
    rho = model.rho  # type: ignore[attr-defined]
    j_s = max(int(source_iface), 1)
    j_r = max(int(receiver_iface), 1)
    al_s, be_s = 1.0 / s_p[j_s], 1.0 / s_s[j_s]
    al_r, be_r = 1.0 / s_p[j_r], 1.0 / s_s[j_r]

    out = np.empty((kxf.size, 9, 9), dtype=complex)
    for t in range(kxf.size):
        g_c = correct_6x6(raw[t], omega, s_s[j_s], s_s[j_r], kxf[t], kyf[t])
        a = strain_from_state(kxf[t], kyf[t], rho[j_r], al_r, be_r)
        b = source_jump_operator(kxf[t], kyf[t], rho[j_s], al_s, be_s)
        out[t] = a @ g_c @ b

    return out.reshape(*shape, 9, 9)
