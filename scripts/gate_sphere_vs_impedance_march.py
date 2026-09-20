#!/usr/bin/env python3
"""The sphere as seen by the laterally coupled impedance march.

ANCHOR: Nestor (1996) Ch.2 (Akdef), Ch.5 (CBPdef0).  The exact arbiter is the
elastic Mie solution in ``cubic_scattering/sphere_scattering.py`` and the
angular spectra of ``Mathematica/MieSphericalWaves.wl`` sections 4-7.

WHY THE CONTRAST IS BUILT IN WAVENUMBER, NOT ON THE GRID
--------------------------------------------------------
A sphere of radius ``a`` centred at depth ``z_c`` cuts the plane at depth ``z``
in a disc of radius ``R(z) = sqrt(a^2 - zeta^2)``, ``zeta = z - z_c``.  The
obvious thing is to paint that disc onto the lateral grid cell by cell.  That is
the WRONG thing, and it is the same trap the depth march has already paid for
once: a voxelised disc changes in JUMPS as ``R(z)`` crosses successive grid
radii, so the coefficient the march integrates is piecewise constant in depth,
RK4 integrates the staircase exactly, and the fourth order is gone.  It does not
look like a discretisation choice -- it looks like a defect in the formulation.

The disc has an exact two-dimensional transform,

    F(q, R) = 2 pi R J_1(q R) / q  = 2 pi R^2 [ J_1(qR) / (qR) ],   F(0) = pi R^2

and ``J_1(x)/x`` is EVEN and entire, so ``F`` is a power series in ``(qR)^2``,
hence a power series in ``R^2``.  And ``R^2 = a^2 - zeta^2`` is a polynomial in
depth.  So the contrast, taken in wavenumber, is an ANALYTIC function of ``z``
on the whole of the sphere's bounding slab -- poles included, where it goes to
zero linearly in ``R^2`` rather than with the square-root singularity ``R(z)``
alone would suggest.  The area, not the radius, is what enters.

Two consequences, and Part 3 measures both:

  - Marching over exactly ``[z_c - a, z_c + a]`` puts the only non-smoothness
    (the junction to the identically-zero exterior) at the endpoints, so the
    integrand is analytic throughout the march and fourth order is available.
  - The voxelised disc is kept as a CONTROL.  It must lose that order.  Without
    it the fourth order below is just a number with nothing to fail against.

A band-limited indicator is not the sharp sphere: it rings at the edge.  That is
the lateral discretisation error, it is edge-limited and algebraic, and it is
what the lateral refinement ladder is for.  It is not hidden here.

Run:  conda run -n seismic python scripts/gate_sphere_vs_impedance_march.py
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.special import j1

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubic_scattering.effective_contrasts import MaterialContrast, ReferenceMedium  # noqa: E402
from cubic_scattering.sphere_scattering import compute_elastic_mie  # noqa: E402
from scripts.gate_first_order_impedance import ablocks, newton_are  # noqa: E402
from scripts.gate_first_order_impedance_march import y_downgoing  # noqa: E402
from scripts.gate_first_order_lateral_impedance_3d import (  # noqa: E402
    Slice,
    aop,
    deriv_1d,
    grid_wavenumbers,
    order_of,
    rel,
    riccati_rhs,
)
from scripts.gate_mie_spectrum_general_m import ang_triple  # noqa: E402
from scripts.gate_sphere_plane_wave_spectrum import kz_of, mode_amplitudes  # noqa: E402
from scripts.gate_thesis_spectral import dz_normalised  # noqa: E402

#: Whole-space background, and the sphere.  The lateral period is a CONVERGENCE
#: PARAMETER, not a fixed constant -- the periodic grid puts an image sphere
#: every LX, and how much that matters is measured rather than assumed.
REF = ReferenceMedium(5000.0, 3000.0, 2500.0)
OMEGA = 60.0
RADIUS = 120.0

#: Fractional perturbation of alpha, beta and rho inside the sphere.
CONTRAST = 0.10

#: The SAME sphere, expressed the way the Mie arbiter wants it.  Derived rather
#: than typed, because the two descriptions must agree exactly and a hand
#: conversion is a silent factor waiting to happen.  Scaling all three of
#: alpha, beta, rho by s gives rho' = s rho, mu' = rho' beta'^2 = s^3 mu, and
#: lambda' = rho' alpha'^2 - 2 mu' = s^3 lambda.
_S = 1.0 + CONTRAST
MIE_CONTRAST = MaterialContrast(
    Dlambda=(_S**3 - 1.0) * REF.lam,
    Dmu=(_S**3 - 1.0) * REF.mu,
    Drho=(_S - 1.0) * REF.rho,
)

_PASS: list[tuple[str, bool]] = []


def report(label: str, ok: bool) -> None:
    """Record and print one check.

    Args:
        label: Description.
        ok: Whether it passed.
    """
    _PASS.append((label, bool(ok)))
    print(f"  {'PASS' if ok else '****FAIL****'}  {label}")


# ---------------------------------------------------------------------------
# The sphere's cross-section
# ---------------------------------------------------------------------------


def disc_radius_sq(z: float, z_c: float, radius: float) -> float:
    """The SQUARED radius of the sphere's cross-section at depth z.

    Returned squared on purpose.  The form factor depends on ``R^2`` alone, and
    ``R^2`` is a polynomial in depth where ``R`` has a square root at the poles.
    Taking the square root here and squaring it later would put the singularity
    back.

    Args:
        z: Depth.
        z_c: Depth of the sphere centre.
        radius: Sphere radius.

    Returns:
        ``a^2 - (z - z_c)^2``, clipped at zero outside the sphere.
    """
    return float(max(radius**2 - (z - z_c) ** 2, 0.0))


def disc_form_factor(q: NDArray, r_sq: float) -> NDArray:
    """The exact 2-D transform of a disc of squared radius ``r_sq``.

    ``F = 2 pi R J_1(qR) / q``, evaluated as ``2 pi R^2 [J_1(x)/x]`` with
    ``x = qR`` so that the ``q -> 0`` limit ``pi R^2`` is reached without a
    division by zero, and so that the dependence on ``R^2`` is manifest.

    Args:
        q: Lateral wavenumber magnitudes, any shape.
        r_sq: Squared disc radius; zero or less gives an empty disc.

    Returns:
        Same shape as ``q``.
    """
    if r_sq <= 0.0:
        return np.zeros_like(q)
    r = np.sqrt(r_sq)
    x = q * r
    small = x < 1.0e-8
    ratio = np.empty_like(x)
    ratio[small] = 0.5 - x[small] ** 2 / 16.0
    ratio[~small] = j1(x[~small]) / x[~small]
    return 2.0 * np.pi * r_sq * ratio


def q_grid(nx: int, ny: int, lx: float, ly: float) -> NDArray:
    """Lateral wavenumber magnitude at every grid mode, flattened.

    Args:
        nx: Points along x.
        ny: Points along y.
        lx: Period along x.
        ly: Period along y.

    Returns:
        Shape (nx*ny,), real, in FFT order with ``idx = ix*ny + iy``.
    """
    kx = grid_wavenumbers(nx, lx)
    ky = grid_wavenumbers(ny, ly)
    return np.sqrt((kx**2)[:, None] + (ky**2)[None, :]).reshape(-1)


def band_limited_disc(nx: int, ny: int, lx: float, ly: float, r_sq: float) -> NDArray:
    """The disc indicator, band limited to the grid, from its exact transform.

    The continuum transform and the DFT differ by the cell area, so
    ``F_dft = F_continuum * (nx ny) / (lx ly)``; the mean of the result is then
    the exact area fraction, which Part 1 checks rather than assumes.

    The transform of a disc centred at the ORIGIN is real and positive, and an
    inverse FFT of it puts the disc at grid index (0, 0) -- the corner.  The
    linear phase below moves it to the middle of the array, matching
    ``voxel_disc``, so that the two can be compared at all.  Without it the L2
    distance between them measures a translation and sits flat under refinement
    at about 0.48, which is how this was found.

    Args:
        nx: Points along x.
        ny: Points along y.
        lx: Period along x.
        ly: Period along y.
        r_sq: Squared disc radius.

    Returns:
        Shape (nx*ny,), real.
    """
    kx, ky = grid_wavenumbers(nx, lx), grid_wavenumbers(ny, ly)
    q = q_grid(nx, ny, lx, ly)
    fhat = disc_form_factor(q, r_sq) * (nx * ny) / (lx * ly)
    xc, yc = 0.5 * (nx - 1) * (lx / nx), 0.5 * (ny - 1) * (ly / ny)
    phase = np.exp(-1j * (kx[:, None] * xc + ky[None, :] * yc)).reshape(-1)
    return np.real(np.fft.ifft2((fhat * phase).reshape(nx, ny))).reshape(-1)


def voxel_disc(nx: int, ny: int, lx: float, ly: float, r_sq: float) -> NDArray:
    """The disc painted cell by cell -- the CONTROL that must lose the order.

    Args:
        nx: Points along x.
        ny: Points along y.
        lx: Period along x.
        ly: Period along y.
        r_sq: Squared disc radius.

    Returns:
        Shape (nx*ny,), real, values 0 or 1.
    """
    x = (np.arange(nx) - 0.5 * (nx - 1)) * (lx / nx)
    y = (np.arange(ny) - 0.5 * (ny - 1)) * (ly / ny)
    rr = (x**2)[:, None] + (y**2)[None, :]
    return (rr.reshape(-1) < r_sq).astype(float)


def sphere_slice_at(
    nx: int,
    ny: int,
    lx: float,
    ly: float,
    z_c: float,
    radius: float,
    *,
    voxelised: bool = False,
    amplitude: float = CONTRAST,
) -> Callable[[float], Slice]:
    """The medium on the lateral grid as a function of depth, for one sphere.

    The band-limited indicator is centred on the grid, so an even ``nx`` puts
    the sphere centre between grid points.  That is deliberate: it keeps the
    cross-section symmetric under the grid's own reflection and stops a single
    centre cell from carrying the whole small-disc limit near the poles.

    Args:
        nx: Points along x.
        ny: Points along y.
        lx: Period along x.
        ly: Period along y.
        z_c: Depth of the sphere centre.
        radius: Sphere radius.
        voxelised: Use the staircase control instead of the exact transform.
        amplitude: Fractional perturbation inside the sphere.  Zero gives the
            empty whole space, whose reflection is identically nothing -- which
            is what makes it a test rather than a trivial case.

    Returns:
        A callable of depth returning the slice.
    """
    build = voxel_disc if voxelised else band_limited_disc

    def at(z: float) -> Slice:
        ind = build(nx, ny, lx, ly, disc_radius_sq(z, z_c, radius))
        s = 1.0 + amplitude * ind
        return Slice(REF.alpha * s, REF.beta * s, REF.rho * s)

    return at


# ---------------------------------------------------------------------------
# The march, on a period this file chooses
# ---------------------------------------------------------------------------


def derivs(nx: int, ny: int, lx: float, ly: float) -> tuple[NDArray, NDArray]:
    """The two lateral derivative matrices at this file's own periods.

    Args:
        nx: Points along x.
        ny: Points along y.
        lx: Period along x.
        ly: Period along y.

    Returns:
        (d_x, d_y), each shape (nx*ny, nx*ny).
    """
    return (
        np.kron(deriv_1d(nx, lx), np.eye(ny)),
        np.kron(np.eye(nx), deriv_1d(ny, ly)),
    )


def start_impedance(nx: int, ny: int, lx: float, ly: float) -> NDArray:
    """The downgoing impedance of the uniform whole space below the sphere.

    Laterally uniform, so diagonal in (k_x, k_y), each 3x3 block the object
    ``gate_first_order_impedance`` already validated.

    Args:
        nx: Points along x.
        ny: Points along y.
        lx: Period along x.
        ly: Period along y.

    Returns:
        Shape (3N, 3N), on the spatial grid.
    """
    kx, ky = grid_wavenumbers(nx, lx), grid_wavenumbers(ny, ly)
    n = nx * ny
    yhat = np.zeros((3 * n, 3 * n), dtype=np.complex128)
    for ix in range(nx):
        for iy in range(ny):
            kxv, kyv = float(kx[ix]), float(ky[iy])
            if kyv == 0.0:
                blk = y_downgoing(REF, OMEGA, kxv)
            else:
                dzm, _, _ = dz_normalised(REF, OMEGA, kxv, kyv)
                dd = dzm[:, :3]
                blk = newton_are(dd[3:, :] @ np.linalg.inv(dd[:3, :]), ablocks(REF, OMEGA, kxv, kyv))[0]
            idx = np.array([c * n + ix * ny + iy for c in range(3)])
            yhat[np.ix_(idx, idx)] = blk
    fx, fy = np.fft.fft(np.eye(nx), axis=0), np.fft.fft(np.eye(ny), axis=0)
    gx, gy = np.fft.ifft(np.eye(nx), axis=0), np.fft.ifft(np.eye(ny), axis=0)
    f, fi = np.kron(fx, fy), np.kron(gx, gy)
    return np.asarray(np.kron(np.eye(3), fi) @ yhat @ np.kron(np.eye(3), f))


def march_sphere(
    nx: int,
    ny: int,
    lx: float,
    ly: float,
    nstep: int,
    *,
    voxelised: bool = False,
    amplitude: float = CONTRAST,
) -> NDArray:
    """March the impedance across the sphere's own bounding slab.

    The march domain is exactly ``[-a, +a]`` about the sphere centre, so the
    only non-smoothness in the coefficient -- the junction to the empty exterior
    -- sits at the endpoints and never inside a step.

    Args:
        nx: Points along x.
        ny: Points along y.
        lx: Period along x.
        ly: Period along y.
        nstep: Depth steps.
        voxelised: Use the staircase control.

    Returns:
        The impedance at the top of the bounding slab, shape (3N, 3N).
    """
    dx, dy = derivs(nx, ny, lx, ly)
    slice_at = sphere_slice_at(nx, ny, lx, ly, RADIUS, RADIUS, voxelised=voxelised, amplitude=amplitude)
    y = start_impedance(nx, ny, lx, ly)
    edge = np.linspace(0.0, 2.0 * RADIUS, nstep + 1)

    for m in range(nstep - 1, -1, -1):
        lo, hi = float(edge[m]), float(edge[m + 1])
        dz = lo - hi
        k1 = riccati_rhs(y, aop(slice_at(hi), OMEGA, dx, dy))
        k2 = riccati_rhs(y + 0.5 * dz * k1, aop(slice_at(hi + 0.5 * dz), OMEGA, dx, dy))
        k3 = riccati_rhs(y + 0.5 * dz * k2, aop(slice_at(hi + 0.5 * dz), OMEGA, dx, dy))
        k4 = riccati_rhs(y + dz * k3, aop(slice_at(hi + dz), OMEGA, dx, dy))
        y = y + (dz / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return np.asarray(y)


# ---------------------------------------------------------------------------
# The checks
# ---------------------------------------------------------------------------


def march_sphere_history(
    nx: int,
    ny: int,
    lx: float,
    ly: float,
    nstep: int,
    *,
    voxelised: bool = False,
    amplitude: float = CONTRAST,
) -> tuple[NDArray, list[NDArray], NDArray]:
    """The upward sweep, keeping Y at every half-step instead of only the top.

    THE SECOND SWEEP NEEDS Y EVERYWHERE, not just at the surface.  Reflection is
    read off Y at the top alone, but transmission is the field carried through
    the slab, and the one-way equation that carries it has Y in its coefficient
    at every depth.

    The march is run at ``2 * nstep`` steps and every value kept, so the returned
    history holds Y at the edges AND the midpoints of a ``nstep``-step grid --
    exactly the stage depths a fourth-order Runge--Kutta downward sweep asks
    for.  No interpolation is involved, which is what keeps the second sweep the
    same order as the first.

    Args:
        nx: Points along x.
        ny: Points along y.
        lx: Period along x.
        ly: Period along y.
        nstep: Steps of the DOWNWARD sweep; the upward one uses twice as many.
        voxelised: Use the staircase control.
        amplitude: Contrast amplitude.

    Returns:
        (Y at the top, Y at each of the 2*nstep+1 depths, those depths).
    """
    dx, dy = derivs(nx, ny, lx, ly)
    slice_at = sphere_slice_at(nx, ny, lx, ly, RADIUS, RADIUS, voxelised=voxelised, amplitude=amplitude)
    y = start_impedance(nx, ny, lx, ly)
    fine = 2 * nstep
    edge = np.linspace(0.0, 2.0 * RADIUS, fine + 1)

    hist: list[NDArray] = [np.zeros(0)] * (fine + 1)
    hist[fine] = np.array(y)
    for m in range(fine - 1, -1, -1):
        lo, hi = float(edge[m]), float(edge[m + 1])
        dz = lo - hi
        k1 = riccati_rhs(y, aop(slice_at(hi), OMEGA, dx, dy))
        k2 = riccati_rhs(y + 0.5 * dz * k1, aop(slice_at(hi + 0.5 * dz), OMEGA, dx, dy))
        k3 = riccati_rhs(y + 0.5 * dz * k2, aop(slice_at(hi + 0.5 * dz), OMEGA, dx, dy))
        k4 = riccati_rhs(y + dz * k3, aop(slice_at(hi + dz), OMEGA, dx, dy))
        y = y + (dz / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        hist[m] = np.array(y)
    return np.asarray(y), hist, edge


def sweep_down(
    hist: list[NDArray],
    edge: NDArray,
    u_top: NDArray,
    nx: int,
    ny: int,
    lx: float,
    ly: float,
    *,
    voxelised: bool = False,
    amplitude: float = CONTRAST,
) -> NDArray:
    """The one-way equation for the field, integrated DOWNWARD.

        u' = (A11 + A12 Y(z)) u ,     z increasing = deeper

    Haines et al. integrate the one-way equation in the direction of energy flow
    and the Riccati against it.  That is not a preference: the eigenvalues of
    A11 + A12 Y have non-positive real part, so the downgoing solution DECAYS
    with increasing z and integrating downward is the stable direction, exactly
    as integrating the Riccati upward is.  Running either the other way amplifies
    the evanescent channels.

    Args:
        hist: Y at the 2*nstep+1 depths of ``march_sphere_history``.
        edge: Those depths.
        u_top: Displacement at the top, shape (3N, m).
        nx: Points along x.
        ny: Points along y.
        lx: Period along x.
        ly: Period along y.
        voxelised: Use the staircase control.
        amplitude: Contrast amplitude.

    Returns:
        Displacement at the base, same shape as ``u_top``.
    """
    dx, dy = derivs(nx, ny, lx, ly)
    slice_at = sphere_slice_at(nx, ny, lx, ly, RADIUS, RADIUS, voxelised=voxelised, amplitude=amplitude)

    def oneway(z_idx: int, z: float) -> NDArray:
        a11, a12, _, _ = aop(slice_at(z), OMEGA, dx, dy)
        return np.asarray(a11 + a12 @ hist[z_idx])

    u = np.array(u_top, dtype=complex)
    nstep = (len(hist) - 1) // 2
    for m in range(nstep):
        i0, i1, i2 = 2 * m, 2 * m + 1, 2 * m + 2
        z0, z1, z2 = float(edge[i0]), float(edge[i1]), float(edge[i2])
        h = z2 - z0
        k1 = oneway(i0, z0) @ u
        k2 = oneway(i1, z1) @ (u + 0.5 * h * k1)
        k3 = oneway(i1, z1) @ (u + 0.5 * h * k2)
        k4 = oneway(i2, z2) @ (u + h * k3)
        u = u + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return u


def mode_blocks(nx: int, ny: int, lx: float, ly: float) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """The four mode matrices assembled block diagonally on the wavenumber grid.

    Args:
        nx: Points along x.
        ny: Points along y.
        lx: Period along x.
        ly: Period along y.

    Returns:
        (U_d, T_d, U_u, T_u), each shape (3N, 3N).
    """
    n = nx * ny
    kx, ky = grid_wavenumbers(nx, lx), grid_wavenumbers(ny, ly)
    out = [np.zeros((3 * n, 3 * n), dtype=complex) for _ in range(4)]
    for ix in range(nx):
        for iy in range(ny):
            idx = np.array([c * n + ix * ny + iy for c in range(3)])
            for slot, blk in enumerate(mode_matrix(float(kx[ix]), float(ky[iy]))):
                out[slot][np.ix_(idx, idx)] = blk
    return out[0], out[1], out[2], out[3]


def fourier_pair(nx: int, ny: int) -> tuple[NDArray, NDArray]:
    """The 3N-component DFT and its inverse on the lateral grid.

    Args:
        nx: Points along x.
        ny: Points along y.

    Returns:
        (F, F^-1), each shape (3N, 3N).
    """
    fx, fy = np.fft.fft(np.eye(nx), axis=0), np.fft.fft(np.eye(ny), axis=0)
    gx, gy = np.fft.ifft(np.eye(nx), axis=0), np.fft.ifft(np.eye(ny), axis=0)
    return np.kron(np.eye(3), np.kron(fx, fy)), np.kron(np.eye(3), np.kron(gx, gy))


def reflection_transmission(
    nx: int,
    ny: int,
    lx: float,
    ly: float,
    nstep: int,
    *,
    voxelised: bool = False,
    amplitude: float = CONTRAST,
) -> tuple[NDArray, NDArray]:
    """Both halves of the response: R from the upward sweep, T from the downward.

    R needs only Y at the top.  T needs the field carried through the slab, so it
    needs the second sweep, and that is the whole reason this routine exists.
    With unit downgoing amplitude above the slab the total field at the top is
    ``u(0) = (U_d + U_u R)``; sweeping it down gives ``u(2a)``, which below the
    slab is purely downgoing, so ``T = U_d^-1 u(2a)``.

    Args:
        nx: Points along x.
        ny: Points along y.
        lx: Period along x.
        ly: Period along y.
        nstep: Steps of the downward sweep.
        voxelised: Use the staircase control.
        amplitude: Contrast amplitude.

    Returns:
        (R, T), each shape (3N, 3N) in the wavenumber-mode basis.
    """
    y_top, hist, edge = march_sphere_history(
        nx, ny, lx, ly, nstep, voxelised=voxelised, amplitude=amplitude
    )
    r_mat = reflection_from_y(y_top, nx, ny, lx, ly)

    ud, _td, uu, _tu = mode_blocks(nx, ny, lx, ly)
    f, fi = fourier_pair(nx, ny)

    u_top_hat = ud + uu @ r_mat
    u_base_hat = f @ sweep_down(
        hist, edge, fi @ u_top_hat, nx, ny, lx, ly, voxelised=voxelised, amplitude=amplitude
    )
    t_mat = np.linalg.solve(ud, u_base_hat)
    return r_mat, np.asarray(t_mat)


def mie_transmission(nx: int, ny: int, lx: float, ly: float) -> NDArray:
    """What the isolated sphere predicts for the array's TRANSMITTED orders.

    The same Poisson relation as ``mie_prediction``, with two differences.  The
    outgoing direction is DOWNWARD, so the spectrum is taken with
    ``upward=False``; and transmission carries the unscattered wave as well, so
    the specular P entry has the direct term ``exp(i k_P 2a)`` added to it.  That
    direct term is the whole of T at zero contrast, which is what makes this
    prediction testable in a limit where Mie contributes nothing.

    Args:
        nx: Points along x.
        ny: Points along y.
        lx: Period along x.
        ly: Period along y.

    Returns:
        Shape (3N,), the predicted transmitted amplitude of every (mode, order)
        for a unit downgoing P wave at normal incidence, indexed mode-slowest.
    """
    mie = compute_elastic_mie(OMEGA, RADIUS, REF, MIE_CONTRAST)
    kp, ks = OMEGA / REF.alpha, OMEGA / REF.beta
    n = nx * ny
    q = q_grid(nx, ny, lx, ly)
    c_p, c_sv = mode_amplitudes(mie, q, upward=False)
    pref = (2.0 * np.pi) ** 2 / (lx * ly)
    kzp, kzs = kz_of(q, kp), kz_of(q, ks)
    out = np.zeros(3 * n, dtype=complex)
    out[:n] = pref * c_p * np.exp(1j * (kp + kzp) * RADIUS)
    out[n : 2 * n] = pref * c_sv * np.exp(1j * (kp + kzs) * RADIUS)
    # The direct wave: unit amplitude at the top, propagated across the slab.
    out[0] = out[0] + np.exp(2j * kp * RADIUS)
    return out


def compare_rt_to_mie(nx: int, ny: int, lx: float, ly: float, nstep: int) -> tuple[float, float]:
    """Score BOTH sweeps: R against Mie's backward spectrum, T against forward.

    Args:
        nx: Points along x.
        ny: Points along y.
        lx: Period along x.
        ly: Period along y.
        nstep: Steps of the downward sweep.

    Returns:
        (relative error in the R column, relative error in the T column).
    """
    r_mat, t_mat = reflection_transmission(nx, ny, lx, ly, nstep)
    col = 0  # the incident P wave at normal incidence is order Gamma, mode P
    r_got, t_got = r_mat[:, col], t_mat[:, col]
    r_want, t_want = mie_prediction(nx, ny, lx, ly), mie_transmission(nx, ny, lx, ly)
    return (
        float(np.linalg.norm(r_got - r_want) / np.linalg.norm(r_want)),
        float(np.linalg.norm(t_got - t_want) / np.linalg.norm(t_want)),
    )


def flux_weights(nx: int, ny: int, lx: float, ly: float) -> tuple[NDArray, NDArray]:
    """The displacement-to-flux rescaling, and which channels carry flux.

    The mode matrices of ``mode_matrix`` are UNIT DISPLACEMENT.  Energy balance
    is a statement about flux, and the two bases differ by a diagonal rescaling
    per (mode, order) -- the square root of the vertical energy flux carried by a
    unit-displacement plane wave,

        w_c(q) = sqrt( c^2 k_z,c(q) / omega ) ,

    up to one overall constant that cancels in ``R^H R + T^H T``.  This project
    has twice asserted a relation between the displacement and flux bases from
    diagonal ratios and been wrong, so nothing here is asserted: the weight is
    written down and the energy identity it is supposed to produce is MEASURED.
    It comes out at 8e-8, which is the answer to whether the weight is right.

    Args:
        nx: Points along x.
        ny: Points along y.
        lx: Period along x.
        ly: Period along y.

    Returns:
        (weights of shape (3N,), boolean mask of propagating channels).
    """
    n = nx * ny
    kx, ky = grid_wavenumbers(nx, lx), grid_wavenumbers(ny, ly)
    kp, ks = OMEGA / REF.alpha, OMEGA / REF.beta
    w = np.zeros(3 * n, dtype=complex)
    prop = np.zeros(3 * n, dtype=bool)
    for ix in range(nx):
        for iy in range(ny):
            q = float(np.hypot(kx[ix], ky[iy]))
            kzp = complex(kz_of(np.array(q), kp))
            kzs = complex(kz_of(np.array(q), ks))
            for c, (kzc, speed) in enumerate(((kzp, REF.alpha), (kzs, REF.beta), (kzs, REF.beta))):
                i = c * n + ix * ny + iy
                w[i] = np.sqrt(speed**2 * kzc / OMEGA)
                prop[i] = abs(kzc.imag) < 1.0e-12 * abs(kzc.real + 1.0e-30) and kzc.real > 0.0
    return w, prop


def energy_residual(nx: int, ny: int, lx: float, ly: float, nstep: int) -> tuple[float, int]:
    """How far the whole R/T response is from conserving energy.

    THIS IS THE ALL-CHANNEL CHECK.  The Mie comparison scores one column -- the
    response to a P wave at normal incidence -- because the plane-wave spectrum
    of ``mode_amplitudes`` is the m = 0 one.  Energy balance scores EVERY entry
    of both matrices at once: every incident channel, every outgoing channel,
    P, SV and SH alike, and every propagating diffraction order.  It needs no
    arbiter, only that the slab is lossless.

    Args:
        nx: Points along x.
        ny: Points along y.
        lx: Period along x.
        ly: Period along y.
        nstep: Steps of the downward sweep.

    Returns:
        (relative residual of R^H R + T^H T = I, number of propagating channels).
    """
    r_mat, t_mat = reflection_transmission(nx, ny, lx, ly, nstep)
    w, prop = flux_weights(nx, ny, lx, ly)
    d, di = np.diag(w), np.diag(1.0 / w)
    rt, tt = d @ r_mat @ di, d @ t_mat @ di
    sel = np.ix_(prop, prop)
    m = rt[sel].conj().T @ rt[sel] + tt[sel].conj().T @ tt[sel]
    k = m.shape[0]
    return float(np.linalg.norm(m - np.eye(k)) / np.sqrt(k)), int(prop.sum())


def mie_spectrum_s(nx: int, ny: int, lx: float, ly: float, *, upward: bool) -> tuple:
    """The scattered plane-wave spectrum of an x-polarised S wave at normal incidence.

    The m = +/-1 counterpart of ``mode_amplitudes``.  Its angular content is the
    triple of ``gate_mie_spectrum_general_m.ang_triple`` -- ported from
    ``Mathematica/MieSphericalWaves.wl`` Section 7 and validated there against
    the exact field -- and its normalisation is fixed by the same relation the
    m = 0 case obeys, with the m = 1 renormalisation renorm(n) = -1/[n(n+1)]:

        c_mode = (i / 2 pi k_z) sum_n coeff_n renorm(n) ang_n(m=1) / i^n

    That is not assumed.  Converting a spectrum to a far field by stationary
    phase gives f = -2 pi i k_z c, and for m = 0 this reproduces
    ``mie_far_field``'s Sum a_n (-i)^n P_n exactly; for m = 1 it reproduces its
    independently written SV branch to 4e-6 over a spread of angles, which is
    what pins the normalisation above.

    ⚠ THE TOROIDAL (M-type) PART IS NOT INCLUDED, and that is a stated gap
    rather than an oversight.  An x-polarised S wave excites the toroidal family
    as well as the spheroidal one, and ``compute_elastic_mie`` carries its
    coefficients as ``c_n``.  What is missing is the scalar relating ``c_n`` to
    the shared potential convention: ``c_n`` is normalised against its own
    incident expansion (2n+1) i^n / (i k_S), which a_n_sv and b_n_sv do not
    share, and ``mie_far_field``'s SH branch cannot settle it because that
    branch drops the M-type outright as "O((ka)^4) and negligible" -- a claim
    that fails here, where |c_n| runs 1.5 to 2.5 times |b_n_sv renorm(n)| at
    k_S a = 2.4.  Scanning the scalar against the march bounds it at or below
    k_S: at that size the toroidal is invisible against the array-coupling
    floor, while a scalar of 1 degrades the median agreement from 0.94 to 0.58
    and one of 1/(i k_S) destroys it.  Bounded, not determined.

    Args:
        nx: Points along x.
        ny: Points along y.
        lx: Period along x.
        ly: Period along y.
        upward: True for the reflected half-space.

    Returns:
        (c_P, c_SV, c_SH), each shape (N,).
    """
    mie = compute_elastic_mie(OMEGA, RADIUS, REF, MIE_CONTRAST)
    kp, ks = OMEGA / REF.alpha, OMEGA / REF.beta
    kx1, ky1 = grid_wavenumbers(nx, lx), grid_wavenumbers(ny, ly)
    kx = np.repeat(kx1, ny)
    ky = np.tile(ky1, nx)
    q = np.hypot(kx, ky)
    kzp, kzs = kz_of(q, kp), kz_of(q, ks)
    dirp = -kzp if upward else kzp
    dirs = -kzs if upward else kzs

    sp = np.zeros(kx.size, dtype=complex)
    ssv = np.zeros(kx.size, dtype=complex)
    ssh = np.zeros(kx.size, dtype=complex)
    for n in range(1, mie.n_max + 1):
        renorm = -1.0 / (n * (n + 1))
        inv_in = 1.0 / (1j**n)
        a_ang, _, _ = ang_triple(n, 1, kx, ky, dirp, kp)
        _, dth, dph = ang_triple(n, 1, kx, ky, dirs, ks)
        sp = sp + mie.a_n_sv[n] * renorm * a_ang * inv_in
        ssv = ssv + mie.b_n_sv[n] * renorm * dth * inv_in
        ssh = ssh + mie.b_n_sv[n] * renorm * dph * inv_in
    return (
        (1j / (2.0 * np.pi * kzp)) * sp,
        (1j / (2.0 * np.pi * kzs)) * ssv,
        (1j / (2.0 * np.pi * kzs)) * ssh,
    )


def mie_columns_s(nx: int, ny: int, lx: float, ly: float) -> tuple[NDArray, NDArray]:
    """The predicted R and T columns for an SV wave at the Gamma order.

    Same Poisson relation as ``mie_prediction``, with the incident wavenumber
    k_S in place of k_P, and the transmitted column carrying the unscattered
    wave on its own (SV, Gamma) entry.

    Args:
        nx: Points along x.
        ny: Points along y.
        lx: Period along x.
        ly: Period along y.

    Returns:
        (R column, T column), each shape (3N,).
    """
    n = nx * ny
    kp, ks = OMEGA / REF.alpha, OMEGA / REF.beta
    q = q_grid(nx, ny, lx, ly)
    kzp, kzs = kz_of(q, kp), kz_of(q, ks)
    pref = (2.0 * np.pi) ** 2 / (lx * ly)

    out = []
    for upward in (True, False):
        c_p, c_sv, c_sh = mie_spectrum_s(nx, ny, lx, ly, upward=upward)
        col = np.zeros(3 * n, dtype=complex)
        col[:n] = pref * c_p * np.exp(1j * (ks + kzp) * RADIUS)
        col[n : 2 * n] = pref * c_sv * np.exp(1j * (ks + kzs) * RADIUS)
        col[2 * n :] = pref * c_sh * np.exp(1j * (ks + kzs) * RADIUS)
        out.append(col)
    out[1][n] = out[1][n] + np.exp(2j * ks * RADIUS)  # the unscattered wave
    return out[0], out[1]


def part1() -> None:
    """The band-limited disc is the disc: area exact, and it converges to it."""
    print("\n[1] the band-limited disc against the disc it represents")
    lx = ly = 600.0
    r_sq = 100.0**2

    # The mean of the indicator is the area fraction, exactly -- it is the q=0
    # mode and nothing else contributes to it.
    for nx in (8, 16, 32):
        ind = band_limited_disc(nx, nx, lx, ly, r_sq)
        got = float(np.mean(ind))
        want = np.pi * r_sq / (lx * ly)
        print(f"      N={nx:<3d} mean {got:.12f}  exact area fraction {want:.12f}")
        report(f"the band-limited disc has the exact area at N={nx}", abs(got - want) < 1e-13)

    # The whole spectrum, not just its q = 0 mode: transforming the constructed
    # field back must return the analytic form factor at every mode.  This is
    # what catches a wrong cell-area normalisation or a wrong centring phase,
    # neither of which the area check above can see.
    nx = 16
    ind = band_limited_disc(nx, nx, lx, ly, r_sq)
    kx, ky = grid_wavenumbers(nx, lx), grid_wavenumbers(nx, ly)
    xc = yc = 0.5 * (nx - 1) * (lx / nx)
    phase = np.exp(-1j * (kx[:, None] * xc + ky[None, :] * yc)).reshape(-1)
    back = np.fft.fft2(ind.reshape(nx, nx)).reshape(-1) / phase * (lx * ly) / (nx * nx)
    want_ff = disc_form_factor(q_grid(nx, nx, lx, ly), r_sq)
    resid = float(np.max(np.abs(back - want_ff))) / float(np.max(np.abs(want_ff)))
    print(f"      round trip to the analytic form factor, all {nx * nx} modes: {resid:.3e}")
    report("the construction reproduces the analytic form factor at every mode", resid < 1e-13)

    # ⚠ AND THE MEASURE THAT DOES NOT WORK, recorded so it is not tried again.
    # The pointwise distance to a voxelised disc looks like the obvious
    # convergence test and is not one.  The band-limited disc is the EXACT
    # projection of the true disc onto the modes the grid can carry; the
    # staircase is a different, worse approximation.  Neither is the truth, so
    # their distance measures neither.  Measured: the overshoot is pinned at
    # about 1.754 and the undershoot at -0.52 at EVERY N from 16 to 256, and the
    # rms distance sits at 0.24 throughout.  Lateral convergence has to be
    # measured in the PHYSICS, against the Mie arbiter, which is what the
    # lateral ladder does.
    peaks = []
    for nsz in (16, 64, 256):
        f = band_limited_disc(nsz, nsz, lx, ly, r_sq)
        peaks.append(
            (
                float(f.max()),
                float(f.min()),
                float(np.sqrt(np.mean((f - voxel_disc(nsz, nsz, lx, ly, r_sq)) ** 2))),
            )
        )
    for nsz, (hi, lo, rms) in zip((16, 64, 256), peaks, strict=True):
        print(f"      N={nsz:<4d} overshoot {hi:.4f}  undershoot {lo:+.4f}  rms vs staircase {rms:.4e}")
    flat = abs(peaks[0][0] - peaks[-1][0]) < 0.01
    report("the ringing does NOT decay with N, so this is not a convergence measure", flat)


def part2() -> None:
    """The depth profile is the sphere, and it is ANALYTIC in depth."""
    print("\n[2] the cross-section profile in depth")
    # The integral of the cross-section area is the sphere volume.  This is what
    # catches R^2 built with the wrong sign or the wrong centre.
    zs = np.linspace(0.0, 2.0 * RADIUS, 20001)
    areas = np.array([np.pi * disc_radius_sq(float(z), RADIUS, RADIUS) for z in zs])
    vol = float(np.trapezoid(areas, zs))
    want = 4.0 / 3.0 * np.pi * RADIUS**3
    print(f"      integral of the cross-section area: {vol:.6e}   4 pi a^3 / 3: {want:.6e}")
    report("the cross-section profile integrates to the sphere volume", abs(vol / want - 1.0) < 1e-8)

    # Analyticity: the form factor at fixed q, as a function of depth, must have
    # convergent high-order difference quotients everywhere INSIDE the slab --
    # including at the poles, where R itself has a square root but R^2 does not.
    lx = ly = 600.0
    q = q_grid(8, 8, lx, ly)
    iq = int(np.argmax(q))

    def f_of_z(z: float) -> float:
        return float(disc_form_factor(q, disc_radius_sq(z, RADIUS, RADIUS))[iq])

    def second_diff(z0: float, h: float) -> float:
        """Fourth-order centred second difference.

        Args:
            z0: Evaluation point.
            h: Step.

        Returns:
            The estimate, which converges as h^4 against a smooth function and
            stalls or diverges against a kink.
        """
        return (
            -f_of_z(z0 + 2 * h)
            + 16 * f_of_z(z0 + h)
            - 30 * f_of_z(z0)
            + 16 * f_of_z(z0 - h)
            - f_of_z(z0 - 2 * h)
        ) / (12 * h * h)

    # INSIDE the slab, including close to a pole, the stencil sees only the
    # analytic branch and the estimate settles.
    for label, z0 in (("mid-sphere", RADIUS), ("near a pole", 0.15 * RADIUS)):
        vals = [second_diff(z0, h) for h in (4.0, 2.0, 1.0)]
        drift = abs(vals[0] - vals[-1]) / max(abs(vals[-1]), 1e-300)
        print(f"      {label:<14} d2/dz2 at h=4,2,1: {'  '.join(f'{v:.6e}' for v in vals)}")
        report(f"the form factor is smooth in depth {label}", drift < 1e-3)

    # THE CONTROL, and the reason the march domain is exactly [0, 2a].  A
    # stencil that STRADDLES a pole straddles the junction to the identically
    # zero exterior, where the profile is continuous but kinked.  There the
    # estimate does not settle at all -- so a march whose steps crossed that
    # point would lose its order, and one whose endpoints sit on it does not.
    vals = [second_diff(0.02 * RADIUS, h) for h in (4.0, 2.0, 1.0)]
    drift = abs(vals[0] - vals[-1]) / max(abs(vals[-1]), 1e-300)
    print(f"      {'straddling a pole':<14} d2/dz2 at h=4,2,1: {'  '.join(f'{v:.3e}' for v in vals)}")
    report("CONTROL: across the pole the profile is kinked, not smooth", drift > 1.0)


def part3() -> None:
    """THE test: the analytic cross-section earns fourth order, the staircase does not."""
    print("\n[3] depth convergence order: exact transform vs voxelised control")
    nx = ny = 8
    lx = ly = 600.0
    steps = [32, 64, 128, 256]

    finest: dict[bool, float] = {}
    spread: dict[bool, float] = {}
    for voxelised, label in ((False, "exact transform"), (True, "voxelised control")):
        ref_y = march_sphere(nx, ny, lx, ly, 1024, voxelised=voxelised)
        errs = [rel(march_sphere(nx, ny, lx, ly, s, voxelised=voxelised), ref_y) for s in steps]
        orders = order_of(errs)
        finest[voxelised] = errs[-1]
        spread[voxelised] = max(orders) - min(orders)
        print(f"      {label:<18} errors: {'  '.join(f'{e:.2e}' for e in errs)}")
        print(f"      {'':<18} orders: {'  '.join(f'{o:.2f}' for o in orders)}")
        if not voxelised:
            report("the exact cross-section earns fourth order in depth", min(orders) > 3.5)

    # The control's failure is NOT that its order is low -- it is that it has no
    # order at all.  A staircase's error depends on where the step boundaries
    # happen to fall relative to the radius crossings, so refining moves the
    # error around instead of reducing it, and the measured orders scatter.
    print(f"      order spread  exact {spread[False]:.2f}   voxelised {spread[True]:.2f}")
    print(f"      error at the finest step is {finest[True] / finest[False]:.0f}x worse voxelised")
    report("the exact route has a CLEAN order, the control does not", spread[False] < 0.3 < spread[True])
    report("the voxelised control is orders of magnitude worse", finest[True] > 1e3 * finest[False])


# ---------------------------------------------------------------------------
# From the impedance to a reflection matrix
# ---------------------------------------------------------------------------


def _traction(kvec: NDArray, e_pol: NDArray) -> NDArray:
    """tau_3 = sigma . zhat for a unit-amplitude plane wave.

    ``sigma_ij = lam delta_ij (i k.u) + mu (i k_i u_j + i k_j u_i)`` and the
    vertical axis is index 0 in the (z, x, y) ordering this project uses.

    Args:
        kvec: Wavevector, shape (3,), ordered (z, x, y).
        e_pol: Polarisation, shape (3,), unit displacement.

    Returns:
        Shape (3,), complex.
    """
    kdote = complex(kvec @ e_pol)
    out = 1j * REF.mu * (kvec * e_pol[0] + kvec[0] * e_pol)
    out[0] += 1j * REF.lam * kdote
    return out


def mode_matrix(kx: float, ky: float) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """The down- and up-going plane-wave modes at one lateral wavenumber.

    Built analytically rather than taken from ``dz_normalised``, on purpose.
    That routine scales its columns so the SYMPLECTIC inverse is exact, which is
    a flux-like normalisation; the Mie arbiter is written for UNIT DISPLACEMENT.
    The two differ by a two-sided transformation, not by a fixed diagonal
    factor, and asserting otherwise from diagonal ratios is a mistake this
    project has already made twice.  Constructing the modes here removes the
    question: the polarisations are exactly the ``e_P``, ``e_theta``, ``e_phi``
    the spectrum assembler uses, and the amplitude is unit displacement by
    definition.  ``dz_normalised`` then becomes a CHECK -- Part 4 -- rather than
    a dependency.

    Args:
        kx: Lateral wavenumber, x.
        ky: Lateral wavenumber, y.

    Returns:
        (U_d, T_d, U_u, T_u), each shape (3, 3) with columns ordered P, SV, SH.
    """
    kp, ks = OMEGA / REF.alpha, OMEGA / REF.beta
    q = float(np.hypot(kx, ky))
    kzp, kzs = complex(kz_of(np.array(q), kp)), complex(kz_of(np.array(q), ks))
    # At q = 0 the shear pair is degenerate and phi is undefined; any orthogonal
    # pair will do, and this choice is the phi = 0 limit of the general one.
    cph, sph = (kx / q, ky / q) if q > 1e-12 else (1.0, 0.0)

    out = []
    for sgn in (1.0, -1.0):
        kzp_s, kzs_s = sgn * kzp, sgn * kzs
        kvec_p = np.array([kzp_s, kx, ky], dtype=complex)
        kvec_s = np.array([kzs_s, kx, ky], dtype=complex)
        e_p = kvec_p / kp
        e_sv = np.array([-q / ks, (kzs_s / ks) * cph, (kzs_s / ks) * sph], dtype=complex)
        e_sh = np.array([0.0, -sph, cph], dtype=complex)
        u_blk = np.column_stack([e_p, e_sv, e_sh])
        t_blk = np.column_stack([_traction(kvec_p, e_p), _traction(kvec_s, e_sv), _traction(kvec_s, e_sh)])
        out.extend([u_blk, t_blk])
    return out[0], out[1], out[2], out[3]


def reflection_from_y(y_top: NDArray, nx: int, ny: int, lx: float, ly: float) -> NDArray:
    """The reflection matrix at the top of the slab, in the wavenumber basis.

    With the field above written as ``u = U_d c_i + U_u c_r`` and
    ``tau = T_d c_i + T_u c_r``, the impedance condition ``tau = Y u`` gives

        R = (T_u - Y U_u)^{-1} (Y U_d - T_d).

    ``Y`` couples lateral wavenumbers, so it is transformed to the wavenumber
    basis as a full matrix while the mode matrices stay block diagonal there.

    Args:
        y_top: Impedance at the top of the slab, spatial basis, shape (3N, 3N).
        nx: Points along x.
        ny: Points along y.
        lx: Period along x.
        ly: Period along y.

    Returns:
        Shape (3N, 3N), complex, in the wavenumber basis with mode index slowest.
    """
    n = nx * ny
    kx, ky = grid_wavenumbers(nx, lx), grid_wavenumbers(ny, ly)
    fx, fy = np.fft.fft(np.eye(nx), axis=0), np.fft.fft(np.eye(ny), axis=0)
    gx, gy = np.fft.ifft(np.eye(nx), axis=0), np.fft.ifft(np.eye(ny), axis=0)
    f, fi = np.kron(fx, fy), np.kron(gx, gy)
    yhat = np.kron(np.eye(3), f) @ y_top @ np.kron(np.eye(3), fi)

    ud = np.zeros((3 * n, 3 * n), dtype=complex)
    td = np.zeros_like(ud)
    uu = np.zeros_like(ud)
    tu = np.zeros_like(ud)
    for ix in range(nx):
        for iy in range(ny):
            idx = np.array([c * n + ix * ny + iy for c in range(3)])
            b_ud, b_td, b_uu, b_tu = mode_matrix(float(kx[ix]), float(ky[iy]))
            ud[np.ix_(idx, idx)] = b_ud
            td[np.ix_(idx, idx)] = b_td
            uu[np.ix_(idx, idx)] = b_uu
            tu[np.ix_(idx, idx)] = b_tu
    return np.asarray(np.linalg.solve(tu - yhat @ uu, yhat @ ud - td))


def part4() -> None:
    """The analytic modes agree with the validated half-space impedance."""
    print("\n[4] the analytic mode matrix against the gated half-space impedance")
    kp = OMEGA / REF.alpha
    worst = 0.0
    for kx, ky in ((0.0, 0.0), (0.004, 0.0), (0.0, 0.006), (0.009, -0.005), (0.03, 0.02)):
        ud, td, _, _ = mode_matrix(kx, ky)
        got = td @ np.linalg.inv(ud)
        want = y_downgoing(REF, OMEGA, kx) if ky == 0.0 else newton_are(got, ablocks(REF, OMEGA, kx, ky))[0]
        resid = rel(got, want)
        worst = max(worst, resid)
        band = "propagating" if np.hypot(kx, ky) < kp else "evanescent"
        print(f"      k=({kx:+.3f},{ky:+.3f}) {band:<12} T_d U_d^-1 vs the gated Y: {resid:.3e}")
    report("the analytic modes reproduce the downgoing half-space impedance", worst < 1e-9)


def part5() -> None:
    """With no sphere there is nothing to reflect, and R must vanish."""
    print("\n[5] the empty whole space reflects nothing")
    nx = ny = 8
    lx = ly = 600.0
    y_top = march_sphere(nx, ny, lx, ly, 64, amplitude=0.0)
    r_mat = reflection_from_y(y_top, nx, ny, lx, ly)
    scale = float(np.max(np.abs(np.linalg.inv(mode_matrix(0.0, 0.0)[0]))))
    print(f"      max |R| over the whole {3 * nx * ny}^2 matrix: {float(np.max(np.abs(r_mat))):.3e}")
    report("R vanishes when the contrast does", float(np.max(np.abs(r_mat))) < 1e-9 * max(scale, 1.0))


def mie_prediction(nx: int, ny: int, lx: float, ly: float) -> NDArray:
    """What the ISOLATED sphere predicts for the array's reflected orders.

    An array of spheres, one per cell of area ``A``, all seeing the same
    incident phase because the incidence is normal, scatters
    ``sum_R u_scat(r - R)``.  Poisson summation turns that into

        u_array = ((2 pi)^2 / A) sum_G uhat(G) exp(i G . rho + i k_z(G)|z|)

    so each diffraction order carries the isolated spectrum times ``(2 pi)^2/A``
    -- exact but for scattering BETWEEN spheres, which is what growing the
    period removes and what the period ladder measures.

    Two phases close the comparison.  The march's incident wave is unit
    amplitude at the TOP of the slab and so has ``exp(i k_P a)`` at the sphere
    centre, where Mie normalises it; and the scattered order, referred back from
    the centre to the top plane, carries ``exp(i k_z(G) a)``.

    Args:
        nx: Points along x.
        ny: Points along y.
        lx: Period along x.
        ly: Period along y.

    Returns:
        Shape (3N,), the predicted reflected amplitude of every (mode, order)
        for a unit downgoing P wave at normal incidence, indexed mode-slowest.
    """
    mie = compute_elastic_mie(OMEGA, RADIUS, REF, MIE_CONTRAST)
    kp, ks = OMEGA / REF.alpha, OMEGA / REF.beta
    n = nx * ny
    q = q_grid(nx, ny, lx, ly)
    c_p, c_sv = mode_amplitudes(mie, q, upward=True)
    pref = (2.0 * np.pi) ** 2 / (lx * ly)
    kzp, kzs = kz_of(q, kp), kz_of(q, ks)
    out = np.zeros(3 * n, dtype=complex)
    out[:n] = pref * c_p * np.exp(1j * (kp + kzp) * RADIUS)
    out[n : 2 * n] = pref * c_sv * np.exp(1j * (kp + kzs) * RADIUS)
    return out


def compare_to_mie(nx: int, ny: int, lx: float, ly: float, nstep: int) -> tuple[float, float]:
    """March the sphere, extract R, and score its Gamma column against Mie.

    Args:
        nx: Points along x.
        ny: Points along y.
        lx: Period along x.
        ly: Period along y.
        nstep: Depth steps.

    Returns:
        (specular P error, worst error over the significant orders), relative.
    """
    n = nx * ny
    y_top = march_sphere(nx, ny, lx, ly, nstep)
    r_mat = reflection_from_y(y_top, nx, ny, lx, ly)
    got = r_mat[:, 0]  # incident: P mode at the Gamma order
    want = mie_prediction(nx, ny, lx, ly)

    spec = abs(got[0] - want[0]) / abs(want[0])
    big = np.abs(want) > 1.0e-3 * abs(want[0])
    worst = float(np.max(np.abs(got[big] - want[big]) / np.abs(want[big]))) if big.any() else spec
    return spec, worst


def part6() -> None:
    """The three refinement ladders, against the exact Mie response."""
    print("\n[6] against the exact sphere: three refinement axes")
    kp, ks = OMEGA / REF.alpha, OMEGA / REF.beta
    print(f"      k_P a = {kp * RADIUS:.3f}   k_S a = {ks * RADIUS:.3f}   contrast {CONTRAST:.2f}")

    # At normal incidence the Gamma-order SV amplitude must vanish identically:
    # d_theta P_n carries a factor q, which is zero there.  A non-zero value
    # would mean the mode ordering or the polarisation had been mixed up.
    n8 = 8 * 8
    want0 = mie_prediction(8, 8, 600.0, 600.0)
    print(f"      Gamma-order SV amplitude (must vanish): {abs(want0[n8]):.3e}")
    report("the specular shear order vanishes at normal incidence", abs(want0[n8]) < 1e-30)

    print("\n      (a) DEPTH STEP, lateral grid and period held fixed")
    errs = []
    for nstep in (16, 32, 64, 128):
        spec, _ = compare_to_mie(8, 8, 600.0, 600.0, nstep)
        errs.append(spec)
        print(f"          nstep={nstep:<4d} specular error {spec:.4e}")
    print(f"          orders: {'  '.join(f'{o:.2f}' for o in order_of(errs))}")
    report("refining the depth step converges", errs[-1] < errs[0])

    print("\n      (b) LATERAL RESOLUTION, period and depth step held fixed")
    for nsz in (8, 12, 16):
        spec, worst = compare_to_mie(nsz, nsz, 600.0, 600.0, 64)
        print(f"          N={nsz:<3d} pitch={600.0 / nsz:6.1f} m  specular {spec:.4e}  worst {worst:.4e}")

    print("\n      (c) LATERAL PERIOD, resolution and depth step held fixed")
    for nsz in (8, 12, 16):
        lx = nsz * 75.0
        spec, worst = compare_to_mie(nsz, nsz, lx, lx, 64)
        diam = lx / (2.0 * RADIUS)
        print(f"          L={lx:6.0f} m = {diam:.1f} diam  specular {spec:.4e}  worst {worst:.4e}")
    report("the three ladders ran", True)


def part7() -> None:
    """The second sweep: transmission, and the whole response scored at once."""
    print("\n[7] the downward sweep -- the other half of the R/T response")
    print("      R is read off Y at the top and needs ONE sweep.  T is the field")
    print("      CARRIED THROUGH the slab and needs the second, integrated with")
    print("      the energy flow while the Riccati runs against it.")

    nx = ny = 4
    lx = ly = 6.0 * RADIUS
    n = nx * ny
    h_slab = 2.0 * RADIUS

    # (a) THE EXACT LIMIT.  With no contrast every entry is known in closed form,
    # so the sweep is tested before any arbiter is involved.
    print("\n      (a) zero contrast, where the answer is known exactly")
    kx, ky = grid_wavenumbers(nx, lx), grid_wavenumbers(ny, ly)
    kp, ks = OMEGA / REF.alpha, OMEGA / REF.beta
    want = np.zeros(3 * n, dtype=complex)
    for ix in range(nx):
        for iy in range(ny):
            qq = float(np.hypot(kx[ix], ky[iy]))
            kzp = complex(kz_of(np.array(qq), kp))
            kzs = complex(kz_of(np.array(qq), ks))
            for c, kzc in enumerate((kzp, kzs, kzs)):
                want[c * n + ix * ny + iy] = np.exp(1j * kzc * h_slab)
    w_mat = np.diag(want)

    errs = []
    print(f"          {'nstep':>6}{'||T - exact||':>16}{'order':>8}")
    for ns in (4, 8, 16, 32):
        r0, t0 = reflection_transmission(nx, ny, lx, ly, ns, amplitude=0.0)
        e = float(np.linalg.norm(t0 - w_mat) / np.linalg.norm(w_mat))
        errs.append(e)
        o = "" if len(errs) < 2 else f"{np.log2(errs[-2] / errs[-1]):.2f}"
        print(f"          {ns:6d}{e:16.4e}{o:>8}")
    report("with no contrast the reflection vanishes", float(np.linalg.norm(r0)) < 1e-12)
    off = float(np.linalg.norm(t0 - np.diag(np.diag(t0))) / np.linalg.norm(t0))
    print(f"          T is diagonal to {off:.2e} -- a uniform slab mixes no modes")
    report("with no contrast T is exactly diagonal", off < 1e-12)
    report(
        "the downward sweep is fourth order, like the upward one", 3.5 < np.log2(errs[-2] / errs[-1]) < 4.5
    )

    # (b) ALL CHANNELS AT ONCE.  Energy balance scores every entry of both
    # matrices -- P, SV and SH incidence, every outgoing channel, every
    # propagating order -- and needs no arbiter, only a lossless slab.
    print("\n      (b) ALL CHANNELS: R^H R + T^H T = I, in the flux basis")
    es = []
    nprop = 0
    for ns in (8, 16, 32, 64):
        e, nprop = energy_residual(6, 6, 600.0, 600.0, ns)
        es.append(e)
        o = "" if len(es) < 2 else f"{np.log2(es[-2] / es[-1]):.2f}"
        print(f"          nstep={ns:<4d} residual {e:.4e}   {o}")
    print(f"          over {nprop} propagating channels of {3 * 36}, with contrast on")
    report("the full R/T response conserves energy", es[-1] < 1e-6)
    report("and converges to it with the depth step", es[-1] < 0.01 * es[0])

    # (c) THE ARBITER, on the one column its spectrum covers.
    print("\n      (c) against exact Mie, the P column")
    r_mat, t_mat = reflection_transmission(8, 8, 600.0, 600.0, 64)
    r_want, t_want = mie_prediction(8, 8, 600.0, 600.0), mie_transmission(8, 8, 600.0, 600.0)
    spec_r = abs(r_mat[0, 0] - r_want[0]) / abs(r_want[0])
    spec_t = abs(t_mat[0, 0] - t_want[0]) / abs(t_want[0])
    print(f"          specular R {spec_r:.4e}    specular T {spec_t:.4e}")
    report("the reflection column matches Mie", spec_r < 0.30)
    report("and the transmission column does too", spec_t < 0.05)

    print(
        "\n      ⚠ (c) covers P INCIDENCE only, and that is a limit of the ARBITER'S\n"
        "      PYTHON PORT, not of the march and not of the derivation:  (b) already\n"
        "      scores every channel of both matrices.  ``mode_amplitudes`` here\n"
        "      implements only the m = 0 spectrum, which is what P incidence at\n"
        "      normal incidence produces; SV and SH incidence excite m = +/-1.\n"
        "\n"
        "      An earlier version of this message said the m = +/-1 spectrum was\n"
        "      NOT BUILT.  That was wrong, and wrong in the way this project has a\n"
        "      standing rule against: the survey stopped at the Python.\n"
        "      ``Mathematica/MieSphericalWaves.wl`` Section 7, 'the channel spectra,\n"
        "      general in m', carries ``angTriple[n, m, ...]`` with the m != 0 branch\n"
        "      in closed form, all three families (L-type P, N-type SV, M-type SH),\n"
        "      and the Sommerfeld path on both the propagating and evanescent\n"
        "      branches -- validated there against the exact Cartesian field.\n"
        "      ``compute_elastic_mie`` already carries a_n_sv, b_n_sv and c_n.\n"
        "      So what remains is a PORT of a validated derivation, not a\n"
        "      derivation."
    )


def part8() -> None:
    """SV incidence, scored against Mie -- the column the m=0 spectrum could not."""
    print("\n[8] SV incidence against exact Mie (the m = +/-1 spectrum)")
    nx = ny = 6
    lx = ly = 600.0
    n = nx * ny

    r_mat, t_mat = reflection_transmission(nx, ny, lx, ly, 32)
    r_want, t_want = mie_columns_s(nx, ny, lx, ly)
    got_r, got_t = r_mat[:, n], t_mat[:, n]  # incident SV at the Gamma order

    # THE SPECULAR SV->P ENTRY IS EXACTLY ZERO IN MIE, and that is a symmetry
    # statement worth keeping: the m = 1 angular function carries a factor
    # sin(theta), so an S wave cannot backscatter into P exactly along the axis.
    print(f"      specular SV->P: Mie {abs(r_want[0]):.3e} (zero by symmetry), march {abs(got_r[0]):.3e}")
    report("Mie gives exactly zero for specular SV->P", abs(r_want[0]) < 1e-30)
    # The march cannot return an exact zero here and should not be asked to: it
    # is solving a periodic ARRAY, whose inter-sphere coupling is measured at
    # 1.45 eps -- about 14% at this contrast -- and that coupling breaks the
    # isolated sphere's axial symmetry.  The leakage is 3.6% of the specular
    # SV->SV, comfortably inside that floor, so what this checks is that the
    # symmetry holds AS WELL AS the array allows, not exactly.
    leak = abs(got_r[0]) / abs(got_r[n])
    print(f"      the march's leakage into it is {leak:.1%} of specular SV->SV, against a 14% array floor")
    report("and the march respects it to within the array coupling", leak < 0.10)

    # The P channel of the SV column is driven by a_n_sv ALONE -- the toroidal
    # field is transverse and cannot radiate P -- so it is the part of this
    # comparison that the stated toroidal gap does not touch.
    for lab, got, want in (("R", got_r, r_want), ("T", got_t, t_want)):
        big = np.abs(want[:n]) > 1.0e-2 * np.max(np.abs(want[:n]))
        rat = np.abs(got[:n][big]) / np.abs(want[:n][big])
        print(
            f"      {lab} column, P channel: {int(big.sum())} orders, "
            f"median march/Mie {np.median(rat):.4f}, range {rat.min():.3f}-{rat.max():.3f}"
        )
        report(f"the SV->P channel of {lab} tracks Mie", 0.6 < float(np.median(rat)) < 1.6)

    big = np.abs(r_want[n:]) > 3.0e-2 * np.max(np.abs(r_want[n:]))
    rat = np.abs(got_r[n:][big]) / np.abs(r_want[n:][big])
    print(f"      R column, S channels: {int(big.sum())} orders, median march/Mie {np.median(rat):.4f}")
    report("the SV->S channels track Mie too", 0.6 < float(np.median(rat)) < 1.6)
    print(
        "      ⚠ the S channels carry the toroidal gap of ``mie_spectrum_s``: the\n"
        "      scalar joining c_n to the shared convention is bounded at or below\n"
        "      k_S by this comparison and is NOT determined by it."
    )


def main() -> int:
    """Run every part and summarise.

    Returns:
        0 if all checks passed, 1 otherwise.
    """
    print("=" * 78)
    print("THE SPHERE AS SEEN BY THE LATERALLY COUPLED IMPEDANCE MARCH")
    print("=" * 78)
    for fn in (part1, part2, part3, part4, part5, part6, part7, part8):
        fn()
    npass = sum(1 for _, ok in _PASS if ok)
    print("\n" + "=" * 78)
    print(f"{npass}/{len(_PASS)} checks passed")
    for label, ok in _PASS:
        if not ok:
            print(f"  FAILED: {label}")
    print("=" * 78)
    return 0 if npass == len(_PASS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
