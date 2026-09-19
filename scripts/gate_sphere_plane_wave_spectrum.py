#!/usr/bin/env python3
"""The sphere's scattered field as PLANE-WAVE MODE AMPLITUDES.

ANCHOR: ``LatexPDFs/SpherePlaneWaveResponse/SpherePlaneWaveResponse.tex`` §4,
verified symbolically in ``Mathematica/MieSphericalWaves.wl`` §5-§7.  This is
the Python twin of that result, assembled for a whole sphere rather than one
multipole, and scored against the committed ``mie_scattered_displacement``.

WHAT THIS PRODUCES, AND WHY IT IS NOT A SINGLE VECTOR SPECTRUM
--------------------------------------------------------------
At one lateral wavenumber ``k_perp`` the outgoing field is not one plane wave.
P and S have DIFFERENT vertical wavenumbers, ``k_z^P = sqrt(k_P^2 - q^2)`` and
``k_z^S = sqrt(k_S^2 - q^2)``, so they carry different depth dependence and
cannot be added into one vector before a depth is chosen.  The natural object is
therefore a pair of MODE amplitudes,

    u_scat(r) = Int dk_x dk_y [ c_P e_P exp(i(k.rho + k_z^P |z|))
                              + c_SV e_SV exp(i(k.rho + k_z^S |z|)) ]

which is also exactly the form the impedance march returns, so the two meet
without anything interpolated between them.

THE TWO SPECTRA, FROM THE BOXED RESULT
--------------------------------------
With ``D = 2 pi K i^n k_z`` and ``A`` the angular function at the OUTGOING
direction, the tex boxes give ``L: i k_P / D . A . e_P`` and
``N: i k_S / D . [d_theta A e_SV + (1/sin theta) d_phi A e_SH]``.  A plane P
wave incident along +z excites m = 0 only, so ``d_phi A`` vanishes, SH is not
driven, and with ``phi = sum a_n h_n(k_P r) P_n`` and
``psi = sum b_n h_n(k_S r) P_n`` -- which is the convention
``_mie_pwave_fields`` and ``_mie_swave_fields`` are written in -- this reduces to

    c_P  = (i / (2 pi k_z^P)) sum_n a_n P_n(k_z^Pdir / k_P) / i^n
    c_SV = (i / (2 pi k_z^S)) sum_n b_n [d_theta P_n] / i^n

``d_theta P_n = -(q/k_S) P_n'(k_z^Sdir / k_S)``, written through ``P_n'`` rather
than through ``lpmv`` because the evanescent band has ``k_z`` imaginary and the
associated-Legendre routines are real-argument only -- the trap that broke the
first Weyl probe.  ``(1 - x^2)^{1/2} = q/K`` exactly on either branch, so no
square root of a complex number is ever taken.

HOW IT IS VERIFIED
------------------
The spectra are integrated back to a field point and compared with
``mie_scattered_displacement``, which this file does not own.  Both 1/k_z poles
are removed by the Sommerfeld substitutions ``q = K sin theta`` below the light
circle and ``q = K cosh t`` above it, separately for each family because the two
poles sit at different q.  Driving Gauss-Legendre straight through either one
plateaus at about 5e-7 however many nodes it is given.

Run:  conda run -n seismic python scripts/gate_sphere_plane_wave_spectrum.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from numpy.polynomial import legendre
from numpy.typing import NDArray
from scipy.special import jv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubic_scattering.effective_contrasts import MaterialContrast, ReferenceMedium  # noqa: E402
from cubic_scattering.sphere_scattering import (  # noqa: E402
    MieResult,
    compute_elastic_mie,
    mie_scattered_displacement,
)

REF = ReferenceMedium(5000.0, 3000.0, 2500.0)
OMEGA = 60.0
RADIUS = 120.0
CONTRAST = MaterialContrast(Dlambda=2.0e9, Dmu=1.0e9, Drho=100.0)

_PASS: list[tuple[str, bool]] = []


def report(label: str, ok: bool) -> None:
    """Record and print one check.

    Args:
        label: Description.
        ok: Whether it passed.
    """
    _PASS.append((label, bool(ok)))
    print(f"  {'PASS' if ok else '****FAIL****'}  {label}")


def legendre_p(n: int, u: NDArray) -> NDArray:
    """P_n at a possibly COMPLEX argument.

    Args:
        n: Degree.
        u: Argument, possibly complex.

    Returns:
        P_n(u).
    """
    c = np.zeros(n + 1)
    c[n] = 1.0
    return legendre.legval(u, c)


def legendre_dp(n: int, u: NDArray) -> NDArray:
    """dP_n/dx at a possibly COMPLEX argument.

    Args:
        n: Degree.
        u: Argument, possibly complex.

    Returns:
        P_n'(u).
    """
    if n == 0:
        return np.zeros_like(np.asarray(u, dtype=complex))
    c = np.zeros(n + 1)
    c[n] = 1.0
    return legendre.legval(u, legendre.legder(c))


def kz_of(q: NDArray, k: float) -> NDArray:
    """Vertical wavenumber on the outgoing branch, Im >= 0.

    Args:
        q: Lateral wavenumber magnitude.
        k: Medium wavenumber.

    Returns:
        k_z, complex.
    """
    s = np.sqrt((k**2 - q**2).astype(complex))
    return np.where(s.imag < 0, -s, s)


def mode_amplitudes(
    mie: MieResult, q: NDArray, *, upward: bool, kz_p: NDArray | None = None, kz_s: NDArray | None = None
) -> tuple[NDArray, NDArray]:
    """The P and SV mode amplitudes of the scattered field at lateral wavenumber q.

    Args:
        mie: The Mie solution, whose a_n and b_n are the potential coefficients.
        q: Lateral wavenumber magnitude, any shape.
        upward: True for the half-space above the sphere, where the outgoing
            direction has k_z reversed.
        kz_p: Optional precomputed P vertical wavenumber, to let a caller supply
            the Sommerfeld-substituted values rather than recompute them.
        kz_s: Optional precomputed S vertical wavenumber.

    Returns:
        (c_P, c_SV), each the same shape as ``q``, complex.
    """
    kp = mie.omega / mie.ref.alpha
    ks = mie.omega / mie.ref.beta
    kzp = kz_of(q, kp) if kz_p is None else kz_p
    kzs = kz_of(q, ks) if kz_s is None else kz_s
    kzp_dir = -kzp if upward else kzp
    kzs_dir = -kzs if upward else kzs

    sum_p = np.zeros_like(np.asarray(q, dtype=complex))
    sum_s = np.zeros_like(np.asarray(q, dtype=complex))
    for n in range(mie.n_max + 1):
        inv_in = 1.0 / (1j**n)
        sum_p = sum_p + mie.a_n[n] * legendre_p(n, kzp_dir / kp) * inv_in
        if n >= 1:
            dtheta = -(q / ks) * legendre_dp(n, kzs_dir / ks)
            sum_s = sum_s + mie.b_n[n] * dtheta * inv_in
    return (1j / (2.0 * np.pi * kzp)) * sum_p, (1j / (2.0 * np.pi * kzs)) * sum_s


def _sommerfeld_nodes(k: float, qmax: float, nq: int) -> tuple[NDArray, NDArray, NDArray]:
    """Nodes for Int q dq / k_z, with the branch point removed by substitution.

    ``q = k sin(theta)`` below the light circle and ``q = k cosh(t)`` above it
    each cancel the ``1/k_z`` exactly, so both pieces are analytic.

    Args:
        k: Medium wavenumber, where the pole sits.
        qmax: Upper limit in q.
        nq: Total quadrature nodes, split between the two pieces.

    Returns:
        (q, k_z, weight) where ``weight`` already carries ``q dq / k_z``.
    """
    gl_t, gl_w = np.polynomial.legendre.leggauss(nq // 2)

    th = 0.25 * np.pi * (gl_t + 1.0)
    wth = 0.25 * np.pi * gl_w
    q1, kz1, w1 = k * np.sin(th), k * np.cos(th) + 0.0j, wth * k * np.sin(th)

    t_max = float(np.arccosh(max(qmax / k, 1.0 + 1e-12)))
    tt = 0.5 * t_max * (gl_t + 1.0)
    wtt = 0.5 * t_max * gl_w
    q2, kz2, w2 = k * np.cosh(tt), 1j * k * np.sinh(tt), wtt * (-1j) * k * np.cosh(tt)

    return (
        np.concatenate([q1, q2]),
        np.concatenate([kz1, kz2]),
        np.concatenate([w1, w2]).astype(complex),
    )


def field_from_modes(mie: MieResult, pt: NDArray, qmax: float, nq: int) -> NDArray:
    """The scattered displacement at one point, by integrating the mode amplitudes.

    Each family is integrated on ITS OWN Sommerfeld path, because the two poles
    sit at different q and a path built for one does not remove the other.

    The azimuthal integral is closed form at m = 0: the z components pair with
    ``2 pi J_0`` and the horizontal ones with ``2 pi i J_1``.  Dropping that
    ``i`` is a real trap -- it leaves the transverse traction wrong by tens of
    per cent while every magnitude still looks plausible.

    Args:
        mie: The Mie solution.
        pt: Field point, ordered (z, x, y), outside the sphere.
        qmax: Upper limit of the q integrals.
        nq: Quadrature nodes per family.

    Returns:
        Shape (3,) complex, ordered (z, x, y).
    """
    z, x, y = float(pt[0]), float(pt[1]), float(pt[2])
    rho = np.hypot(x, y)
    cph, sph = (x / rho, y / rho) if rho > 1e-14 else (1.0, 0.0)
    upward = z < 0.0
    kp, ks = mie.omega / mie.ref.alpha, mie.omega / mie.ref.beta

    u_z = 0.0j
    u_h = 0.0j

    # The P family, on the k_P path.
    q, kz, w = _sommerfeld_nodes(kp, qmax, nq)
    c_p, _ = mode_amplitudes(mie, q, upward=upward, kz_p=kz)
    kz_dir = -kz if upward else kz
    prop = np.exp(1j * kz * abs(z))
    # c_P carries 1/k_z, which the weight also carries; divide it back out once.
    core = w * (c_p * kz) * prop
    u_z += 2.0 * np.pi * np.sum(core * (kz_dir / kp) * jv(0, q * rho))
    u_h += 2.0 * np.pi * 1j * np.sum(core * (q / kp) * jv(1, q * rho))

    # The SV family, on the k_S path.
    q, kz, w = _sommerfeld_nodes(ks, qmax, nq)
    _, c_sv = mode_amplitudes(mie, q, upward=upward, kz_s=kz)
    kz_dir = -kz if upward else kz
    prop = np.exp(1j * kz * abs(z))
    core = w * (c_sv * kz) * prop
    # e_SV = e_theta = (-sin theta, cos theta cos phi, cos theta sin phi).
    u_z += 2.0 * np.pi * np.sum(core * (-q / ks) * jv(0, q * rho))
    u_h += 2.0 * np.pi * 1j * np.sum(core * (kz_dir / ks) * jv(1, q * rho))

    return np.array([u_z, u_h * cph, u_h * sph])


def rel(a: NDArray, b: NDArray) -> float:
    """Relative difference in the max norm.

    Args:
        a: First.
        b: Second.

    Returns:
        max|a-b| / max|b|.
    """
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-300))


def part1() -> None:
    """The assembled spectrum reproduces the committed Mie field."""
    print("\n[1] spectrum integrated back, against mie_scattered_displacement")
    mie = compute_elastic_mie(OMEGA, RADIUS, REF, CONTRAST)
    kp, ks = OMEGA / REF.alpha, OMEGA / REF.beta
    print(f"      n_max={mie.n_max}  k_P a={kp * RADIUS:.3f}  k_S a={ks * RADIUS:.3f}")

    pts = np.array(
        [
            [-400.0, 0.0, 0.0],
            [-400.0, 150.0, 0.0],
            [-400.0, -90.0, 130.0],
            [-700.0, 220.0, -160.0],
            [500.0, 120.0, 80.0],
        ]
    )
    want = mie_scattered_displacement(mie, pts)
    got = np.array([field_from_modes(mie, p, 8.0 * ks, 400) for p in pts])
    for i, p in enumerate(pts):
        side = "above" if p[0] < 0 else "below"
        print(f"      ({p[0]:6.0f},{p[1]:6.0f},{p[2]:6.0f}) {side:<5}  rel {rel(got[i], want[i]):.3e}")
    worst = max(rel(got[i], want[i]) for i in range(len(pts)))
    report("the mode amplitudes reproduce the Mie field", worst < 1e-9)


def part2() -> None:
    """The q integral is converged, not merely stable."""
    print("\n[2] quadrature convergence")
    mie = compute_elastic_mie(OMEGA, RADIUS, REF, CONTRAST)
    ks = OMEGA / REF.beta
    pt = np.array([-400.0, 150.0, -80.0])
    ref_u = field_from_modes(mie, pt, 12.0 * ks, 800)
    for nq in (100, 200, 400):
        err = rel(field_from_modes(mie, pt, 8.0 * ks, nq), ref_u)
        print(f"      nq={nq:<4d} rel to the fine quadrature: {err:.3e}")
    base_err = rel(field_from_modes(mie, pt, 8.0 * ks, 400), ref_u)
    report("the quadrature is converged", base_err < 1e-10)

    for qm in (4.0, 8.0, 16.0):
        err = rel(field_from_modes(mie, pt, qm * ks, 400), ref_u)
        print(f"      q_max={qm:<5.1f} k_S  rel: {err:.3e}")
    report("the evanescent cutoff is converged", base_err < 1e-10)


def main() -> int:
    """Run every part and summarise.

    Returns:
        0 if all checks passed, 1 otherwise.
    """
    print("=" * 78)
    print("THE SPHERE'S SCATTERED FIELD AS PLANE-WAVE MODE AMPLITUDES")
    print("=" * 78)
    for fn in (part1, part2):
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
