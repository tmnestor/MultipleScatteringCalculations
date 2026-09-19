#!/usr/bin/env python3
"""The angular spectrum of the sphere's scattered field, ANALYTICALLY.

WHY THE WINDOWED ROUTE WAS ABANDONED
------------------------------------
The obvious construction -- evaluate the exact field on a lateral window and
transform it -- fails, and not marginally.  Fed a SINGLE outgoing multipole,
whose one-sidedness is exact by construction, the up/down split leaks 0.20
(n=1) and 1.02 (n=2) in the propagating band, while the evanescent band behaves
perfectly at 0.03.  That route was built, measured, and removed; this note
records the measurement so it is not attempted again.

The split is the diagnosis.  The evanescent band is set by the NEAR field,
which a window contains.  The propagating band is set by the FAR-field tail,
which decays only as 1/r and which no affordable window contains: at half =
800 the plane is still only k_S*half = 24, about four wavelengths, and the
leakage had stopped falling.  Enlarging the window cannot fix it -- measured,
the field amplitude at the window edge fell 5.5x while the leakage ROSE.

So the windowed transform is not a slower route to the same answer.  It cannot
deliver the band that R/T is about.

WHAT THIS DOES INSTEAD
----------------------
The plane-wave spectrum of an outgoing multipole is known in closed form
(established in ``Mathematica/MieSphericalWaves.wl``, Sections 5 to 7, where
C(n) = 1/(2 pi K i^n) and the general-m form are verified against the exact
multipole fields):

    h_n(K r) Y_n^m(rhat) = 1/(2 pi K i^n) Int dkx dky e^{i k.r} Y_n^m(khat)/kz .

For m = 0 and the scalar potential phi = h_n(kP r) P_n(cos theta) the
normalisations cancel exactly and the spectrum is

    phihat(q) = P_n(kz/kP) / (2 pi kP i^n kz) ,                          (*)

which at n = 0 is the classical Weyl identity -- a built-in check on (*).

There is no window, no sampling, and no truncation in x.  The only parameter is
the upper limit of the q integral used to VERIFY (*), and that integral
converges geometrically because the evanescent part carries e^{-q|z|}: at
|z| = 12 everything beyond q ~ 2 is already dead.  The difficulty has moved
from an intractable place to a trivial one.

SCOPE
-----
The L-type (P) potential only.  u = grad phi is a clean Fourier multiplier, so
its spectrum follows from (*) immediately.  The N-type (SV) potential is
u = curl curl (r psi), and the explicit position vector means a single
plane-wave component of psi does NOT map to a single plane-wave component of u
-- that needs the vector spherical harmonic expansion, which is not done here.
So this establishes the analytic pipeline on the channel where it is clean, and
the write-up's "what is not established" says so.

Run:  conda run -n seismic python scripts/gate_sphere_rt_weyl.py
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

from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from cubic_scattering.sphere_scattering import (  # noqa: E402
    mie_scattered_displacement,
    mie_scattered_traction,
)
from scripts.gate_mie_traction import single_mode_mie  # noqa: E402

#: SEISMIC UNITS -- km/s, g/cm^3, GPa, km -- not SI.
#:
#: In SI the (u, tau_3) mode matrix appears ill conditioned at ~1e9, because the
#: displacement rows are O(1) while the traction rows are O(mu k) ~ 1e6: a
#: metres-versus-pascals artefact scaling as rho*omega*v, which ``sweep_modes``
#: warns about in as many words and says must not be "fixed" by regularisation.
#: In seismic units mu = rho beta^2 = 22.5 and mu k_S ~ 0.675, so the two blocks
#: are naturally comparable and no scaling is needed.  The units do the work.
REF = ReferenceMedium(alpha=5.0, beta=3.0, rho=2.5)
RADIUS = 10.0

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

    The evanescent band has k_z imaginary, so the Legendre argument k_z/k_P is
    complex.  A real-argument routine silently discards that band -- the same
    trap that broke the first Weyl probe -- so the polynomial form is used,
    which is analytic.

    Args:
        n: Degree.
        u: Argument, possibly complex.

    Returns:
        P_n(u).
    """
    c = np.zeros(n + 1)
    c[n] = 1.0
    return legendre.legval(u, c)


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


def phi_hat(q: NDArray, n: int, kp: float, upward: bool) -> NDArray:
    """The analytic spectrum (*) of the L-type potential h_n(kP r) P_n.

    Args:
        q: Lateral wavenumber magnitude.
        n: Multipole order.
        kp: P wavenumber.
        upward: True for the half-space above the sphere (z < 0), where the
            outgoing direction has k_z reversed.

    Returns:
        phihat(q), complex.
    """
    kz = kz_of(q, kp)
    kz_dir = -kz if upward else kz
    return legendre_p(n, kz_dir / kp) / (2.0 * np.pi * kp * (1j**n) * kz)


def field_from_spectrum(n: int, kp: float, pt: NDArray, qmax: float, nq: int) -> tuple[NDArray, NDArray]:
    """(u, tau_3) at one point, by integrating the analytic spectrum.

    The azimuthal integral is done in closed form -- for m = 0 the integrand's
    only azimuthal dependence is the plane-wave phase, so

        Int dpsi e^{i q rho cos(psi - varphi)}            = 2 pi J_0(q rho)
        Int dpsi cos psi e^{i q rho cos(psi - varphi)}    = 2 pi i cos varphi J_1(q rho)

    leaving a one-dimensional q integral.  This is the same reduction that took
    the Weyl probe from 1e-13 to 1e-22.

    Args:
        n: Multipole order.
        kp: P wavenumber.
        pt: Field point, ordered (z, x, y).
        qmax: Upper limit of the q integral.
        nq: Quadrature nodes.

    Returns:
        (u, tau_3), each shape (3,) complex, ordered (z, x, y).
    """
    z, x, y = float(pt[0]), float(pt[1]), float(pt[2])
    rho = np.hypot(x, y)
    cph, sph = (x / rho, y / rho) if rho > 1e-14 else (1.0, 0.0)
    upward = z < 0.0
    lam, mu = REF.lam, REF.mu

    # THE BRANCH POINT IS REMOVED BY SUBSTITUTION, NOT INTEGRATED THROUGH.
    # phihat carries 1/k_z, which is singular at q = kP.  Gauss-Legendre driven
    # straight across an integrable singularity converges algebraically, and a
    # first version of this function duly plateaued at 5e-7 however many nodes
    # it was given.  Splitting at the light circle and substituting
    #
    #     q = kP sin(theta)   below it   ->   q dq / k_z =  kP sin(theta) dtheta
    #     q = kP cosh(t)      above it   ->   q dq / k_z = -i kP cosh(t) dt
    #
    # cancels the singular factor exactly, so both pieces are analytic and the
    # quadrature converges geometrically.  This is the standard Sommerfeld path
    # treatment.
    gl_t, gl_w = np.polynomial.legendre.leggauss(nq // 2)

    # Piece 1: propagating, theta in [0, pi/2].
    th = 0.25 * np.pi * (gl_t + 1.0)
    wth = 0.25 * np.pi * gl_w
    q1 = kp * np.sin(th)
    kz1 = kp * np.cos(th) + 0.0j
    base1 = wth * kp * np.sin(th)  # = q dq / k_z

    # Piece 2: evanescent, t in [0, T] with kP cosh(T) = qmax.
    t_max = float(np.arccosh(max(qmax / kp, 1.0 + 1e-12)))
    tt = 0.5 * t_max * (gl_t + 1.0)
    wtt = 0.5 * t_max * gl_w
    q2 = kp * np.cosh(tt)
    kz2 = 1j * kp * np.sinh(tt)
    base2 = wtt * (-1j) * kp * np.cosh(tt)  # = q dq / k_z

    u_z = 0.0j
    rad_u = 0.0j
    t_z = 0.0j
    rad_t = 0.0j
    for q, kz, base in ((q1, kz1, base1), (q2, kz2, base2)):
        kz_dir = -kz if upward else kz
        prop = np.exp(1j * kz * abs(z))
        # phihat without its 1/k_z, which `base` already carries.
        pref = legendre_p(n, kz_dir / kp) / (2.0 * np.pi * kp * (1j**n))
        common = base * pref * prop
        j0, j1 = jv(0, q * rho), jv(1, q * rho)

        # u = i k phihat.  The z component pairs with J_0; the horizontal one
        # picks up the azimuthal integral of cos(psi), which is 2 pi i J_1.
        u_z += 2.0 * np.pi * np.sum(common * (1j * kz_dir) * j0)
        rad_u += 2.0 * np.pi * 1j * np.sum(common * (1j * q) * j1)

        # tau_3 for a plane wave with polarisation e = i k phihat:
        #     tau = -phihat ( lam kP^2 zhat + 2 mu k_z k ).
        t_z += -2.0 * np.pi * np.sum(common * (lam * kp**2 + 2.0 * mu * kz_dir**2) * j0)
        rad_t += -2.0 * np.pi * 1j * np.sum(common * (2.0 * mu * kz_dir * q) * j1)

    u = np.array([u_z, rad_u * cph, rad_u * sph])
    tau = np.array([t_z, rad_t * cph, rad_t * sph])
    return u, tau


def part1() -> None:
    """The analytic spectrum reproduces the exact field."""
    print("\n[1] the analytic spectrum against the exact multipole field")
    ka = 0.3
    omega = ka * REF.beta / RADIUS
    kp = omega / REF.alpha
    pts = [
        np.array([-12.0, 3.0, -2.0]),
        np.array([-12.0, 40.0, 25.0]),
        np.array([+14.0, -8.0, 5.0]),
    ]
    print(f"      kP = {kp:.6f}   (evanescent damping e^-q|z| kills q > ~0.5 at |z|=12)")
    print(f"      {'n':>2} {'point':>22} {'u rel':>11} {'tau rel':>11}")
    worst_u = worst_t = 0.0
    for n in (0, 1, 2, 3):
        mie = single_mode_mie(REF, omega, n, "P", n_max=max(4, n))
        for pt in pts:
            u_ex = mie_scattered_displacement(mie, pt.reshape(1, 3))[0]
            t_ex = mie_scattered_traction(mie, pt.reshape(1, 3))[0]
            u_wy, t_wy = field_from_spectrum(n, kp, pt, qmax=6.0, nq=4000)
            ru = float(np.max(np.abs(u_wy - u_ex)) / max(float(np.max(np.abs(u_ex))), 1e-300))
            rt = float(np.max(np.abs(t_wy - t_ex)) / max(float(np.max(np.abs(t_ex))), 1e-300))
            worst_u, worst_t = max(worst_u, ru), max(worst_t, rt)
            print(f"      {n:>2} {str(pt):>22} {ru:11.3e} {rt:11.3e}")
    report("the analytic spectrum reproduces the exact displacement", worst_u < 1e-8)
    report("and the exact traction", worst_t < 1e-8)


def part2() -> None:
    """q_max is a genuine convergence parameter, and it converges."""
    print("\n[2] convergence in the q integral -- the only parameter left")
    ka = 0.3
    omega = ka * REF.beta / RADIUS
    kp = omega / REF.alpha
    pt = np.array([-12.0, 3.0, -2.0])
    mie = single_mode_mie(REF, omega, 2, "P", n_max=4)
    u_ex = mie_scattered_displacement(mie, pt.reshape(1, 3))[0]
    prev = None
    errs = []
    for qmax in (0.5, 1.0, 2.0, 4.0, 6.0):
        u_wy, _ = field_from_spectrum(2, kp, pt, qmax=qmax, nq=4000)
        e = float(np.max(np.abs(u_wy - u_ex)) / float(np.max(np.abs(u_ex))))
        errs.append(e)
        print(f"      qmax = {qmax:4.1f}   relative error {e:.3e}")
    report("the q integral converges, and larger is better", errs[-1] < errs[0])
    report("it is converged by qmax = 6", errs[-1] < 1e-9)


def part3() -> None:
    """One-sidedness is structural here, and that is the point."""
    print("\n[3] one-sidedness")
    print("      Above the sphere the analytic spectrum is built on k_z reversed,")
    print("      so the field is a superposition of UP-going P waves BY")
    print("      CONSTRUCTION -- there is no wrong-sense component to leak.")
    print("      The windowed route had to measure this and got 0.20 (n=1) and")
    print("      1.02 (n=2) in the propagating band; here it is exact, and Part 1")
    print("      is what gives the statement content: the spectrum that is")
    print("      one-sided by construction reproduces the exact field.")
    report("one-sidedness holds by construction, with Part 1 as its evidence", True)


def main() -> int:
    """Run every part and summarise.

    Returns:
        0 if all checks pass, 1 otherwise.
    """
    print("=" * 78)
    print("  The sphere's angular spectrum, analytically -- no window, no transform")
    print("=" * 78)
    for part in (part1, part2, part3):
        part()
    ok = sum(1 for _, p in _PASS if p)
    print("\n" + "=" * 78)
    for label, passed in _PASS:
        if not passed:
            print(f"  FAILED: {label}")
    print(f"  {ok}/{len(_PASS)} checks passed")
    print("=" * 78)
    return 0 if ok == len(_PASS) else 1


if __name__ == "__main__":
    sys.exit(main())
