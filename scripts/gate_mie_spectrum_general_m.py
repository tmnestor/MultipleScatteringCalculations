#!/usr/bin/env python3
"""The Mie plane-wave spectrum for GENERAL m, ported from the Mathematica.

WHY THIS EXISTS
---------------
``scripts/gate_sphere_plane_wave_spectrum.py`` implements the plane-wave
spectrum of a Mie multipole for ``m = 0`` only.  That is what a P wave at
normal incidence excites, so it is enough to score the P column of a
reflection/transmission response and nothing else.  SV and SH incidence excite
``m = +/-1``, and scoring those columns needs the general-m spectrum.

⚠ THE DERIVATION IS NOT NEW AND WAS NOT MISSING.  An earlier note in this
project said the m = +/-1 spectrum "is NOT built".  That was wrong, and wrong in
the way the survey rule exists to prevent: the survey stopped at the Python.
``Mathematica/MieSphericalWaves.wl`` Section 7, "the channel spectra, general in
m", carries it in closed form -- the angular triple for m != 0, all three
families, and the Sommerfeld path on both the propagating and the evanescent
branch -- validated there against the exact Cartesian field.  This file is a
PORT of that, and the gate below is the cross-check between the two
implementations, which is the evidence standard this project works to.

WHAT IS PORTED
--------------
For a multipole of degree n and order m, with the scalar potential

    phi = h_n(K r) P_n^m(cos theta) cos(m phi)

the three elastic families are the L-type ``u = grad phi`` (P), the N-type
``u = curl curl (r phi)`` (SV) and the M-type ``u = curl (r phi)`` (SH).  Each
has a plane-wave (Weyl) representation whose angular content is the triple

    {A, dA/dtheta, (1/sin theta) dA/dphi}

evaluated AT THE PLANE-WAVE DIRECTION.  For m = 0 the third entry vanishes and
the spectrum is the familiar P_n / dP_n pair; for m != 0 it does not, and the
consequence is physical: an N-type multipole radiates BOTH SV and SH polarised
plane waves, and so does an M-type.  That azimuthal mixing is the whole reason
the m = 0 routine cannot be reused.

THE ARBITER
-----------
``Mathematica/MieSphericalWaves_reference.json`` carries the EXACT field of
every multipole -- 3 families x 3 degrees x 2 orders x 4 points = 72 cases --
computed from the Cartesian construction at 30 digits with exact rational
parameters.  The superposition built here must reproduce it.  Nothing in this
file is scored against anything written in this file.

Run:
    conda run -n seismic python scripts/gate_mie_spectrum_general_m.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.gate_sphere_plane_wave_spectrum import (  # noqa: E402
    legendre_dp,
    legendre_p,
)

REFERENCE = ROOT / "Mathematica" / "MieSphericalWaves_reference.json"

_PASS: list[tuple[str, bool]] = []


def report(label: str, ok: bool) -> None:
    """Record and print one check.

    Args:
        label: What was checked.
        ok: Whether it passed.
    """
    _PASS.append((label, ok))
    print(f"  {'PASS' if ok else '****FAIL****'}  {label}")


def legendre_ddp(n: int, u: NDArray) -> NDArray:
    """d2P_n/dx2 at a possibly complex argument.

    The second derivative is needed only for m != 0: the theta derivative of
    ``P_n^1(cos theta) = -sin theta P_n'(cos theta)`` carries it.

    Args:
        n: Degree.
        u: Argument, possibly complex.

    Returns:
        P_n''(u).
    """
    if n < 2:
        return np.zeros_like(np.asarray(u, dtype=complex))
    c = np.zeros(n + 1)
    c[n] = 1.0
    from numpy.polynomial import legendre

    return legendre.legval(u, legendre.legder(c, 2))


def ang_triple(
    n: int, m: int, kx: NDArray, ky: NDArray, kz_dir: NDArray, k: float
) -> tuple[NDArray, NDArray, NDArray]:
    """``{A, dA/dtheta, (1/sin theta) dA/dphi}`` at the plane-wave direction.

    A port of ``angTriple`` in Section 7 of ``MieSphericalWaves.wl``.  With
    ``A = P_n^m(cos theta) cos(m phi)``, ``u = cos theta = kz/K`` and
    ``s = sin theta = q/K``, the m = 1 entries follow from the Condon--Shortley
    relation ``P_n^1(u) = -sqrt(1-u^2) P_n'(u) = -s P_n'(u)``:

        A            = -s P_n'(u) cos psi
        dA/dtheta    = (-u P_n'(u) + s^2 P_n''(u)) cos psi
        (1/sin) dA/dphi = P_n'(u) sin psi

    Args:
        n: Degree.
        m: Order, 0 or 1.
        kx: Wavenumber component, x.
        ky: Wavenumber component, y.
        kz_dir: Vertical wavenumber WITH the propagation direction's sign.
        k: Medium wavenumber.

    Returns:
        The three angular functions, each broadcast to the shape of ``kx``.
    """
    u = kz_dir / k
    q = np.sqrt(np.asarray(kx, dtype=complex) ** 2 + np.asarray(ky) ** 2)
    q = np.where(np.abs(q) < 1e-300, 1e-300, q)
    s = q / k
    cps, sps = kx / q, ky / q
    p1 = legendre_dp(n, u)
    if m == 0:
        return legendre_p(n, u), -s * p1, np.zeros_like(p1)
    p2 = legendre_ddp(n, u)
    return -s * p1 * cps, (-u * p1 + s * s * p2) * cps, p1 * sps


def polarisations(kx: NDArray, ky: NDArray, kz_dir: NDArray, k: float) -> tuple[NDArray, NDArray, NDArray]:
    """The three unit polarisations at one plane-wave direction, Cartesian (x,y,z).

    Args:
        kx: Wavenumber component, x.
        ky: Wavenumber component, y.
        kz_dir: Vertical wavenumber with the direction's sign.
        k: Medium wavenumber.

    Returns:
        (e_P, e_SV, e_SH), each of shape (..., 3).
    """
    q = np.sqrt(np.asarray(kx, dtype=complex) ** 2 + np.asarray(ky) ** 2)
    q = np.where(np.abs(q) < 1e-300, 1e-300, q)
    e_p = np.stack([kx / k, ky / k, kz_dir / k], axis=-1)
    e_sv = np.stack([kz_dir * kx / q / k, kz_dir * ky / q / k, -q / k], axis=-1)
    e_sh = np.stack([-ky / q, kx / q, np.zeros_like(q)], axis=-1)
    return e_p, e_sv, e_sh


def sommerfeld_pieces(
    k: float, nq: int, *, z_abs: float = 1.0, decades: float = 30.0
) -> list[tuple[NDArray, NDArray, NDArray]]:
    """The two branches of the Weyl integral, each with the 1/k_z cancelled.

    ``q = K sin theta`` on the propagating branch and ``q = K cosh t`` on the
    evanescent one; each substitution removes the branch point exactly, so both
    integrands are analytic and Gauss--Legendre is appropriate.

    ⚠ NOT a Chebyshev rule.  The Mathematica notes that a first version used
    Chebyshev nodes reweighted by sqrt(1-x^2) -- the natural rule for a
    1/sqrt weight and a poor one for the smooth integrand the substitution
    leaves -- and it held the same check at 3e-4.

    ⚠ THE EVANESCENT CUTOFF MUST FOLLOW THE OBSERVATION HEIGHT.  The Mathematica
    uses a fixed ``q_max = 6``, which is tuned to its own test points.  The
    evanescent integrand decays as ``exp(-q |z|)``, so a fixed cutoff is a
    fixed number of e-foldings only at one height: at ``|z| = 1.1`` it leaves
    ``exp(-6.6) ~ 1.4e-3`` of the tail behind, and the port duly reproduced the
    reference to 1.02 at exactly that point and to 1.0000 everywhere else.
    Scaling the cutoff with ``1/|z|`` removes it.

    Args:
        k: Medium wavenumber.
        nq: Nodes per branch.
        z_abs: Height of the observation point above the source plane.
        decades: Required e-foldings of evanescent decay at the cutoff.

    Returns:
        A list of (q, k_z, base weight) for each branch.
    """
    gl_t, gl_w = np.polynomial.legendre.leggauss(nq)

    th = 0.25 * np.pi * (gl_t + 1.0)
    wth = 0.25 * np.pi * gl_w
    prop = (k * np.sin(th), k * np.cos(th) + 0.0j, wth * k * np.sin(th) + 0.0j)

    q_max = max(6.0, decades / max(z_abs, 1.0e-3))
    t_max = float(np.arccosh(max(q_max / k, 1.0 + 1e-12)))
    tt = 0.5 * t_max * (gl_t + 1.0)
    wtt = 0.5 * t_max * gl_w
    evan = (k * np.cosh(tt), 1j * k * np.sinh(tt), wtt * (-1j) * k * np.cosh(tt))
    return [prop, evan]


def superpose(
    family: str, n: int, m: int, k: float, point: NDArray, *, nq: int = 400, npsi: int = 64
) -> NDArray:
    """The plane-wave superposition of one multipole, at one observation point.

    A port of ``superpose`` in Section 7 of ``MieSphericalWaves.wl``.  The
    family combinations are the content of Sections 5 and 6:

        P  (L-type):   i K A e_P
        SV (N-type):   i K (dtheta e_SV + dphi e_SH)
        SH (M-type):   -(-dphi e_SV + dtheta e_SH)

    ⚠ The M-type carries neither the ``i K`` of the other two nor their sign;
    that relative normalisation is the one thing here a dimensional argument
    cannot supply, and it is what the reference check pins down.

    Args:
        family: "P", "SV" or "SH".
        n: Degree.
        m: Order, 0 or 1.
        k: Medium wavenumber (k_P for P, k_S otherwise).
        point: Observation point, Cartesian (x, y, z).
        nq: Quadrature nodes per Sommerfeld branch.
        npsi: Azimuthal nodes; the integrand is smooth and periodic in psi, so a
            uniform rule is spectrally accurate.

    Returns:
        The displacement, shape (3,), Cartesian (x, y, z).
    """
    x, y, z = (float(v) for v in point)
    up = z < 0.0
    psi = 2.0 * np.pi * np.arange(npsi) / npsi
    w_psi = 2.0 * np.pi / npsi

    total = np.zeros(3, dtype=complex)
    for q_1d, kz_1d, base_1d in sommerfeld_pieces(k, nq, z_abs=abs(z)):
        kz_dir_1d = -kz_1d if up else kz_1d
        kx = q_1d[:, None] * np.cos(psi)[None, :]
        ky = q_1d[:, None] * np.sin(psi)[None, :]
        kz = np.broadcast_to(kz_1d[:, None], kx.shape)
        kz_dir = np.broadcast_to(kz_dir_1d[:, None], kx.shape)
        base = np.broadcast_to(base_1d[:, None], kx.shape)

        a_ang, dth, dph = ang_triple(n, m, kx, ky, kz_dir, k)
        e_p, e_sv, e_sh = polarisations(kx, ky, kz_dir, k)
        common = base * w_psi / (2.0 * np.pi * k * (1j**n)) * np.exp(1j * (kx * x + ky * y + kz * abs(z)))

        if family == "P":
            vec = (1j * k * common * a_ang)[..., None] * e_p
        elif family == "SV":
            vec = (1j * k * common)[..., None] * (dth[..., None] * e_sv + dph[..., None] * e_sh)
        else:
            vec = (-common)[..., None] * (-dph[..., None] * e_sv + dth[..., None] * e_sh)
        total = total + vec.sum(axis=(0, 1))
    return total


def load_reference() -> tuple[dict, list[dict]]:
    """Read the Mathematica reference dump.

    Returns:
        (header fields, the 72 cases).
    """
    with REFERENCE.open() as fh:
        data = json.load(fh)
    return data, data["cases"]


def part1() -> None:
    """The ported spectrum against the exact multipole field."""
    print("\n[1] the general-m spectrum against the exact Mathematica field")
    head, cases = load_reference()
    kp, ks = float(head["kP"]), float(head["kS"])
    print(f"      k_P = {kp:.6f}   k_S = {ks:.6f}   {len(cases)} cases")

    worst_by: dict[tuple[str, int], float] = {}
    for case in cases:
        fam, n, m = case["family"], int(case["n"]), int(case["m"])
        k = kp if fam == "P" else ks
        point = np.array([float(v) for v in case["point"]])
        want = np.array([complex(re, im) for re, im in case["u"]])
        got = superpose(fam, n, m, k, point)
        rel = float(np.max(np.abs(got - want)) / np.max(np.abs(want)))
        key = (fam, m)
        worst_by[key] = max(worst_by.get(key, 0.0), rel)

    print(f"      {'family':>8}{'m':>4}{'worst over n and point':>26}")
    for (fam, m), val in sorted(worst_by.items()):
        print(f"      {fam:>8}{m:>4}{val:>26.3e}")

    worst = max(worst_by.values())
    print(f"      worst over all {len(cases)} cases: {worst:.3e}")
    report(
        "the m = 0 spectrum reproduces the exact field",
        max(v for (_f, m), v in worst_by.items() if m == 0) < 1e-4,
    )
    report(
        "and so does m = 1, which is the part that was missing",
        max(v for (_f, m), v in worst_by.items() if m == 1) < 1e-4,
    )
    report("every family works, including the M-type", worst < 1e-4)


def part2() -> None:
    """The azimuthal mixing that makes m = 0 unusable for S incidence."""
    print("\n[2] at m = 1 an N-type multipole radiates SH as well as SV")
    head, _cases = load_reference()
    ks = float(head["kS"])
    kx = np.array([0.3 * ks])
    ky = np.array([0.2 * ks])
    kzd = np.array([np.sqrt(complex(ks**2 - 0.13 * ks**2))])

    for m in (0, 1):
        _a, dth, dph = ang_triple(2, m, kx, ky, kzd, ks)
        print(f"      m={m}: |dtheta| = {abs(dth[0]):.4e}   |dphi| = {abs(dph[0]):.4e}")
        if m == 0:
            report("at m = 0 the SH channel of an N-type multipole is empty", abs(dph[0]) < 1e-30)
        else:
            report("at m = 1 it is not -- this is why m = 0 cannot be reused", abs(dph[0]) > 1e-3)

    # The Condon-Shortley identity the m=1 entries are built on, checked rather
    # than trusted: P_n^1(u) = -sqrt(1-u^2) P_n'(u), so A = -s P_n' cos psi.
    u = np.array([0.37])
    for n in (1, 2, 3):
        lhs = -np.sqrt(1.0 - u**2) * legendre_dp(n, u)
        from scipy.special import lpmv

        rhs = lpmv(1, n, u)
        ok = bool(np.allclose(lhs, rhs, rtol=1e-12))
        report(f"P_{n}^1 = -sqrt(1-u^2) P_{n}' (Condon-Shortley)", ok)


def main() -> int:
    """Run every part and summarise.

    Returns:
        0 if all checks passed, 1 otherwise.
    """
    print("=" * 78)
    print("THE MIE PLANE-WAVE SPECTRUM, GENERAL IN m")
    print("=" * 78)
    for fn in (part1, part2):
        fn()
    npass = sum(1 for _, ok in _PASS if ok)
    print("\n" + "=" * 78)
    for label, ok in _PASS:
        if not ok:
            print(f"  FAILED: {label}")
    print(f"  {npass}/{len(_PASS)} checks passed")
    print("=" * 78)
    return 0 if npass == len(_PASS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
