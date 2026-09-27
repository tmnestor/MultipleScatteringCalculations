#!/usr/bin/env python3
"""Gate: the single-site moment formulation, applied to a SPHERE, against exact elastic Mie.

The continuum-limit paper (section 6) derives the single-site T-matrix from moments of the elastodynamic
Green's tensor over the scatterer. It evaluates them as distributions: every derivative is moved onto the
boundary by the divergence theorem, so the r = 0 delta term is carried and never added by hand. The
first-gradient truncation of the resulting hierarchy is the uniform-strain closure. Its claims are:

  * the closure is EXACT for a sphere in the static limit (Eshelby's uniformity theorem);
  * the cube's dilatational channel equals the sphere's;
  * the isotropic average (3 S_off + 2 S_diag)/5 of the cube's shear depolarisations equals the sphere's.

Until now they had been checked against the textbook Eshelby formula, never against the scattering
solution. Here the arbiter is the exact Mie solve (``sphere_scattering.compute_elastic_mie``). The
concentration of channel n is E_n = a_n / a_n^Born. a_n^Born is a central difference in the contrast
(O(eps^2) = 1e-10). The static value is a Richardson extrapolation from k_S a = 0.01, 0.02 (O(w^4)).

The formulation's side shares nothing with the Mie solve. The ball's moment
M_in,pk = Int_V d_p d_k G_in dV is evaluated by the paper's route, the surface integral
Oint n_p d_k G_in dA over |r| = a, with this script's own closed-form derivatives of the Kupradze tensor
(checked against finite differences in [0]).

  [0] the Green's-tensor derivatives here against central finite differences;
  [1] static: the ball's dilatational channel == Mie E_0;
  [2] static: the ball's shear channel == Mie E_2;
  [3] static: the density channel == Mie E_1 == 1;
  [4] static: the cube's dilatational channel (closed form) == Mie E_0;
  [5] static: the cube's isotropic shear average (3 S_off + 2 S_diag)/5 == the depolarisation from Mie E_2;
  [C] control: the same moment by a SPHERICAL EXCISION (PV integral, no delta) must disagree with Mie.
  [D] dynamic (measured, not gated): the closure's amplification with the dynamic ball moments against
      Mie's E_n(w), with the order of the departure.
  [E] the departure's Born-order (ka)^2 coefficient, extracted by Richardson in the contrast and in ka,
      against the closed forms of Mathematica/SphereClosureDynamic.wl:
        n=0  -(k_P a)^2 Dlam / (10 (lam + 2 mu))
        n=1  (2 (k_S a)^2 + (k_P a)^2) Drho / (30 rho)
        n=2  -(3 (k_S a)^2 + 2 (k_P a)^2 beta^2/alpha^2) Dmu / (75 mu)

Background and contrasts are the validated ones (alpha = 5, beta = 3, rho = 2.5; dlam = 2, dmu = 1,
drho = 0.1). The static channel checks use each contrast alone, so that each Mie order sees one channel.

Run:  conda run -n seismic python scripts/gate_sphere_closure_vs_mie.py
"""

import sys

import numpy as np

sys.path.insert(0, "/Users/tod/Desktop/MultipleScatteringCalculations")
from cubic_scattering.effective_contrasts import MaterialContrast, ReferenceMedium  # noqa: E402
from cubic_scattering.sphere_scattering import compute_elastic_mie  # noqa: E402

ALPHA, BETA, RHO = 5.0, 3.0, 2.5
MU = RHO * BETA**2
LAM = RHO * ALPHA**2 - 2 * MU
DLAM, DMU, DRHO = 2.0, 1.0, 0.1
RADIUS = 1.0
REF = ReferenceMedium(alpha=ALPHA, beta=BETA, rho=RHO)
EPS = 1e-4
TOL_STATIC = 1e-6
SQ3 = np.sqrt(3.0)
EYE = np.eye(3)


# ---------------------------------------------------------------------------------------------------
# the Kupradze tensor's first derivative, closed form
# ---------------------------------------------------------------------------------------------------
def radial_derivs(k: complex, r: float) -> tuple[complex, complex, complex]:
    """f', f'', f''' of f(r) = exp(i k r)/r."""
    e = np.exp(1j * k * r)
    d1 = e * (1j * k / r - 1 / r**2)
    d2 = e * (-(k**2) / r - 2j * k / r**2 + 2 / r**3)
    d3 = e * (-1j * k**3 / r + 3 * k**2 / r**2 + 6j * k / r**3 - 6 / r**4)
    return d1, d2, d3


def dgreen(x: np.ndarray, omega: float) -> np.ndarray:
    """d_k G_in(x) as array [i, n, k], dynamic, G = (1/4 pi mu)[delta f_S + k_S^-2 dd (f_S - f_P)]."""
    kp, ks = omega / ALPHA, omega / BETA
    r = np.linalg.norm(x)
    xh = x / r
    fs1, _, _ = radial_derivs(ks, r)
    s1, s2, s3 = radial_derivs(ks, r)
    p1, p2, p3 = radial_derivs(kp, r)
    h1, h2, h3 = s1 - p1, s2 - p2, s3 - p3
    a, b = h2 - h1 / r, h1 / r
    da, db = h3 - h2 / r + h1 / r**2, h2 / r - h1 / r**2
    t = (
        da * np.einsum("k,i,n->ink", xh, xh, xh)
        + (a / r)
        * (
            np.einsum("ik,n->ink", EYE, xh)
            + np.einsum("nk,i->ink", EYE, xh)
            - 2 * np.einsum("i,n,k->ink", xh, xh, xh)
        )
        + db * np.einsum("k,in->ink", xh, EYE)
    )
    return (ks**2 * np.einsum("in,k->ink", EYE, fs1 * xh) + t) / (4 * np.pi * MU * ks**2)


def dgreen_static(x: np.ndarray) -> np.ndarray:
    """d_k G_in(x) of the Kelvin tensor a0 delta/r + b0 x x / r^3."""
    a0 = (1 / MU + 1 / (LAM + 2 * MU)) / (8 * np.pi)
    b0 = (1 / MU - 1 / (LAM + 2 * MU)) / (8 * np.pi)
    r = np.linalg.norm(x)
    return (
        -a0 * np.einsum("in,k->ink", EYE, x) / r**3
        + b0 * (np.einsum("ik,n->ink", EYE, x) + np.einsum("nk,i->ink", EYE, x)) / r**3
        - 3 * b0 * np.einsum("i,n,k->ink", x, x, x) / r**5
    )


def kelvin(x: np.ndarray) -> np.ndarray:
    """The static Kelvin tensor, for the finite-difference check only."""
    a0 = (1 / MU + 1 / (LAM + 2 * MU)) / (8 * np.pi)
    b0 = (1 / MU - 1 / (LAM + 2 * MU)) / (8 * np.pi)
    r = np.linalg.norm(x)
    return a0 * EYE / r + b0 * np.outer(x, x) / r**3


def green(x: np.ndarray, omega: float) -> np.ndarray:
    """G_in(x) itself, for the finite-difference check only."""
    kp, ks = omega / ALPHA, omega / BETA
    r = np.linalg.norm(x)
    xh = x / r
    fs = np.exp(1j * ks * r) / r
    s1, s2, _ = radial_derivs(ks, r)
    p1, p2, _ = radial_derivs(kp, r)
    h1, h2 = s1 - p1, s2 - p2
    dd = (h2 - h1 / r) * np.outer(xh, xh) + (h1 / r) * EYE
    return (ks**2 * fs * EYE + dd) / (4 * np.pi * MU * ks**2)


# ---------------------------------------------------------------------------------------------------
# the ball's moments by the surface route
# ---------------------------------------------------------------------------------------------------
NT, NP = 24, 48
_ct, _wt = np.polynomial.legendre.leggauss(NT)
_ph = 2 * np.pi * np.arange(NP) / NP
NODES = [
    (np.array([c, np.sqrt(1 - c * c) * np.cos(p), np.sqrt(1 - c * c) * np.sin(p)]), w * 2 * np.pi / NP)
    for c, w in zip(_ct, _wt, strict=True)
    for p in _ph
]


def surface_moment(dg, radius: float) -> np.ndarray:
    """M_in,pk = Oint_{|r| = radius} n_p d_k G_in dA, as array [i, n, p, k]."""
    m = np.zeros((3, 3, 3, 3), dtype=complex)
    for nh, w in NODES:
        m += w * radius**2 * np.einsum("p,ink->inpk", nh, dg(radius * nh))
    return m


def ball_displacement_moment(omega: float) -> complex:
    """g with Int_ball G_ij dV = g delta_ij: tr G = 2 f_S/(4 pi mu) + f_P/(4 pi (lam + 2 mu))."""

    def radial(k: float) -> complex:  # Int_0^a 4 pi r^2 exp(ikr)/r dr
        return 4 * np.pi * (np.exp(1j * k * RADIUS) * (RADIUS / (1j * k) + 1 / k**2) - 1 / k**2)

    ks, kp = omega / BETA, omega / ALPHA
    return (2 * radial(ks) / (4 * np.pi * MU) + radial(kp) / (4 * np.pi * (LAM + 2 * MU))) / 3


def channels(m: np.ndarray, dlam: float, dmu: float) -> tuple[complex, complex, complex]:
    """Eigenvalues (A1g, Eg, T2g) of the first-gradient block I - M . dc, as in CubeA22Block.wl."""
    dc = dlam * np.einsum("ij,kl->ijkl", EYE, EYE) + dmu * (
        np.einsum("ik,jl->ijkl", EYE, EYE) + np.einsum("il,jk->ijkl", EYE, EYE)
    )
    a22 = np.einsum("pr,ij->ipjr", EYE, EYE) - np.einsum("inpk,nkrj->ipjr", m, dc)
    a22 = a22.reshape(9, 9)
    v_a1g = (EYE / SQ3).reshape(9)
    v_eg = (np.diag([1.0, -1.0, 0.0]) / np.sqrt(2)).reshape(9)
    v_t2g = ((np.outer(EYE[0], EYE[1]) + np.outer(EYE[1], EYE[0])) / np.sqrt(2)).reshape(9)
    return tuple(v @ a22 @ v for v in (v_a1g, v_eg, v_t2g))


# ---------------------------------------------------------------------------------------------------
# Mie concentrations
# ---------------------------------------------------------------------------------------------------
def mie_concentration(w: float, dlam: float, dmu: float, drho: float, n: int) -> complex:
    """E_n(w) = a_n / a_n^Born at k_S a = w; a_n^Born by a central difference in the contrast."""
    omega = w * BETA / RADIUS

    def a_n(scale: float) -> complex:
        c = MaterialContrast(Dlambda=dlam * scale, Dmu=dmu * scale, Drho=drho * scale)
        return compute_elastic_mie(omega, RADIUS, REF, c, n_max=4).a_n[n]

    born = (a_n(EPS) - a_n(-EPS)) / (2 * EPS)
    return a_n(1.0) / born


def mie_static(dlam: float, dmu: float, drho: float, n: int) -> float:
    e1, e2 = mie_concentration(0.01, dlam, dmu, drho, n), mie_concentration(0.02, dlam, dmu, drho, n)
    return ((4 * e1 - e2) / 3).real


def main() -> int:
    oks: list[bool] = []

    def check(label: str, err: float, tol: float) -> None:
        ok = err < tol
        oks.append(ok)
        print(f"  {'PASS' if ok else 'FAIL'}  {label}: {err:.2e}")

    # [0] the derivatives
    x0, om0, hh = np.array([0.31, -0.52, 0.77]), 1.7, 1e-5
    fd = np.stack(
        [(green(x0 + hh * EYE[k], om0) - green(x0 - hh * EYE[k], om0)) / (2 * hh) for k in range(3)],
        axis=-1,
    )
    check(
        "[0] d_k G_in against finite differences (rel.)",
        np.abs(fd - dgreen(x0, om0)).max() / np.abs(fd).max(),
        1e-8,
    )
    kelvin_fd = np.stack(
        [(kelvin(x0 + hh * EYE[k]) - kelvin(x0 - hh * EYE[k])) / (2 * hh) for k in range(3)], axis=-1
    )
    check(
        "[0] static d_k G_in against finite differences of the Kelvin tensor (rel.)",
        np.abs(kelvin_fd - dgreen_static(x0)).max() / np.abs(kelvin_fd).max(),
        1e-8,
    )

    m_ball = surface_moment(dgreen_static, RADIUS).real
    print(
        f"  static ball moment: M_iiii = {m_ball[0, 0, 0, 0]:.12f}   M_1122 = {m_ball[0, 0, 1, 1]:.12f}"
        f"   M_1212 = {m_ball[0, 1, 0, 1]:.12f}"
    )

    # [1]-[3] the ball's channels against Mie, each contrast alone
    e0 = mie_static(DLAM, 0.0, 0.0, 0)
    bulk_ball = 1 / channels(m_ball, DLAM, 0.0)[0].real
    print(f"  E_0: Mie {e0:.12f}   ball closure {bulk_ball:.12f}")
    check("[1] ball dilatational channel == Mie E_0", abs(bulk_ball - e0) / e0, TOL_STATIC)
    e2 = mie_static(0.0, DMU, 0.0, 2)
    ch = channels(m_ball, 0.0, DMU)
    shear_ball = 1 / ch[2].real
    print(
        f"  E_2: Mie {e2:.12f}   ball closure {shear_ball:.12f}"
        f"   (ball Eg - T2g = {abs(ch[1] - ch[2]):.1e})"
    )
    check("[2] ball shear channel == Mie E_2", abs(shear_ball - e2) / e2, TOL_STATIC)
    e1 = mie_static(0.0, 0.0, DRHO, 1)
    print(f"  E_1: Mie {e1:.12f}")
    check("[3] density channel: Mie E_1 == 1", abs(e1 - 1), TOL_STATIC)

    # [4]-[5] the cube's closed forms (CubeA22Block.wl) against the same Mie numbers
    s_off = (np.pi * (LAM + 2 * MU) - SQ3 * (LAM + MU)) / (3 * np.pi * MU * (LAM + 2 * MU))
    s_diag = (3 * SQ3 * (LAM + MU) + 2 * np.pi * MU) / (6 * np.pi * MU * (LAM + 2 * MU))
    cube_bulk = 1 / (1 + DLAM / (LAM + 2 * MU))
    check("[4] cube dilatational channel == Mie E_0", abs(cube_bulk - e0) / e0, TOL_STATIC)
    s_mie = (1 / e2 - 1) / (2 * DMU)
    print(
        f"  shear depolarisation 2 mu S: Mie {2 * MU * s_mie:.10f}   cube T2g {2 * MU * s_off:.10f}"
        f"   Eg {2 * MU * s_diag:.10f}   cube average {2 * MU * (3 * s_off + 2 * s_diag) / 5:.10f}"
    )
    check(
        "[5] cube isotropic shear average == Mie",
        abs((3 * s_off + 2 * s_diag) / 5 - s_mie) / s_mie,
        TOL_STATIC,
    )

    # [C] a spherical excision drops the delta: PV = outer surface minus a small inner sphere
    m_pv = m_ball - surface_moment(dgreen_static, 1e-3).real
    pv_shear = 1 / channels(m_pv, 0.0, DMU)[2].real
    pv_bulk = 1 / channels(m_pv, DLAM, 0.0)[0].real
    err_c = min(abs(pv_shear - e2) / e2, abs(pv_bulk - e0) / e0)
    print(
        f"  [C] excision: E_0 {pv_bulk:.6f} (Mie {e0:.6f}), E_2 {pv_shear:.6f} (Mie {e2:.6f});"
        f" its depolarisations: bulk {1 / pv_bulk - 1:+.2e}, shear {1 / pv_shear - 1:+.2e}"
    )
    ok = err_c > 1e-3
    oks.append(ok)
    print(
        f"  {'PASS' if ok else 'FAIL'}  [C] control: the excised moment disagrees with Mie by"
        f" >= {err_c:.2e}"
    )

    # [D] dynamic, measured: closure amplification with the dynamic ball moments vs Mie E_n(w)
    print(
        "  [D] dynamic: (closure / Mie E_n) - 1   [closure = uniform internal field, dynamic ball moments]"
    )
    print("       k_S a      n=0 (dlam)      n=1 (drho)      n=2 (dmu)")
    rows = []
    for w in (0.05, 0.1, 0.2, 0.4):
        omega = w * BETA / RADIUS
        m_dyn = surface_moment(lambda x, om=omega: dgreen(x, om), RADIUS)
        c0 = 1 / channels(m_dyn, DLAM, 0.0)[0]
        c1 = 1 / (1 - omega**2 * DRHO * ball_displacement_moment(omega))
        c2 = 1 / channels(m_dyn, 0.0, DMU)[2]
        row = [
            c0 / mie_concentration(w, DLAM, 0, 0, 0) - 1,
            c1 / mie_concentration(w, 0, 0, DRHO, 1) - 1,
            c2 / mie_concentration(w, 0, DMU, 0, 2) - 1,
        ]
        rows.append((w, row))
        print(f"       {w:5.2f}   " + "   ".join(f"{abs(v):.3e}" for v in row))
    w, half = 0.2, 0.5
    omega = w * BETA / RADIUS
    m_dyn = surface_moment(lambda x: dgreen(x, omega), RADIUS)
    h0 = 1 / channels(m_dyn, half * DLAM, 0.0)[0] / mie_concentration(w, half * DLAM, 0, 0, 0) - 1
    h1 = 1 / (1 - omega**2 * half * DRHO * ball_displacement_moment(omega))
    h1 = h1 / mie_concentration(w, 0, 0, half * DRHO, 1) - 1
    h2 = 1 / channels(m_dyn, 0.0, half * DMU)[2] / mie_concentration(w, 0, half * DMU, 0, 2) - 1
    full = rows[2][1]
    print(
        "       halving the contrast at k_S a = 0.2 divides the departure by: "
        + "   ".join(f"{abs(f) / abs(hv):.3f}" for f, hv in zip(full, (h0, h1, h2), strict=True))
        + "   (2 = first order in the contrast, 4 = second)"
    )
    for j, name in enumerate(("n=0", "n=1", "n=2")):
        (w1, r1), (w2, r2) = rows[1], rows[2]
        print(
            f"       order of the departure, {name}:"
            f" {np.log(abs(r2[j]) / abs(r1[j])) / np.log(w2 / w1):.2f}"
        )

    # [E] the Born-order (ka)^2 coefficient, extracted and compared with the closed forms of
    # Mathematica/SphereClosureDynamic.wl:  closure/Mie - 1 = d_n (k a)^2 x contrast + O(contrast^2, (ka)^3)
    kp2, ks2 = (BETA / ALPHA) ** 2, 1.0  # (k_P a)^2 and (k_S a)^2 per w^2
    closed = [
        -kp2 * (DLAM / (LAM + 2 * MU)) / 10,
        (2 * ks2 + kp2) * (DRHO / RHO) / 30,
        -(3 * ks2 + 2 * kp2 * (BETA / ALPHA) ** 2) * (DMU / MU) / 75,
    ]

    def departure(w: float, scale: float) -> list[complex]:
        omega = w * BETA / RADIUS
        m_dyn = surface_moment(lambda x: dgreen(x, omega), RADIUS)
        c0 = 1 / channels(m_dyn, scale * DLAM, 0.0)[0] / mie_concentration(w, scale * DLAM, 0, 0, 0)
        c1 = 1 / (1 - omega**2 * scale * DRHO * ball_displacement_moment(omega))
        c1 = c1 / mie_concentration(w, 0, 0, scale * DRHO, 1)
        c2 = 1 / channels(m_dyn, 0.0, scale * DMU)[2] / mie_concentration(w, 0, scale * DMU, 0, 2)
        return [c0 - 1, c1 - 1, c2 - 1]

    def born_coeff(w: float) -> np.ndarray:  # first order in the contrast: 4 D(1/2) - D(1)
        d1, dh = np.array(departure(w, 1.0)), np.array(departure(w, 0.5))
        return (4 * dh - d1).real / w**2

    b1, b2 = born_coeff(0.2), born_coeff(0.1)
    extracted = (4 * b2 - b1) / 3  # Richardson in w: the next real term is O(w^2) relative
    worst = max(abs(e - c) / abs(c) for e, c in zip(extracted, closed, strict=True))
    print(
        "  [E] Born-order (k_S a)^2 coefficient: extracted "
        + ", ".join(f"{v:+.6e}" for v in extracted)
        + "\n      closed form (SphereClosureDynamic.wl)  "
        + ", ".join(f"{v:+.6e}" for v in closed)
    )
    check("[E] the departure's (ka)^2 coefficient == the closed forms (rel.)", worst, 2e-3)

    verdict = f"ALL {len(oks)} CHECKS PASS" if all(oks) else "CHECKS FAILED"
    print(f"==== gate_sphere_closure_vs_mie: {verdict}")
    return 0 if all(oks) else 1


if __name__ == "__main__":
    sys.exit(main())
