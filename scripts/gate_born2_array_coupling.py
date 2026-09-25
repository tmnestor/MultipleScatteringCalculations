#!/usr/bin/env python3
"""GATE: second-order Born predicts the array-coupling residual, independently of the march.

WHAT IS BEING PREDICTED.  The impedance march solves a periodic ARRAY of spheres;
exact Mie solves ONE.  Their specular P reflections share the first-order Born
term exactly (Poisson summation), so the relative departure between them is
linear in the contrast, and its coefficient is set by second-order Born:

    departure / eps  ->  |Delta_2| / |f_1| ,
    Delta_2 = (1/A) Sum_g I(g)  -  Int d^2q/(2 pi)^2 I(q) ,

where I(q) is the per-lateral-wavenumber double-scattering integral of
``scripts/born2_kernel.py``.  The march measures 0.157 eps at N=8, 0.190 eps at
N=9, rising under lateral refinement towards about 0.3 eps at 2.5 diameters and
k_P a = 1.44 (``gate_sphere_vs_impedance_march.py`` part 10).  Nothing here uses
the march, the voxel route or any multipole lattice machinery.

WHY A WINDOW.  The isolated sphere's own q-integral converges slowly -- its
large-q tail is the local self-interaction, falling only as ~q^-1.4 -- but
Delta_2 does not need it.  By Poisson summation Delta_2 couples a sphere only to
images at least L - 2a away, so the SAME smooth window W(q) = exp(-(q/Q)^8) can
be applied to the lattice sum and the integral: its real-space leakage from the
self-region to the nearest image is exp(-(360 Q)^(8/7)), and it is flat to 1e-5
through the propagating orders.  Convergence in Q is checked, not assumed.

NOT YET DONE: the isolated sphere's full integral against Mie's own eps^2
coefficient, which would test the kernel at every q against the exact sphere.
It needs the slow large-q tail handled asymptotically.  The kernel itself is
derived twice (Mathematica, and sympy in born2_kernel.py, agreeing to
1.7e-15), and the prediction is tested against the march below.

Run:  conda run -n seismic python scripts/gate_born2_array_coupling.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.special import j1

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubic_scattering.effective_contrasts import MaterialContrast, ReferenceMedium  # noqa: E402
from cubic_scattering.sphere_scattering import compute_elastic_mie, mie_far_field  # noqa: E402
from scripts.born2_kernel import kernel_numeric  # noqa: E402

REF = ReferenceMedium(5000.0, 3000.0, 2500.0)
OMEGA = 60.0
RADIUS = 120.0
KP, KS = OMEGA / REF.alpha, OMEGA / REF.beta
#: First-order contrasts per unit eps, for alpha, beta, rho all scaled by 1 + eps.
DRHO, DLAM, DMU = REF.rho, 3.0 * REF.lam, 3.0 * REF.mu
PREF = 1.0 / (4.0 * np.pi * REF.rho * REF.alpha**2)

_PASS: list[tuple[str, bool]] = []


def report(label: str, ok: bool) -> None:
    """Record and print one check.

    Args:
        label: What is checked.
        ok: Whether it passed.
    """
    _PASS.append((label, ok))
    print(f"  {'PASS' if ok else '****FAIL****'}  {label}")


def form_factor(q: float, z: NDArray) -> NDArray:
    """Lateral transform of the sphere's cross-section at depth z (centre at z = 0).

    Args:
        q: Lateral wavenumber magnitude.
        z: Depths.

    Returns:
        F(q, z) = 2 pi R^2 J_1(qR)/(qR), R^2 = a^2 - z^2, zero outside.
    """
    r2 = np.clip(RADIUS**2 - z**2, 0.0, None)
    r = np.sqrt(r2)
    x = q * r
    ratio = np.where(
        x < 1e-8, 0.5 - x**2 / 16.0, j1(np.where(x < 1e-8, 1.0, x)) / np.where(x < 1e-8, 1.0, x)
    )
    return 2.0 * np.pi * r2 * ratio


def _gauss(n: int) -> tuple[NDArray, NDArray]:
    x, w = np.polynomial.legendre.leggauss(n)
    return x, w


def per_q(q: float, n_outer: int = 96, n_inner: int = 48) -> tuple[complex, complex]:
    """I(q): the double-scattering integral at one lateral wavenumber.

    Split as regular part (the double z-integral) and contact part.  The regular
    kernel is sum_K A_{K,sign} e^{i kap_K |z - z'|}; each ordered half of the
    square is integrated with the inner interval split at a distance set by the
    evanescent decay, where the kernel concentrates.

    Args:
        q: Lateral wavenumber magnitude.
        n_outer: Gauss nodes for the outer depth.
        n_inner: Gauss nodes for each inner panel.

    Returns:
        (regular part, contact part).
    """
    kn = kernel_numeric()
    # Complex q so that sqrt(k^2 - q^2) takes the branch Im >= 0 past the branch
    # points, the same branch as kap below.
    args = (complex(q), KP, KS, REF.rho, OMEGA, DRHO, DLAM, DMU)
    kap = {"P": np.sqrt(complex(KP**2 - q**2)), "S": np.sqrt(complex(KS**2 - q**2))}
    coef = {k: complex(kn[k](*args)) for k in ("P+", "P-", "S+", "S-")}
    c0 = complex(kn["c0"](*args))

    xo, wo = _gauss(n_outer)
    zo = RADIUS * xo
    wo = RADIUS * wo
    go = np.exp(1j * KP * zo) * form_factor(q, zo)
    decay = max(abs(kap["P"].imag), abs(kap["S"].imag), 1e-30)
    width = min(2.0 * RADIUS, 12.0 / decay)
    xi, wi = _gauss(n_inner)

    reg = 0.0j
    # zeta = z - z' > 0 half: z' in [-a, z]; the other half by symmetry of the
    # ordered pair (z' in [z, a]).
    for sign, tag in ((1.0, "+"), (-1.0, "-")):
        total = np.zeros_like(zo, dtype=complex)
        for zc, wc in zip(zo, np.arange(zo.size), strict=True):
            lo, hi = (-RADIUS, zc) if sign > 0 else (zc, RADIUS)
            span = hi - lo
            if span <= 0.0:
                continue
            # two panels: the one adjacent to z (width `width`) and the rest
            near = min(width, span)
            pts, wts = [], []
            if sign > 0:
                panels = [(hi - near, hi)] + ([(lo, hi - near)] if span > near else [])
            else:
                panels = [(lo, lo + near)] + ([(lo + near, hi)] if span > near else [])
            for a_, b_ in panels:
                pts.append(0.5 * (b_ - a_) * xi + 0.5 * (b_ + a_))
                wts.append(0.5 * (b_ - a_) * wi)
            zp = np.concatenate(pts)
            wp = np.concatenate(wts)
            gi = np.exp(1j * KP * zp) * form_factor(q, zp)
            zeta = np.abs(zc - zp)
            kern = coef["P" + tag] * np.exp(1j * kap["P"] * zeta) + coef["S" + tag] * np.exp(
                1j * kap["S"] * zeta
            )
            total[wc] = np.sum(wp * gi * kern)
        reg += np.sum(wo * go * total)
    contact = c0 * np.sum(wo * go**2)
    return complex(reg), complex(contact)


def mie_backscatter(eps: float) -> complex:
    """Exact backscattered P amplitude of the sphere at contrast eps.

    Args:
        eps: Fractional perturbation of alpha, beta and rho.

    Returns:
        f(pi), the far-field amplitude in the convention u = f e^{ikr}/r.
    """
    s = 1.0 + eps
    con = MaterialContrast(
        Dlambda=(s**3 - 1.0) * REF.lam, Dmu=(s**3 - 1.0) * REF.mu, Drho=(s - 1.0) * REF.rho
    )
    f_p, _, _ = mie_far_field(
        compute_elastic_mie(OMEGA, RADIUS, REF, con), np.array([np.pi]), incident_type="P"
    )
    return complex(f_p[0])


def first_order() -> complex:
    """The first-order Born backscatter amplitude per unit eps.

    Returns:
        f_1, in the convention of ``mie_backscatter``.
    """
    chi = (
        4.0 * np.pi * (np.sin(2 * KP * RADIUS) - 2 * KP * RADIUS * np.cos(2 * KP * RADIUS)) / (2 * KP) ** 3
    )
    bracket = -(OMEGA**2) * DRHO - KP**2 * (DLAM + 2.0 * DMU)
    return complex(bracket * chi * PREF)


def window(q: NDArray | float, qw: float) -> NDArray | float:
    """The common spectral window, flat through the propagating orders.

    Args:
        q: Lateral wavenumber magnitude(s).
        qw: Window scale.

    Returns:
        exp(-(q/qw)^8).
    """
    return np.exp(-((np.asarray(q) / qw) ** 8))


def continuum(qw: float, n_seg: int = 40) -> complex:
    """(1/(2 pi)) Int_0^inf q W(q) I(q) dq, i.e. Int d^2q/(2 pi)^2 W I for radial I.

    Three segments, each mapped to remove the inverse-square-root branch
    points of 1/kap_P and 1/kap_S at its ends.

    Args:
        qw: Window scale.
        n_seg: Gauss nodes per segment.

    Returns:
        The windowed continuum integral.
    """
    t, w = _gauss(n_seg)
    total = 0.0j
    # [0, kP]: q = kP sin(th)
    th = 0.25 * np.pi * (t + 1.0)
    for q, wq in zip(KP * np.sin(th), 0.25 * np.pi * w * KP * np.cos(th), strict=True):
        total += wq * q * window(q, qw) * sum(per_q(q))
    # [kP, kS]: q = c - d cos(th)
    c, d = 0.5 * (KP + KS), 0.5 * (KS - KP)
    th = 0.5 * np.pi * (t + 1.0)
    for q, wq in zip(c - d * np.cos(th), 0.5 * np.pi * w * d * np.sin(th), strict=True):
        total += wq * q * window(q, qw) * sum(per_q(q))
    # [kS, qmax]: q = kS cosh(u), in two pieces so the flat part is well sampled
    umax = float(np.arccosh(1.6 * qw / KS))
    for u0, u1 in ((0.0, 0.5 * umax), (0.5 * umax, umax)):
        u = 0.5 * (u1 - u0) * (t + 1.0) + u0
        for q, wq in zip(KS * np.cosh(u), 0.5 * (u1 - u0) * w * KS * np.sinh(u), strict=True):
            total += wq * q * window(q, qw) * sum(per_q(q))
    return complex(total / (2.0 * np.pi))


def lattice(period: float, qw: float) -> complex:
    """(1/A) Sum_g W(g) I(g) over the square lattice of diffraction orders.

    Args:
        period: Lattice period.
        qw: Window scale.

    Returns:
        The windowed lattice sum.
    """
    g0 = 2.0 * np.pi / period
    mmax = int(np.ceil(1.6 * qw / g0))
    radii: dict[int, int] = {}
    for m in range(-mmax, mmax + 1):
        for n in range(-mmax, mmax + 1):
            radii[m * m + n * n] = radii.get(m * m + n * n, 0) + 1
    total = 0.0j
    for r2, count in radii.items():
        q = g0 * np.sqrt(r2)
        wq = window(q, qw)
        if wq < 1e-30:
            continue
        total += count * wq * sum(per_q(q))
    return complex(total / period**2)


def main() -> int:
    """Run the checks.

    Returns:
        0 if all pass.
    """
    print("=" * 78)
    print("SECOND-ORDER BORN: THE ARRAY'S DEPARTURE FROM THE ISOLATED SPHERE")
    print(f"  k_P a = {KP * RADIUS:.2f}   k_S a = {KS * RADIUS:.2f}")
    print("=" * 78)
    f1 = first_order()
    f1_mie = (mie_backscatter(1e-6) - mie_backscatter(-1e-6)) / 2e-6
    print("\n[1] the first-order term, against Mie's linear coefficient")
    print(f"      first-order Born f_1 = {f1:.7e}   Mie = {f1_mie:.7e}")
    report("first-order Born is Mie's linear coefficient", abs(f1 - f1_mie) / abs(f1_mie) < 1e-5)

    # (2) THE WINDOW.  Sum and integral each move with Q; their difference must
    # not, because the window's real-space leakage from the self-region to the
    # nearest image (360 m away) is exp(-(360 Q)^(8/7)) and it is flat to 1e-5
    # through the propagating orders.
    period = 600.0
    print(f"\n[2] Delta_2 at period {period:.0f} m ({period / (2 * RADIUS):.1f} diameters)")
    print(f"      {'Q':>6}{'(1/A) sum':>28}{'continuum':>28}{'Delta_2 / f_1':>26}")
    preds = []
    for qw in (0.08, 0.12, 0.16):
        s = lattice(period, qw)
        c = continuum(qw)
        preds.append(PREF * (s - c) / f1)
        print(f"      {qw:6.2f}{s:>28.6e}{c:>28.6e}{preds[-1]:>26.6f}")
    pred = preds[-1]
    spread = max(abs(p - pred) for p in preds) / abs(pred)
    report("Delta_2 is independent of the window", spread < 1e-4)

    # (3)-(4) AGAINST THE MARCH, which shares nothing with this calculation.
    import scripts.gate_sphere_vs_impedance_march as march

    print("\n[3] the imaginary part, against the depth-converged march (N=8, 256 steps)")
    got, want = march._specular_at(8, period, 1e-3, 256)
    d8 = (got - want) / want / 1e-3
    print(f"      march departure/eps {d8:.5f}   Born prediction {pred:.5f}")
    report("the imaginary part matches to 1% of |departure|", abs(d8.imag - pred.imag) < 0.01 * abs(pred))

    print("\n[4] the real part, against the march's lateral ladder extrapolated in 1/N")
    ns = np.array([8.0, 12.0, 16.0, 20.0])
    ds = []
    for nx in ns.astype(int):
        got, want = march._specular_at(int(nx), period, 1e-3, 64)
        ds.append((got - want) / want / 1e-3)
        print(f"      N={int(nx):<3d} departure/eps {ds[-1]:.5f}")
    fit = np.linalg.lstsq(
        np.column_stack([np.ones_like(ns), 1 / ns, 1 / ns**2]), np.array(ds).real, rcond=None
    )[0]
    print(f"      extrapolated real part {fit[0]:.4f}   Born prediction {pred.real:.4f}")
    report("the real part matches to 5%", abs(fit[0] - pred.real) < 0.05 * abs(pred.real))

    print(
        "\n      So the residual the march converges to at 2.5 diameters, 0.29 eps,\n"
        "      is the ARRAY's double scattering: second-order Born of the periodic\n"
        "      medium less that of the isolated sphere predicts it in real and\n"
        "      imaginary part, with no march, voxel or multipole machinery."
    )
    npass = sum(ok for _, ok in _PASS)
    print("\n" + "=" * 78)
    print(f"{npass}/{len(_PASS)} checks passed")
    print("=" * 78)
    return 0 if npass == len(_PASS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
