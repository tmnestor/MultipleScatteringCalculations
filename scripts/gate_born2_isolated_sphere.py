#!/usr/bin/env python3
"""GATE: second-order Born of the ISOLATED sphere against Mie's own eps^2 coefficient.

WHAT THIS TESTS.  ``gate_born2_array_coupling`` tested the double-scattering
kernel on the array's DIFFERENCE from one sphere, where the sphere's
self-interaction -- the kernel's large-q part and its contact term -- cancels.
Here the sphere is alone, so every q and the contact term count, and the
answer is known exactly: Mie's eps^2 coefficient equals the first-order term
carrying the eps^2 part of the contrast plus the double-scattering integral.

HOW: SINGULARITY SUBTRACTION.  The q-integral's tail is the local
self-interaction, falling only as ~q^-1.4.  With the static kernel of
``born2_static_kernel`` (the same vertices, Kelvin tensor between them):

    f2 = PREF [ Int d^2q/(2 pi)^2 (I_dyn - I_static)  +  S_static ]

where the remainder decays like (k/q)^2 faster and S_static is the static
self-term in real space, exactly (``Mathematica/BornSecondOrderStatic.wl``).
Both kernels are integrated on IDENTICAL nodes, so quadrature error in the
shared form factors cancels in the remainder.

Run:  conda run -n seismic python scripts/gate_born2_isolated_sphere.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.born2_static_kernel import kernel_functions  # noqa: E402
from scripts.gate_born2_array_coupling import (  # noqa: E402
    DLAM,
    DMU,
    DRHO,
    KP,
    KS,
    OMEGA,
    RADIUS,
    REF,
    _gauss,
    form_factor,
)

PAR = (KP, KS, REF.rho, OMEGA, REF.lam, REF.mu, DRHO, DLAM, DMU)


def remainder(q: float, n_outer: int = 96, n_inner: int = 48) -> complex:
    """I_dyn(q) - I_static(q), both on the same nodes.

    Args:
        q: Lateral wavenumber magnitude.
        n_outer: Gauss nodes for the outer depth.
        n_inner: Gauss nodes per inner panel.

    Returns:
        The remainder (the contact terms are identical and cancel exactly).
    """
    kd, ks = kernel_functions("dynamic"), kernel_functions("static")
    xo, wo = _gauss(n_outer)
    zo, wo = RADIUS * xo, RADIUS * wo
    go = np.exp(1j * KP * zo) * form_factor(q, zo)
    width = min(2.0 * RADIUS, 12.0 / max(q, 1e-30))
    xi, wi = _gauss(n_inner)
    total = 0.0j
    for sgn, key in ((1.0, "+"), (-1.0, "-")):
        acc = np.zeros(len(zo), dtype=complex)
        for idx, zc in enumerate(zo):
            lo, hi = (-RADIUS, zc) if sgn > 0 else (zc, RADIUS)
            span = hi - lo
            if span <= 0.0:
                continue
            near = min(width, span)
            panels = (
                [(hi - near, hi)] + ([(lo, hi - near)] if span > near else [])
                if sgn > 0
                else [(lo, lo + near)] + ([(lo + near, hi)] if span > near else [])
            )
            zp = np.concatenate([0.5 * (b - a) * xi + 0.5 * (b + a) for a, b in panels])
            wp = np.concatenate([0.5 * (b - a) * wi for a, b in panels])
            t = np.abs(zc - zp)
            kern = np.asarray(kd[key](t, complex(q), *PAR), dtype=np.complex128) - np.asarray(
                ks[key](t, q, *PAR), dtype=np.complex128
            )
            acc[idx] = np.sum(wp * np.exp(1j * KP * zp) * form_factor(q, zp) * kern)
        total += np.sum(wo * go * acc)
    return complex(total)


def remainder_integral(qmax: float, n_seg: int = 48) -> complex:
    """(1/(2 pi)) Int_0^qmax q R(q) dq = Int d^2q/(2 pi)^2 R for radial R.

    The same three segments as the array gate, each mapped to remove the
    inverse-square-root branch points of the DYNAMIC kernel at k_P and k_S
    (the static kernel has none).

    Args:
        qmax: Cutoff; the remainder falls like q^-5, so its tail is checked, not modelled.
        n_seg: Gauss nodes per segment.

    Returns:
        The integral.
    """
    t, w = _gauss(n_seg)
    total = 0.0j
    th = 0.25 * np.pi * (t + 1.0)
    for q, wq in zip(KP * np.sin(th), 0.25 * np.pi * w * KP * np.cos(th), strict=True):
        total += wq * q * remainder(q)
    c, d = 0.5 * (KP + KS), 0.5 * (KS - KP)
    th = 0.5 * np.pi * (t + 1.0)
    for q, wq in zip(c - d * np.cos(th), 0.5 * np.pi * w * d * np.sin(th), strict=True):
        total += wq * q * remainder(q)
    umax = float(np.arccosh(qmax / KS))
    for u0, u1 in ((0.0, 0.4 * umax), (0.4 * umax, umax)):
        u = 0.5 * (u1 - u0) * (t + 1.0) + u0
        for q, wq in zip(KS * np.cosh(u), 0.5 * (u1 - u0) * w * KS * np.sinh(u), strict=True):
            total += wq * q * remainder(q)
    return complex(total / (2.0 * np.pi))


def mie_eps2() -> complex:
    """Mie's eps^2 coefficient of the backscatter, by Richardson on the even part.

    g(eps) = (f(eps) + f(-eps)) / (2 eps^2) = c2 + c4 eps^2 + ..., so
    (4 g(eps) - g(2 eps)) / 3 = c2 + O(eps^4).

    Returns:
        c2.
    """
    from scripts.gate_born2_array_coupling import mie_backscatter

    def g(e: float) -> complex:
        return (mie_backscatter(e) + mie_backscatter(-e)) / (2.0 * e**2)

    eps = 1e-3
    return (4.0 * g(eps) - g(2.0 * eps)) / 3.0


def first_order_eps2() -> complex:
    """The first-order term carrying the eps^2 part of the contrast.

    s^3 = 1 + 3 eps + 3 eps^2 + eps^3, so at eps^2: dlam = 3 lam, dmu = 3 mu,
    drho = 0.

    Returns:
        B1^(2).
    """
    from scripts.gate_born2_array_coupling import PREF

    chi = (
        4.0 * np.pi * (np.sin(2 * KP * RADIUS) - 2 * KP * RADIUS * np.cos(2 * KP * RADIUS)) / (2 * KP) ** 3
    )
    return complex(-(KP**2) * (3.0 * REF.lam + 6.0 * REF.mu) * chi * PREF)


def main() -> int:
    """Run the checks.

    Returns:
        0 if all pass.
    """
    print("remainder I_dyn - I_static: decay and resolution")
    for q in (0.03, 0.1, 0.3):
        r = remainder(q)
        r2 = remainder(q, n_outer=192, n_inner=96)
        print(f"  q = {q:5.2f}   q R = {q * r:.3e}   (refined {abs(r2 - r) / abs(r):.1e})")
    print("\nremainder integral against its cutoff")
    for qmax in (0.2, 0.3, 0.4):
        print(f"  q_max = {qmax:4.2f}: {remainder_integral(qmax):.10e}")
    import json

    from scripts.gate_born2_array_coupling import PREF

    rint = remainder_integral(0.4)
    s_static = complex(
        *json.loads((ROOT / "Mathematica" / "BornSecondOrderStatic.json").read_text())["S_static"]
    )
    mie2 = mie_eps2()
    b12 = first_order_eps2()
    born = b12 + PREF * (rint + s_static)
    print("\nthe eps^2 coefficient of the backscatter")
    print(f"  first-order term, eps^2 contrast        {b12:.8f}")
    print(f"  double scattering: remainder, PREF x    {PREF * rint:.8f}")
    print(f"  double scattering: static, PREF x       {PREF * s_static:.8f}")
    print(f"  second-order Born, total                {born:.8f}")
    print(f"  Mie                                     {mie2:.8f}")
    rel = abs(born - mie2) / abs(mie2 - b12)
    print(f"  |Born - Mie| / |double scattering|      {rel:.2e}")
    ok = rel < 1e-3
    verdict = "PASS" if ok else "****FAIL****"
    print(f"  {verdict}  second-order Born of the isolated sphere is Mie's eps^2 coefficient")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
