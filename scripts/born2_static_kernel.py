#!/usr/bin/env python3
"""The static (Kelvin) counterpart of the second-order Born kernel, for singularity subtraction.

WHY.  The isolated sphere's second-order Born integral over lateral wavenumber
q converges slowly: its large-q tail is the sphere's LOCAL self-interaction,
where points so close that the wave hardly propagates between them see the
static Green's tensor.  Subtracting the same kernel built on the static tensor
leaves a remainder that decays like (k/q)^2 faster; the static part is then
added back in real space, exactly (``Mathematica/BornSecondOrderStatic.wl``).

THE STATIC TENSOR, spectrally.  As omega -> 0 at fixed q,
    kS^2 gS / (rho w^2)          -> g0 / mu,           g0 = e^{-q|z|}/(2q)
    (gS - gP) / (rho w^2)        -> h,                 h  = c (1 + q|z|) e^{-q|z|} / q^3,
    c = (1/beta^2 - 1/alpha^2) / (4 rho) = (1/mu - 1/(lam + 2 mu)) / 4,
the lateral transforms of 1/(4 pi mu r) and of -(1/beta^2 - 1/alpha^2) r/(8 pi rho),
i.e. the Kelvin tensor.  The vertices are unchanged: they carry the incident and
scattered wavenumbers, which are not being subtracted.

DISTRIBUTIONAL DERIVATIVES, for any even f(z) = f+(|z|): the regular part of
f^(n) at sign s is s^n f+^(n)(|z|), and its contact part is
    sum_m 2 f+^(2m+1)(0) delta^(n-2m-2) ,
which reproduces the dynamic table (-delta; -delta'; kap^2 delta - delta'').

Run:  conda run -n seismic python scripts/born2_static_kernel.py
"""

from __future__ import annotations

import functools
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.born2_kernel import _cten, kernel_numeric  # noqa: E402

_q, _t, _p = sp.symbols("q t p", positive=True)
_kp, _ks, _rho, _w, _lam, _mu = sp.symbols("kP kS rho w lam mu", positive=True)
_drho, _dlam, _dmu = sp.symbols("drho dlam dmu")


def _two_vertex(gh: list) -> sp.Expr:
    """The kernel as a polynomial in p acting on the Green's-tensor basis symbols.

    Args:
        gh: 3x3 spectral Green's tensor, entries polynomial in p over basis symbols.

    Returns:
        Expanded kernel.
    """
    dd = [sp.I * _q, 0, _p]
    dc = _cten(_dlam, _dmu)
    e, n = [0, 0, 1], [0, 0, -1]
    kin, ksc = [0, 0, _kp], [0, 0, -_kp]
    u1 = []
    for i in range(3):
        term = _w**2 * _drho * gh[i][2]
        for j in range(3):
            for k in range(3):
                term += dc[j][k][2][2] * dd[k] * gh[i][j] * sp.I * kin[2] * e[2]
        u1.append(term)
    kern = _w**2 * _drho * n[2] * u1[2]
    for ll in range(3):
        for m in range(3):
            c = dc[2][2][ll][m]
            if c != 0:
                kern += sp.I * c * n[2] * ksc[2] * dd[m] * u1[ll]
    return sp.expand(kern)


def _decompose(kern: sp.Expr, basis: dict[sp.Symbol, sp.Expr]) -> dict[str, sp.Expr]:
    """Regular parts (as functions of t = |zeta|) and contact coefficients.

    Args:
        kern: Polynomial in p over the basis symbols.
        basis: symbol -> its even function's expression f+(t) for t = |zeta|.

    Returns:
        {"+": regular part for zeta > 0, "-": for zeta < 0, "c0", "c1", "c2"}.
    """
    reg = {1: sp.Integer(0), -1: sp.Integer(0)}
    contact = {0: sp.Integer(0), 1: sp.Integer(0), 2: sp.Integer(0)}
    for sym, fplus in basis.items():
        for nn in range(5):
            c = kern.coeff(sym).coeff(_p, nn)
            if c == 0:
                continue
            dn = sp.diff(fplus, _t, nn)
            for s in (1, -1):
                reg[s] += c * s**nn * dn
            for m in range(0, 3):
                r = nn - 2 * m - 2
                if r >= 0:
                    contact[r] += c * 2 * sp.diff(fplus, _t, 2 * m + 1).subs(_t, 0)
    return {"+": reg[1], "-": reg[-1], "c0": contact[0], "c1": contact[1], "c2": contact[2]}


@functools.cache
def kernels() -> dict[str, dict[str, sp.Expr]]:
    """The dynamic and static kernels through one code path.

    Returns:
        {"dynamic": ..., "static": ...}, each as ``_decompose`` returns.
    """
    gs, gp, g0, hh = sp.symbols("gS gP g0 h")
    dd = [sp.I * _q, 0, _p]
    kap = {gp: sp.sqrt(_kp**2 - _q**2), gs: sp.sqrt(_ks**2 - _q**2)}
    dyn_gh = [
        [
            (_ks**2 * gs * (1 if i == j else 0) + dd[i] * dd[j] * (gs - gp)) / (_rho * _w**2)
            for j in range(3)
        ]
        for i in range(3)
    ]
    dyn = _decompose(
        _two_vertex(dyn_gh), {g: sp.I / (2 * kap[g]) * sp.exp(sp.I * kap[g] * _t) for g in (gs, gp)}
    )
    st_gh = [[(g0 * (1 if i == j else 0)) / _mu + dd[i] * dd[j] * hh for j in range(3)] for i in range(3)]
    c = (1 / _mu - 1 / (_lam + 2 * _mu)) / 4
    st = _decompose(
        _two_vertex(st_gh),
        {g0: sp.exp(-_q * _t) / (2 * _q), hh: c * (1 + _q * _t) * sp.exp(-_q * _t) / _q**3},
    )
    return {"dynamic": dyn, "static": st}


@functools.cache
def kernel_functions(kind: str) -> dict[str, Callable[..., Any]]:
    """Numeric kernels: f(t, q, kP, kS, rho, w, lam, mu, drho, dlam, dmu).

    Args:
        kind: "dynamic" or "static".

    Returns:
        {"+", "-", "c0"}: lambdified.
    """
    args = (_t, _q, _kp, _ks, _rho, _w, _lam, _mu, _drho, _dlam, _dmu)
    k = kernels()[kind]
    return {key: sp.lambdify(args, k[key], "numpy") for key in ("+", "-", "c0")}


def main() -> int:
    """Checks on the static kernel.

    Returns:
        0 if all pass.
    """
    ok = True
    k = kernels()
    st, dyn = k["static"], k["dynamic"]
    print("[1] contact terms")
    for name in ("c1", "c2"):
        v = sp.simplify(st[name])
        print(f"    static {name}: {v}")
        ok &= v == 0
    diff = sp.simplify(st["c0"] - dyn["c0"].subs({_ks: _w / sp.sqrt(_mu / _rho), _kp: _kp}))
    print(f"    static c0 - dynamic c0: {diff}")

    rho, al, be, w = 2500.0, 5000.0, 3000.0, 60.0
    mu, lam = rho * be**2, rho * (al**2 - 2 * be**2)
    kp, ks = w / al, w / be
    par = (kp, ks, rho, w, lam, mu, rho, 3 * lam, 3 * mu)

    print("[2] the generic path reproduces the committed dynamic kernel")
    kd = kernel_functions("dynamic")
    kn = kernel_numeric()
    worst = 0.0
    for q in (0.004, 0.015, 0.05, 0.3):
        for t in (0.0, 3.0, 40.0):
            for sgn in ("+", "-"):
                got = complex(kd[sgn](t, complex(q), *par))
                kap_p = np.sqrt(complex(kp**2 - q**2))
                kap_s = np.sqrt(complex(ks**2 - q**2))
                a = (complex(q), kp, ks, rho, w, rho, 3 * lam, 3 * mu)
                want = complex(kn["P" + sgn](*a)) * np.exp(1j * kap_p * t) + complex(
                    kn["S" + sgn](*a)
                ) * np.exp(1j * kap_s * t)
                worst = max(worst, abs(got - want) / max(abs(want), 1e-300))
    print(f"    worst relative difference {worst:.1e}")
    ok &= worst < 1e-10

    print("[3] the dynamic kernel tends to the static one at large q (should fall like (k/q)^2)")
    ks_ = kernel_functions("static")
    for q in (0.1, 0.3, 1.0, 3.0):
        t = 1.0 / q
        d = complex(kd["+"](t, complex(q), *par))
        s = complex(ks_["+"](t, q, *par))
        rel = abs(d - s) / abs(d)
        print(f"    q = {q:5.2f}: |K_dyn - K_st| / |K_dyn| = {rel:.2e}   (k_S/q)^2 = {(ks / q) ** 2:.1e}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
