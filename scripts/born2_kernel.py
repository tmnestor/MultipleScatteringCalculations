#!/usr/bin/env python3
"""The second-order elastic Born kernel per lateral wavenumber -- the Python twin.

Re-derives, with sympy, the kernel that ``Mathematica/ElasticBornSecondOrderKernel.wl``
derives, sharing nothing with it but the statement of the problem:

    f2(q) = (1/(4 pi rho alpha^2)) Int dz Int dz' e^{i kP (z+z')} F(q,z) F(q,z') K(q; z-z')

for a P wave at normal incidence scattered back into P.  Both vertex derivatives
are D = (i qx, i qy, p), p = d/dzeta acting on the spectral Green's tensor
Ghat = [kS^2 gS I + D D (gS - gP)]/(rho w^2), gK = (i/(2 kapK)) e^{i kapK |zeta|}.

The kernel is a polynomial in p acting on gS and gP.  Each p^n gK has the regular
part (i/(2 kap)) (i kap s)^n e^{i kap |zeta|}, s = sign(zeta), and the contact parts
-delta (n = 2), -delta' (n = 3), kap^2 delta - delta'' (n = 4).

Run:  conda run -n seismic python scripts/born2_kernel.py   (checks against the Mathematica dump)
"""

from __future__ import annotations

import functools
import json
import sys
from pathlib import Path

import sympy as sp

ROOT = Path(__file__).resolve().parent.parent
REF = ROOT / "Mathematica" / "ElasticBornSecondOrderKernel.json"

_qx, _qy, _p, _kp, _ks, _rho, _w, _drho, _dlam, _dmu = sp.symbols("qx qy p kP kS rho w drho dlam dmu")
_gs, _gp = sp.symbols("gS gP")


def _cten(lam: sp.Expr, mu: sp.Expr) -> list:
    """The isotropic elastic tensor as a nested list.

    Args:
        lam: Lame lambda.
        mu: Shear modulus.

    Returns:
        c[i][j][k][l].
    """
    d = [[1 if i == j else 0 for j in range(3)] for i in range(3)]
    return [
        [
            [
                [lam * d[i][j] * d[k][l] + mu * (d[i][k] * d[j][l] + d[i][l] * d[j][k]) for l in range(3)]
                for k in range(3)
            ]
            for j in range(3)
        ]
        for i in range(3)
    ]


@functools.cache
def kernel_symbolic() -> dict[str, sp.Expr]:
    """The kernel's regular and contact coefficients, symbolic in q, k, rho, w, contrasts.

    Returns:
        Keys "P+", "P-", "S+", "S-" (regular coefficient of e^{i kap |zeta|} for
        zeta of that sign) and "c0", "c1", "c2" (delta, delta', delta'').
    """
    dd = [sp.I * _qx, sp.I * _qy, _p]
    gh = [
        [
            (_ks**2 * _gs * (1 if i == j else 0) + dd[i] * dd[j] * (_gs - _gp)) / (_rho * _w**2)
            for j in range(3)
        ]
        for i in range(3)
    ]
    dc = _cten(_dlam, _dmu)
    e = [0, 0, 1]
    n = [0, 0, -1]
    kin = [0, 0, _kp]
    ksc = [0, 0, -_kp]
    u1 = []
    for i in range(3):
        term = _w**2 * _drho * sum(gh[i][j] * e[j] for j in range(3))
        for j in range(3):
            for k in range(3):
                for ll in range(3):
                    for m in range(3):
                        c = dc[j][k][ll][m]
                        if c != 0 and kin[m] != 0 and e[ll] != 0:
                            term += c * dd[k] * gh[i][j] * sp.I * kin[m] * e[ll]
        u1.append(term)
    kern = sum(_w**2 * _drho * n[ll] * u1[ll] for ll in range(3))
    for j in range(3):
        for k in range(3):
            for ll in range(3):
                for m in range(3):
                    c = dc[j][k][ll][m]
                    if c != 0 and n[j] != 0 and ksc[k] != 0:
                        kern += sp.I * c * n[j] * ksc[k] * dd[m] * u1[ll]
    kern = sp.expand(kern)

    kap = {_gp: sp.sqrt(_kp**2 - _qx**2 - _qy**2), _gs: sp.sqrt(_ks**2 - _qx**2 - _qy**2)}

    def coef(g: sp.Symbol, nn: int) -> sp.Expr:
        return sp.expand(kern).coeff(g).coeff(_p, nn)

    out: dict[str, sp.Expr] = {}
    for name, g in (("P", _gp), ("S", _gs)):
        for sgn, tag in ((1, "+"), (-1, "-")):
            out[name + tag] = sum(
                coef(g, nn) * (sp.I / (2 * kap[g])) * (sp.I * kap[g] * sgn) ** nn for nn in range(5)
            )
    out["c0"] = sum(-coef(g, 2) + coef(g, 4) * kap[g] ** 2 for g in (_gs, _gp))
    out["c1"] = sum(-coef(g, 3) for g in (_gs, _gp))
    out["c2"] = sum(-coef(g, 4) for g in (_gs, _gp))
    return out


@functools.cache
def kernel_numeric() -> dict[str, object]:
    """The kernel coefficients as numpy functions of (q, kP, kS, rho, w, drho, dlam, dmu).

    Rotation invariance makes them functions of |q| alone; they are evaluated
    with q along x.

    Returns:
        Lambdified coefficient functions, keys as ``kernel_symbolic``.
    """
    sym = kernel_symbolic()
    args = (_qx, _kp, _ks, _rho, _w, _drho, _dlam, _dmu)
    return {
        k: sp.lambdify(args, v.subs(_qy, 0), "numpy") for k, v in sym.items() if k != "c1" and k != "c2"
    }


def main() -> int:
    """Check the twin against the Mathematica dump.

    Returns:
        0 on agreement.
    """
    sym = kernel_symbolic()
    c1, c2 = sp.simplify(sym["c1"]), sp.simplify(sym["c2"])
    c0 = sp.factor(sp.simplify(sym["c0"]))
    print(f"delta' coefficient: {c1}   delta'' coefficient: {c2}")
    print(f"delta coefficient:  {c0}")
    ok = c1 == 0 and c2 == 0
    data = json.loads(REF.read_text())
    b, c = data["background"], data["contrast_per_eps"]
    worst = 0.0
    for row in data["rows"]:
        qx, qy = row["q"]
        vals = {
            k: complex(
                sym[k]
                .subs(
                    {
                        _qx: qx,
                        _qy: qy,
                        _kp: b["omega"] / b["alpha"],
                        _ks: b["omega"] / b["beta"],
                        _rho: b["rho"],
                        _w: b["omega"],
                        _drho: c["drho"],
                        _dlam: c["dlam"],
                        _dmu: c["dmu"],
                    }
                )
                .evalf(30)
            )
            for k in ("P+", "P-", "S+", "S-", "c0")
        }
        want = {k: complex(*row[k if k == "c0" else "reg" + k]) for k in vals}
        scale = max(abs(v) for v in want.values())
        worst = max(worst, max(abs(vals[k] - want[k]) for k in vals) / scale)
    print(f"sympy twin vs Mathematica, worst relative coefficient difference: {worst:.2e}")
    ok = ok and worst < 1e-12
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
