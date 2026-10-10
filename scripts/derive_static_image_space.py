"""Twin of Mathematica/StaticImageSpace.wl: the elastostatic image of a welded interface, in space.

Reads Mathematica/StaticInterfaceImage.json (the reflected static kernel in the lateral-wavenumber domain)
and transforms it with zeta = -(z + zp) > 0, rho the lateral separation, R = sqrt(rho^2 + zeta^2):

    q^m e^{-q zeta}  ->  (-d_zeta)^(m+1) Phi0 (m >= -1),  Phi1 (m = -2),  Phi2 (m = -3),
    Phi0 = 1/(2 pi R),  Phi1 = -log(R + zeta)/(2 pi),  Phi2 = (zeta log(R + zeta) - R)/(2 pi);

i qhat_h -> d_rho_h on the transform of (entry)/q, qhat_h qhat_h' -> -d_rho_h d_rho_h' on (entry)/q^2.  Phi1
and Phi2 are the Mindlin-type terms of Rongved's solution.

Checks: the spatial form against direct 2-D Fourier integration of the spectral kernel (measured 7e-13,
5e-13, 1.3e-12 at three points, one near touching); and, when Mathematica/StaticImageSpace.json exists, the
two derivations against each other at its test points.

Run:  PYTHONPATH=. python scripts/derive_static_image_space.py
"""

import json
from pathlib import Path

import numpy as np
import sympy as sp
from numpy.polynomial.legendre import leggauss

ROOT = Path(__file__).resolve().parent.parent
J = json.loads((ROOT / "Mathematica" / "StaticInterfaceImage.json").read_text())
q = sp.Symbol("q", positive=True)
z, zp = sp.symbols("z zp", real=True)
lamA, muA, lamB, muB = sp.symbols("lamA muA lamB muB", positive=True)
loc = {
    "q": q,
    "z": z,
    "zp": zp,
    "lamA": lamA,
    "muA": muA,
    "lamB": lamB,
    "muB": muB,
    "E": sp.E,
    "I": sp.I,
}
ent = [
    [sp.sympify(s.replace("^", "**"), locals=loc) for s in row] for row in J["entries"]
]
# frame entries (order z, x', y'): a=zz, b=z x', c=x' z, d=x'x', e=y'y'
a, b, c, d, e = ent[0][0], ent[0][1], ent[1][0], ent[1][1], ent[2][2]
X = sp.exp(q * (z + zp))


def qpoly(f):  # f = X * sum_n p_n q^(n-1)  ->  {n-1: p_n}
    g = sp.expand(sp.simplify(f / X * q))
    P = sp.Poly(g, q)
    return {mon[0] - 1: coef for mon, coef in zip(P.monoms(), P.coeffs())}


rx, ry, zeta = sp.symbols("rho_x rho_y zeta", real=True)
R = sp.sqrt(rx**2 + ry**2 + zeta**2)
Phi = {
    0: 1 / (2 * sp.pi * R),
    1: -sp.log(R + zeta) / (2 * sp.pi),
    2: (zeta * sp.log(R + zeta) - R) / (2 * sp.pi),
}


def T(pw):  # spatial transform of q^pw e^{-q zeta}
    if pw >= -1:
        return sp.diff(Phi[0], zeta, pw + 1) * (-1) ** (pw + 1)
    return Phi[-1 - pw]


def Tsum(dct, shift=0):
    return sum(coef * T(pw + shift) for pw, coef in dct.items())


dh = {0: rx, 1: ry}
pa, pb, pc, pd, pe = map(qpoly, (a, b, c, d, e))
pdm = {k: pd.get(k, 0) - pe.get(k, 0) for k in set(pd) | set(pe)}
G = sp.zeros(3, 3)
G[0, 0] = Tsum(pa)
for hh in (1, 2):
    G[0, hh] = sp.diff(Tsum({k: v / sp.I for k, v in pb.items()}, -1), dh[hh - 1])
    G[hh, 0] = sp.diff(Tsum({k: v / sp.I for k, v in pc.items()}, -1), dh[hh - 1])
    for h2 in (1, 2):
        G[hh, h2] = (Tsum(pe) if hh == h2 else 0) - sp.diff(
            Tsum(pdm, -2), dh[hh - 1], dh[h2 - 1]
        )
pars = {lamA: 1.7, muA: 1.1, lamB: 2.6, muB: 1.9}
fs = sp.lambdify((q, z, zp), sp.Matrix(ent).subs(pars), "numpy")
gs = sp.lambdify((rx, ry, zeta, z, zp), G.subs(pars), "numpy")
for zz, zzp, px, py in [
    (-0.3, -0.5, 0.4, -0.2),
    (-0.1, -0.05, 0.02, 0.03),
    (-0.7, -0.2, 1.1, 0.6),
]:
    zt = -(zz + zzp)
    qmax = 60 / zt
    x, w = leggauss(400)
    qs = (x + 1) / 2 * qmax
    wq = w * qmax / 2
    nphi = 256
    phis = 2 * np.pi * np.arange(nphi) / nphi
    tot = np.zeros((3, 3), complex)
    for qq, ww in zip(qs, wq):
        F = np.array(fs(qq, zz, zzp), complex)
        for ph in phis:
            cc, ss = np.cos(ph), np.sin(ph)
            Q = np.array([[1, 0, 0], [0, cc, -ss], [0, ss, cc]])
            tot += (
                ww
                * qq
                * (2 * np.pi / nphi)
                / (4 * np.pi**2)
                * np.exp(1j * qq * (cc * px + ss * py))
                * (Q @ F @ Q.T)
            )
    gg = np.array(gs(px, py, zt, zz, zzp), complex)
    print((zz, zzp, px, py), f"rel err {np.abs(tot - gg).max() / np.abs(gg).max():.2e}")

wl = ROOT / "Mathematica" / "StaticImageSpace.json"
if wl.exists():
    m = json.loads(wl.read_text())
    f = sp.lambdify((rx, ry, zeta, z, zp, lamA, muA, lamB, muB), G, "numpy")
    err = 0.0
    for t, re, im in zip(m["tests"], m["values_re"], m["values_im"], strict=True):
        mine = np.array(f(*t), dtype=complex)
        theirs = np.array(re) + 1j * np.array(im)
        err = max(err, np.abs(mine - theirs).max() / np.abs(theirs).max())
    print(f"against Mathematica: {err:.1e}", "PASS" if err < 1e-12 else "FAIL")
else:
    print(
        "Mathematica/StaticImageSpace.json not found: run the .wl to compare (SKIPPED, not PASS)"
    )
