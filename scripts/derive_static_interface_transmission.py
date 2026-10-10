"""Twin of Mathematica/StaticInterfaceTransmission.wl: the elastostatic field transmitted through a welded interface.

Medium A fills z < 0, medium B z > 0, welded at z = 0; a unit point force in A at depth zp < 0, the receiver in
B at z > 0.  As for the reflected image (derive_static_interface_image.py): in the frame whose x-axis lies
along the lateral wavevector (q, 0), Kelvin's field in A plus a reflected part, a transmitted part in B, u and
the z-plane traction continuous at z = 0.  The TRANSMITTED part GT(q; z, zp), 3 x 3 (rows u, columns force,
order z, x, y), is the whole static coupling between a source in A and a receiver in B.  Every entry is
e^{-q (z - zp)} / q times a polynomial in q z, q zp: the same three potentials as the image, with
zeta = z - zp > 0.

Checks: for identical media GT is Kelvin's field (z > zp).  Writes scripts/data/static_interface_transmission.json
and, when Mathematica/StaticInterfaceTransmission.json exists, compares the two derivations.

Run:  PYTHONPATH=. python scripts/derive_static_interface_transmission.py
"""

import json
from pathlib import Path

import numpy as np
import sympy as sp

z, zp = sp.symbols("z zp", real=True)
q = sp.Symbol("q", positive=True)
lamA, muA, lamB, muB = sp.symbols("lamA muA lamB muB", positive=True)


def stresses(u, lam, mu):
    ux, uy, uz = u["x"], u["y"], u["z"]  # frame: e^{i q x}, no y dependence
    dx = lambda f: sp.I * q * f
    dz = lambda f: sp.diff(f, z)
    szz = (lam + 2 * mu) * dz(uz) + lam * dx(ux)
    sxz = mu * (dz(ux) + dx(uz))
    syz = mu * dz(uy)
    sxx = lam * dz(uz) + (lam + 2 * mu) * dx(ux)
    sxy = mu * dx(uy)
    return {"zz": szz, "xz": sxz, "yz": syz, "xx": sxx, "xy": sxy}


def navier_ok(u, lam, mu):
    s = stresses(u, lam, mu)
    eqx = sp.I * q * s["xx"] + sp.diff(s["xz"], z)
    eqz = sp.I * q * s["xz"] + sp.diff(s["zz"], z)
    eqy = sp.I * q * s["xy"] + sp.diff(s["yz"], z)
    return [sp.simplify(e) for e in (eqx, eqy, eqz)]


def basis(lam, mu, s):
    """general solution ~ e^{s q z}, s = +-1: P-SV two-parameter, SH one."""
    a, b, c, d = sp.symbols("a b c d")
    u = {
        "x": (a + b * z) * sp.exp(s * q * z),
        "z": (c + d * z) * sp.exp(s * q * z),
        "y": 0,
    }
    eqs = navier_ok(u, lam, mu)
    eqs = [sp.expand(sp.simplify(e * sp.exp(-s * q * z))) for e in eqs]
    coeffs = []
    for e in eqs:
        coeffs += sp.Poly(e, z).coeffs()
    sol = sp.solve(coeffs, [c, d], dict=True)[0]
    sols = []
    for pa, pb in ((1, 0), (0, 1)):
        uu = {
            k: (sp.simplify(v.subs(sol).subs({a: pa, b: pb})) if v != 0 else 0)
            for k, v in u.items()
        }
        sols.append(uu)
    sh = {"x": 0, "z": 0, "y": sp.exp(s * q * z)}
    return sols, sh


def traction(u, lam, mu):
    s = stresses(u, lam, mu)
    return {"x": s["xz"], "y": s["yz"], "z": s["zz"]}


def combo(sols, sh, cs):
    out = {"x": 0, "y": 0, "z": 0}
    for c, u in zip(cs, sols + [sh]):
        for k in out:
            out[k] += c * u[k]
    return out


comps = ("x", "y", "z")
GT, GK = {}, {}
for f in comps:
    F = {k: (1 if k == f else 0) for k in comps}
    cb = sp.symbols("cb0:3")
    ca = sp.symbols("ca0:3")
    cr = sp.symbols("cr0:3")
    ct = sp.symbols("ct0:3")
    solsD, shD = basis(lamA, muA, -1)
    solsU, shU = basis(lamA, muA, 1)
    below = combo(solsD, shD, cb)
    above = combo(solsU, shU, ca)
    eqs = []
    tb, ta = traction(below, lamA, muA), traction(above, lamA, muA)
    for k in comps:
        eqs.append((below[k] - above[k]).subs(z, zp))
        eqs.append((tb[k] - ta[k]).subs(z, zp) + F[k])
    kel = sp.solve(eqs, list(cb) + list(ca), dict=True)[0]
    uK = {k: below[k].subs(kel) for k in comps}
    R_ = combo(solsU, shU, cr)
    solsB, shB = basis(lamB, muB, -1)
    T_ = combo(solsB, shB, ct)
    tK, tR, tT = (
        traction(uK, lamA, muA),
        traction(R_, lamA, muA),
        traction(T_, lamB, muB),
    )
    eqs = []
    for k in comps:
        eqs.append((uK[k] + R_[k] - T_[k]).subs(z, 0))
        eqs.append((tK[k] + tR[k] - tT[k]).subs(z, 0))
    rs = sp.solve(eqs, list(cr) + list(ct), dict=True)[0]
    for k in comps:
        GT[(k, f)] = sp.simplify(T_[k].subs(rs))
        GK[(k, f)] = sp.simplify(uK[k])

idx = {"z": 0, "x": 1, "y": 2}
G = sp.zeros(3, 3)
K = sp.zeros(3, 3)
for (k, f), v in GT.items():
    G[idx[k], idx[f]] = v
    K[idx[k], idx[f]] = GK[(k, f)]

print(
    "identical media give Kelvin's field:",
    sp.simplify(G.subs({lamB: lamA, muB: muA}) - K) == sp.zeros(3, 3),
)
X = sp.exp(-q * (z - zp))
print(
    "entries are e^{-q(z - zp)}/q times a polynomial:",
    all(sp.simplify(e / X * q).is_polynomial(q, z, zp) for e in G if e != 0),
)

TESTS = [(2.3, 0.4, -0.9, 1.7, 1.1, 2.6, 1.9), (11.0, 0.05, -0.12, 0.6, 0.9, 1.4, 2.2)]
fn = sp.lambdify((q, z, zp, lamA, muA, lamB, muB), G, "numpy")
vals = [np.array(fn(*t), dtype=complex) for t in TESTS]
root = Path(__file__).resolve().parent
out = root / "data" / "static_interface_transmission.json"
out.write_text(
    json.dumps(
        {
            "order": "rows u_(z,x,y), columns force (z,x,y); frame x along the lateral wavevector; source in A (zp < 0), receiver in B (z > 0)",
            "entries": [[str(G[i, j]) for j in range(3)] for i in range(3)],
            "tests": TESTS,
            "values_re": [v.real.tolist() for v in vals],
            "values_im": [v.imag.tolist() for v in vals],
        },
        indent=1,
    )
)
print("written", out.relative_to(root.parent))
wl = root.parent / "Mathematica" / "StaticInterfaceTransmission.json"
if wl.exists():
    m = json.loads(wl.read_text())
    mv = [
        np.array(r) + 1j * np.array(i)
        for r, i in zip(m["values_re"], m["values_im"], strict=True)
    ]
    err = max(
        np.abs(a - b).max() / np.abs(b).max() for a, b in zip(vals, mv, strict=True)
    )
    print(f"against Mathematica: {err:.1e}", "PASS" if err < 1e-12 else "FAIL")
else:
    print(
        "Mathematica/StaticInterfaceTransmission.json not found: run the .wl to compare (SKIPPED, not PASS)"
    )
