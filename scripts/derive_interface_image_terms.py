"""Canonical terms of the 9 x 9 static interface image: poly(z, zp) x d^alpha Phi_j.

Reads Mathematica/StaticInterfaceImage.json (the reflected static kernel of a welded interface in the lateral-
wavenumber domain, from StaticInterfaceImage.wl) and writes scripts/data/interface_image_terms.json: for every
entry (row, column) of the 9 x 9 kernel (rows u, engineering strain; columns force, engineering stress, as
graded_voxel.kernel.kernel_9x9), a list of terms

    coefficient(z, zp; lamA, muA, lamB, muB)  x  d_rho_x^ax d_rho_y^ay d_zeta^az Phi_j(rho_x, rho_y, zeta),

with zeta = -(z + zp), rho = x - x', and Phi0 = 1/(2 pi R), Phi1 = -log(R + zeta)/(2 pi),
Phi2 = (zeta log(R + zeta) - R)/(2 pi), R^2 = rho^2 + zeta^2 (see derive_static_image_space.py).  A zeta
derivative of Phi1 or Phi2 is reduced (d_zeta Phi1 = -Phi0, d_zeta Phi2 = -Phi1), so the Mindlin potentials
appear only under lateral derivatives.  Receiver derivatives: d_x = d_rho_x, d_z = d/dz - d_zeta; source
columns carry -d': -d'_x = d_rho_x, -d'_z = -d/dzp + d_zeta.

Checked: the 81 entries against direct differentiation of the spatial kernel, 3e-15.

Run:  PYTHONPATH=. python scripts/derive_interface_image_terms.py [reflected | transmitted]
"""

import json
import sys
from pathlib import Path

import sympy as sp

ROOT = Path(__file__).resolve().parent.parent
#: "reflected" (receiver and source in A, zeta = -(z + zp)) or "transmitted" (source in A, receiver in B,
#: zeta = z - zp).  The transmitted spectrum is read from Mathematica's output when it exists, else from the
#: sympy twin's (scripts/derive_static_interface_transmission.py).
KIND = sys.argv[1] if len(sys.argv) > 1 else "reflected"
if KIND == "reflected":
    SOURCE = ROOT / "Mathematica" / "StaticInterfaceImage.json"
    OUT = ROOT / "scripts" / "data" / "interface_image_terms.json"
elif KIND == "transmitted":
    SOURCE = ROOT / "Mathematica" / "StaticInterfaceTransmission.json"
    if not SOURCE.exists():
        SOURCE = ROOT / "scripts" / "data" / "static_interface_transmission.json"
    OUT = ROOT / "scripts" / "data" / "interface_transmission_terms.json"
else:
    raise SystemExit(f"kind must be 'reflected' or 'transmitted', got {KIND!r}")
J = json.loads(SOURCE.read_text())
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
a, b, c, d, e = ent[0][0], ent[0][1], ent[1][0], ent[1][1], ent[2][2]
X = sp.exp(q * (z + zp)) if KIND == "reflected" else sp.exp(-q * (z - zp))


def qpoly(f):
    P = sp.Poly(sp.expand(sp.simplify(f / X * q)), q)
    return {m[0] - 1: co for m, co in zip(P.monoms(), P.coeffs())}


# a term: dict key (j, ax, ay, az) -> coefficient (poly in z, zp, materials)
def T(pw):  # q^pw e^{-q zeta} -> {(j, 0,0,az): coef}
    if pw >= -1:
        return {(0, 0, 0, pw + 1): (-1) ** (pw + 1)}
    return {(-1 - pw, 0, 0, 0): 1}


def add(acc, t, s=1):
    for k, v in t.items():
        acc[k] = acc.get(k, 0) + s * v
    return acc


def scale(t, s):
    return {k: v * s for k, v in t.items()}


def dlat(t, h):  # lateral derivative h in {0: x, 1: y}
    return {(k[0], k[1] + (h == 0), k[2] + (h == 1), k[3]): v for k, v in t.items()}


def tsum(dct, shift=0):
    acc = {}
    for pw, co in dct.items():
        add(acc, scale(T(pw + shift), co))
    return acc


pa, pb, pc, pd, pe = map(qpoly, (a, b, c, d, e))
pdm = {k: pd.get(k, 0) - pe.get(k, 0) for k in set(pd) | set(pe)}
G = [[{} for _ in range(3)] for _ in range(3)]
G[0][0] = tsum(pa)
for hh in (1, 2):
    G[0][hh] = dlat(tsum({k: v / sp.I for k, v in pb.items()}, -1), hh - 1)
    G[hh][0] = dlat(tsum({k: v / sp.I for k, v in pc.items()}, -1), hh - 1)
    for h2 in (1, 2):
        G[hh][h2] = add(
            tsum(pe) if hh == h2 else {}, dlat(dlat(tsum(pdm, -2), hh - 1), h2 - 1), -1
        )


# derivative operators on terms: receiver d_x, d_y, d_z (= d/dz explicit - d/dzeta); source -d'_x, -d'_y, -d'_z (= -d/dzp + d/dzeta)
def dzeta(t):
    out = {}
    for (j, ax, ay, az), v in t.items():
        out[(j, ax, ay, az + 1)] = out.get((j, ax, ay, az + 1), 0) + v
    return out


def d_rec(t, i):
    if i == 1:
        return dlat(t, 0)
    if i == 2:
        return dlat(t, 1)
    # d/dz of zeta: -1 for the image (zeta = -(z + zp)), +1 across the interface (zeta = z - zp)
    return add(
        {k: sp.diff(v, z) for k, v in t.items()},
        dzeta(t),
        -1 if KIND == "reflected" else 1,
    )


def d_src(t, i):  # -d'
    if i == 1:
        return dlat(t, 0)
    if i == 2:
        return dlat(t, 1)
    return add({k: -sp.diff(v, zp) for k, v in t.items()}, dzeta(t), 1)


def canon(t):
    """reduce d_zeta on Phi_j (j>0): d_zeta Phi1 = -Phi0, d_zeta Phi2 = -Phi1; drop zeros."""
    out = {}
    for (j, ax, ay, az), v in t.items():
        while az > 0 and j > 0:
            j, az, v = j - 1, az - 1, -v
        out[(j, ax, ay, az)] = out.get((j, ax, ay, az), 0) + v
    return {k: sp.factor(sp.expand(v)) for k, v in out.items() if sp.simplify(v) != 0}


VP = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


def column(gi: list, cc: int) -> dict:
    """Column cc (force, then engineering stress) of a displacement row given by its force components gi."""
    if cc < 3:
        return gi[cc]
    m, n = VP[cc - 3]
    if m == n:
        return d_src(gi[m], n)
    return scale(add(d_src(gi[m], n), d_src(gi[n], m)), sp.Rational(1, 2))


def row(r: int) -> list:
    """Row r (displacement, then engineering strain) as its three force components."""
    if r < 3:
        return G[r]
    p, s_ = VP[r - 3]
    if p == s_:
        return [d_rec(G[p][k], p) for k in range(3)]
    return [add(d_rec(G[s_][k], p), d_rec(G[p][k], s_)) for k in range(3)]


def entry(r: int, cc: int) -> dict:
    """The (r, cc) entry; the source derivative commutes with the receiver one."""
    if r < 3 or cc < 3:
        return column(row(r), cc)
    p, s_ = VP[r - 3]

    def strain_of(kk: int, der: int) -> dict:
        if p == s_:
            return d_rec(d_src(G[p][kk], der), p)
        return add(d_rec(d_src(G[s_][kk], der), p), d_rec(d_src(G[p][kk], der), s_))

    m, n = VP[cc - 3]
    if m == n:
        return strain_of(m, n)
    return scale(add(strain_of(m, n), strain_of(n, m)), sp.Rational(1, 2))


K = [[canon(entry(r, cc)) for cc in range(9)] for r in range(9)]

out = []
for r in range(9):
    out_row = []
    for cc in range(9):
        terms = []
        for (j, ax, ay, az), v in sorted(K[r][cc].items()):
            poly = sp.Poly(sp.expand(v), z, zp)
            for (pa_, pb_), co in zip(poly.monoms(), poly.coeffs(), strict=True):
                terms.append(
                    {
                        "j": j,
                        "alpha": [ax, ay, az],
                        "z_power": pa_,
                        "zp_power": pb_,
                        "coef": str(sp.simplify(co)),
                    }
                )
        out_row.append(terms)
    out.append(out_row)
path = OUT
path.write_text(
    json.dumps({"variables": ["lamA", "muA", "lamB", "muB"], "terms": out}, indent=0)
)
n = sum(len(t) for row in out for t in row)
print(f"written {path.relative_to(ROOT)}: {n} terms")
