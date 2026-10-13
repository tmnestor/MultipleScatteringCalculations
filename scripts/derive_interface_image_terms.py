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

ORDER 1: the omega^2 term G1 (Mathematica/InterfaceOmega2.json, keys R1 and T1; densities enter), written to
scripts/data/interface_omega2_terms_{reflected,transmitted}.json.  Its lowest powers q^-4, q^-5 give
Phi3 and Phi4 (Phi_j for q^(-1-j)), whose transforms exist only as finite parts.  A term with n_lat lateral
derivatives has an integrand ~ q^(n_lat - 1 - j) at q -> 0, integrable in the plane (no finite part) iff
n_lat >= j; the summary printed at the end lists every family and flags those with n_lat < j.

MEDIA (optional third argument): "symbolic" (default) keeps the Lame constants and densities as symbols;
"media1" or "media2" substitutes exact rational media first, (alpha, beta, rho) = (5, 3, 5/2) over
(13/2, 18/5, 14/5), or (4, 9/4, 2) over (7, 4, 3).  The algebra is then far lighter (minutes, not hours, at
order 1), and the families are those of the symbolic kernel unless a coefficient happens to vanish for those
media, which two unrelated media guard against.  Written to ..._{media}.json.

Run:  PYTHONPATH=. python scripts/derive_interface_image_terms.py [reflected | transmitted] [0 | 1] [media]
"""

import json
import os
import sys
from pathlib import Path

import sympy as sp

ROOT = Path(__file__).resolve().parent.parent
#: "reflected" (receiver and source in A, zeta = -(z + zp)) or "transmitted" (source in A, receiver in B,
#: zeta = z - zp).  The transmitted spectrum is read from Mathematica's output when it exists, else from the
#: sympy twin's (scripts/derive_static_interface_transmission.py).
KIND = sys.argv[1] if len(sys.argv) > 1 else "reflected"
ORDER = int(sys.argv[2]) if len(sys.argv) > 2 else 0
if ORDER not in (0, 1):
    raise SystemExit(f"order must be 0 or 1, got {ORDER}")
if KIND not in ("reflected", "transmitted"):
    raise SystemExit(f"kind must be 'reflected' or 'transmitted', got {KIND!r}")
MEDIA = sys.argv[3] if len(sys.argv) > 3 else "symbolic"
MEDIA_SETS = {
    "media1": (
        (5, 3, sp.Rational(5, 2)),
        (sp.Rational(13, 2), sp.Rational(18, 5), sp.Rational(14, 5)),
    ),
    "media2": ((4, sp.Rational(9, 4), 2), (7, 4, 3)),
}
if MEDIA != "symbolic" and MEDIA not in MEDIA_SETS:
    raise SystemExit(f"media must be 'symbolic', 'media1' or 'media2', got {MEDIA!r}")
if ORDER == 1:
    SOURCE = ROOT / "Mathematica" / "InterfaceOmega2.json"
    OUT = ROOT / "scripts" / "data" / f"interface_omega2_terms_{KIND}.json"
elif KIND == "reflected":
    SOURCE = ROOT / "Mathematica" / "StaticInterfaceImage.json"
    OUT = ROOT / "scripts" / "data" / "interface_image_terms.json"
else:
    SOURCE = ROOT / "Mathematica" / "StaticInterfaceTransmission.json"
    if not SOURCE.exists():
        SOURCE = ROOT / "scripts" / "data" / "static_interface_transmission.json"
    OUT = ROOT / "scripts" / "data" / "interface_transmission_terms.json"
if MEDIA != "symbolic":
    OUT = OUT.with_name(f"{OUT.stem}_{MEDIA}.json")
J = json.loads(SOURCE.read_text())
q = sp.Symbol("q", positive=True)
z, zp = sp.symbols("z zp", real=True)
lamA, muA, lamB, muB = sp.symbols("lamA muA lamB muB", positive=True)
rhoA, rhoB = sp.symbols("rhoA rhoB", positive=True)
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
if ORDER == 1:
    from sympy.parsing.mathematica import parse_mathematica

    subs = {
        sp.Symbol(k): v
        for k, v in {**loc, "rhoA": rhoA, "rhoB": rhoB}.items()
        if k not in ("E", "I")
    }
    ent = [
        [parse_mathematica(s).subs(subs) for s in row]
        for row in J["R1" if KIND == "reflected" else "T1"]
    ]
else:
    ent = [
        [sp.sympify(s.replace("^", "**"), locals=loc) for s in row]
        for row in J["entries"]
    ]
if MEDIA != "symbolic":
    vals = {}
    for tag, (al, be, rh) in zip("AB", MEDIA_SETS[MEDIA], strict=True):
        vals[{"A": lamA, "B": lamB}[tag]] = rh * (al**2 - 2 * be**2)
        vals[{"A": muA, "B": muB}[tag]] = rh * be**2
        vals[{"A": rhoA, "B": rhoB}[tag]] = rh
    ent = [[e_.subs(vals) for e_ in row_] for row_ in ent]
a, b, c, d, e = ent[0][0], ent[0][1], ent[1][0], ent[1][1], ent[2][2]
X = sp.exp(q * (z + zp)) if KIND == "reflected" else sp.exp(-q * (z - zp))


#: Powers of q down to q^-(N_SHIFT) are admitted (G0 has q^-1, G1 lower).
N_SHIFT = 1 if ORDER == 0 else 8


def qpoly(f):
    P = sp.Poly(sp.expand(sp.simplify(f / X * q) * q ** (N_SHIFT - 1)), q)
    return {m[0] - N_SHIFT: co for m, co in zip(P.monoms(), P.coeffs())}


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


#: Optional checkpoint directory (environment DERIVE_CHECKPOINT): each reduced entry is written there as it is
#: finished (sympy srepr) and read back on a rerun, so a long symbolic run survives a restart.
CHECKPOINT = os.environ.get("DERIVE_CHECKPOINT")


def reduced(r: int, cc: int) -> dict:
    if CHECKPOINT is None:
        return canon(entry(r, cc))
    path = Path(CHECKPOINT) / f"{KIND}_order{ORDER}_{MEDIA}_{r}_{cc}.txt"
    if path.exists():
        return {
            tuple(json.loads(k)): sp.sympify(v)
            for k, v in json.loads(path.read_text()).items()
        }
    out = canon(entry(r, cc))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps({json.dumps(list(k)): sp.srepr(v) for k, v in out.items()})
    )
    tmp.replace(path)  # atomic: a restart never sees a half-written entry
    print(f"entry ({r}, {cc}) done", flush=True)
    return out


K = [[reduced(r, cc) for cc in range(9)] for r in range(9)]

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
variables = ["lamA", "muA", "lamB", "muB"] + (["rhoA", "rhoB"] if ORDER == 1 else [])
path.write_text(json.dumps({"variables": variables, "terms": out}, indent=0))
n = sum(len(t) for row in out for t in row)
print(f"written {path.relative_to(ROOT)}: {n} terms")

# the families (j, alpha) and the lateral derivatives each potential occurs under
fam: dict = {}
for row_ in out:
    for terms_ in row_:
        for t in terms_:
            key = (t["j"], *t["alpha"])
            fam[key] = fam.get(key, 0) + 1
print(f"{len(fam)} families (j, alpha):")
for j in sorted({k[0] for k in fam}):
    lat = sorted({k[1] + k[2] for k in fam if k[0] == j})
    az = sorted({k[3] for k in fam if k[0] == j})
    nt = sum(v for k, v in fam.items() if k[0] == j)
    bad = [k for k in fam if k[0] == j and j > 0 and k[1] + k[2] < j]
    nb = sum(fam[k] for k in bad)
    print(
        f"  Phi{j}: {nt} terms in {sum(1 for k in fam if k[0] == j)} families; lateral derivatives {lat},"
        f" zeta derivatives {az}"
        + (
            f"; finite part needed (n_lat < {j}): {nb} terms in {len(bad)} families {sorted(bad)}"
            if bad
            else ""
        )
    )
