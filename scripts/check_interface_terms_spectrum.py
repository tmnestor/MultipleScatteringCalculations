"""Check: the canonical terms of the interface kernels against their spectrum, entry by entry.

Plan: docs/2026-10-10-stratified-reference-legendre-cells-3d.md (12 October: K1 in space, the families).

The terms (scripts/derive_interface_image_terms.py, order 0 or 1) are c z^A zp^B d^alpha Phi_j(rho, zeta).
Each has an exact lateral-wavenumber transform, (1/4 pi^2) int e^{i k.rho} F d^2k = term, with
    d_rho_h -> i k_h,   Phi_j -> q^(-1-j) e^{-q zeta} (j >= 1),   d_zeta^n Phi0 -> (-q)^n q^-1 e^{-q zeta},
valid for q > 0 whether or not the term's spatial form needs a finite part (that concerns q -> 0 only).
The reference is built without the term machinery: the 3 x 3 frame kernel (Mathematica's G0 or G1) rotated to
fixed axes by Q(phi) G Q(phi)^T, then the receiver derivatives (d_x, d_y -> i k_x, i k_y; d_z explicit) and
the source derivatives (-d'_x, -d'_y -> i k_x, i k_y; -d'_z explicit) of the 9 x 9 point propagator (rows u and
engineering strain, columns force and engineering stress, the stress columns halved for x != y).

At random wavenumbers, depths and rational media, every one of the 81 entries; error relative to the largest
entry of the reference.  Expected: rounding (the comparison is in 30-digit arithmetic).

Run:  PYTHONPATH=. python scripts/check_interface_terms_spectrum.py [reflected | transmitted] [0 | 1]
"""

import json
import sys
from pathlib import Path

import mpmath as mp
import sympy as sp
from sympy.parsing.mathematica import parse_mathematica

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "scripts" / "data"
KIND = sys.argv[1] if len(sys.argv) > 1 else "reflected"
ORDER = int(sys.argv[2]) if len(sys.argv) > 2 else 0
VP = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]

q, z, zp = (
    sp.Symbol("q", positive=True),
    sp.Symbol("z", real=True),
    sp.Symbol("zp", real=True),
)
NAMES = ("lamA", "muA", "lamB", "muB", "rhoA", "rhoB")
SYM = {n: sp.Symbol(n, positive=True) for n in NAMES}
#: rational media (alpha, beta, rho) = (5, 3, 5/2) above, (13/2, 18/5, 14/5) below
MEDIA = {}
for tag, (al, be, rho) in (
    ("A", (5, 3, sp.Rational(5, 2))),
    ("B", (sp.Rational(13, 2), sp.Rational(18, 5), sp.Rational(14, 5))),
):
    MEDIA[SYM["lam" + tag]] = rho * (al**2 - 2 * be**2)
    MEDIA[SYM["mu" + tag]] = rho * be**2
    MEDIA[SYM["rho" + tag]] = rho


def frame_kernel() -> sp.Matrix:
    """The 3 x 3 frame kernel (rows u_(z,x,y), columns force (z,x,y)), x along k, the media substituted."""
    loc = {"q": q, "z": z, "zp": zp, **SYM}
    if ORDER == 0:
        src = (
            ROOT
            / "Mathematica"
            / (
                "StaticInterfaceImage.json"
                if KIND == "reflected"
                else "StaticInterfaceTransmission.json"
            )
        )
        if not src.exists():
            src = DATA / "static_interface_transmission.json"
        rows = json.loads(src.read_text())["entries"]
        G = sp.Matrix(
            [
                [
                    sp.sympify(
                        e.replace("^", "**"), locals={**loc, "E": sp.E, "I": sp.I}
                    )
                    for e in r
                ]
                for r in rows
            ]
        )
    else:
        rows = json.loads((ROOT / "Mathematica" / "InterfaceOmega2.json").read_text())[
            "R1" if KIND == "reflected" else "T1"
        ]
        subs = {sp.Symbol(k): v for k, v in loc.items()}
        G = sp.Matrix([[parse_mathematica(e).subs(subs) for e in r] for r in rows])
    return G.subs(MEDIA)


def reference(G: sp.Matrix, kx, ky) -> list:
    """9 x 9 spectrum at (kx, ky) as sympy expressions in z, zp (no term machinery)."""
    qv = sp.sqrt(kx**2 + ky**2)
    c, s = kx / qv, ky / qv
    Q = sp.Matrix([[1, 0, 0], [0, c, -s], [0, s, c]])
    Gf = (Q * G.subs(q, qv) * Q.T).applyfunc(sp.expand)
    k = [None, sp.I * kx, sp.I * ky]

    def drec(e, i):
        return sp.diff(e, z) if i == 0 else k[i] * e

    def dsrc(e, i):  # -d'
        return -sp.diff(e, zp) if i == 0 else k[i] * e

    def col(gi, cc):
        if cc < 3:
            return gi[cc]
        m, n = VP[cc - 3]
        return dsrc(gi[m], n) if m == n else (dsrc(gi[m], n) + dsrc(gi[n], m)) / 2

    def row(r):
        if r < 3:
            return [Gf[r, kk] for kk in range(3)]
        p, t = VP[r - 3]
        if p == t:
            return [drec(Gf[p, kk], p) for kk in range(3)]
        return [drec(Gf[t, kk], p) + drec(Gf[p, kk], t) for kk in range(3)]

    return [[col(row(r), cc) for cc in range(9)] for r in range(9)]


def from_terms(entries: list, kx, ky) -> list:
    """9 x 9 spectrum at (kx, ky) from the canonical terms."""
    qv = sp.sqrt(kx**2 + ky**2)
    zeta = -(z + zp) if KIND == "reflected" else z - zp
    out = []
    for r in range(9):
        row_ = []
        for cc in range(9):
            acc = 0
            for t in entries[r][cc]:
                j, (ax, ay, az) = t["j"], t["alpha"]
                coef = sp.sympify(t["coef"], locals=SYM).subs(MEDIA)
                if j == 0:
                    pot = (-qv) ** az / qv
                else:
                    assert az == 0, t
                    pot = qv ** (-1 - j)
                acc += (
                    coef
                    * z ** t["z_power"]
                    * zp ** t["zp_power"]
                    * (sp.I * kx) ** ax
                    * (sp.I * ky) ** ay
                    * pot
                    * sp.exp(-qv * zeta)
                )
            row_.append(acc)
        out.append(row_)
    return out


def main() -> None:
    name = (
        (
            "interface_image_terms"
            if KIND == "reflected"
            else "interface_transmission_terms"
        )
        if ORDER == 0
        else f"interface_omega2_terms_{KIND}"
    )
    entries = json.loads((DATA / f"{name}.json").read_text())["terms"]
    G = frame_kernel()
    mp.mp.dps = 30
    worst = 0.0
    # depths: both in A (z, zp < 0) when reflected; receiver in B (z > 0) when transmitted
    for kx, ky, zv, zpv in (
        (
            sp.Rational(7, 3),
            sp.Rational(-5, 4),
            sp.Rational(-3, 10),
            sp.Rational(-1, 5),
        ),
        (
            sp.Rational(-1, 2),
            sp.Rational(11, 5),
            sp.Rational(-7, 10),
            sp.Rational(-2, 5),
        ),
        (
            sp.Rational(9, 4),
            sp.Rational(1, 3),
            sp.Rational(-1, 10),
            sp.Rational(-9, 10),
        ),
    ):
        if KIND == "transmitted":
            zv = -zv
        ref = reference(G, kx, ky)
        got = from_terms(entries, kx, ky)
        at = {z: zv, zp: zpv}
        rv = [[complex(sp.N(e.subs(at), 30)) for e in r] for r in ref]
        gv = [[complex(sp.N(e.subs(at), 30)) for e in r] for r in got]
        scale = max(abs(v) for r in rv for v in r)
        err = max(
            abs(a - b)
            for ra, rb in zip(rv, gv, strict=True)
            for a, b in zip(ra, rb, strict=True)
        )
        worst = max(worst, err / scale)
        print(
            f"  k = ({kx}, {ky}), z = {zv}, zp = {zpv}: {err / scale:.1e}", flush=True
        )
    print(
        f"{KIND}, order {ORDER}: 81 entries, worst {worst:.1e}  {'PASS' if worst < 1e-13 else 'FAIL'}"
    )


if __name__ == "__main__":
    main()
