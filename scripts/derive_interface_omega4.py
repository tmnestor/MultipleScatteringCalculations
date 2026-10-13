"""Derive: the O(omega^4) term of the reflected and transmitted spectral kernels of a welded interface.

Plan: docs/2026-10-10-stratified-reference-legendre-cells-3d.md (13 October: the remainder, its tail by omega^4).
Paper: LatexPDFs/InterfaceCouplingIntegrals, the omega^2 section.  Order omega^2: scripts/derive_interface_omega2.py
(whose building blocks this script reuses); Mathematica twin of this order: Mathematica/InterfaceOmega4.wl.

At large q, G(q; z, zp; omega) = G0 + omega^2 G1 + omega^4 G2 + O(omega^6).  The perturbation of the static problem
continues one order: div sigma(u_n) = -rho u_(n-1) with no source, u_n and its traction continuous at zp (Kelvin's
term) and at the interface (with the reflected and transmitted terms).  Each u_n is a polynomial two degrees higher
than u_(n-1) times the same exponential, plus the regular homogeneous solutions.

The materials are exact rationals (SymPy's solve with symbolic materials exceeds memory already at omega^2): by
default the media of the gates, A = (5, 3, 2.5) and B = (6.5, 3.6, 2.8) as (alpha, beta, rho), whose binary values
are converted exactly, as ``interface_spectral.frame_laurent`` does.  The symbolic result is the Mathematica twin's.

CHECKS: (G - G0 - omega^2 G1 - omega^4 G2) / omega^6 tends to a constant as omega -> 0 against the exact kernels at
50 digits (the residual falls 64-fold when omega halves; the omega^2 residual, 16-fold, for comparison); identical
media give R2 = 0 and T2 = Kelvin's omega^4 term; G0, G1 equal those of derive_interface_omega2.py's chain.

OUTPUT: scripts/data/interface_omega4_<media>.json: R2 and T2 (sympy strings in q, z, zp), the media, and the
powers of q of each monomial z^a zp^b (which potentials the spatial form would need).

Run:  PYTHONPATH=.:scripts python scripts/derive_interface_omega4.py [gate | omega2test]
"""

import json
import sys
import time
from pathlib import Path

import derive_interface_omega2 as d2
import mpmath as mp
import sympy as sp

z, zp, q = d2.z, d2.zp, d2.q
COMPS = d2.COMPS
ORDER = 2  # omega^(2 ORDER)


def _rational_media(alpha, beta, rho):
    al, be, rh = (sp.Rational(float(v)) for v in (alpha, beta, rho))
    return rh * (al**2 - 2 * be**2), rh * be**2, rh


MEDIA = {
    "gate": (_rational_media(5.0, 3.0, 2.5), _rational_media(6.5, 3.6, 2.8)),
    "omega2test": (
        tuple(list(d2.MATS.values())[:3]),
        tuple(list(d2.MATS.values())[3:]),
    ),
}


def solve_force(f, ma, mb, order=ORDER):
    """Lists (R_n, T_n, K_n), n = 0..order, of the columns for a unit force along f."""
    la, mua, rha = ma
    lb, mub, rhb = mb
    F = {k: sp.Integer(1 if k == f else 0) for k in COMPS}
    below = d2.homogeneous(la, mua, -1, sp.symbols("kb0:3"))
    above = d2.homogeneous(la, mua, 1, sp.symbols("ka0:3"))
    tb, ta = d2.traction(below, la, mua), d2.traction(above, la, mua)
    eqs = [(below[k] - above[k]).subs(z, zp) for k in COMPS]
    eqs += [(tb[k] - ta[k]).subs(z, zp) + F[k] for k in COMPS]
    sol = sp.solve(
        eqs, list(sp.symbols("kb0:3")) + list(sp.symbols("ka0:3")), dict=True
    )[0]
    kb = [{k: d2._clean(below[k].subs(sol)) for k in COMPS}]
    ka = [{k: d2._clean(above[k].subs(sol)) for k in COMPS}]
    R, T = [], []
    for n in range(order + 1):
        if n:
            nb = d2.add(
                d2.particular(kb[-1], la, mua, rha, -1),
                d2.homogeneous(la, mua, -1, sp.symbols(f"lb{n}_0:3")),
            )
            na = d2.add(
                d2.particular(ka[-1], la, mua, rha, 1),
                d2.homogeneous(la, mua, 1, sp.symbols(f"la{n}_0:3")),
            )
            tb, ta = d2.traction(nb, la, mua), d2.traction(na, la, mua)
            eqs = [(nb[k] - na[k]).subs(z, zp) for k in COMPS] + [
                (tb[k] - ta[k]).subs(z, zp) for k in COMPS
            ]
            unk = list(sp.symbols(f"lb{n}_0:3")) + list(sp.symbols(f"la{n}_0:3"))
            sol = sp.solve(eqs, unk, dict=True)[0]
            kb.append({k: d2._clean(nb[k].subs(sol)) for k in COMPS})
            ka.append({k: d2._clean(na[k].subs(sol)) for k in COMPS})
        r = d2.homogeneous(la, mua, 1, sp.symbols(f"r{n}_0:3"))
        t = d2.homogeneous(lb, mub, -1, sp.symbols(f"t{n}_0:3"))
        if n:
            r = d2.add(d2.particular(R[-1], la, mua, rha, 1), r)
            t = d2.add(d2.particular(T[-1], lb, mub, rhb, -1), t)
        tk, tr, tt = (
            d2.traction(kb[n], la, mua),
            d2.traction(r, la, mua),
            d2.traction(t, lb, mub),
        )
        eqs = [(kb[n][k] + r[k] - t[k]).subs(z, 0) for k in COMPS]
        eqs += [(tk[k] + tr[k] - tt[k]).subs(z, 0) for k in COMPS]
        unk = list(sp.symbols(f"r{n}_0:3")) + list(sp.symbols(f"t{n}_0:3"))
        sol = sp.solve(eqs, unk, dict=True)[0]
        R.append({k: d2._clean(r[k].subs(sol)) for k in COMPS})
        T.append({k: d2._clean(t[k].subs(sol)) for k in COMPS})
    return R, T, kb


def tensors(ma, mb, order=ORDER):
    """{"R": [G_0..], "T": [...], "K": [...]}: 3 x 3 sympy matrices in (q, z, zp), order (z, x, y)."""
    out = {name: [sp.zeros(3, 3) for _ in range(order + 1)] for name in ("R", "T", "K")}
    for f in COMPS:
        R, T, K = solve_force(f, ma, mb, order)
        for name, cols in (("R", R), ("T", T), ("K", K)):
            for n, col in enumerate(cols):
                for k in COMPS:
                    out[name][n][d2.IDX[k], d2.IDX[f]] = col[k]
    return out


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "gate"
    ma, mb = MEDIA[which]
    t0 = time.time()
    g = tensors(ma, mb)
    print(f"derived to omega^{2 * ORDER} ({time.time() - t0:.0f} s)", flush=True)
    same = tensors(ma, ma)
    print(
        "identical media: R_n = 0 for all n:",
        all(d2._clean(m) == sp.zeros(3, 3) for m in same["R"]),
    )
    print(
        "identical media: T_n = Kelvin's term for all n:",
        all(
            d2._clean(same["T"][n] - same["K"][n]) == sp.zeros(3, 3)
            for n in range(ORDER + 1)
        ),
    )

    mp.mp.dps = 50
    med_a, med_b = d2._med(*[float(v) for v in ma]), d2._med(*[float(v) for v in mb])
    for kind, (qq, zr, zs) in (("R", (3.0, -0.4, -0.7)), ("T", (3.0, 0.4, -0.7))):
        fns = [sp.lambdify((q, z, zp), m, "mpmath") for m in g[kind]]
        gs = [mp.matrix(fn(mp.mpf(qq), mp.mpf(zr), mp.mpf(zs))) for fn in fns]
        print(f"{kind}: q = {qq}, z = {zr}, zp = {zs}")
        prev2 = prev4 = None
        for om in (0.4, 0.2, 0.1, 0.05):
            ex = d2.dynamic(
                mp.mpf(qq), mp.mpf(om), mp.mpf(zr), mp.mpf(zs), med_a, med_b, kind
            )
            r2 = mp.mnorm(ex - gs[0] - om**2 * gs[1], 1) / mp.mnorm(gs[0], 1)
            r4 = mp.mnorm(ex - gs[0] - om**2 * gs[1] - om**4 * gs[2], 1) / mp.mnorm(
                gs[0], 1
            )
            rat = (
                ""
                if prev4 is None
                else f"  ratios {mp.nstr(prev2 / r2, 4)} (w^2), {mp.nstr(prev4 / r4, 4)} (w^4)"
            )
            print(
                f"  omega {om:5.2f}: residual after w^2 {mp.nstr(r2, 3)}, after w^4 {mp.nstr(r4, 3)}{rat}",
                flush=True,
            )
            prev2, prev4 = r2, r4

    powers = {}
    for kind in ("R", "T"):
        powers[kind] = [
            [d2.q_powers(g[kind][2][i, j], kind) for j in range(3)] for i in range(3)
        ]
        low = min(
            m for row in powers[kind] for dct in row for ms in dct.values() for m in ms
        )
        print(
            f"{kind}2: lowest power of q {low} (q^m e^-q zeta: Phi_j, j = -1 - m, so up to Phi{-1 - low})"
        )
    out = {
        "order": "rows u_(z,x,y), columns force (z,x,y); frame x along k; G = G0 + w^2 G1 + w^4 G2",
        "media": {
            "A": [str(v) for v in ma],
            "B": [str(v) for v in mb],
            "names": ["lam", "mu", "rho"],
        },
        "R2": [[str(g["R"][2][i, j]) for j in range(3)] for i in range(3)],
        "T2": [[str(g["T"][2][i, j]) for j in range(3)] for i in range(3)],
        "q_powers": {
            k: [
                [{f"{a},{b}": ms for (a, b), ms in dct.items()} for dct in row]
                for row in powers[k]
            ]
            for k in powers
        },
    }
    path = Path(__file__).resolve().parent / "data" / f"interface_omega4_{which}.json"
    path.write_text(json.dumps(out, indent=1))
    print("written", path.name)


if __name__ == "__main__":
    main()
