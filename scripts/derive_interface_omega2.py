"""Derive: the O(omega^2) term of the reflected and transmitted spectral kernels of a welded interface.

Paper: LatexPDFs/InterfaceCouplingIntegrals (the asymptotic subtraction beyond the static image).
Mathematica twin: Mathematica/InterfaceOmega2.wl.

WHAT.  Medium A (lamA, muA, rhoA) fills z < 0, medium B (lamB, muB, rhoB) fills z > 0, welded at z = 0; a unit
point force in A at depth zp < 0; frame x along the lateral wavevector (q, 0).  At large q the reflected and
transmitted kernels expand in powers of omega^2 at fixed (q, z, zp):

    G(q; z, zp; omega) = G0 + omega^2 G1 + O(omega^4),

G0 the static kernels (Rongved's problem, scripts/derive_static_interface_image.py and
..._transmission.py).  This script derives G1, the reflected part R1 (both points in A) and the transmitted
part T1 (receiver in B).

HOW: perturbation in omega^2.  With u = u0 + omega^2 u1 + ..., the Navier equations
div sigma(u) + rho omega^2 u + F delta = 0 give at order omega^2  div sigma(u1) = -rho u0, with no source:
u1 and its traction are continuous at z = zp and at z = 0.  In each region u0 is a polynomial times e^{+-q z}
(a double root of the static operator); u1 is a particular solution, a polynomial of degree two higher times the
same exponential, plus the homogeneous solutions regular in that region.  The whole-space part (Kelvin's
omega^2 term) is solved first; it forces the interface conditions of R1 and T1.

CHECK: against the exact dynamic kernels in 50-digit arithmetic (the mode sums of
cubic_scattering.stratified_march, re-implemented in mpmath; in double precision their P and SV terms cancel as
(c q / omega)^4): (G(omega) - G0 - omega^2 G1) / omega^4 must tend to a constant as omega -> 0, i.e. the residual
must fall sixteenfold when omega halves.  And identical media must give R1 = 0 and T1 = Kelvin's omega^2 term.

OUTPUT: scripts/data/interface_omega2.json: entries of R1 and T1 (sympy strings), their values at test points,
and the powers of q that multiply each monomial z^a zp^b once e^{q(z+zp)} or e^{-q(z-zp)} is removed (these
decide which potentials the spatial form needs: q^m e^{-q zeta} with m <= -4 is a line of images of higher order).

Run:  PYTHONPATH=. python scripts/derive_interface_omega2.py
"""

import json
from pathlib import Path

import mpmath as mp
import sympy as sp

z, zp = sp.symbols("z zp", real=True)
q = sp.Symbol("q", positive=True)
lamA, muA, rhoA, lamB, muB, rhoB = sp.symbols("lamA muA rhoA lamB muB rhoB", positive=True)
COMPS = ("x", "y", "z")


def stresses(u, lam, mu):
    dx = lambda f: sp.I * q * f
    dz = lambda f: sp.diff(f, z)
    return {
        "zz": (lam + 2 * mu) * dz(u["z"]) + lam * dx(u["x"]),
        "xz": mu * (dz(u["x"]) + dx(u["z"])),
        "yz": mu * dz(u["y"]),
        "xx": lam * dz(u["z"]) + (lam + 2 * mu) * dx(u["x"]),
        "xy": mu * dx(u["y"]),
    }


def navier(u, lam, mu):
    """div sigma, components (x, y, z)."""
    s = stresses(u, lam, mu)
    return {
        "x": sp.I * q * s["xx"] + sp.diff(s["xz"], z),
        "y": sp.I * q * s["xy"] + sp.diff(s["yz"], z),
        "z": sp.I * q * s["xz"] + sp.diff(s["zz"], z),
    }


def traction(u, lam, mu):
    s = stresses(u, lam, mu)
    return {"x": s["xz"], "y": s["yz"], "z": s["zz"]}


def homogeneous(lam, mu, s, names):
    """The three homogeneous solutions ~ e^{s q z}: two P-SV, one SH, with coefficients `names`."""
    a, b, c, d = sp.symbols("a b c d")
    u = {"x": (a + b * z) * sp.exp(s * q * z), "z": (c + d * z) * sp.exp(s * q * z), "y": sp.Integer(0)}
    eqs = navier(u, lam, mu)
    coeffs = []
    for k in ("x", "z"):
        coeffs += sp.Poly(sp.expand(sp.simplify(eqs[k] * sp.exp(-s * q * z))), z).coeffs()
    sol = sp.solve(coeffs, [c, d], dict=True)[0]
    c1, c2, c3 = names
    out = {k: sp.Integer(0) for k in COMPS}
    for cc, (pa, pb) in ((c1, (1, 0)), (c2, (0, 1))):
        for k in ("x", "z"):
            out[k] += cc * u[k].subs(sol).subs({a: pa, b: pb})
    out["y"] += c3 * sp.exp(s * q * z)
    return out


def _poly_part(expr, s):
    """expr * e^{-s q z} with the exponential in z cancelled exactly: a polynomial in z."""
    e = sp.powsimp(sp.expand(sp.simplify(expr * sp.exp(-s * q * z))), combine="exp")
    e = sp.expand(sp.simplify(e))
    assert not e.has(sp.exp(q * z)) and z not in sp.exp(q).free_symbols, e
    return e


def particular(u0, lam, mu, rho, s):
    """A solution of div sigma(u1) = -rho u0, u0 = poly(z) e^{s q z}: u1 = poly of degree + 2 times e^{s q z}."""
    deg = max(sp.Poly(_poly_part(u0[k], s), z).degree() if u0[k] != 0 else 0 for k in COMPS)
    n = deg + 3
    cs = {k: sp.symbols(f"p{k}0:{n}") for k in COMPS}
    u1 = {k: sum(cs[k][i] * z**i for i in range(n)) * sp.exp(s * q * z) for k in COMPS}
    res = navier(u1, lam, mu)
    eqs = []
    for k in COMPS:
        e = _poly_part(res[k] + rho * u0[k], s)
        eqs += sp.Poly(e, z).coeffs() if e != 0 else []
    unknowns = [c for k in COMPS for c in cs[k]]
    sol = sp.solve(eqs, unknowns, dict=True)[0]
    free = {c: 0 for c in unknowns if c not in sol}
    return {k: sp.expand(u1[k].subs(sol).subs(free)) for k in COMPS}


def add(*us):
    return {k: sum(u[k] for u in us) for k in COMPS}


def solve_force(f):
    """(R0, R1, T0, T1, K0, K1) columns for a unit force along f."""
    F = {k: sp.Integer(1 if k == f else 0) for k in COMPS}
    # order 0: Kelvin, then reflection and transmission
    below = homogeneous(lamA, muA, -1, sp.symbols("kb0:3"))  # z > zp
    above = homogeneous(lamA, muA, 1, sp.symbols("ka0:3"))  # z < zp
    tb, ta = traction(below, lamA, muA), traction(above, lamA, muA)
    eqs = [(below[k] - above[k]).subs(z, zp) for k in COMPS]
    eqs += [(tb[k] - ta[k]).subs(z, zp) + F[k] for k in COMPS]
    sol = sp.solve(eqs, list(sp.symbols("kb0:3")) + list(sp.symbols("ka0:3")), dict=True)[0]
    k0b = {k: sp.simplify(below[k].subs(sol)) for k in COMPS}
    k0a = {k: sp.simplify(above[k].subs(sol)) for k in COMPS}
    r0 = homogeneous(lamA, muA, 1, sp.symbols("r0:3"))
    t0 = homogeneous(lamB, muB, -1, sp.symbols("t0:3"))
    tk, tr, tt = traction(k0b, lamA, muA), traction(r0, lamA, muA), traction(t0, lamB, muB)
    eqs = [(k0b[k] + r0[k] - t0[k]).subs(z, 0) for k in COMPS]
    eqs += [(tk[k] + tr[k] - tt[k]).subs(z, 0) for k in COMPS]
    sol = sp.solve(eqs, list(sp.symbols("r0:3")) + list(sp.symbols("t0:3")), dict=True)[0]
    r0 = {k: sp.simplify(r0[k].subs(sol)) for k in COMPS}
    t0 = {k: sp.simplify(t0[k].subs(sol)) for k in COMPS}
    # order omega^2: Kelvin's term, no source, continuous across zp
    k1b = add(particular(k0b, lamA, muA, rhoA, -1), homogeneous(lamA, muA, -1, sp.symbols("lb0:3")))
    k1a = add(particular(k0a, lamA, muA, rhoA, 1), homogeneous(lamA, muA, 1, sp.symbols("la0:3")))
    tb, ta = traction(k1b, lamA, muA), traction(k1a, lamA, muA)
    eqs = [(k1b[k] - k1a[k]).subs(z, zp) for k in COMPS]
    eqs += [(tb[k] - ta[k]).subs(z, zp) for k in COMPS]
    sol = sp.solve(eqs, list(sp.symbols("lb0:3")) + list(sp.symbols("la0:3")), dict=True)[0]
    k1b = {k: sp.simplify(k1b[k].subs(sol)) for k in COMPS}
    # the reflected and transmitted omega^2 terms
    r1 = add(particular(r0, lamA, muA, rhoA, 1), homogeneous(lamA, muA, 1, sp.symbols("s0:3")))
    t1 = add(particular(t0, lamB, muB, rhoB, -1), homogeneous(lamB, muB, -1, sp.symbols("u0:3")))
    tk, tr, tt = traction(k1b, lamA, muA), traction(r1, lamA, muA), traction(t1, lamB, muB)
    eqs = [(k1b[k] + r1[k] - t1[k]).subs(z, 0) for k in COMPS]
    eqs += [(tk[k] + tr[k] - tt[k]).subs(z, 0) for k in COMPS]
    sol = sp.solve(eqs, list(sp.symbols("s0:3")) + list(sp.symbols("u0:3")), dict=True)[0]
    r1 = {k: sp.simplify(r1[k].subs(sol)) for k in COMPS}
    t1 = {k: sp.simplify(t1[k].subs(sol)) for k in COMPS}
    return r0, r1, t0, t1, k0b, k1b


IDX = {"z": 0, "x": 1, "y": 2}


def tensors():
    mats = {name: sp.zeros(3, 3) for name in ("R0", "R1", "T0", "T1", "K0", "K1")}
    for f in COMPS:
        cols = solve_force(f)
        for name, col in zip(("R0", "R1", "T0", "T1", "K0", "K1"), cols, strict=True):
            for k in COMPS:
                mats[name][IDX[k], IDX[f]] = col[k]
    return mats


# ------------------------------------------------------------------------------------- the exact dynamic kernels
def _modes(qq, om, med, sign):
    out = []
    sh = [mp.mpf(0), mp.mpf(0), mp.mpf(1)]
    for c, wave in ((med["alpha"], "P"), (med["beta"], "SV"), (med["beta"], "SH")):
        kz = mp.sqrt(mp.mpc((om / c) ** 2 - qq * qq))
        kz = -kz if mp.im(kz) < 0 else kz
        k = [sign * kz, qq, mp.mpf(0)]
        if wave == "P":
            e = [ki * c / om for ki in k]
        elif wave == "SH":
            e = sh
        else:
            e = [(k[1] * sh[2] - k[2] * sh[1]) * c / om, (k[2] * sh[0] - k[0] * sh[2]) * c / om,
                 (k[0] * sh[1] - k[1] * sh[0]) * c / om]
        out.append((kz, k, e, mp.mpc(0, 1) / (2 * med["rho"] * c * c * kz)))
    return out


def _ut(k, e, med):
    """(u, t) of a plane wave: t = (sigma_zz, sigma_xz, sigma_yz) with eps_ij = i/2 (k_i e_j + k_j e_i)."""
    i = mp.mpc(0, 1)
    eps = lambda a, b: i / 2 * (k[a] * e[b] + k[b] * e[a])
    tr = eps(0, 0) + eps(1, 1) + eps(2, 2)
    return [e[0], e[1], e[2], med["lam"] * tr + 2 * med["mu"] * eps(0, 0), 2 * med["mu"] * eps(0, 1),
            2 * med["mu"] * eps(0, 2)]


def dynamic(qq, om, zr, zs, ma, mb, kind):
    """The exact reflected (zr < 0) or transmitted (zr > 0) displacement kernel, 3 x 3, order (z, x, y)."""
    dn_a, up_a, dn_b = _modes(qq, om, ma, 1), _modes(qq, om, ma, -1), _modes(qq, om, mb, 1)
    lhs, rhs = mp.matrix(6, 6), mp.matrix(6, 3)
    for j in range(3):
        ua = _ut(up_a[j][1], up_a[j][2], ma)
        ub = _ut(dn_b[j][1], dn_b[j][2], mb)
        ud = _ut(dn_a[j][1], dn_a[j][2], ma)
        for i in range(6):
            lhs[i, j], lhs[i, 3 + j], rhs[i, j] = ua[i], -ub[i], -ud[i]
    sol = mp.inverse(lhs) * rhs
    out = mp.matrix(3, 3)
    i = mp.mpc(0, 1)
    for m in range(3):
        if kind == "R":
            kzm, em = up_a[m][0], up_a[m][2]
            ph = mp.exp(i * kzm * abs(zr))
            amp = [sol[m, n] for n in range(3)]
        else:
            kzm, em = dn_b[m][0], dn_b[m][2]
            ph = mp.exp(i * kzm * zr)
            amp = [sol[3 + m, n] for n in range(3)]
        for n in range(3):
            kzn, _, en, wn = dn_a[n]
            c = ph * amp[n] * wn * mp.exp(i * kzn * abs(zs))
            for a in range(3):
                for b in range(3):
                    out[a, b] += c * em[a] * en[b]
    return out


def _med(lam, mu, rho):
    lam, mu, rho = mp.mpf(lam), mp.mpf(mu), mp.mpf(rho)
    return {"lam": lam, "mu": mu, "rho": rho, "alpha": mp.sqrt((lam + 2 * mu) / rho), "beta": mp.sqrt(mu / rho)}


def q_powers(expr, kind):
    """{(a, b): sorted powers of q} of expr / X, X = e^{q(z+zp)} (R) or e^{-q(z-zp)} (T), as a polynomial in z, zp."""
    x = sp.exp(q * (z + zp)) if kind == "R" else sp.exp(-q * (z - zp))
    e = sp.expand(sp.powsimp(sp.expand(expr / x), combine="exp"))
    assert not e.has(sp.exp), e
    out: dict = {}
    for term in sp.Add.make_args(e):
        pa, pb = sp.degree(term, z), sp.degree(term, zp)
        rest = sp.simplify(term / (z**pa * zp**pb))
        num, den = sp.fraction(sp.together(rest))
        m = sp.degree(num, q) - sp.degree(den, q)
        out.setdefault((int(pa), int(pb)), set()).add(int(m))
    return {k: sorted(v) for k, v in sorted(out.items())}


def _values(fn, t):
    """[[re, im], ...] as 20-digit strings, row by row."""
    m = mp.matrix(fn(*map(mp.mpf, t)))
    return [[[mp.nstr(mp.re(m[i, j]), 20), mp.nstr(mp.im(m[i, j]), 20)] for j in range(3)] for i in range(3)]


def main() -> None:
    mats = tensors()
    print("identical media: R0 = 0, R1 = 0:",
          all(sp.simplify(mats[n].subs({lamB: lamA, muB: muA, rhoB: rhoA})) == sp.zeros(3, 3) for n in ("R0", "R1")))
    print("identical media: T1 = Kelvin's omega^2 term:",
          sp.simplify(mats["T1"].subs({lamB: lamA, muB: muA, rhoB: rhoA}) - mats["K1"]) == sp.zeros(3, 3))

    # against the exact kernels at 50 digits: residual / omega^4 tends to a constant
    mp.mp.dps = 50
    pars = {lamA: 1.7, muA: 1.1, rhoA: 2.5, lamB: 2.6, muB: 1.9, rhoB: 2.8}
    ma, mb = _med(1.7, 1.1, 2.5), _med(2.6, 1.9, 2.8)
    syms = (q, z, zp, lamA, muA, rhoA, lamB, muB, rhoB)
    vals = (1.7, 1.1, 2.5, 2.6, 1.9, 2.8)
    fns = {n: sp.lambdify(syms, mats[n], "mpmath") for n in ("R0", "R1", "T0", "T1")}
    for kind, (qq, zr, zs) in (("R", (3.0, -0.4, -0.7)), ("T", (3.0, 0.4, -0.7))):
        g0 = mp.matrix(fns[kind + "0"](mp.mpf(qq), mp.mpf(zr), mp.mpf(zs), *map(mp.mpf, vals)))
        g1 = mp.matrix(fns[kind + "1"](mp.mpf(qq), mp.mpf(zr), mp.mpf(zs), *map(mp.mpf, vals)))
        prev = None
        print(f"{kind}: q = {qq}, z = {zr}, zp = {zs}")
        for om in (0.4, 0.2, 0.1, 0.05):
            g = dynamic(mp.mpf(qq), mp.mpf(om), mp.mpf(zr), mp.mpf(zs), ma, mb, kind)
            res = mp.mnorm(g - g0 - om**2 * g1, 1) / mp.mnorm(g0, 1)
            first = mp.mnorm(g - g0, 1) / mp.mnorm(g0, 1)
            ratio = "" if prev is None else f"  residual ratio {mp.nstr(prev / res, 4)}"
            print(f"  omega {om:5.2f}: |G - G0|/|G0| {mp.nstr(first, 3)}   |G - G0 - w^2 G1|/|G0| {mp.nstr(res, 3)}{ratio}")
            prev = res
    del pars

    powers = {
        "R1": [[q_powers(mats["R1"][i, j], "R") for j in range(3)] for i in range(3)],
        "T1": [[q_powers(mats["T1"][i, j], "T") for j in range(3)] for i in range(3)],
    }
    lowest = {n: min(m for row in powers[n] for d in row for ms in d.values() for m in ms) for n in powers}
    print("lowest power of q (after the exponential and the z, zp monomials):", lowest)

    tests = [(2.3, -0.4, -0.9, 1.7, 1.1, 2.5, 2.6, 1.9, 2.8), (11.0, -0.05, -0.12, 0.6, 0.9, 2.2, 1.4, 2.2, 2.6)]
    tests_t = [(2.3, 0.4, -0.9, *t[3:]) for t in tests]
    out = {
        "order": "rows u_(z,x,y), columns force (z,x,y); frame x along the lateral wavevector; G = G0 + omega^2 G1",
        "symbols": [str(s) for s in syms],
        "R1": [[str(mats["R1"][i, j]) for j in range(3)] for i in range(3)],
        "T1": [[str(mats["T1"][i, j]) for j in range(3)] for i in range(3)],
        "tests_R": tests,
        "tests_T": tests_t,
        "R1_values": [_values(fns["R1"], t) for t in tests],
        "T1_values": [_values(fns["T1"], t) for t in tests_t],
        "q_powers": {n: [[{f"{a},{b}": ms for (a, b), ms in d.items()} for d in row] for row in powers[n]]
                     for n in powers},
    }
    path = Path(__file__).resolve().parent / "data" / "interface_omega2.json"
    path.write_text(json.dumps(out, indent=1))
    print("written", path.name)

    wl = Path(__file__).resolve().parent.parent / "Mathematica" / "InterfaceOmega2.json"
    if not wl.exists():
        print("Mathematica/InterfaceOmega2.json not found: run the .wl to compare (SKIPPED, not PASS)")
        return
    m = json.loads(wl.read_text())
    worst = 0.0
    for name, key in (("R1", "R1_values"), ("T1", "T1_values")):
        for k, mine in enumerate(out[key]):
            ref = [[complex(float(m[name + "_re"][k][i][j]), float(m[name + "_im"][k][i][j])) for j in range(3)]
                   for i in range(3)]
            got = [[complex(float(mine[i][j][0]), float(mine[i][j][1])) for j in range(3)] for i in range(3)]
            scale = max(abs(x) for row in ref for x in row)
            worst = max(worst, max(abs(got[i][j] - ref[i][j]) for i in range(3) for j in range(3)) / scale)
    print(f"against Mathematica: {worst:.1e}", "PASS" if worst < 1e-14 else "FAIL")


if __name__ == "__main__":
    main()
