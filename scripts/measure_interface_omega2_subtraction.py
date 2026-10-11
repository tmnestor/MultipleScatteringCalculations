"""Measure: convergence of the plane-wave integral of a reflected or transmitted block with the static field and
the O(omega^2) term subtracted.

Paper: LatexPDFs/InterfaceCouplingIntegrals (the asymptotic subtraction beyond the static image).

THE BLOCKS.  Reflected (argument R, the default): that of scripts/measure_interface_static_subtraction.py, a
cell of half-width h touching the interface from above, field function 1, source monomial 1, coupled to itself
through the reflection.  Transmitted (argument T): the source cell the same, the field cell its mirror image in
medium B, the two touching across the interface, coupled through the transmission (their whole coupling).
In either case

    K = (1 / 4 pi^2) int d^2k  L(k) Y(k; omega).

At large q the kernel expands as Y = Y0 + omega^2 Y1 + O(omega^4 / q^4 relative), Y0 the static image or the
static transmitted field and Y1 the omega^2 term (R1 or T1 of Mathematica/InterfaceOmega2.wl, verified in
scripts/derive_interface_omega2.py).
Y1 does not depend on omega, so the split

    K = K0 + omega^2 K1(Q0) + int [Y - Y0 - omega^2 Y1 theta(q > Q0)] L d^2k,    K1(Q0) = int_{q > Q0} L Y1 d^2k,

computes K0 and K1 once per geometry; only the last integral, whose integrand falls by two more powers of q
than with the static image alone, is evaluated at each frequency.

MEASURED, at omega h / beta = 1/6 and 1 (omega = 10, 60): the contribution of each shell Q_{j-1} < q h < Q_j,
Q_j = 10 * 2^j, relative to the whole block, of (i) Y, (ii) Y - Y0, (iii) Y - Y0 - omega^2 Y1, (iv) omega^2 Y1
(the integrand of K1); the local order log2(previous / this); and the tail beyond each cutoff, the measured
shells to q h = 320 plus a geometric extrapolation at the last measured order.

THE KERNELS.  Both Y0 and Y1 are X times a polynomial in (z, zp) with coefficients rational in q,
X = e^{q(z + zp)} (reflected) or e^{-q(z - zp)} (transmitted); the 9 x 9 frame kernel is built from the 3 x 3
displacement kernel by the receiver derivatives (d_z, i q, 0) and the source derivatives (-d_zp, i q, 0) acting
on the polynomial (d_z -> +-q + d_z), and each monomial z^a zp^b is integrated over the cells' depths in closed
form.  The dynamic kernels are 40-digit mode sums (the reflected one that of the static measurement), and the
quadrature rules are the static measurement's, with every branch point of both media a panel end below q h = 10.

OUTPUT: scripts/data/interface_omega2_reference_{R,T}.json, the 9 x 9 integrals behind the tables: for each omega
the part q h < 10 of Y, and for each shell those of Y, Y0 and Y1 -- the 40-digit reference of
scripts/gate_interface_spectral.py.

Checks printed: (reflected) the frame Y0 built here against the static measurement's; (Y - Y0) / omega^2
against Y1 at large q (the difference falls as q^-2); one shell with finer rules.

Run:  PYTHONPATH=.:scripts python scripts/measure_interface_omega2_subtraction.py [R|T]
      (about 20 minutes on 4 cores each)
"""

import json
import sys
from itertools import pairwise
from multiprocessing import Pool
from pathlib import Path

import measure_interface_static_subtraction as st
import mpmath as mp
import numpy as np
import sympy as sp
from sympy.parsing.mathematica import parse_mathematica

OMEGAS = [10.0, 60.0]
CUTS = [10, 20, 40, 80, 160, 320]  # q h
DPS = st.DPS
H = st.H

KIND = "R"  # "R" reflected, "T" transmitted; set from the command line before the workers fork
z, zp, q_ = st.z, st.zp, st.q_
_ROOT = Path(__file__).resolve().parent.parent


def _materials():
    """The sweep's media as exact rationals of their binary values (as the 40-digit dynamic kernel sees them),
    keyed by the Mathematica names: a decimal rounding of 1e-16 would exceed the remainder being measured."""
    out = {}
    for tag, med in (("A", st.A), ("B", st.B)):
        rho, al, be = (sp.Rational(v) for v in (med.rho, med.alpha, med.beta))
        out[f"lam{tag}"], out[f"mu{tag}"], out[f"rho{tag}"] = (
            rho * (al**2 - 2 * be**2),
            rho * be**2,
            rho,
        )
    return out


def _omega2_kernel():
    """R1 or T1 (rows u_(z,x,y), columns force (z,x,y)) at the sweep's media, sympy, from Mathematica's general
    result."""
    m = json.loads((_ROOT / "Mathematica" / "InterfaceOmega2.json").read_text())
    subs = {sp.Symbol(k): v for k, v in _materials().items()}
    subs |= {sp.Symbol("q"): q_, sp.Symbol("z"): z, sp.Symbol("zp"): zp}
    return sp.Matrix(
        [[parse_mathematica(e).subs(subs) for e in row] for row in m[KIND + "1"]]
    )


def _static_kernel():
    """The static image (R) or static transmitted field (T) at the sweep's media, sympy."""
    mats = _materials()
    subs = {getattr(st, k): mats[k] for k in ("lamA", "muA", "lamB", "muB")}
    if KIND == "R":
        return st._G.subs(subs)
    path = _ROOT / "scripts" / "data" / "static_interface_transmission.json"
    rows = json.loads(path.read_text())["entries"]
    return sp.Matrix(
        [[sp.sympify(e, locals=st._LOC) for e in row] for row in rows]
    ).subs(subs)


def _sign():
    """s in X = e^{s q z + q zp}: +1 reflected, -1 transmitted."""
    return 1 if KIND == "R" else -1


def _polys(G):
    """G = X P: P as a 3 x 3 matrix of rational functions of q, polynomial in z, zp."""
    P = G.applyfunc(
        lambda e: e.replace(sp.exp, lambda a: sp.Integer(1)) if e.has(sp.exp) else e
    )
    for e in G:
        assert {a.args[0] for a in e.atoms(sp.exp)} <= {q_ * (_sign() * z + zp)}, e
    return P.applyfunc(sp.cancel)


def frame_coefficients(G):
    """The 9 x 9 frame kernel of G as {(a, b): 9 x 9 sympy matrix in q}, the coefficient of z^a zp^b X."""
    P = _polys(G)
    dz = lambda e: _sign() * q_ * e + sp.diff(e, z)
    dzp = lambda e: q_ * e + sp.diff(e, zp)
    drec = lambda e, j: [dz(e), sp.I * q_ * e, sp.Integer(0)][j]
    dsrc = lambda e, l: [-dzp(e), sp.I * q_ * e, sp.Integer(0)][l]
    out = sp.zeros(9, 9)
    for i in range(3):
        for k in range(3):
            out[i, k] = P[i, k]
    for b, (m, n) in enumerate(st.VP):
        for i in range(3):
            out[i, 3 + b] = (
                dsrc(P[i, m], n)
                if m == n
                else (dsrc(P[i, m], n) + dsrc(P[i, n], m)) / 2
            )
    for a, (p, r) in enumerate(st.VP):
        for k in range(3):
            out[3 + a, k] = (
                drec(P[p, k], p) if p == r else drec(P[r, k], p) + drec(P[p, k], r)
            )

        def term(mm, nn, p=p, r=r):
            e = lambda j, i: drec(dsrc(P[i, mm], nn), j)
            return e(p, p) if p == r else e(p, r) + e(r, p)

        for b, (m, n) in enumerate(st.VP):
            out[3 + a, 3 + b] = term(m, n) if m == n else (term(m, n) + term(n, m)) / 2
    coeffs: dict = {}
    for i in range(9):
        for j in range(9):
            num, den = sp.fraction(sp.cancel(sp.together(out[i, j])))
            if num == 0:
                continue
            for (a, b), c in sp.Poly(sp.expand(num), z, zp).terms():
                coeffs.setdefault((a, b), sp.zeros(9, 9))[i, j] += c / den
    return coeffs


def vmom(a, n):
    """int_{-2h}^{0} z^n e^{a z} dz in mpmath, Re a >= 0: the recurrence I_n = -(-L)^n e^{-aL}/a - (n/a) I_{n-1}."""
    L = 2 * mp.mpf(H)
    if abs(a * L) < 4 * (n + 1):
        return mp.quad(lambda t: t**n * mp.exp(a * t), [-L, 0])
    e = mp.exp(-a * L)
    out = (1 - e) / a
    for k in range(1, n + 1):
        out = -((-L) ** k) * e / a - k / a * out
    return out


_KERNELS: dict = {}


def build() -> None:
    """Lambdify the frame coefficients of Y0 and Y1 (once, before the workers fork)."""
    for name, G in (("Y0", _static_kernel()), ("Y1", _omega2_kernel())):
        co = frame_coefficients(G)
        keys = sorted(co)
        _KERNELS[name] = (keys, sp.lambdify(q_, [co[k] for k in keys], "mpmath"))


def frame(name, q):
    """The frame kernel of Y0 or Y1 at q (mp), lateral moments excluded, 9 x 9 complex numpy."""
    keys, fn = _KERNELS[name]
    out = mp.matrix(9, 9)
    for (a, b), c in zip(keys, fn(q), strict=True):
        # the field cell: z in [-2h, 0] against e^{q z} (R), or z in [0, 2h] against e^{-q z} (T): z -> -z
        field = vmom(q, a) * (1 if KIND == "R" else (-1) ** a)
        out += mp.matrix(c) * (field * vmom(q, b))
    return np.array(out.tolist(), dtype=complex)


def dynamic_transmitted(q):
    """The transmitted frame kernel at k = (q, 0), source cell in A on [-2h, 0], field cell in B on [0, 2h]."""
    ma, mb, om = st._med(st.A), st._med(st.B), mp.mpf(st.OMEGA)
    dn_a, up_a, dn_b = (
        st._modes(q, om, ma, 1),
        st._modes(q, om, ma, -1),
        st._modes(q, om, mb, 1),
    )
    lhs, rhs = mp.matrix(6, 6), mp.matrix(6, 3)
    for j in range(3):
        ua, ub, ud = (
            st._ut(up_a[j][1], ma),
            st._ut(dn_b[j][1], mb),
            st._ut(dn_a[j][1], ma),
        )
        for i in range(6):
            lhs[i, j], lhs[i, 3 + j], rhs[i, j] = ua[i], -ub[i], -ud[i]
    t_d = mp.inverse(lhs) * rhs  # rows 3..5: transmitted down-going amplitudes in B
    pair = [1, 1, 1, 1, 1, 1, mp.mpf(1) / 2, mp.mpf(1) / 2, mp.mpf(1) / 2]
    out = mp.matrix(9, 9)
    for m in range(3):
        kzm, dm, _ = dn_b[m]
        vm = vmom(-mp.mpc(0, 1) * kzm, 0)  # int_0^{2h} e^{i kz z} dz
        for n in range(3):
            kzn, dn, wn = dn_a[n]
            c = vm * t_d[3 + m, n] * wn * vmom(-mp.mpc(0, 1) * kzn, 0)
            for i in range(9):
                for j in range(9):
                    out[i, j] += c * dm[i] * dn[j] * pair[j]
    return out


def dynamic(q):
    return st.dynamic_frame(q) if KIND == "R" else dynamic_transmitted(q)


def kernels(q_h):
    """{'Y0', 'Y1', omega: Y} at q h."""
    mp.mp.dps = DPS
    q = mp.mpf(q_h) / H
    out = {"Y0": frame("Y0", q), "Y1": frame("Y1", q)}
    for om in OMEGAS:
        st.OMEGA = om
        out[om] = np.array(dynamic(q).tolist(), dtype=complex)
    return out


def dynamic_only(args):
    q_h, om = args
    mp.mp.dps = DPS
    st.OMEGA = om
    return np.array(dynamic(mp.mpf(q_h) / H).tolist(), dtype=complex)


def integrate(nodes, wts, kern, fn, n_phi=None):
    """(1/4 pi^2) int q dq dphi L fn(kernels) over the given radial nodes."""
    out = np.zeros((9, 9), complex)
    for qh, w, k in zip(nodes, wts, kern, strict=True):
        n = None if n_phi is None else n_phi(qh)
        out += w * (qh / H) / H / (4 * np.pi**2) * st.azimuthal(qh, fn(k), n_phi=n)
    return out


def low_nodes(om):
    """Nodes on [0, 10] in q h with every branch point of both media (q h = omega h / c) a panel end, the
    square-root singularity there removed by a quadratic map."""
    bp = sorted({om * H / c for c in (st.A.alpha, st.A.beta, st.B.alpha, st.B.beta)})
    x, w = st.leggauss(24)
    u, wu = (x + 1) / 2, w / 2
    nodes, wts = [], []

    def towards(a, b):  # singular at b
        t = 1 - u
        nodes.append(b - (b - a) * t**2)
        wts.append(wu * 2 * (b - a) * t)

    def away(a, b):  # singular at a
        nodes.append(a + (b - a) * u**2)
        wts.append(wu * 2 * (b - a) * u)

    towards(0.0, bp[0])
    for a, b in pairwise(bp):
        towards((a + b) / 2, b)
        away(a, (a + b) / 2)
    end = max(1.0, 2 * bp[-1])
    away(bp[-1], end)
    n2, w2 = st.panels(end, 10.0, width=0.5, order=16)
    return np.concatenate([*nodes, n2]), np.concatenate([*wts, w2])


def main() -> None:
    global KIND
    KIND = sys.argv[1] if len(sys.argv) > 1 else "R"
    assert KIND in ("R", "T"), KIND
    print("reflected block" if KIND == "R" else "transmitted block", flush=True)
    mp.mp.dps = DPS
    build()
    if KIND == "R":
        q = mp.mpf(160) / H
        y0 = frame("Y0", q)
        ref = np.array(st.static_frame(q).tolist(), dtype=complex)
        print(
            f"check: Y0 frame here vs the static measurement's, q h = 160: "
            f"{np.abs(y0 - ref).max() / np.abs(ref).max():.1e}"
        )
    print("check: |(Y - Y0)/omega^2 - Y1| / |Y1| at large q (falls as q^-2):")
    for om in OMEGAS:
        row = []
        for q_h in (20, 40, 80, 160):
            k = kernels(q_h)
            row.append(
                np.abs((k[om] - k["Y0"]) / om**2 - k["Y1"]).max()
                / np.abs(k["Y1"]).max()
            )
        print(f"  omega {om:5.1f}: " + "  ".join(f"{v:.2e}" for v in row), flush=True)

    with Pool(4) as pool:
        lows = {}
        for om in OMEGAS:
            n, w = low_nodes(om)
            kern = pool.map(dynamic_only, [(x, om) for x in n])
            lows[om] = integrate(n, w, kern, lambda k: k)
        shells = []
        for a, b in pairwise(CUTS):
            n, w = st.panels(a, b)
            kern = pool.map(kernels, list(n))
            parts = {
                "Y0": integrate(n, w, kern, lambda k: k["Y0"]),
                "Y1": integrate(n, w, kern, lambda k: k["Y1"]),
            }
            for om in OMEGAS:
                parts[om] = integrate(n, w, kern, lambda k, om=om: k[om])
            shells.append(parts)
            print(f"  shell {a}-{b}: {len(n)} radial nodes", flush=True)

        # resolution check on the shell 160-320 of the doubly subtracted integrand: azimuth doubled, panels halved
        n, w = st.panels(160, 320, width=np.pi / 4)
        kern = pool.map(kernels, list(n))

    def pack(m):
        return {"re": np.real(m).tolist(), "im": np.imag(m).tolist()}

    ref = {
        "kind": KIND,
        "h": H,
        "media": {t: [m.alpha, m.beta, m.rho] for t, m in (("A", st.A), ("B", st.B))},
        "block": "field function 1, source monomial 1; R: cell centre (-h, 0, 0) with its image;"
        " T: source centre (-h, 0, 0), field centre (h, 0, 0)",
        "cuts_qh": CUTS,
        "low": {str(om): pack(lows[om]) for om in OMEGAS},
        "shells": [
            {k if isinstance(k, str) else str(k): pack(v) for k, v in sh.items()}
            for sh in shells
        ],
    }
    out = (
        Path(__file__).resolve().parent
        / "data"
        / f"interface_omega2_reference_{KIND}.json"
    )
    out.write_text(json.dumps(ref))
    print("written", out.name)

    for om in OMEGAS:
        total = lows[om] + sum(s[om] for s in shells)
        scale = np.abs(total).max()
        rem = lambda s, om=om: s[om] - s["Y0"] - om**2 * s["Y1"]
        fine = integrate(
            n,
            w,
            kern,
            lambda k, om=om: k[om] - k["Y0"] - om**2 * k["Y1"],
            n_phi=lambda qh: int(8 * np.ceil(qh) + 128),
        )
        print(f"\nomega = {om}, omega h / beta = {om * H / st.A.beta:.3f}")
        print(
            f"check: shell 160-320 of Y - Y0 - w^2 Y1, finer rules: {np.abs(fine - rem(shells[-1])).max() / scale:.1e}"
            f" (contribution {np.abs(rem(shells[-1])).max() / scale:.1e})"
        )
        cols = {
            "Y": lambda s, om=om: s[om],
            "Y - Y0": lambda s, om=om: s[om] - s["Y0"],
            "Y - Y0 - w2 Y1": rem,
            "w2 Y1": lambda s, om=om: om**2 * s["Y1"],
        }
        vals = {
            c: [np.abs(f(s)).max() / scale for s in shells] for c, f in cols.items()
        }
        print(
            "shell contribution relative to the block (max entry), local order log2(previous / this):"
        )
        print(f"{'q h':>10}" + "".join(f"{c:>17} {'order':>6}" for c in cols))
        for j, (a, b) in enumerate(pairwise(CUTS)):
            line = f"{a:>4}-{b:<5}"
            for c in cols:
                v = vals[c]
                o = f"{np.log2(v[j - 1] / v[j]):6.2f}" if j else " " * 6
                line += f"{v[j]:17.2e} {o}"
            print(line)
        print(
            "tail beyond the cutoff (measured shells + geometric extrapolation at the last order):"
        )
        print(f"{'Q h':>6}" + "".join(f"{c:>17}" for c in ("Y - Y0", "Y - Y0 - w2 Y1")))
        for j, a in enumerate(CUTS[:-1]):
            line = f"{a:>6}"
            for c, f in (("Y - Y0", cols["Y - Y0"]), ("Y - Y0 - w2 Y1", rem)):
                p = np.log2(vals[c][-2] / vals[c][-1])
                tail = sum(f(s) for s in shells[j:])
                extra = f(shells[-1]) / (2**p - 1)
                line += f"{np.abs(tail + extra).max() / scale:17.2e}"
            print(line)


if __name__ == "__main__":
    main()
