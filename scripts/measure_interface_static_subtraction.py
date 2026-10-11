"""Measure: convergence of the plane-wave integral of a reflected block, with and without the static image.

Paper: LatexPDFs/InterfaceCouplingIntegrals, Table 1 ("Where the plane waves fail").

THE BLOCK.  A cell of half-width h touching a welded interface from above (centre at z = -h), field function 1
and source monomial 1 (p = 0), coupled to itself through the reflection (medium A above, B below):

    K = (1 / 4 pi^2) int d^2k  L(k) * Y(k),

with L = (lateral moment of the cell)^2, a product of sinc functions of k_x h and k_y h, and Y the reflected
9 x 9 spectral kernel integrated over the cell's depth.  The integral converges only algebraically: the
evanescent growth of the two vertical moments cancels the decay between the cell and its image.

THE SUBTRACTION.  Y_static, the same with the static image of the interface (Rongved's problem, from
scripts/data/static_interface_image.json), is subtracted under the integral sign.  What is measured is the
contribution of each shell Q_{j-1} < q h < Q_j of the lateral wavenumber, Q_j = 10 * 2^j, relative to the
whole block, with and without the subtraction; the local order of convergence is log2 of the ratio of the
contributions of successive shells (the shell from Q to 2Q of a tail ~ Q^-p contributes ~ Q^-p).

NUMERICS.
- Y depends on |k| only up to a rotation of the 9-state: Y(k) = T(phi) Y(q, 0) M^-1 T(phi)^T M.  The frame
  kernel Y(q, 0) is computed once per radial node in 40-digit arithmetic (mpmath): in double precision the
  P and SV terms of the mode sum cancel as (c q / omega)^4, which at omega h / beta = 0.17 already leaves
  errors of 6e-7 at q h = 20 and makes the remainder noise beyond q h ~ 100.
- The azimuthal integral of L(q, phi) T Y T^T is a trapezoid rule of 4 q h + 64 points (L is periodic and
  analytic); the radial one Gauss-Legendre of 12 points on panels of width pi/2 in q h (L oscillates with
  period pi in q h), fixed for every cutoff, so a shell's contribution does not depend on the others.
- The part q h < 10 is common to every cutoff and enters only the normalisation; it is integrated with the
  1/k_z branch points at q = omega/alpha and omega/beta as panel ends, under a square-root substitution.

Checks printed: the rotation against direct evaluation at three azimuths, the 40-digit kernel at 40 and 60
digits, and one shell with the azimuth rule doubled and the radial panels halved.

Run:  PYTHONPATH=. python scripts/measure_interface_static_subtraction.py   (about 10 minutes on 4 cores)
"""

import json
from itertools import pairwise
from multiprocessing import Pool
from pathlib import Path

import mpmath as mp
import numpy as np
import sympy as sp
from numpy.polynomial.legendre import leggauss
from scipy.special import spherical_jn

from cubic_scattering.effective_contrasts import ReferenceMedium

A = ReferenceMedium(5.0, 3.0, 2.5)
B = ReferenceMedium(6.5, 3.6, 2.8)
H = 0.05
OMEGA = 10.0
DPS = 40
CUTS = [10, 20, 40, 80, 160, 320, 640]  # q h
VP = [
    (0, 0),
    (1, 1),
    (2, 2),
    (1, 2),
    (0, 2),
    (0, 1),
]  # Voigt order of the state: zz, xx, yy, xy, zy, zx
M = np.diag([1, 1, 1, 1, 1, 1, 0.5, 0.5, 0.5])

# ---------------------------------------------------------------------------------------------- static image
z, zp = sp.symbols("z zp", real=True)
q_ = sp.Symbol("q", positive=True)
lamA, muA, lamB, muB = sp.symbols("lamA muA lamB muB", positive=True)
_LOC = {"z": z, "zp": zp, "q": q_, "lamA": lamA, "muA": muA, "lamB": lamB, "muB": muB}
_JSON = Path(__file__).resolve().parent / "data" / "static_interface_image.json"
_G = sp.Matrix(
    [
        [sp.sympify(e, locals=_LOC) for e in row]
        for row in json.loads(_JSON.read_text())["entries"]
    ]
)
assert _G.free_symbols <= set(_LOC.values())


def _strip(expr):
    """expr * e^{-q (z + zp)}: every entry is e^{q (z + zp)} times a rational function."""
    e = sp.powsimp(sp.expand(expr * sp.exp(-q_ * (z + zp))), combine="exp")
    assert not e.has(sp.exp)
    return e


def _static_frame_symbolic():
    """The static 9 x 9 image in the frame of k = (q, 0), e^{q(z+zp)} removed, as sympy (rows u, eps)."""
    drec = lambda e, j: [sp.diff(e, z), sp.I * q_ * e, sp.Integer(0)][j]
    dsrc = lambda e, l: [-sp.diff(e, zp), sp.I * q_ * e, sp.Integer(0)][l]
    out = sp.zeros(9, 9)
    for i in range(3):
        for k in range(3):
            out[i, k] = _G[i, k]
    for b, (m, n) in enumerate(VP):
        for i in range(3):
            v = (
                dsrc(_G[i, m], n)
                if m == n
                else (dsrc(_G[i, m], n) + dsrc(_G[i, n], m)) / 2
            )
            out[i, 3 + b] = v
    for a, (p, r) in enumerate(VP):
        for k in range(3):
            out[3 + a, k] = (
                drec(_G[p, k], p) if p == r else drec(_G[r, k], p) + drec(_G[p, k], r)
            )

        def e(j, i, mm, nn):
            return drec(dsrc(_G[i, mm], nn), j)

        def term(mm, nn, p=p, r=r):
            return e(p, p, mm, nn) if p == r else e(p, r, mm, nn) + e(r, p, mm, nn)

        for b, (m, n) in enumerate(VP):
            out[3 + a, 3 + b] = term(m, n) if m == n else (term(m, n) + term(n, m)) / 2
    return out.applyfunc(_strip)


_STATIC = None


def _vmom(a, n):
    """int_{-2h}^{0} z^n e^{a z} dz, n = 0 or 1, in mpmath (Re a >= 0 or small |a|)."""
    h2 = 2 * mp.mpf(H)
    if abs(a * h2) < mp.mpf("1e-4"):
        return mp.quad(lambda t: t**n * mp.exp(a * t), [-h2, 0])
    e = mp.exp(-a * h2)
    return (1 - e) / a if n == 0 else (e * (1 + a * h2) - 1) / a**2


def static_frame(q):
    """The static image's frame kernel, lateral moments excluded, 9 x 9 mp matrix (field 1, source 1)."""
    global _STATIC
    if _STATIC is None:
        _STATIC = sp.lambdify(
            (q_, z, zp, lamA, muA, lamB, muB), _static_frame_symbolic(), "mpmath"
        )
    pts = [(-0.01, -0.02), (-0.03, -0.02), (-0.01, -0.05), (-0.03, -0.05)]
    m4 = mp.inverse(mp.matrix([[1, a, b, a * b] for a, b in pts]))
    vals = [
        mp.matrix(_STATIC(q, mp.mpf(a), mp.mpf(b), A.lam, A.mu, B.lam, B.mu))
        for a, b in pts
    ]
    out = mp.matrix(9, 9)
    for i, (ea, eb) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)]):
        c = sum((m4[i, j] * vals[j] for j in range(4)), mp.matrix(9, 9))
        out += c * (_vmom(q, ea) * _vmom(q, eb))
    return out


# ------------------------------------------------------------------------------------------- dynamic, 40 digits
def _med(r):
    rho, al, be = mp.mpf(r.rho), mp.mpf(r.alpha), mp.mpf(r.beta)
    return {
        "alpha": al,
        "beta": be,
        "rho": rho,
        "lam": rho * (al**2 - 2 * be**2),
        "mu": rho * be**2,
    }


def _state(k, e):
    out = [e[0], e[1], e[2]]
    for p, r in VP:
        v = mp.mpc(0, 1) / 2 * (k[p] * e[r] + k[r] * e[p])
        out.append(v if p == r else 2 * v)
    return out


def _modes(q, om, med, sign):
    """[(k_z, state, weight)] for P, SV, SH with bilinear-unit polarisations, as stratified_march.mode_family."""
    out = []
    sh = [mp.mpf(0), mp.mpf(0), mp.mpf(1)]
    for c, wave in ((med["alpha"], "P"), (med["beta"], "SV"), (med["beta"], "SH")):
        kz = mp.sqrt(mp.mpc((om / c) ** 2 - q * q))
        kz = -kz if mp.im(kz) < 0 else kz
        k = [sign * kz, mp.mpf(q), mp.mpf(0)]
        if wave == "P":
            e = [ki * c / om for ki in k]
        elif wave == "SH":
            e = sh
        else:
            e = [
                (k[1] * sh[2] - k[2] * sh[1]) * c / om,
                (k[2] * sh[0] - k[0] * sh[2]) * c / om,
                (k[0] * sh[1] - k[1] * sh[0]) * c / om,
            ]
        out.append((kz, _state(k, e), mp.mpc(0, 1) / (2 * med["rho"] * c * c * kz)))
    return out


def _ut(st, med):
    eps = st[3:]
    return [
        st[0],
        st[1],
        st[2],
        med["lam"] * (eps[0] + eps[1] + eps[2]) + 2 * med["mu"] * eps[0],
        med["mu"] * eps[5],
        med["mu"] * eps[4],
    ]


def dynamic_frame(q):
    """The reflected frame kernel at k = (q, 0), lateral moments excluded, 9 x 9 mp matrix."""
    ma, mb, om = _med(A), _med(B), mp.mpf(OMEGA)
    dn_a, up_a, dn_b = _modes(q, om, ma, 1), _modes(q, om, ma, -1), _modes(q, om, mb, 1)
    lhs, rhs = mp.matrix(6, 6), mp.matrix(6, 3)
    for j in range(3):
        ua, ub, ud = _ut(up_a[j][1], ma), _ut(dn_b[j][1], mb), _ut(dn_a[j][1], ma)
        for i in range(6):
            lhs[i, j], lhs[i, 3 + j], rhs[i, j] = ua[i], -ub[i], -ud[i]
    r_d = mp.inverse(lhs) * rhs  # rows 0..2: reflected up-going amplitudes
    pair = [1, 1, 1, 1, 1, 1, mp.mpf(1) / 2, mp.mpf(1) / 2, mp.mpf(1) / 2]
    out = mp.matrix(9, 9)
    for m in range(3):
        kzm, dm, _ = up_a[m]
        vm = _vmom(-mp.mpc(0, 1) * kzm, 0)
        for n in range(3):
            kzn, dn, wn = dn_a[n]
            c = vm * r_d[m, n] * wn * _vmom(-mp.mpc(0, 1) * kzn, 0)
            for i in range(9):
                for j in range(9):
                    out[i, j] += c * dm[i] * dn[j] * pair[j]
    return out


def frames(q_h):
    """(dynamic, static) frame kernels at q h, as complex numpy arrays."""
    mp.mp.dps = DPS
    q = mp.mpf(q_h) / H
    d = np.array(dynamic_frame(q).tolist(), dtype=complex)
    s = np.array(static_frame(q).tolist(), dtype=complex)
    return d, s


# ----------------------------------------------------------------------------------------- the azimuth
def rotation(phi):
    """9 x 9 rotations of the (u, engineering strain) state about z, shape (n, 9, 9)."""
    c, s = np.cos(phi), np.sin(phi)
    n = len(phi)
    Q = np.zeros((n, 3, 3))
    Q[:, 0, 0] = 1
    Q[:, 1, 1] = c
    Q[:, 1, 2] = -s
    Q[:, 2, 1] = s
    Q[:, 2, 2] = c
    T = np.zeros((n, 9, 9))
    T[:, :3, :3] = Q
    for a, (p, r) in enumerate(VP):
        for b, (m, k) in enumerate(VP):
            v = Q[:, p, m] * Q[:, r, k] + (Q[:, p, k] * Q[:, r, m] if m != k else 0.0)
            v = v / 2.0 if m != k else v
            T[:, 3 + a, 3 + b] = 2.0 * v if p != r else v
    return T


def lateral(q_h, phi):
    """L = (h^2 * 2 j0(k_x h) * 2 j0(k_y h))^2: the lateral moments of field and source cell."""
    lat = (
        H**2
        * 4
        * spherical_jn(0, q_h * np.cos(phi))
        * spherical_jn(0, q_h * np.sin(phi))
    )
    return lat * lat


def azimuthal(q_h, Y, n_phi=None):
    """int_0^{2 pi} L T (Y M^-1) T^T M dphi, by the trapezoid rule."""
    n_phi = n_phi or int(4 * np.ceil(q_h) + 64)
    phi = 2 * np.pi * np.arange(n_phi) / n_phi
    T = rotation(phi)
    w = lateral(q_h, phi) * (2 * np.pi / n_phi)
    return np.einsum("p,pab,bc,pdc->ad", w, T, Y @ np.linalg.inv(M), T) @ M


# ----------------------------------------------------------------------------------------- the radial rule
def panels(a, b, width=np.pi / 2, order=12):
    """Gauss nodes and weights in q h on [a, b], panels of at most `width`."""
    n = max(1, int(np.ceil((b - a) / width)))
    x, w = leggauss(order)
    edges = np.linspace(a, b, n + 1)
    nodes = np.concatenate(
        [(e0 + e1) / 2 + (e1 - e0) / 2 * x for e0, e1 in pairwise(edges)]
    )
    wts = np.concatenate([(e1 - e0) / 2 * w for e0, e1 in pairwise(edges)])
    return nodes, wts


def low_nodes(order=24):
    """Nodes on [0, 10] in q h with the branch points omega h/alpha, omega h/beta as panel ends (sqrt maps)."""
    bp = sorted([OMEGA * H / A.alpha, OMEGA * H / A.beta])
    x, w = leggauss(order)
    u, wu = (x + 1) / 2, w / 2
    nodes, wts = [], []

    def towards(a, b):  # singular at b: q = b - (b - a) (1 - u)^2 ... cluster at b
        t = 1 - u
        nodes.append(b - (b - a) * t**2)
        wts.append(wu * 2 * (b - a) * t)

    def away(a, b):  # singular at a
        nodes.append(a + (b - a) * u**2)
        wts.append(wu * 2 * (b - a) * u)

    towards(0.0, bp[0])
    mid = (bp[0] + bp[1]) / 2
    away(bp[0], mid)
    towards(mid, bp[1])
    away(bp[1], 1.0)
    n2, w2 = panels(1.0, 10.0, width=0.5, order=16)
    return np.concatenate(nodes + [n2]), np.concatenate(wts + [w2])


def contribution(nodes, wts, pool):
    """(dynamic, static) integrals over the given radial nodes: (1/4 pi^2) int q dq dphi L Y."""
    kern = pool.map(frames, list(nodes))
    dyn, stat = np.zeros((9, 9), complex), np.zeros((9, 9), complex)
    for qh, w, (d, s) in zip(nodes, wts, kern):
        f = w * (qh / H) / H / (4 * np.pi**2)  # q dq with q = qh / H, dq = d(qh) / H
        dyn += f * azimuthal(qh, d)
        stat += f * azimuthal(qh, s)
    return dyn, stat


def main() -> None:
    mp.mp.dps = DPS
    d40 = frames(320.0)[
        0
    ]  # also builds the static kernel once, before the workers fork
    mp.mp.dps = 60
    d60 = np.array(dynamic_frame(mp.mpf(320) / H).tolist(), dtype=complex)
    mp.mp.dps = DPS
    print(
        f"check: frame kernel at q h = 320, 40 vs 60 digits: {np.abs(d40 - d60).max() / np.abs(d60).max():.1e}",
        flush=True,
    )
    with Pool(4) as pool:
        low_n, low_w = low_nodes()
        low_d, _ = contribution(low_n, low_w, pool)
        shells = []
        for a, b in pairwise(CUTS):
            n, w = panels(a, b)
            shells.append(contribution(n, w, pool))
            print(f"  shell {a}-{b}: {len(n)} radial nodes", flush=True)
        total = low_d + sum(s[0] for s in shells)
        scale = np.abs(total).max()

        # resolution check on the shell 160-320: azimuth doubled, panels halved
        n, w = panels(160, 320, width=np.pi / 4)
        kern2 = pool.map(frames, list(n))
        dd = sum(
            wi
            * (qh / H)
            / H
            / (4 * np.pi**2)
            * azimuthal(qh, k[0] - k[1], n_phi=int(8 * np.ceil(qh) + 128))
            for qh, wi, k in zip(n, w, kern2)
        )
        ref = shells[4][0] - shells[4][1]
        print(
            f"check: shell 160-320 of the subtracted integrand, finer rules: {np.abs(dd - ref).max() / scale:.1e}"
            f" (contribution {np.abs(ref).max() / scale:.1e})"
        )

    print(
        "\nshell contribution relative to the block (max entry), and local order log2(previous / this):"
    )
    print(
        f"{'q h':>10} {'total':>10} {'order':>6} {'static subtracted':>18} {'order':>6}"
    )
    prev = None
    for (a, b), (d, s) in zip(pairwise(CUTS), shells, strict=True):
        t, r = np.abs(d).max() / scale, np.abs(d - s).max() / scale
        o1 = f"{np.log2(prev[0] / t):6.2f}" if prev else " " * 6
        o2 = f"{np.log2(prev[1] / r):6.2f}" if prev else " " * 6
        print(f"{a:>4}-{b:<5} {t:10.2e} {o1} {r:18.2e} {o2}")
        prev = (t, r)


if __name__ == "__main__":
    main()
