"""Gate: interface blocks of cells touching the interface against their plane-wave integral, the algebraic tail
integrated analytically.

Plan: docs/2026-10-10-stratified-reference-legendre-cells-3d.md (13 October: K1, touching pairs).
Paper: LatexPDFs/InterfaceCouplingIntegrals, the omega^2 section and Table "verification".

THE TAIL.  For cells touching the interface the plane-wave integrand decays only algebraically.  Its vertical
moments are int v^k e^{c(v - 1)} dv, c = q h, and split exactly as P_k(1/c) + e^{-2c} Q_k(1/c) (P, Q polynomials:
P_0 = 1/c, P_k = 1/c - (k/c) P_(k-1); Q_0 = -1/c, Q_k = -(-1)^k/c - (k/c) Q_(k-1)).  A cell touching the interface
has an O(1) prefactor, so the integrand is
    Y = Y_alg + (terms with e^{-2qh}),   Y_alg = sum_p q^p sum_n e^{i n phi} H_p[n] x lateral moments x phase,
Y_alg the product of the receiver's and the source's P parts: finitely many powers p, nine harmonics n.

THE ALGEBRAIC PART OVER THE WHOLE PLANE, IN CLOSED FORM.  By Weber-Schafheitlin,
    (1/4 pi^2) FP int q^p e^{i n phi} e^{i k.rho} d^2k = (i^n e^{i n theta} / 2 pi) M(p, |n|) rho^(-p-2) (-1)^(n<0 ? n : 0),
    M(p, nu) = 2^(p+1) Gamma((nu + p + 2)/2) / Gamma((nu - p)/2),
the finite part taken by continuation in p (the convention of the potentials Phi_j).  The lateral moments and the
phase turn this into a 2-D integral over the separation u = sigma + Delta (units of h) against the exact lateral
weights w(sigma) = int P_E(v) (v - sigma)^F dv: pieces [Delta - 2, Delta], [Delta, Delta + 2] per axis.  On a piece
with the origin at a corner, in polar coordinates about it, every monomial u_x^a u_y^b times the kernel is rho^d A(t):
the radial integral is exact (with its pole and logarithm when d + 2 = 0), and the angular one is a Gauss rule on
a smooth integrand.  M, the radial integral and the scaling h^(-eps) are expanded together in eps = (shift of p)
and the constant term taken: at most one simple pole occurs (M's when the kernel is a polynomial times nothing
singular, the radial one when the kernel is too singular), and the products give the logarithms.  Pieces away
from the origin are smooth: Gauss.

THE REFERENCE.  K = FP int_0^{q_s} Y (head: q^-2 times an entire function, Chebyshev modes exactly)
              + int_{q_s}^{Q} (Y - Y_alg) (Gauss; it falls as e^{-2qh}, below rounding by Q h = 24)
              + FP int_0^oo Y_alg (above) - FP int_0^{q_s} Y_alg (each power q^(p+1) against the Chebyshev modes of
                the entire lateral harmonics S_n(q)).
No step is shared with the closed form of interface_image (no Phi_j, no 3-D corner pieces).

Checked first at order 0 against the static blocks (known to rounding against an extended-precision reference),
then at order 1 against K1.  Measured 13 October 2026, full 4 x 10 x 9 x 9 blocks (closed form vs plane waves;
moving q_s from h^-1 to h^-1/2 changes the reference by the amount in brackets, its own precision):
    order 0: self-image 5.2e-14 [3.8e-14], edge 7.5e-14 [8.7e-14], corner 4.9e-13 [4.3e-13], across 3.4e-14 [4.2e-14];
    order 1: self-image 4.8e-14 [1.1e-13], edge 1.3e-13 [1.1e-13], corner 4.2e-13 [2.1e-13], across 3.9e-14 [6.2e-14].
RECORDED: the head of Y_alg below q_s first by Chebyshev sampling (as for Y): it and the plane integral are each
about 20 times the block and cancel, and the sampling's rounding, amplified by the powers down to q^-7, left 3e-10
(and an imaginary part of that size).  The head is now in closed form too (``lateral_head``).

Run:  PYTHONPATH=. python scripts/gate_interface_touching_blocks.py [order ...]
"""

import sys
import time
from functools import cache

import mpmath as mp
import numpy as np
import sympy as sp
from numpy.polynomial import chebyshev as C
from numpy.polynomial import polynomial as nppoly

from cubic_scattering import interface_spectral as isp
from cubic_scattering.effective_contrasts import ReferenceMedium
from cubic_scattering.gauss_rules import gauss_interval
from cubic_scattering.interface_image import image_block

A = ReferenceMedium(5.0, 3.0, 2.5)
B = ReferenceMedium(6.5, 3.6, 2.8)
H = 0.05
N_TEST, N_SOURCE = 4, 10
PAIRS = [
    ((-1, 0, 0), (-1, 0, 0)),
    ((-1, 2, 0), (-1, 0, 0)),
    ((-1, 2, 2), (-1, 0, 0)),
    ((1, 0, 0), (-1, 0, 0)),
]


# ---------------------------------------------------------------------------
# The algebraic part of the vertical moments and of the integrand
# ---------------------------------------------------------------------------


@cache
def _p_coeffs(n_max: int) -> list:
    """P_k(u) = sum_j c[k][j] u^(j+1), u = 1/|c|: the algebraic part of E_k (``isp._exp_moments``), k <= n_max."""
    out, prev = [], None
    for k in range(n_max + 1):
        cur = np.zeros(n_max + 1)
        cur[0] = 1.0
        if k:
            cur[1:] -= k * prev[:-1]
        out.append(cur)
        prev = cur
    return out


def _vertical_alg(poly, zc, s, a_max) -> dict:
    """The algebraic part of V[a] = int_cell f(v) z^a e^{s q z} dz as {power of q: array (a_max + 1)}; empty if the
    cell does not touch the interface (then the prefactor is exponentially small)."""
    if not np.isclose(s * zc, -H):
        return {}
    n_max = len(poly) - 1 + a_max
    pc = _p_coeffs(n_max)
    out: dict = {}
    for a in range(a_max + 1):
        p = (
            nppoly.polymul(poly, nppoly.polypow([zc, H], a))
            if a
            else np.asarray(poly, float)
        )
        for k, pk in enumerate(p):
            if pk == 0:
                continue
            sign = (-1.0) ** k if s < 0 else 1.0  # E_k(-c) = (-1)^k E_k(c)
            for j, cj in enumerate(pc[k]):
                if cj:
                    out.setdefault(-(j + 1), np.zeros(a_max + 1))[a] += (
                        H * pk * cj * sign * H ** (-(j + 1))
                    )
    return out


def alg_harmonics(order: int, kind: str, rec, src) -> dict:
    """{(l, e): {p: (9 harmonics, 81)}}: Y_alg's coefficient of q^p, in the form of ``isp._static_frames``."""
    Cf, m0 = isp.frame_laurent(kind, order, isp._key(A), isp._key(B))
    s = 1 if kind == "R" else -1
    exps_f = isp.SOURCE_EXPONENTS_QUARTIC[:N_TEST]
    exps_s = isp.SOURCE_EXPONENTS_QUARTIC[:N_SOURCE]
    pf, ps = isp._vertical_polys(exps_f, True), isp._vertical_polys(exps_s, False)
    zr, zs = rec[0] * H, src[0] * H
    out = {}
    for l, polyl in pf.items():
        vr = _vertical_alg(polyl, zr, s, Cf.shape[2] - 1)
        for e, polye in ps.items():
            vs = _vertical_alg(polye, zs, 1, Cf.shape[3] - 1)
            acc: dict = {}
            for pr, ar in vr.items():
                for pq, bq in vs.items():
                    for m in range(Cf.shape[4]):
                        p = m0 + m + pr + pq
                        g = np.einsum("ijab,a,b->ij", Cf[:, :, :, :, m], ar, bq)
                        acc[p] = acc.get(p, 0) + g
            out[l, e] = {
                p: isp._rotate_harmonics((g @ isp._M_INV)[None])[0]
                for p, g in acc.items()
            }
    return out


# ---------------------------------------------------------------------------
# The lateral finite-part integrals
# ---------------------------------------------------------------------------


@cache
def _weight(E: int, F: int) -> tuple:
    """w(sigma) = int P_E(v) (v - sigma)^F dv on the two pieces [-2, 0], [0, 2]: power coefficients in sigma."""
    v, sg = sp.symbols("v sigma")
    pe = sp.legendre(E, v)
    lo = sp.integrate(pe * (v - sg) ** F, (v, -1, sg + 1))
    hi = sp.integrate(pe * (v - sg) ** F, (v, sg - 1, 1))
    return tuple(
        np.array([float(c) for c in sp.Poly(sp.expand(x), sg).all_coeffs()[::-1]])
        for x in (lo, hi)
    )


def _compose(c: np.ndarray, delta: float) -> np.ndarray:
    out = np.zeros(len(c))
    for k, ck in enumerate(c):
        out[: k + 1] += ck * nppoly.polypow([-delta, 1.0], k)[: k + 1]
    return out


@cache
def _m_laurent(p: int, nu: int) -> tuple:
    """M(p + eps, nu) = 2^(p+1+eps) Gamma((nu+p+2+eps)/2) / Gamma((nu-p-eps)/2) as (m_-1, m_0, m_1) in eps."""
    with mp.workdps(40):
        f = lambda e: (
            mp.mpf(2) ** (p + 1 + e)
            * mp.gamma((nu + p + 2 + e) / 2)
            * mp.rgamma((nu - p - e) / 2)
        )
        if nu + p + 2 <= 0 and (nu + p + 2) % 2 == 0:  # pole of the numerator
            g = mp.taylor(
                lambda e: e * f(e) if e != 0 else mp.limit(lambda t: t * f(t), 0), 0, 2
            )
            return float(g[0]), float(g[1]), float(g[2])
        t = mp.taylor(f, 0, 1)
        return 0.0, float(t[0]), float(t[1])


_T_NODES = 48


def _box_laurent(
    a: float, b: float, sx: int, sy: int, alpha: int, beta: int, n: int, s: int
) -> tuple:
    """int over [0,a] x [0,b] (reflected by sx, sy) of (sx x)^alpha (sy y)^beta e^{i n theta_actual} rho^(s - eps)
    as (B_-1, B_0, B_1) in eps: polar about the corner, the radial integral exact, Gauss in the angle."""
    d = alpha + beta + s
    tstar = np.arctan2(b, a)
    out = np.zeros(3, dtype=complex)
    for t0, t1, which in ((0.0, tstar, "a"), (tstar, np.pi / 2, "b")):
        t, w = gauss_interval(_T_NODES, t0, t1)
        ct, st = np.cos(t), np.sin(t)
        R = a / ct if which == "a" else b / st
        ang = (sx * ct) ** alpha * (sy * st) ** beta * (sx * ct + 1j * sy * st) ** n * w
        L = np.log(R)
        if d + 2 == 0:  # int_0^R rho^(-1-eps) = -1/eps + log R - eps log^2 R / 2
            out += np.array([-np.sum(ang), np.sum(ang * L), -np.sum(ang * L**2) / 2])
        else:  # R^(d+2-eps)/(d+2-eps)
            base = R ** (d + 2) / (d + 2)
            out += np.array(
                [0.0, np.sum(ang * base), np.sum(ang * base * (1 / (d + 2) - L))]
            )
    return tuple(out)


def _box_regular(
    lo: tuple, hi: tuple, cx: np.ndarray, cy: np.ndarray, n: int, s: int
) -> tuple:
    """A piece away from the origin: (0, B_0, B_1) by a tensor Gauss rule, B_1 = -int ... log rho."""
    (x, wx), (y, wy) = (
        gauss_interval(40, lo[0], hi[0]),
        gauss_interval(40, lo[1], hi[1]),
    )
    X, Y = np.meshgrid(x, y, indexing="ij")
    rho = np.hypot(X, Y)
    f = (
        nppoly.polyval(X, cx)
        * nppoly.polyval(Y, cy)
        * ((X + 1j * Y) / rho) ** n
        * rho**s
        * np.outer(wx, wy)
    )
    return 0.0, np.sum(f), -np.sum(f * np.log(rho))


@cache
def lateral_fp(p: int, n: int, ea: tuple, ec: tuple, delta: tuple) -> complex:
    """T = (1/4 pi^2) FP int d^2k q^p e^{i n phi} L_a(k) L_c(k) e^{i k.Delta h}, q in km^-1."""
    s = -p - 2
    nu = abs(n)
    m = _m_laurent(p, nu)
    pref = 1j**n * (-1.0) ** (nu if n < 0 else 0) / (2 * np.pi)
    pieces_x = [_compose(c, float(delta[0])) for c in _weight(ea[1], ec[1])]
    pieces_y = [_compose(c, float(delta[1])) for c in _weight(ea[2], ec[2])]
    ends = lambda dl: [(dl - 2, dl), (dl, dl + 2)]
    bsum = np.zeros(3, dtype=complex)
    for (x0, x1), cx in zip(ends(delta[0]), pieces_x, strict=True):
        for (y0, y1), cy in zip(ends(delta[1]), pieces_y, strict=True):
            if 0 in (x0, x1) and 0 in (y0, y1):
                a, b = max(abs(x0), abs(x1)), max(abs(y0), abs(y1))
                sx = -1 if x0 < 0 else 1
                sy = -1 if y0 < 0 else 1
                for al, ca in enumerate(cx):
                    for be, cb in enumerate(cy):
                        if ca and cb:
                            bsum += (
                                ca
                                * cb
                                * np.array(_box_laurent(a, b, sx, sy, al, be, n, s))
                            )
            else:
                bsum += np.array(_box_regular((x0, y0), (x1, y1), cx, cy, n, s))
    # scaling: rho = h u, rho^(s - eps) = h^(s - eps) u^(s - eps); d^2 rho -> h^2; lateral moments carry h^4 / h^2
    hs = (
        H ** (s + 4)
    )  # h^4 (moments) x h^s, the d^2u of the s-form being in units of h (absorbed in the weights)
    h_eps = (1.0, -np.log(H), np.log(H) ** 2 / 2)
    # constant term of M(eps) * B(eps) * h^(-eps)
    mm = (m[0], m[1], m[2])
    const = 0.0
    for i, mi in zip((-1, 0, 1), mm, strict=True):
        for j, bj in zip((-1, 0, 1), bsum, strict=True):
            k = -(i + j)
            if 0 <= k <= 2:
                const += mi * bj * h_eps[k]
    return pref * hs * const


# ---------------------------------------------------------------------------
# The reference
# ---------------------------------------------------------------------------


def fp_alg_plane(order, kind, rec, src) -> np.ndarray:
    """FP int Y_alg d^2k (1/4 pi^2), shape (n_test, n_source, 9, 9)."""
    exps_f = isp.SOURCE_EXPONENTS_QUARTIC[:N_TEST]
    exps_s = isp.SOURCE_EXPONENTS_QUARTIC[:N_SOURCE]
    delta = (rec[1] - src[1], rec[2] - src[2])
    out = np.zeros((N_TEST, N_SOURCE, 81), dtype=complex)
    for (l, e), coeffs in alg_harmonics(order, kind, rec, src).items():
        for a, ea in enumerate(exps_f):
            if ea[0] != l:
                continue
            for c, ec in enumerate(exps_s):
                if ec[0] != e:
                    continue
                for p, harm in coeffs.items():
                    for k, n in enumerate(isp._HARM):
                        out[a, c] += lateral_fp(p, int(n), ea, ec, delta) * harm[k]
    return out.reshape(N_TEST, N_SOURCE, 9, 9)


@cache
def _cheb_shifted(k: int) -> tuple:
    """Power coefficients of T_k(2t - 1)."""
    t = sp.Symbol("t")
    return tuple(
        float(c) for c in sp.Poly(sp.chebyshevt(k, 2 * t - 1), t).all_coeffs()[::-1]
    )


def _chebyshev_fp(values: np.ndarray, x: np.ndarray, qs: float, powers: list) -> dict:
    """{power r: FP int_0^{q_s} q^r S(q) dq} for S given at the Chebyshev nodes x (on [-1, 1])."""
    nc = len(x)
    coef = np.einsum("kn,n...->k...", np.linalg.inv(C.chebvander(x, nc - 1)), values)
    out = {}
    for r in powers:
        acc = 0
        for k in range(nc):
            co = _cheb_shifted(k)
            val = 0.0
            for j, cj in enumerate(co):
                e = r + j
                val += (
                    float(cj)
                    * (np.log(qs) if e == -1 else qs ** (e + 1) / (e + 1))
                    * qs ** (-j)
                )
            acc = acc + val * coef[k]
        out[r] = acc
    return out


@cache
def _rect_poly_moments(x0, x1, y0, y1, deg: int) -> np.ndarray:
    """int over the rectangle of u_x^i u_y^j, i, j <= deg, exactly."""
    ix = np.array([(x1 ** (i + 1) - x0 ** (i + 1)) / (i + 1) for i in range(deg + 1)])
    iy = np.array([(y1 ** (j + 1) - y0 ** (j + 1)) / (j + 1) for j in range(deg + 1)])
    return np.outer(ix, iy)


@cache
def _bessel_poly(n: int, k: int) -> np.ndarray:
    """(u_x + i sgn(n) u_y)^|n| (u_x^2 + u_y^2)^k as a 2-D power-coefficient array."""
    nu, sg = abs(n), (1 if n >= 0 else -1)
    out = np.zeros((nu + 2 * k + 1, nu + 2 * k + 1), dtype=complex)
    base = np.zeros((nu + 1, nu + 1), dtype=complex)
    for i in range(nu + 1):  # binomial (u_x + i sg u_y)^nu
        base[i, nu - i] = float(sp.binomial(nu, i)) * (1j * sg) ** (nu - i)
    rad = np.zeros((2 * k + 1, 2 * k + 1))
    for i in range(k + 1):
        rad[2 * i, 2 * (k - i)] = float(sp.binomial(k, i))
    for (i, j), v in np.ndenumerate(base):
        if v:
            out[i : i + 2 * k + 1, j : j + 2 * k + 1] += v * rad
    return out


@cache
def lateral_head(
    p: int, n: int, ea: tuple, ec: tuple, delta: tuple, qs: float, kmax: int = 30
) -> complex:
    """(1/4 pi^2) FP int_{q < q_s} d^2k q^p e^{i n phi} L_a L_c e^{i k.Delta h}, exactly: J_n(q rho) by its Taylor
    series, FP int_0^{q_s} q^e dq = q_s^(e+1)/(e+1) (log q_s at e = -1), and the polynomials rho^(|n|+2k) e^{i n theta}
    integrated against the lateral weights over each piece."""
    nu = abs(n)
    pref = 1j**n * (-1.0) ** (nu if n < 0 else 0) / (2 * np.pi)
    pieces_x = [_compose(c, float(delta[0])) for c in _weight(ea[1], ec[1])]
    pieces_y = [_compose(c, float(delta[1])) for c in _weight(ea[2], ec[2])]
    ends = lambda dl: [(dl - 2, dl), (dl, dl + 2)]
    total = 0.0
    for k in range(kmax + 1):
        # J_nu(x) = sum_k (-1)^k (x/2)^(nu+2k) / (k! (nu+k)!), x = q h |u|
        ck = (-1) ** k / (
            2.0 ** (nu + 2 * k) * float(sp.factorial(k) * sp.factorial(nu + k))
        )
        e = p + 1 + nu + 2 * k
        fq = np.log(qs) if e == -1 else qs ** (e + 1) / (e + 1)
        poly = _bessel_poly(n, k)
        mom = 0.0
        for (x0, x1), cx in zip(ends(delta[0]), pieces_x, strict=True):
            for (y0, y1), cy in zip(ends(delta[1]), pieces_y, strict=True):
                w2 = np.outer(cx, cy)
                full = np.zeros(
                    (w2.shape[0] + poly.shape[0], w2.shape[1] + poly.shape[1]),
                    dtype=complex,
                )
                for (i, j), v in np.ndenumerate(w2):
                    if v:
                        full[i : i + poly.shape[0], j : j + poly.shape[1]] += v * poly
                rm = _rect_poly_moments(x0, x1, y0, y1, max(full.shape) - 1)
                mom += np.sum(full * rm[: full.shape[0], : full.shape[1]])
        total += ck * H ** (nu + 2 * k) * fq * mom
    return pref * H**4 * total


def fp_alg_head(order, kind, rec, src, qsh: float) -> np.ndarray:
    """FP int_0^{q_s} I_alg dq, in closed form (``lateral_head``)."""
    qs = qsh / H
    exps_f = isp.SOURCE_EXPONENTS_QUARTIC[:N_TEST]
    exps_s = isp.SOURCE_EXPONENTS_QUARTIC[:N_SOURCE]
    delta = (rec[1] - src[1], rec[2] - src[2])
    out = np.zeros((N_TEST, N_SOURCE, 81), dtype=complex)
    for (l, e), coeffs in alg_harmonics(order, kind, rec, src).items():
        for a, ea in enumerate(exps_f):
            if ea[0] != l:
                continue
            for c, ec in enumerate(exps_s):
                if ec[0] != e:
                    continue
                for p, harm in coeffs.items():
                    for k, n in enumerate(isp._HARM):
                        out[a, c] += (
                            lateral_head(p, int(n), ea, ec, delta, qs) * harm[k]
                        )
    return out.reshape(N_TEST, N_SOURCE, 9, 9)


def radial(order, kind, rec, src, qh: np.ndarray, alg: bool = False) -> np.ndarray:
    """I(q) per unit q at nodes q = qh / h (alg: the algebraic part only)."""
    orig = isp._exp_moments
    if alg:
        isp._exp_moments = _alg_exp_moments
    try:
        return np.stack(
            [
                H
                * isp._static_integral(
                    order,
                    kind,
                    rec,
                    src,
                    H,
                    isp._key(A),
                    isp._key(B),
                    np.array([v]),
                    np.array([1.0]),
                    N_SOURCE,
                    N_TEST,
                )
                for v in qh
            ]
        )
    finally:
        isp._exp_moments = orig


def _alg_exp_moments(n_max, c):
    c = np.asarray(c, float)
    a = np.abs(c)
    out = np.empty((n_max + 1, c.size))
    pc = _p_coeffs(n_max)
    for k in range(n_max + 1):
        out[k] = sum(cj * a ** (-(j + 1)) for j, cj in enumerate(pc[k]) if cj)
    sgn = np.where(c < 0, -1.0, 1.0)
    for k in range(1, n_max + 1, 2):
        out[k] *= sgn
    return out


def fp_head_full(order, kind, rec, src, qsh: float, n_cheb: int = 24) -> np.ndarray:
    """FP int_0^{q_s} I dq: I = q^-N x entire, N = -(1 + the frame kernel's lowest power of q) (2 for the static
    and omega^2 terms, 4 for omega^4)."""
    _, m0 = isp.frame_laurent(kind, order, isp._key(A), isp._key(B))
    n_sing = -(m0 + 1)
    qs = qsh / H
    x = np.cos(np.pi * (np.arange(n_cheb) + 0.5) / n_cheb)
    q = qs * (x + 1) / 2
    vals = radial(order, kind, rec, src, q * H) * (q**n_sing)[:, None, None, None, None]
    return _chebyshev_fp(vals, x, qs, [-n_sing])[-n_sing]


def body(order, kind, rec, src, qsh: float, qmax_h: float) -> np.ndarray:
    qh, w = isp._panels(qsh, qmax_h, 0.5, 20)
    full = radial(order, kind, rec, src, qh)
    alg = radial(order, kind, rec, src, qh, alg=True)
    return np.einsum("n,n...->...", w / H, full - alg)


def reference(order, rec, src, qsh=1.0, qmax_h=24.0):
    kind = "R" if rec[0] < 0 else "T"
    return (
        fp_head_full(order, kind, rec, src, qsh)
        + body(order, kind, rec, src, qsh, qmax_h)
        + fp_alg_plane(order, kind, rec, src)
        - fp_alg_head(order, kind, rec, src, qsh)
    )


def main() -> None:
    orders = [int(a) for a in sys.argv[1:]] or [0, 1]
    ok = True
    for order in orders:
        for rec, src in PAIRS:
            t0 = time.time()
            ref = reference(order, rec, src)
            t1 = time.time()
            closed = image_block(rec, src, H, A, B, order=order)
            scale = np.abs(closed).max()
            err = np.abs(ref - closed).max() / scale
            imag = np.abs(ref.imag).max() / scale
            alt = reference(
                order, rec, src, qsh=0.5
            )  # moving q_s must not change the sum
            split = np.abs(alt - ref).max() / scale
            good = err < 1e-12
            ok &= good
            print(
                f"order {order}, {rec} <- {src}: closed form vs plane waves {err:.1e} (imaginary part {imag:.1e},"
                f" q_s moved {split:.1e}; reference {t1 - t0:.0f} s)  {'PASS' if good else 'FAIL'}",
                flush=True,
            )
    print("ALL PASS" if ok else "SOME FAILED")


if __name__ == "__main__":
    main()
