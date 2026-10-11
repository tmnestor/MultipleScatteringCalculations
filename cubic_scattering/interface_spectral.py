"""Blocks of cells near a welded interface at many frequencies: a short plane-wave integral per frequency and
two corrections computed once.

Paper: LatexPDFs/InterfaceCouplingIntegrals, Section "The next term: the order omega^2".

THE SPLIT.  For the reflected coupling of two cells in medium A, or the transmitted coupling of a cell in A to
one in B, the block is K = (1/4 pi^2) int Y(k; omega) d^2k, Y the integrand of the plane-wave representation
(the mode sum against the cells' moments).  At large q = |k|, Y = Y0 + omega^2 Y1 + O(omega^4 / q^4 relative),
Y0 the static field (Rongved's problem) and Y1 the next term (Mathematica/InterfaceOmega2.wl), neither
depending on omega.  With a cutoff Q0,

    K(omega) = D(omega) + A0 + omega^2 A1 + E,
    D  = (1/4 pi^2) int_{q < Q0} Y d^2k                     per frequency, double-precision mode sums;
    A0 = K0 - (1/4 pi^2) int_{q < Q0} Y0 d^2k               once: K0 the static block in closed form
                                                            (``interface_image.image_block``);
    A1 = (1/4 pi^2) int_{q > Q0} Y1 d^2k                    once: to Q1, and the tail beyond Q1 by its q^-4 law;
    E  = (1/4 pi^2) int_{q > Q0} (Y - Y0 - omega^2 Y1) d^2k  dropped; it converges as Q0^-6.

For a cell touching the interface with its own image, measured against 40-digit references
(scripts/gate_interface_spectral.py): E ~ 4e-10 of the block at Q0 h = 10 when omega h / beta = 1/6, and
~ 6e-11 at Q0 h = 40 when omega h / beta = 1, while the double-precision mode sums of D stay below 1e-11.

THE STATIC AND OMEGA^2 INTEGRANDS.  In the frame whose x axis lies along k, both kernels are X times a
polynomial in (z, zp) with coefficients that are Laurent polynomials in q, X = e^{q(z + zp)} (reflected) or
e^{-q(z - zp)} (transmitted).  The 9 x 9 frame kernel follows from the 3 x 3 displacement kernel by the
receiver derivatives (d_z, i q, 0) and the source derivatives (-d_zp, i q, 0); its vertical moments over the
cells are elementary (``_exp_moments``), and the frame is rotated to k by the 9-state rotation T(phi):
Y(k) = T Y(q, 0) M^-1 T^T M, times the cells' lateral moments and phase.

Conventions as ``stratified_march``: axes z = 0 (down), x = 1, y = 2; medium A (``above``) fills z < 0, B
(``below``) z > 0; centres in units of h, z odd.
"""

import itertools
import json
from functools import cache
from pathlib import Path

import numpy as np
import sympy as sp
from numpy.polynomial import legendre as npleg
from numpy.polynomial import polynomial as nppoly
from numpy.typing import NDArray
from scipy.special import spherical_jn

from .effective_contrasts import ReferenceMedium
from .gauss_rules import gauss_interval, gauss_legendre
from .graded_voxel.basis import SOURCE_EXPONENTS_QUARTIC
from .graded_voxel.farfield import monomial_fourier
from .interface_image import MIRROR_9, image_block
from .resonance_tmatrix import VOIGT_PAIRS
from .stratified_march import (
    SOURCE_PAIRING,
    interface_rt,
    mode_matrices,
    receiver_moments,
    source_moments_local,
)

_ROOT = Path(__file__).resolve().parent.parent
_Z, _ZP = sp.symbols("z zp", real=True)
_Q = sp.Symbol("q", positive=True)
_NAMES = ("lamA", "muA", "rhoA", "lamB", "muB", "rhoB")
_MSYM = dict(zip(_NAMES, sp.symbols(" ".join(_NAMES), positive=True), strict=True))
_M_INV = np.diag(1.0 / SOURCE_PAIRING)


# ---------------------------------------------------------------------------
# The kernels in the frame of k
# ---------------------------------------------------------------------------


@cache
def _general(kind: str, order: int) -> sp.Matrix:
    """The 3 x 3 displacement kernel (rows u_(z,x,y), columns force (z,x,y)) in general materials, sympy.

    order 0: the static image (R) or transmitted static field (T), scripts/data/static_interface_*.json;
    order 1: the omega^2 term, Mathematica/InterfaceOmega2.json.
    """
    loc = {"z": _Z, "zp": _ZP, "q": _Q, **_MSYM}
    if order == 0:
        name = "image" if kind == "R" else "transmission"
        rows = json.loads(
            (_ROOT / "scripts" / "data" / f"static_interface_{name}.json").read_text()
        )["entries"]
        return sp.Matrix([[sp.sympify(e, locals=loc) for e in row] for row in rows])
    from sympy.parsing.mathematica import parse_mathematica

    rows = json.loads((_ROOT / "Mathematica" / "InterfaceOmega2.json").read_text())[
        kind + "1"
    ]
    subs = {sp.Symbol(k): v for k, v in loc.items()}
    return sp.Matrix([[parse_mathematica(e).subs(subs) for e in row] for row in rows])


def _materials(above: tuple, below: tuple) -> dict:
    """Exact rationals of the media's binary values, keyed by the sympy symbols."""
    out = {}
    for tag, (al, be, rho) in (("A", above), ("B", below)):
        al, be, rho = (sp.Rational(float(v)) for v in (al, be, rho))
        out[_MSYM["lam" + tag]] = rho * (al**2 - 2 * be**2)
        out[_MSYM["mu" + tag]] = rho * be**2
        out[_MSYM["rho" + tag]] = rho
    return out


def frame_coefficients(G: sp.Matrix, s: int) -> dict:
    """The 9 x 9 frame kernel of G = X P, X = e^{s q z + q zp}, as {(a, b): 9 x 9 sympy matrix in q}: the
    coefficient of z^a zp^b X.  Rows (u, engineering strain), columns (force, engineering stress)."""
    x = _Q * (s * _Z + _ZP)
    for e in G:
        assert {a.args[0] for a in e.atoms(sp.exp)} <= {x}, e
    P = G.applyfunc(lambda e: sp.cancel(e.replace(sp.exp, lambda a: sp.Integer(1))))
    dz = lambda e: s * _Q * e + sp.diff(e, _Z)
    dzp = lambda e: _Q * e + sp.diff(e, _ZP)
    drec = lambda e, j: [dz(e), sp.I * _Q * e, sp.Integer(0)][j]
    dsrc = lambda e, l: [-dzp(e), sp.I * _Q * e, sp.Integer(0)][l]
    out = sp.zeros(9, 9)
    for i in range(3):
        for k in range(3):
            out[i, k] = P[i, k]
    for b, (m, n) in enumerate(VOIGT_PAIRS):
        for i in range(3):
            out[i, 3 + b] = (
                dsrc(P[i, m], n)
                if m == n
                else (dsrc(P[i, m], n) + dsrc(P[i, n], m)) / 2
            )
    for a, (p, r) in enumerate(VOIGT_PAIRS):
        for k in range(3):
            out[3 + a, k] = (
                drec(P[p, k], p) if p == r else drec(P[r, k], p) + drec(P[p, k], r)
            )

        def term(mm, nn, p=p, r=r):
            e = lambda j, i: drec(dsrc(P[i, mm], nn), j)
            return e(p, p) if p == r else e(p, r) + e(r, p)

        for b, (m, n) in enumerate(VOIGT_PAIRS):
            out[3 + a, 3 + b] = term(m, n) if m == n else (term(m, n) + term(n, m)) / 2
    coeffs: dict = {}
    for i in range(9):
        for j in range(9):
            num, den = sp.fraction(sp.cancel(sp.together(out[i, j])))
            if num == 0:
                continue
            for (a, b), c in sp.Poly(sp.expand(num), _Z, _ZP).terms():
                coeffs.setdefault((a, b), sp.zeros(9, 9))[i, j] += c / den
    return coeffs


@cache
def frame_laurent(
    kind: str, order: int, above: tuple, below: tuple
) -> tuple[NDArray, int]:
    """The frame kernel's coefficients as Laurent polynomials in q, for media (alpha, beta, rho).

    Returns:
        (C, m0): C[i, j, a, b, m] complex, shape (9, 9, na, nb, nm), the coefficient of q^(m0 + m) z^a zp^b X.
    """
    G = _general(kind, order).subs(_materials(above, below))
    co = frame_coefficients(G, 1 if kind == "R" else -1)
    terms = {}
    for (a, b), mat in co.items():
        for i in range(9):
            for j in range(9):
                e = sp.together(sp.expand(mat[i, j]))
                if e == 0:
                    continue
                num, den = sp.fraction(e)
                dpoly = sp.Poly(den, _Q)
                assert len(dpoly.terms()) == 1, (
                    f"frame_laurent: denominator not a power of q: {den}"
                )
                (dk,), dc = dpoly.terms()[0]
                for (nk,), nc in sp.Poly(sp.expand(num), _Q).terms():
                    terms[i, j, a, b, nk - dk] = complex(sp.N(nc / dc, 30))
    na = 1 + max(k[2] for k in terms)
    nb = 1 + max(k[3] for k in terms)
    m0 = min(k[4] for k in terms)
    nm = 1 + max(k[4] for k in terms) - m0
    C = np.zeros((9, 9, na, nb, nm), dtype=complex)
    for (i, j, a, b, m), v in terms.items():
        C[i, j, a, b, m - m0] = v
    return C, m0


def _exp_moments(n_max: int, c: NDArray) -> NDArray:
    """E[n] = int_{-1}^{1} v^n e^{c (v - sgn c)} dv for real c, n = 0 .. n_max, shape (n_max + 1, N).

    For |c| > 2 (n_max + 2) the forward recurrence J_n = (1 - (-1)^n e^{-2|c|}) / |c| - (n / |c|) J_{n-1}
    (stable there); below, a 40-point Gauss rule (exact to rounding for |c| <= 2 (n_max + 2) <= 20).
    For c < 0, E_n(c) = (-1)^n E_n(|c|).
    """
    c = np.asarray(c, dtype=float)
    a = np.abs(c)
    out = np.empty((n_max + 1, c.size))
    big = a > 2 * (n_max + 2)
    if big.any():
        ab = a[big]
        e2 = np.exp(-2 * ab)
        j = (1 - e2) / ab
        out[0, big] = j
        for n in range(1, n_max + 1):
            j = (1 - (-1) ** n * e2) / ab - n / ab * j
            out[n, big] = j
    if (~big).any():
        x, w = gauss_legendre(40)
        g = np.exp(np.outer(a[~big], x - 1)) * w
        for n in range(n_max + 1):
            out[n, ~big] = g @ x**n
    sgn = np.where(c < 0, -1.0, 1.0)
    for n in range(1, n_max + 1, 2):
        out[n] *= sgn
    return out


def _vertical_moments(
    poly: NDArray, zc: float, h: float, s: int, q: NDArray, a_max: int
) -> NDArray:
    """V[a] = int_cell f(v) z^a e^{s q z} dz, z = zc + h v, for a = 0 .. a_max, shape (a_max + 1, N).

    f has power coefficients ``poly`` in v; the exponential is referred to the cell's edge nearest the
    interface, where it is largest, so nothing overflows.
    """
    n_max = len(poly) - 1 + a_max
    E = _exp_moments(n_max, s * q * h)
    scale = h * np.exp(s * q * zc + q * h)
    out = np.empty((a_max + 1, len(q)))
    for a in range(a_max + 1):
        p = (
            nppoly.polymul(poly, nppoly.polypow([zc, h], a))
            if a
            else np.asarray(poly, float)
        )
        out[a] = scale * (p @ E[: len(p)])
    return out


def rotation(phi: NDArray) -> NDArray:
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
    for a, (p, r) in enumerate(VOIGT_PAIRS):
        for b, (m, k) in enumerate(VOIGT_PAIRS):
            v = Q[:, p, m] * Q[:, r, k] + (Q[:, p, k] * Q[:, r, m] if m != k else 0.0)
            v = v / 2.0 if m != k else v
            T[:, 3 + a, 3 + b] = 2.0 * v if p != r else v
    return T


# ---------------------------------------------------------------------------
# Quadrature in the lateral wavenumber
# ---------------------------------------------------------------------------


def _panels(
    a: float, b: float, width: float, order: int = 12
) -> tuple[NDArray, NDArray]:
    n = max(1, int(np.ceil((b - a) / width)))
    edges = np.linspace(a, b, n + 1)
    parts = [gauss_interval(order, e0, e1) for e0, e1 in itertools.pairwise(edges)]
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def radial_rule(
    qh_max: float, branch_qh: list, width: float, order: int = 24
) -> tuple[NDArray, NDArray]:
    """Nodes and weights in q h on [0, qh_max] with the branch points as panel ends, the square-root
    singularity at each removed by a quadratic map, then panels of at most ``width``."""
    bp = sorted(b for b in set(branch_qh) if 0 < b < qh_max)
    x, w = gauss_legendre(order)
    u, wu = (x + 1) / 2, w / 2
    nodes, wts = [], []
    if bp:
        t = 1 - u
        nodes.append(bp[0] - bp[0] * t**2)
        wts.append(wu * 2 * bp[0] * t)
        for a, b in itertools.pairwise(bp):
            m = (a + b) / 2
            nodes += [a + (m - a) * u**2, b - (b - m) * (1 - u) ** 2]
            wts += [wu * 2 * (m - a) * u, wu * 2 * (b - m) * (1 - u)]
        end = min(qh_max, max(1.0, 2 * bp[-1]))
        nodes.append(bp[-1] + (end - bp[-1]) * u**2)
        wts.append(wu * 2 * (end - bp[-1]) * u)
    else:
        end = 0.0
    if end < qh_max:
        n2, w2 = _panels(end, qh_max, width)
        nodes.append(n2)
        wts.append(w2)
    return np.concatenate(nodes), np.concatenate(wts)


def _n_phi(qh: float, reach: float) -> int:
    """Azimuthal trapezoid nodes at q h: the cells' lateral moments and the phase over ``reach`` (in h)."""
    return int(4 * np.ceil((4 * qh + 2 * qh * reach + 64) / 4))


def _grid(
    qh: NDArray, wqh: NDArray, h: float, reach: float
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Flattened polar points: (index of the radial node, kx, ky, weight of (1/4 pi^2) d^2k)."""
    idx, kx, ky, wt = [], [], [], []
    for i, (x, w) in enumerate(zip(qh, wqh, strict=True)):
        n = _n_phi(x, reach)
        phi = 2 * np.pi * np.arange(n) / n
        q = x / h
        idx.append(np.full(n, i))
        kx.append(q * np.cos(phi))
        ky.append(q * np.sin(phi))
        wt.append(np.full(n, w / h * q * (2 * np.pi / n) / (4 * np.pi**2)))
    return (
        np.concatenate(idx),
        np.concatenate(kx),
        np.concatenate(ky),
        np.concatenate(wt),
    )


# ---------------------------------------------------------------------------
# The cells
# ---------------------------------------------------------------------------


def _lateral(kx: NDArray, ky: NDArray, h: float, exps: tuple, field: bool) -> NDArray:
    """The lateral (x, y) factor of each field function (Legendre) or source monomial, shape (N, n)."""
    if field:
        f = {
            (i, e): 2.0 * 1j**e * spherical_jn(e, k * h)
            for i, k in ((1, kx), (2, ky))
            for e in range(3)
        }
    else:
        f = {
            (i, e): monomial_fourier(e, -k * h)
            for i, k in ((1, kx), (2, ky))
            for e in range(5)
        }
    return np.stack([h**2 * f[1, e[1]] * f[2, e[2]] for e in exps], -1)


def _vertical_polys(exps: tuple, field: bool) -> dict:
    """{vertical exponent: power coefficients in v} of P_l(v) (field) or v^e (source)."""
    out = {}
    for e in exps:
        l = e[0]
        if l not in out:
            out[l] = npleg.leg2poly([0] * l + [1]) if field else np.eye(l + 1)[l]
    return out


#: Angular harmonics of the rotated frame: T(phi) G T(phi)^T has e^{i n phi} with |n| <= 4 only, since the
#: entries of T are trigonometric polynomials of degree two.
_HARM = np.arange(-4, 5)


@cache
def _harmonic_rotation() -> NDArray:
    """H[n, a, d, b, c]: T(phi) G T(phi)^T = sum_n e^{i n phi} sum_bc H[n, a, d, b, c] G[b, c], exactly."""
    n = 16  # samples; aliasing would need harmonics beyond 8
    phi = 2 * np.pi * np.arange(n) / n
    T = rotation(phi)
    K = np.einsum("kab,kdc->kadbc", T, T)
    return np.einsum("nk,kadbc->nadbc", np.exp(-1j * np.outer(_HARM, phi)) / n, K)


def _static_integral(
    order: int,
    kind: str,
    rec: tuple,
    src: tuple,
    h: float,
    above: tuple,
    below: tuple,
    qh: NDArray,
    wqh: NDArray,
    n_source: int,
    n_test: int,
    chunk: int = 50000,
) -> NDArray:
    """(1/4 pi^2) int Y0 or Y1 d^2k over the radial nodes ``qh`` (with weights), shape (n_test, n_source, 9, 9).

    The frame kernel at each radial node is split into its nine angular harmonics once; the azimuthal rule
    then only sums the cells' lateral moments against e^{i n phi}.
    """
    C, m0 = frame_laurent(kind, order, above, below)
    s = 1 if kind == "R" else -1
    exps_f = SOURCE_EXPONENTS_QUARTIC[:n_test]
    exps_s = SOURCE_EXPONENTS_QUARTIC[:n_source]
    pf, ps = _vertical_polys(exps_f, True), _vertical_polys(exps_s, False)
    q = qh / h
    zr, zs = rec[0] * h, src[0] * h
    Vr = {l: _vertical_moments(p, zr, h, s, q, C.shape[2] - 1) for l, p in pf.items()}
    Vs = {e: _vertical_moments(p, zs, h, 1, q, C.shape[3] - 1) for e, p in ps.items()}
    qpow = q[None, :] ** (m0 + np.arange(C.shape[4]))[:, None]  # (nm, Nq)
    Cq = np.einsum("ijabm,mn->nijab", C, qpow)  # (Nq, 9, 9, na, nb)
    harm = _harmonic_rotation()
    groups, rot = {}, {}
    for l in pf:
        for e in ps:
            fa = [a for a, x in enumerate(exps_f) if x[0] == l]
            sc = [c for c, x in enumerate(exps_s) if x[0] == e]
            groups[l, e] = (np.array(fa), np.array(sc))
            g = np.einsum("nijab,an,bn->nij", Cq, Vr[l], Vs[e]) @ _M_INV
            rot[l, e] = (
                np.einsum("hadbc,nbc->nhad", harm, g) * SOURCE_PAIRING
            ).reshape(len(q), 9, 81)
    dx, dy = (rec[1] - src[1]) * h, (rec[2] - src[2]) * h
    reach = np.hypot(rec[1] - src[1], rec[2] - src[2])
    idx, kx, ky, wt = _grid(qh, wqh, h, reach)
    starts = np.searchsorted(idx, np.arange(len(q) + 1))
    out = np.zeros((n_test, n_source, 81), dtype=complex)
    i = 0
    while i < len(q):
        j = i + 1
        while j < len(q) and starts[j + 1] - starts[i] <= chunk:
            j += 1
        sl = slice(starts[i], starts[j])
        coef = wt[sl] * np.exp(1j * (kx[sl] * dx + ky[sl] * dy))
        lf = _lateral(kx[sl], ky[sl], h, exps_f, True) * coef[:, None]
        ls = _lateral(kx[sl], ky[sl], h, exps_s, False)
        eph = np.exp(1j * np.outer(np.arctan2(ky[sl], kx[sl]), _HARM))
        x = (lf[:, :, None] * eph[:, None, :]).reshape(len(lf), -1)  # (P, n_test * 9)
        for r in range(i, j):
            p = slice(starts[r] - starts[i], starts[r + 1] - starts[i])
            S = (x[p].T @ ls[p]).reshape(n_test, 9, n_source)
            for key, (fa, sc) in groups.items():
                out[np.ix_(fa, sc)] += np.einsum(
                    "anc,nk->ack", S[np.ix_(fa, np.arange(9), sc)], rot[key][r]
                )
        i = j
    return out.reshape(n_test, n_source, 9, 9)


# ---------------------------------------------------------------------------
# The blocks
# ---------------------------------------------------------------------------


def _key(m: ReferenceMedium) -> tuple:
    return (float(m.alpha), float(m.beta), float(m.rho))


def _check(rec: tuple, src: tuple) -> str:
    if rec[0] % 2 == 0 or src[0] % 2 == 0:
        raise ValueError(
            "interface_spectral: cell centres must have odd z in units of h."
        )
    if src[0] > 0:
        raise ValueError(
            "interface_spectral: internal routines take a source in A; mirror first."
        )
    return "R" if rec[0] < 0 else "T"


def _mirrored(c: tuple) -> tuple:
    return (-c[0], c[1], c[2])


def _mirror(m: NDArray) -> NDArray:
    """A block computed for the mirrored cells (z -> -z, media exchanged) mapped back, as ``image_block``."""
    n_test, n_source = m.shape[:2]
    sa = np.array([(-1.0) ** e[0] for e in SOURCE_EXPONENTS_QUARTIC[:n_test]])
    sc = np.array([(-1.0) ** e[0] for e in SOURCE_EXPONENTS_QUARTIC[:n_source]])
    return np.einsum("a,c,i,j,acij->acij", sa, sc, MIRROR_9, MIRROR_9, m)


def frequency_independent_parts(
    rec_centre: tuple[int, int, int],
    src_centre: tuple[int, int, int],
    h: float,
    above: ReferenceMedium,
    below: ReferenceMedium,
    n_source: int = 10,
    n_test: int = 4,
    q0h: float = 10.0,
    q1h: float = 160.0,
) -> tuple[NDArray, NDArray]:
    """(A0, A1) of the split, computed once per arrangement of cells and reused at every frequency.

    A0 = K0 - (1/4 pi^2) int_{q < Q0} Y0, K0 the closed-form static block; A1 = (1/4 pi^2) int_{q > Q0} Y1, to
    Q1 by the plane-wave rule and beyond by the q^-4 law of its last octave.

    Args:
        rec_centre: Receiver cell centre in units of h, (z, x, y); z odd.
        src_centre: Source cell centre in units of h.
        h: Cell half-width.
        above: Medium A (z < 0).
        below: Medium B (z > 0).
        n_source: Source monomials (10, 20 or 35).
        n_test: Field functions (4 or 10).
        q0h: The cutoff Q0 h.
        q1h: The plane-wave rule of A1 ends at Q1 h, a power-of-two multiple of Q0 h.

    Returns:
        (A0, A1), each shape (n_test, n_source, 9, 9), complex.
    """
    rec, src = tuple(rec_centre), tuple(src_centre)
    if src[0] > 0:
        a0, a1 = frequency_independent_parts(
            _mirrored(rec), _mirrored(src), h, below, above, n_source, n_test, q0h, q1h
        )
        return _mirror(a0), _mirror(a1)
    kind = _check(rec, src)
    ka, kb = _key(above), _key(below)
    reach = np.hypot(rec[1] - src[1], rec[2] - src[2])
    width = np.pi / 2 / max(1.0, reach / 4)
    k0 = image_block(rec, src, h, above, below, n_source, n_test)
    qh, w = _panels(0.0, q0h, width, 16)
    a0 = k0 - _static_integral(0, kind, rec, src, h, ka, kb, qh, w, n_source, n_test)
    octaves = []
    lo = q0h
    while lo < q1h * (1 - 1e-12):
        qh, w = _panels(lo, 2 * lo, width)
        octaves.append(
            _static_integral(1, kind, rec, src, h, ka, kb, qh, w, n_source, n_test)
        )
        lo *= 2
    a1 = sum(octaves) + octaves[-1] / 15.0
    return a0, a1


def truncated_blocks(
    pairs: list[tuple[tuple[int, int, int], tuple[int, int, int]]],
    h: float,
    omega: float,
    above: ReferenceMedium,
    below: ReferenceMedium,
    n_source: int = 10,
    n_test: int = 4,
    q0h: float = 10.0,
    chunk: int = 20000,
) -> list[NDArray]:
    """D = (1/4 pi^2) int_{q < Q0} Y d^2k for many pairs (receiver, source) with sources in A, on one grid
    (a source in B is mirrored onto A by ``interface_block``).

    The reflection and transmission coefficients and the mode states are computed once per point of a
    polar grid that every pair shares (radial nodes with the branch points of both media as panel ends;
    the azimuthal rule set by the widest lateral offset).

    Returns:
        One block per pair, shape (n_test, n_source, 9, 9), complex.
    """
    for rec, src in pairs:
        _check(rec, src)
    reach = max(np.hypot(r[1] - s[1], r[2] - s[2]) for r, s in pairs)
    branch = [omega * h / c for m in (above, below) for c in (m.alpha, m.beta)]
    qh, w = radial_rule(q0h, branch, np.pi / 2 / max(1.0, reach / 4))
    _, kx_all, ky_all, wt_all = _grid(qh, w, h, reach)
    out = [np.zeros((n_test, n_source, 9, 9), dtype=complex) for _ in pairs]
    for i0 in range(0, len(kx_all), chunk):
        kx, ky, wt = (
            kx_all[i0 : i0 + chunk],
            ky_all[i0 : i0 + chunk],
            wt_all[i0 : i0 + chunk],
        )
        st_a, _, w_a, kz_a = mode_matrices(kx, ky, omega, above)
        st_b, _, _, kz_b = mode_matrices(kx, ky, omega, below)
        rt = interface_rt(kx, ky, omega, above, below)
        n = len(kx)
        lat = np.stack([np.zeros(n), kx, ky], -1)
        k_dn_a = [lat + np.outer(kz_a[:, m], [1, 0, 0]) for m in range(3)]
        k_up_a = [lat - np.outer(kz_a[:, m], [1, 0, 0]) for m in range(3)]
        k_dn_b = [lat + np.outer(kz_b[:, m], [1, 0, 0]) for m in range(3)]
        src_right = (
            st_a[:, :, :3].swapaxes(1, 2) * w_a[:, :3, None] * SOURCE_PAIRING
        )  # (n, 3 modes, 9)
        for (rec, src), acc in zip(pairs, out, strict=True):
            cr, cs = np.array(rec, float) * h, np.array(src, float) * h
            if rec[0] < 0:
                k_rec, states, amp = k_up_a, st_a[:, :, 3:], rt["R_d"]
            else:
                k_rec, states, amp = k_dn_b, st_b[:, :, :3], rt["T_d"]
            fm = np.stack(
                [
                    receiver_moments(k, h, n_test) * np.exp(1j * (k @ cr))[:, None]
                    for k in k_rec
                ],
                1,
            )  # (n, 3, a)
            sm = np.stack(
                [
                    source_moments_local(k, h, n_source)
                    * np.exp(-1j * (k @ cs))[:, None]
                    for k in k_dn_a
                ],
                1,
            )  # (n, 3, c)
            left = np.einsum("pma,pim->pmai", fm * wt[:, None, None], states).reshape(
                n * 3, -1
            )
            right = np.einsum("pmn,pnc,pnj->pmcj", amp, sm, src_right).reshape(
                n * 3, -1
            )
            acc += (
                (left.T @ right).reshape(n_test, 9, n_source, 9).transpose(0, 2, 1, 3)
            )
    return out


def interface_block(
    rec_centre: tuple[int, int, int],
    src_centre: tuple[int, int, int],
    h: float,
    omega: float,
    above: ReferenceMedium,
    below: ReferenceMedium,
    n_source: int = 10,
    n_test: int = 4,
    q0h: float = 10.0,
    q1h: float = 160.0,
    parts: tuple[NDArray, NDArray] | None = None,
) -> NDArray:
    """The reflected (cells on one side) or transmitted (opposite sides) block at one frequency: D + A0 + w^2 A1.

    ``parts`` = ``frequency_independent_parts(...)`` with the same arguments, passed when sweeping frequency.
    """
    rec, src = tuple(rec_centre), tuple(src_centre)
    a0, a1 = (
        parts
        if parts is not None
        else frequency_independent_parts(
            rec, src, h, above, below, n_source, n_test, q0h, q1h
        )
    )
    if src[0] > 0:
        d = _mirror(
            truncated_blocks(
                [(_mirrored(rec), _mirrored(src))],
                h,
                omega,
                below,
                above,
                n_source,
                n_test,
                q0h,
            )[0]
        )
    else:
        d = truncated_blocks(
            [(rec, src)], h, omega, above, below, n_source, n_test, q0h
        )[0]
    return d + a0 + omega**2 * a1
