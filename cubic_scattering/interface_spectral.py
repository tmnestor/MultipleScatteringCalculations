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
~ 6e-11 at Q0 h = 40 when omega h / beta = 1, while the rounding of D's double-precision mode sums (their P and
SV terms cancel) is 1e-11 to 3e-11 at the lower frequency and below 1e-12 at the higher.

THE WINDOW.  With a smooth high-pass window chi (``Window``: 0 below start Q0, 1 above Q0) in place of the
sharp cutoff,

    D  = (1/4 pi^2) int_{q < Q0} [Y - chi (Y0 + omega^2 Y1)] d^2k,
    A0 = K0 - (1/4 pi^2) int (1 - chi) Y0 d^2k,     A1 = (1/4 pi^2) int chi Y1 d^2k,

the same sum with the same remainder E (chi = 1 beyond Q0): the ramp integrals (``window_parts``) move from D
to A0, A1.  The sharp cutoff makes A0, A1 ring in space and decay slowly with the cells' lateral offset; the
smooth window makes them local (scripts/measure_interface_correction_locality.py).

THE SWEEP.  D is evaluated in the same frame: Y(q, phi) = T(phi) Y(q, 0) M^-1 T(phi)^T M times the cells'
lateral moments and phase, and the rotation holds only the harmonics e^{i n phi}, |n| <= 4.  So

    D = sum over radial nodes  w q  sum_n  R_n(q; omega) . S_n(q),

R_n the harmonics of the dynamic frame kernel (one 6 x 6 interface solve per radial node, the cells' vertical
moments in closed form) and S_n the azimuthal integrals of the lateral moments and phase, which depend on
neither the frequency, the media nor the depths.  S_n is tabulated once per lateral offset as piecewise
Chebyshev series (``lateral_harmonics``) and interpolated to each frequency's radial rule, whose panel ends
follow the branch points omega / c.  Per block and frequency this costs tens of milliseconds, against seconds
for the mode sum at every (q, phi) (``_truncated_blocks_direct``, kept as its reference: the two agree to
3e-15 .. 2e-11, the latter the rounding above).

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
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import sympy as sp
from numpy.polynomial import legendre as npleg
from numpy.polynomial import polynomial as nppoly
from numpy.typing import NDArray
from scipy.special import betainc, spherical_jn

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
    order 1: the omega^2 term, Mathematica/InterfaceOmega2.json;
    order 2: the omega^4 term, Mathematica/InterfaceOmega4.json (checked exactly against the python twin
    scripts/derive_interface_omega4.py at its rational media).
    """
    loc = {"z": _Z, "zp": _ZP, "q": _Q, **_MSYM}
    if order == 0:
        name = "image" if kind == "R" else "transmission"
        rows = json.loads(
            (_ROOT / "scripts" / "data" / f"static_interface_{name}.json").read_text()
        )["entries"]
        return sp.Matrix([[sp.sympify(e, locals=loc) for e in row] for row in rows])
    from sympy.parsing.mathematica import parse_mathematica

    name = {1: "InterfaceOmega2.json", 2: "InterfaceOmega4.json"}[order]
    rows = json.loads((_ROOT / "Mathematica" / name).read_text())[kind + str(order)]
    subs = {sp.Symbol(k): v for k, v in loc.items()}
    canon = _canon_exp(kind)
    # one exponential in the form frame_coefficients expects, exp(q (s z + zp))
    return sp.Matrix(
        [
            [
                sp.powsimp(parse_mathematica(e).subs(subs), combine="exp").replace(
                    sp.exp, canon
                )
                for e in row
            ]
            for row in rows
        ]
    )


def _canon_exp(kind: str):
    """exp(a) -> exp(q (s z + zp)) when a equals it (frame_coefficients compares the exponent structurally)."""
    target = _Q * ((1 if kind == "R" else -1) * _Z + _ZP)

    def f(a):
        if sp.expand(a - target) != 0:
            raise ValueError(f"_general: unexpected exponential exp({a})")
        return sp.exp(target)

    return f


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
    if not terms:  # the reflection vanishes identically (identical media)
        return np.zeros((9, 9, 1, 1, 1), dtype=complex), 0
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
    qh_max: float,
    branch_qh: list,
    width: float,
    order: int = 24,
    breaks: tuple = (),
) -> tuple[NDArray, NDArray]:
    """Nodes and weights in q h on [0, qh_max] with the branch points as panel ends, the square-root
    singularity at each removed by a quadratic map, then panels of at most ``width``, split also at
    ``breaks`` (plain panel ends, e.g. the start of a window's ramp; ignored among the branch points)."""
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
        ends = [end, *sorted(b for b in breaks if end < b < qh_max), qh_max]
        for e0, e1 in itertools.pairwise(ends):
            n2, w2 = _panels(e0, e1, width)
            nodes.append(n2)
            wts.append(w2)
    return np.concatenate(nodes), np.concatenate(wts)


@dataclass(frozen=True)
class Window:
    """A smooth high-pass window chi(q): 0 for q < start Q0, 1 for q >= Q0, rising on the ramp between.

    With it the split is K = D + A0 + omega^2 A1 with D = int_{q < Q0} [Y - chi (Y0 + omega^2 Y1)],
    A0 = K0 - int (1 - chi) Y0 and A1 = int chi Y1: the same sum, the same dropped remainder (beyond Q0,
    where chi = 1), but A0 and A1 are high-pass parts with no sharp edge, so they decay quickly with the
    cells' lateral offset instead of ringing (the sharp cutoff, ``window=None``, is the Gibbs case).

    Attributes:
        start: The ramp starts at start * Q0 (0 < start < 1).
        shape: "cinf", the C-infinity step e^{-1/t} / (e^{-1/t} + e^{-1/(1-t)}); or "poly", the polynomial
            step of degree 2 order + 1 (the regularised incomplete beta I_t(order + 1, order + 1)), C^order.
        order: Smoothness of the "poly" step.
    """

    start: float = 0.5
    shape: str = "cinf"
    order: int = 3

    def __post_init__(self) -> None:
        if not 0.0 < self.start < 1.0:
            raise ValueError("Window: start must lie in (0, 1).")
        if self.shape not in ("cinf", "poly"):
            raise ValueError("Window: shape must be 'cinf' or 'poly'.")

    def __call__(self, qh: NDArray, q0h: float) -> NDArray:
        t = np.clip((np.asarray(qh, float) / q0h - self.start) / (1 - self.start), 0, 1)
        if self.shape == "poly":
            return betainc(self.order + 1, self.order + 1, t)
        with np.errstate(divide="ignore"):
            a = np.where(t > 0, np.exp(-1 / np.where(t > 0, t, 1)), 0.0)
            b = np.where(t < 1, np.exp(-1 / np.where(t < 1, 1 - t, 1)), 0.0)
        return a / (a + b)

    def ramp_edges(self, q0h: float) -> NDArray:
        """Panel ends on the ramp [start Q0, Q0]: 16 panels, enough for the C-infinity step's steep flanks
        (order-12 Gauss on panels of a quarter of the ramp leaves ~5e-11; on a sixteenth, rounding)."""
        return np.linspace(self.start * q0h, q0h, 17)

    def ramp_rule(self, q0h: float) -> tuple[NDArray, NDArray]:
        """Nodes and weights in q h on the ramp [start Q0, Q0], weights times chi."""
        parts = [
            gauss_interval(24, e0, e1)
            for e0, e1 in itertools.pairwise(self.ramp_edges(q0h))
        ]
        qh = np.concatenate([x for x, _ in parts])
        w = np.concatenate([w for _, w in parts])
        return qh, w * self(qh, q0h)


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


def _rotate_harmonics(g: NDArray) -> NDArray:
    """The nine angular harmonics of T(phi) g T(phi)^T M for frame kernels g (N, 9, 9): shape (N, 9, 81)."""
    harm = _harmonic_rotation().reshape(9 * 81, 81)
    out = (g.reshape(len(g), 81) @ harm.T).reshape(len(g), 9, 9, 9)
    return (out * SOURCE_PAIRING).reshape(len(g), 9, 81)


def _static_frames(
    order: int,
    kind: str,
    rec: tuple,
    src: tuple,
    h: float,
    above: tuple,
    below: tuple,
    q: NDArray,
    exps_f: tuple,
    exps_s: tuple,
    rotate: bool = True,
) -> dict:
    """The angular harmonics of the static (order 0) or omega^2 (order 1) frame kernel at wavenumbers q,
    with the cells' vertical moments, {(l, e): (N, 9, 81)}: the form of the dynamic frames of
    ``truncated_blocks``.  ``rotate=False``: the kernel in the frame of k, times M^-1, {(l, e): (N, 9, 9)}."""
    C, m0 = frame_laurent(kind, order, above, below)
    s = 1 if kind == "R" else -1
    pf, ps = _vertical_polys(exps_f, True), _vertical_polys(exps_s, False)
    zr, zs = rec[0] * h, src[0] * h
    Vr = {l: _vertical_moments(p, zr, h, s, q, C.shape[2] - 1) for l, p in pf.items()}
    Vs = {e: _vertical_moments(p, zs, h, 1, q, C.shape[3] - 1) for e, p in ps.items()}
    qpow = q[None, :] ** (m0 + np.arange(C.shape[4]))[:, None]  # (nm, Nq)
    Cq = np.einsum("ijabm,mn->nijab", C, qpow)  # (Nq, 9, 9, na, nb)
    frames = {
        (l, e): np.einsum("nijab,an,bn->nij", Cq, Vr[l], Vs[e]) @ _M_INV
        for l in pf
        for e in ps
    }
    return {k: _rotate_harmonics(v) for k, v in frames.items()} if rotate else frames


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
    exps_f = SOURCE_EXPONENTS_QUARTIC[:n_test]
    exps_s = SOURCE_EXPONENTS_QUARTIC[:n_source]
    rot = _static_frames(order, kind, rec, src, h, above, below, qh / h, exps_f, exps_s)
    groups = {
        (l, e): (
            np.array([a for a, x in enumerate(exps_f) if x[0] == l]),
            np.array([c for c, x in enumerate(exps_s) if x[0] == e]),
        )
        for l, e in rot
    }
    q = qh / h
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
    window: Window | None = None,
    a1_route: str = "closed",
    omega4: bool = False,
) -> tuple:
    """(A0, A1) of the split, or (A0, A1, A2) with ``omega4``, computed once per arrangement of cells and reused
    at every frequency.

    A0 = K0 - (1/4 pi^2) int_{q < Q0} Y0, K0 the closed-form static block; A1 = (1/4 pi^2) int_{q > Q0} Y1, to
    Q1 by the plane-wave rule and beyond by the q^-4 law of its last octave.  With a ``window`` chi, A0 and A1
    gain the ramp integrals of ``window_parts``: A0 = K0 - int (1 - chi) Y0, A1 = int chi Y1.

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
        window: The smooth window (None: the sharp cutoff at Q0).  ``truncated_blocks`` must use the same.
        a1_route: "closed" (default): A1 = K1 - FP int_{q < Q0} Y1, K1 from ``interface_image.image_block(...,
            order=1)``, exact; "octaves": the plane-wave octaves to Q1 and the q^-4 law of the last (``q1h``).
        omega4: Also A2 = (1/4 pi^2) FP int_{q > Q0} Y2 = K2 - FP int_{q < Q0} Y2, K2 the closed-form omega^4 block
            (``image_block(..., order=2)``, any media), so that the dropped
            remainder falls as Q0^-8 instead of Q0^-6.  Not with a window.

    Returns:
        (A0, A1) or (A0, A1, A2), each shape (n_test, n_source, 9, 9), complex.
    """
    if omega4 and window is not None:
        raise ValueError(
            "frequency_independent_parts: omega4 with a window is not implemented."
        )
    rec, src = tuple(rec_centre), tuple(src_centre)
    if src[0] > 0:
        parts_m = frequency_independent_parts(
            _mirrored(rec),
            _mirrored(src),
            h,
            below,
            above,
            n_source,
            n_test,
            q0h,
            q1h,
            window,
            a1_route,
            omega4,
        )
        return tuple(_mirror(a) for a in parts_m)
    kind = _check(rec, src)
    ka, kb = _key(above), _key(below)
    reach = np.hypot(rec[1] - src[1], rec[2] - src[2])
    width = np.pi / 2 / max(1.0, reach / 4)
    k0 = image_block(rec, src, h, above, below, n_source, n_test)
    qh, w = _panels(0.0, q0h, width, 16)
    a0 = k0 - _static_integral(0, kind, rec, src, h, ka, kb, qh, w, n_source, n_test)
    if a1_route == "closed":
        # A1 = int_{q > Q0} Y1 = K1 - FP int_{q < Q0} Y1, K1 the closed-form omega^2 block: no asymptotic tail
        k1 = image_block(rec, src, h, above, below, n_source, n_test, order=1)
        a1 = k1 - fp_low_integral(1, kind, rec, src, h, ka, kb, q0h, n_source, n_test)
    else:  # the plane-wave octaves to Q1 and the q^-4 law of the last (the earlier route, kept to compare)
        octaves = []
        lo = q0h
        while lo < q1h * (1 - 1e-12):
            qh, w = _panels(lo, 2 * lo, width)
            octaves.append(
                _static_integral(1, kind, rec, src, h, ka, kb, qh, w, n_source, n_test)
            )
            lo *= 2
        a1 = sum(octaves) + octaves[-1] / 15.0
    if window is not None:
        r0, r1 = window_parts(rec, src, h, above, below, window, n_source, n_test, q0h)
        a0, a1 = a0 + r0, a1 + r1
    if omega4:
        k2 = image_block(rec, src, h, above, below, n_source, n_test, order=2)
        a2 = k2 - fp_low_integral(2, kind, rec, src, h, ka, kb, q0h, n_source, n_test)
        return a0, a1, a2
    return a0, a1


@cache
def _cheb_mode_moments(n: int, n_sing: int = 2) -> tuple:
    """For T_k(2t - 1), k < n: (sum_{l != N-1} a_kl / (l - N + 1), a_k,N-1), N = n_sing: the finite parts of
    int_0^1 t^(l - N), and the coefficient of the logarithm."""
    t = sp.Symbol("t")
    out = []
    for k in range(n):
        co = sp.Poly(sp.chebyshevt(k, 2 * t - 1), t).all_coeffs()[::-1]
        pw = sum(
            sp.Rational(c) / (j - n_sing + 1)
            for j, c in enumerate(co)
            if j != n_sing - 1
        )
        # no logarithm without a q^-1 term (n_sing = 0: co[-1] would be the last coefficient)
        lg = sp.Rational(co[n_sing - 1]) if 1 <= n_sing <= len(co) else sp.Integer(0)
        out.append((float(pw), float(lg)))
    return tuple(out)


def fp_low_integral(
    order: int,
    kind: str,
    rec: tuple,
    src: tuple,
    h: float,
    above: tuple,
    below: tuple,
    q0h: float,
    n_source: int,
    n_test: int,
    qsh: float | None = None,
    n_cheb: int | None = None,
    head: str = "analytic",
) -> NDArray:
    """(1/4 pi^2) FP int_{q < Q0} Y_order d^2k (Y0, Y1 or Y2), the finite part by continuation in the power of q
    (the convention of the potentials Phi_j of ``interface_image``), shape (n_test, n_source, 9, 9).

    The radial integrand behaves as q^-N times an entire function at q -> 0, N = -(1 + the frame kernel's lowest
    power of q) (2 for the static and omega^2 terms, 4 for omega^4): on [0, q_s] that function is sampled at
    Chebyshev nodes and each mode integrated against q^-N exactly (the q^-1 term giving log q_s, q in km^-1);
    Gauss panels on [q_s, Q0].  q_s h = 0.5 for N = 2 and 3 for N = 4, with 24 and 40 nodes (the sampling's
    rounding is amplified by q_s^(1-N); for N = 4 the window is narrow: 4e-11 of K2 at 40 nodes, 6e-9 at 24 and
    5e-10 at 48).  ``head="analytic"`` (default) replaces the sampling: ``interface_lateral.head``, power series
    of the frame, vertical and lateral moments, every term's finite part exact, with q_s = ``qs_for``.
    """
    if head == "analytic":
        # power series throughout (interface_lateral.head); no sampling
        from .interface_lateral import head as fp_head
        from .interface_lateral import qs_for

        qs_a = qs_for(rec, src, h) if qsh is None else qsh / h
        out = fp_head(order, kind, rec, src, h, above, below, qs_a, n_source, n_test)
        qh, w = _panels(qs_a * h, q0h, 0.5, 20)
        return out + _static_integral(
            order, kind, rec, src, h, above, below, qh, w, n_source, n_test
        )
    _, m0 = frame_laurent(kind, order, above, below)
    n_sing = -(m0 + 1)
    if qsh is None:
        qsh = 0.5 if n_sing <= 2 else 3.0
    if n_cheb is None:
        # q^-4 (omega^4): the head over [0, 3/h] needs 40 nodes (24 left 6e-9 of K2, 48 lose 5e-10 to the
        # q_s^(1-N) amplification of the modes' rounding); measured against K2 and the octaves of Y2
        n_cheb = 24 if n_sing <= 2 else 40
    qs = qsh / h
    x = np.cos(np.pi * (np.arange(n_cheb) + 0.5) / n_cheb)
    q = qs * (x + 1) / 2
    vals = np.stack(
        [
            h
            * _static_integral(
                order,
                kind,
                rec,
                src,
                h,
                above,
                below,
                np.array([v * h]),
                np.array([1.0]),
                n_source,
                n_test,
            )
            * v**n_sing
            for v in q
        ]
    )
    coef = np.einsum(
        "kn,n...->k...",
        np.linalg.inv(np.polynomial.chebyshev.chebvander(x, n_cheb - 1)),
        vals,
    )
    mom = _cheb_mode_moments(n_cheb, n_sing)
    head = (
        np.einsum("k,k...->...", np.array([m[0] for m in mom]), coef)
        + np.log(qs) * np.einsum("k,k...->...", np.array([m[1] for m in mom]), coef)
    ) * qs ** (1 - n_sing)
    width = _radial_width(np.hypot(rec[1] - src[1], rec[2] - src[2]))
    qh, w = _panels(qsh, q0h, min(width, 0.5), 20)
    return head + _static_integral(
        order, kind, rec, src, h, above, below, qh, w, n_source, n_test
    )


def window_parts(
    rec_centre: tuple[int, int, int],
    src_centre: tuple[int, int, int],
    h: float,
    above: ReferenceMedium,
    below: ReferenceMedium,
    window: Window,
    n_source: int = 10,
    n_test: int = 4,
    q0h: float = 10.0,
) -> tuple[NDArray, NDArray]:
    """The ramp integrals (1/4 pi^2) int chi Y0 and (1/4 pi^2) int chi Y1 over start Q0 < q < Q0.

    They turn the sharp-cutoff parts into the windowed ones (A0 + int chi Y0, A1 + int chi Y1) and are taken
    from D, so the block is unchanged; cheap, since q < Q0.

    Returns:
        Two arrays of shape (n_test, n_source, 9, 9), complex.
    """
    rec, src = tuple(rec_centre), tuple(src_centre)
    if src[0] > 0:
        r0, r1 = window_parts(
            _mirrored(rec),
            _mirrored(src),
            h,
            below,
            above,
            window,
            n_source,
            n_test,
            q0h,
        )
        return _mirror(r0), _mirror(r1)
    kind = _check(rec, src)
    ka, kb = _key(above), _key(below)
    qh, w = window.ramp_rule(q0h)
    return tuple(
        _static_integral(order, kind, rec, src, h, ka, kb, qh, w, n_source, n_test)
        for order in (0, 1)
    )


# ---------------------------------------------------------------------------
# The per-frequency integral in the frame of k
# ---------------------------------------------------------------------------


def _lateral_harmonics_at(
    qh: float, dxy: tuple, h: float, exps_f: tuple, exps_s: tuple
) -> NDArray:
    """S_n(q) = (1/4 pi^2) int dphi  lateral_a lateral_c e^{i k.Delta} e^{i n phi}, shape (9, n_test, n_source)."""
    reach = float(np.hypot(*dxy))
    n = _n_phi(qh, reach)
    phi = 2 * np.pi * np.arange(n) / n
    q = qh / h
    kx, ky = q * np.cos(phi), q * np.sin(phi)
    coef = (
        np.exp(1j * (kx * dxy[0] + ky * dxy[1]) * h) * (2 * np.pi / n) / (4 * np.pi**2)
    )
    lf = _lateral(kx, ky, h, exps_f, True) * coef[:, None]
    ls = _lateral(kx, ky, h, exps_s, False)
    x = (lf[:, :, None] * np.exp(1j * np.outer(phi, _HARM))[:, None, :]).reshape(n, -1)
    return (x.T @ ls).reshape(len(exps_f), 9, len(exps_s)).transpose(1, 0, 2)


def _radial_width(reach: float) -> float:
    """Panel width in q h: the lateral moments oscillate with period pi, the phase with 2 pi / reach."""
    return np.pi / 2 / max(1.0, reach / 4)


def lateral_harmonics(
    offsets: list[tuple[int, int]],
    h: float,
    q0h: float,
    n_source: int = 10,
    n_test: int = 4,
    order: int = 20,
) -> dict:
    """The angular harmonics S_n(q) of the cells' lateral moments, once for a frequency sweep.

    For each lateral offset (in units of h) of receiver from source, S_n(q), |n| <= 4, on [0, Q0] as
    piecewise Chebyshev series of ``order`` terms on panels of the radial rule's width.  S_n is entire in q
    (the lateral moments are), so the series reach rounding; it depends on neither the frequency nor the
    media nor the depths of the cells.

    Returns:
        {"q0h", "n_source", "n_test", "order", offset: (edges, coeffs)}, coeffs shape
        (n_panels, order, 9, n_test, n_source).
    """
    exps_f = SOURCE_EXPONENTS_QUARTIC[:n_test]
    exps_s = SOURCE_EXPONENTS_QUARTIC[:n_source]
    x = np.cos(np.pi * (np.arange(order) + 0.5) / order)
    vinv = np.linalg.inv(np.polynomial.chebyshev.chebvander(x, order - 1))
    out: dict = {"q0h": q0h, "n_source": n_source, "n_test": n_test, "order": order}
    for off in {tuple(o) for o in offsets}:
        width = _radial_width(float(np.hypot(*off)))
        n_pan = max(1, int(np.ceil(q0h / width)))
        edges = np.linspace(0.0, q0h, n_pan + 1)
        coeffs = []
        for e0, e1 in itertools.pairwise(edges):
            vals = np.stack(
                [
                    _lateral_harmonics_at(
                        e0 + (e1 - e0) * (t + 1) / 2, off, h, exps_f, exps_s
                    )
                    for t in x
                ]
            )
            coeffs.append(np.einsum("kj,j...->k...", vinv, vals))
        out[off] = (edges, np.stack(coeffs))
    return out


def _interpolate(table_entry: tuple, qh: NDArray) -> NDArray:
    """S_n at the radial nodes, shape (N, 9, n_test, n_source)."""
    edges, coeffs = table_entry
    i = np.clip(np.searchsorted(edges, qh, side="right") - 1, 0, len(edges) - 2)
    x = 2 * (qh - edges[i]) / (edges[i + 1] - edges[i]) - 1
    v = np.polynomial.chebyshev.chebvander(x, coeffs.shape[1] - 1)
    # one matrix product per panel (no gathered copy of the coefficients per node)
    out = np.empty((len(qh),) + coeffs.shape[2:], dtype=coeffs.dtype)
    flat = coeffs.reshape(coeffs.shape[0], coeffs.shape[1], -1)
    for p in np.unique(i):
        sel = i == p
        out[sel] = (v[sel] @ flat[p]).reshape((-1,) + coeffs.shape[2:])
    return out


def _dynamic_frames(
    q: NDArray,
    omega: float,
    above: ReferenceMedium,
    below: ReferenceMedium,
    rec: tuple,
    src: tuple,
    h: float,
    lz: list,
    ez: list,
) -> dict:
    """The dynamic frame kernel at k = (q, 0) with the cells' vertical moments, {(l, e): (N, 9, 9)}.

    Columns paired as the plane-wave integrand (engineering stress weighted by SOURCE_PAIRING); lateral
    moments excluded.
    """
    zero = np.zeros_like(q)
    st_a, _, w_a, kz_a = mode_matrices(q, zero, omega, above)
    rt = interface_rt(q, zero, omega, above, below)
    if rec[0] < 0:  # reflected: up-going in A
        states, kzw, amp = st_a[:, :, 3:], -kz_a[:, 3:], rt["R_d"]
    else:  # transmitted: down-going in B
        st_b, _, _, kz_b = mode_matrices(q, zero, omega, below)
        states, kzw, amp = st_b[:, :, :3], kz_b[:, :3], rt["T_d"]
    zr, zs = rec[0] * h, src[0] * h
    kzs = kz_a[:, :3]
    vf = {
        l: h * np.exp(1j * kzw * zr) * 2.0 * 1j**l * spherical_jn(l, kzw * h)
        for l in lz
    }
    vs = {e: h * np.exp(-1j * kzs * zs) * monomial_fourier(e, -kzs * h) for e in ez}
    right = (
        st_a[:, :, :3].swapaxes(1, 2) * w_a[:, :3, None] * SOURCE_PAIRING
    )  # (N, n, j)
    out = {}
    for l in lz:
        left = states * vf[l][:, None, :]  # (N, i, m)
        mid = left @ amp  # (N, i, n)
        for e in ez:
            out[l, e] = mid @ (right * vs[e][:, :, None])
    return out


def truncated_blocks(
    pairs: list[tuple[tuple[int, int, int], tuple[int, int, int]]],
    h: float,
    omega: float,
    above: ReferenceMedium,
    below: ReferenceMedium,
    n_source: int = 10,
    n_test: int = 4,
    q0h: float = 10.0,
    table: dict | None = None,
    window: Window | None = None,
    low_pass: Window | None = None,
) -> list[NDArray]:
    """D = (1/4 pi^2) int_{q < Q0} Y d^2k for many pairs (receiver, source) with sources in A (a source in B
    is mirrored onto A by ``interface_block``).

    In the frame of k: per radial node one 6 x 6 interface solve and the dynamic frame kernel with the
    cells' vertical moments in closed form, split into its nine angular harmonics; the azimuthal integral
    is the table S_n(q) of ``lateral_harmonics``, shared by every frequency.  The radial rule has the branch
    points of both media as panel ends, so it changes with the frequency; S_n is interpolated to it.

    With a ``window`` chi, D = (1/4 pi^2) int_{q < Q0} [Y - chi (Y0 + omega^2 Y1)] d^2k: the static and omega^2
    frame kernels, in the same harmonics, are subtracted on the ramp.

    Args:
        table: ``lateral_harmonics`` for the pairs' lateral offsets with the same q0h, n_source, n_test;
            built here when absent (pass it when sweeping frequency).
        window: The window of ``frequency_independent_parts`` (None: the sharp cutoff).
        low_pass: The low band of the split D = D_low + D_high (``high_band_blocks``): the integrand times
            psi = 1 - low_pass(q h, q0h), so it ends smoothly on low_pass's ramp at q0h (here the band's upper end,
            not the split's Q0).  Not with a window: the band lies below it.

    Returns:
        One block per pair, shape (n_test, n_source, 9, 9), complex.
    """
    for rec, src in pairs:
        _check(rec, src)
    if low_pass is not None and window is not None:
        raise ValueError(
            "truncated_blocks: low_pass is the band below the window; pass one or the other."
        )
    offsets = [(r[1] - s[1], r[2] - s[2]) for r, s in pairs]
    if table is None:
        table = lateral_harmonics(offsets, h, q0h, n_source, n_test)
    if (table["q0h"], table["n_source"], table["n_test"]) != (q0h, n_source, n_test):
        raise ValueError(
            "truncated_blocks: the table was built for another q0h, n_source or n_test."
        )
    exps_f = SOURCE_EXPONENTS_QUARTIC[:n_test]
    exps_s = SOURCE_EXPONENTS_QUARTIC[:n_source]
    lz = sorted({e[0] for e in exps_f})
    ez = sorted({e[0] for e in exps_s})
    fa = {l: np.array([a for a, x in enumerate(exps_f) if x[0] == l]) for l in lz}
    sc = {e: np.array([c for c, x in enumerate(exps_s) if x[0] == e]) for e in ez}
    branch = [omega * h / c for m in (above, below) for c in (m.alpha, m.beta)]
    if window is not None and window.start * q0h < max(1.0, 2 * max(branch)):
        raise ValueError(
            "truncated_blocks: the window's ramp must start beyond the branch-point panels "
            "(start Q0 h >= max(1, 2 omega h / beta)); raise Q0 or the window's start."
        )
    rules: dict = {}
    frames: dict = {}
    harmonics: dict = {}
    out: list = [None] * len(pairs)
    widths = [_radial_width(float(np.hypot(*off))) for off in offsets]
    # by radial rule: the frames and harmonics of one rule are reused within it and dropped after it (kept for every
    # rule at once they grew as rules x pairs of depths x nodes: 13 GB for 576 cells in four layers)
    current = None
    for i in sorted(range(len(pairs)), key=lambda n: widths[n]):
        (rec, src), off, width = pairs[i], offsets[i], widths[i]
        if width != current:
            frames.clear()
            harmonics.clear()
            current = width
        if width not in rules:
            ramp = window if window is not None else low_pass
            breaks = () if ramp is None else tuple(ramp.ramp_edges(q0h))
            qh_r, w_r = radial_rule(q0h, branch, width, breaks=breaks)
            if low_pass is not None:
                w_r = w_r * (1.0 - low_pass(qh_r, q0h))
            rules[width] = (qh_r, w_r)
        qh, w = rules[width]
        key = (width, rec[0], src[0])
        if key not in frames:
            fr = _dynamic_frames(qh / h, omega, above, below, rec, src, h, lz, ez)
            frames[key] = {k: _rotate_harmonics(v @ _M_INV) for k, v in fr.items()}
            if window is not None:
                chi = window(qh, q0h)
                on = chi > 0
                stat = [
                    _static_frames(
                        order,
                        _check(rec, src),
                        rec,
                        src,
                        h,
                        _key(above),
                        _key(below),
                        qh[on] / h,
                        exps_f,
                        exps_s,
                    )
                    for order in (0, 1)
                ]
                for k, v in frames[key].items():
                    v[on] -= chi[on, None, None] * (stat[0][k] + omega**2 * stat[1][k])
        if (
            (off, width) not in harmonics
        ):  # S_n depends on the lateral offset and the rule, not on the depths
            harmonics[off, width] = (
                _interpolate(table[off], qh) * (w / h * qh / h)[:, None, None, None]
            )
        S = harmonics[off, width]
        blk = np.zeros((n_test, n_source, 81), dtype=complex)
        for (l, e), rot in frames[key].items():
            sub = S[:, :, fa[l]][:, :, :, sc[e]].reshape(-1, len(fa[l]) * len(sc[e]))
            blk[np.ix_(fa[l], sc[e])] += (sub.T @ rot.reshape(-1, 81)).reshape(
                len(fa[l]), len(sc[e]), 81
            )
        out[i] = blk.reshape(n_test, n_source, 9, 9)
    return out


def integrand_points(
    rec_z: int,
    src_z: int,
    kx: NDArray,
    ky: NDArray,
    h: float,
    omega: float,
    above: ReferenceMedium,
    below: ReferenceMedium,
    n_source: int = 10,
    n_test: int = 4,
    q0h: float = 10.0,
    window: Window | None = None,
) -> NDArray:
    """The integrand of D at lateral wavenumbers (kx, ky), cells at lateral offset zero: the dynamic frame
    kernel less the window's static and omega^2 parts, rotated to k's direction, times the cells' lateral
    moments.  D(Delta) = (1/4 pi^2) int F(k) e^{i k . Delta h} d^2k over q < Q0.

    Args:
        rec_z, src_z: The cells' centres' z in h (odd); the source in A (mirror first).

    Returns:
        Shape (P, n_test, n_source, 81), complex.
    """
    rec, src = (rec_z, 0, 0), (src_z, 0, 0)
    kind = _check(rec, src)
    exps_f = SOURCE_EXPONENTS_QUARTIC[:n_test]
    exps_s = SOURCE_EXPONENTS_QUARTIC[:n_source]
    lz = sorted({e[0] for e in exps_f})
    ez = sorted({e[0] for e in exps_s})
    pairs_le = [(l, e) for l in lz for e in ez]
    # block[a, c]: the (l, e) of field function a and source monomial c
    block = np.array(
        [
            [pairs_le.index((exps_f[a][0], exps_s[c][0])) for c in range(n_source)]
            for a in range(n_test)
        ]
    )
    qh = np.hypot(kx, ky) * h
    q = qh / h
    # the frame kernel in the frame of k (times M^-1), less the window's static and omega^2 parts
    fr = {
        k: v @ _M_INV
        for k, v in _dynamic_frames(q, omega, above, below, rec, src, h, lz, ez).items()
    }
    if window is not None:
        chi = window(qh, q0h)
        ramp = chi > 0
        if np.any(ramp):
            stat = [
                _static_frames(
                    order,
                    kind,
                    rec,
                    src,
                    h,
                    _key(above),
                    _key(below),
                    q[ramp],
                    exps_f,
                    exps_s,
                    rotate=False,
                )
                for order in (0, 1)
            ]
            for k, v in fr.items():
                v[ramp] -= chi[ramp, None, None] * (stat[0][k] + omega**2 * stat[1][k])
    # rotated to k's direction, T G T^T, columns paired (as _rotate_harmonics at phi)
    T = rotation(np.arctan2(ky, kx))
    g = np.stack(
        [T @ fr[l, e] @ T.swapaxes(1, 2) * SOURCE_PAIRING for l, e in pairs_le], 1
    )
    lf = _lateral(kx, ky, h, exps_f, True)
    ls = _lateral(kx, ky, h, exps_s, False)
    return (
        lf[:, :, None, None]
        * ls[:, None, :, None]
        * g[:, block].reshape(len(q), n_test, n_source, 81)
    )


def high_band_blocks(
    rec_z: int,
    src_z: int,
    n_grid: int,
    h: float,
    omega: float,
    above: ReferenceMedium,
    below: ReferenceMedium,
    n_source: int = 10,
    n_test: int = 4,
    q0h: float = 10.0,
    window: Window | None = None,
    low_pass: Window | None = None,
    band_qh: float = 2.5,
    offsets: list[tuple[int, int]] | None = None,
    chunk: int = 4096,
) -> NDArray:
    """D_high, the part of D above the low band, for every lateral offset of a periodic grid of n_grid cells.

    D = D_low + D_high, with D_low = ``truncated_blocks(..., q0h=band_qh, low_pass=low_pass)`` and D_high the
    integrand times low_pass(q h, band_qh) (0 below the band, 1 above it) on q < Q0.  The band lies above every
    branch point, so D_high's spectrum is smooth and its kernel decays faster than any power of the offset; it
    is sampled at the wavenumbers of the periodic grid, 2 pi m / (n_grid 2h), and summed onto the grid's offsets
    (Poisson: the periodised kernel, sum over images of D_high(d + j n_grid)).

    Args:
        rec_z, src_z: The cells' centres' z in h (odd); the source in A (mirror first, as ``interface_block``).
        n_grid: Cells of the periodic grid along x and y; its images lie n_grid cells apart.
        window: The split's window (as ``truncated_blocks``); its ramp must start at or above band_qh.
        low_pass: The band: its ramp from low_pass.start * band_qh to band_qh.
        offsets: Only these lateral offsets (cells), by the lateral symmetries: the integrand is sampled on the
            wedge 0 <= ky <= kx alone (an eighth of the grid), F(Q k) = T_Q F(k) for the eight signed permutations
            Q fixing z, so D_high(x) = sum_Q T_Q D_W(Q^T x), D_W the wedge's sum (points on its edges weighted 1/2).
            None: every offset of the grid, from the whole grid (the reference).

    Returns:
        Shape (n_grid, n_grid, n_test, n_source, 9, 9): [dx % n_grid, dy % n_grid] holds the block of lateral
        offset (dx, dy) cells (2h each) of receiver from source.  With ``offsets``: (len(offsets), n_test,
        n_source, 9, 9), in their order.
    """
    _check((rec_z, 0, 0), (src_z, 0, 0))
    if low_pass is None:
        raise ValueError("high_band_blocks: pass the low band (low_pass).")
    branch = [omega * h / c for m in (above, below) for c in (m.alpha, m.beta)]
    lo = low_pass.start * band_qh
    if lo < max(branch) * (1 + 1e-9) or (
        window is not None and window.start * q0h < band_qh
    ):
        raise ValueError(
            "high_band_blocks: the band must lie above the branch points and below the window's ramp\n"
            f"  (band from {lo:g} to {band_qh:g}; branch points to {max(branch):g}"
            + (
                f"; window from {window.start * q0h:g})."
                if window is not None
                else ")."
            )
        )
    dk = 2 * np.pi / (n_grid * 2 * h)
    tiles = int(np.ceil(q0h / h / dk / n_grid)) + 1
    acc = np.zeros((n_grid * n_grid, n_test, n_source, 81), dtype=complex)
    j = np.arange(n_grid)
    wedge = offsets is not None
    span = range(tiles) if wedge else range(-tiles, tiles)
    for tx, ty in itertools.product(span, repeat=2):
        if wedge and ty > tx:
            continue  # wholly above the diagonal
        mx, my = np.meshgrid(tx * n_grid + j, ty * n_grid + j, indexing="ij")
        kx, ky = (mx * dk).ravel(), (my * dk).ravel()
        qh_all = np.hypot(kx, ky) * h
        keep = (qh_all > lo) & (qh_all < q0h)
        if wedge:
            keep &= mx.ravel() >= my.ravel()
            # the wedge's edges (ky = 0, ky = kx) are shared with a neighbouring wedge
            edge = 1.0 - 0.5 * (my.ravel() == 0) - 0.5 * (mx.ravel() == my.ravel())
        on = np.flatnonzero(keep)
        for i0 in range(0, len(on), chunk):
            sel = on[i0 : i0 + chunk]
            qx, qy, qh = kx[sel], ky[sel], qh_all[sel]
            wgt = low_pass(qh, band_qh) * dk**2 / (4 * np.pi**2)
            if wedge:
                wgt = wgt * edge[sel]
            vals = integrand_points(
                rec_z,
                src_z,
                qx,
                qy,
                h,
                omega,
                above,
                below,
                n_source,
                n_test,
                q0h,
                window,
            )
            idx = (mx.ravel()[sel] % n_grid) * n_grid + (my.ravel()[sel] % n_grid)
            acc[idx] += (
                wgt[:, None, None, None] * vals
            )  # residues are distinct within a tile
    # sum_m F(k_m) e^{i k_m . 2 d h} over the residues m = j (mod n_grid): an inverse DFT
    out = np.fft.ifft2(acc.reshape(n_grid, n_grid, -1), axes=(0, 1)) * n_grid**2
    out = out.reshape(n_grid, n_grid, n_test, n_source, 9, 9)
    if not wedge:
        return out
    from .graded_voxel.fft import _transform, signed_permutations

    lateral = [q for q in signed_permutations() if q[0, 0] == 1.0]
    res = np.zeros((len(offsets), n_test, n_source, 9, 9), dtype=complex)
    for i, x in enumerate(offsets):
        for q in lateral:
            y = q[1:, 1:].T @ np.asarray(x, float)
            res[i] += _transform(q, out[int(y[0]) % n_grid, int(y[1]) % n_grid])
    return res


def _truncated_blocks_direct(
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
    """D by the mode sum at every point of a polar grid: the reference for ``truncated_blocks``.

    The reflection and transmission coefficients, the mode states and the cells' 3-D moments are evaluated
    at each (q, phi); everything is recomputed at each frequency.

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
    table: dict | None = None,
    window: Window | None = None,
) -> NDArray:
    """The reflected (cells on one side) or transmitted (opposite sides) block at one frequency: D + A0 + w^2 A1.

    ``parts`` = ``frequency_independent_parts(...)`` and ``table`` = ``lateral_harmonics(...)`` for the pair's
    lateral offset, with the same arguments (``window`` included): both independent of the frequency, passed
    when sweeping it.
    """
    rec, src = tuple(rec_centre), tuple(src_centre)
    parts = (
        parts
        if parts is not None
        else frequency_independent_parts(
            rec, src, h, above, below, n_source, n_test, q0h, q1h, window
        )
    )
    a0, a1 = parts[0], parts[1]
    a2 = parts[2] if len(parts) > 2 else None  # with omega4: the omega^4 correction
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
                table,
                window,
            )[0]
        )
    else:
        d = truncated_blocks(
            [(rec, src)], h, omega, above, below, n_source, n_test, q0h, table, window
        )[0]
    out = d + a0 + omega**2 * a1
    if a2 is not None:
        out = out + omega**4 * a2
    return out


def sweep(
    rec_centre: tuple[int, int, int],
    src_centre: tuple[int, int, int],
    h: float,
    omegas: list[float],
    above: ReferenceMedium,
    below: ReferenceMedium,
    n_source: int = 10,
    n_test: int = 4,
    q0h: float = 10.0,
    q1h: float = 160.0,
    window: Window | None = None,
) -> list[NDArray]:
    """The block at each frequency, with A0, A1 and the table S_n computed once.

    Q0 must suit the highest frequency: Q0 h = 10 gives about 4e-10 of the block at omega h / beta = 1/6 and
    Q0 h = 40 about 6e-11 at omega h / beta = 1 (the remainder falls as (omega / beta Q0)^4 Q0^-2).
    """
    rec, src = tuple(rec_centre), tuple(src_centre)
    parts = frequency_independent_parts(
        rec, src, h, above, below, n_source, n_test, q0h, q1h, window
    )
    table = lateral_harmonics(
        [(rec[1] - src[1], rec[2] - src[2])], h, q0h, n_source, n_test
    )
    return [
        interface_block(
            rec,
            src,
            h,
            om,
            above,
            below,
            n_source,
            n_test,
            q0h,
            q1h,
            parts,
            table,
            window,
        )
        for om in omegas
    ]
