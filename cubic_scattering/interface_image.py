"""Galerkin blocks of the static interface image: Legendre cells coupled through a welded interface's image.

Plan: ``docs/2026-10-10-stratified-reference-legendre-cells-3d.md``, option 1, route A.

THE KERNEL.  Medium A fills z < 0, medium B z > 0, welded at z = 0 (z down).  For receiver and source both in A,
the reflected static 9 x 9 kernel (rows u, engineering strain; columns force, engineering stress, as
``graded_voxel.kernel.kernel_9x9``) is a sum of canonical terms (``scripts/derive_interface_image_terms.py``)

    c_t(lamA, muA, lamB, muB)  z^A zp^B  d^alpha Phi_j(rho_x, rho_y, zeta),   zeta = -(z + zp) >= 0,  rho = x - x',

with Phi0 = 1/(2 pi R), Phi1 = -log(R + zeta)/(2 pi), Phi2 = (zeta log(R + zeta) - R)/(2 pi), R^2 = rho^2 + zeta^2.
Every d^alpha Phi_j that occurs is homogeneous in (rho, zeta), of degree -1 - |alpha|, -|alpha|, 1 - |alpha|.

THE S-FORM.  With x = c + h v and x' = c' + h v' per axis (v, v' in [-1, 1]), the six-fold integral of a field
function, a source monomial and a term is a three-fold integral over the separation, against weights that
are piecewise polynomials, computed exactly:
    lateral   rho = (c - c') + h sigma,  sigma = v - v',  w(sigma) = int v^E (v - sigma)^F dv;
    vertical  zeta = Z0 - h tau,  Z0 = -(c_z + c'_z),  tau = v + v',  u(tau) = int v^E (tau - v)^F dv,
each on the two pieces [-2, 0], [0, 2].  The factors z^A, zp^B are expanded into the vertical weights.  All
lengths are in units of h; the block scales as h^(6 + degree + A + B).

THE PIECES.  The singular point rho = 0, zeta = 0 is reached only when both cells touch the interface and
touch laterally, and then it is a corner of the pieces that contain it.  On such a piece Euler's identity for
a homogeneous kernel f of degree d, with the box anchored at the singular corner,

    (3 + |n| + d) int_box s^n f = sum over the three FAR faces  |A_i| A_i^(n_i) int_face s^(n without i) f,

turns the volume integral into face integrals of a smooth integrand (the faces lie at distance >= 2 from the
singular point, and the region zeta >= 0 meets the logarithm's singular ray R + zeta = 0 only at the corner).
The weight is expanded about the corner in exact rationals, so the low powers of zeta that vanish (the
vertical weight vanishes as zeta^(1 + A + B) at the touching corner) are exact zeros and no divergent monomial
is ever formed.  Every other piece is smooth and is integrated by Gauss rules in coordinates centred on it.
The face integrals are where the closed forms of the Mathematica phase enter; here they are Gauss rules,
exponentially convergent.
"""

import json
from fractions import Fraction
from functools import cache
from pathlib import Path

import numpy as np
import sympy as sp
from numpy.polynomial.legendre import leg2poly, leggauss
from numpy.typing import NDArray

from .effective_contrasts import ReferenceMedium
from .graded_voxel.basis import SOURCE_EXPONENTS_QUARTIC
from .image_moments import corner_moment

_DATA = Path(__file__).resolve().parent.parent / "scripts" / "data"
#: Canonical terms per kind: the image (both cells in A) and the transmission (source in A, receiver in B).
_TERMS_PATH = {
    "reflected": _DATA / "interface_image_terms.json",
    "transmitted": _DATA / "interface_transmission_terms.json",
}
#: The 9-state under the mirror z -> -z: u_z, gamma_zy and gamma_zx change sign (Voigt order zz, xx, yy, xy, zy, zx).
MIRROR_9 = np.array([-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0])
_DEG0 = {0: -1, 1: 0, 2: 1}

Poly = tuple[Fraction, ...]  # ascending coefficients


# ---------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------


@cache
def _terms(kind: str = "reflected") -> tuple:
    """The canonical terms of one kind, each material coefficient compiled: ((row, col, j, alpha, A, B, fn), ...)."""
    data = json.loads(_TERMS_PATH[kind].read_text())
    lam_a, mu_a, lam_b, mu_b = sp.symbols("lamA muA lamB muB", positive=True)
    loc = {"lamA": lam_a, "muA": mu_a, "lamB": lam_b, "muB": mu_b}
    out = []
    for r, row in enumerate(data["terms"]):
        for c, terms in enumerate(row):
            for t in terms:
                expr = sp.sympify(t["coef"], locals=loc)
                fn = sp.lambdify((lam_a, mu_a, lam_b, mu_b), expr, "numpy")
                out.append(
                    (r, c, t["j"], tuple(t["alpha"]), t["z_power"], t["zp_power"], fn)
                )
    return tuple(out)


@cache
def kernel_function(j: int, alpha: tuple[int, int, int]):
    """d^alpha Phi_j as a vectorised function of (rho_x, rho_y, zeta)."""
    rx, ry, zeta = sp.symbols("rx ry zeta", real=True)
    r = sp.sqrt(rx**2 + ry**2 + zeta**2)
    phi = {
        0: 1 / (2 * sp.pi * r),
        1: -sp.log(r + zeta) / (2 * sp.pi),
        2: (zeta * sp.log(r + zeta) - r) / (2 * sp.pi),
    }[j]
    expr = (
        sp.diff(phi, rx, alpha[0], ry, alpha[1], zeta, alpha[2]) if sum(alpha) else phi
    )
    # No simplify: on a fourth derivative it takes about a minute; common subexpressions suffice.
    return sp.lambdify((rx, ry, zeta), expr, "numpy", cse=True)


def kernel_degree(j: int, alpha: tuple[int, int, int]) -> int:
    """Degree of homogeneity of d^alpha Phi_j in (rho, zeta)."""
    return _DEG0[j] - sum(alpha)


# ---------------------------------------------------------------------------
# Exact one-dimensional weights
# ---------------------------------------------------------------------------


def _pmul(a: Poly, b: Poly) -> Poly:
    out = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for k, y in enumerate(b):
            out[i + k] += x * y
    return tuple(out)


def _padd(a: Poly, b: Poly, s: Fraction = Fraction(1)) -> Poly:
    n = max(len(a), len(b))
    return tuple(
        (a[i] if i < len(a) else 0) + s * (b[i] if i < len(b) else 0) for i in range(n)
    )


def _ppow(a: Poly, n: int) -> Poly:
    out: Poly = (Fraction(1),)
    for _ in range(n):
        out = _pmul(out, a)
    return out


@cache
def _weight_1d(e: int, f: int, kind: str) -> tuple[Poly, Poly]:
    """w(sigma) = int v^e (v - sigma)^f dv (kind 'lat'), or u(tau) = int v^e (tau - v)^f dv (kind 'vert'),
    over v in [-1, 1] intersected with the second cell, on the pieces [-2, 0] and [0, 2]; exact rationals."""
    v, s = sp.symbols("v s")
    g = v**e * ((v - s) ** f if kind == "lat" else (s - v) ** f)
    if kind == "lat":
        lims = ((-1, s + 1), (s - 1, 1))
    else:
        lims = ((-1, s + 1), (s - 1, 1))
    pieces = []
    for lo, hi in lims:
        expr = sp.expand(sp.integrate(g, (v, lo, hi)))
        coeffs = sp.Poly(expr, s).all_coeffs()[::-1] if expr != 0 else [0]
        pieces.append(
            tuple(
                Fraction(int(sp.Rational(c).p), int(sp.Rational(c).q)) for c in coeffs
            )
        )
    return pieces[0], pieces[1]


@cache
def _legendre_monomials(e: int) -> Poly:
    """P_e(v) in ascending monomials."""
    return tuple(Fraction(c).limit_denominator(10**6) for c in leg2poly([0] * e + [1]))


def _lateral_weight(e_field: int, f_source: int) -> tuple[Poly, Poly]:
    """Weight of a Legendre field factor P_e(v) against a source monomial v'^f along a lateral axis."""
    out: list[Poly] = [(Fraction(0),), (Fraction(0),)]
    for big_e, coef in enumerate(_legendre_monomials(e_field)):
        if coef == 0:
            continue
        w = _weight_1d(big_e, f_source, "lat")
        out = [_padd(out[k], w[k], coef) for k in range(2)]
    return out[0], out[1]


def _vertical_weight(
    e_field: int,
    f_source: int,
    a_pow: int,
    b_pow: int,
    cz: int,
    czp: int,
    kind: str = "vert",
) -> tuple[Poly, Poly]:
    """Vertical weight of P_e(v) (cz + v)^A against v'^f (czp + v')^B, centres in units of h: a convolution in
    tau = v + v' (kind 'vert', the image) or a correlation in sigma = v - v' (kind 'lat', the transmission)."""
    left = _pmul(
        _legendre_monomials(e_field), _ppow((Fraction(cz), Fraction(1)), a_pow)
    )
    right = _pmul(
        tuple(Fraction(int(k == f_source)) for k in range(f_source + 1)),
        _ppow((Fraction(czp), Fraction(1)), b_pow),
    )
    out: list[Poly] = [(Fraction(0),), (Fraction(0),)]
    for big_e, cl in enumerate(left):
        for big_f, cr in enumerate(right):
            if cl == 0 or cr == 0:
                continue
            w = _weight_1d(big_e, big_f, kind)
            out = [_padd(out[k], w[k], cl * cr) for k in range(2)]
    return out[0], out[1]


def _shift(p: Poly, a: Fraction, b: Fraction) -> Poly:
    """p(a + b t) as a polynomial in t, exactly."""
    out: Poly = (Fraction(0),)
    for k, c in enumerate(p):
        if c:
            out = _padd(out, _ppow((a, b), k), c)
    return out


# ---------------------------------------------------------------------------
# Moments of one kernel over one piece
# ---------------------------------------------------------------------------


def _gauss(n: int, lo: float, hi: float) -> tuple[NDArray, NDArray]:
    x, w = leggauss(n)
    return lo + (hi - lo) * (x + 1) / 2, w * (hi - lo) / 2


def _regular_moments(
    f, box: tuple, centre: tuple, deg: tuple[int, int, int], n: int
) -> NDArray:
    """int_box prod_i (s_i - centre_i)^p_i f(s) for p_i <= deg_i, by a tensor Gauss rule."""
    rules = [_gauss(n, lo, hi) for lo, hi in box]
    sx, sy, sz = np.meshgrid(rules[0][0], rules[1][0], rules[2][0], indexing="ij")
    fw = f(sx, sy, sz) * np.einsum("i,j,k->ijk", rules[0][1], rules[1][1], rules[2][1])
    px = np.vander(rules[0][0] - centre[0], deg[0] + 1, increasing=True)
    py = np.vander(rules[1][0] - centre[1], deg[1] + 1, increasing=True)
    pz = np.vander(rules[2][0] - centre[2], deg[2] + 1, increasing=True)
    return np.einsum("ijk,ia,jb,kc->abc", fw, px, py, pz)


def _corner_moments(
    f, d: int, box: tuple, deg: tuple[int, int, int], n: int
) -> NDArray:
    """int_box s^p f(s) for a box with a corner at the origin (the singular point), by Euler's identity."""
    far = [lo if hi == 0 else hi for lo, hi in box]  # the coordinate of each far face
    m = np.zeros((deg[0] + 1, deg[1] + 1, deg[2] + 1))
    for i in range(3):
        others = [k for k in range(3) if k != i]
        r1, r2 = (_gauss(n, *box[k]) for k in others)
        g1, g2 = np.meshgrid(r1[0], r2[0], indexing="ij")
        pts = [None, None, None]
        pts[i] = np.full_like(g1, far[i])
        pts[others[0]], pts[others[1]] = g1, g2
        fw = f(*pts) * np.outer(r1[1], r2[1])
        v1 = np.vander(r1[0], deg[others[0]] + 1, increasing=True)
        v2 = np.vander(r2[0], deg[others[1]] + 1, increasing=True)
        face = np.einsum("ab,ap,bq->pq", fw, v1, v2)  # (deg_o0 + 1, deg_o1 + 1)
        ai = np.array([abs(far[i]) * far[i] ** p for p in range(deg[i] + 1)])
        contrib = np.einsum("k,pq->kpq", ai, face)
        m += (
            np.moveaxis(contrib, 0, i)
            if i == 0
            else np.moveaxis(contrib, [0, 1, 2], [i, *others])
        )
    total = np.add.outer(
        np.add.outer(np.arange(deg[0] + 1), np.arange(deg[1] + 1)),
        np.arange(deg[2] + 1),
    )
    denom = 3 + total + d
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0, m / np.where(denom == 0, 1, denom), np.nan)


def _corner_moments_closed(
    j: int, alpha: tuple[int, int, int], d: int, box: tuple, deg: tuple[int, int, int]
) -> NDArray:
    """``_corner_moments`` in closed form (``image_moments.corner_moment``): the same array, NaN where divergent."""
    sides = tuple(int(max(abs(lo), abs(hi))) for lo, hi in box)
    signs = tuple(-1 if lo < 0 else 1 for lo, _ in box[:2])
    m = np.full((deg[0] + 1, deg[1] + 1, deg[2] + 1), np.nan)
    for p in range(deg[0] + 1):
        for q in range(deg[1] + 1):
            for r in range(deg[2] + 1):
                if 3 + p + q + r + d > 0:
                    m[p, q, r] = float(corner_moment(j, alpha, (p, q, r), sides, signs))
    return m


# ---------------------------------------------------------------------------
# The block
# ---------------------------------------------------------------------------


def image_block(
    rec_centre: tuple[int, int, int],
    src_centre: tuple[int, int, int],
    h: float,
    above: ReferenceMedium,
    below: ReferenceMedium,
    n_source: int = 10,
    n_test: int = 4,
    n_gauss: int = 32,
    closed: bool = True,
) -> NDArray:
    """The static interface block K[a, c] between two cells: the image when both lie on one side of the
    interface z = 0, the transmission when they lie on opposite sides.

    Cells in A (z < 0) use the image's terms; a receiver in B with a source in A uses the transmission's.
    The other two arrangements are mirrored (z -> -z, media exchanged) onto these: the 9-state changes sign
    in u_z, gamma_zy, gamma_zx (``MIRROR_9``), and each cell polynomial by (-1)^(its power of xi_z).

    Args:
        rec_centre: Receiver cell centre in units of h, (z, x, y); z odd (z = -1 or 1 touches the interface).
        src_centre: Source cell centre in units of h, likewise.
        h: Cell half-width, km.
        above: Medium A (z < 0).
        below: Medium B (z > 0).
        n_source: Source monomials (10, 20 or 35).
        n_test: Field functions (4 or 10).
        n_gauss: Gauss nodes per dimension on regular pieces (and on the far faces when not closed).
        closed: Corner pieces in closed form (``image_moments``); False uses Euler's reduction to the far
            faces with Gauss rules there (the two agree to 1e-15).

    Returns:
        K, shape (n_test, n_source, 9, 9), as ``graded_voxel.blocks.coupling_block``.

    Raises:
        ValueError: when a cell is not in medium A, or a divergent monomial would be needed.
    """
    cz, czp = rec_centre[0], src_centre[0]
    if cz % 2 == 0 or czp % 2 == 0:
        raise ValueError(
            "image_block: cell centres must have odd z in units of h (cells lie on one side)."
        )
    exps_f = SOURCE_EXPONENTS_QUARTIC[:n_test]
    exps_s = SOURCE_EXPONENTS_QUARTIC[:n_source]
    if czp > 0:
        # mirror z -> -z, exchanging the media, onto a source in A
        mirrored = image_block(
            (-cz, rec_centre[1], rec_centre[2]),
            (-czp, src_centre[1], src_centre[2]),
            h,
            below,
            above,
            n_source,
            n_test,
            n_gauss,
            closed,
        )
        sa = np.array([(-1.0) ** e[0] for e in exps_f])
        sc = np.array([(-1.0) ** e[0] for e in exps_s])
        return np.einsum("a,c,i,j,acij->acij", sa, sc, MIRROR_9, MIRROR_9, mirrored)
    kind = "reflected" if cz < 0 else "transmitted"
    dx, dy = rec_centre[1] - src_centre[1], rec_centre[2] - src_centre[2]
    z0 = -(cz + czp) if kind == "reflected" else cz - czp
    mats = (above.lam, above.mu, below.lam, below.mu)
    out = np.zeros((n_test, n_source, 9, 9))

    # pieces along each axis, in the separation coordinates: lateral rho = D + sigma, vertical zeta = z0 - tau
    lat_x = [(dx - 2, dx), (dx, dx + 2)]
    lat_y = [(dy - 2, dy), (dy, dy + 2)]
    if kind == "reflected":
        # zeta = z0 - tau: tau in [-2, 0] -> zeta in [z0, z0 + 2]; tau in [0, 2] -> [z0 - 2, z0]
        ver = [(z0, z0 + 2), (z0 - 2, z0)]
    else:
        # zeta = z0 + sigma: sigma in [-2, 0] -> zeta in [z0 - 2, z0]; sigma in [0, 2] -> [z0, z0 + 2]
        ver = [(z0 - 2, z0), (z0, z0 + 2)]

    groups: dict = {}
    for r, c, j, alpha, a_pow, b_pow, fn in _terms(kind):
        groups.setdefault((j, alpha, a_pow, b_pow), []).append((r, c, fn(*mats)))

    for (j, alpha, a_pow, b_pow), entries in groups.items():
        f = kernel_function(j, alpha)
        d = kernel_degree(j, alpha)
        # weights per (field exponent, source exponent), as polynomials in the separation coordinate
        wx = {}
        for ef in {e[1] for e in exps_f}:
            for fs in {e[1] for e in exps_s}:
                p0, p1 = _lateral_weight(ef, fs)
                wx[ef, fs] = [
                    _shift(p0, Fraction(-dx), Fraction(1)),
                    _shift(p1, Fraction(-dx), Fraction(1)),
                ]
        wy = {}
        for ef in {e[2] for e in exps_f}:
            for fs in {e[2] for e in exps_s}:
                p0, p1 = _lateral_weight(ef, fs)
                wy[ef, fs] = [
                    _shift(p0, Fraction(-dy), Fraction(1)),
                    _shift(p1, Fraction(-dy), Fraction(1)),
                ]
        wz = {}
        for ef in {e[0] for e in exps_f}:
            for fs in {e[0] for e in exps_s}:
                if kind == "reflected":
                    p0, p1 = _vertical_weight(ef, fs, a_pow, b_pow, cz, czp, "vert")
                    # tau = z0 - zeta
                    wz[ef, fs] = [
                        _shift(p0, Fraction(z0), Fraction(-1)),
                        _shift(p1, Fraction(z0), Fraction(-1)),
                    ]
                else:
                    p0, p1 = _vertical_weight(ef, fs, a_pow, b_pow, cz, czp, "lat")
                    # sigma = zeta - z0
                    wz[ef, fs] = [
                        _shift(p0, Fraction(-z0), Fraction(1)),
                        _shift(p1, Fraction(-z0), Fraction(1)),
                    ]
        deg = (
            max(len(p) for v in wx.values() for p in v) - 1,
            max(len(p) for v in wy.values() for p in v) - 1,
            max(len(p) for v in wz.values() for p in v) - 1,
        )
        val = np.zeros((n_test, n_source))
        for ix in range(2):
            for iy in range(2):
                for iz in range(2):
                    box = (lat_x[ix], lat_y[iy], ver[iz])
                    singular = all(lo <= 0 <= hi for lo, hi in box)
                    if singular:
                        if any(lo < 0 < hi for lo, hi in box):
                            raise ValueError(
                                "image_block: the singular point lies inside a piece, not at a corner."
                            )
                        centre = (0.0, 0.0, 0.0)
                        mom = (
                            _corner_moments_closed(j, alpha, d, box, deg)
                            if closed
                            else _corner_moments(f, d, box, deg, n_gauss)
                        )
                    else:
                        centre = tuple((lo + hi) / 2 for lo, hi in box)
                        mom = _regular_moments(f, box, centre, deg, n_gauss)
                    for a, ef in enumerate(exps_f):
                        for cc, es in enumerate(exps_s):
                            px = _shift(
                                wx[ef[1], es[1]][ix], Fraction(centre[0]), Fraction(1)
                            )
                            py = _shift(
                                wy[ef[2], es[2]][iy], Fraction(centre[1]), Fraction(1)
                            )
                            pz = _shift(
                                wz[ef[0], es[0]][iz], Fraction(centre[2]), Fraction(1)
                            )
                            nzx = [k for k, x in enumerate(px) if x]
                            nzy = [k for k, x in enumerate(py) if x]
                            nzz = [k for k, x in enumerate(pz) if x]
                            if not (nzx and nzy and nzz):
                                continue
                            sub = mom[np.ix_(nzx, nzy, nzz)]
                            if np.isnan(sub).any():
                                raise ValueError(
                                    f"image_block: a divergent monomial for Phi{j} d^{alpha} (degree {d}); "
                                    "the weight should vanish faster at the touching corner."
                                )
                            cx = np.array([float(px[k]) for k in nzx])
                            cy = np.array([float(py[k]) for k in nzy])
                            cz_ = np.array([float(pz[k]) for k in nzz])
                            val[a, cc] += np.einsum("i,j,k,ijk->", cx, cy, cz_, sub)
        scale = h ** (6 + d + a_pow + b_pow)
        for r, c, coef in entries:
            out[:, :, r, c] += scale * coef * val
    return out
