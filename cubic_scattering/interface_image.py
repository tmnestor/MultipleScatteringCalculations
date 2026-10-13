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
from numpy.polynomial.legendre import leg2poly, legval, legvander
from numpy.typing import NDArray

from .effective_contrasts import ReferenceMedium
from .gauss_rules import gauss_interval
from .graded_voxel.basis import SOURCE_EXPONENTS_QUARTIC
from .image_moments import corner_moment

_DATA = Path(__file__).resolve().parent.parent / "scripts" / "data"
#: Canonical terms per kind: the image (both cells in A) and the transmission (source in A, receiver in B).
_TERMS_PATH = {
    ("reflected", 0): _DATA / "interface_image_terms.json",
    ("transmitted", 0): _DATA / "interface_transmission_terms.json",
    ("reflected", 1): _DATA / "interface_omega2_terms_reflected.json",
    ("transmitted", 1): _DATA / "interface_omega2_terms_transmitted.json",
    ("reflected", 2): _DATA / "interface_omega4_terms_reflected.json",
    ("transmitted", 2): _DATA / "interface_omega4_terms_transmitted.json",
}
#: The potentials Phi_j, j >= 1, as Hadamard finite parts (scripts/derive_finite_part_potentials.py).
_FP_PATH = _DATA / "finite_part_potentials.json"
#: The 9-state under the mirror z -> -z: u_z, gamma_zy and gamma_zx change sign (Voigt order zz, xx, yy, xy, zy, zx).
MIRROR_9 = np.array([-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0])
_DEG0 = {0: -1, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}

Poly = tuple[Fraction, ...]  # ascending coefficients


# ---------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------


@cache
def _terms(kind: str = "reflected", order: int = 0) -> tuple:
    """The canonical terms of one kind and order (0: the static field; 1: the omega^2 term, whose coefficients
    also hold the densities), each material coefficient compiled: ((row, col, j, alpha, A, B, fn), ...); fn
    takes the materials in the order of the file's "variables"."""
    data = json.loads(_TERMS_PATH[kind, order].read_text())
    names = data.get("variables", ["lamA", "muA", "lamB", "muB"])
    syms = sp.symbols(" ".join(names), positive=True) if names else ()  # omega^4: media already in
    loc = dict(zip(names, syms, strict=True))
    syms = syms if isinstance(syms, tuple) else (syms,)
    if not names:
        syms = ()
    out = []
    for r, row in enumerate(data["terms"]):
        for c, terms in enumerate(row):
            for t in terms:
                expr = sp.sympify(t["coef"], locals=loc)
                fn = sp.lambdify(syms, expr, "numpy")
                out.append(
                    (r, c, t["j"], tuple(t["alpha"]), t["z_power"], t["zp_power"], fn)
                )
    return tuple(out)


_RX, _RY, _ZETA = sp.symbols("rx ry zeta", real=True)
_R3 = sp.sqrt(_RX**2 + _RY**2 + _ZETA**2)


@cache
def _potential(j: int) -> sp.Expr:
    """Phi_j in (rx, ry, zeta): Phi0 = 1/(2 pi R); for j >= 1 Hadamard's finite part of the transform of
    q^(-1-j) e^(-q zeta), the continuation in the power of q.  Phi1, Phi2 differ from Mindlin's
    -log(R + zeta)/2pi, (zeta log(R + zeta) - R)/2pi by a constant and a multiple of zeta, which every static
    term annihilates by its lateral derivatives; the omega^2 term needs the finite parts themselves."""
    if j == 0:
        return 1 / (2 * sp.pi * _R3)
    r_, z_ = sp.symbols("R zeta", positive=True)
    text = json.loads(_FP_PATH.read_text())["phi"][j]
    return sp.sympify(text, locals={"R": r_, "zeta": z_}).subs({r_: _R3, z_: _ZETA})


@cache
def kernel_function(j: int, alpha: tuple[int, int, int]):
    """d^alpha Phi_j as a vectorised function of (rho_x, rho_y, zeta)."""
    phi = _potential(j)
    expr = (
        sp.diff(phi, _RX, alpha[0], _RY, alpha[1], _ZETA, alpha[2])
        if sum(alpha)
        else phi
    )
    # No simplify: on a fourth derivative it takes about a minute; common subexpressions suffice.
    return sp.lambdify((_RX, _RY, _ZETA), expr, "numpy", cse=True)


@cache
def _log_coefficient_expr(j: int, alpha: tuple[int, int, int]) -> sp.Expr:
    """g = d^alpha g0, g0 = -(-1)^n R^n P_n(zeta/R) / (2 pi n!), n = j - 1: Phi_j(lam x) = lam^n Phi_j(x)
    + lam^n log(lam) g0(x), so d^alpha Phi_j is homogeneous of degree d up to log(lam) d^alpha g0 (a homogeneous
    polynomial of degree d, zero whenever the lateral derivatives annihilate the solid harmonic)."""
    if j == 0:
        return sp.Integer(0)
    n = j - 1
    g0 = sp.cancel(
        sp.expand(
            -((-1) ** n)
            * _R3**n
            * sp.legendre(n, _ZETA / _R3)
            / (2 * sp.pi * sp.factorial(n))
        )
    )
    return sp.expand(
        sp.diff(g0, _RX, alpha[0], _RY, alpha[1], _ZETA, alpha[2]) if sum(alpha) else g0
    )


@cache
def kernel_log_coefficient(j: int, alpha: tuple[int, int, int]):
    """g of d^alpha Phi_j (f(lam x) = lam^d [f(x) + log(lam) g(x)]) as a vectorised function, or None if zero."""
    g = _log_coefficient_expr(j, alpha)
    if g == 0:
        return None
    fn = sp.lambdify((_RX, _RY, _ZETA), g, "numpy")
    return lambda x, y, z: fn(x, y, z) + 0.0 * x  # broadcast a constant polynomial


def kernel_degree(j: int, alpha: tuple[int, int, int]) -> int:
    """Degree of homogeneity of d^alpha Phi_j in (rho, zeta) (up to the log term of ``kernel_log_coefficient``)."""
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


@cache
def _mono_to_legendre(n: int) -> tuple[tuple[Fraction, ...], ...]:
    """x^j = sum_l M[j][l] P_l(x) on [-1, 1], exactly, for j < n (from x P_l = ((l+1) P_{l+1} + l P_{l-1})/(2l+1))."""
    rows = [(Fraction(1),)]
    for _ in range(1, n):
        prev = rows[-1]
        nxt = [Fraction(0)] * (len(prev) + 1)
        for l, c in enumerate(prev):
            if c:
                nxt[l + 1] += c * Fraction(l + 1, 2 * l + 1)
                if l:
                    nxt[l - 1] += c * Fraction(l, 2 * l + 1)
        rows.append(tuple(nxt))
    return tuple(rows)


def _legendre_coeffs(
    p: Poly, a: Fraction = Fraction(0), b: Fraction = Fraction(1)
) -> NDArray:
    """Legendre coefficients in x of p(a + b x), computed exactly and then rounded."""
    q = _shift(p, a, b)
    table = _mono_to_legendre(len(q))
    out = [Fraction(0)] * len(q)
    for j, c in enumerate(q):
        if c:
            for l, m in enumerate(table[j]):
                out[l] += c * m
    return np.array([float(c) for c in out])


def _corner_factor(p: Poly, far: Fraction) -> tuple[int, NDArray]:
    """p(s) on [0, far] (or [far, 0]) written as (far t)^m r(far t), t in [0, 1], r(0) != 0: the order m of the
    zero at the corner, exact, and the Legendre coefficients of r(far t) in x = 2 t - 1."""
    m = next((k for k, c in enumerate(p) if c), None)
    if m is None:
        return 0, np.zeros(1)
    r = p[m:]
    # r(far t) with t = (1 + x)/2:  argument far/2 + (far/2) x
    return m, _legendre_coeffs(r, far / 2, far / 2)


# ---------------------------------------------------------------------------
# Moments of one kernel over one piece
# ---------------------------------------------------------------------------


def _contract3(fw: NDArray, a: NDArray, b: NDArray, c: NDArray) -> NDArray:
    """sum_ijk fw[i,j,k] a[i,p] b[j,q] c[k,r], one axis at a time: a single einsum loop accumulates all n^3
    terms in one running sum and loses about 2e-15 at n = 32; three sums of n terms lose about 2e-16."""
    t = np.tensordot(fw, c, axes=(2, 0))  # (i, j, r)
    t = np.tensordot(t, b, axes=(1, 0))  # (i, r, q)
    t = np.tensordot(t, a, axes=(0, 0))  # (r, q, p)
    return np.transpose(t, (2, 1, 0))


def _gauss(n: int, lo: float, hi: float) -> tuple[NDArray, NDArray]:
    # not numpy's leggauss, whose weights are wrong by up to 7e-13 at n = 40 (see gauss_rules)
    return gauss_interval(n, lo, hi)


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
    return _contract3(fw, px, py, pz)


def _corner_moments(
    f, d: int, box: tuple, deg: tuple[int, int, int], n: int, g=None
) -> NDArray:
    """int_box s^p f(s) for a box with a corner at the origin (the singular point), by Euler's identity.

    With a log term (f(lam s) = lam^d [f(s) + log(lam) g(s)], g a polynomial), s.grad f = d f + g, and the
    identity gains -int_box s^p g, integrated exactly by Gauss.
    """
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
        face = (fw @ v2).T @ v1  # (deg_o1 + 1, deg_o0 + 1), one axis at a time
        face = face.T
        ai = np.array([abs(far[i]) * far[i] ** p for p in range(deg[i] + 1)])
        contrib = np.einsum("k,pq->kpq", ai, face)
        m += (
            np.moveaxis(contrib, 0, i)
            if i == 0
            else np.moveaxis(contrib, [0, 1, 2], [i, *others])
        )
    if g is not None:
        ng = sum(deg) + max(d, 0) + 2
        (x0, w0), (x1, w1), (x2, w2) = (_gauss(ng, *box[i]) for i in range(3))
        gx = g(*np.meshgrid(x0, x1, x2, indexing="ij")) * np.einsum(
            "i,j,k->ijk", w0, w1, w2
        )
        m -= np.einsum(
            "ijk,ip,jq,kr->pqr",
            gx,
            np.vander(x0, deg[0] + 1, increasing=True),
            np.vander(x1, deg[1] + 1, increasing=True),
            np.vander(x2, deg[2] + 1, increasing=True),
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


def _regular_legendre_moments(
    f, box: tuple, deg: tuple[int, int, int], n: int
) -> NDArray:
    """int_box P_a(x) P_b(y) P_c(z) f for a box of side 2 with local coordinates x = s - centre in [-1, 1]."""
    rules = [_gauss(n, lo, hi) for lo, hi in box]
    sx, sy, sz = np.meshgrid(rules[0][0], rules[1][0], rules[2][0], indexing="ij")
    fw = f(sx, sy, sz) * np.einsum("i,j,k->ijk", rules[0][1], rules[1][1], rules[2][1])
    lx, ly, lz = (
        legvander(rules[i][0] - (box[i][0] + box[i][1]) / 2, deg[i]) for i in range(3)
    )
    return _contract3(fw, lx, ly, lz)


@cache
def _log_rho_moment(e: int, k: int) -> float:
    """int_0^1 rho^e log(rho) P_k(2 rho - 1) d rho, exactly (int rho^m log rho = -1/(m+1)^2), in floating point."""
    coeffs = sp.Poly(
        sp.legendre(k, 2 * sp.Symbol("t") - 1), sp.Symbol("t")
    ).all_coeffs()[::-1]
    return float(sum(-sp.Rational(c) / (e + m + 1) ** 2 for m, c in enumerate(coeffs)))


def _corner_pyramids(
    f, d: int, box: tuple, weights: list[dict], pairs: list[tuple], n: int, g=None
) -> NDArray:
    """int_box W f for each pair of weights, the box anchored at the singular corner, by three Duffy pyramids.

    The box is {s_i = A_i t_i, t in [0, 1]^3}, A_i = +-2 the far coordinates.  In pyramid k, t_k = rho and
    t_j = rho v_j, so f(s) = rho^d f(A e_k(v)) and each 1-D weight is (A_i rho w_i)^m_i r_i(A_i rho w_i), its zero
    at the corner of order m_i factored out exactly and r_i evaluated in its Legendre basis.  The rho integrand is
    then a polynomial times rho^(2 + d + sum m_i), integrated exactly by Gauss; the face integrand is smooth (the
    far face lies at distance 2 from the singular point).

    Args:
        f: The kernel d^alpha Phi_j.
        d: Its degree of homogeneity.
        box: ((lo, hi), ...) with 0 an end of each interval.
        weights: Per axis, {key: Poly in s about the corner}.
        pairs: (key_x, key_y, key_z) per output.
        n: Gauss nodes per face dimension.
        g: The log coefficient (``kernel_log_coefficient``), or None.  Then f(s) = rho^d [f(A e_k(v)) +
            log(rho) g(A e_k(v))] in each pyramid; the rho integrand of the log part is a polynomial P(rho) times
            rho^e log(rho), integrated exactly from P's shifted-Legendre coefficients (``_log_rho_moment``).

    Returns:
        The integrals, shape (len(pairs),).

    Raises:
        ValueError: when a pair's weight does not vanish fast enough for the integral to exist.
    """
    far = [Fraction(lo if hi == 0 else hi) for lo, hi in box]
    fac = [
        {k: _corner_factor(p, far[i]) for k, p in weights[i].items()} for i in range(3)
    ]
    max_deg = sum(max(len(p) for p in w.values()) - 1 for w in weights)
    n_rho = max(1, (3 + max(d, 0) + max_deg) // 2 + 2)
    if g is not None:
        n_rho = max(
            n_rho, max_deg + 1
        )  # P(rho) of degree <= max_deg: its Legendre coefficients exactly
    rho, w_rho = _gauss(n_rho, 0.0, 1.0)
    if g is not None:
        leg_rho = np.stack(
            [legval(2 * rho - 1, np.eye(n_rho)[k]) for k in range(n_rho)]
        )  # (k, node)
    v, w_v = _gauss(n, 0.0, 1.0)
    vol = abs(float(far[0] * far[1] * far[2]))
    out = np.zeros(len(pairs))
    for k in range(3):
        j1, j2 = [i for i in range(3) if i != k]
        g1, g2 = np.meshgrid(v, v, indexing="ij")
        pts = [None, None, None]
        pts[k] = np.full_like(g1, float(far[k]))
        pts[j1], pts[j2] = float(far[j1]) * g1, float(far[j2]) * g2
        face = f(*pts) * np.outer(w_v, w_v)  # (n, n)
        face_g = None if g is None else g(*pts) * np.outer(w_v, w_v)
        cache_face_g: dict = {}
        # per key: r_k(A_k rho) on the pyramid axis; v^m r_j(A_j rho v) on the other two
        on_k = {key: legval(2 * rho - 1, c) for key, (m, c) in fac[k].items()}
        on_j = [
            {
                key: v[None, :] ** m * legval(2 * np.outer(rho, v) - 1, c)
                for key, (m, c) in fac[i].items()
            }
            for i in (j1, j2)
        ]
        cache_face: dict = {}
        for idx, keys in enumerate(pairs):
            m_tot = sum(fac[i][keys[i]][0] for i in range(3))
            if (
                fac[0][keys[0]][1].any()
                and fac[1][keys[1]][1].any()
                and fac[2][keys[2]][1].any()
            ):
                e = 2 + d + m_tot
                if e < 0:
                    raise ValueError(
                        f"image_block: a divergent pyramid integral (degree {d}, zero of order {m_tot})."
                    )
            else:
                continue
            fk = (keys[j1], keys[j2])
            if fk not in cache_face:
                # sum_ab face[a, b] A[r, a] B[r, b], one axis at a time
                cache_face[fk] = np.sum(
                    (on_j[0][keys[j1]] @ face) * on_j[1][keys[j2]], axis=1
                )
            inner = cache_face[fk]
            scale = float(np.prod([far[i] ** fac[i][keys[i]][0] for i in range(3)]))
            out[idx] += vol * scale * np.sum(w_rho * rho**e * on_k[keys[k]] * inner)
            if face_g is not None:
                if fk not in cache_face_g:
                    cache_face_g[fk] = np.sum(
                        (on_j[0][keys[j1]] @ face_g) * on_j[1][keys[j2]], axis=1
                    )
                poly = on_k[keys[k]] * cache_face_g[fk]  # P(rho) at the nodes
                c = (2 * np.arange(n_rho) + 1) * (leg_rho @ (w_rho * poly))
                out[idx] += (
                    vol
                    * scale
                    * sum(c[kk] * _log_rho_moment(e, kk) for kk in range(n_rho))
                )
    return out


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
    basis: str = "legendre",
    order: int = 0,
) -> NDArray:
    """The static interface block K[a, c] between two cells: the image when both lie on one side of the
    interface z = 0, the transmission when they lie on opposite sides.  With ``order=1``, the block K1 of the
    omega^2 term instead (K = K0 + omega^2 K1 + ...), from the omega^2 terms with the finite-part potentials.

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
        closed: With basis 'power', corner pieces in closed form (``image_moments``); False uses Euler's
            reduction to the far faces with Gauss rules there (the two agree to 1e-15).
        basis: 'legendre' (default): every piece in a well-conditioned basis, regular pieces by Legendre
            moments of their own coordinates, corner pieces by Duffy pyramids with the weights' zeros at the
            corner factored out exactly and the rest in Legendre form: 4e-16 to 1.2e-15 against an
            extended-precision reference, degree one and two (scripts/measure_interface_image_bases.py).
            'power': the moments in powers about each piece's centre or the singular corner, contracted
            with the weights' power coefficients: the same at degree one, up to 1e-14 at degree two.

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
            basis,
            order,
        )
        sa = np.array([(-1.0) ** e[0] for e in exps_f])
        sc = np.array([(-1.0) ** e[0] for e in exps_s])
        return np.einsum("a,c,i,j,acij->acij", sa, sc, MIRROR_9, MIRROR_9, mirrored)
    if basis not in ("legendre", "power"):
        raise ValueError(
            f"image_block: basis must be 'legendre' or 'power', got {basis!r}"
        )
    geo = family_geometry(rec_centre, src_centre, n_source, n_test, order)
    mats = (above.lam, above.mu, below.lam, below.mu)
    if order == 1:
        mats += (above.rho, below.rho)
    elif order == 2:
        # the omega^4 terms hold their media (exact rationals, scripts/derive_interface_omega4.py): only those
        held = json.loads(_TERMS_PATH[geo["kind"], 2].read_text())["media"]
        want = [(m.lam, m.mu, m.rho) for m in (above, below)]
        have = [tuple(float(sp.Rational(v)) for v in held[t]) for t in ("A", "B")]
        if not np.allclose(np.array(want), np.array(have), rtol=1e-13):
            raise ValueError(
                "image_block: the omega^4 terms were derived for other media; run "
                "scripts/derive_interface_omega4.py for these (or use the symbolic Mathematica result)."
            )
        mats = ()
    groups: dict = {}
    for r, c, j, alpha, a_pow, b_pow, fn in _terms(geo["kind"], order):
        groups.setdefault((j, alpha, a_pow, b_pow), []).append((r, c, fn(*mats)))
    logs = family_log_integrals(geo, n_gauss)
    out = np.zeros((n_test, n_source, 9, 9))
    for key, val in family_integrals(geo, n_gauss, closed, basis).items():
        scale = h ** (6 + kernel_degree(key[0], key[1]) + key[2] + key[3])
        if key in logs:
            # a finite-part family is homogeneous only up to log: f(h s) = h^d [f(s) + log(h) g(s)]
            val = val + np.log(h) * logs[key]
        for r, c, coef in groups[key]:
            out[:, :, r, c] += scale * coef * val
    return out


def family_geometry(
    rec_centre: tuple[int, int, int],
    src_centre: tuple[int, int, int],
    n_source: int,
    n_test: int,
    order: int = 0,
) -> dict:
    """The pieces and the exact weights of every kernel family, for a source in A (src_centre[0] < 0).

    Returns:
        {"kind", "lat_x", "lat_y", "ver", "exps_f", "exps_s", "families": {(j, alpha, A, B): (wx, wy, wz, deg)}},
        the weights per (field exponent, source exponent) as exact polynomials in the separation coordinate,
        one per piece of the axis.
    """
    cz, czp = rec_centre[0], src_centre[0]
    exps_f = SOURCE_EXPONENTS_QUARTIC[:n_test]
    exps_s = SOURCE_EXPONENTS_QUARTIC[:n_source]
    kind = "reflected" if cz < 0 else "transmitted"
    dx, dy = rec_centre[1] - src_centre[1], rec_centre[2] - src_centre[2]
    z0 = -(cz + czp) if kind == "reflected" else cz - czp
    # pieces along each axis, in the separation coordinates: lateral rho = D + sigma, vertical zeta = z0 - tau
    lat_x = [(dx - 2, dx), (dx, dx + 2)]
    lat_y = [(dy - 2, dy), (dy, dy + 2)]
    if kind == "reflected":
        # zeta = z0 - tau: tau in [-2, 0] -> zeta in [z0, z0 + 2]; tau in [0, 2] -> [z0 - 2, z0]
        ver = [(z0, z0 + 2), (z0 - 2, z0)]
    else:
        # zeta = z0 + sigma: sigma in [-2, 0] -> zeta in [z0 - 2, z0]; sigma in [0, 2] -> [z0, z0 + 2]
        ver = [(z0 - 2, z0), (z0, z0 + 2)]
    families = {}
    for _, _, j, alpha, a_pow, b_pow, _ in _terms(kind, order):
        key = (j, alpha, a_pow, b_pow)
        if key in families:
            continue
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
        families[key] = (wx, wy, wz, deg)
    return {
        "kind": kind,
        "order": order,
        "lat_x": lat_x,
        "lat_y": lat_y,
        "ver": ver,
        "exps_f": exps_f,
        "exps_s": exps_s,
        "families": families,
    }


def family_integrals(
    geo: dict, n_gauss: int = 32, closed: bool = True, basis: str = "legendre"
) -> dict:
    """The (n_test, n_source) integrals of each kernel family over the pieces, in units of h (no material
    coefficient, no power of h): {(j, alpha, A, B): array}."""
    lat_x, lat_y, ver = geo["lat_x"], geo["lat_y"], geo["ver"]
    exps_f, exps_s = geo["exps_f"], geo["exps_s"]
    out = {}
    for (j, alpha, a_pow, b_pow), (wx, wy, wz, deg) in geo["families"].items():
        f = kernel_function(j, alpha)
        d = kernel_degree(j, alpha)
        g = kernel_log_coefficient(j, alpha)
        if basis == "legendre":
            out[j, alpha, a_pow, b_pow] = _block_legendre(
                f, d, wx, wy, wz, exps_f, exps_s, lat_x, lat_y, ver, deg, n_gauss, g
            )
            continue
        # image_moments' closed forms use the line of images: valid for j <= 2 with enough lateral
        # derivatives (it is another finite-part convention otherwise; scripts/derive_finite_part_potentials.py)
        closed_here = closed and j <= 2 and (j == 0 or alpha[0] + alpha[1] >= j)
        val = np.zeros((len(exps_f), len(exps_s)))
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
                            if closed_here
                            else _corner_moments(f, d, box, deg, n_gauss, g)
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
        out[j, alpha, a_pow, b_pow] = val
    return out


def family_log_integrals(geo: dict, n_gauss: int = 32) -> dict:
    """For each family with a log term, the integral of its log coefficient g (a polynomial) against the weights
    over the pieces, in units of h: {(j, alpha, A, B): array}.  The block of such a family scales as
    h^(6 + d + A + B) [int f + log(h) int g], f(h s) = h^d [f(s) + log(h) g(s)]."""
    out = {}
    for (j, alpha, a_pow, b_pow), (wx, wy, wz, deg) in geo["families"].items():
        g = kernel_log_coefficient(j, alpha)
        if g is None:
            continue
        out[j, alpha, a_pow, b_pow] = _block_legendre(
            g,
            0,
            wx,
            wy,
            wz,
            geo["exps_f"],
            geo["exps_s"],
            geo["lat_x"],
            geo["lat_y"],
            geo["ver"],
            deg,
            n_gauss,
            regular_only=True,
        )
    return out


def _block_legendre(
    f,
    d,
    wx,
    wy,
    wz,
    exps_f,
    exps_s,
    lat_x,
    lat_y,
    ver,
    deg,
    n_gauss,
    g=None,
    regular_only=False,
) -> NDArray:
    """One kernel family's (n_test, n_source) integrals in the Legendre basis of each piece.

    ``g``: the family's log coefficient, passed to the corner pyramids.  ``regular_only``: every piece by Gauss
    rules (for a polynomial kernel, exact).
    """
    val = np.zeros((len(exps_f), len(exps_s)))
    pair_keys = [
        ((ef[1], es[1]), (ef[2], es[2]), (ef[0], es[0]))
        for ef in exps_f
        for es in exps_s
    ]
    for ix in range(2):
        for iy in range(2):
            for iz in range(2):
                box = (lat_x[ix], lat_y[iy], ver[iz])
                ws = [
                    {k: v[ix] for k, v in wx.items()},
                    {k: v[iy] for k, v in wy.items()},
                    {k: v[iz] for k, v in wz.items()},
                ]
                if not regular_only and all(lo <= 0 <= hi for lo, hi in box):
                    if any(lo < 0 < hi for lo, hi in box):
                        raise ValueError(
                            "image_block: the singular point lies inside a piece, not at a corner."
                        )
                    val += _corner_pyramids(
                        f, d, box, ws, pair_keys, n_gauss, g
                    ).reshape(val.shape)
                    continue
                centre = [Fraction(lo + hi, 2) for lo, hi in box]
                mom = _regular_legendre_moments(f, box, deg, n_gauss)
                leg = [
                    {k: _legendre_coeffs(p, centre[i]) for k, p in ws[i].items()}
                    for i in range(3)
                ]
                for idx, keys in enumerate(pair_keys):
                    cx, cy, cz_ = (leg[i][keys[i]] for i in range(3))
                    sub = mom[: len(cx), : len(cy), : len(cz_)]
                    val.flat[idx] += np.einsum("i,j,k,ijk->", cx, cy, cz_, sub)
    return val
