"""Closed-form corner moments of the static interface image (route A, the face closed forms).

Plan: ``docs/2026-10-10-stratified-reference-legendre-cells-3d.md``.  Used by ``interface_image``.

THE MOMENTS.  On a piece of the s-form holding the singular corner (a box [0, a] x [0, b] x [0, c] in
(|rho_x|, |rho_y|, zeta), after reflecting the lateral axes),

    M[p, q, r] = int_box rho_x^p rho_y^q zeta^r  d^alpha Phi_j(rho_x, rho_y, zeta),

for Phi0 = 1/(2 pi R), Phi1 = -log(R + zeta)/(2 pi), Phi2 = (zeta log(R + zeta) - R)/(2 pi).

PHI0.  d^alpha (1/R) = P_alpha(x, y, z) R^-(2|alpha| + 1), with P_alpha homogeneous of degree |alpha|, so M
is a sum of Paper 2's master integrals  I[p, q, r; a, b, c; m] = int_box x^p y^q z^r R^m  (m odd), here
needed down to m = -9 (Paper 2's ``graded_voxel.moments`` stops at -3).

THE MINDLIN POTENTIALS: A LINE OF IMAGES.  Since d_zeta Phi1 = -Phi0 and d_zeta Phi2 = -Phi1, and every
Mindlin term carries a lateral derivative (so it vanishes as zeta -> oo),

    d^alpha Phi1(rho, zeta) = int_zeta^oo d^alpha Phi0(rho, t) dt,
    d^alpha Phi2(rho, zeta) = int_zeta^oo (t - zeta) d^alpha Phi0(rho, t) dt.

Doing the zeta integral of the box first leaves Phi0 against a polynomial weight in t: on [0, c]
t^(r+1)/(r+1) (Phi1) or t^(r+2)/((r+1)(r+2)) (Phi2), and on the column [c, oo) c^(r+1)/(r+1) (Phi1) or
t c^(r+1)/(r+1) - c^(r+2)/(r+2) (Phi2).  The column integral converges (the Mindlin terms decay at least as
t^-2), and is the semi-infinite master integral minus the finite one.

SEMI-INFINITE MASTER INTEGRALS.  Euler's identity holds on the semi-infinite box, the face at infinity
carrying no flux; the strip faces reduce to one-dimensional integrals to infinity,

    int_0^oo z^r (A^2 + z^2)^(m/2) dz = A^(r+m+1) B((r+1)/2, -(r+m+1)/2) / 2,

and to integrals int_0^b y^q (A^2 + y^2)^k dy with k an integer or half-integer (logarithm, arctangent,
square roots).  Every value is elementary; with integer sides they are universal constants, cached, at 40
digits.
"""

from functools import cache

import mpmath as mp
import sympy as sp

DPS = 40


# ---------------------------------------------------------------------------
# One dimension
# ---------------------------------------------------------------------------


@cache
def line(r: int, a2: int, c: int, s: int) -> mp.mpf:
    """int_0^c z^r (a2 + z^2)^(s/2) dz for any integer s, a2 >= 0 (a2 = 0 needs r + s > -1)."""
    with mp.workdps(DPS):
        if a2 == 0:
            if r + s + 1 <= 0:
                raise ValueError(f"line: z^{r + s} diverges at z = 0")
            return mp.mpf(c) ** (r + s + 1) / (r + s + 1)
        a2m = mp.mpf(a2)
        top = a2m + c * c
        if s % 2 == 0 and s >= 0:
            k = s // 2
            return mp.fsum(
                mp.binomial(k, i)
                * a2m ** (k - i)
                * mp.mpf(c) ** (r + 2 * i + 1)
                / (r + 2 * i + 1)
                for i in range(k + 1)
            )
        if r >= 2:
            # d/dz (z^(r-1) top^((s+2)/2)) = (r-1) z^(r-2) top^((s+2)/2) + (s+2) z^r top^(s/2)
            if s == -2:
                return line(r - 2, a2, c, 0) - a2m * line(r - 2, a2, c, -2)
            return (
                mp.mpf(c) ** (r - 1) * top ** (mp.mpf(s + 2) / 2)
                - (r - 1) * line(r - 2, a2, c, s + 2)
            ) / (s + 2)
        if r == 1:
            if s == -2:
                return mp.log(top / a2m) / 2
            return (top ** (mp.mpf(s + 2) / 2) - a2m ** (mp.mpf(s + 2) / 2)) / (s + 2)
        # r == 0
        if s == -1:
            return mp.asinh(c / mp.sqrt(a2m))
        if s == -2:
            return mp.atan(c / mp.sqrt(a2m)) / mp.sqrt(a2m)
        if s < -2:
            # (s+3) L[s+2] = c top^((s+2)/2) + (s+2) a2 L[s]
            return (
                (s + 3) * line(0, a2, c, s + 2) - c * top ** (mp.mpf(s + 2) / 2)
            ) / ((s + 2) * a2m)
        # s >= 1 odd: (s+1) L[s] = c top^(s/2) + s a2 L[s-2]
        return (c * top ** (mp.mpf(s) / 2) + s * a2m * line(0, a2, c, s - 2)) / (s + 1)


@cache
def line_inf(r: int, a2: int, s: int) -> mp.mpf:
    """int_0^oo z^r (a2 + z^2)^(s/2) dz = a^(r+s+1) B((r+1)/2, -(r+s+1)/2) / 2, for r + s < -1, a2 > 0."""
    if r + s + 1 >= 0 or a2 <= 0:
        raise ValueError(f"line_inf: diverges (r = {r}, s = {s}, a2 = {a2})")
    with mp.workdps(DPS):
        a = mp.sqrt(mp.mpf(a2))
        return a ** (r + s + 1) * mp.beta(mp.mpf(r + 1) / 2, -mp.mpf(r + s + 1) / 2) / 2


# ---------------------------------------------------------------------------
# Faces and boxes, finite and semi-infinite (the third side to infinity)
# ---------------------------------------------------------------------------


@cache
def face(q: int, r: int, a: int, b: int, c: int | None, m: int) -> mp.mpf:
    """int_0^b int_0^c y^q z^r (a^2 + y^2 + z^2)^(m/2) dz dy, m odd; c = None means z to infinity (a > 0)."""
    with mp.workdps(DPS):
        if c is None:
            # z first, to infinity: (a^2 + y^2)^((r+m+1)/2) B(...) / 2, then y over [0, b]
            if r + m + 1 >= 0:
                raise ValueError(f"face: z^{r} R^{m} does not decay to infinity")
            beta = mp.beta(mp.mpf(r + 1) / 2, -mp.mpf(r + m + 1) / 2) / 2
            return beta * line(q, a * a, b, r + m + 1)
        if a == 0:
            deg = q + r + m + 2
            if deg <= 0:
                raise ValueError("face: diverges at the origin of a face through it")
            return (
                mp.mpf(b) ** (q + 1) * line(r, b * b, c, m)
                + mp.mpf(c) ** (r + 1) * line(q, c * c, b, m)
            ) / deg
        if q >= 1:
            edge = mp.mpf(b) ** (q - 1) * line(r, a * a + b * b, c, m + 2)
            if q == 1:
                return (edge - line(r, a * a, c, m + 2)) / (m + 2)
            return (edge - (q - 1) * face(q - 2, r, a, b, c, m + 2)) / (m + 2)
        if r >= 1:
            return face(r, q, a, c, b, m)
        if m == -3:
            return (
                mp.atan(mp.mpf(b * c) / (a * mp.sqrt(mp.mpf(a * a + b * b + c * c))))
                / a
            )
        edges = lambda mm: (
            b * line(0, a * a + b * b, c, mm) + c * line(0, a * a + c * c, b, mm)
        )
        if m < -3:
            # (m+4) J[m+2] = (m+2) a^2 J[m] + edges(m+2)
            return ((m + 4) * face(0, 0, a, b, c, m + 2) - edges(m + 2)) / (
                (m + 2) * a * a
            )
        return (m * a * a * face(0, 0, a, b, c, m - 2) + edges(m)) / (m + 2)


class EulerPole(ValueError):
    """Euler's identity is degenerate (degree zero): the integral needs another reduction."""


@cache
def box(
    p: int,
    q: int,
    r: int,
    a: int,
    b: int,
    c: int | None,
    m: int,
    finite_part: bool = False,
) -> mp.mpf:
    """int over [0, a] x [0, b] x [0, c] of x^p y^q z^r R^m, m odd; c = None means z to infinity.

    Euler's identity: (p + q + r + m + 3) I = sum over the far faces; the face at infinity carries none.
    With ``finite_part`` a negative degree is allowed: the result is then Hadamard's finite part, the
    analytic continuation in the degree.  Two such values over boxes sharing the corner differ by the exact
    integral over their difference, which is how the columns use it.
    """
    deg = p + q + r + m + 3
    if deg == 0 and finite_part:
        raise EulerPole(f"box: degree zero (x^{p} y^{q} z^{r} R^{m})")
    if deg <= 0 and not finite_part:
        raise ValueError(f"box: x^{p} y^{q} z^{r} R^{m} diverges at the origin")
    if c is None and r + m + 1 >= 0:
        raise ValueError(f"box: z^{r} R^{m} does not decay to infinity")
    with mp.workdps(DPS):
        total = mp.mpf(a) ** (p + 1) * face(q, r, a, b, c, m) + mp.mpf(b) ** (
            q + 1
        ) * face(p, r, b, a, c, m)
        if c is not None:
            total += mp.mpf(c) ** (r + 1) * face(p, q, c, a, b, m)
        return total / deg


@cache
def column(p: int, q: int, r: int, a: int, b: int, c: int, m: int) -> mp.mpf:
    """int over [0, a] x [0, b] x [c, oo) of x^p y^q z^r R^m: the semi-infinite box less the finite one,
    each by Euler's identity (finite parts where the degree is negative)."""
    try:
        return box(p, q, r, a, b, None, m, True) - box(p, q, r, a, b, c, m, True)
    except EulerPole:
        return _column_degree_zero(p, q, r, a, b, c, m)


def _strip(q: int, a: int, b: int, c: int, m: int) -> mp.mpf:
    """int_0^b int_c^oo y^q (a^2 + y^2 + t^2)^(m/2) dt dy, a > 0."""
    return face(q, 0, a, b, None, m) - face(q, 0, a, b, c, m)


@cache
def _column_degree_zero(
    p: int, q: int, r: int, a: int, b: int, c: int, m: int
) -> mp.mpf:
    """The column when p + q + r + m + 3 = 0, where Euler's identity is degenerate.

    Integration by parts in t (from c to infinity) lowers r by two, its boundary a rectangle at height c;
    then in x and in y, the boundaries strips at x = a or y = b (never through the origin).  What is left
    is one of two base integrals, B1 = column of R^-3 and B2 = column of x y R^-5.
    """
    with mp.workdps(DPS):
        if r >= 2:
            return (
                -(mp.mpf(c) ** (r - 1)) * face(p, q, c, a, b, m + 2)
                - (r - 1) * column(p, q, r - 2, a, b, c, m + 2)
            ) / (m + 2)
        if r == 1:
            return -face(p, q, c, a, b, m + 2) / (m + 2)
        if p >= 2:
            return (
                mp.mpf(a) ** (p - 1) * _strip(q, a, b, c, m + 2)
                - (p - 1) * column(p - 2, q, 0, a, b, c, m + 2)
            ) / (m + 2)
        if q >= 2:
            return (
                mp.mpf(b) ** (q - 1) * _strip(p, b, a, c, m + 2)
                - (q - 1) * column(p, q - 2, 0, a, b, c, m + 2)
            ) / (m + 2)
        if (p, q, m) == (0, 0, -3):
            return base_b1(a, b, c)
        if (p, q, m) == (1, 1, -5):
            return base_b2(a, b, c)
        raise ValueError(
            f"_column_degree_zero: no reduction for p={p}, q={q}, r={r}, m={m}"
        )


def _polar(a: int, b: int, g) -> mp.mpf:
    """int over the rectangle [0, a] x [0, b] in polar form: sum over its two triangles of int g(theta, R(theta))."""
    th0 = mp.atan2(b, a)
    return mp.quad(lambda th: g(th, a / mp.cos(th)), [0, th0]) + mp.quad(
        lambda th: g(th, b / mp.sin(th)), [th0, mp.pi / 2]
    )


@cache
def base_b1(a: int, b: int, c: int) -> mp.mpf:
    """B1 = int over [0, a] x [0, b] x [c, oo) of R^-3: the radial integral is log(c + sqrt(r^2 + c^2)) - log(2c),
    leaving one smooth angular integral (here by quadrature at 40 digits; its closed form is the Mathematica phase)."""
    with mp.workdps(DPS):
        c = mp.mpf(c)
        return _polar(
            a, b, lambda th, rr: mp.log(c + mp.sqrt(rr * rr + c * c)) - mp.log(2 * c)
        )


@cache
def base_b2(a: int, b: int, c: int) -> mp.mpf:
    """B2 = int over [0, a] x [0, b] x [c, oo) of x y R^-5: radial antiderivative c/(3 rho) + (2/3) log(c + rho),
    rho = sqrt(r^2 + c^2), times cos sin, leaving one smooth angular integral (quadrature at 40 digits)."""
    with mp.workdps(DPS):
        c = mp.mpf(c)

        def g2(rr):
            rho = mp.sqrt(rr * rr + c * c)
            return c / (3 * rho) + 2 * mp.log(c + rho) / 3

        return _polar(a, b, lambda th, rr: mp.cos(th) * mp.sin(th) * (g2(rr) - g2(0)))


# ---------------------------------------------------------------------------
# The corner moments
# ---------------------------------------------------------------------------


@cache
def phi0_derivative(
    alpha: tuple[int, int, int],
) -> tuple[tuple[tuple[int, int, int], sp.Rational], ...]:
    """d^alpha (1/R) = sum coef x^i y^j z^k  R^-(2|alpha|+1): the monomials and their coefficients."""
    x, y, z = sp.symbols("x y z", real=True)
    r = sp.sqrt(x**2 + y**2 + z**2)
    n = sum(alpha)
    expr = sp.diff(1 / r, x, alpha[0], y, alpha[1], z, alpha[2]) if n else 1 / r
    poly = sp.Poly(sp.expand(sp.simplify(expr * r ** (2 * n + 1))), x, y, z)
    return tuple(
        (mon, sp.Rational(co))
        for mon, co in zip(poly.monoms(), poly.coeffs(), strict=True)
    )


def corner_moment(
    j: int,
    alpha: tuple[int, int, int],
    n: tuple[int, int, int],
    sides: tuple[int, int, int],
    signs: tuple[int, int],
) -> mp.mpf:
    """int over the corner box of rho_x^p rho_y^q zeta^r d^alpha Phi_j, in closed form.

    Args:
        j: The potential, 0, 1 or 2.
        alpha: Derivative orders in (rho_x, rho_y, zeta); for j > 0 lateral only.
        n: The monomial (p, q, r).
        sides: (a, b, c), the box's extents in |rho_x|, |rho_y|, zeta (integers).
        signs: The sign of rho_x and of rho_y on the box (the box is reflected onto the positive octant).

    Returns:
        The moment, 40 digits.
    """
    if j > 0 and alpha[2]:
        raise ValueError(
            "corner_moment: a zeta derivative of a Mindlin potential is to be reduced first"
        )
    p, q, r = n
    a, b, c = sides
    m = -(2 * sum(alpha) + 1)
    total = mp.mpf(0)
    with mp.workdps(DPS):
        for (i, k, l), co in phi0_derivative(alpha):
            # reflect: rho_x = sx u, rho_y = sy v; the monomial and P_alpha pick up the signs
            sgn = signs[0] ** (p + i) * signs[1] ** (q + k)
            coef = mp.mpf(co.p) / co.q * sgn
            pp, qq = p + i, q + k
            if j == 0:
                val = box(pp, qq, r + l, a, b, c, m)
            elif j == 1:
                val = (
                    box(pp, qq, r + 1 + l, a, b, c, m)
                    + mp.mpf(c) ** (r + 1) * column(pp, qq, l, a, b, c, m)
                ) / (r + 1)
            else:
                inner = box(pp, qq, r + 2 + l, a, b, c, m) / ((r + 1) * (r + 2))
                col1 = column(pp, qq, l + 1, a, b, c, m)
                col0 = column(pp, qq, l, a, b, c, m)
                val = (
                    inner
                    + mp.mpf(c) ** (r + 1) / (r + 1) * col1
                    - mp.mpf(c) ** (r + 2) / (r + 2) * col0
                )
            total += coef * val
        return total / (2 * mp.pi)
