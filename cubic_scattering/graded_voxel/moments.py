"""Closed-form master integrals: a monomial times an odd power of the distance, over a box at the origin.

    box_integral   I[p, q, r; a, b, c; m] = int_0^a int_0^b int_0^c x^p y^q z^r R^m dz dy dx
    face_integral  J[q, r; a; b, c; m]    = int_0^b int_0^c y^q z^r S^m dz dy
    line_integral  L[r; A^2; c; m]        = int_0^c z^r (A^2 + z^2)^(m/2) dz

with R^2 = x^2 + y^2 + z^2, S^2 = a^2 + y^2 + z^2 and m odd.  Every one is elementary (square roots, one
logarithm, one arctangent), by three reductions.

VOLUME TO FACES.  x^p y^q z^r R^m is homogeneous of degree d = p + q + r + m.  Euler's identity
div(x f) = (d + 3) f and the divergence theorem on the box (the faces through the origin carry no flux):

    (d + 3) I[p, q, r; m] = a^(p+1) J[q, r; a; b, c; m] + b^(q+1) J[p, r; b; a, c; m]
                            + c^(r+1) J[p, q; c; a, b; m].

FACES TO EDGES.  d/dy (y^(q-1) S^(m+2)) = (q - 1) y^(q-2) S^(m+2) + (m + 2) y^q S^m, over the face:

    (m + 2) J[q, r; m] = b^(q-1) L[r; a^2 + b^2; c; m + 2] - [q = 1] L[r; a^2; c; m + 2]
                         - (q - 1) J[q - 2, r; m + 2],

which lowers q by two at a time (and r likewise) down to q, r in {0, 1}; q = 1 is then a single edge term.
For q = r = 0, writing S^m = S^(m-2) (a^2 + y^2 + z^2) and using the same derivative identity,

    (m + 2) J[0, 0; m] = m a^2 J[0, 0; m - 2] + b L[0; a^2 + b^2; c; m] + c L[0; a^2 + c^2; b; m],
    J[0, 0; -3] = (1/a) arctan(b c / (a sqrt(a^2 + b^2 + c^2))),

the arctangent being the solid angle the rectangle subtends.  On a face THROUGH the origin (a = 0) the
integrand is homogeneous in (y, z) and Euler's identity in the plane gives
(q + r + m + 2) J = b^(q+1) L[r; b^2; c; m] + c^(r+1) L[q; c^2; b; m].

EDGES.  d/dz (z^(r-1) (A^2 + z^2)^((m+2)/2)) lowers r by two:

    (m + 2) L[r; m] = c^(r-1) (A^2 + c^2)^((m+2)/2) - [r = 1] A^(m+2) - (r - 1) L[r - 2; m + 2],
    (m + 1) L[0; m] = c (A^2 + c^2)^(m/2) + m A^2 L[0; m - 2],   L[0; -1] = asinh(c / A),
    L[0; -3] = c / (A^2 sqrt(A^2 + c^2)),   and for A = 0:  L[r; m] = c^(r+m+1) / (r + m + 1).

Arguments are integers (lengths in units of the cell's half-width: the box sides are 2 or 4), so every value
is a universal constant, cached.  Values are mpmath numbers at 40 digits; the recursions divide by small
odd integers only.
"""

from functools import cache

import mpmath as mp

DPS = 40


def _odd(m: int, who: str) -> None:
    if m % 2 == 0:
        raise ValueError(f"{who}: the power m of the distance must be odd, got {m}")


@cache
def line_integral(r: int, a2: int, c: int, m: int) -> mp.mpf:
    """L[r; A^2; c; m] = int_0^c z^r (A^2 + z^2)^(m/2) dz, m odd, A^2 = a2 >= 0.

    Raises:
        ValueError: when m is even, m < -3 with A > 0, or the integral diverges at z = 0 (A = 0, r + m < 0).
    """
    _odd(m, "line_integral")
    with mp.workdps(DPS):
        if a2 == 0:
            if r + m + 1 <= 0:
                raise ValueError(
                    f"line_integral: int_0^c z^{r + m} dz diverges at z = 0 (r = {r}, m = {m})"
                )
            return mp.mpf(c) ** (r + m + 1) / (r + m + 1)
        top = mp.mpf(a2 + c * c)
        if r >= 2:
            return (
                mp.mpf(c) ** (r - 1) * top ** (mp.mpf(m + 2) / 2)
                - (r - 1) * line_integral(r - 2, a2, c, m + 2)
            ) / (m + 2)
        if r == 1:
            return (top ** (mp.mpf(m + 2) / 2) - mp.mpf(a2) ** (mp.mpf(m + 2) / 2)) / (m + 2)
        if m == -1:
            return mp.asinh(c / mp.sqrt(a2))
        if m == -3:
            return c / (a2 * mp.sqrt(top))
        if m < -3:
            raise ValueError(f"line_integral: m = {m} < -3 is not needed and not implemented")
        return (c * top ** (mp.mpf(m) / 2) + m * a2 * line_integral(0, a2, c, m - 2)) / (m + 1)


@cache
def face_integral(q: int, r: int, a: int, b: int, c: int, m: int) -> mp.mpf:
    """J[q, r; a; b, c; m] = int_0^b int_0^c y^q z^r (a^2 + y^2 + z^2)^(m/2) dz dy, m odd.

    Raises:
        ValueError: when m is even, or the integral diverges (a = 0 and q + r + m <= -2), or m < -3.
    """
    _odd(m, "face_integral")
    with mp.workdps(DPS):
        if a == 0:
            deg = q + r + m + 2
            if deg <= 0:
                raise ValueError(
                    f"face_integral: y^{q} z^{r} rho^{m} diverges at the origin of the face "
                    f"(q + r + m = {q + r + m})"
                )
            return (
                mp.mpf(b) ** (q + 1) * line_integral(r, b * b, c, m)
                + mp.mpf(c) ** (r + 1) * line_integral(q, c * c, b, m)
            ) / deg
        if q >= 1:
            edge = mp.mpf(b) ** (q - 1) * line_integral(r, a * a + b * b, c, m + 2)
            if q == 1:
                return (edge - line_integral(r, a * a, c, m + 2)) / (m + 2)
            return (edge - (q - 1) * face_integral(q - 2, r, a, b, c, m + 2)) / (m + 2)
        if r >= 1:
            return face_integral(r, q, a, c, b, m)
        if m == -3:
            return mp.atan(mp.mpf(b * c) / (a * mp.sqrt(a * a + b * b + c * c))) / a
        if m < -3:
            raise ValueError(f"face_integral: m = {m} < -3 is not needed and not implemented")
        edges = b * line_integral(0, a * a + b * b, c, m) + c * line_integral(0, a * a + c * c, b, m)
        return (m * a * a * face_integral(0, 0, a, b, c, m - 2) + edges) / (m + 2)


@cache
def box_integral(p: int, q: int, r: int, a: int, b: int, c: int, m: int) -> mp.mpf:
    """I[p, q, r; a, b, c; m] = int over [0, a] x [0, b] x [0, c] of x^p y^q z^r R^m, m odd.

    Raises:
        ValueError: when m is even or the integral diverges at the origin (p + q + r + m <= -3).
    """
    _odd(m, "box_integral")
    deg = p + q + r + m + 3
    if deg <= 0:
        raise ValueError(
            f"box_integral: x^{p} y^{q} z^{r} R^{m} diverges at the origin "
            f"(p + q + r + m = {p + q + r + m})"
        )
    with mp.workdps(DPS):
        return (
            mp.mpf(a) ** (p + 1) * face_integral(q, r, a, b, c, m)
            + mp.mpf(b) ** (q + 1) * face_integral(p, r, b, a, c, m)
            + mp.mpf(c) ** (r + 1) * face_integral(p, q, c, a, b, m)
        ) / deg
