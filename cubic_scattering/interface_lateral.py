"""Finite parts of the interface plane-wave integrals at small q, analytically: no sampling.

Plan: docs/2026-10-10-stratified-reference-legendre-cells-3d.md (13 October: the analytic small-q head).
Paper: LatexPDFs/InterfaceCouplingIntegrals, Eq. (headseries) and the omega^2 section.

THE HEAD.  FP int_{q < q_s} Y_order d^2k, Y_order the static (0), omega^2 (1) or omega^4 (2) integrand of a
block (``interface_spectral``), with the finite part by continuation in the power of q (the convention of
the potentials Phi_j).  Near q = 0 every factor of the integrand is an exact power series in q:
  * the frame kernel, a Laurent polynomial q^(m0+m) (``interface_spectral.frame_laurent``);
  * the vertical moments int_cell f(v) z^a e^{s q z} dz = sum_n (s q)^n / n! int_cell f(v) z^(a+n) dz, exact
    polynomial moments (entire in q);
  * the lateral moments with the phase, for each harmonic e^{i n phi}: the Bessel series of J_|n|(q rho)
    against the exact lateral weights w(sigma) = int P_E(v) (v - sigma)^F dv (``lateral_head``).
The product is a Laurent series sum_p R_p[n] q^p; each power and harmonic is integrated exactly,
    FP int_0^{q_s} q^(p+1) S_n(q) dq = sum_k c_k ... q_s^(e+1)/(e+1)   (log q_s at e = -1),
and the sum truncated where the series' terms fall below rounding.  The series converge for every q
(entire), but lose digits as q_s times the largest distance of the pair grows: q_s is taken as min(1/h,
4/R), R that distance.  q is in km^-1: a finite part depends on the unit of length, as the closed forms do.
"""

from functools import cache
from math import factorial

import numpy as np
import sympy as sp
from numpy.polynomial import polynomial as nppoly

# ---------------------------------------------------------------------------
# The lateral moments: exact weights and the Bessel series
# ---------------------------------------------------------------------------


@cache
def weight(E: int, F: int) -> tuple:
    """w(sigma) = int P_E(v) (v - sigma)^F dv on the two pieces [-2, 0], [0, 2]: power coefficients in
    sigma."""
    v, sg = sp.symbols("v sigma")
    pe = sp.legendre(E, v)
    lo = sp.integrate(pe * (v - sg) ** F, (v, -1, sg + 1))
    hi = sp.integrate(pe * (v - sg) ** F, (v, sg - 1, 1))
    return tuple(
        np.array([float(c) for c in sp.Poly(sp.expand(x), sg).all_coeffs()[::-1]]) for x in (lo, hi)
    )


def compose(c: np.ndarray, delta: float) -> np.ndarray:
    """Coefficients of w(u - delta) in u from those of w(sigma)."""
    out = np.zeros(len(c))
    for k, ck in enumerate(c):
        out[: k + 1] += ck * nppoly.polypow([-delta, 1.0], k)[: k + 1]
    return out


@cache
def rect_poly_moments(x0, x1, y0, y1, deg: int) -> np.ndarray:
    """int over the rectangle of u_x^i u_y^j, i, j <= deg, exactly."""
    # floats: numpy integer corners would raise to high powers in int64 and wrap silently
    x0, x1, y0, y1 = (float(v) for v in (x0, x1, y0, y1))
    ix = np.array([(x1 ** (i + 1) - x0 ** (i + 1)) / (i + 1) for i in range(deg + 1)])
    iy = np.array([(y1 ** (j + 1) - y0 ** (j + 1)) / (j + 1) for j in range(deg + 1)])
    return np.outer(ix, iy)


@cache
def bessel_poly(n: int, k: int) -> np.ndarray:
    """(u_x + i sgn(n) u_y)^|n| (u_x^2 + u_y^2)^k as a 2-D power-coefficient array."""
    nu, sg = abs(n), (1 if n >= 0 else -1)
    out = np.zeros((nu + 2 * k + 1, nu + 2 * k + 1), dtype=complex)
    base = np.zeros((nu + 1, nu + 1), dtype=complex)
    for i in range(nu + 1):
        base[i, nu - i] = float(sp.binomial(nu, i)) * (1j * sg) ** (nu - i)
    rad = np.zeros((2 * k + 1, 2 * k + 1))
    for i in range(k + 1):
        rad[2 * i, 2 * (k - i)] = float(sp.binomial(k, i))
    for (i, j), v in np.ndenumerate(base):
        if v:
            out[i : i + 2 * k + 1, j : j + 2 * k + 1] += v * rad
    return out


@cache
def _poly_moments(n: int, k: int, ea: tuple, ec: tuple, delta: tuple) -> complex:
    """int over the separation u of w_x w_y (u_x + i sgn u_y)^|n| |u|^(2k), u in units of h."""
    pieces_x = [compose(c, float(delta[0])) for c in weight(ea[1], ec[1])]
    pieces_y = [compose(c, float(delta[1])) for c in weight(ea[2], ec[2])]
    poly = bessel_poly(n, k)
    mom = 0.0
    for (x0, x1), cx in zip(((delta[0] - 2, delta[0]), (delta[0], delta[0] + 2)), pieces_x, strict=True):
        for (y0, y1), cy in zip(
            ((delta[1] - 2, delta[1]), (delta[1], delta[1] + 2)), pieces_y, strict=True
        ):
            w2 = np.outer(cx, cy)
            full = np.zeros(
                (w2.shape[0] + poly.shape[0], w2.shape[1] + poly.shape[1]),
                dtype=complex,
            )
            for (i, j), v in np.ndenumerate(w2):
                if v:
                    full[i : i + poly.shape[0], j : j + poly.shape[1]] += v * poly
            rm = rect_poly_moments(x0, x1, y0, y1, max(full.shape) - 1)
            mom += np.sum(full * rm[: full.shape[0], : full.shape[1]])
    return mom


def lateral_head(
    p: int, n: int, ea: tuple, ec: tuple, delta: tuple, qs: float, h: float, kmax: int
) -> complex:
    """(1/4 pi^2) FP int_{q < q_s} d^2k q^p e^{i n phi} L_a L_c e^{i k.Delta h}, exactly: J_|n|(q rho) by
    its Taylor series, FP int_0^{q_s} q^e dq = q_s^(e+1)/(e+1) (log q_s at e = -1), the polynomials
    rho^(|n|+2k) e^{i n theta} against the lateral weights.  q_s in km^-1, h in km."""
    nu = abs(n)
    pref = 1j**n * (-1.0) ** (nu if n < 0 else 0) / (2 * np.pi)
    total = 0.0
    for k in range(kmax + 1):
        ck = (-1) ** k / (2.0 ** (nu + 2 * k) * factorial(k) * factorial(nu + k))
        e = p + 1 + nu + 2 * k
        fq = np.log(qs) if e == -1 else qs ** (e + 1) / (e + 1)
        total += ck * h ** (nu + 2 * k) * fq * _poly_moments(n, k, ea, ec, delta)
    return pref * h**4 * total


# ---------------------------------------------------------------------------
# The vertical moments as power series and the head
# ---------------------------------------------------------------------------


def _vertical_series(poly, zc: float, h: float, s: int, a_max: int, n_terms: int) -> np.ndarray:
    """V[a] = int_cell f(v) z^a e^{s q z} dz = sum_n (s q)^n / n! int f(v) z^(a+n) dz: coefficients of q^n,
    shape (a_max + 1, n_terms).  f has power coefficients ``poly`` in v, z = zc + h v."""
    poly = np.asarray(poly, float)
    out = np.zeros((a_max + 1, n_terms))
    zp = np.array([1.0])  # (zc + h v)^j, power coefficients in v
    moments = []
    for _ in range(a_max + n_terms):
        prod = nppoly.polymul(poly, zp)
        moments.append(h * sum(c * (1 - (-1) ** (i + 1)) / (i + 1) for i, c in enumerate(prod)))
        zp = nppoly.polymul(zp, [zc, h])
    for a in range(a_max + 1):
        for nn in range(n_terms):
            out[a, nn] = s**nn / factorial(nn) * moments[a + nn]
    return out


def head(
    order: int,
    kind: str,
    rec: tuple,
    src: tuple,
    h: float,
    above: tuple,
    below: tuple,
    qs: float,
    n_source: int,
    n_test: int,
    n_terms: int = 36,
) -> np.ndarray:
    """(1/4 pi^2) FP int_{q < q_s} Y_order d^2k in closed form (power series throughout), shape
    (n_test, n_source, 9, 9).  q_s in km^-1; ``qs_for`` gives the recommended value."""
    from . import (
        interface_spectral as isp,
    )  # the frame kernels and the rotation; isp imports this module

    C, m0 = isp.frame_laurent(kind, order, above, below)
    s = 1 if kind == "R" else -1
    exps_f = isp.SOURCE_EXPONENTS_QUARTIC[:n_test]
    exps_s = isp.SOURCE_EXPONENTS_QUARTIC[:n_source]
    pf, ps = isp._vertical_polys(exps_f, True), isp._vertical_polys(exps_s, False)
    zr, zs = rec[0] * h, src[0] * h
    delta = (rec[1] - src[1], rec[2] - src[2])
    reach = qs * h * (np.hypot(*delta) + 2 * np.sqrt(2))
    kmax = 6
    while (reach / 2) ** (2 * kmax) / factorial(kmax) ** 2 > 1e-19:
        kmax += 1
    na, nb, nm = C.shape[2], C.shape[3], C.shape[4]
    out = np.zeros((n_test, n_source, 81), dtype=complex)
    for ell, polyl in pf.items():
        vr = _vertical_series(polyl, zr, h, s, na - 1, n_terms)
        fa = np.array([a for a, x in enumerate(exps_f) if x[0] == ell])
        for e, polye in ps.items():
            vs = _vertical_series(polye, zs, h, 1, nb - 1, n_terms)
            sc = np.array([c for c, x in enumerate(exps_s) if x[0] == e])
            # the product's coefficient of q^(m0 + t), t < n_terms: sum over m + n1 + n2 = t
            g = np.zeros((n_terms, 9, 9), dtype=complex)
            for m in range(nm):
                for n1 in range(n_terms - m):
                    n2 = np.arange(n_terms - m - n1)
                    g[m + n1 + n2] += np.einsum("ijab,a,bn->nij", C[:, :, :, :, m], vr[:, n1], vs[:, n2])
            rot = isp._rotate_harmonics(g @ isp._M_INV)  # (t, 9 harmonics, 81)
            for t in range(n_terms):
                if not np.any(rot[t]):
                    continue
                p = m0 + t
                for a in fa:
                    ea = exps_f[a]
                    for c in sc:
                        ec = exps_s[c]
                        for k, n in enumerate(isp._HARM):
                            out[a, c] += lateral_head(p, int(n), ea, ec, delta, qs, h, kmax) * rot[t, k]
    return out.reshape(n_test, n_source, 9, 9)


def qs_for(rec: tuple, src: tuple, h: float) -> float:
    """The split point q_s (km^-1): min(1/h, 4/R), R the largest distance between points of the two cells
    (or of the receiver and the source's image), so that q_s R <= 4 and the series lose few digits."""
    dist = h * (np.hypot(rec[1] - src[1], rec[2] - src[2]) + 2 * np.sqrt(2) + abs(rec[0]) + abs(src[0]) + 2)
    return min(1.0 / h, 4.0 / dist)
