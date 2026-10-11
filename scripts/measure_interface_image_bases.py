"""Measure: the accuracy of the interface image blocks in the power basis and in the Legendre basis.

Paper: LatexPDFs/InterfaceCouplingIntegrals, the accuracy of the touching blocks.

WHAT.  For each kernel family d^alpha Phi_j z^A z'^B, the integrals over the pieces of the separation form, in
units of h and without material coefficients (``interface_image.family_integrals``), by
  - 'power':    moments in powers about each piece's centre (Gauss) or the singular corner (closed forms of
                ``image_moments``), contracted with the weights' power coefficients, in double precision;
  - 'legendre': regular pieces by Legendre moments of their own coordinates, corner pieces by Duffy pyramids
                with the weights' zeros at the corner factored out exactly, in double precision;
against a REFERENCE: the power route carried out in 80-bit extended precision (numpy.longdouble, eps 1.1e-19),
with the closed-form corner moments at 40 digits, Gauss nodes and weights refined to extended precision, the
kernels evaluated in extended precision (their rational constants are all dyadic; pi is supplied), and the
exact rational weights rounded to extended precision.  The reference shares the pieces and the closed forms with
'power', but not its rounding; it shares nothing with 'legendre' but the pieces.

Requires numpy.longdouble to be wider than double (x86-64 Linux); the script refuses otherwise.

Run:  PYTHONPATH=. python scripts/measure_interface_image_bases.py
"""

import time

import mpmath as mp
import numpy as np
import sympy as sp
from numpy.polynomial.legendre import leggauss

from cubic_scattering import interface_image as ii
from cubic_scattering.image_moments import corner_moment

LD = np.longdouble
LD_PI = LD("3.14159265358979323846264338327950288")
CASES = [  # (receiver centre, source centre, n_test, n_source, label)
    ((-1, 0, 0), (-1, 0, 0), 4, 10, "reflected, cell with its own image"),
    ((-1, 0, 0), (-1, 2, 0), 4, 10, "reflected, lateral neighbour"),
    ((-1, 0, 0), (-1, 2, 2), 4, 10, "reflected, diagonal neighbour"),
    ((1, 0, 0), (-1, 0, 0), 4, 10, "transmitted, face across"),
    ((1, 0, 0), (-1, 2, 2), 4, 10, "transmitted, corner across"),
    ((-1, 0, 0), (-1, 0, 0), 10, 35, "reflected, own image, degree two"),
    ((-1, 0, 0), (-1, 2, 2), 10, 35, "reflected, diagonal, degree two"),
    ((1, 0, 0), (-1, 0, 0), 10, 35, "transmitted, face across, degree two"),
]


def _ld_gauss(n: int, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights in extended precision (Newton-refined from double)."""
    x0, _ = leggauss(n)
    x = x0.astype(LD)
    for _ in range(4):
        p0, p1 = np.ones_like(x), x.copy()
        for k in range(2, n + 1):
            p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
        dp = n * (x * p1 - p0) / (x * x - 1)
        x = x - p1 / dp
    p0, p1 = np.ones_like(x), x.copy()
    for k in range(2, n + 1):
        p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
    dp = n * (x * p1 - p0) / (x * x - 1)
    w = 2 / ((1 - x * x) * dp * dp)
    return LD(lo) + (LD(hi) - LD(lo)) * (x + 1) / 2, w * (LD(hi) - LD(lo)) / 2


_KERN: dict = {}


def _ld_kernel(j: int, alpha: tuple[int, int, int]):
    if (j, alpha) not in _KERN:
        rx, ry, zeta = sp.symbols("rx ry zeta", real=True)
        r = sp.sqrt(rx**2 + ry**2 + zeta**2)
        phi = {
            0: 1 / (2 * sp.pi * r),
            1: -sp.log(r + zeta) / (2 * sp.pi),
            2: (zeta * sp.log(r + zeta) - r) / (2 * sp.pi),
        }[j]
        expr = (
            sp.diff(phi, rx, alpha[0], ry, alpha[1], zeta, alpha[2])
            if sum(alpha)
            else phi
        )
        _KERN[j, alpha] = sp.lambdify(
            (rx, ry, zeta), expr, [{"pi": LD_PI}, "numpy"], cse=True
        )
    return _KERN[j, alpha]


def _ld(fr) -> np.longdouble:
    return LD(fr.numerator) / LD(fr.denominator)


def reference(geo: dict, n_gauss: int = 40) -> dict:
    """``family_integrals(geo, basis='power')`` in extended precision."""
    out = {}
    exps_f, exps_s = geo["exps_f"], geo["exps_s"]
    for (j, alpha, a_pow, b_pow), (wx, wy, wz, deg) in geo["families"].items():
        f = _ld_kernel(j, alpha)
        d = ii.kernel_degree(j, alpha)
        val = np.zeros((len(exps_f), len(exps_s)), dtype=LD)
        for ix in range(2):
            for iy in range(2):
                for iz in range(2):
                    box = (geo["lat_x"][ix], geo["lat_y"][iy], geo["ver"][iz])
                    if all(lo <= 0 <= hi for lo, hi in box):
                        centre = (0, 0, 0)
                        sides = tuple(int(max(abs(lo), abs(hi))) for lo, hi in box)
                        signs = tuple(-1 if lo < 0 else 1 for lo, _ in box[:2])
                        mom = np.full(
                            (deg[0] + 1, deg[1] + 1, deg[2] + 1), LD("nan"), dtype=LD
                        )
                        for p in range(deg[0] + 1):
                            for q in range(deg[1] + 1):
                                for r in range(deg[2] + 1):
                                    if 3 + p + q + r + d > 0:
                                        mom[p, q, r] = LD(
                                            mp.nstr(
                                                corner_moment(
                                                    j, alpha, (p, q, r), sides, signs
                                                ),
                                                30,
                                            )
                                        )
                    else:
                        centre = tuple((lo + hi) // 2 for lo, hi in box)
                        rules = [_ld_gauss(n_gauss, lo, hi) for lo, hi in box]
                        sx, sy, sz = np.meshgrid(
                            rules[0][0], rules[1][0], rules[2][0], indexing="ij"
                        )
                        fw = f(sx, sy, sz) * np.einsum(
                            "i,j,k->ijk", rules[0][1], rules[1][1], rules[2][1]
                        )
                        vs = [
                            np.vander(
                                rules[i][0] - LD(centre[i]), deg[i] + 1, increasing=True
                            )
                            for i in range(3)
                        ]
                        mom = np.einsum("ijk,ia,jb,kc->abc", fw, *vs)
                    for a, ef in enumerate(exps_f):
                        for c, es in enumerate(exps_s):
                            ps = [
                                ii._shift(
                                    wx[ef[1], es[1]][ix],
                                    ii.Fraction(centre[0]),
                                    ii.Fraction(1),
                                ),
                                ii._shift(
                                    wy[ef[2], es[2]][iy],
                                    ii.Fraction(centre[1]),
                                    ii.Fraction(1),
                                ),
                                ii._shift(
                                    wz[ef[0], es[0]][iz],
                                    ii.Fraction(centre[2]),
                                    ii.Fraction(1),
                                ),
                            ]
                            nz = [[k for k, x in enumerate(p) if x] for p in ps]
                            if not all(nz):
                                continue
                            sub = mom[np.ix_(*nz)]
                            cs = [
                                np.array([_ld(ps[i][k]) for k in nz[i]], dtype=LD)
                                for i in range(3)
                            ]
                            val[a, c] += np.einsum("i,j,k,ijk->", *cs, sub)
        out[j, alpha, a_pow, b_pow] = val
    return out


def main() -> None:
    if np.finfo(LD).eps > 1e-18:
        raise SystemExit(
            "numpy.longdouble is not wider than double here: no extended-precision reference."
        )
    print(
        f"{'case':42s} {'power':>9s} {'legendre':>9s}   (max over families of max|error| / max|reference|)"
    )
    for rc, sc, nt, ns, label in CASES:
        geo = ii.family_geometry(rc, sc, ns, nt)
        t = time.time()
        ref = reference(geo)
        t_ref = time.time() - t
        res = {}
        for basis in ("power", "legendre"):
            t = time.time()
            res[basis] = ii.family_integrals(geo, basis=basis)
            res[basis + "_t"] = time.time() - t
        errs = {}
        for basis in ("power", "legendre"):
            errs[basis] = max(
                float(
                    np.abs(res[basis][k].astype(LD) - ref[k]).max()
                    / np.abs(ref[k]).max()
                )
                for k in ref
                if np.abs(ref[k]).max() > 0
            )
        print(
            f"{label:42s} {errs['power']:9.1e} {errs['legendre']:9.1e}   "
            f"(times: power {res['power_t']:.0f}s, legendre {res['legendre_t']:.0f}s, reference {t_ref:.0f}s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
