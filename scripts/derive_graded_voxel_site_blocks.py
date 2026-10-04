#!/usr/bin/env python3
"""The static single-site blocks of the uniform graded voxel, entry by entry, in exact arithmetic.

For a cell of half-width h with a contrast uniform in it, T36 = F A^-1 M with A = M - K(0) E reduces on
the irreducible representations of the cube group (``check_graded_voxel_site_symmetry.py``).  This script
derives the entries of the reduced blocks in the static limit (omega -> 0) as exact expressions:

  1. the master integrals of ``graded_voxel.moments`` re-derived in exact arithmetic (sympy), by the same
     three reductions, and compared with the 40-digit values;
  2. the universal moments U_ac(m, idx) of the self cell for the static terms (m = -1, 1), a, c over the
     four functions 1, xi_i, from the pieces of the autocorrelation W, exactly;
  3. the static self block K(0), compared with ``blocks.near_block`` evaluated numerically;
  4. a symmetry-adapted basis, and the reduced blocks of K(0) in it.

Run:  conda run -n seismic python -u scripts/derive_graded_voxel_site_blocks.py
"""

import itertools
import sys
from functools import cache
from pathlib import Path

import numpy as np
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cubic_scattering import ReferenceMedium  # noqa: E402
from cubic_scattering.graded_voxel.blocks import (  # noqa: E402
    _moves_closed,
    family_tables,
    near_block,
    radial_monomials,
    static_term_integral_closed,
    static_term_table,
)
from cubic_scattering.graded_voxel.moments import box_integral, face_integral  # noqa: E402
from cubic_scattering.graded_voxel.site import contrast_operator, single_site_t36  # noqa: E402

R = sp.Rational


# ---------------------------------------------------------------------------
# 1. the master integrals, exactly
# ---------------------------------------------------------------------------


@cache
def line_x(r: int, a2: int, c: int, m: int) -> sp.Expr:
    if a2 == 0:
        return sp.Integer(c) ** (r + m + 1) / (r + m + 1)
    top = sp.Integer(a2 + c * c)
    if r >= 2:
        return (sp.Integer(c) ** (r - 1) * top ** R(m + 2, 2) - (r - 1) * line_x(r - 2, a2, c, m + 2)) / (
            m + 2
        )
    if r == 1:
        return (top ** R(m + 2, 2) - sp.Integer(a2) ** R(m + 2, 2)) / (m + 2)
    if m == -1:
        return sp.asinh(c / sp.sqrt(a2))
    if m == -3:
        return c / (a2 * sp.sqrt(top))
    return (c * top ** R(m, 2) + m * a2 * line_x(0, a2, c, m - 2)) / (m + 1)


@cache
def face_x(q: int, r: int, a: int, b: int, c: int, m: int) -> sp.Expr:
    if a == 0:
        return (
            sp.Integer(b) ** (q + 1) * line_x(r, b * b, c, m)
            + sp.Integer(c) ** (r + 1) * line_x(q, c * c, b, m)
        ) / (q + r + m + 2)
    if q >= 1:
        edge = sp.Integer(b) ** (q - 1) * line_x(r, a * a + b * b, c, m + 2)
        if q == 1:
            return (edge - line_x(r, a * a, c, m + 2)) / (m + 2)
        return (edge - (q - 1) * face_x(q - 2, r, a, b, c, m + 2)) / (m + 2)
    if r >= 1:
        return face_x(r, q, a, c, b, m)
    if m == -3:
        return sp.atan(sp.Integer(b * c) / (a * sp.sqrt(a * a + b * b + c * c))) / a
    edges = b * line_x(0, a * a + b * b, c, m) + c * line_x(0, a * a + c * c, b, m)
    return (m * a * a * face_x(0, 0, a, b, c, m - 2) + edges) / (m + 2)


@cache
def box_x(p: int, q: int, r: int, a: int, b: int, c: int, m: int) -> sp.Expr:
    return (
        sp.Integer(a) ** (p + 1) * face_x(q, r, a, b, c, m)
        + sp.Integer(b) ** (q + 1) * face_x(p, r, b, a, c, m)
        + sp.Integer(c) ** (r + 1) * face_x(p, q, c, a, b, m)
    ) / (p + q + r + m + 3)


# ---------------------------------------------------------------------------
# 2. the universal moments of the self cell, exactly
# ---------------------------------------------------------------------------

SG = sp.Symbol("sigma", real=True)
TEST = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))


@cache
def autocorrelation_x(e_t: int, e_s: int) -> tuple[sp.Expr, sp.Expr]:
    """w(sigma) on [-2, 0] and [0, 2], h = 1."""
    v = sp.Symbol("v", real=True)
    f = v**e_t * (v - SG) ** e_s
    return sp.expand(sp.integrate(f, (v, -1, SG + 1))), sp.expand(sp.integrate(f, (v, SG - 1, 1)))


def axis_parts_x(e_t: int, e_s: int, n_der: int) -> list[tuple[str, int, int, sp.Expr]]:
    left, right = autocorrelation_x(e_t, e_s)
    parts = [("piece", -2, 0, sp.diff(left, SG, n_der)), ("piece", 0, 2, sp.diff(right, SG, n_der))]
    if n_der == 2:
        dl, dr = sp.diff(left, SG), sp.diff(right, SG)
        jumps = (dl.subs(SG, -2), dr.subs(SG, 0) - dl.subs(SG, 0), -dr.subs(SG, 2))
        parts += [("delta", b, b, j) for b, j in zip((-2, 0, 2), jumps, strict=True)]
    return parts


def anchored_x(poly: sp.Expr, lo: int, hi: int, alpha: int) -> dict[int, sp.Expr]:
    """{e: coefficient}: int_lo^hi poly(t) t^alpha g = sum_e coef[e] int_0^2 t^(e + alpha) g (sides 2)."""
    out: dict[int, sp.Expr] = {}
    coeffs = sp.Poly(poly, SG).all_coeffs()[::-1] if poly != 0 else []
    for e, c in enumerate(coeffs):
        total = sp.Integer(0)
        if hi == 2:
            total += c
        if lo == -2:
            total += -((-1) ** (e + alpha + 1)) * c  # -F(lo), F(-2) = (-1)^(e + alpha + 1) I(2)
        if total != 0:
            out[e] = total
    return out


@cache
def moment_x(m: int, idx: tuple[int, ...], a: int, c: int) -> sp.Expr:
    """U_ac(m, idx) of the self cell at h = 1, exactly."""
    k = _moves_closed(m, idx)
    moved, rest = idx[:k], idx[k:]
    orders = (moved.count(0), moved.count(1), moved.count(2))
    parts = [axis_parts_x(TEST[a][i], TEST[c][i], orders[i]) for i in range(3)]
    total = sp.Integer(0)
    for choice in itertools.product(*parts):
        fixed = [i for i in range(3) if choice[i][0] == "delta"]
        free = [i for i in range(3) if i not in fixed]
        for coef, alpha, power in radial_monomials(m, rest):
            cf = sp.nsimplify(coef)
            if fixed:
                f = fixed[0]
                x0 = choice[f][1]
                if x0 == 0 and alpha[f] > 0:
                    continue
                weight = choice[f][3] * sp.Integer(x0) ** alpha[f]
                if weight == 0:
                    continue
                j, j2 = free
                vj = anchored_x(choice[j][3], choice[j][1], choice[j][2], alpha[j])
                vk = anchored_x(choice[j2][3], choice[j2][1], choice[j2][2], alpha[j2])
                for (ej, cj), (ek, ck) in itertools.product(vj.items(), vk.items()):
                    total += (
                        cf * weight * cj * ck * face_x(ej + alpha[j], ek + alpha[j2], abs(x0), 2, 2, power)
                    )
            else:
                vs = [anchored_x(choice[i][3], choice[i][1], choice[i][2], alpha[i]) for i in range(3)]
                for (e0, c0), (e1, c1), (e2, c2) in itertools.product(*(v.items() for v in vs)):
                    total += (
                        cf
                        * c0
                        * c1
                        * c2
                        * box_x(e0 + alpha[0], e1 + alpha[1], e2 + alpha[2], 2, 2, 2, power)
                    )
    return sp.nsimplify((-1) ** k) * total


def main() -> int:
    ok = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        ok.append(bool(cond))
        print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}", flush=True)

    # 1
    worst = 0.0
    for args in (
        (1, 1, 0, 2, 2, 2, -1),
        (2, 0, 1, 2, 2, 2, -3),
        (3, 2, 2, 2, 2, 2, -3),
        (0, 2, 2, 2, 2, 2, 1),
    ):
        worst = max(worst, abs(float(box_x(*args).evalf(30)) / float(box_integral(*args)) - 1))
    for args in ((0, 0, 2, 2, 2, -3), (3, 2, 2, 2, 2, -3), (1, 1, 0, 2, 2, -3), (0, 0, 0, 2, 2, -1)):
        worst = max(worst, abs(float(face_x(*args).evalf(30)) / float(face_integral(*args)) - 1))
    check("1. exact master integrals equal the 40-digit ones", worst < 1e-14, f"{worst:.1e}")

    # 2 and 3: every static moment of the self cell, against the floating closed forms
    ref = ReferenceMedium(5000.0, 3000.0, 2500.0)
    terms = sorted(static_term_table(ref.alpha, ref.beta, ref.rho))
    worst = 0.0
    exact: dict[tuple[int, tuple[int, ...]], sp.Matrix] = {}
    for m, idx in terms:
        mat = sp.Matrix(4, 4, lambda a, c, m=m, idx=idx: moment_x(m, idx, a, c))
        exact[(m, idx)] = mat
        num = static_term_integral_closed(m, idx, (0, 0, 0), 1.0)[:, :4]
        got = np.array(mat.evalf(25), dtype=float)
        worst = max(worst, float(np.abs(got - num).max() / max(np.abs(num).max(), 1e-300)))
    check(
        f"2. {len(terms)} exact static moments equal the floating closed forms",
        worst < 1e-10,
        f"{worst:.1e}",
    )

    ta, tb = family_tables()
    print(f"   families: {len(ta)} terms with delta_ij r^m, {len(tb)} with d_i d_j r^m")
    out = (
        Path(__file__).resolve().parent.parent
        / "LatexPDFs"
        / "ExactCouplingIntegrals"
        / "data"
        / "site_moments_exact.txt"
    )
    lines = []
    for (m, idx), mat in exact.items():
        for a, c in itertools.product(range(4), repeat=2):
            if mat[a, c] != 0:
                lines.append(f"U[{m}; {idx}; {a},{c}] = {sp.simplify(mat[a, c])}")
    out.write_text("\n".join(lines) + "\n")
    print(f"   wrote {out} ({len(lines)} non-zero moments)")
    # 3. the static strain-strain self block K_SS[a, c] (6 x 6 each), exactly, in units of h^3 / (4 pi mu):
    #    K = TA . U(-1, idx) + b2 TB . U(1, idx), |idx| = 2 and 4
    b2 = sp.Symbol("b2", real=True)

    def rat(mat: np.ndarray) -> sp.Matrix:
        return sp.Matrix(6, 6, lambda i, j: sp.nsimplify(mat[3 + i, 3 + j], rational=True))

    kss = {}
    for a, c in itertools.product(range(4), repeat=2):
        tot = sp.zeros(6, 6)
        for idx, coef in ta.items():
            if len(idx) == 2:
                tot += rat(coef) * exact[(-1, idx)][a, c]
        for idx, coef in tb.items():
            if len(idx) == 4:
                tot += b2 * rat(coef) * exact[(1, idx)][a, c]
        kss[(a, c)] = tot.applyfunc(sp.simplify)
    full = np.zeros((4, 4, 6, 6))
    b2v = -(1.0 - ref.beta**2 / ref.alpha**2) / 2.0
    for (a, c), mat in kss.items():
        full[a, c] = np.array(mat.subs(b2, b2v).evalf(25), dtype=float)
    num = near_block((0, 0, 0), 1.0, 1e-6, ref, n_q=14)[:, :4, 3:, 3:].real * 4.0 * np.pi * ref.mu
    err = np.abs(full - num).max() / np.abs(num).max()
    check(
        "3. the exact static strain-strain block equals the quadrature block as omega -> 0",
        err < 1e-9,
        f"{err:.1e}",
    )

    # 4. the reduced blocks.  A strain-type vector is a 4 x 6 array v[a, alpha] (function a, Voigt alpha).
    #    Voigt order (zz, xx, yy, 2xy, 2zy, 2zx): shear index 3 is the pair of axes (1, 2), 4 is (0, 2),
    #    5 is (0, 1).  The vectors have integer entries and are not normalised; the block of an operator X
    #    is its matrix in them, (V^T V)^-1 V^T X V.
    def vec(entries: dict[tuple[int, int], int]) -> sp.Matrix:
        v = sp.zeros(24, 1)
        for (a, al), val in entries.items():
            v[6 * a + al] = val
        return v

    basis = {
        "A1g": [vec({(0, 0): 1, (0, 1): 1, (0, 2): 1})],
        "Eg": [vec({(0, 0): 2, (0, 1): -1, (0, 2): -1})],
        "T2g": [vec({(0, 3): 1})],
        "T1u": [vec({(1, 0): 1}), vec({(1, 1): 1, (1, 2): 1}), vec({(2, 5): 1, (3, 4): 1})],
        "T2u": [vec({(1, 1): 1, (1, 2): -1}), vec({(2, 5): 1, (3, 4): -1})],
        "A2u": [vec({(1, 3): 1, (2, 4): 1, (3, 5): 1})],
        "Eu": [vec({(2, 4): 1, (3, 5): -1})],
    }
    big = sp.zeros(24, 24)
    for (a, c), mat in kss.items():
        for ii, jj in itertools.product(range(6), repeat=2):
            big[6 * a + ii, 6 * c + jj] = mat[ii, jj]
    names = [(g, k) for g, vs in basis.items() for k in range(len(vs))]
    cols = sp.Matrix.hstack(*[basis[g][k] for g, k in names])
    gram_v = cols.T * cols
    red = (gram_v.inv() * cols.T * big * cols).applyfunc(sp.simplify)
    leak = max(
        abs(complex(red[a, c].subs(b2, b2v).evalf(20)))
        for a, (ga, _) in enumerate(names)
        for c, (gc, _) in enumerate(names)
        if ga != gc
    )
    check("4. the adapted vectors of different representations do not couple", leak < 1e-25, f"{leak:.1e}")

    # N = M^-1 K_SS in units of 1 / mu; M = 8 h^3 on the uniform functions, 8 h^3 / 3 on the linear ones
    gram = dict.fromkeys(("A1g", "Eg", "T2g"), sp.Integer(8))
    gram.update(dict.fromkeys(("T1u", "T2u", "A2u", "Eu"), sp.Rational(8, 3)))
    l1, l2 = sp.log(1 + sp.sqrt(2)), sp.asinh(sp.sqrt(2) / 2)
    consts = [sp.Integer(1), 1 / sp.pi, sp.sqrt(2) / sp.pi, sp.sqrt(3) / sp.pi, (l1 - l2) / sp.pi]
    labels = ["1", "1/pi", "sqrt2/pi", "sqrt3/pi", "L"]

    def decompose(expr: sp.Expr) -> list[sp.Rational]:
        """Rational coefficients of expr on (1, 1/pi, sqrt2/pi, sqrt3/pi, L),
        L = (ln(1 + sqrt2) - arsinh(1/sqrt2)) / pi."""
        a1, a2 = sp.symbols("a1 a2")
        e = sp.expand(sp.expand(expr).subs({l1: a1, l2: a2}) * sp.pi)
        c_l1, c_l2 = e.coeff(a1), e.coeff(a2)
        assert sp.simplify(c_l1 + c_l2) == 0, expr
        rest = sp.expand(e - c_l1 * a1 - c_l2 * a2)
        c_pi = rest.coeff(sp.pi)
        rest = sp.expand(rest - c_pi * sp.pi)
        c_2, c_3 = rest.coeff(sp.sqrt(2)), rest.coeff(sp.sqrt(3))
        c_1 = sp.expand(rest - c_2 * sp.sqrt(2) - c_3 * sp.sqrt(3))
        out = [c_pi, c_1, c_2, c_3, c_l1]
        assert all(c.is_Rational for c in out), (expr, out)
        assert sp.simplify(sum(c * k for c, k in zip(out, consts, strict=True)) - expr) == 0
        return out

    blocks_n: dict[str, sp.Matrix] = {}
    text = [
        f"constants: {labels}; N = (p + b2 q) / mu, each of p, q as rational coefficients on the constants"
    ]
    for g in basis:
        rows = [a for a, (ga, _) in enumerate(names) if ga == g]
        blk = red.extract(rows, rows) / (4 * sp.pi * gram[g])
        blocks_n[g] = blk
        text.append(f"{g}: block {len(rows)} x {len(rows)}")
        for a, c in itertools.product(range(len(rows)), repeat=2):
            e = sp.expand(blk[a, c])
            pq = (decompose(e.subs(b2, 0)), decompose(sp.diff(e, b2)))
            text.append(
                f"  [{a},{c}] p = {pq[0]}  q = {pq[1]}   "
                f"({float(e.subs(b2, 0)):+.10f} {float(sp.diff(e, b2)):+.10f} b2)"
            )
    out_tex = out.with_name("site_blocks_exact.txt")
    out_tex.write_text("\n".join(text) + "\n")
    print("\n".join(text))

    # 5. T36 from the blocks: T = M Delta (1 - N Delta)^-1, against the 36 x 36 inverse as omega -> 0.
    #    Delta on the adapted vectors, from Delta = dlam J + 2 dmu I on the six strain components.
    dlam, dmu, hh = 2.0e9, 1.0e9, 1.25
    delta_blocks = {
        "A1g": sp.Matrix([[3 * dlam + 2 * dmu]]),
        "Eg": sp.Matrix([[2 * dmu]]),
        "T2g": sp.Matrix([[2 * dmu]]),
        "T1u": sp.Matrix([[dlam + 2 * dmu, 2 * dlam, 0], [dlam, 2 * dlam + 2 * dmu, 0], [0, 0, 2 * dmu]]),
        "T2u": sp.Matrix([[2 * dmu, 0], [0, 2 * dmu]]),
        "A2u": sp.Matrix([[2 * dmu]]),
        "Eu": sp.Matrix([[2 * dmu]]),
    }
    om = 1e-6
    k0 = near_block((0, 0, 0), hh, om, ref, n_q=14)
    dl = np.zeros((4, 9, 9), dtype=complex)
    dl[0] = contrast_operator(dlam, dmu, 0.0, om)
    t36 = single_site_t36(hh, dl, k0)
    worst = 0.0
    for g, vs in basis.items():
        n = len(vs)
        nb = np.array((blocks_n[g].subs(b2, b2v) / ref.mu).evalf(25), dtype=float)
        db = np.array(delta_blocks[g], dtype=float)
        mg = float(gram[g]) * hh**3
        t_blk = mg * db @ np.linalg.inv(np.eye(n) - nb @ db)
        v36 = np.zeros((36, n))
        for k, v in enumerate(vs):
            for a in range(4):
                v36[9 * a + 3 : 9 * a + 9, k] = np.array(v[6 * a : 6 * a + 6], dtype=float).ravel()
        got = np.linalg.solve(v36.T @ v36, v36.T @ t36.real @ v36)
        worst = max(worst, float(np.abs(got - t_blk).max() / np.abs(t_blk).max()))
    check(
        "5. T36 from the closed-form blocks equals the 36 x 36 single site as omega -> 0",
        worst < 1e-8,
        f"{worst:.1e}",
    )
    print(f"   wrote {out_tex}")
    print(f"{sum(ok)}/{len(ok)} checks passed")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
