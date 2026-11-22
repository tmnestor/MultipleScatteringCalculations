#!/usr/bin/env python3
"""What a cell loses of the product (profile x plane wave), as it acts on a smooth field: the symbolic
step of the factor F of the second error term at a finite frequency.

THE QUESTION. A cell of half-width h holds the product of the profile f and a plane wave w = exp(i k.x)
only approximately. Call the part it loses g. The error of the second-order term T2 needs <g, phi> for a
smooth field phi (the field scattered once, or its adjoint). Summed over the cells this is an integral
over the body of a density, and to leading order in h the density is a bilinear form in the derivatives
of f and of phi at a point, with coefficients polynomial in k. This script derives that form.

THE THREE CASES, by what the scheme does:

  constant        g = f w - mean(f) mean(w)                 both sides of constant cells (p = r = 0)
  degree-one in   g = f w - (P1 f)(P1 w)                    the source of degree-one cells: the product of
                                                            the two linear forms, kept whole
  degree-one out  g = f w - P1[ w (P1 f) ]                  the readout of degree-one cells: the product
                                                            is projected

with P1 the L2 projection onto 1, x, y, z on the cell. The density is (1 / cell volume) int_cell g phi,
expanded to h^2 (constant) or h^4 (degree one), and returned as the coefficients of phi, d_i phi and
d_i d_j phi: three arrays L0, L1[i], L2[i][j], each a polynomial in the derivatives of f and in k, such
that

    density = w(centre) h^n ( L0 phi + sum_i L1_i d_i phi + sum_ij L2_ij d_i d_j phi ).

``residual_operators(case)`` returns numerical functions of these; running the script prints them and
checks the constant case against the form derived by hand,
(h^2 / 3) (grad f . grad phi + i (k . grad f) phi + i f k . grad phi ... ) rearranged, and both degree-one
cases against the static second moments h^4 (f_ii phi_ii / 45 + f_ij phi_ij / 9).

Run:  conda run -n seismic python -u scripts/derive_cell_residual_moments.py
"""

import itertools
import sys
from functools import lru_cache

import sympy as sp

X = sp.symbols("x1:4", real=True)
KV = sp.symbols("k1:4", real=True)
H, T = sp.symbols("h t", positive=True)
MAX_DEG = 4


def _taylor(name: str, max_deg: int) -> tuple[sp.Expr, dict[tuple[int, ...], sp.Symbol]]:
    """A generic Taylor polynomial about the cell centre and its derivative symbols, keyed by the sorted
    tuple of the axes differentiated (() the value, (0,) d/dx, (0, 1) d2/dxdy, ...)."""
    syms: dict[tuple[int, ...], sp.Symbol] = {}
    poly = sp.Integer(0)
    for deg in range(max_deg + 1):
        for idx in itertools.combinations_with_replacement(range(3), deg):
            s = sp.Symbol(f"{name}_" + ("".join(str(i + 1) for i in idx) or "0"))
            syms[idx] = s
            mult = sp.factorial(deg) / sp.prod([sp.factorial(idx.count(a)) for a in range(3)])
            poly += s * mult * sp.prod([X[i] for i in idx]) / sp.factorial(deg)
    return poly, syms


def _cube_mean(expr: sp.Expr) -> sp.Expr:
    """The mean of a polynomial over the cube [-h, h]^3."""
    out = sp.expand(expr)
    for x in X:
        out = sp.integrate(out, (x, -H, H)) / (2 * H)
    return sp.expand(out)


def _truncate(expr: sp.Expr, order: int) -> sp.Expr:
    """Drop every term whose total degree in (x, h) exceeds ``order``."""
    scaled = sp.expand(expr.subs({**{x: T * x for x in X}, H: T * H}, simultaneous=True))
    return sp.expand(sum(scaled.coeff(T, n) for n in range(order + 1)).subs(T, 1))


def _project1(expr: sp.Expr) -> sp.Expr:
    """The L2 projection onto 1, x, y, z over the cube."""
    return _cube_mean(expr) + sum(_cube_mean(expr * x) / (H**2 / 3) * x for x in X)


@lru_cache(maxsize=3)
def residual_density(case: str) -> tuple[sp.Expr, dict, dict, int]:
    """(density / w(centre), the symbols of f, the symbols of phi, the power of h) for one case."""
    f, fs = _taylor("f", MAX_DEG)
    phi, ps = _taylor("p", MAX_DEG)
    kx = sum(k * x for k, x in zip(KV, X, strict=True))
    w = sum((sp.I * kx) ** n / sp.factorial(n) for n in range(MAX_DEG + 1))
    if case == "constant":
        power, g = 2, f * w - _cube_mean(f) * _cube_mean(w)
    elif case == "degree-one in":
        power, g = 4, f * w - _project1(f) * _project1(w)
    elif case == "degree-one out":
        power, g = 4, f * w - _project1(_truncate(w * _project1(f), MAX_DEG))
    else:
        raise ValueError(
            f"residual_density: case must be 'constant', 'degree-one in' or 'degree-one out', got {case!r}"
        )
    density = _truncate(_cube_mean(_truncate(sp.expand(g * phi), MAX_DEG + 2)), power)
    lower = [density.coeff(H, n) for n in range(power)]
    if any(sp.simplify(c) != 0 for c in lower):
        raise AssertionError(f"{case}: the density has a term below h^{power}: {lower}")
    return sp.expand(density.coeff(H, power)), fs, ps, power


def residual_operators(case: str):
    """Numerical L0, L1, L2 of the docstring: ``ops(fd, k)`` with ``fd`` a dict from the sorted axis tuple
    to an array of that derivative of f (orders 0 to 3) and ``k`` the wave vector, returning
    (L0, [L1_i], [[L2_ij]]) as arrays; and the power of h."""
    density, fs, ps, power = residual_density(case)
    used = sorted((s for s in density.free_symbols if str(s).startswith("f_")), key=str)
    by_symbol = {v: key for key, v in fs.items()}
    args = [*used, *KV]

    def coefficient(idx: tuple[int, ...]) -> sp.Expr:
        return density.coeff(ps[idx])

    higher = [idx for idx in ps if len(idx) > 2 and coefficient(idx) != 0]
    if higher:
        raise AssertionError(f"{case}: the density involves derivatives of phi above the second: {higher}")
    l0 = sp.lambdify(args, coefficient(()), "numpy")
    l1 = [sp.lambdify(args, coefficient((i,)), "numpy") for i in range(3)]
    # the coefficient of the symbol p_ij (i <= j) multiplies d_i d_j phi once: it carries both orders
    l2 = {(i, j): sp.lambdify(args, coefficient((i, j)), "numpy") for i in range(3) for j in range(i, 3)}

    def ops(fd: dict, k):
        vals = [fd[by_symbol[s]] for s in used] + [complex(v) for v in k]
        return l0(*vals), [fn(*vals) for fn in l1], {key: fn(*vals) for key, fn in l2.items()}

    return ops, power


def main() -> int:
    ok = True
    for case in ("constant", "degree-one in", "degree-one out"):
        density, fs, ps, power = residual_density(case)
        print(f"\n{case}: density / (w h^{power}) =")
        for idx in sorted(ps, key=lambda t: (len(t), t)):
            c = sp.factor(density.coeff(ps[idx]))
            if c != 0:
                print(f"   {ps[idx]}:  {c}")
        static = density.subs({k: 0 for k in KV})
        if case == "constant":
            want = sum(fs[(i,)] * ps[(i,)] for i in range(3)) / 3
        else:
            want = (
                sum(fs[(i, i)] * ps[(i, i)] for i in range(3)) / 45
                + sum(fs[(i, j)] * ps[(i, j)] for i in range(3) for j in range(i + 1, 3)) / 9
            )
        good = sp.simplify(static - want) == 0
        ok &= good
        print(f"   static limit equals the hand-derived second moments: {'yes' if good else 'NO'}")
    # the constant case against the form used in measure_t2_factor_dynamic.py, after integration by parts:
    # density = (h^2 / 3) [ grad f . grad phi + i (k . grad f) phi + i f k . grad phi + ... ] must equal
    # (h^2 / 3) [ (grad f + i k f) . grad phi + i k . grad f phi ]
    density, fs, ps, _ = residual_density("constant")
    hand = (
        sum(
            (fs[(i,)] + sp.I * KV[i] * fs[()]) * ps[(i,)] + sp.I * KV[i] * fs[(i,)] * ps[()]
            for i in range(3)
        )
        / 3
    )
    good = sp.simplify(density - hand) == 0
    ok &= good
    print(
        f"\nconstant case equals (1/3) [(grad f + i k f) . grad phi + i (k . grad f) phi]: {'yes' if good else 'NO'}"
    )
    ok &= check_paper_forms()
    return 0 if ok else 1


def check_paper_forms() -> bool:
    """The two degree-one densities in the compact form printed in the Error paper.

    With <A, B> = sum_i A_ii B_ii / 45 + sum_{i != j} A_ij B_ij / 18, H the second derivatives of f, and
    k a for the symmetrised pair k a + a k:

        out:  < H + i (k grad f + grad f k) - k k f , grad grad phi >
              + < H , i (k grad phi + grad phi k) - k k phi >
        in:   < H , grad grad phi + i (k grad phi + grad phi k) - k k phi >
              - < k k , f grad grad phi + grad f grad phi + grad phi grad f >
    """
    _, fs, ps, _ = residual_density("degree-one out")

    def sym2(syms: dict) -> sp.Matrix:
        return sp.Matrix(3, 3, lambda i, j: syms[tuple(sorted((i, j)))])

    def pair(a: sp.Matrix, b: sp.Matrix) -> sp.Matrix:
        return a * b.T + b * a.T

    def form(a: sp.Matrix, b: sp.Matrix) -> sp.Expr:
        return (
            sum(a[i, i] * b[i, i] for i in range(3)) / 45
            + sum(a[i, j] * b[i, j] for i in range(3) for j in range(3) if i != j) / 18
        )

    k = sp.Matrix(KV)
    grad_f = sp.Matrix([fs[(i,)] for i in range(3)])
    grad_p = sp.Matrix([ps[(i,)] for i in range(3)])
    hess_f, hess_p = sym2(fs), sym2(ps)
    f0, p0 = fs[()], ps[()]
    kk = k * k.T
    out = form(hess_f + sp.I * pair(k, grad_f) - kk * f0, hess_p) + form(
        hess_f, sp.I * pair(k, grad_p) - kk * p0
    )
    inn = form(hess_f, hess_p + sp.I * pair(k, grad_p) - kk * p0) - form(
        kk, f0 * hess_p + pair(grad_f, grad_p)
    )
    good = True
    for case, expr in (("degree-one out", out), ("degree-one in", inn)):
        same = sp.simplify(residual_density(case)[0] - sp.expand(expr)) == 0
        good &= same
        print(f"{case}: equals the compact form printed in the paper: {'yes' if same else 'NO'}")
    return good


if __name__ == "__main__":
    sys.exit(main())
