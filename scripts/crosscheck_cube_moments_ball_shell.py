#!/usr/bin/env python3
"""Independent check of the scalar cube moments E[m; d1..dD; w1..wW] by a different route.

The moment engine (``Mathematica/CubeMomentCore.wl``) evaluates

    E[m; ds; w] = < d_{d1} .. d_{dD} r^m ,  x_{w1} .. x_{wW} 1_V >,     V = [-1/2, 1/2]^3,

as a distribution, by moving every derivative onto the FACES of the cube. This script evaluates the same
number without touching the faces:

    cube = inscribed ball + (cube minus ball).

* On the ball of radius h = 1/2 the derivatives are moved onto the SPHERE, where the surface integrals
  of monomials are closed-form Gamma functions. All the distributional content (the delta function of
  the inclusion problem and its derivatives) is in this part.
* On the cube minus the ball the integrand is an ordinary smooth function. In spherical coordinates the
  radial integral is elementary, and the angular integral is taken over the six faces by Gauss-Legendre
  (the map from a face to the directions is smooth on the face).

The two routes share nothing but the definition, so agreement is evidence for both.

Self-checks, with answers known independently of either route:
  [1] the delta rule  Sum_p E[-1; {p,p}+rest; w] = -4 pi (-1)^|rest| (d_rest w)(0);
  [2] the chain rule  Sum_p E[m; {p,p}+rest; w] = m (m+1) E[m-2; rest; w];
  [3] D = 0 against direct numerical integration.
Cross-check:
  [4] every value in ``Mathematica/cube_higher_moments.json`` (exported from the engine's stored values),
      when that file exists.

Run:  python scripts/crosscheck_cube_moments_ball_shell.py
"""

import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

ROOT = Path(__file__).resolve().parent.parent
H = 0.5
GX, GW = leggauss(48)

# A function is a dict {(a, b, c, p): coefficient} meaning sum coeff x^a y^b z^c r^p.
Poly = dict[tuple[int, int, int, int], float]


def deriv(f: Poly, axis: int) -> Poly:
    """d/dx_axis of sum c x^a y^b z^c r^p (pointwise, away from the origin)."""
    out: Poly = defaultdict(float)
    for (a, b, c, p), coef in f.items():
        e = [a, b, c]
        if e[axis] > 0:
            g = list(e)
            g[axis] -= 1
            out[(g[0], g[1], g[2], p)] += coef * e[axis]
        if p != 0:
            g = list(e)
            g[axis] += 1
            out[(g[0], g[1], g[2], p - 2)] += coef * p
    return {k: v for k, v in out.items() if v != 0.0}


def times_monomial(f: Poly, axes: list[int]) -> Poly:
    """f times x_{axes[0]} x_{axes[1]} ..."""
    out: Poly = {}
    for (a, b, c, p), coef in f.items():
        e = [a, b, c]
        for ax in axes:
            e[ax] += 1
        out[(e[0], e[1], e[2], p)] = out.get((e[0], e[1], e[2], p), 0.0) + coef
    return out


def sphere_monomial(a: int, b: int, c: int) -> float:
    """Integral of n_x^a n_y^b n_z^c over the unit sphere."""
    if a % 2 or b % 2 or c % 2:
        return 0.0
    return (
        2.0
        * math.gamma((a + 1) / 2)
        * math.gamma((b + 1) / 2)
        * math.gamma((c + 1) / 2)
        / math.gamma((a + b + c + 3) / 2)
    )


def ball_volume(f: Poly) -> float:
    """Integral of f over the ball of radius H (f integrable at the origin)."""
    tot = 0.0
    for (a, b, c, p), coef in f.items():
        s = a + b + c + p + 2
        ang = sphere_monomial(a, b, c)
        if ang == 0.0:
            continue
        if s + 1 <= 0:
            raise ValueError(f"non-integrable term {(a, b, c, p)} reached the ball's volume integral")
        tot += coef * ang * H ** (s + 1) / (s + 1)
    return tot


def sphere_surface(f: Poly, normal_axis: int) -> float:
    """Integral over the sphere of radius H of n_axis f dA."""
    tot = 0.0
    for (a, b, c, p), coef in f.items():
        e = [a, b, c]
        e[normal_axis] += 1
        ang = sphere_monomial(e[0], e[1], e[2])
        tot += coef * ang * H ** (a + b + c + p + 2)
    return tot


def ball_distribution(m: int, ds: list[int], w: list[int]) -> float:
    """< d_ds r^m, w 1_ball >, the derivatives peeled onto the sphere."""
    if not ds:
        return ball_volume(times_monomial({(0, 0, 0, m): 1.0}, w))
    q, rest = ds[0], ds[1:]
    kern: Poly = {(0, 0, 0, m): 1.0}
    for ax in rest:
        kern = deriv(kern, ax)
    tot = sphere_surface(times_monomial(kern, w), q)
    for u, wu in enumerate(w):
        if wu == q:
            tot -= ball_distribution(m, rest, w[:u] + w[u + 1 :])
    return tot


def shell_classical(m: int, ds: list[int], w: list[int]) -> float:
    """Integral over the cube minus the inscribed ball of w d_ds r^m, an ordinary function there."""
    f: Poly = {(0, 0, 0, m): 1.0}
    for ax in ds:
        f = deriv(f, ax)
    f = times_monomial(f, w)
    # face z = H (and its five images): direction through the face point (u, v, H)
    u = H * GX
    wu = H * GW
    uu, vv = np.meshgrid(u, u, indexing="ij")
    ww = np.outer(wu, wu)
    big_r = np.sqrt(uu**2 + vv**2 + H**2)
    d_omega = ww * H / big_r**3
    tot = 0.0
    for face_axis in range(3):
        for sign in (1.0, -1.0):
            n = [None, None, None]
            others = [ax for ax in range(3) if ax != face_axis]
            n[face_axis] = sign * H / big_r
            n[others[0]] = uu / big_r
            n[others[1]] = vv / big_r
            for (a, b, c, p), coef in f.items():
                s = a + b + c + p + 2
                if s + 1 == 0:
                    radial = np.log(big_r / H)
                else:
                    radial = (big_r ** (s + 1) - H ** (s + 1)) / (s + 1)
                tot += coef * float(np.sum(d_omega * n[0] ** a * n[1] ** b * n[2] ** c * radial))
    return tot


def moment(m: int, ds: list[int], w: list[int]) -> float:
    """E[m; ds; w] over the cube [-1/2, 1/2]^3; axes are 0, 1, 2."""
    return ball_distribution(m, list(ds), list(w)) + shell_classical(m, list(ds), list(w))


def deriv_at_zero(rest: list[int], w: list[int]) -> float:
    """(d_rest w)(0) for the monomial w."""
    cw = [w.count(ax) for ax in range(3)]
    cr = [rest.count(ax) for ax in range(3)]
    if cw != cr:
        return 0.0
    return float(math.prod(math.factorial(k) for k in cw))


def main() -> int:
    ok = True
    axes = (0, 1, 2)

    worst_delta = worst_chain = 0.0
    count = 0
    for d_rest, n_w in ((0, 0), (1, 1), (2, 0), (2, 2), (1, 3), (0, 2)):
        for rest in itertools.combinations_with_replacement(axes, d_rest):
            for w in itertools.combinations_with_replacement(axes, n_w):
                rest_l, w_l = list(rest), list(w)
                lhs = sum(moment(-1, [p, p, *rest_l], w_l) for p in axes)
                rhs = -4 * math.pi * (-1) ** len(rest_l) * deriv_at_zero(rest_l, w_l)
                worst_delta = max(worst_delta, abs(lhs - rhs))
                lhs = sum(moment(1, [p, p, *rest_l], w_l) for p in axes)
                rhs = 2 * moment(-1, rest_l, w_l)
                worst_chain = max(worst_chain, abs(lhs - rhs))
                count += 1
    good = worst_delta < 1e-10
    ok = ok and good
    verdict = "PASS" if good else "FAIL"
    print(f"[1] delta rule, {count} traces up to grade (4,2): worst {worst_delta:.1e}   {verdict}")
    good = worst_chain < 1e-10
    ok = ok and good
    print(f"[2] chain rule, {count} traces: worst {worst_chain:.1e}   {'PASS' if good else 'FAIL'}")

    # [3] D = 0: 1/r and x^2/r over the cube, against the classical closed forms' numerical value
    x = H * GX
    wq = H * GW
    xx, yy, zz = np.meshgrid(x, x, x, indexing="ij")
    w3 = np.einsum("i,j,k->ijk", wq, wq, wq)
    direct = float(np.sum(w3 * xx**2 * yy**2 * np.sqrt(xx**2 + yy**2 + zz**2)))  # smooth: x^2 y^2 r
    got = moment(1, [], [0, 0, 1, 1])
    good = abs(got - direct) < 1e-6 * abs(direct)
    ok = ok and good
    verdict = "PASS" if good else "FAIL"
    print(f"[3] D = 0, x^2 y^2 r over the cube: {got:.10e} vs quadrature {direct:.10e}   {verdict}")

    ref = ROOT / "Mathematica" / "cube_higher_moments.json"
    if ref.exists():
        data = json.loads(ref.read_text())
        worst, n_cmp, worst_key = 0.0, 0, ""
        for row in data["moments"]:
            ds = [d - 1 for d in row["d"]]
            w = [v - 1 for v in row["w"]]
            got = moment(row["m"], ds, w)
            err = abs(got - row["value"]) / max(1.0, abs(row["value"]))
            n_cmp += 1
            if err > worst:
                worst, worst_key = err, f"m={row['m']} d={row['d']} w={row['w']}"
        good = worst < 1e-9
        ok = ok and good
        print(
            f"[4] against the engine's {n_cmp} stored values: worst {worst:.1e} ({worst_key})   "
            f"{'PASS' if good else 'FAIL'}"
        )
    else:
        print(f"[4] {ref.name} not found: export the engine's values first (CubeMomentExportScalars.wl)")
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
