#!/usr/bin/env python3
"""The gradient hierarchy on a GRADED layer in one dimension: does the order survive a varying contrast?

``measure_layer_taylor_hierarchy.py`` established the orders of the hierarchy (2, 2, 4 for q = 1, 2, 3 with
both contrasts) on a layer of UNIFORM contrast. In three dimensions on the graded sphere the same scheme,
with the contrast expanded in each voxel, did not show them. This script puts the same question in one
dimension, where everything is elementary, to separate the formulation from the three-dimensional code.

The layer 0 < z < D has the contrast  s(z) (drho, dM)  with a smooth profile s. In each cell the profile is
replaced by its L2 projection on polynomials of degree r_c, the field by its Taylor polynomial of degree
q about the centre, the strain by the derivative of that, and the sources are the products. The equation
and its first q derivatives are imposed at each centre, the delta terms kept:

    u = u0 + A + B',   A = int g (a s u),  B = int g (b s u'),
    A'' = -(a s u)/M - k^2 A,   B'' = -(b s u')/M - k^2 B   inside a cell.

Exact answer: the ordinary differential equation (M u')' + rho omega^2 u = 0 integrated to 1e-12.

Profiles:
  "sin2"   s = sin^2(pi z / D): analytic inside, zero with zero slope at both faces;
  "smooth" the smoothstep bump used radially in the graded sphere (full in the middle half, C^2 joins).

Run:  python scripts/measure_layer_taylor_hierarchy_graded.py
"""

import math
import sys

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import solve_ivp

ALPHA, RHO = 5000.0, 2500.0
M_P = RHO * ALPHA**2
D_LAYER = 2.0
D_RHO, D_M = 100.0, 4.0e9
GX, GW = leggauss(24)


def profile(name: str, z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    if name == "sin2":
        return np.where((z > 0) & (z < D_LAYER), np.sin(np.pi * z / D_LAYER) ** 2, 0.0)
    # smoothstep bump: 1 for |z - D/2| < D/4, falling to 0 at the faces
    x = np.clip((D_LAYER / 2 - np.abs(z - D_LAYER / 2)) / (D_LAYER / 4), 0.0, 1.0)
    return 10 * x**3 - 15 * x**4 + 6 * x**5


def exact_scattered(name, omega, drho, dm, z_r, z_t):
    k0 = omega / ALPHA

    def rhs(z, y):
        s = float(profile(name, np.array([z]))[0])
        m, rho = M_P + dm * s, RHO + drho * s
        return [y[1] / m, -rho * omega**2 * y[0]]

    sol = solve_ivp(rhs, [D_LAYER, 0.0], [1.0 + 0j, 1j * k0 * M_P], method="DOP853", rtol=1e-12, atol=1e-14)
    u0, t0 = sol.y[0, -1], sol.y[1, -1]
    # u(0) = c (1 + R), M u'(0) = c i k M (1 - R); the integration assumed T = 1
    inc = 0.5 * (u0 + t0 / (1j * k0 * M_P))
    refl = 0.5 * (u0 - t0 / (1j * k0 * M_P))
    r, t = refl / inc, 1.0 / inc
    return r * np.exp(-1j * k0 * z_r), t * np.exp(1j * k0 * (z_t - D_LAYER)) - np.exp(1j * k0 * z_t)


def cell_integrals(k, h, delta, deg):
    """int g(delta - xi) xi^m dxi and the same with dg/dz, m = 0..deg (plain monomials)."""
    c = 1j / (2.0 * M_P * k)
    edges = [-h, delta, h] if -h < delta < h else [-h, h]
    gint = np.zeros(deg + 1, dtype=complex)
    dint = np.zeros(deg + 1, dtype=complex)
    for lo, hi in zip(edges, edges[1:], strict=False):
        xi = 0.5 * (hi - lo) * GX + 0.5 * (hi + lo)
        w = 0.5 * (hi - lo) * GW
        sep = delta - xi
        g = c * np.exp(1j * k * np.abs(sep))
        dg = 1j * k * np.sign(sep) * g
        for m in range(deg + 1):
            gint[m] += np.sum(w * g * xi**m)
            dint[m] += np.sum(w * dg * xi**m)
    return gint, dint


def project(name, zc, h, r_c):
    """Monomial coefficients of the L2 projection of the profile on degree <= r_c in the cell."""
    xi = h * GX
    phi = np.stack([xi**m for m in range(r_c + 1)], axis=1)
    gram = phi.T @ (GW[:, None] * phi)
    return np.linalg.solve(gram, phi.T @ (GW * profile(name, zc + xi)))


def solve(name, omega, n, q, drho, dm, z_r, z_t, r_c=None):
    r_c = q if r_c is None else r_c
    k = omega / ALPHA
    a, b = omega**2 * drho, dm
    h = D_LAYER / n / 2.0
    zc = (np.arange(n) + 0.5) * 2.0 * h
    nu = q + 1
    deg = q + r_c
    prof = [project(name, z, h, r_c) for z in zc]

    # source polynomials as linear maps of the unknowns U^(m) (derivatives at the centre):
    #   u  = sum_m U^(m) xi^m / m!        u' = sum_m U^(m+1) xi^m / m!
    def source_maps(pc):
        f = np.zeros((deg + 1, nu))  # coefficients of xi^p in s u
        t = np.zeros((deg + 1, nu))  # coefficients of xi^p in s u'
        for j, sj in enumerate(pc):
            for m in range(nu):
                f[j + m, m] += sj / math.factorial(m)
                if m >= 1:
                    t[j + m - 1, m] += sj / math.factorial(m - 1)
        return f, t

    maps = [source_maps(pc) for pc in prof]
    big = np.eye(n * nu, dtype=complex)
    rhs = np.zeros(n * nu, dtype=complex)
    for i in range(n):
        for j in range(n):
            gint, dint = cell_integrals(k, h, zc[i] - zc[j], deg)
            f, t = maps[j]
            a_der = [a * (gint @ f), a * (dint @ f)]
            b_der = [b * (gint @ t), b * (dint @ t)]
            own = 1.0 if i == j else 0.0
            for m in range(2, q + 2):
                # (s u)^(m-2)(0) = (m-2)! times the coefficient of xi^(m-2)
                loc_a = own * a * math.factorial(m - 2) * f[m - 2]
                loc_b = own * b * math.factorial(m - 2) * t[m - 2]
                a_der.append(-loc_a / M_P - k**2 * a_der[m - 2])
                b_der.append(-loc_b / M_P - k**2 * b_der[m - 2])
            for m in range(nu):
                big[i * nu + m, j * nu : (j + 1) * nu] -= a_der[m] + b_der[m + 1]
        for m in range(nu):
            rhs[i * nu + m] = (1j * k) ** m * np.exp(1j * k * zc[i])
    sol = np.linalg.solve(big, rhs).reshape(n, nu)
    out = []
    for z_o in (z_r, z_t):
        tot = 0.0j
        for j in range(n):
            gint, dint = cell_integrals(k, h, z_o - zc[j], deg)
            f, t = maps[j]
            tot += a * (gint @ f) @ sol[j] + b * (dint @ t) @ sol[j]
        out.append(tot)
    return out[0], out[1]


def main() -> int:
    omega, z_r, z_t = 300.0, -1.0, 4.0
    ns = [4, 8, 16, 32]
    want = {1: 2, 2: 2, 3: 4}
    ok = True
    for name in ("sin2", "smooth"):
        ex = exact_scattered(name, omega, D_RHO, D_M, z_r, z_t)
        print(f"\nprofile {name}, both contrasts: relative error R / T, omega = {omega:.0f}")
        for q in (1, 2, 3):
            errs = np.array(
                [
                    [
                        abs(g - e) / abs(e)
                        for g, e in zip(solve(name, omega, n, q, D_RHO, D_M, z_r, z_t), ex, strict=True)
                    ]
                    for n in ns
                ]
            )
            orders = np.log2(errs[:-1] / errs[1:])
            good = bool(np.all(np.abs(orders[-2] - want[q]) < 0.6))
            ok = ok and good
            print(
                f"  q = {q}:  "
                + "   ".join(f"n={n:2d} {e[0]:.2e}/{e[1]:.2e}" for n, e in zip(ns, errs, strict=True))
            )
            print(
                "          orders R "
                + " ".join(f"{o:.2f}" for o in orders[:, 0])
                + "   T "
                + " ".join(f"{o:.2f}" for o in orders[:, 1])
                + f"   predicted {want[q]}   {'PASS' if good else '****FAIL****'}"
            )
    print("\nALL PASS" if ok else "\nSOME PREDICTIONS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
