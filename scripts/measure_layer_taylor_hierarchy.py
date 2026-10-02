#!/usr/bin/env python3
"""The single-site hierarchy in gradients about the centre, tested as a scheme on the layer.

The question. The closed hierarchy of the single site expands the internal displacement of a voxel in a
Taylor series about its centre, u, du, d2u, ..., and closes the Lippmann-Schwinger equation by evaluating
it and its derivatives AT THE CENTRE. Truncated at the first gradient it is the uniform-strain closure
(the collocation voxel). Does keeping the second gradients, which is a more complete representation,
improve the result, and by how much?

The test bed is the layer of the paper at normal incidence, where the exact answer is elementary and the
moments of the hierarchy are one-dimensional integrals: a layer 0 < z < D with density contrast drho and
modulus contrast dM, a unit plane wave exp(i k z) incident from above, cut into n cells.

The scheme of degree q. In cell j the displacement is its Taylor polynomial about the centre z_j,
    u(z_j + xi) = sum_{m <= q} U_j^(m) xi^m / m!,
and the strain is its derivative (degree q - 1). The equation
    u(z) = u0(z) + a A(z) + b dB/dz,   A = int g(z - z') u(z') dz',  B = int g(z - z') u'(z') dz',
with a = omega^2 drho, b = dM, g = i exp(i k |z - z'|) / (2 M k), is imposed on the m-th derivative at each
centre for m = 0..q. Derivatives of the integrals above the first are reduced by M g'' + rho omega^2 g =
-delta, that is  A'' = -u/M - k^2 A  and  B'' = -u'/M - k^2 B  inside a cell: the delta function is kept,
as in the distributional moments of the paper. The cell integrals of g and dg/dz against the monomials
are the hierarchy's moments; here they are done by Gauss-Legendre on each half of the source cell (the
integrands are entire on each half).

q = 1 is the collocation voxel (uniform strain). q = 2 is the hierarchy with the second gradients kept.
q = 3 adds the third gradients.

Predicted, before running. The source that a cell radiates is the integral of its polynomial against the
kernel. A Taylor polynomial of degree d leaves a remainder that starts at xi^(d+1). If d + 1 is odd the
remainder integrates to zero against the even part of the kernel and the error is two orders smaller; if
d + 1 is even it does not. So:
    density only (source u, degree q):         q = 1 -> order 2,  q = 2 -> order 4,  q = 3 -> order 4;
    modulus only (source u', degree q - 1):    q = 1 -> order 2,  q = 2 -> order 2,  q = 3 -> order 4.
The second gradients should therefore raise the order for a density contrast and change only the constant
for a modulus contrast; the modulus contrast needs the THIRD gradients. This is the parity statement of
the hierarchy (even gradients couple to even, odd to odd) seen as an order of convergence.

Run:  python scripts/measure_layer_taylor_hierarchy.py
"""

import math
import sys

import numpy as np
from numpy.polynomial.legendre import leggauss

ALPHA, RHO = 5000.0, 2500.0
M_P = RHO * ALPHA**2
D_LAYER = 2.0
D_RHO, D_M = 100.0, 4.0e9  # the paper's contrast: dM = dlambda + 2 dmu
GX, GW = leggauss(24)


def exact_scattered(
    omega: float, drho: float, dm: float, z_r: float, z_t: float
) -> tuple[complex, complex]:
    """Reflected field at z_r < 0 and the change of the transmitted field at z_t > D, unit incident wave."""
    k0 = omega / ALPHA
    m1, rho1 = M_P + dm, RHO + drho
    k1 = omega * math.sqrt(rho1 / m1)
    z0, z1 = M_P * k0, m1 * k1
    # unknowns: R, A (down in layer), B (up in layer), T; continuity of u and M u' at 0 and D
    e1 = np.exp(1j * k1 * D_LAYER)
    mat = np.array(
        [
            [-1.0, 1.0, 1.0, 0.0],
            [z0, z1, -z1, 0.0],
            [0.0, e1, 1.0 / e1, -1.0],
            [0.0, z1 * e1, -z1 / e1, -z0],
        ],
        dtype=complex,
    )
    r, _, _, t = np.linalg.solve(mat, np.array([1.0, z0, 0.0, 0.0], dtype=complex))
    return r * np.exp(-1j * k0 * z_r), t * np.exp(1j * k0 * (z_t - D_LAYER)) - np.exp(1j * k0 * z_t)


def cell_integrals(k: float, h: float, delta: float, q: int) -> tuple[np.ndarray, np.ndarray]:
    """int g(delta - xi) xi^m / m! dxi and the same with dg/dz, over a cell of half-width h, m = 0..q.

    delta is the field point minus the source cell's centre; the cell is split at the field point when it
    lies inside, where |z - z'| has its kink.
    """
    c = 1j / (2.0 * M_P * k)
    edges = [-h, h]
    if -h < delta < h:
        edges = [-h, delta, h]
    gint = np.zeros(q + 1, dtype=complex)
    dint = np.zeros(q + 1, dtype=complex)
    for lo, hi in zip(edges, edges[1:], strict=False):
        xi = 0.5 * (hi - lo) * GX + 0.5 * (hi + lo)
        w = 0.5 * (hi - lo) * GW
        sep = delta - xi
        g = c * np.exp(1j * k * np.abs(sep))
        dg = 1j * k * np.sign(sep) * g
        for m in range(q + 1):
            mono = xi**m / math.factorial(m)
            gint[m] += np.sum(w * g * mono)
            dint[m] += np.sum(w * dg * mono)
    return gint, dint


def solve(
    omega: float, n: int, q: int, drho: float, dm: float, z_r: float, z_t: float
) -> tuple[complex, complex]:
    """The hierarchy of degree q on n cells: scattered field at the two observers."""
    k = omega / ALPHA
    a, b = omega**2 * drho, dm
    d = D_LAYER / n
    h = d / 2.0
    zc = (np.arange(n) + 0.5) * d
    nu = q + 1
    big = np.eye(n * nu, dtype=complex)
    rhs = np.zeros(n * nu, dtype=complex)
    for i in range(n):
        for j in range(n):
            gint, dint = cell_integrals(k, h, zc[i] - zc[j], q)
            # A^(0), A^(1) from the displacement polynomial (coefficients U^(m), m = 0..q)
            a0, a1 = gint, dint
            # B^(0), B^(1) from the strain polynomial: coefficient of xi^m/m! is U^(m+1)
            b0 = np.concatenate([[0.0], gint[:q]])
            b1 = np.concatenate([[0.0], dint[:q]])
            # local terms of the reduction, present in the cell's own equations only
            own = 1.0 if i == j else 0.0
            # derivative order m of u = u0 + a A + b B'; A^(m), B^(m) as rows over the unknowns U^(0..q)
            a_der = [a0.astype(complex), a1.astype(complex)]
            b_der = [b0.astype(complex), b1.astype(complex)]
            for m in range(2, q + 2):
                loc_a = np.zeros(nu, dtype=complex)
                loc_b = np.zeros(nu, dtype=complex)
                if m - 2 <= q:
                    loc_a[m - 2] = own  # u^(m-2) at the centre is U^(m-2)
                if m - 1 <= q:
                    loc_b[m - 1] = own  # (u')^(m-2) at the centre is U^(m-1)
                a_der.append(-loc_a / M_P - k**2 * a_der[m - 2])
                b_der.append(-loc_b / M_P - k**2 * b_der[m - 2])
            for m in range(nu):
                big[i * nu + m, j * nu : (j + 1) * nu] -= a * a_der[m] + b * b_der[m + 1]
        for m in range(nu):
            rhs[i * nu + m] = (1j * k) ** m * np.exp(1j * k * zc[i])
    sol = np.linalg.solve(big, rhs).reshape(n, nu)
    out = []
    for z_o in (z_r, z_t):
        tot = 0.0j
        for j in range(n):
            gint, dint = cell_integrals(k, h, z_o - zc[j], q)
            tot += a * np.dot(gint, sol[j]) + b * np.dot(dint[:q], sol[j][1:])
        out.append(tot)
    return out[0], out[1]


def born_factor(degree: int, kh: float) -> tuple[float, float]:
    """Leading departure from one of the first-order term, (reflection, transmission), for a source held
    to a Taylor polynomial of this degree about the cell centre.

    A cell radiates  int P_d(i k xi) exp(-/+ i k xi) dxi  in place of  int exp(i k xi) exp(-/+ i k xi) dxi,
    with P_d the Taylor polynomial of exp(i k xi).  Expanding in k h (derived symbolically):
        d odd :  R and T alike   (-1)^((d-1)/2) (k h)^(d+1) / (d+2)!
        d even:  R   (-1)^(d/2) (k h)^(d+2) / (d+2)!
                 T   (-1)^(d/2+1) (d+1) (k h)^(d+2) / (d+3)!
    d = 0 gives (k d)^2/8 and -(k d)^2/24, the collocation constants of the uniform-field voxel.
    """
    if degree % 2 == 1:
        c = (-1) ** ((degree - 1) // 2) * kh ** (degree + 1) / math.factorial(degree + 2)
        return c, c
    j = degree // 2
    r = (-1) ** j * kh ** (degree + 2) / math.factorial(degree + 2)
    t = (-1) ** (j + 1) * (degree + 1) * kh ** (degree + 2) / math.factorial(degree + 3)
    return r, t


def check_born_constants() -> bool:
    """At weak contrast the scheme over the exact layer, less one, is the closed-form factor above."""
    omega, n, scale = 600.0, 1, 1.0e-6
    z_r, z_t = -1.0, 4.0
    kh = (omega / ALPHA) * D_LAYER / n / 2.0
    ok = True
    print(f"\nfirst-order factor against its closed form, weak contrast, n = {n}, k h = {kh:.3f}")
    for label, drho, dm, shift in (("density", D_RHO * scale, 0.0, 0), ("modulus", 0.0, D_M * scale, 1)):
        ex = exact_scattered(omega, drho, dm, z_r, z_t)
        for q in (1, 2, 3, 4, 5):
            if q - shift > 3:
                # the factor is below 1e-8 from degree 4, and the weak-contrast scattered field (1e-7 of
                # the incident) carries a relative round-off of that size in double precision: those
                # degrees are checked in Mathematica/ContinuumLimit_GradientHierarchy.wl at 60 digits
                continue
            got = solve(omega, n, q, drho, dm, z_r, z_t)
            meas = [(got[0] / ex[0] - 1.0).real, (got[1] / ex[1] - 1.0).real]
            pred = born_factor(q - shift, kh)
            rel = [abs(m - p) / abs(p) for m, p in zip(meas, pred, strict=True)]
            good = max(rel) < 0.05
            ok = ok and good
            print(
                f"  {label} q = {q} (source degree {q - shift}):  R {meas[0]:+.4e} vs {pred[0]:+.4e}   "
                f"T {meas[1]:+.4e} vs {pred[1]:+.4e}   {'PASS' if good else '****FAIL****'}"
            )
    return ok


def main() -> int:
    omega = 300.0
    z_r, z_t = -1.0, 4.0
    ns = [2, 4, 8, 16, 32]
    ok = True
    for label, drho, dm, want in (
        ("density only", D_RHO, 0.0, {1: 2, 2: 4, 3: 4}),
        ("modulus only", 0.0, D_M, {1: 2, 2: 2, 3: 4}),
        ("both", D_RHO, D_M, {1: 2, 2: 2, 3: 4}),
    ):
        ex = exact_scattered(omega, drho, dm, z_r, z_t)
        print(f"\n{label}: relative error of the scattered field, R / T, omega = {omega:.0f}")
        for q in (1, 2, 3):
            errs = []
            for n in ns:
                got = solve(omega, n, q, drho, dm, z_r, z_t)
                errs.append([abs(got[0] - ex[0]) / abs(ex[0]), abs(got[1] - ex[1]) / abs(ex[1])])
            errs = np.array(errs)
            order = np.log2(errs[-2] / errs[-1])
            line = "   ".join(f"n={n:2d} {e[0]:.2e}/{e[1]:.2e}" for n, e in zip(ns, errs, strict=True))
            good = bool(np.all(np.abs(order - want[q]) < 0.4))
            ok = ok and good
            print(f"  q = {q}:  {line}")
            print(
                f"          order (16 -> 32): R {order[0]:.2f}  T {order[1]:.2f}   predicted {want[q]}   "
                f"{'PASS' if good else '****FAIL****'}"
            )
    ok = check_born_constants() and ok
    print("\nALL PASS" if ok else "\nSOME PREDICTIONS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
