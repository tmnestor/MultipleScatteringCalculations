#!/usr/bin/env python3
"""Why the Legendre cell carries the strain as an independent unknown: a test on the exact layer.

The governing equation is in displacement; the strain enters through the stiffness contrast. On a layer
of thickness D at normal incidence (P channel, one dimension), with g(z) = i e^(ik|z|) / (2 k M_P) the
plane Green's function of the background,

    u(z)   = u0(z)   + int g(z - z')   w2 drho u(z') dz' + int g'(z - z')  dM eps(z') dz',
    eps(z) = eps0(z) + int g'(z - z')  w2 drho u(z') dz' + int g''(z - z') dM eps(z') dz',
    g'' = -k^2 g - delta / M_P.

Each cell carries Legendre polynomials of degree p, tested against the same (Galerkin). Three schemes:

  A  independent strain: u and eps each of degree p, both equations tested (the Legendre cell);
  B  derived strain: u of degree p only, eps the cellwise derivative of each cell's u, the displacement
     equation tested;
  C  as B, with the derivative taken in the sense of distributions: the jumps [u] of the cellwise
     displacement across the interior faces add dM [u] delta(z - z_f) to the stiffness source.

B drops the face terms that C keeps; C is the derived-strain scheme made consistent. The reflected
displacement is compared with the exact layer's. Measured: the error and its order for p = 1, 2, 3.

Run:  python -u scripts/measure_derived_vs_independent_strain.py
"""

import math
import sys

import numpy as np
from numpy.polynomial.legendre import Legendre, leggauss

ALPHA, RHO = 5000.0, 2500.0
M_P = RHO * ALPHA**2
DM, DRHO = 4.0e9, 100.0  # Delta(lambda + 2 mu), Delta rho
D_LAYER, Z_OBS = 2.0, -1.0
NG = 24  # Gauss points per (sub)interval


def green(z: np.ndarray, k: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """g, g' and the smooth part of g'' (the -delta / M_P part is handled separately)."""
    e = 1j * np.exp(1j * k * np.abs(z)) / (2.0 * k * M_P)
    return e, 1j * k * np.sign(z) * e, -(k**2) * e


def exact_reflection(omega: float) -> complex:
    """Scattered displacement at Z_OBS < 0 for the incident wave e^(ikz) on the layer [0, D]."""
    k0 = omega / ALPHA
    m1, r1 = M_P + DM, RHO + DRHO
    k1 = omega * math.sqrt(r1 / m1)
    z0, z1 = math.sqrt(M_P * RHO), math.sqrt(m1 * r1)
    r = (z0 - z1) / (z0 + z1)
    ph = np.exp(2j * k1 * D_LAYER)
    big_r = r * (1 - ph) / (1 - r**2 * ph)
    return big_r * np.exp(-1j * k0 * Z_OBS)


def solve(scheme: str, p: int, n: int, omega: float) -> complex:
    k = omega / ALPHA
    w2 = omega**2
    h = D_LAYER / (2 * n)
    centres = h * (2 * np.arange(n) + 1)
    x, wq = leggauss(NG)
    polys = [Legendre.basis(b) for b in range(p + 1)]
    dpolys = [P.deriv() for P in polys]

    def phi(b, s):
        return polys[b](s / h)

    def dphi(b, s):
        return dpolys[b](s / h) / h

    # outer points in every cell
    zo = (centres[:, None] + h * x[None, :])  # (n, NG)
    wo = h * wq
    nb = p + 1
    ncomp = 2 if scheme == "A" else 1
    size = n * nb * ncomp
    mat = np.zeros((size, size), dtype=complex)
    rhs = np.zeros(size, dtype=complex)

    def idx(j, b, c=0):
        return (j * nb + b) * ncomp + c

    def inner(zv: float, j: int):
        """Gauss nodes and weights over cell j for an outer point zv, split at zv in the same cell."""
        a, b = centres[j] - h, centres[j] + h
        if a < zv < b:
            n1 = 0.5 * (zv - a) * x + 0.5 * (zv + a)
            n2 = 0.5 * (b - zv) * x + 0.5 * (b + zv)
            return np.r_[n1, n2], np.r_[0.5 * (zv - a) * wq, 0.5 * (b - zv) * wq]
        return centres[j] + h * x, h * wq

    for i in range(n):
        for qo in range(NG):
            z = zo[i, qo]
            s_i = z - centres[i]
            u0, e0 = np.exp(1j * k * z), 1j * k * np.exp(1j * k * z)
            for a in range(nb):
                ta = phi(a, s_i) * wo[qo]
                rhs[idx(i, a, 0)] += ta * u0
                if scheme == "A":
                    rhs[idx(i, a, 1)] += ta * e0
            for j in range(n):
                zp, wp = inner(z, j)
                g, g1, g2 = green(z - zp, k)
                sj = zp - centres[j]
                for b in range(nb):
                    pb = phi(b, sj)
                    if scheme == "A":
                        # unknowns u_b (c=0), eps_b (c=1)
                        cu = (g * w2 * DRHO * pb) @ wp, (g1 * w2 * DRHO * pb) @ wp
                        ce = (g1 * DM * pb) @ wp, (g2 * DM * pb) @ wp
                        for a in range(nb):
                            ta = phi(a, s_i) * wo[qo]
                            mat[idx(i, a, 0), idx(j, b, 0)] -= ta * cu[0]
                            mat[idx(i, a, 1), idx(j, b, 0)] -= ta * cu[1]
                            mat[idx(i, a, 0), idx(j, b, 1)] -= ta * ce[0]
                            mat[idx(i, a, 1), idx(j, b, 1)] -= ta * ce[1]
                    else:
                        val = (g * w2 * DRHO * pb + g1 * DM * dphi(b, sj)) @ wp
                        for a in range(nb):
                            mat[idx(i, a), idx(j, b)] -= phi(a, s_i) * wo[qo] * val
            if scheme == "C":  # the face deltas: dM [u]_f g'(z - z_f), [u]_f = u_{j+1}(-h) - u_j(+h)
                for f in range(n - 1):
                    zf = centres[f] + h
                    _, g1f, _ = green(np.array([z - zf]), k)
                    for b in range(nb):
                        for a in range(nb):
                            t = phi(a, s_i) * wo[qo] * DM * g1f[0]
                            mat[idx(i, a), idx(f + 1, b)] -= t * phi(b, -h)
                            mat[idx(i, a), idx(f, b)] += t * phi(b, h)
    # the mass terms, and the local part of g'' in A: -(1/M_P) dM int phi_a phi_b over the cell
    for i in range(n):
        for a in range(nb):
            norm = 2 * h / (2 * a + 1)
            for c in range(ncomp):
                mat[idx(i, a, c), idx(i, a, c)] += norm
            if scheme == "A":
                mat[idx(i, a, 1), idx(i, a, 1)] += DM / M_P * norm
    coef = np.linalg.solve(mat, rhs)
    # the scattered displacement at Z_OBS
    out = 0.0j
    for j in range(n):
        zp = centres[j] + h * x
        g, g1, _ = green(Z_OBS - zp, k)
        sj = zp - centres[j]
        for b in range(nb):
            if scheme == "A":
                u_b, e_b = coef[idx(j, b, 0)], coef[idx(j, b, 1)]
                out += ((g * w2 * DRHO * u_b + g1 * DM * e_b) * phi(b, sj)) @ (h * wq)
            else:
                d = coef[idx(j, b)]
                out += ((g * w2 * DRHO * phi(b, sj) + g1 * DM * dphi(b, sj)) * d) @ (h * wq)
    if scheme == "C":
        for f in range(n - 1):
            zf = centres[f] + h
            _, g1f, _ = green(np.array([Z_OBS - zf]), k)
            jump = sum(coef[idx(f + 1, b)] * phi(b, -h) - coef[idx(f, b)] * phi(b, h) for b in range(nb))
            out += DM * g1f[0] * jump
    return out


def main() -> int:
    omega = 1500.0
    ref = exact_reflection(omega)
    k = omega / ALPHA
    print(f"layer D = {D_LAYER} m, omega = {omega}, k_P D = {k * D_LAYER:.2f}; exact reflected u = {ref:.6e}")
    ns = [2, 4, 8, 16, 32]
    for p in (1, 2, 3):
        print(f"p = {p}")
        for scheme in "ABC":
            errs = [abs(solve(scheme, p, n, omega) - ref) / abs(ref) for n in ns]
            ords = [math.log(errs[i] / errs[i + 1]) / math.log(2) for i in range(len(ns) - 1)]
            print(f"   {scheme}: err " + " ".join(f"{e:.2e}" for e in errs)
                  + " | order " + " ".join(f"{o:.2f}" for o in ords))
    return 0


if __name__ == "__main__":
    sys.exit(main())
