#!/usr/bin/env python3
"""CROSS-CHECK: an independent implementation of the Legendre Galerkin voxel of any degree p.

``Mathematica/ContinuumLimit_SecondMoment.wl`` (notebook 14) builds the normal-incidence P layer of
notebook 7 with Legendre moments up to degree p, symbolically: closed-form moments and same-cell double
integrals, 40-digit arithmetic. This script shares only the DEFINITION of the scheme and computes every
ingredient another way, in double precision:

  * every single and double integral by Gauss-Legendre quadrature; within a cell the inner integral is
    split at the receiver, where the kernel's derivative jumps, so both pieces are smooth;
  * the exact layer from its own 4 x 4 interface system;
  * the closed-form Born error by sympy.

THE SCHEME (notebook 7).  Field w = (u, e), e = du/dz, polarisation q = (omega^2 drho u, dM e), continuum
equation w = w0 + int K(z - z') q(z') dz', K = (i / 2 M k) e^{i k |z - z'|} [[1, i k s], [i k s, -k^2]]
with s = sign(z - z'), plus -delta(z - z')/M in the strain-strain entry. Each voxel carries (u, e) as a
sum of Legendre polynomials P_0 .. P_p of the local coordinate; the equation is tested with the same.

CHECKS: [1] c_p = ((p+1)!)^2 / ((2p+2)! (2p+3)!) for p = 0 .. 5, reflection and transmission, by sympy;
[2] the errors against the exact layer agree with notebook 14's (Mathematica/
ContinuumLimit_second_moment_data.json) for p = 0 .. 3 wherever they lie above double-precision round-off;
[3] at Born order (contrast x 1e-6) the measured error is the closed form, p = 1 and 2;
[4] the second-moment voxel is sixth order (and the first-moment voxel fourth) at omega = 1200;
[5] one plane of degree-p voxels differs from the exact layer at O(D^{2p+3}) as the layer thins.

Run:  conda run -n seismic python scripts/crosscheck_second_moment_voxel.py
SI units; e^{-i omega t}; z down.
"""

import json
import sys
from math import factorial
from pathlib import Path

import numpy as np
import sympy as sp
from numpy.polynomial.legendre import Legendre, leggauss

ROOT = Path(__file__).resolve().parent.parent
REF = ROOT / "Mathematica" / "ContinuumLimit_second_moment_data.json"

ALPHA, BETA, RHO = 5000.0, 3000.0, 2500.0
M_P = RHO * ALPHA**2
CONTRAST = (2e9, 1e9, 100.0)  # d_lambda, d_mu, d_rho
D_LAYER, Z_SRC, Z_OBS_R, Z_OBS_T = 2.0, -12.0, -1.0, 4.0
NODES, WEIGHTS = leggauss(24)


def g_inc(k: float, z: float) -> complex:
    return 1j / (2 * M_P * k) * np.exp(1j * k * abs(z - Z_SRC))


def kernel(k: float, dz: np.ndarray) -> np.ndarray:
    """The regular part of K(dz), dz = z - z' != 0, shape dz.shape + (2, 2)."""
    s = np.sign(dz)
    pre = 1j / (2 * M_P * k) * np.exp(1j * k * np.abs(dz))
    return pre[..., None, None] * np.stack(
        [
            np.stack([np.ones_like(s), 1j * k * s], -1),
            np.stack([1j * k * s, -(k**2) * np.ones_like(s)], -1),
        ],
        -2,
    )


def gauss(a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
    return 0.5 * (b - a) * NODES + 0.5 * (a + b), 0.5 * (b - a) * WEIGHTS


def exact_scattered(
    omega: float, contrast: tuple[float, float, float], d_layer: float = D_LAYER
) -> np.ndarray:
    """Scattered u at the two observers for the exact homogeneous layer."""
    d_lam, d_mu, d_rho = contrast
    k0 = omega / ALPHA
    m1 = M_P + d_lam + 2 * d_mu
    k1 = omega * np.sqrt((RHO + d_rho) / m1)
    ep, em = np.exp(1j * k1 * d_layer), np.exp(-1j * k1 * d_layer)
    # unknowns rr, tt, bb, cc: continuity of u and of the traction M du/dz at z = 0 and z = D
    a = np.array(
        [
            [1, 0, -1, -1],
            [-M_P * 1j * k0, 0, -m1 * 1j * k1, m1 * 1j * k1],
            [0, -1, ep, em],
            [0, -M_P * 1j * k0, m1 * 1j * k1 * ep, -m1 * 1j * k1 * em],
        ],
        dtype=complex,
    )
    rhs = np.array([-1, -M_P * 1j * k0, 0, 0], dtype=complex)
    rr, tt, _, _ = np.linalg.solve(a, rhs)
    g0 = g_inc(k0, 0.0)
    return np.array(
        [
            rr * g0 * np.exp(-1j * k0 * Z_OBS_R),
            tt * g0 * np.exp(1j * k0 * (Z_OBS_T - d_layer)) - g_inc(k0, Z_OBS_T),
        ]
    )


def discrete_scattered(
    omega: float, n: int, p: int, contrast: tuple[float, float, float], d_layer: float = D_LAYER
) -> np.ndarray:
    """The Legendre Galerkin voxel of degree p on n cells: scattered u at the two observers."""
    d_lam, d_mu, d_rho = contrast
    k = omega / ALPHA
    d = d_layer / n
    h = d / 2
    centres = (np.arange(n) + 0.5) * d
    dq = np.diag([omega**2 * d_rho, d_lam + 2 * d_mu])
    legs = [Legendre.basis(j) for j in range(p + 1)]
    nb = p + 1
    s, ws = gauss(-h, h)
    phi = np.array([leg(s / h) for leg in legs])  # (nb, nodes)

    def self_block() -> np.ndarray:
        """int int phi_a(s) K(s - t) phi_b(t) dt ds over one cell, split at t = s, plus the delta."""
        out = np.zeros((nb, nb, 2, 2), dtype=complex)
        for si, wi in zip(s, ws, strict=True):
            for lo, hi in ((-h, si), (si, h)):
                t, wt = gauss(lo, hi)
                kt = kernel(k, si - t)  # (nodes, 2, 2)
                pt = np.array([leg(t / h) for leg in legs])  # (nb, nodes)
                inner = np.einsum("bn,n,nij->bij", pt, wt, kt)
                out += wi * np.einsum("a,bij->abij", np.array([leg(si / h) for leg in legs]), inner)
        for a in range(nb):
            out[a, a, 1, 1] -= (2 * h / (2 * a + 1)) / M_P  # int P_a P_b = delta_ab 2h/(2a+1)
        return out

    selfb = self_block()
    size = 2 * nb * n
    mat = np.zeros((size, size), dtype=complex)
    rhs = np.zeros(size, dtype=complex)
    for i in range(n):
        zi = centres[i] + s
        for a in range(nb):
            r = 2 * (nb * i + a)
            mat[r : r + 2, r : r + 2] += (2 * h / (2 * a + 1)) * np.eye(2)
            w0 = np.array([[g_inc(k, z), 1j * k * g_inc(k, z)] for z in zi])
            rhs[r : r + 2] = np.einsum("n,n,ni->i", phi[a], ws, w0)
        for j in range(n):
            if i == j:
                blk = selfb
            else:
                zj = centres[j] + s
                kk = kernel(k, zi[:, None] - zj[None, :])  # (n_i, n_j, 2, 2)
                blk = np.einsum("an,n,bm,m,nmij->abij", phi, ws, phi, ws, kk)
            for a in range(nb):
                for b in range(nb):
                    r, c = 2 * (nb * i + a), 2 * (nb * j + b)
                    mat[r : r + 2, c : c + 2] -= blk[a, b] @ dq
    sol = np.linalg.solve(mat, rhs)
    out = []
    for zo in (Z_OBS_R, Z_OBS_T):
        total = 0j
        for j in range(n):
            zj = centres[j] + s
            kz = kernel(k, zo - zj)  # (nodes, 2, 2)
            for b in range(nb):
                c = 2 * (nb * j + b)
                total += (np.einsum("n,n,nij->ij", phi[b], ws, kz) @ dq @ sol[c : c + 2])[0]
        out.append(total)
    return np.array(out)


def closed_form() -> bool:
    k, h, s, x = sp.symbols("k h s x", positive=True)

    def moment(j: int, c: sp.Expr) -> sp.Expr:
        return sp.integrate(sp.legendre(j, s / h) * sp.exp(sp.I * c * s), (s, -h, h))

    ok = True
    for p in range(6):
        cp = sp.Rational(factorial(p + 1) ** 2, factorial(2 * p + 2) * factorial(2 * p + 3))
        for name, q, sign in (("R", k, (-1) ** p), ("T", -k, -1)):
            num = sum(moment(j, k) * moment(j, q) / (2 * h / (2 * j + 1)) for j in range(p + 1))
            den = 2 * h if name == "T" else moment(0, 2 * k)
            ser = sp.series((num / den).subs(h, x / (2 * k)), x, 0, 2 * p + 3).removeO()
            got = sp.nsimplify(sp.expand(sp.simplify(ser))) - 1
            ok &= sp.simplify(got - sign * cp * x ** (2 * p + 2)) == 0
        print(f"      p = {p}: c_p = {cp}  {'ok' if ok else 'MISMATCH'}", flush=True)
    return bool(ok)


def main() -> int:
    oks = []
    print("==== crosscheck_second_moment_voxel :: Legendre Galerkin voxels of degree p, independently ====")
    print(
        "  [1] the closed-form Born error, by sympy "
        "(reflection (-1)^p c_p (kd)^(2p+2), transmission -c_p (kd)^(2p+2)):"
    )
    oks.append(closed_form())
    print(f"      {'PASS' if oks[-1] else 'FAIL'}")

    ref = json.loads(REF.read_text())
    omega = ref["omega"]
    ex = exact_scattered(omega, CONTRAST)
    print(f"  [2] errors against the exact layer, omega = {omega}: Python (notebook 14), {{R, T}}")
    rows = []
    for p in range(4):
        for n, want in zip(ref["n"], ref["errors_R_T"][f"G{p}"], strict=True):
            got = np.abs(discrete_scattered(omega, n, p, CONTRAST) / ex - 1)
            rows.append((got, np.array(want)))
            row = "   ".join(f"{g:.3e} ({w:.3e})" for g, w in zip(got, want, strict=True))
            print(f"      p = {p}, n = {n:2d}:  {row}")
    # double precision cannot resolve an error below its round-off floor: the transmitted scattered field is
    # total minus incident. The floor is what Python returns where notebook 14's error is below 1e-19.
    floor = np.max([g for g, w in rows if np.all(w < 1e-19)], axis=0)
    diff = max(float(np.max(np.abs(g - w) / (3 * floor + 1e-6 * w))) for g, w in rows)
    resolved = sum(int(np.sum(w > 100 * floor)) for _, w in rows)
    oks.append(diff < 1)
    print(
        f"      round-off floor {floor[0]:.1e} (R), {floor[1]:.1e} (T); every error agrees within "
        f"3 x floor ({resolved} of them resolved above 100 x floor): {'PASS' if oks[-1] else 'FAIL'}"
    )

    # contrast x 1e-3: large enough that the scattered field clears round-off, small enough that the
    # O(contrast) remainder (0.4 x contrast in notebook 14) is far below the next power of k d; omega = 1200
    # puts the sixth-order error at n = 2 (2e-9) far above the floor
    eps = 1e-3
    small = (eps * CONTRAST[0], eps * CONTRAST[1], eps * CONTRAST[2])
    om = 1200.0
    ex_b = exact_scattered(om, small)
    k = om / ALPHA
    worst = 0.0
    print(f"  [3] Born order (contrast x {eps:g}), omega = {om:g}: disc/exact - 1, measured (closed form)")
    for p in (1, 2):
        cp = factorial(p + 1) ** 2 / (factorial(2 * p + 2) * factorial(2 * p + 3))
        for n in (1, 2):
            x = k * D_LAYER / n
            got = discrete_scattered(om, n, p, small) / ex_b - 1
            pred = np.array([(-1) ** p * cp * x ** (2 * p + 2), -cp * x ** (2 * p + 2)])
            worst = max(worst, *np.abs(got.real / pred - 1))
            print(
                f"      p = {p}, n = {n}:  R {got[0].real:.4e} ({pred[0]:.4e})   "
                f"T {got[1].real:.4e} ({pred[1]:.4e})"
            )
    # the closed form is the leading term; the next power of k d is a relative O((k d)^2) correction
    oks.append(worst < 0.1)
    verdict = "PASS" if oks[-1] else "FAIL"
    print(f"      worst relative difference {worst:.2e} (the next term in k d): {verdict}")

    ex6 = exact_scattered(om, CONTRAST)
    print(f"  [4] orders at the full contrast, omega = {om:g}, n = 1, 2, 4:")
    for p, target in ((1, 4), (2, 6)):
        e = np.array([np.abs(discrete_scattered(om, n, p, CONTRAST) / ex6 - 1) for n in (1, 2, 4)])
        order = np.log2(e[0] / e[2]) / 2
        print(f"      p = {p}: errors {e[:, 0]} (R)  order R {order[0]:.3f}, T {order[1]:.3f}")
        oks.append(bool(np.all(np.abs(order - target) < 0.1)))
    print(f"      fourth and sixth order: {'PASS' if all(oks[-2:]) else 'FAIL'}")

    # ONE plane of voxels (n = 1) as the layer thins: notebook 14 finds R_plane - R_exact = O(D^{2p+3})
    thick = (1.0, 0.5, 0.25)
    print(
        f"  [5] one plane of voxels as the layer thins, omega = {om:g}, D = {thick} m: |R_plane - R_exact|"
    )
    slopes_ok = True
    for p in (0, 1, 2):
        gap = np.array(
            [
                abs(discrete_scattered(om, 1, p, CONTRAST, dl)[0] - exact_scattered(om, CONTRAST, dl)[0])
                for dl in thick
            ]
        )
        slope = np.log2(gap[:-1] / gap[1:])
        print(f"      p = {p}: {gap}  power of D {slope.round(3)} (expect {2 * p + 3})")
        slopes_ok &= bool(np.all(np.abs(slope - (2 * p + 3)) < 0.15))
    oks.append(slopes_ok)
    print(f"      the first difference is at D^(2p+3): {'PASS' if slopes_ok else 'FAIL'}")

    summary = f"ALL {len(oks)} CHECKS PASS" if all(oks) else "CHECKS FAILED"
    print(f"==== crosscheck_second_moment_voxel: {summary} ====")
    return 0 if all(oks) else 1


if __name__ == "__main__":
    sys.exit(main())
