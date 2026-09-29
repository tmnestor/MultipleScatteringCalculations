#!/usr/bin/env python3
"""CROSS-CHECK: cells whose contrast carries a Legendre expansion (notebook 15), independently.

``Mathematica/ContinuumLimit_GradedContrast.wl`` builds the scheme from closed forms: Legendre moments of
the exponential kernel, same-cell double integrals and Legendre linearisation coefficients, then solves
the graded layer against a 90-digit ODE reference. This script tests those closed forms themselves, not
only the assembled result, and then rebuilds the scheme without them:

  [1] Legendre moments int_{-h}^{h} P_n(s/h) e^{i c s} ds: notebook 15's closed form against the
      independent closed form 2h i^n j_n(c h) (spherical Bessel) and against Gauss quadrature;
  [2] linearisation P_c P_b = sum_e A_cbe P_e: notebook 15's table against the Wigner 3j closed form
      A_cbe = (2e+1) (c b e; 0 0 0)^2 and against quadrature;
  [3] same-cell double integrals int int P_a K P_b (with the local term): notebook 15's closed form
      against quadrature split at the receiver;
  [4] the exact graded layer by mpmath's 30-digit Taylor-series ODE solver on the scaled state
      (u, M u'/M_P): the closed form for a constant contrast;
  [5] the scheme assembled WITHOUT linearisation (the projected contrast evaluated at the quadrature
      nodes of every integral): its errors against notebook 15's, within the reference's round-off;
  [6] orders at omega = 1200 against the rule min(2p + 2, 2r + 2), or 2p + 2 when degree r is exact.

Run:  conda run -n seismic python scripts/crosscheck_graded_contrast.py
SI units; e^{-i omega t}; z down.
"""

import json
import sys
from pathlib import Path

import mpmath as mp
import numpy as np
from numpy.polynomial.legendre import Legendre
from scipy.special import spherical_jn
from sympy.physics.wigner import wigner_3j

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crosscheck_second_moment_voxel import (  # noqa: E402
    ALPHA,
    CONTRAST,
    D_LAYER,
    M_P,
    RHO,
    Z_OBS_R,
    Z_OBS_T,
    Z_SRC,
    exact_scattered,
    g_inc,
    gauss,
    kernel,
)

ROOT = Path(__file__).resolve().parent.parent
REF = ROOT / "Mathematica" / "ContinuumLimit_graded_contrast_data.json"
PROFILES = {
    "const": (lambda z: np.ones_like(z), 0),
    "linear": (lambda z: 1 + (2 * z / D_LAYER - 1) / 2, 1),
    "smooth": (lambda z: 1 + np.sin(2 * np.pi * z / D_LAYER) / 2, None),  # None: no finite degree
}


def c_(pair: list[float]) -> complex:
    return complex(pair[0], pair[1])


def leg(n: int, x: np.ndarray) -> np.ndarray:
    return Legendre.basis(n)(x)


# ---------------------------------------------------------------- closed forms, independently
def moment_bessel(n: int, c: float, h: float) -> complex:
    """int_{-h}^{h} P_n(s/h) e^{i c s} ds = 2h i^n j_n(c h)."""
    return complex(2 * h * (1j**n) * spherical_jn(n, c * h))


# the referee for the closed forms is 30-digit quadrature: the high-degree moments and blocks are small
# differences of O(h) terms, which double-precision quadrature resolves only to round-off
mp.mp.dps = 30


def moment_mp(n: int, c: float, h: float) -> mp.mpc:
    hh, cc = mp.mpf(h), mp.mpf(c)
    return mp.quad(lambda s: mp.legendre(n, s / hh) * mp.expj(cc * s), [-hh, hh])


def self_mp(a: int, b: int, k: float, h: float) -> list[list[mp.mpc]]:
    """int int P_a(s/h) K(s - t) P_b(t/h) dt ds by nested 30-digit quadrature, split at t = s."""
    hh, kk = mp.mpf(h), mp.mpf(k)

    def inner(s: mp.mpf, sgn: int) -> mp.mpc:
        below = mp.quad(lambda t: mp.legendre(b, t / hh) * mp.expj(kk * (s - t)), [-hh, s])
        above = mp.quad(lambda t: mp.legendre(b, t / hh) * mp.expj(kk * (t - s)), [s, hh])
        return below + sgn * above

    i0 = mp.quad(lambda s: mp.legendre(a, s / hh) * inner(s, 1), [-hh, hh])  # e^{ik|s-t|}
    i1 = mp.quad(lambda s: mp.legendre(a, s / hh) * inner(s, -1), [-hh, hh])  # sign(s - t) e^{ik|s-t|}
    pre = 1j / (2 * mp.mpf(M_P) * kk)
    local = mp.quad(lambda s: mp.legendre(a, s / hh) * mp.legendre(b, s / hh), [-hh, hh]) / mp.mpf(M_P)
    return [[pre * i0, pre * 1j * kk * i1], [pre * 1j * kk * i1, -pre * kk**2 * i0 - local]]


def lin_3j(c: int, b: int, e: int) -> float:
    return float((2 * e + 1) * wigner_3j(c, b, e, 0, 0, 0) ** 2)


def lin_quad(c: int, b: int, e: int) -> float:
    x, w = gauss(-1.0, 1.0)
    return float((2 * e + 1) / 2 * np.sum(w * leg(c, x) * leg(b, x) * leg(e, x)))


def self_quad(a: int, b: int, k: float, h: float, weight=None) -> np.ndarray:
    """int int P_a(s/h) K(s - t) w(t) P_b(t/h) dt ds over one cell, split at t = s, plus the local term."""
    wt_fn = weight if weight is not None else (lambda t: np.ones_like(t))
    s, ws = gauss(-h, h)
    out = np.zeros((2, 2), dtype=complex)
    for si, wi in zip(s, ws, strict=True):
        for lo, hi in ((-h, si), (si, h)):
            t, wt = gauss(lo, hi)
            inner = np.einsum("n,nij->ij", wt * leg(b, t / h) * wt_fn(t), kernel(k, si - t))
            out += wi * leg(a, np.array(si / h)) * inner
    out[1, 1] -= np.sum(ws * leg(a, s / h) * leg(b, s / h) * wt_fn(s)) / M_P
    return out


# ---------------------------------------------------------------- the exact graded layer
MP_PROFILES = {
    "const": lambda _z: mp.mpf(1),
    "linear": lambda z: 1 + (2 * z / D_LAYER - 1) / 2,
    "smooth": lambda z: 1 + mp.sin(2 * mp.pi * z / D_LAYER) / 2,
}


def exact_graded(omega: float, name: str, contrast=CONTRAST) -> np.ndarray:
    """Scattered u at the two observers for the graded layer, by mpmath's Taylor-series ODE solver on
    (u, y = M u'/M_P) at 30 digits, kept in 30 digits through the final total-minus-incident subtraction."""
    d_lam, d_mu, d_rho = (mp.mpf(c) for c in contrast)
    d_m = d_lam + 2 * d_mu
    mpp, rho, om = mp.mpf(M_P), mp.mpf(RHO), mp.mpf(omega)
    k0 = om / ALPHA
    f = MP_PROFILES[name]

    def rhs(z, y):
        return [mpp * y[1] / (mpp + d_m * f(z)), -(om**2) * (rho + d_rho * f(z)) * y[0] / mpp]

    cols = [mp.odefun(rhs, 0, ic)(D_LAYER) for ic in ([mp.mpf(1), mp.mpf(0)], [mp.mpf(0), mp.mpf(1)])]
    pm = [[cols[0][0], cols[1][0]], [cols[0][1], cols[1][1]]]
    # pm @ (1 + R, i k0 (1 - R)) = (T, i k0 T): linear in (R, T)
    a = mp.matrix([[pm[0][0] - 1j * k0 * pm[0][1], -1], [pm[1][0] - 1j * k0 * pm[1][1], -1j * k0]])
    rhs0 = -mp.matrix([pm[0][0] + 1j * k0 * pm[0][1], pm[1][0] + 1j * k0 * pm[1][1]])
    rr, tt = mp.lu_solve(a, rhs0)

    def g(z):
        return 1j / (2 * mpp * k0) * mp.expj(k0 * abs(mp.mpf(z) - Z_SRC))

    scat = [rr * g(0) * mp.expj(-k0 * Z_OBS_R), tt * g(0) * mp.expj(k0 * (Z_OBS_T - D_LAYER)) - g(Z_OBS_T)]
    return np.array([complex(v) for v in scat])


# ---------------------------------------------------------------- the scheme, without linearisation
def graded_scattered(omega: float, n: int, p: int, r: int, profile) -> np.ndarray:
    """Field degree p, contrast projected to degree r per cell, the product never re-expanded."""
    d_lam, d_mu, d_rho = CONTRAST
    k = omega / ALPHA
    h = D_LAYER / (2 * n)
    centres = (np.arange(n) + 0.5) * 2 * h
    dq = np.diag([omega**2 * d_rho, d_lam + 2 * d_mu])
    nb = p + 1
    s, ws = gauss(-h, h)
    phi = np.array([leg(a, s / h) for a in range(nb)])
    # each cell's contrast, projected to degree r by quadrature: a function of the local coordinate
    coef = np.array(
        [
            [(2 * c + 1) / (2 * h) * np.sum(ws * profile(zc + s) * leg(c, s / h)) for c in range(r + 1)]
            for zc in centres
        ]
    )

    def f_r(j: int, loc: np.ndarray) -> np.ndarray:
        return sum(coef[j, c] * leg(c, loc / h) for c in range(r + 1))

    size = 2 * nb * n
    mat = np.zeros((size, size), dtype=complex)
    rhs = np.zeros(size, dtype=complex)
    for i in range(n):
        zi = centres[i] + s
        for a in range(nb):
            row = 2 * (nb * i + a)
            mat[row : row + 2, row : row + 2] += (2 * h / (2 * a + 1)) * np.eye(2)
            w0 = np.array([[g_inc(k, z), 1j * k * g_inc(k, z)] for z in zi])
            rhs[row : row + 2] = np.einsum("n,n,ni->i", phi[a], ws, w0)
        for j in range(n):
            if i == j:
                blocks = [
                    [self_quad(a, b, k, h, lambda t, j=j: f_r(j, t)) for b in range(nb)] for a in range(nb)
                ]
            else:
                zj = centres[j] + s
                kk = kernel(k, zi[:, None] - zj[None, :])
                fj = f_r(j, s)
                full = np.einsum("an,n,bm,m,m,nmij->abij", phi, ws, phi, ws, fj, kk)
                blocks = [[full[a, b] for b in range(nb)] for a in range(nb)]
            for a in range(nb):
                for b in range(nb):
                    rr, cc = 2 * (nb * i + a), 2 * (nb * j + b)
                    mat[rr : rr + 2, cc : cc + 2] -= blocks[a][b] @ dq
    sol = np.linalg.solve(mat, rhs)
    out = []
    for zo in (Z_OBS_R, Z_OBS_T):
        total = 0j
        for j in range(n):
            kz = kernel(k, zo - (centres[j] + s))
            fj = f_r(j, s)
            for b in range(nb):
                col = 2 * (nb * j + b)
                total += (np.einsum("n,n,n,nij->ij", phi[b], ws, fj, kz) @ dq @ sol[col : col + 2])[0]
        out.append(total)
    return np.array(out)


def main() -> int:
    ref = json.loads(REF.read_text())
    oks = []
    print("==== crosscheck_graded_contrast :: graded cells, closed forms first ====")

    worst_b, worst_q = 0.0, 0.0
    for cf in ref["closed_forms"]:
        k, h = cf["k"], cf["h"]
        for sign, key in ((1, "moment_plus"), (-1, "moment_minus")):
            for n, val in enumerate(cf[key]):
                want = c_(val)
                worst_b = max(worst_b, abs(moment_bessel(n, sign * k, h) - want) / abs(want))
                worst_q = max(worst_q, float(abs(moment_mp(n, sign * k, h) - want) / abs(want)))
    oks.append(worst_b < 1e-12 and worst_q < 1e-12)
    print(
        f"  [1] Legendre moments, P_0..P_4 at three (k, h), c = +-k: vs 2h i^n j_n(ch) {worst_b:.1e}, "
        f"vs 30-digit quadrature {worst_q:.1e}: {'PASS' if oks[-1] else 'FAIL'}"
    )

    lin = np.array(ref["linearisation"])
    worst_3j = max(
        abs(lin_3j(c, b, e) - lin[c, b, e]) for c in range(3) for b in range(3) for e in range(5)
    )
    worst_lq = max(
        abs(lin_quad(c, b, e) - lin[c, b, e]) for c in range(3) for b in range(3) for e in range(5)
    )
    oks.append(worst_3j < 1e-15 and worst_lq < 1e-14)
    print(
        f"  [2] linearisation A_cbe, c, b <= 2, e <= 4: vs Wigner 3j {worst_3j:.1e}, vs quadrature "
        f"{worst_lq:.1e}: {'PASS' if oks[-1] else 'FAIL'}"
    )

    worst_s = 0.0
    for cf in ref["closed_forms"]:
        k, h = cf["k"], cf["h"]
        for a in range(5):
            for b in range(5):
                blk = np.array([[c_(v) for v in row] for row in cf["self"][a][b]])
                got = self_mp(a, b, k, h)
                dev = max(abs(got[i][j] - blk[i, j]) for i in range(2) for j in range(2))
                worst_s = max(worst_s, float(dev / np.max(np.abs(blk))))
    oks.append(worst_s < 1e-12)
    print(
        f"  [3] same-cell double integrals with the local term, P_0..P_4 squared, three (k, h), "
        f"vs 30-digit quadrature: {worst_s:.1e}: {'PASS' if oks[-1] else 'FAIL'}"
    )

    omega = ref["omega"]
    dev_c = float(np.max(np.abs(exact_graded(omega, "const") / exact_scattered(omega, CONTRAST) - 1)))
    oks.append(dev_c < 1e-11)
    print(
        f"  [4] graded layer by the 30-digit Taylor ODE, constant contrast vs closed form: {dev_c:.1e}: "
        f"{'PASS' if oks[-1] else 'FAIL'}"
    )

    exact = {name: exact_graded(omega, name) for name in ("linear", "smooth")}
    rows = []
    for run in ref["runs"]:
        for n, want in zip(ref["n"], run["errors_R_T"], strict=True):
            got = np.abs(
                graded_scattered(omega, n, run["p"], run["r"], PROFILES[run["profile"]][0])
                / exact[run["profile"]]
                - 1
            )
            rows.append((run["profile"], run["p"], run["r"], n, got, np.array(want)))
    # the reference is exact to 30 digits, so the only limit is the double-precision discrete solve: about
    # 1e-15 of the scattered field (a few machine epsilons times the conditioning). Every error must agree
    # with notebook 15's to 1e-15 absolute plus 1e-3 relative.
    floor = 1e-15
    ratio = max(float(np.max(np.abs(g - w) / (floor + 1e-3 * w))) for *_, g, w in rows)
    resolved = sum(int(np.sum(w > 100 * floor)) for *_, w in rows)
    worst_abs = max(float(np.max(np.abs(g - w))) for *_, g, w in rows if np.all(w < 100 * floor))
    oks.append(ratio < 1)
    print(
        f"  [5] the scheme without linearisation vs notebook 15, {len(rows)} runs x R, T: all agree within "
        f"1e-15 + 1e-3 relative; {resolved} errors above 1e-13 agree to 1e-3, the rest (below 1e-13) to "
        f"{worst_abs:.1e} absolute: {'PASS' if oks[-1] else 'FAIL'}"
    )

    om = 1200.0
    print(f"  [6] orders at omega = {om:g} (two finest halvings), R / T:")
    ladders = {"linear": (2, 4, 8), "smooth": (4, 8, 16)}
    cases = {"linear": ((0, 0), (1, 0), (1, 1), (2, 1)), "smooth": ((1, 0), (1, 1), (2, 1), (2, 2))}
    order_ok = True
    for name in ("linear", "smooth"):
        prof, deg = PROFILES[name]
        ex = exact_graded(om, name)
        for p, r in cases[name]:
            e = np.array([np.abs(graded_scattered(om, n, p, r, prof) / ex - 1) for n in ladders[name]])
            order = np.log2(e[0] / e[-1]) / 2
            want = 2 * p + 2 if (deg is not None and r >= deg) else min(2 * p + 2, 2 * r + 2)
            print(
                f"      {name:6s} p = {p}, r = {r}:  errors (R) {e[:, 0]}  order {order.round(3)}  "
                f"predicted {want}"
            )
            order_ok &= bool(np.all(np.abs(order - want) < 0.25))
    oks.append(order_ok)
    print(f"      as predicted: {'PASS' if order_ok else 'FAIL'}")

    summary = f"ALL {len(oks)} CHECKS PASS" if all(oks) else "CHECKS FAILED"
    print(f"==== crosscheck_graded_contrast: {summary} ====")
    return 0 if all(oks) else 1


if __name__ == "__main__":
    sys.exit(main())
