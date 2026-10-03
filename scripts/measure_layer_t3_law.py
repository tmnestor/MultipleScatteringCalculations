#!/usr/bin/env python3
"""The layer: the error of the third-order term, and of the whole nonlinear part, against the local law.

In the long-wave limit the kernel that maps a strain source to the strain it scatters is local, a
constant c, and with field and contrast of the same degree p in each cell the scheme's n-th term is
c^(n-1) int (Pi f)^n against the exact c^(n-1) int f^n (Pi the L2 projection onto the cell's Legendre
polynomials, f the profile). So the relative error of each term is predicted, from the profile alone, as

    e_n = (int f^n - int (Pi f)^n) / int f^n ,

e_2 being the relative projection error E (Paper 1, Appendix G). Measured here, on the layer of
``measure_layer_bases.py`` (profile 1 + sin(2 pi z / D) / 2): the scheme's T2 and T3 and those of the
exact layer, from solutions at scaled contrasts +-d and +-2d,

    T2 = (16 (u(d) + u(-d)) - (u(2d) + u(-2d))) / (24 d^2),
    T3 = ((u(2d) - u(-2d)) - 2 (u(d) - u(-d))) / (12 d^3),

and the relative error of each, (T - T_scheme) / T, in reflection and in transmission, against e_n; and
the relative error of T2 + T3 against the same combination of the predictions.

Run:  conda run -n seismic python -u scripts/measure_layer_t3_law.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crosscheck_graded_contrast as cg  # noqa: E402
import measure_layer_bases as mlb  # noqa: E402
from crosscheck_second_moment_voxel import CONTRAST, D_LAYER, M_P, gauss  # noqa: E402

D = 1e-2


def series(f) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v = {s: f(s) for s in (D, -D, 2 * D, -2 * D)}
    o1, o2 = v[D] - v[-D], v[2 * D] - v[-2 * D]
    e1, e2 = v[D] + v[-D], v[2 * D] + v[-2 * D]
    t1 = (8 * o1 - o2) / (12 * D)
    t2 = (16 * e1 - e2) / (24 * D * D)
    t3 = (o2 - 2 * o1) / (12 * D**3)
    return t1, t2, t3


def moments(n: int, p: int) -> tuple[float, float, float, float]:
    """int f^2, int (Pi f)^2, int f^3, int (Pi f)^3 over the layer cut into n cells."""
    h = D_LAYER / (2 * n)
    s, w = gauss(-h, h)
    prof = cg.PROFILES[mlb.PROFILE][0]
    out = np.zeros(4)
    for zc in (np.arange(n) + 0.5) * 2 * h:
        vals = prof(zc + s)
        fit = sum(
            (2 * c + 1) / (2 * h) * np.sum(w * vals * cg.leg(c, s / h)) * cg.leg(c, s / h)
            for c in range(p + 1)
        )
        out += [np.sum(w * vals**2), np.sum(w * fit**2), np.sum(w * vals**3), np.sum(w * fit**3)]
    return tuple(out)


def main() -> int:
    print("layer, profile 1 + sin(2 pi z/D)/2: relative errors (T - T_scheme)/T, reflection | transmission")
    for omega in (30.0, 300.0):
        ex = series(lambda s, _o=omega: mlb.exact(_o, s))
        print(
            f"\nomega = {omega:.0f} rad/s:  |T3|/|T2| = {np.abs(ex[2] / ex[1])[0]:.3f} (R), "
            f"{np.abs(ex[2] / ex[1])[1]:.3f} (T);"
            f"  T3/T2 = {(ex[2] / ex[1])[0].real:+.3f} (R)"
        )
        for p in (0, 1, 2):
            for n in (4, 8, 16):
                sc = series(lambda s, _o=omega, _n=n, _p=p: mlb.scattered(_o, _n, _p, _p, s))
                f2, pf2, f3, pf3 = moments(n, p)
                e2p, e3p = (f2 - pf2) / f2, (f3 - pf3) / f3
                e2 = (ex[1] - sc[1]) / ex[1]
                e3 = (ex[2] - sc[2]) / ex[2]
                rest = ((ex[1] + ex[2]) - (sc[1] + sc[2])) / (ex[1] + ex[2])
                rest_p = (e2p * ex[1] + e3p * ex[2]) / (ex[1] + ex[2])
                print(
                    f"  p = r = {p}, n = {n:2d}:  e2/E2 {(e2 / e2p)[0].real:.4f} | {(e2 / e2p)[1].real:.4f}"
                    f"   e3/e3_pred {(e3 / e3p)[0].real:.4f} | {(e3 / e3p)[1].real:.4f}"
                    f"   (e3_pred/E2 = {e3p / e2p:.3f})"
                    f"   nonlinear part: error/E2 {(rest / e2p)[0].real:.4f} | {(rest / e2p)[1].real:.4f}"
                    f", predicted {(rest_p / e2p)[0].real:.4f} | {(rest_p / e2p)[1].real:.4f}",
                    flush=True,
                )
    # the whole nonlinear part at the full contrast: in the local limit the response is int f / (1 - c f)
    # and the scheme's int Pi f / (1 - c Pi f)
    print(
        "\nthe whole nonlinear part at the full contrast, u - T1 (omega = 30): error / E2, "
        "measured | predicted"
    )
    omega = 30.0
    ex = series(lambda s: mlb.exact(omega, s))
    u_ex = mlb.exact(omega, 1.0)
    for p in (0, 1):
        for n in (4, 8, 16):
            sc = series(lambda s, _n=n, _p=p: mlb.scattered(omega, _n, _p, _p, s))
            u_sc = mlb.scattered(omega, n, p, p, 1.0)
            nl_ex, nl_sc = u_ex - ex[0], u_sc - sc[0]
            h = D_LAYER / (2 * n)
            zs, w = gauss(-h, h)
            prof = cg.PROFILES[mlb.PROFILE][0]
            vals, fits = [], []
            for zc in (np.arange(n) + 0.5) * 2 * h:
                v = prof(zc + zs)
                vals.append(v)
                fits.append(
                    sum(
                        (2 * c + 1) / (2 * h) * np.sum(w * v * cg.leg(c, zs / h)) * cg.leg(c, zs / h)
                        for c in range(p + 1)
                    )
                )
            vals, fits = np.concatenate(vals), np.concatenate(fits)
            ww = np.tile(w, n)
            # the local strain-strain term of the kernel, c = -dM / M_P, from the contrast: nothing fitted
            c_k = -(CONTRAST[0] + 2 * CONTRAST[1]) / M_P
            c_check = (ex[2] / ex[1])[0].real * np.sum(ww * vals**2) / np.sum(ww * vals**3)
            phi = lambda x, _c=c_k: _c * x**2 / (1 - _c * x)  # noqa: E731
            pred = (np.sum(ww * phi(vals)) - np.sum(ww * phi(fits))) / np.sum(ww * phi(vals))
            # the scheme in the local limit: in each cell the Galerkin solution of eps = 1 + c Pi f eps on
            # the Legendre polynomials of degree p (its intermediate fields projected), nonlinear part
            # int Pi f (eps - 1); for p = 0 this is int phi(Pi f)
            scheme_nl = 0.0
            for k in range(n):
                v_k, f_k, w_k = vals[k * len(w) : (k + 1) * len(w)], fits[k * len(w) : (k + 1) * len(w)], w
                basis = np.array([cg.leg(a, zs / h) for a in range(p + 1)])
                gram = basis @ (w_k[:, None] * basis.T)
                amat = basis @ ((w_k * f_k)[:, None] * basis.T)
                rhs = basis @ w_k
                coef = np.linalg.solve(gram - c_k * amat, rhs)
                eps = coef @ basis
                scheme_nl += float(np.sum(w_k * f_k * (eps - 1.0)))
            exact_nl = float(np.sum(ww * vals * (1.0 / (1.0 - c_k * vals) - 1.0)))
            pred_galerkin = (exact_nl - scheme_nl) / exact_nl
            e2p = (np.sum(ww * vals**2) - np.sum(ww * fits**2)) / np.sum(ww * vals**2)
            meas = (nl_ex - nl_sc) / nl_ex
            print(
                f"  p = r = {p}, n = {n:2d}:  {(meas / e2p)[0].real:.4f} | {(meas / e2p)[1].real:.4f}"
                f"   predicted {pred / e2p:.4f}, cell Galerkin {pred_galerkin / e2p:.4f}"
                f"   (c = {c_k:.5f}; from T3/T2 {c_check:.5f})",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
