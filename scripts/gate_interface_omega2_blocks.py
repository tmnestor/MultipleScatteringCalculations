"""Gate: the omega^2 interface blocks K1 in closed form against the finite part of their plane-wave integral.

Plan: docs/2026-10-10-stratified-reference-legendre-cells-3d.md (13 October: K1 in space).
Paper: LatexPDFs/InterfaceCouplingIntegrals, "The potentials as finite parts" and the omega^2 section.

K1 = (1/4 pi^2) FP int Y1 d^2k, Y1 the omega^2 integrand (``interface_spectral``, order 1): the finite part is
the continuation in the power of q, the convention of the potentials Phi_j (``interface_image``, order 1).
This route shares no step with the closed form: no potentials, no master integrals, no corner pieces.

THE FINITE PART.  The radial integrand I(q) (the azimuthal and lateral parts done, ``_static_integral`` with
one node) behaves as q^-2 times an entire function at q -> 0.  On [0, q_s], E(q) = q^2 I(q) is sampled at
Chebyshev nodes; with E = sum_l b_l (q/q_s)^l,
    FP int_0^{q_s} I dq = q_s^-1 [ sum_{l != 1} b_l / (l - 1) + b_1 log q_s ],
each Chebyshev mode's power coefficients exact (rationals).  q in km^-1: a finite part depends on the unit of
length, and the closed form uses km.  On [q_s, Q] Gauss panels; beyond Q nothing for cells whose image is
not touching (the integrand falls as e^{-q zeta_min}, zeta_min >= 2 h).

PAIRS.  Reflected, cells not touching their images: (-3, 0, 0) <- (-1, 0, 0) (zeta >= 2h) and
(-3, 2, 0) <- (-1, 0, 0); full 4 x 10 x 9 x 9 blocks.  Also checked: the head's Chebyshev order and q_s.

Run:  PYTHONPATH=. python scripts/gate_interface_omega2_blocks.py
"""

import time
from functools import cache

import numpy as np
import sympy as sp
from numpy.polynomial import chebyshev as C

from cubic_scattering import interface_spectral as isp
from cubic_scattering.effective_contrasts import ReferenceMedium
from cubic_scattering.interface_image import image_block

A = ReferenceMedium(5.0, 3.0, 2.5)
B = ReferenceMedium(6.5, 3.6, 2.8)
H = 0.05
PAIRS = [((-3, 0, 0), (-1, 0, 0)), ((-3, 2, 0), (-1, 0, 0))]


@cache
def _mode_moments(n: int) -> tuple:
    """For each shifted Chebyshev mode T_k(2t - 1), k < n: (sum_{l != 1} a_kl / (l - 1), a_k1), exact."""
    t = sp.Symbol("t")
    out = []
    for k in range(n):
        co = sp.Poly(sp.chebyshevt(k, 2 * t - 1), t).all_coeffs()[::-1]
        pw = sum(sp.Rational(c) / (l - 1) for l, c in enumerate(co) if l != 1)
        lg = sp.Rational(co[1]) if len(co) > 1 else sp.Integer(0)
        out.append((float(pw), float(lg)))
    return tuple(out)


def radial(rec, src, qh: np.ndarray, n_test=4, n_source=10) -> np.ndarray:
    """I(q) per unit q (km^-1) at the nodes q = qh / h, shape (N, n_test, n_source, 9, 9)."""
    kind = "R" if rec[0] < 0 else "T"
    return np.stack(
        [
            H
            * isp._static_integral(
                1,
                kind,
                rec,
                src,
                H,
                isp._key(A),
                isp._key(B),
                np.array([x]),
                np.array([1.0]),
                n_source,
                n_test,
            )
            for x in qh
        ]
    )


def fp_head(rec, src, qsh: float, n_cheb: int) -> np.ndarray:
    """FP int_0^{q_s} I dq, q_s = qsh / h."""
    qs = qsh / H
    x = np.cos(np.pi * (np.arange(n_cheb) + 0.5) / n_cheb)  # Chebyshev nodes on [-1, 1]
    q = qs * (x + 1) / 2
    e = radial(rec, src, q * H) * (q**2)[:, None, None, None, None]
    coef = np.einsum("kn,n...->k...", np.linalg.inv(C.chebvander(x, n_cheb - 1)), e)
    mom = _mode_moments(n_cheb)
    pw = np.array([m[0] for m in mom])
    lg = np.array([m[1] for m in mom])
    return (
        np.einsum("k,k...->...", pw, coef)
        + np.log(qs) * np.einsum("k,k...->...", lg, coef)
    ) / qs


def body(rec, src, qsh: float, qmax_h: float) -> np.ndarray:
    """int_{q_s}^{Q} I dq by Gauss panels."""
    kind = "R" if rec[0] < 0 else "T"
    qh, w = isp._panels(qsh, qmax_h, 0.5, 20)
    return isp._static_integral(
        1, kind, rec, src, H, isp._key(A), isp._key(B), qh, w, 10, 4
    )


def main() -> None:
    ok = True
    for rec, src in PAIRS:
        t0 = time.time()
        closed = image_block(rec, src, H, A, B, order=1)
        t1 = time.time()
        b = body(rec, src, 0.5, 60.0)
        heads = {
            (qsh, n): fp_head(rec, src, qsh, n) for qsh in (0.5, 0.25) for n in (16, 24)
        }
        t2 = time.time()
        ref = heads[0.5, 24] + b
        scale = np.abs(ref).max()
        err = np.abs(closed - ref).max() / scale
        cheb = np.abs(heads[0.5, 24] - heads[0.5, 16]).max() / scale
        # moving q_s moves [q_s', q_s] from the head to the body; the sum must not change
        mid = isp._static_integral(
            1,
            "R" if rec[0] < 0 else "T",
            rec,
            src,
            H,
            isp._key(A),
            isp._key(B),
            *isp._panels(0.25, 0.5, 0.5, 20),
            10,
            4,
        )
        split = np.abs(heads[0.25, 24] + mid - heads[0.5, 24]).max() / scale
        good = err < 1e-12
        ok &= good
        print(
            f"{rec} <- {src}: closed form vs FP plane-wave integral {err:.1e}  (head: Chebyshev 16 vs 24 {cheb:.1e},"
            f" q_s h 0.25 vs 0.5 {split:.1e}; closed {t1 - t0:.0f} s, plane waves {t2 - t1:.0f} s)"
            f"  {'PASS' if good else 'FAIL'}",
            flush=True,
        )
    print("ALL PASS" if ok else "SOME FAILED")


if __name__ == "__main__":
    main()
