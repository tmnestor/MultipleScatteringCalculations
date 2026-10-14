"""Gate: the omega^4 interface blocks K2 in closed form against the finite part of their plane-wave integral,
for any media.

Plan: docs/2026-10-10-stratified-reference-legendre-cells-3d.md (14 October: the symbolic omega^4 term).
Paper: LatexPDFs/InterfaceCouplingIntegrals, "The next order" (eq:omega4split).

K2 = (1/4 pi^2) FP int Y2 d^2k, Y2 the omega^4 integrand (``interface_spectral``, order 2), its kernel G2 read
from Mathematica/InterfaceOmega4.json with symbolic media; the closed form (``interface_image``, order 2) uses
the canonical terms derived from the same G2 (scripts/derive_interface_image_terms.py ... 2).  The plane-wave
route shares no step with the closed form: the head FP int_{q < q_0} in closed form (``fp_low_integral``, the
Laurent x power-series x Bessel-series route), the body by Gauss panels to q h = 60, beyond which nothing for
cells not touching their images (the integrand falls as e^{-q zeta_min}, zeta_min >= 2 h).

MEDIA.  "gate", (alpha, beta, rho) = (5, 3, 2.5) over (6.5, 3.6, 2.8), the media the omega^4 term was first
derived for (scripts/derive_interface_omega4.py); "other", (4, 2.25, 2) over (7, 4, 3), which only the symbolic
G2 reaches.  Both arrangements, the pairs of gate_interface_omega2_blocks.py; full 4 x 10 x 9 x 9 blocks.

Run:  PYTHONPATH=. python scripts/gate_interface_omega4_blocks.py
"""

import time

import numpy as np

from cubic_scattering import interface_spectral as isp
from cubic_scattering.effective_contrasts import ReferenceMedium
from cubic_scattering.interface_image import image_block

MEDIA = {
    "gate": (ReferenceMedium(5.0, 3.0, 2.5), ReferenceMedium(6.5, 3.6, 2.8)),
    "other": (ReferenceMedium(4.0, 2.25, 2.0), ReferenceMedium(7.0, 4.0, 3.0)),
}
H = 0.05
Q0H = 1.0
PAIRS = [
    ((-3, 0, 0), (-1, 0, 0)),
    ((-3, 2, 0), (-1, 0, 0)),
    ((3, 0, 0), (-1, 0, 0)),
    ((3, 2, 0), (-1, 0, 0)),
]


def main() -> None:
    ok = True
    for name, (a, b) in MEDIA.items():
        ka, kb = isp._key(a), isp._key(b)
        for rec, src in PAIRS:
            kind = "R" if rec[0] < 0 else "T"
            t0 = time.time()
            closed = image_block(rec, src, H, a, b, order=2)
            t1 = time.time()
            head = isp.fp_low_integral(2, kind, rec, src, H, ka, kb, Q0H, 10, 4)
            qh, w = isp._panels(Q0H, 60.0, 0.5, 20)
            body = isp._static_integral(2, kind, rec, src, H, ka, kb, qh, w, 10, 4)
            t2 = time.time()
            ref = head + body
            err = np.abs(closed - ref).max() / np.abs(ref).max()
            good = err < 1e-12
            ok &= good
            print(
                f"{name} {rec} <- {src}: closed form vs FP plane-wave integral {err:.1e}"
                f"  (closed {t1 - t0:.0f} s, plane waves {t2 - t1:.0f} s)  {'PASS' if good else 'FAIL'}",
                flush=True,
            )
    print("ALL PASS" if ok else "SOME FAILED")


if __name__ == "__main__":
    main()
