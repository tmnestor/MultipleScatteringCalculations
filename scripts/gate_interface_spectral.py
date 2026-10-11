"""Gate: the frequency-sweep form of the interface blocks, ``cubic_scattering.interface_spectral``.

Paper: LatexPDFs/InterfaceCouplingIntegrals, Section "The next term: the order omega^2".

K(omega) = D(omega) + A0 + omega^2 A1, D the plane-wave integral to Q0 in double precision, A0 and A1 computed
once (the static block in closed form less its plane waves below Q0; the omega^2 term above Q0).

S1  THE FRAME KERNELS.  The double-precision Laurent form of the static and omega^2 frame kernels (vertical
    moments included, field 1, source 1) against the 40-digit kernels of
    scripts/measure_interface_omega2_subtraction.py, at q h = 0.5 .. 160.  Measured, 11 October 2026:
        R: Y0 1.7e-14, Y1 8.9e-13;  T: Y0 1.9e-14, Y1 1.2e-12  (Y1 enters multiplied by omega^2).
S2  IDENTICAL MEDIA, THE WHOLE BLOCK.  With A = B the transmitted coupling is the homogeneous one, so a block
    of two cells touching across the plane must be Paper 2's ``coupling_block`` (closed forms, no plane waves,
    no interface code), every field function and source monomial (4 x 10 x 9 x 9).  Measured:
        offset (1,0,0): 3.0e-10 (omega h/beta = 1/6, Q0 h = 10), 5.4e-11 (omega h/beta = 1, Q0 h = 40);
        offset (1,1,0): 3.2e-10, 3.9e-11;   offset (1,1,1): 7.7e-11, 1.7e-12.
    The source in B (the mirror), offset (-1,0,0): run below.
S3  DIFFERENT MEDIA, AGAINST 40 DIGITS.  Entry (field 1, source 1) of the reflected block of a touching cell
    with its own image, and of the transmitted block of two cells touching across the interface, against
    K0 + the 40-digit plane-wave integral of Y - Y0 to q h = 320 (scripts/data/interface_omega2_reference_*.json)
    + its tail beyond 320 from the last octave's Y1 and remainder (the static plane waves below q h = 10, which
    the reference does not hold, from the package's static integrand, checked in S1).  Expected: the dropped remainder, 4e-10
    (R) and 2e-10 (T) at omega h/beta = 1/6, Q0 h = 10; 6e-11 and 4e-11 at omega h/beta = 1, Q0 h = 40.

By default S1, S2 at offset (1,0,0) and its mirror, and S3 run (about 10 minutes); ``--full`` adds the
lateral neighbours of S2.

Run:  PYTHONPATH=.:scripts python scripts/gate_interface_spectral.py [--full]
"""

import json
import sys
import time
from pathlib import Path

import mpmath as mp
import numpy as np

from cubic_scattering import interface_spectral as isp
from cubic_scattering.effective_contrasts import ReferenceMedium
from cubic_scattering.graded_voxel.blocks import coupling_block
from cubic_scattering.interface_image import image_block

A = ReferenceMedium(5.0, 3.0, 2.5)
B = ReferenceMedium(6.5, 3.6, 2.8)
H = 0.05
CASES = [(10.0, 10.0), (60.0, 40.0)]  # (omega, Q0 h): omega h / beta = 1/6 and 1
DATA = Path(__file__).resolve().parent / "data"


def gate_frames() -> bool:
    import measure_interface_omega2_subtraction as m

    ok = True
    for kind in "RT":
        m.KIND = kind
        m._KERNELS.clear()
        m.build()
        s = 1 if kind == "R" else -1
        for order, name, tol in ((0, "Y0", 1e-13), (1, "Y1", 1e-11)):
            C, m0 = isp.frame_laurent(kind, order, isp._key(A), isp._key(B))
            worst = 0.0
            for qh in (0.5, 3.0, 10.0, 40.0, 160.0):
                q = np.array([qh / H])
                zr = -H if kind == "R" else H
                vr = isp._vertical_moments(np.array([1.0]), zr, H, s, q, C.shape[2] - 1)
                vs = isp._vertical_moments(np.array([1.0]), -H, H, 1, q, C.shape[3] - 1)
                qp = q[None, :] ** (m0 + np.arange(C.shape[4]))[:, None]
                f = np.einsum("ijabm,mn,an,bn->ij", C, qp, vr, vs)
                mp.mp.dps = 40
                ref = m.frame(name, mp.mpf(qh) / H)
                worst = max(worst, np.abs(f - ref).max() / np.abs(ref).max())
            good = worst < tol
            ok &= good
            print(
                f"S1 {kind} {name}: {worst:.1e}  {'PASS' if good else 'FAIL'}",
                flush=True,
            )
    return ok


def gate_identical(rec, src, tol=(1e-9, 2e-10)) -> bool:
    off = tuple((r - s) // 2 for r, s in zip(rec, src, strict=True))
    ok = True
    for (om, q0), t in zip(CASES, tol, strict=True):
        t0 = time.time()
        parts = isp.frequency_independent_parts(rec, src, H, A, A, q0h=q0)
        t1 = time.time()
        k = isp.interface_block(rec, src, H, om, A, A, q0h=q0, parts=parts)
        t2 = time.time()
        ref = coupling_block(off, H, om, A)
        err = np.abs(k - ref).max() / np.abs(ref).max()
        good = err < t
        ok &= good
        print(
            f"S2 {rec} <- {src}, omega h/beta = {om * H / A.beta:.3f}, Q0 h = {q0:g}: {err:.1e}"
            f"  (once {t1 - t0:.0f} s, per frequency {t2 - t1:.1f} s)  {'PASS' if good else 'FAIL'}",
            flush=True,
        )
    return ok


def _unpack(d):
    return np.array(d["re"]) + 1j * np.array(d["im"])


def gate_reference(kind: str, tol=(1e-9, 2e-10)) -> bool:
    path = DATA / f"interface_omega2_reference_{kind}.json"
    if not path.exists():
        print(
            f"S3 {kind}: {path.name} missing -- run measure_interface_omega2_subtraction.py {kind} (SKIPPED)"
        )
        return False
    ref = json.loads(path.read_text())
    rec, src = ((-1, 0, 0), (-1, 0, 0)) if kind == "R" else ((1, 0, 0), (-1, 0, 0))
    k0 = image_block(rec, src, H, A, B, n_source=1, n_test=1)[0, 0]
    ok = True
    for (om, q0), t in zip(CASES, tol, strict=True):
        sh = ref["shells"]
        low = _unpack(ref["low"][str(om)])
        y = sum(_unpack(s[str(om)]) - _unpack(s["Y0"]) for s in sh)
        last = sh[-1]
        rem = _unpack(last[str(om)]) - _unpack(last["Y0"]) - om**2 * _unpack(last["Y1"])
        tail = om**2 * _unpack(last["Y1"]) / 15 + rem / 63
        # the reference's low part is of Y; Y0 below q h = 10 is in K0 - (int Y0 beyond 10), so subtract
        # Y0 over q h < 10 the same way the package does
        qh, w = isp._panels(0.0, 10.0, np.pi / 2, 16)
        y0_low = isp._static_integral(
            0, kind, rec, src, H, isp._key(A), isp._key(B), qh, w, 1, 1
        )[0, 0]
        full = k0 - y0_low + low + y + tail
        pkg = isp.interface_block(rec, src, H, om, A, B, n_source=1, n_test=1, q0h=q0)[
            0, 0
        ]
        err = np.abs(pkg - full).max() / np.abs(full).max()
        good = err < t
        ok &= good
        print(
            f"S3 {kind}, omega h/beta = {om * H / A.beta:.3f}, Q0 h = {q0:g}: {err:.1e}  {'PASS' if good else 'FAIL'}",
            flush=True,
        )
    return ok


def main() -> None:
    full = "--full" in sys.argv
    ok = gate_frames()
    ok &= gate_identical((1, 0, 0), (-1, 0, 0))
    ok &= gate_identical((-1, 0, 0), (1, 0, 0))
    if full:
        ok &= gate_identical((1, 2, 0), (-1, 0, 0))
        ok &= gate_identical((1, 2, 2), (-1, 0, 0))
    ok &= gate_reference("R")
    ok &= gate_reference("T")
    print("ALL PASS" if ok else "SOME FAILED")


if __name__ == "__main__":
    main()
