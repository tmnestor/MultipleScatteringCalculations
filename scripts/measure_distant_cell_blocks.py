#!/usr/bin/env python3
"""The Galerkin coupling blocks of cells that do not touch: accuracy and cost of each route.

Routes for an offset o (cells 2h wide, centres 2h o apart):

  gauss       ``blocks.coupling_block``: the s-form with the full kernel, Gauss rules of the package's
              order (14, 10 or 8 points per axis on each of the eight pieces, by distance);
  multipole   ``multipole.far_block_multipole``: the series about the centre separation, the cells'
              polynomial moments exact and the propagator's derivatives at R from spherical Hankel
              functions; refused when (2 sqrt 3 h) / R exceeds 0.6;
  piecewise   ``multipole.piecewise_multipole_block``: the same series about the centre of each piece of
              the s-form, for every pair that does not touch;
  kseries     ``blocks.far_block_series``: the power series in k with Gauss coefficients computed once.

The reference is the s-form with 32 Gauss points per axis on each piece (checked against 24: the
difference is printed). Relative Frobenius error of each route, and its cost per block (warm: the
one-off moments and coefficients excluded and reported separately).

Run:  python -u scripts/measure_distant_cell_blocks.py [linear|quadratic]
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from cubic_scattering.graded_voxel import blocks, multipole  # noqa: E402
from cubic_scattering.graded_voxel.kernel import kernel_9x9  # noqa: E402

REF = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
H = 1.0
OFFSETS = [(2, 0, 0), (2, 1, 1), (3, 0, 0), (4, 3, 2), (8, 0, 0), (16, 0, 0), (32, 0, 0), (64, 0, 0)]
KS_H = (1e-4, 0.0625, 0.5, 2.0)
CELLS = {"linear": (10, 4), "quadratic": (35, 10)}


def sform_block(off, omega, n_q, n_source, n_test):
    out = blocks._sform(off, H, (0, 0, 0), lambda X: kernel_9x9(X, omega, REF).reshape(len(X), 81),
                        n_q, n_source, n_test)
    return blocks._to_field_rows(out.reshape(n_test, n_source, 9, 9))


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


def main(cell: str) -> int:
    n_source, n_test = CELLS[cell]
    print(f"{cell} field ({n_test} x {n_source}); relative error against the s-form at 32 points; "
          "time per block (warm)")
    for ks_h in KS_H:
        omega = ks_h * REF.beta / H
        print(f"k_S h = {ks_h:g}", flush=True)
        for off in OFFSETS:
            ref = sform_block(off, omega, 32, n_source, n_test)
            ref24 = sform_block(off, omega, 24, n_source, n_test)
            row = [f"  o = {str(off):12s} ref 24 vs 32 {rel(ref24, ref):.0e} |"]
            g, tg = timed(blocks.coupling_block, off, H, omega, REF, n_source, n_test)
            row.append(f"gauss {rel(g, ref):.0e}/{tg * 1e3:.0f}ms")
            for name, fn in (("multipole", multipole.far_block_multipole),
                             ("piecewise", multipole.piecewise_multipole_block)):
                try:
                    fn(off, H, omega, REF, n_source=n_source, n_test=n_test, tol=1e-13)  # warm the moments
                    m, tm = timed(fn, off, H, omega, REF, n_source=n_source, n_test=n_test, tol=1e-13)
                    row.append(f"{name} {rel(m, ref):.0e}/{tm * 1e3:.0f}ms")
                except ValueError:
                    row.append(f"{name} refused")
            try:
                blocks.far_block_series(off, H, omega, REF, n_source, n_test)
                k, tk = timed(blocks.far_block_series, off, H, omega, REF, n_source, n_test)
                row.append(f"kseries {rel(k, ref):.0e}/{tk * 1e3:.1f}ms")
            except ValueError:
                row.append("kseries refused")
            print(" ".join(row), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "linear"))
