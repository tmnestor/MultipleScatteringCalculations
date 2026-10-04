#!/usr/bin/env python3
"""The Galerkin coupling blocks of touching cells: closed forms against quadrature, in accuracy and cost.

For the self cell and its touching neighbours the double average over two cubes that share a face, an
edge or a corner has a singular integrand, and quadrature is the usual weak point. Three routes:

  quadrature  ``blocks.near_block(static="quadrature", n_q)``: the static (Kelvin) part, a
              distribution, by Duffy pyramids and Gauss rules of order n_q on the pieces of the s-form; the
              dynamic remainder, at most 1/r, by the same rules;
  closed      ``near_block(static="closed", n_q)``: the static part in closed form
              (``static_term_integral_closed``: Legendre modified moments away from the singular point,
              Duffy pyramids at it), the dynamic remainder still by quadrature;
  series      ``blocks.near_block_series``: the whole block as the power series of the propagator in the
              wavenumber, every term a universal moment in closed form; no quadrature at all.

The series is the reference: its terms are exact (the closed forms for odd powers of r, exact
polynomial quadrature for even ones), evaluated to round-off, and summed to 1e-17. It is checked against
the closed route at the highest n_q. Printed per geometry, cell degree and frequency: the relative
error (Frobenius) of each route and its cost, cold. The series' universal moments depend on neither
the frequency nor the cell size, so they are computed once (reported as the one-off cost) and its
per-block cost is the assembly.

Run:  conda run -n seismic python -u scripts/measure_galerkin_near_blocks.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from cubic_scattering.graded_voxel import blocks  # noqa: E402

REF = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
H = 1.25
OFFSETS = {"self": (0, 0, 0), "face": (1, 0, 0), "edge": (1, 1, 0), "corner": (1, 1, 1)}
CELLS = {"linear": (10, 4), "quadratic": (35, 10)}
N_Q = (4, 6, 8, 10, 14, 20)


def clear() -> None:
    blocks._NEAR_CACHE.clear()


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


def rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main() -> int:
    # the one-off cost: every universal moment the series needs, for both cell degrees
    blocks.universal_moment.cache_clear()
    t0 = time.perf_counter()
    for n_s, n_t in CELLS.values():
        for off in OFFSETS.values():
            blocks.near_block_series(off, H, 150.0, REF, n_source=n_s, n_test=n_t)
    dt = time.perf_counter() - t0
    print(f"universal moments for every touching geometry and both degrees (one-off): {dt:.1f} s")
    for ks_h in (0.0625, 0.5):
        omega = ks_h * REF.beta / H
        print(f"\nk_S h = {ks_h}")
        for cell, (n_s, n_t) in CELLS.items():
            for name, off in OFFSETS.items():
                ref, t_ser = timed(blocks.near_block_series, off, H, omega, REF, n_source=n_s, n_test=n_t)
                quad = []
                for nq in N_Q:
                    clear()
                    q, t = timed(blocks.near_block, off, H, omega, REF, n_q=nq, n_source=n_s, n_test=n_t)
                    quad.append((nq, rel(q, ref), t))
                clos = []
                for nq in (6, 10, 20):
                    clear()
                    kw = {"n_q": nq, "n_source": n_s, "n_test": n_t, "static": "closed"}
                    c, t = timed(blocks.near_block, off, H, omega, REF, **kw)
                    clos.append((nq, rel(c, ref), t))
                print(
                    f"  {cell:9s} {name:6s}  series {t_ser * 1e3:6.0f} ms | quadrature "
                    + "  ".join(f"n{nq}:{e:.0e}/{t:.2f}s" for nq, e, t in quad)
                    + " | closed static "
                    + "  ".join(f"n{nq}:{e:.0e}/{t:.2f}s" for nq, e, t in clos),
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
