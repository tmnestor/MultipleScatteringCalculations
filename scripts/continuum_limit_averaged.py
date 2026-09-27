#!/usr/bin/env python3
"""Reference numbers for ``Mathematica/ContinuumLimit_AveragedSums.wl``, notebook 3 of the continuum limit.

The SOURCE-CELL-AVERAGED specular lattice sums S_avg(m) = sum_R <G>(R + m d z^), from the package's
exact route (``build_slab_kernels(..., volume_averaged=True, exact_cell_average=True)``: sinc form
factors per mode between planes, Ewald with an analytic tail within the plane; the self cell is
excluded, it lives in the T-matrix), whose k = 0 block is the full lattice sum, for m = -4..4 at three
pitches; and the same-plane sum near-static at d = 1 for three settings of its two convergence
parameters (near-shell radius, near-shell Gauss order).  Every 9x9 in the order (u_z, u_x, u_y, e_zz,
e_xx, e_yy, 2e_xy, 2e_zy, 2e_zx), [re, im].

Run:  conda run -n seismic python scripts/continuum_limit_averaged.py
Writes Mathematica/ContinuumLimit_averaged.json.  SI units, real (lossless) whole-space background.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubic_scattering.cell_averaged_lattice import averaged_same_plane_9x9  # noqa: E402
from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from cubic_scattering.slab_scattering import SlabGeometry, build_slab_kernels  # noqa: E402

OUT = ROOT / "Mathematica" / "ContinuumLimit_averaged.json"
REF = ReferenceMedium(5000.0, 3000.0, 2500.0)
OM = 60.0
PITCHES = (0.5, 1.0, 2.0)
M_MAX = 4
OM_STATIC = 0.6  # k_S d = 2e-4 at d = 1: the static limit, before the omega -> 0 cancellation bites
STATIC_SETTINGS = ((2, 6), (2, 20), (8, 20))  # (r0_cells, n_gauss); (2, 6) is the default


def reim(a: np.ndarray) -> list:
    """Complex array -> nested [re, im]."""
    return np.stack([np.real(a), np.imag(a)], axis=-1).tolist()


def main() -> int:
    """Dump the reference.

    Returns:
        0.
    """
    lattice: list = []
    for d in PITCHES:
        geom = SlabGeometry(M=1, N_z=M_MAX + 1, a=d / 2)
        kh = build_slab_kernels(
            geom, OM, REF, periodic=True, lattice_ewald=True, volume_averaged=True, exact_cell_average=True
        )
        for k in range(2 * M_MAX + 1):
            lattice.append({"d": d, "m": k - M_MAX, "S": reim(np.asarray(kh[k, 0, 0]))})
    # the same-plane sum, NEAR-STATIC (k d ~ 2e-4), at its two convergence parameters: the near-shell
    # radius r0_cells (beyond it an O(d^2) Taylor tail) and the Gauss order of the near-shell average.
    # (8, 20) takes ~9 minutes.
    static_s0: list = []
    for r0, ng in STATIC_SETTINGS:
        s0 = averaged_same_plane_9x9(1.0, OM_STATIC, REF, np.array([0.0, 0.0]), r0_cells=r0, n_gauss=ng)
        static_s0.append({"r0_cells": r0, "n_gauss": ng, "S": reim(np.asarray(s0))})
        print(f"  static same-plane sum, r0_cells {r0}, n_gauss {ng}", flush=True)
    out = {
        "omega": OM,
        "alpha": REF.alpha,
        "beta": REF.beta,
        "rho": REF.rho,
        "lattice": lattice,
        "omega_static": OM_STATIC,
        "static_s0_d1": static_s0,
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}: {len(lattice)} averaged specular sums")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
