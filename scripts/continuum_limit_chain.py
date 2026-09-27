#!/usr/bin/env python3
"""Reference numbers for ``Mathematica/ContinuumLimit_Chain.wl``, notebook 5 of the continuum limit.

The package's 9x9 cube T-matrix (``compute_slab_tmatrices``) and self term (``cube_self_9x9``) for the
refinement ladder of a D = 2 m layer, d = D/n with n = 1, 2, 4, 8, 16, at three frequencies (k_S a from
0.02 to 0.2 at n = 1), for the gate's contrast.  The notebook assembles the chain itself.
Every 9x9 in the order (u_z, u_x, u_y, e_zz, e_xx, e_yy, 2e_xy, 2e_zy, 2e_zx), [re, im].

Run:  conda run -n seismic python scripts/continuum_limit_chain.py
Writes Mathematica/ContinuumLimit_chain.json.  SI units.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubic_scattering.cell_averaged_lattice import cube_self_9x9  # noqa: E402
from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from cubic_scattering.slab_scattering import (  # noqa: E402
    SlabGeometry,
    SlabMaterial,
    compute_slab_tmatrices,
)

OUT = ROOT / "Mathematica" / "ContinuumLimit_chain.json"
REF = ReferenceMedium(5000.0, 3000.0, 2500.0)
D_LAYER = 2.0
LADDER = (1, 2, 4, 8, 16)
OMEGAS = (60.0, 300.0, 600.0)
D_LAM, D_MU, D_RHO = 2.0e9, 1.0e9, 100.0  # the gate's contrast


def reim(a: np.ndarray) -> list:
    """Complex array -> nested [re, im]."""
    return np.stack([np.real(a), np.imag(a)], axis=-1).tolist()


def main() -> int:
    """Dump the reference.

    Returns:
        0.
    """
    ones = np.ones((1, 1, 1))
    mat = SlabMaterial(Dlambda=D_LAM * ones, Dmu=D_MU * ones, Drho=D_RHO * ones, ref=REF)
    cells = []
    for om in OMEGAS:
        for n in LADDER:
            d = D_LAYER / n
            t9 = compute_slab_tmatrices(SlabGeometry(M=1, N_z=1, a=d / 2), mat, om)[0, 0, 0]
            cells.append({"omega": om, "n": n, "T": reim(t9), "self": reim(cube_self_9x9(d, om, REF))})
    out = {
        "alpha": REF.alpha,
        "beta": REF.beta,
        "rho": REF.rho,
        "D": D_LAYER,
        "contrast": {"dlambda": D_LAM, "dmu": D_MU, "drho": D_RHO},
        "cells": cells,
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}: {len(cells)} (omega, n) cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
