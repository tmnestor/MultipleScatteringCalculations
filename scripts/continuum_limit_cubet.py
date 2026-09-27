#!/usr/bin/env python3
"""Reference numbers for ``Mathematica/ContinuumLimit_CubeT.wl``, notebook 4 of the continuum limit.

For a cube of side d = 1 m at five frequencies (k_S a from 0.01 to 0.3, a = d/2, the validated range):
  * ``cube_self_9x9`` -- the cube's own cell integral of the Green's tensor, the term the exact
    normal-incidence kernel subtracts (built from the T-matrix's Gamma0 and A^c, B^c, C^c);
  * the package's 9x9 cube T-matrix as the Foldy-Lax chain uses it (``compute_slab_tmatrices``), for
    the gate's contrast.
Every 9x9 in the order (u_z, u_x, u_y, e_zz, e_xx, e_yy, 2e_xy, 2e_zy, 2e_zx), [re, im].

Run:  conda run -n seismic python scripts/continuum_limit_cubet.py
Writes Mathematica/ContinuumLimit_cubet.json.  SI units.
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

OUT = ROOT / "Mathematica" / "ContinuumLimit_cubet.json"
REF = ReferenceMedium(5000.0, 3000.0, 2500.0)
D = 1.0
OMEGAS = (60.0, 300.0, 600.0, 1200.0, 1800.0)  # k_S a = 0.01, 0.05, 0.1, 0.2, 0.3
D_LAM, D_MU, D_RHO = 2.0e9, 1.0e9, 100.0  # the gate's contrast


def reim(a: np.ndarray) -> list:
    """Complex array -> nested [re, im]."""
    return np.stack([np.real(a), np.imag(a)], axis=-1).tolist()


def main() -> int:
    """Dump the reference.

    Returns:
        0.
    """
    geom = SlabGeometry(M=1, N_z=1, a=D / 2)
    ones = np.ones((1, 1, 1))
    mat = SlabMaterial(Dlambda=D_LAM * ones, Dmu=D_MU * ones, Drho=D_RHO * ones, ref=REF)
    cases = []
    for om in OMEGAS:
        t9 = compute_slab_tmatrices(geom, mat, om)[0, 0, 0]
        cases.append({"omega": om, "self": reim(cube_self_9x9(D, om, REF)), "T": reim(t9)})
    out = {
        "alpha": REF.alpha,
        "beta": REF.beta,
        "rho": REF.rho,
        "d": D,
        "contrast": {"dlambda": D_LAM, "dmu": D_MU, "drho": D_RHO},
        "cases": cases,
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}: {len(cases)} frequencies")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
