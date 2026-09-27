#!/usr/bin/env python3
"""Gate: the stratified reference of notebook 13 against the package's Kennett recursion.

``Mathematica/ContinuumLimit_Heterogeneous.wl`` (notebook 13) builds the exact reflection of a layer of
eight random model cells as a product of propagator matrices, cell by cell. The package's
``kennett_layers`` computes the same object by a different algorithm: the Kennett reflection-transmission
recursion over interfaces. The two share only the model.

Compared are the SAME-TYPE coefficients, which do not depend on how the up- and downgoing waves are
normalised (displacement or energy flux), because incident and reflected wave are the same wave in the
same medium:
  [1] normal incidence, R_PP;
  [2] normal incidence, R_SS (the 1-D S model is the SH recursion at p = 0);
  [3] P at 20 degrees, R_PP.
The gate accepts a sign of +-1 and reports it. One is expected: the 1-D model's R is the ratio of
VERTICAL displacements, whereas Kennett (like notebook 9's oblique solve) refers amplitudes to direction
vectors, and an upgoing P wave at normal incidence points in -z -- so R_PP at p = 0 differs by exactly -1,
and nowhere else. Converted coefficients (R_PS) are not compared here, because their value depends on the
normalisation.

Run:  conda run -n seismic python scripts/gate_heterogeneous_reference_vs_kennett.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubic_scattering.kennett_layers import IsotropicLayer, LayerStack, kennett_layers  # noqa: E402

REF = ROOT / "Mathematica" / "ContinuumLimit_heterogeneous_ref.json"
TOL = 1e-10


def main() -> int:
    ref = json.loads(REF.read_text())
    al, be, rho, om, dc = ref["alpha"], ref["beta"], ref["rho"], ref["omega"], ref["cell_thickness"]
    mu, lam = rho * be**2, rho * al**2 - 2 * rho * be**2
    layers = [IsotropicLayer(al, be, rho, 100.0)]
    for dl, dm, dr in ref["cells_dlambda_dmu_drho"]:
        m1, l1, r1 = mu + dm, lam + dl, rho + dr
        layers.append(IsotropicLayer(np.sqrt((l1 + 2 * m1) / r1), np.sqrt(m1 / r1), r1, dc))
    layers.append(IsotropicLayer(al, be, rho, np.inf))
    stack = LayerStack(layers)

    def cplx(v: list) -> complex:
        return complex(v[0], v[1])

    k0 = kennett_layers(stack, 0.0, np.array([om]))
    p20 = np.sin(np.deg2rad(20.0)) / al
    k20 = kennett_layers(stack, p20, np.array([om]))
    pairs = [
        ("[1] normal incidence R_PP", cplx(ref["exact_normal_P_RT"][0]), complex(k0.RD_psv[0][0, 0])),
        ("[2] normal incidence R_SS", cplx(ref["exact_normal_S_RT"][0]), complex(k0.RD_sh[0])),
        (
            "[3] P at 20 deg, R_PP",
            cplx(ref["exact_oblique_20deg_RPP_RPS"][0]),
            complex(k20.RD_psv[0][0, 0]),
        ),
    ]
    print("=" * 84)
    print("  The stratified reference (propagators, Mathematica) against Kennett's recursion (Python)")
    print("=" * 84)
    oks = []
    for label, mm, kn in pairs:
        sign = 1 if abs(mm - kn) <= abs(mm + kn) else -1
        err = abs(mm - sign * kn) / abs(mm)
        oks.append(err < TOL)
        print(
            f"  {'PASS' if err < TOL else 'FAIL'}  {label}: propagator {mm:.12f}   Kennett {kn:.12f}"
            f"   (sign {sign:+d})   rel. diff {err:.2e}"
        )
    verdict = f"ALL {len(oks)} CHECKS PASS" if all(oks) else "CHECKS FAILED"
    print(f"==== gate_heterogeneous_reference_vs_kennett: {verdict}")
    return 0 if all(oks) else 1


if __name__ == "__main__":
    sys.exit(main())
