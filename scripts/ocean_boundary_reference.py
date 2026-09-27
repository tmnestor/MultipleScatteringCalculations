#!/usr/bin/env python3
"""Reference numbers for ``Mathematica/OceanBoundary.wl``: the layered propagator under a fluid layer 0.

``GlobalMatrix.layered_greens`` treats layer 0 as an OCEAN whatever ``beta[0]`` says: SH sees a
traction-free boundary at interface 0, and P-SV a fluid-solid interface. The notebook derives the
exact response of a uniform solid below a fluid half-space with the SAME P velocity and density, and
compares it with this dump.

Model: layer 0 fluid (beta = 0), then uniform solid layers of 1.25 m down to a half-space, all with the
same alpha and rho. Interface j is the BASE of layer j, so interface 0 is the fluid-solid boundary.
Every 9x9 in the order (u_z, u_x, u_y, e_zz, e_xx, e_yy, 2e_xy, 2e_zy, 2e_zx), [re, im].

Run:  conda run -n seismic python scripts/ocean_boundary_reference.py
Writes Mathematica/OceanBoundary_reference.json. SI units.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "PhD_fortran_code"))
sys.path.insert(0, "/Users/tod/Desktop/SeismicInversion")

import cubic_scattering.layered_correction as LC  # noqa: E402
from Kennett_Reflectivity.layer_model import LayerModel  # noqa: E402

OUT = ROOT / "Mathematica" / "OceanBoundary_reference.json"
A0, B0, R0 = 5000.0, 3000.0, 2500.0
OM = 60.0
Q = 1e4
H = 1.25
N_LAY = 16
PAIRS = ((2, 8), (8, 2), (5, 5), (3, 11))
K_FRACS = (1e-6, 0.05, 0.3, 0.7, 1.5)  # of k_S; 1.5 is evanescent for P and S


def reim(a: np.ndarray) -> list:
    """Complex array -> nested [re, im]."""
    return np.stack([np.real(a), np.imag(a)], axis=-1).tolist()


def main() -> int:
    """Dump the reference.

    Returns:
        0.
    """
    model = LayerModel.from_arrays(
        alpha=[A0] * N_LAY,
        beta=[0.0, *([B0] * (N_LAY - 1))],
        rho=[R0] * N_LAY,
        thickness=[3000.0, *([H] * (N_LAY - 2)), np.inf],
        Q_alpha=[Q] * N_LAY,
        Q_beta=[1e10, *([Q] * (N_LAY - 1))],
    )
    s_p, s_s = model.complex_slowness_p(), model.complex_slowness_s()
    k_s = OM / B0
    cases: list = []
    for src, rcv in PAIRS:
        for frac in K_FRACS:
            kx = frac * k_s
            g = LC.corrected_layered_9x9(model, OM, np.array([kx]), np.array([0.0]), src, rcv)[0]
            cases.append({"src": src, "rcv": rcv, "kx": kx, "G": reim(g)})
    out = {
        "omega": OM,
        "rho": R0,
        "alpha_c": reim(np.asarray(1.0 / s_p[1])),
        "beta_c": reim(np.asarray(1.0 / s_s[1])),
        "alpha_fluid_c": reim(np.asarray(1.0 / s_p[0])),
        "h": H,
        "note": "interface j at depth j*h below the fluid-solid boundary (interface 0)",
        "cases": cases,
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}: {len(cases)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
