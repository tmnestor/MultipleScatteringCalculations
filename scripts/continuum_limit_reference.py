#!/usr/bin/env python3
"""Reference numbers for ``Mathematica/ContinuumLimit_Reference.wl``, notebook 1 of the continuum limit.

The continuum-limit notebooks derive, independently, what the package computes. This dumps the package's
side at normal incidence (k_par -> 0), where a vertical plane force drives a pure 1-D P problem and a
horizontal one a pure 1-D S problem:

  * the whole-space specular kernels -- ``vertical_kernel_9x9`` (dz != 0) and ``same_depth_kernel_9x9``
    (dz = 0) -- at several dz;
  * the layered propagator ``corrected_layered_9x9`` for a contrast layer in a uniform solid below a
    matched fluid layer 0 (the refinement study's "exact" answer for its control arm), source 12 m
    above the layer, receivers above, inside and below it;
  * the complex slownesses and densities of the two media, so both routes use identical attenuation.

Every 9x9 is dumped whole (rows and columns in the package's order: u_z, u_x, u_y, e_zz, e_xx, e_yy,
2e_xy, 2e_zy, 2e_zx), as [re, im] pairs.

Run:  conda run -n seismic python scripts/continuum_limit_reference.py
Writes Mathematica/ContinuumLimit_reference.json.  SI units.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "PhD_fortran_code"))
sys.path.insert(0, "/Users/tod/Desktop/SeismicInversion")

from cubic_scattering.pair_propagators import plane_reference_medium  # noqa: E402
from cubic_scattering.sweep_kernels import same_depth_kernel_9x9, vertical_kernel_9x9  # noqa: E402
from scripts import measure_thesis_refinement as refine  # noqa: E402

OUT = ROOT / "Mathematica" / "ContinuumLimit_reference.json"
N = 4  # sublayers of D/(2N) = 0.25 m: receivers on a 0.25 m grid
DZ_WHOLE = (0.25, 1.0, 5.0)
Z_RECEIVERS = (-2.75, -1.25, -0.25, 0.25, 0.75, 1.25, 1.75)  # above, inside; the half-space starts at D = 2


def reim(a: np.ndarray) -> list:
    """Complex array -> nested [re, im]."""
    return np.stack([np.real(a), np.imag(a)], axis=-1).tolist()


def main() -> int:
    """Dump the reference.

    Returns:
        0.
    """
    geo = refine.Geometry(N)
    # Layer 0 is a fluid with the background's alpha and rho: transparent to P at normal incidence,
    # but a traction-free boundary for S at interface 0 (Mathematica/OceanBoundary.wl).
    m_bg = geo.model(contrast=False, uniform=True)
    m_full = geo.model(contrast=True, uniform=True)
    src = geo.iface(refine.Z_SRC)
    bg = plane_reference_medium(m_bg, geo.iface(-1.0))
    inside = plane_reference_medium(m_full, geo.iface(1.0))
    s_p, s_s = m_full.complex_slowness_p(), m_full.complex_slowness_s()
    j_in = geo.iface(1.0)
    out = {
        "omega": refine.OM,
        "eps_kpar": refine.EPS,
        "D": refine.D_SLAB,
        "z_src": refine.Z_SRC,
        # interface 0, the fluid-solid boundary: interface 1 sits at the source, layer 1 above it
        "z_ocean": refine.Z_SRC - float(m_full.thickness[1]),
        "background": {
            "alpha": [bg.alpha.real, bg.alpha.imag],
            "beta": [bg.beta.real, bg.beta.imag],
            "rho": bg.rho,
        },
        "layer": {
            "alpha": [inside.alpha.real, inside.alpha.imag],
            "beta": [inside.beta.real, inside.beta.imag],
            "rho": inside.rho,
            "slowness_p": [s_p[j_in].real, s_p[j_in].imag],
            "slowness_s": [s_s[j_in].real, s_s[j_in].imag],
        },
        "contrast_SI": {"dlambda": refine.gate.D_LAM, "dmu": refine.gate.D_MU, "drho": refine.gate.D_RHO},
    }
    whole: list = []
    layered: list = []
    for dz in DZ_WHOLE:
        k = vertical_kernel_9x9(np.array([refine.EPS]), 0.0, dz, refine.OM, bg)[:, :, 0]
        whole.append({"dz": dz, "G": reim(k)})
    k0 = same_depth_kernel_9x9(np.array([refine.EPS]), 0.0, refine.OM, bg)[:, :, 0]
    whole.append({"dz": 0.0, "G": reim(k0)})
    for z in Z_RECEIVERS:
        g = refine.p_tilde(m_full, src, geo.iface(z))
        g0 = refine.p_tilde(m_bg, src, geo.iface(z))
        layered.append({"z": z, "G_layer": reim(g), "G_background": reim(g0)})
    out["whole_space"], out["layered"] = whole, layered
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}: {len(whole)} whole-space kernels, {len(layered)} receivers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
