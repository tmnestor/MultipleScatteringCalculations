#!/usr/bin/env python3
"""GATE: the GMM marine stack against the Kennett recursion.

THE STACK. Free surface | ocean | two elastic layers | homogeneous half-space.
Two elastic layers rather than one on purpose: a single interface cannot
exercise interbed multiples, so a recursion that handles one interface
correctly and compounds them wrongly would pass.

THE OBJECT. The plane-wave PP reflection response R(p, omega) referenced at the
FREE SURFACE -- the full marine response, water column and all. That is the
object that actually exercises the water reverberation, and the water thickness
appears in it explicitly.

WHY PP ONLY, and why that is not a dodge. The two constructions normalise their
mode amplitudes differently -- `kennett_layers` and `psv_fluid_solid` use the
flux convention `sqrt(eta*rho)`, the GMM its own eigenvectors. Those differ by a
DIAGONAL similarity, and a diagonal similarity leaves every diagonal entry
exactly invariant. Comparing the PP component is therefore convention-free,
where comparing P<->SV conversions would require the flux factor first (see
`gate_dg0_absolute_magnitude` [M3] for that factor and what it costs to miss).

THE TWO COMPARISONS, the first sharper than the second:

  [M1] R at the top of the water, free surface NOT applied. Tests the seabed
       fluid-solid step, the interbed multiples, and the one-way-down/one-way-up
       water leg, with no reverberation series on top to mask an error.

  [M2] R at the free surface. Adds the pressure-release series R -> R/(1+R),
       with R_fs = -1. Both sides close it the same way -- the GMM by its
       `free_surface` flag, Kennett by `seismic_survey.free_surface_reverberations`
       -- so this checks that the closure is applied to the same underlying R.

  [M3] WATER-THICKNESS CONTROL, mandatory. The response must change with the
       water thickness on BOTH sides. This is the property the Green's-function
       path of the same module fails: `riccati_greens_psv` treats the ocean as a
       half-space, so `corrected_layered_9x9` and the 3-D table inherit no water
       column and no free surface at all (`gate_free_surface_reverberation`
       [F4]). A comparison that happened to use one water thickness would not
       see that, and would certify a stack that has no ocean in it.

KENNETT ASSEMBLY, and the reference-depth trap it must avoid. `kennett_layers`
references RD at the first INTERFACE, not at the top of the stack: its recursion
phases the layer BELOW each interface, so the leading layer's own two-way delay
is excluded. Feeding the sub-ocean layers directly therefore gives an RD short by
one layer's delay, and because that shifts P and SV alike it reads as a physics
failure rather than a bookkeeping one. A zero-thickness layer of the first
solid's own medium is prepended so RD refers to the seabed.

Run:  conda run -n seismic python scripts/gate_gmm_marine_stack.py
Seismic units (km, km/s, g/cm3) throughout, BOTH sides.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "PhD_fortran_code"))
sys.path.insert(0, "/Users/tod/Desktop/SeismicInversion")

from cubic_scattering.kennett_layers import (  # noqa: E402
    IsotropicLayer,
    LayerStack,
    _vertical_slowness,
    kennett_layers,
    psv_fluid_solid,
)
from GlobalMatrix.global_matrix import gmm_reflectivity  # noqa: E402

# Free surface | ocean | elastic 1 | elastic 2 | half-space
W_ALPHA, W_RHO = 1.5, 1.03
L1 = (4.0, 2.30, 2.20, 2.0)  # alpha, beta, rho, thickness
L2 = (5.5, 3.20, 2.60, 3.0)
HS = (6.5, 3.80, 3.00, np.inf)
Q = 1e6  # near-lossless: the reverberation must be visible, not damped away


def gmm_model(water_thick: float):
    """The stack as a LayerModel: layer 0 is the ocean (beta = 0)."""
    import Kennett_Reflectivity.layer_model as lm

    return lm.LayerModel.from_arrays(
        alpha=[W_ALPHA, L1[0], L2[0], HS[0]],
        beta=[0.0, L1[1], L2[1], HS[1]],
        rho=[W_RHO, L1[2], L2[2], HS[2]],
        thickness=[water_thick, L1[3], L2[3], HS[3]],
        Q_alpha=[Q] * 4,
        Q_beta=[1e12, Q, Q, Q],
    )


def kennett_response(water_thick: float, p: float, om: np.ndarray) -> tuple:
    """(R at top of water, R at the free surface) by the Kennett route.

    Returns both so [M1] can be read without the reverberation series on top.

    THE WATER SLOWNESS IS TAKEN FROM THE MODEL, not rebuilt from the real
    velocity. Both sides must attenuate the water identically or the two-way
    water phase differs by omega * h_w * eta_w / (2Q), which shows up as a
    residual proportional to omega and to the water thickness and nearly
    independent of p -- measured at 1e-5 for Q = 1e6 before this was fixed.
    """
    # Sub-ocean solid stack, RD referenced AT THE SEABED via the zero-thickness
    # leading layer -- see the reference-depth trap in the module docstring.
    stack = LayerStack(
        [
            IsotropicLayer(L1[0], L1[1], L1[2], 1e-9, Q, Q),
            IsotropicLayer(L1[0], L1[1], L1[2], L1[3], Q, Q),
            IsotropicLayer(L2[0], L2[1], L2[2], L2[3], Q, Q),
            IsotropicLayer(HS[0], HS[1], HS[2], np.inf, Q, Q),
        ]
    )
    rd = kennett_layers(stack, p, om).RD_psv  # (nfreq, 2, 2), flux convention

    # Fluid-solid seabed, then the water reverberation:
    #   R = Rd + Tu . RD . (I - Ru . RD)^-1 . Td
    # mirroring ocean_bottom._kennett_water_step, which needs an
    # OceanBottomConfig this gate has no other use for.
    mod = gmm_model(water_thick)
    s_p, s_s = mod.complex_slowness_p(), mod.complex_slowness_s()
    eta_w = _vertical_slowness(s_p[0], p)
    eta_1 = _vertical_slowness(s_p[1], p)
    neta_1 = _vertical_slowness(s_s[1], p)
    coeff = psv_fluid_solid(p, eta_w, W_RHO, eta_1, neta_1, L1[2], 1.0 / s_s[1])

    eye = np.eye(2, dtype=complex)
    r_seabed = np.empty(om.size, dtype=complex)
    for i in range(om.size):
        u = np.linalg.inv(eye - coeff.Ru @ rd[i])
        r_seabed[i] = (coeff.Rd + coeff.Tu @ rd[i] @ u @ coeff.Td)[0, 0]

    e2 = np.exp(2j * om * eta_w * water_thick)  # two-way water phase
    r_top = e2 * r_seabed
    return r_top, r_top / (1.0 + r_top)


def main() -> int:
    print("=" * 78)
    print("GATE -- GMM marine stack vs the Kennett recursion")
    print("  free surface | ocean | 2 elastic layers | half-space")
    print(f"  water a={W_ALPHA} rho={W_RHO}; L1={L1}; L2={L2}; HS={HS[:3]}")
    print("  PP component only -- convention-free under a diagonal similarity")
    print("=" * 78)

    om = 2 * np.pi * np.array([3.0, 5.0, 8.0])
    ok1 = ok2 = True
    worst1 = worst2 = 0.0

    for water in (1.0, 2.5):
        print(f"\n  water thickness {water} km")
        print(f"    {'f (Hz)':>7} {'p':>7} {'[M1] top of water':>19} {'[M2] free surface':>19}")
        for p in (0.02, 0.08, 0.14):
            g_raw = gmm_reflectivity(gmm_model(water), p, om, free_surface=False)
            g_fs = gmm_reflectivity(gmm_model(water), p, om, free_surface=True)
            k_raw, k_fs = kennett_response(water, p, om)
            for i, f in enumerate(om / (2 * np.pi)):
                d1 = abs(g_raw[i] - k_raw[i]) / max(abs(k_raw[i]), 1e-300)
                d2 = abs(g_fs[i] - k_fs[i]) / max(abs(k_fs[i]), 1e-300)
                worst1, worst2 = max(worst1, d1), max(worst2, d2)
                ok1 = ok1 and d1 < 1e-8
                ok2 = ok2 and d2 < 1e-8
                print(f"    {f:7.1f} {p:7.2f} {d1:19.3e} {d2:19.3e}")

    print(f"\n  [M1] top of water, no free surface : worst {worst1:.3e}")
    print(f"  [M2] at the free surface           : worst {worst2:.3e}")

    print("\n  [M3] water-thickness control -- the response MUST move")
    a = gmm_reflectivity(gmm_model(1.0), 0.08, om, free_surface=True)
    b = gmm_reflectivity(gmm_model(2.5), 0.08, om, free_surface=True)
    moves_gmm = float(np.abs(a - b).max() / np.abs(a).max())
    ka = kennett_response(1.0, 0.08, om)[1]
    kb = kennett_response(2.5, 0.08, om)[1]
    moves_ken = float(np.abs(ka - kb).max() / np.abs(ka).max())
    print(f"    GMM     moves by {moves_gmm:.3e}")
    print(f"    Kennett moves by {moves_ken:.3e}")
    ok3 = moves_gmm > 1e-2 and moves_ken > 1e-2
    print(f"    [M3] both carry a real water column : {'PASS' if ok3 else 'FAIL'}")

    ok = ok1 and ok2 and ok3
    print("\n" + "=" * 78)
    print(f"GATE GMM marine stack: {'PASS' if ok else 'FAIL'}")
    if not ok1:
        print("  [M1] failed -- the disagreement is in the stack itself, before any")
        print("  free-surface closure. Check the reference depth first (the")
        print("  zero-thickness leading layer), then the fluid-solid step, then")
        print("  the sign of the two-way water phase.")
    elif not ok2:
        print("  [M1] passed but [M2] failed -- the underlying R agrees and only")
        print("  the pressure-release closure differs. One side is applying")
        print("  R/(1+R) to a differently-referenced R.")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
