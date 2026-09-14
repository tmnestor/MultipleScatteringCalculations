#!/usr/bin/env python3
"""GATE: what the layered table's TOP reflector actually is.

Rung 7's prerequisite, and it overturns the premise rung 7 was resting on.

The stage-1 plan records rung 7 as unblocked because "its free surface is part
of the layered background, which the solver now carries". Every gate written
before this one damps the top away on purpose (`q = 2`, so the uniform-background
reduction holds at machine precision), so that claim had never been exercised.

MEASURED, the claim is WRONG. There IS a top reflector and it is alive, but it
is the OCEAN-BOTTOM (fluid-solid) interface, not a free surface. The water
column above it does not reverberate at all -- it is a half-space.

The model has NO internal contrast: ocean over a uniform elastic half-space, so
the same-plane diagonal of the layered table isolates the top reflector exactly.

  [F1] A TOP REFLECTOR EXISTS. At high Q the diagonal is O(0.1) of the
       whole-space scale, at every depth tested. Were there no top boundary this
       would be round-off everywhere.

  [F2] IT ATTENUATES. The path is two-way to the top and back, so Q must kill
       it: 2.2e-1 -> 1.2e-5 at 3 km as Q falls 1e4 -> 2. This separates a
       reverberation from a constant offset, which would not care about Q.

  [F3] IT IS THE FLUID-SOLID INTERFACE. Changing the water velocity and density
       changes the diagonal by tens of percent, as a reflection coefficient must.
       So the interface is real and correctly parameterised.

  [F4] THERE IS NO FREE SURFACE -- reported, not gated, because it is a
       structural fact rather than a tolerance. The diagonal is INSENSITIVE to
       the water thickness: identical to four significant figures across a 100x
       change, 3 km to 300 km. A free surface at the top of the water would make
       the two-way water path, and hence the phase and amplitude, depend on that
       thickness. The ocean is a half-space.

WHY IT MATTERS. `FFTProp` carries a genuine free surface with Rayleigh
reflection and P-SV coupling (`free_surface_reflect`). This background does not.
Rung 7 therefore cannot simply be run: either a free surface is added here or
`FFTProp`'s is disabled, and that is a decision, not a detail.

ALSO NOT ESTABLISHED: the absolute magnitude of the interface reverberation.
[F1]-[F3] pin existence, attenuation and parameterisation, all of which a
wrongly scaled interface satisfies too. The route is the one that closed the
layered case -- extract the upward reflection from the diagonal and compare it
against `ocean_bottom.py`, which already builds water | slab | half-space with
`psv_fluid_solid` and the `_kennett_water_step` series.

Run:  conda run -n seismic python scripts/gate_free_surface_reverberation.py
Seismic units (km, km/s, g/cm3), time convention e^{-i omega t}.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "PhD_fortran_code"))
sys.path.insert(0, "/Users/tod/Desktop/SeismicInversion")

from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from cubic_scattering.pair_propagators import TransverseRule, layered_stack_table  # noqa: E402

PITCH = 1.0
OM = 2 * np.pi * 6.0
RULE = TransverseRule(kr_max=10.0 / PITCH, n_axis=48)


def model(
    n_lay: int = 18,
    q: float = 1e4,
    *,
    water_thick: float = 3.0,
    water_alpha: float = 1.5,
    water_rho: float = 1.03,
):
    """Ocean over a UNIFORM elastic half-space -- no internal contrast at all."""
    import Kennett_Reflectivity.layer_model as lm

    a, b, r = 5.0, 3.0, 2.5
    return lm.LayerModel.from_arrays(
        alpha=[water_alpha, *([a] * n_lay), a],
        beta=[0.0, *([b] * n_lay), b],
        rho=[water_rho, *([r] * n_lay), r],
        thickness=[water_thick, *([PITCH] * n_lay), np.inf],
        Q_alpha=[q] * (n_lay + 2),
        Q_beta=[1e10, *([q] * n_lay), q],
    )


def diagonal_ratio(planes: tuple[int, int], **kw) -> float:
    """|same-plane reverberation| relative to the whole-space inter-plane scale."""
    mod = model(**kw)
    s_p, s_s = mod.complex_slowness_p(), mod.complex_slowness_s()
    j = max(planes[0], 1)
    ref = ReferenceMedium(1.0 / s_p[j], 1.0 / s_s[j], mod.rho[j])
    tab = layered_stack_table(2, 2, 2, PITCH, OM, ref, model=mod, plane_ifaces=planes, transverse=RULE)
    ws = layered_stack_table(2, 2, 2, PITCH, OM, ref)
    return float(np.abs(tab[0, 0]).max() / np.abs(ws[0, 1]).max())


def main() -> int:
    print("=" * 78)
    print("GATE -- what the layered table's TOP reflector actually is")
    print("  ocean over a UNIFORM half-space: the only reverberation is off the top")
    print("=" * 78)

    print("\n  [F1]/[F2] diagonal vs depth and Q")
    print(f"    {'plane':>6} {'depth km':>9} {'Q=2':>11} {'Q=50':>11} {'Q=1e4':>11}")
    rows = {}
    for planes, depth in (((3, 4), 3.0), ((8, 9), 8.0), ((14, 15), 14.0)):
        vals = [diagonal_ratio(planes, q=q) for q in (2.0, 50.0, 1e4)]
        rows[depth] = vals
        print(f"    {planes[0]:6d} {depth:9.1f} {vals[0]:11.3e} {vals[1]:11.3e} {vals[2]:11.3e}")

    present = all(v[2] > 1e-2 for v in rows.values())
    dies = all(v[0] < v[2] / 100.0 for v in rows.values())
    print(f"    [F1] a top reflector exists                : {'PASS' if present else 'FAIL'}")
    print(f"    [F2] it attenuates (>100x, Q 1e4 -> 2)     : {'PASS' if dies else 'FAIL'}")

    print("\n  [F3] does it depend on the WATER PROPERTIES? (a reflection must)")
    print(f"    {'alpha':>8} {'rho':>8} {'diagonal':>12}")
    props = []
    for wa, wr in ((1.5, 1.03), (0.5, 1.03), (1.5, 3.0), (4.9, 2.49)):
        v = diagonal_ratio((8, 9), water_alpha=wa, water_rho=wr)
        props.append(v)
        print(f"    {wa:8.2f} {wr:8.2f} {v:12.4e}")
    responds = (max(props) - min(props)) / max(props) > 0.1
    print(f"    [F3] it is the fluid-solid interface       : {'PASS' if responds else 'FAIL'}")

    print("\n  [F4] does it depend on the WATER THICKNESS? (a free surface would)")
    print(f"    {'water km':>10} {'diagonal':>14}")
    thick = []
    for water in (3.0, 30.0, 300.0):
        v = diagonal_ratio((8, 9), water_thick=water)
        thick.append(v)
        print(f"    {water:10.1f} {v:14.6e}")
    insensitive = (max(thick) - min(thick)) / max(thick) < 1e-6
    print(f"    -> insensitive to thickness: {insensitive}")
    print("       THE OCEAN IS A HALF-SPACE. There is no free surface here, and")
    print("       no water-column reverberation. Reported, not gated: it is a")
    print("       structural fact about the background, not a tolerance.")

    ok = present and dies and responds
    print("\n" + "=" * 78)
    print(f"GATE top reflector: {'PASS' if ok else 'FAIL'}")
    print("  ESTABLISHED: an ocean-bottom fluid-solid interface, alive,")
    print("  attenuating, and parameterised by the water properties.")
    print("  OVERTURNED: the stage-1 claim that the free surface is part of this")
    print("  background. It is not. Rung 7 must either add one here or disable")
    print("  FFTProp's -- a decision, not a detail.")
    print("  STILL OPEN: the absolute magnitude, via ocean_bottom.py.")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
