#!/usr/bin/env python3
"""GATE: what the layered table's TOP reflector is, with and without a free surface.

Rung 7's prerequisite. It overturned the premise rung 7 was resting on, and the
fix that followed is gated here too.

The stage-1 plan records rung 7 as unblocked because "its free surface is part
of the layered background, which the solver now carries". Every gate written
before this one damps the top away on purpose (`q = 2`, so the uniform-background
reduction holds at machine precision), so that claim had never been exercised.

MEASURED, the claim was WRONG. The default background has a live top reflector,
but it is the OCEAN-BOTTOM (fluid-solid) interface: the water column above it
does not reverberate at all, being a half-space. `free_surface=True` now closes
the ocean with a pressure release and makes it finite; the default is unchanged,
bit for bit.

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

  [F5] WITH `free_surface=True` THE WATER REVERBERATES -- and periodically, at
       FIXED (kx, ky), returning to itself after exactly one round-trip cycle of
       the two-way phase. Periodicity is the discriminating property: a response
       that merely drifted with thickness could be a leak.

  [F6] THE FLAG REACHES THE 3-D TABLE, through `corrected_layered_9x9` and
       `layered_stack_table`.

WHY IT MATTERS. `FFTProp` carries a genuine free surface with Rayleigh
reflection and P-SV coupling (`free_surface_reflect`). The default background
here does not, so any comparison against it must set `free_surface=True`.

ALSO NOT ESTABLISHED: the absolute MAGNITUDE of either the interface
reverberation or the surface one. [F1]-[F6] pin existence, attenuation,
parameterisation, periodicity and plumbing -- all of which a wrongly scaled
surface satisfies too. The route is the one that closed the layered case:
extract the upward reflection from the diagonal and compare it against
`ocean_bottom.py`, which already builds water | slab | half-space with
`psv_fluid_solid` and the `_kennett_water_step` series. The plane-wave
reflectivity version of exactly this stack is already reconciled against Kennett
at 1e-15 in `gate_gmm_marine_stack`, which gives that comparison a validated
target.

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

import cubic_scattering.layered_correction as LC  # noqa: E402
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


def diagonal_vec(planes: tuple[int, int], *, free_surface: bool = False, **kw) -> np.ndarray:
    """The same-plane reverberation block itself, for phase-sensitive tests."""
    mod = model(**kw)
    s_p, s_s = mod.complex_slowness_p(), mod.complex_slowness_s()
    j = max(planes[0], 1)
    ref = ReferenceMedium(1.0 / s_p[j], 1.0 / s_s[j], mod.rho[j])
    tab = layered_stack_table(
        2,
        2,
        2,
        PITCH,
        OM,
        ref,
        model=mod,
        plane_ifaces=planes,
        transverse=RULE,
        free_surface=free_surface,
    )
    return tab[0, 0]


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

    print("\n  [F5] with free_surface=True the water column REVERBERATES")
    print("       AT FIXED (kx, ky), stepping the thickness by quarter-cycles.")
    print("       Fixed k is essential: the two-way phase is 2 w eta_w h_w and")
    print("       eta_w varies across the transverse grid, so the INTEGRATED")
    print("       table has no single period and cannot be periodic in h_w. An")
    print("       earlier draft tested the integrated table and read its")
    print("       aperiodicity as a failure; it is a property of the integral.")
    kx, ky = np.array([0.4]), np.array([0.3])
    p_ray = float(np.hypot(0.4, 0.3) / OM)
    eta_w = float(np.sqrt(1.0 / 1.5**2 - p_ray**2))
    step = float((np.pi / 2) / (2 * OM * eta_w))

    def spectral(wt: float) -> np.ndarray:
        return LC.corrected_layered_9x9(model(water_thick=wt), OM, kx, ky, 3, 2, free_surface=True)[0]

    base = spectral(1.0)
    print(f"    quarter-cycle step {step:.5f} km")
    print(f"    {'water km':>10} {'phase (rad)':>13} {'change vs base':>16}")
    changes = []
    for k in range(5):
        wt = 1.0 + k * step
        ch = float(np.abs(spectral(wt) - base).max() / np.abs(base).max())
        changes.append(ch)
        print(f"    {wt:10.5f} {2 * OM * eta_w * wt:13.2f} {ch:16.3e}")
    # PERIODICITY is the discriminating property, not mere sensitivity. A
    # response that merely drifted with thickness could be a leak; one that
    # returns to itself after exactly one round-trip cycle is a reverberation.
    swings = max(changes) > 0.1
    returns = changes[4] < max(changes) / 1e4
    print(f"    [F5] swings within the cycle (> 0.1)      : {'PASS' if swings else 'FAIL'}")
    print(f"    [F5] returns after a FULL cycle (< 1e-4x) : {'PASS' if returns else 'FAIL'}")

    print("\n  [F6] the INTEGRATED table moves too (sensitivity, not periodicity)")
    a = diagonal_vec((8, 9), water_thick=1.0, free_surface=True)
    b = diagonal_vec((8, 9), water_thick=1.0 + 2 * step, free_surface=True)
    moved = float(np.abs(a - b).max() / np.abs(a).max())
    off = diagonal_vec((8, 9), water_thick=1.0, free_surface=False)
    flag_bites = float(np.abs(a - off).max() / np.abs(off).max())
    print(f"    thickness change, flag on  : {moved:.3e}")
    print(f"    flag on vs flag off        : {flag_bites:.3e}")
    integ = moved > 1e-3 and flag_bites > 1e-2
    print(f"    [F6] the flag reaches the 3-D table       : {'PASS' if integ else 'FAIL'}")

    ok = present and dies and responds and swings and returns and integ
    print("\n" + "=" * 78)
    print(f"GATE top reflector: {'PASS' if ok else 'FAIL'}")
    print("  DEFAULT (free_surface=False): an ocean-bottom fluid-solid interface,")
    print("  alive, attenuating, parameterised by the water properties -- but the")
    print("  ocean above it is a HALF-SPACE. The stage-1 claim that the free")
    print("  surface came for free with the layered background was wrong.")
    print("  WITH free_surface=True: the water column is finite and reverberates,")
    print("  periodically in the round-trip phase at fixed k, and the flag reaches")
    print("  the 3-D table. Use it for any comparison against a solver that")
    print("  carries a free surface -- FFTProp does.")
    print("  STILL OPEN: the absolute magnitude of either. Existence, attenuation,")
    print("  parameterisation and periodicity are all satisfied by a wrongly")
    print("  scaled surface too. Arbiter: ocean_bottom.py.")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
