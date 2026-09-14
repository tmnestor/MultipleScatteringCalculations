"""MEASUREMENT: is there a geometry where the reverberation ADDS at the observer?

`measure_reflector_lever` showed that strengthening the reflector does not
sharpen the thesis gate: the relative separation rises to 5.4x, but only because
|exact| falls 3.4x by DESTRUCTIVE interference at the observation plane, while
the absolute dress-after error falls. The obstacle was the observation-plane
amplitude, not the discretisation floor.

That is a PHASE, not a law. The reverberation returns to the observer carrying
2 (omega/alpha) dz for a reflector dz below, so the reflector depth and the
frequency together decide whether it adds to the direct arrival or cancels it.
The lever sweep held both fixed and only ever changed the reflector's strength,
which is why it could only ever find one side of that interference.

WHY THE FREQUENCY MUST MOVE, and this is the thing that makes the scan
non-trivial. At the gate's omega = 60 the round trip to a reflector 2 m below is
2 * (60/5000) * 2 = 0.05 rad -- under one percent of a cycle. Across the whole
available depth range the phase barely moves at all, so scanning DEPTH ALONE at
the gate's frequency cannot flip the sign of the interference. A full cycle
needs omega of a few hundred. ka = omega/alpha * a stays far inside the
validated ka < 0.3 up to omega ~ 1500, so there is room; ka IS PRINTED for every
row and flagged if it leaves that range.

WHAT WOULD COUNT. A cell where the ABSOLUTE dress-after error [T2] is large
against the floor [T3] AND |exact| is not depressed relative to its neighbours.
Both conditions matter: a big ratio bought by a small denominator is the
artifact the lever already found, and taking it would be the bar-lowering the
gate exists to refuse. The absolute columns are therefore the ones to read.

Run:  conda run -n seismic python scripts/measure_constructive_geometry.py
SI units (m, m/s, kg/m3, Pa) -- the slab machinery's own convention.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import gate_thesis_formulation_periodic as G  # noqa: E402

# Layers below the voxel plane are 1 m thick, so interface j sits (j - 20) m
# below the scattering plane. 22 is the committed value (immediately below the
# voxels); 38 is as deep as the 40-layer stack allows.
REFL_LAYERS = (22, 26, 30, 34, 38)
# omega = 60 is the gate's own value and anchors the scan to the known result.
# The rest climb until the round trip covers a full cycle at the deeper
# reflectors. ka = omega/5000 * 1.0, so even 900 is ka = 0.18.
OMEGAS = (60.0, 200.0, 400.0, 600.0, 900.0)


def main() -> int:
    print("=" * 96)
    print("MEASUREMENT -- scanning reflector depth x frequency for CONSTRUCTIVE return")
    print("  the lever failed on destructive interference at the observer, not on the floor")
    print("  cube contrast and reflector strength are HELD FIXED; only the phase moves")
    print("=" * 96)

    best, amp_ref = None, None
    for omega in OMEGAS:
        ka = omega / G.A0 * G.A_HALF
        flag = "" if ka < 0.3 else "   <-- ka OUT OF VALIDATED RANGE"
        # The floor depends on the frequency but NOT on the reflector, so it is
        # computed once per row-block and reused down the block.
        t3, s3 = G._run(dressed=True, uniform=True, omega=omega)
        a3 = t3 * s3
        print(f"\n  omega = {omega:6.1f}   ka = {ka:.4f}   floor [T3]abs = {a3:.4e}{flag}")
        # GAIN = [T2]/[T1] is the column that carries the physics. It is the
        # factor by which dressing the kernel with DeltaG0 IMPROVES the answer,
        # and unlike the T2/T3 and T1/T3 ratios it is free of the floor and of
        # |exact| alike -- both arms share the same normalisation and the same
        # discretisation. Gain > 1 means the dressing helps; gain < 1 means it
        # actively HURTS, which is a defect and not a resolution limit.
        print(
            f"    {'refl':>5} {'dz(m)':>6} {'phase/2pi':>10} {'|exact|':>11} "
            f"{'[T1]abs':>11} {'[T2]abs':>11} {'T2/T3':>7} {'T1/T3':>7} {'GAIN':>6}"
        )
        for layer in REFL_LAYERS:
            dz = float(layer - G.SCAT_IFACE)  # 1 m half-pitch layers
            cycles = 2.0 * (omega / G.A0) * dz / (2.0 * 3.141592653589793)
            t1, s1 = G._run(dressed=True, uniform=False, refl_layer=layer, omega=omega)
            t2, _ = G._run(dressed=False, uniform=False, refl_layer=layer, omega=omega)
            a1, a2 = t1 * s1, t2 * s1
            if amp_ref is None:
                amp_ref = s1  # the gate's own cell, the yardstick for "collapsed"
            # A cell whose exact field has collapsed is in an interference null,
            # where every ratio is formed on a denominator that is itself nearly
            # cancelling. Flagged, because such cells must not be read as physics.
            null = "  << null" if s1 < 0.2 * amp_ref else ""
            print(
                f"    {layer:5d} {dz:6.1f} {cycles:10.3f} {s1:11.4e} "
                f"{a1:11.4e} {a2:11.4e} {a2 / a3:7.2f} {a1 / a3:7.2f} {a2 / a1:6.2f}{null}"
            )
            if ka < 0.3 and (best is None or a2 / a3 > best[0]):
                best = (a2 / a3, a1 / a3, omega, layer, s1)

    print("\n" + "=" * 96)
    if best is None:
        print("  NOTHING INSIDE THE VALIDATED ka RANGE -- the scan says nothing.")
        return 0
    sep, thesis_arm, omega, layer, amp = best
    print(f"  BEST SEPARATION: {sep:.2f}x at omega = {omega:.1f}, reflector layer {layer}")
    print(f"    thesis arm there: {thesis_arm:.2f}x the floor;  |exact| = {amp:.4e}")
    if sep > 5.0 and thesis_arm < 2.0:
        print("  This clears the gate's 5x bar with the thesis arm still AT the floor.")
        print("  CHECK |exact| against its neighbours before believing it -- a big")
        print("  ratio bought by a depressed denominator is the lever's artifact.")
    elif sep > 5.0:
        print("  Separation clears 5x but the thesis arm is ALSO above the floor.")
        print("  That is the refutation signature, not a win. Read it carefully.")
    else:
        print("  Still short of 5x. The phase is not the missing ingredient, and")
        print("  the floor itself has to go -- replace the cube T-matrix with the")
        print("  exact normal-incidence slab T-matrix, which removes the")
        print("  touching-face error rather than fighting it.")
    print("\n  READ THE GAIN COLUMN, not the separation. Separation is what the")
    print("  gate needs to CONCLUDE; gain is what says whether the formulation")
    print("  is right. Gain falling towards 1 as omega rises would mean the")
    print("  dressing stops helping; gain below 1 would mean it actively hurts,")
    print("  which no amount of floor reduction would repair.")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
