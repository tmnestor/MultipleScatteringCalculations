"""MEASUREMENT: can the thesis test be sharpened WITHOUT raising its own floor?

`gate_thesis_formulation_periodic` returned a positive indication that could not
be called conclusive: it demanded 5x separation between dress-after and the
discretisation floor, and got 3.01x. The obvious response -- scatter harder --
does not work, because the cube contrast that raises the signal raises the
touching-face floor with it.

THE WAY OUT IS THAT THE SIGNAL IS A PRODUCT OF TWO INDEPENDENT FACTORS.
The ordering effect is |DeltaG0 . T|:

    DeltaG0  is the layer reverberation, set by the REFLECTIVITY OF THE
             BACKGROUND STRATIFICATION -- nothing to do with the cubes;
    T        is the single-site T-matrix, set by the CUBE CONTRAST -- which is
             what drives the discretisation floor.

The floor arm [T3] is measured on a UNIFORM background, so it cannot see the
reflector at all. If that reasoning holds, strengthening the reflector moves
[T2] and leaves [T3] where it is, and the separation rises at fixed floor.

THIS SCRIPT MEASURES THAT, and it is written so it can come out negative. Three
things would sink it, and each is reported rather than assumed:

  * [T3] must be IDENTICAL across the sweep. It is recomputed independently and
    compared, not carried over. Any drift means `uniform=True` is not the
    reflector-free control it is claimed to be.
  * |exact| must not move much. The three arms each normalise by their own
    exact field, so a reflector that changes the amplitude at the observation
    plane could shift every ratio for a reason unrelated to the orderings.
  * [T1] must STAY at the floor as the reflector strengthens. That is the real
    prize. [T2] rising while [T1] holds is the formulation being right at
    increasing stress; BOTH rising together means the dressing is only
    cancelling part of the reverberation, which is a refutation and not a
    calibration problem.

Run:  conda run -n seismic python scripts/measure_reflector_lever.py
SI units (m, m/s, kg/m3, Pa) -- the slab machinery's own convention.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import gate_thesis_formulation_periodic as G  # noqa: E402

# Jump multipliers on the committed reflector (dalpha, dbeta, drho) =
# (2000, 1200, 600). Scale 0 is the null control -- no reflector, hence no
# reverberation, hence no ordering effect: [T1] and [T2] must COINCIDE there.
SCALES = (0.0, 0.5, 1.0, 2.0, 3.0)


def _refl_coefficient(scale: float) -> float:
    """Normal-incidence impedance reflection coefficient of the reflector."""
    d_al, _, d_rh = (c * scale for c in G.REFL_JUMP)
    z1 = G.R0 * G.A0
    z2 = (G.R0 + d_rh) * (G.A0 + d_al)
    return (z2 - z1) / (z2 + z1)


def main() -> int:
    ka = G.OM / G.A0 * G.A_HALF
    print("=" * 78)
    print("MEASUREMENT -- the reflector lever: more signal at a FIXED floor")
    print(f"  ka = {ka:.4f} (validated range is ka < 0.3); N_z = {G.N_Z}, M = {G.M}")
    print("  cube contrast is HELD FIXED throughout -- only the background moves")
    print("=" * 78)

    # THE REAL NULL. Scale 0 still leaves the WATER LAYER above the scattering
    # plane, so it is not a reverberation-free background at all -- the ocean
    # bottom is itself a reflector. The only genuinely unstratified case is the
    # uniform one, and there the dressing has nothing to do, so dressed and
    # undressed must coincide.
    n_dressed, _ = G._run(dressed=True, uniform=True)
    n_plain, _ = G._run(dressed=False, uniform=True)
    print(f"\n  NULL (uniform background): dressed {n_dressed:.4e}  undressed {n_plain:.4e}")
    print(f"    they differ by {abs(n_dressed - n_plain) / n_dressed:.2e} -- must be ~0")

    print(
        f"\n  {'scale':>6} {'R_refl':>8} {'|exact|':>11} "
        f"{'[T1]rel':>10} {'[T2]rel':>10} {'T2/T3':>7} {'T1/T3':>7} "
        f"{'[T1]abs':>11} {'[T2]abs':>11}"
    )

    t3_first, rows = None, []
    for scale in SCALES:
        jump = tuple(c * scale for c in G.REFL_JUMP)
        t1, s1 = G._run(dressed=True, uniform=False, refl_jump=jump)
        t2, _ = G._run(dressed=False, uniform=False, refl_jump=jump)
        # Recomputed every row ON PURPOSE. It should not move; that is the
        # control on the whole idea, and carrying it over would hide a drift.
        t3, _ = G._run(dressed=True, uniform=True, refl_jump=jump)
        if t3_first is None:
            t3_first = t3
        rows.append((scale, s1, t1 * s1, t2 * s1))
        print(
            f"  {scale:6.2f} {_refl_coefficient(scale):8.3f} {s1:11.4e} "
            f"{t1:10.3e} {t2:10.3e} {t2 / t3:7.2f} {t1 / t3:7.2f} "
            f"{t1 * s1:11.4e} {t2 * s1:11.4e}"
        )

    drift = abs(t3 - t3_first) / t3_first
    print(f"\n  floor drift across the sweep: {drift:.2e}")
    if drift > 1e-10:
        print("  ** [T3] moved with the reflector -- the uniform arm is not")
        print("  ** reflector-free and the separations are not comparable.")

    # THE NORMALISATION CHECK, and it is the one that decides whether any of
    # this is real. Each arm divides by its OWN |exact|. If that amplitude moves
    # across the sweep, a rising RELATIVE separation can be nothing but a
    # shrinking denominator, so the absolute errors are the arbiter.
    amp_ratio = rows[0][1] / rows[-1][1]
    abs_t2_ratio = rows[-1][3] / rows[0][3]
    print(f"\n  |exact| fell by {amp_ratio:.2f}x across the sweep")
    print(f"  ABSOLUTE dress-after error changed by {abs_t2_ratio:.2f}x")
    if amp_ratio > 1.2 and abs_t2_ratio < 1.0:
        print("  ** THE LEVER DOES NOT WORK. The relative separation rises only")
        print("  ** because |exact| shrinks; in absolute terms the dress-after")
        print("  ** error FALLS as the reflector strengthens. The reflector is")
        print("  ** interfering destructively at the observation plane, which")
        print("  ** deflates the denominator faster than it inflates the signal.")
        print("  ** A 5x reached this way would be an artifact, and taking it")
        print("  ** would be exactly the bar-lowering the gate refuses to do.")

    print("\n  Reading it. T2/T3 rising while T1/T3 stays flat would be the")
    print("  thesis ordering holding under increasing stress; both rising")
    print("  together is a refutation -- but neither reading is available while")
    print("  the denominator is moving. Fix the amplitude first, then re-read.")
    print("  Reported, not gated: this measures whether the gate can be")
    print("  sharpened, and the gate keeps its own 5x bar either way.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
