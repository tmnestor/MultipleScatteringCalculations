#!/usr/bin/env python3
"""MEASUREMENT, not a gate: does the transverse cutoff scale with the distance to the nearest contrast?

WHY
---
scripts/measure_stack_table_cross_material.py found layered_stack_table's
documented rule, kr_max = 10/pitch, wrong by up to 25% of the operator's scale
when a material contrast lies half a pitch from the voxel planes -- in
cross-material blocks and in the reverberation block of a plane next to the
contrast -- while a no-contrast control is exact. The HYPOTHESIS recorded there
is that the needed cutoff scales with the distance h from a plane to the nearest
contrast, kr_max ~ c/h, rather than with the pitch. This tests it.

PREDICTIONS
-----------
  Hypothesis: the cutoff at which a block converges satisfies kr_max * h ~ const
              across h (so kr_max * pitch grows as h shrinks).
  Null:       kr_max * pitch ~ const, whatever h is.

GEOMETRY
--------
Quarter-pitch sublayers (0.25 km, pitch 1 km) under 3 km of water, Q = 2, 6 Hz.
Two planes, at interfaces 40 and 44 (one pitch apart, as the table's
d_z = (lz - mz) pitch requires). A plane may not sit on a material jump, so the
sublayering is what lets a contrast approach it:

  Family A, contrast BELOW both planes: fast region (2 km) starting h below the
            lower plane, h = 0.25, 0.5, 0.75, 1.0, 1.5 pitch. Both planes stay in
            the background medium -- the same-material blocks that were 25% off.
  Family B, contrast BETWEEN the planes: interface at h = 0.25 or 0.5 pitch
            below the upper plane, the lower plane inside the fast region --
            cross-material pairs.

METHOD
------
Tables at kr_max = 10, 15, 20, 30, 40 /pitch (plus 60 where h = 0.25), node
spacing dk fixed. Error of each block against the widest rule, relative to the
OPERATOR's scale (the largest off-diagonal block). For each block, the cutoff
where that error first falls below 1e-3 and 1e-5, log-linearly interpolated,
reported as both kr*pitch and kr*h_block, where h_block is the distance from the
block's planes to the nearest contrast.

Run:  conda run -n seismic python scripts/measure_transverse_rule_vs_contrast_distance.py [--probe]
Seismic units (km/s, g/cm3), time convention e^{-i omega t}.
"""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/Users/tod/Desktop/SeismicInversion")

from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from cubic_scattering.pair_propagators import TransverseRule, layered_stack_table  # noqa: E402

PITCH = 1.0
SUB = 0.25 * PITCH
OMEGA = 2 * np.pi * 6.0
PLANES = (40, 44)
N_XY = 2
DK = 0.15625
TOLS = (1e-3, 1e-5)
BG = (4.0, 2.22, 2.6)
FAST = (6.5, 3.7, 3.3)


def model(fast_layers: range):
    """Quarter-pitch sublayers; the given sublayers are fast."""
    from Kennett_Reflectivity.layer_model import LayerModel

    n_sub, q = 80, 2.0
    a = [1.5, *([BG[0]] * n_sub), BG[0]]
    b = [0.0, *([BG[1]] * n_sub), BG[1]]
    r = [1.03, *([BG[2]] * n_sub), BG[2]]
    for lay in fast_layers:
        a[lay], b[lay], r[lay] = FAST
    return LayerModel.from_arrays(
        alpha=a,
        beta=b,
        rho=r,
        thickness=[3.0, *([SUB] * n_sub), np.inf],
        Q_alpha=[q] * (n_sub + 2),
        Q_beta=[1e10, *([q] * n_sub), q],
    )


def geometries():
    """(label, fast sublayers, per-plane distance to the nearest contrast in pitches)."""
    out = []
    for k in (1, 2, 3, 4, 6):  # family A: boundary k sublayers below plane 44
        top = PLANES[1] + k
        out.append((f"A  h={k * SUB:.2f}", range(top + 1, top + 9), (k * SUB + PITCH, k * SUB)))
    for k in (1, 2):  # family B: boundary k sublayers below plane 40
        top = PLANES[0] + k
        out.append((f"B  h={k * SUB:.2f}", range(top + 1, top + 17), (k * SUB, PITCH - k * SUB)))
    return out


def build(mod, kr_pitch: float):
    """The table at cutoff kr_pitch/pitch; returns (table, seconds)."""
    kr = kr_pitch / PITCH
    rule = TransverseRule(kr_max=kr, n_axis=int(round(2 * kr / DK)))
    ref = ReferenceMedium(*BG)
    t = time.time()
    tab = layered_stack_table(
        2, N_XY, N_XY, PITCH, OMEGA, ref, model=mod, plane_ifaces=PLANES, transverse=rule
    )
    return tab, time.time() - t


def crossing(krs: np.ndarray, errs: np.ndarray, tol: float) -> float:
    """First kr where err falls below tol, log-linearly interpolated; nan if never."""
    for i in range(1, len(krs)):
        if errs[i] < tol <= errs[i - 1]:
            t = (np.log(tol) - np.log(errs[i - 1])) / (np.log(errs[i]) - np.log(errs[i - 1]))
            return float(krs[i - 1] + t * (krs[i] - krs[i - 1]))
    return float(krs[0]) if errs[0] < tol else float("nan")


def main() -> None:
    """All geometries, or a single timing probe."""
    print("=" * 100)
    print("MEASUREMENT -- converging transverse cutoff vs distance h to the nearest contrast")
    print(f"  planes {PLANES}, pitch {PITCH} km, sublayers {SUB} km, 6 Hz, Q = 2, dk = {DK}")
    print("=" * 100)
    if "--probe" in sys.argv:
        _, sec = build(model(geometries()[0][1]), 10.0)
        print(f"  probe: one 2-plane build at 10/pitch took {sec:.1f} s")
        return

    blocks = ((0, 0), (1, 1), (1, 0), (0, 1))
    summary = []
    chosen = geometries()
    if "--only" in sys.argv:  # one geometry per process, so they can run in parallel
        chosen = [chosen[int(sys.argv[sys.argv.index("--only") + 1])]]
    for label, fast, h_plane in chosen:
        mod = model(fast)
        rules = (10.0, 15.0, 20.0, 30.0, 40.0) + ((60.0,) if min(h_plane) < 0.3 else ())
        tabs, total = {}, 0.0
        for kr in rules:
            tabs[kr], sec = build(mod, kr)
            total += sec
        ref_tab = tabs[rules[-1]]
        scale = max(np.abs(ref_tab[0, 1]).max(), np.abs(ref_tab[1, 0]).max())
        dist = f"{h_plane[0]:.2f}, {h_plane[1]:.2f} pitch"
        print(f"\n  {label}   (planes' distances to contrast: {dist}; {total:.0f} s)")
        print(f"    {'block':>12} {'h_block':>8} " + " ".join(f"{f'{kr:.0f}/p':>9}" for kr in rules[:-1]))
        for lz, mz in blocks:
            h_block = min(h_plane[lz], h_plane[mz])
            errs = np.array([np.abs(tabs[kr][lz, mz] - ref_tab[lz, mz]).max() / scale for kr in rules[:-1]])
            name = f"{PLANES[mz]}->{PLANES[lz]}"
            print(f"    {name:>12} {h_block:8.2f} " + " ".join(f"{e:9.1e}" for e in errs))
            krs = np.array(rules[:-1])
            summary.append((label, name, h_block, [crossing(krs, errs, tol) for tol in TOLS]))

    print("\n  SUMMARY: cutoff where the block error first falls below tol (operator scale)")
    print(
        f"    {'geometry':>10} {'block':>8} {'h':>5} "
        + " ".join(f"{f'kr*p @{tol:.0e}':>12} {f'kr*h @{tol:.0e}':>12}" for tol in TOLS)
    )
    for label, name, h, cuts in summary:
        cells = " ".join(f"{c:12.1f} {c * h:12.1f}" for c in cuts)
        print(f"    {label:>10} {name:>8} {h:5.2f} {cells}")


if __name__ == "__main__":
    main()
