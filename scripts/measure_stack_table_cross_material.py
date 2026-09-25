#!/usr/bin/env python3
"""MEASUREMENT, not a gate: is layered_stack_table's transverse rule adequate across materials?

WHY
---
Gate A1b (scripts/gate_a1b_receiver_transpose.py) found that
layered_incident_field, at the documented rule TransverseRule(kr_max=10/pitch),
is truncation-limited when source and receiver planes lie in DIFFERENT
materials: 2.4e-2 at 10/pitch, 3.3e-10 at 20/pitch, flat in the node spacing.
The construction subtracts the whole-space kernel of the RECEIVER plane's medium
spectrally and adds it back in closed form; across a material change the
remainder carries the difference between the direct paths, not reverberation
alone. The 10/pitch rule was fixed (measure_dg0_transverse_rule.py) on one
same-medium pair.

layered_stack_table -- the 3-D forward operator's real-space tables -- uses the
same subtract-transform-add and the same rule. This measures whether its
cross-material blocks carry the same truncation.

GEOMETRY, AND A CONSTRAINT IT EXPOSES
-------------------------------------
The table takes d_z = (lz - mz) pitch, so its planes must be consecutive and
one pitch apart; and every plane must be interior to a layer
(assert_interface_continuous). With the pitch-thick layers the docstrings
prescribe, those two rules make a cross-material pair IMPOSSIBLE -- the material
change would have to sit on a plane. Half-pitch sublayers, planes on every
second interface and material jumps on the interfaces between them, are the
construction that reaches it:

    sublayers 0.5 km; planes at interfaces 14, 16, 18 (pitch 1.0 km);
    fast slab = sublayers 16-17, so plane 16 is IN the slab, 14 and 18 are not.
    Control: the same stack with no slab (every pair same-material).

METHOD
------
Self-convergence in kr_max at FIXED node spacing dk, against the widest rule;
plus one dk refinement at 10/pitch, to tell truncation (moves with kr_max) from
quadrature (moves with dk). Per block (lz, mz): max |T - T_ref| / max |T_ref|.
Build time per rule is reported, because the cure has a cost.

RESULT (26 September 2026). Errors of the 10/pitch table against 30/pitch,
quoted against the OPERATOR's scale (the largest off-diagonal block), since a
block's error relative to itself overstates a small block:

    block (source -> receiver)          kind                        10/pitch   20/pitch*
    16 -> 14, 16 -> 18                  cross-material              2.5e-1     8.8e-5
    14 -> 16, 18 -> 16                  cross-material              7.1e-2     7.6e-6
    14 -> 14, 18 -> 18                  diagonal, slab 0.5 p away   2.5e-1     2.8e-4
    16 -> 16                            diagonal, inside the slab   1.2e-1     1.3e-5
    14 -> 18, 18 -> 14                  same material, slab between 4.2e-4     2.7e-10
    CONTROL, no slab, every block                                   <= 6e-13
    (* 20/pitch against 40/pitch, relative to the block itself.)

Truncation, not quadrature: halving dk at 10/pitch moves every block by
<= 8e-5, while raising the cutoff removes the error. The control is exact to
1e-16 (its diagonals are the damped seabed return, ~5e-13 of the operator), so
the documented rule is right exactly where it was calibrated and wrong near a
contrast. Not only cross-material PAIRS are affected: a diagonal block whose
plane sits half a pitch from a contrast is as bad. Consistent with -- NOT proven
by -- a rule that must scale with the distance h from a plane to the nearest
contrast (kr_max ~ 10/h), not with the pitch: here h = pitch/2 and 20/pitch is
the first rule to converge. Build cost scales as n_axis^2: 21.6 s at 10/pitch,
83.9 s at 20, 194 s at 30, 353 s at 40, for this 3-plane 2x2 table.

No existing test or gate exercised this. test_layered_stack_table and
gate_free_surface_reverberation put every plane in one medium, with any
contrast >= 2 pitches away; the 3-D solver tests (test_apply_g0_3d,
test_sweep_solver_3d) use synthetic or whole-space tables and never build a
stratified one.

Run:  conda run -n seismic python scripts/measure_stack_table_cross_material.py
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
SUB = 0.5 * PITCH
OMEGA = 2 * np.pi * 6.0
PLANES = (14, 16, 18)
N_XY = 2
DK = 0.15625  # node spacing held fixed while the cutoff grows


def model(slab: bool):
    """Half-pitch sublayers under 3 km of water, Q = 2; optional fast slab in sublayers 16-17."""
    from Kennett_Reflectivity.layer_model import LayerModel

    n_sub, q = 40, 2.0
    al, be, rh = 4.0, 2.22, 2.6
    a = [1.5, *([al] * n_sub), al]
    b = [0.0, *([be] * n_sub), be]
    r = [1.03, *([rh] * n_sub), rh]
    if slab:
        for lay in (16, 17):
            a[lay], b[lay], r[lay] = 6.5, 3.7, 3.3
    return LayerModel.from_arrays(
        alpha=a,
        beta=b,
        rho=r,
        thickness=[3.0, *([SUB] * n_sub), np.inf],
        Q_alpha=[q] * (n_sub + 2),
        Q_beta=[1e10, *([q] * n_sub), q],
    )


def build(mod, kr_pitch: float, dk: float):
    """The table at cutoff kr_pitch/pitch and spacing dk; returns (table, seconds)."""
    kr = kr_pitch / PITCH
    rule = TransverseRule(kr_max=kr, n_axis=int(round(2 * kr / dk)))
    s_p, s_s = mod.complex_slowness_p(), mod.complex_slowness_s()
    ref = ReferenceMedium(1.0 / s_p[PLANES[0]], 1.0 / s_s[PLANES[0]], mod.rho[PLANES[0]])
    t = time.time()
    tab = layered_stack_table(
        len(PLANES), N_XY, N_XY, PITCH, OMEGA, ref, model=mod, plane_ifaces=PLANES, transverse=rule
    )
    return tab, time.time() - t


def block_errors(tab, ref_tab) -> np.ndarray:
    """Relative difference per (lz, mz) block."""
    n = len(PLANES)
    out = np.zeros((n, n))
    for lz in range(n):
        for mz in range(n):
            out[lz, mz] = np.abs(tab[lz, mz] - ref_tab[lz, mz]).max() / np.abs(ref_tab[lz, mz]).max()
    return out


def medium_label(slab: bool, iface: int) -> str:
    return "slab" if slab and iface == 16 else "bg"


def run(slab: bool, rules: tuple[float, ...]) -> None:
    """Convergence table for one model."""
    mod = model(slab)
    name = "WITH fast slab (plane 16 in the slab)" if slab else "CONTROL, no slab (all pairs same-material)"
    print(f"\n{name}")
    tabs = {}
    for kr in rules:
        tabs[kr], sec = build(mod, kr, DK)
        print(
            f"    built kr_max = {kr:>4.0f}/pitch, n_axis = {int(round(2 * kr / DK)):4d}  in {sec:7.1f} s"
        )
    ref_tab = tabs[rules[-1]]
    fine_dk, sec = build(mod, rules[0], DK / 2)
    print(f"    built kr_max = {rules[0]:>4.0f}/pitch, dk halved              in {sec:7.1f} s")

    pairs = [(lz, mz) for lz in range(len(PLANES)) for mz in range(len(PLANES))]
    head = " ".join(f"{f'{PLANES[m]}->{PLANES[l]}':>10}" for l, m in pairs)

    def kind(lz: int, mz: int) -> str:
        if lz == mz:
            return "diag"
        cross = medium_label(slab, PLANES[lz]) != medium_label(slab, PLANES[mz])
        return "CROSS" if cross else "same"

    kinds = " ".join(f"{kind(l, m):>10}" for l, m in pairs)
    print(f"    {'rule':>18} {head}")
    print(f"    {'':>18} {kinds}")
    for kr in rules[:-1]:
        err = block_errors(tabs[kr], ref_tab)
        print(
            f"    {f'{kr:.0f}/pitch vs {rules[-1]:.0f}':>18} "
            + " ".join(f"{err[l, m]:10.2e}" for l, m in pairs)
        )
    err = block_errors(tabs[rules[0]], fine_dk)
    print(
        f"    {f'{rules[0]:.0f}/pitch: dk vs dk/2':>18} " + " ".join(f"{err[l, m]:10.2e}" for l, m in pairs)
    )


def main() -> None:
    """Timing probe first when asked; otherwise both models."""
    print("=" * 110)
    print("MEASUREMENT -- layered_stack_table transverse cutoff, same- vs cross-material plane pairs")
    print(f"  planes {PLANES} (interfaces), pitch {PITCH} km, sublayers {SUB} km")
    print(f"  f = {OMEGA / (2 * np.pi):.0f} Hz, Q = 2")
    print(f"  {N_XY}x{N_XY} lateral lattice, node spacing dk = {DK} held fixed")
    print("=" * 110)
    if "--probe" in sys.argv:
        _, sec = build(model(True), 10.0, DK)
        print(f"  probe: one build at 10/pitch took {sec:.1f} s")
        return
    rules = (10.0, 20.0, 30.0, 40.0)
    run(True, rules)
    run(False, rules)


if __name__ == "__main__":
    main()
