#!/usr/bin/env python3
"""MEASUREMENT: what the dress-after ordering omits.

================================================================================
RETRACTED 2026-09-14 -- THE NUMBERS BELOW WERE TAKEN OUTSIDE THE T-MATRIX'S
VALIDATED REGIME. DO NOT QUOTE THEM.
================================================================================
This script runs at PITCH = 0.25 km, a = 0.125 km, omega = 2 pi 6, alpha = 5,
so ka = 7.54 * 0.125 = 0.94. `compute_slab_tmatrices` uses the ANALYTIC cube
T-matrix, which this project's regime table validates only for ka < 0.3; 0.3-1.0
requires the subdivided resonance treatment. Every T-matrix here was therefore
outside its range, and the error attributed to the ORDERING may be the T-matrix.

What it reported and what is withdrawn:
  * "the dress-after error reaches 70-110% of the field"  -- WITHDRAWN
  * "DeltaG0 is O(1) relative to G0"                      -- WITHDRAWN as a
    statement about a physical configuration

Re-run inside the Rayleigh regime (0.6 Hz gives ka = 0.094) the ordering effect
is 1-3% and indistinguishable from the discretisation error -- see
`scripts/gate_thesis_formulation.py`, which measures both and refuses to
conclude. The METHOD below is sound and worth keeping; only the regime was
wrong. Fix ka before reading any number from it.
================================================================================

THE THESIS'S CENTRAL CLAIM, and the one idea in this programme that has never
been tested. The thesis solves the stratified depth-average reference EXACTLY
and carries the lateral contrast as a multiple-scattering series on that
reference. `slab_scattering` and `ocean_bottom` do the opposite: they solve the
slab against a HOMOGENEOUS reference and dress the layering on afterwards, as
`MT = E R_bg E + R_slab`.

WHERE THE TWO DIFFER, precisely. Both illuminate through the layered background
and both dress the final reflectivity through it. The only difference is the
INTER-VOXEL coupling inside the Krylov solve:

    thesis ordering : G0 = whole-space + layer reverberation   (DeltaG0 present)
    dress-after     : G0 = whole-space                         (DeltaG0 absent)

So dress-after omits every voxel -> layer -> voxel path. That omission IS
DeltaG0, the object gated all through this programme -- rank 3, magnitude fixed
against Kennett at 2e-10. No third arbiter is needed to measure it: hold the
illumination and the T-matrices fixed, vary only G0, and the difference is
exactly what the approximation drops.

WHAT IS DELIBERATELY HELD FIXED. The same layered incident field drives both
solves. `ocean_bottom` in fact also illuminates its slab homogeneously, so a
faithful reproduction of it would differ in two ways at once; holding the
illumination fixed isolates the ORDERING, which is the question being asked.

REPORTED, not gated. This is a measurement of a physical quantity, not a
correctness check -- there is no target value it must hit. What it answers is
whether the omitted coupling is negligible, which is what would justify the
dress-after ordering.

Run:  conda run -n seismic python scripts/measure_ordering_difference.py
Seismic units (km, km/s, g/cm3), time convention e^{-i omega t}.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "PhD_fortran_code"))
sys.path.insert(0, "/Users/tod/Desktop/SeismicInversion")

from cubic_scattering.directional_sweeps import (  # noqa: E402
    SweepGrid3D,
    apply_g0_3d,
    build_g0_cache_3d,
)
from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from cubic_scattering.pair_propagators import TransverseRule, layered_incident_field  # noqa: E402
from cubic_scattering.slab_scattering import (  # noqa: E402
    SlabGeometry,
    SlabMaterial,
    compute_slab_tmatrices,
)
from cubic_scattering.sweep_solver import solve_foldy_lax_3d  # noqa: E402

PITCH = 0.25
OM = 2 * np.pi * 6.0
N_Z, N_X, N_Y = 2, 4, 4
PLANES = (18, 19)
SOURCE_IFACE = 14
RULE = TransverseRule(kr_max=10.0 / PITCH, n_axis=48)
SRC_VEC = np.array([1.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0], dtype=complex)


def model(contrast_scale: float, n_lay: int = 24, q: float = 1e4):
    """Ocean over elastic layers, with a contrast BELOW the scattering planes.

    The deep contrast ADDS to DeltaG0; it does not create it. Setting
    `contrast_scale = 0` leaves the SEABED, whose reverberation is already
    O(0.1) of the whole-space scale (gate_free_surface_reverberation), so there
    is no reflector-free configuration here and the null must come from
    attenuation instead. `q` is therefore the control knob, not the contrast.
    """
    import Kennett_Reflectivity.layer_model as lm

    a, b, r = 5.0, 3.0, 2.5
    al = [1.5, *([a] * n_lay), a]
    be = [0.0, *([b] * n_lay), b]
    rh = [1.03, *([r] * n_lay), r]
    for j in range(21, n_lay + 2):
        al[j] = a + 2.0 * contrast_scale
        be[j] = b + 1.2 * contrast_scale
        rh[j] = r + 0.6 * contrast_scale
    return lm.LayerModel.from_arrays(
        alpha=al,
        beta=be,
        rho=rh,
        thickness=[3.0, *([PITCH] * n_lay), np.inf],
        Q_alpha=[q] * (n_lay + 2),
        Q_beta=[1e12, *([q] * n_lay), q],
    )


def t_blocks(ref: ReferenceMedium, scale: float) -> np.ndarray:
    """A physical cube T-matrix at every voxel, modulated per site.

    Reuses `compute_slab_tmatrices`, which already returns exactly the
    (n_z, n_x, n_y, 9, 9) block array solve_foldy_lax_3d wants. Seismic units
    throughout: GPa, g/cm3, km -- self-consistent with the layered model, since
    rho v^2 = 2.5 * 25 = 62.5 GPa.
    """
    geom = SlabGeometry(M=N_X, N_z=N_Z, a=0.5 * PITCH)
    rng = np.random.default_rng(20260914)
    s = scale * (0.5 + rng.random((N_Z, N_X, N_Y)))
    material = SlabMaterial(Dlambda=2.0 * s, Dmu=1.0 * s, Drho=0.1 * s, ref=ref)
    return compute_slab_tmatrices(geom, material, OM)


def main() -> int:
    print("=" * 78)
    print("MEASUREMENT -- what the dress-after ordering omits")
    print(f"  lattice {N_Z} x {N_X} x {N_Y}, pitch {PITCH} km, planes {PLANES}")
    print("  identical illumination and T-matrices; ONLY G0 differs")
    print("=" * 78)

    # THE OMITTED OPERATOR IS REPORTED DIRECTLY, not inferred from a control
    # row. A first draft used "no deep reflector" as the null and expected the
    # two orderings to coincide there; they did not, by 1e-2 to 4e-2. The reason
    # is that the SEABED is always present -- removing the deep reflector does
    # not remove the ocean-bottom interface, and gate_free_surface_reverberation
    # already measured that reverberation at O(0.1) of the whole-space scale. So
    # the null is not a geometry with no reflector; it is a configuration in
    # which DeltaG0 itself is small, which heavy attenuation delivers.
    print(
        f"\n  {'reflector':>10} {'Q':>7} {'T':>5} {'|dG0|/|G0|':>11} "
        f"{'|psi_A-psi_B|/|psi_A|':>22} {'Born est':>11}"
    )
    for c_scale, q in ((0.0, 2.0), (0.0, 1e4), (0.3, 1e4), (1.0, 1e4)):
        mod = model(c_scale, q=q)
        s_p, s_s = mod.complex_slowness_p(), mod.complex_slowness_s()
        j = PLANES[0]
        ref = ReferenceMedium(1.0 / s_p[j], 1.0 / s_s[j], mod.rho[j])
        dz_planes = tuple(float(PLANES[i] - SOURCE_IFACE) * PITCH for i in range(N_Z))

        psi_inc = layered_incident_field(
            N_Z,
            N_X,
            N_Y,
            PITCH,
            OM,
            ref,
            source_xy=(0, 0),
            source_vec=SRC_VEC,
            dz_planes=dz_planes,
            model=mod,
            plane_ifaces=PLANES,
            source_iface=SOURCE_IFACE,
            transverse=RULE,
        )

        grid = SweepGrid3D(n_z=N_Z, n_x=N_X, n_y=N_Y, pitch=PITCH)
        cache_thesis = build_g0_cache_3d(grid, ref, OM, model=mod, plane_ifaces=PLANES, transverse=RULE)
        cache_dress = build_g0_cache_3d(grid, ref, OM)  # whole space: DeltaG0 absent

        # How big is the omitted operator itself, independent of any solve?
        rng = np.random.default_rng(7)
        probe = rng.normal(size=(N_Z, N_X, N_Y, 9)) + 1j * rng.normal(size=(N_Z, N_X, N_Y, 9))
        g_full = apply_g0_3d(probe, cache_thesis)
        d_g0 = float(np.abs(g_full - apply_g0_3d(probe, cache_dress)).max() / np.abs(g_full).max())

        for t_scale in (0.3, 1.0):
            t0 = t_blocks(ref, t_scale)
            a = solve_foldy_lax_3d(cache_thesis, t0, psi_inc, tol=1e-12).psi
            b = solve_foldy_lax_3d(cache_dress, t0, psi_inc, tol=1e-12).psi
            rel = float(np.abs(a - b).max() / np.abs(a).max())

            # Born estimate of the omitted term: one application of the missing
            # coupling to the dress-after field.
            tb = np.einsum("zxyab,zxyb->zxya", t0, b)
            omitted = apply_g0_3d(tb, cache_thesis) - apply_g0_3d(tb, cache_dress)
            born = float(np.abs(omitted).max() / np.abs(a).max())

            print(f"  {c_scale:10.1f} {q:7.0f} {t_scale:5.1f} {d_g0:11.3e} {rel:22.3e} {born:11.3e}")

    print("\n  Reading it. |dG0|/|G0| is the omitted operator's own size, measured")
    print("  without any solve -- the column the whole table hangs on. The Q = 2")
    print("  row is the null: heavy attenuation kills the path to the seabed, so")
    print("  the omitted coupling is small and the two orderings must nearly")
    print("  coincide. That is the control, and it is a control over dG0 rather")
    print("  than over the deep reflector, which does NOT remove the seabed.")
    print("\n  The Born column is one application of the omitted term. Where it")
    print("  tracks the measured difference, the omission is first-order and")
    print("  estimable; where the difference exceeds it, the omitted coupling is")
    print("  being compounded by the multiple scattering, which is precisely the")
    print("  regime the thesis ordering exists for.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
