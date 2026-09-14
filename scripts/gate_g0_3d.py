#!/usr/bin/env python3
"""GATE: the composed three-dimensional G0, end to end.

The unit tests exercise the pieces -- the tables against the closed form, the
partition as an integer count -- but always on whole-space or synthetic tables.
This is the first thing that runs the LAYERED 3-D operator all the way through,
which is where a defect in the composition of table build and application would
actually live.

  [G1] VALUE. Whole-space G0 against an explicit O(N^2) pairwise sum over the
       closed form, with a DISTINCT source at every site.

  [G2] VACUITY CONTROL, mandatory. A uniform source would pass [G1] even for an
       implementation that silently averaged the sites, so this demonstrates
       that it does -- the claim being made is disorder-resolved coupling, and a
       test that a site-averaging implementation also passes is not evidence for
       it.

  [G3] LAYERED REDUCTION, end to end. The operator built on a UNIFORM stratified
       background must equal the whole-space operator. This runs the stratified
       propagator, the transverse transform, the table assembly and the
       application, and demands the closed-form answer back from all four.

  [G4] LAYERED IS NOT TRIVIAL. With a real contrast the operator must DIFFER
       from whole space. Without this, an implementation that silently ignored
       the background would pass [G3] perfectly.

Run:  conda run -n seismic python scripts/gate_g0_3d.py
Seismic units (km/s, g/cm3), time convention e^{-i omega t}.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/Users/tod/Desktop/SeismicInversion")

from cubic_scattering.directional_sweeps import (  # noqa: E402
    SweepGrid3D,
    apply_g0_3d,
    build_g0_cache_3d,
)
from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from cubic_scattering.horizontal_greens import exact_propagator_9x9  # noqa: E402
from cubic_scattering.pair_propagators import TransverseRule  # noqa: E402

OM = 2 * np.pi * 6.0
PITCH = 1.0
PLANES = (8, 9)


def _model(contrast: bool, n_lay: int = 16, q: float = 2.0):
    """A stack with one interface per pitch. q = 2 damps the free surface.

    Attenuation is load-bearing, not cosmetic: the layered propagator carries
    surface reverberations the whole-space kernel does not, and only damping
    attenuates them enough for [G3] to reduce. With q large, a CORRECT
    implementation fails [G3] at ~0.5.
    """
    import Kennett_Reflectivity.layer_model as lm

    a, b, r = 5.0, 3.0, 2.5
    al, be, rh = [1.5, *([a] * n_lay), a], [0.0, *([b] * n_lay), b], [1.03, *([r] * n_lay), r]
    if contrast:
        for j in range(11, n_lay + 2):
            al[j], be[j], rh[j] = 6.8, 3.9, 3.2
    return lm.LayerModel.from_arrays(
        alpha=al,
        beta=be,
        rho=rh,
        thickness=[3.0, *([PITCH] * n_lay), np.inf],
        Q_alpha=[q] * (n_lay + 2),
        Q_beta=[1e10, *([q] * n_lay), q],
    )


def _ref(model) -> ReferenceMedium:
    s_p, s_s = model.complex_slowness_p(), model.complex_slowness_s()
    j = max(PLANES[0], 1)
    return ReferenceMedium(1.0 / s_p[j], 1.0 / s_s[j], model.rho[j])


def _pairwise(src, grid, ref) -> np.ndarray:
    out = np.zeros_like(src)
    for lz in range(grid.n_z):
        for ix in range(grid.n_x):
            for iy in range(grid.n_y):
                for mz in range(grid.n_z):
                    for jx in range(grid.n_x):
                        for jy in range(grid.n_y):
                            if (lz, ix, iy) == (mz, jx, jy):
                                continue
                            out[lz, ix, iy] += (
                                exact_propagator_9x9(
                                    (ix - jx) * PITCH,
                                    (iy - jy) * PITCH,
                                    (lz - mz) * PITCH,
                                    OM,
                                    ref,
                                )
                                @ src[mz, jx, jy]
                            )
    return out


def main() -> int:
    grid = SweepGrid3D(n_z=2, n_x=3, n_y=2, pitch=PITCH)
    uni = _model(contrast=False)
    ref = _ref(uni)
    shape = (grid.n_z, grid.n_x, grid.n_y, 9)
    rng = np.random.default_rng(20260914)
    distinct = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    uniform = np.ones(shape, dtype=complex)

    print("=" * 78)
    print("GATE -- the composed three-dimensional G0, end to end")
    print(f"  lattice {grid.n_z} x {grid.n_x} x {grid.n_y}, pitch {PITCH} km")
    print("=" * 78)

    ok = True
    ws = build_g0_cache_3d(grid, ref, OM)

    got = apply_g0_3d(distinct, ws)
    want = _pairwise(distinct, grid, ref)
    g1 = float(np.abs(got - want).max() / np.abs(want).max())
    ok = ok and g1 < 1e-13 and np.isfinite(got).all()
    print(f"\n  [G1] whole-space vs pairwise, distinct sources : {g1:.3e}")

    avg = np.broadcast_to(distinct.mean(axis=(0, 1, 2)), shape).copy()
    g2_d = float(np.abs(apply_g0_3d(distinct, ws) - apply_g0_3d(avg, ws)).max())
    # The control is a statement about the INPUT, not the output: for a uniform
    # source the site-averaged array IS the original, so no implementation can
    # be distinguished from a site-averaging one. Comparing the two outputs
    # would be tautological; comparing the two inputs is the actual claim.
    g2_u = float(np.abs(uniform - np.broadcast_to(uniform.mean(axis=(0, 1, 2)), shape)).max())
    print(f"  [G2] vacuity control: distinct vs site-averaged : {g2_d:.3e} (must be LARGE)")
    print(f"       uniform source is its own site-average     : {g2_u:.3e} (hence blind)")
    ok = ok and g2_d > 1e-3 and g2_u < 1e-14

    rule = TransverseRule(kr_max=10.0 / PITCH, n_axis=48)
    lay_u = build_g0_cache_3d(grid, ref, OM, model=uni, plane_ifaces=PLANES, transverse=rule)
    g3 = float(
        np.abs(apply_g0_3d(distinct, lay_u) - apply_g0_3d(distinct, ws)).max()
        / np.abs(apply_g0_3d(distinct, ws)).max()
    )
    ok = ok and g3 < 1e-11
    print(f"  [G3] uniform layered reduces to whole space     : {g3:.3e}")

    lay_c = build_g0_cache_3d(
        grid, ref, OM, model=_model(contrast=True), plane_ifaces=PLANES, transverse=rule
    )
    g4 = float(
        np.abs(apply_g0_3d(distinct, lay_c) - apply_g0_3d(distinct, ws)).max()
        / np.abs(apply_g0_3d(distinct, ws)).max()
    )
    ok = ok and g4 > 1e-6
    print(f"  [G4] contrasted layered DIFFERS from whole space: {g4:.3e} (must be LARGE)")

    print("\n" + "=" * 78)
    print(f"GATE 3-D G0: {'PASS' if ok else 'FAIL'}")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
