#!/usr/bin/env python3
"""MEASUREMENT, not a gate: what does the class-A (Dz != 0) table actually cost?

Task 2 of docs/plans/2026-09-14-cartesian-directional-sweeps-stage2.md.

A CORRECTION TO HOW THE PLAN FRAMED THIS. The plan justified the real-space
architecture with a 149 MB storage figure and then asked whether the
reverberation needs fewer transverse nodes. But the real-space table size is
n_z^2 * (2n_x-1)(2n_y-1) * 81 -- it carries NO n_k at all. The node count does
not touch storage; it drives BUILD TIME. So the question Task 2 has to answer is
not "does it fit" (it does, and always did) but "can it be built".

Two things are measured here.

  [A] CONVERGENCE. How fine a transverse rule does DeltaG0 need? There is no
      closed form for the reverberation, so this is a self-convergence test:
      the relative change between successive rules. The expectation from
      gate_dg0_absolute_magnitude [M1] is that DeltaG0 is rank 3 and smooth,
      hence cheaper than the full kernel -- an EXPECTATION, which is what this
      measures rather than assumes.

  [B] BUILD TIME, with the transform done SEPARABLY. The map from
      DeltaG0(k_x, k_y) to P(dx, dy) is a 2-D Fourier sum over a tensor grid,
      so it factorises: transform along k_x to get dx, then along k_y to get dy.
      Done densely it is n_out * n_k per component and hopeless; done separably
      it is n_k * sqrt(n_out) and merely expensive. This measures which.

If [A] says DeltaG0 needs the same rule as the singular kernel AND [B] says the
build does not fit in a sensible wall-clock, STOP -- do not raise the node count
until the number looks acceptable. That is the failure mode that produced the
first version of the stage-2 plan.

Run:  conda run -n seismic python scripts/measure_dg0_transverse_rule.py
Seismic units (km/s, g/cm3), time convention e^{-i omega t}.
"""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "PhD_fortran_code"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, "/Users/tod/Desktop/SeismicInversion")

import cubic_scattering.layered_correction as LC  # noqa: E402
from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from cubic_scattering.sweep_kernels import vertical_kernel_9x9  # noqa: E402
from gate_sh_impedance import OM as OM_REAL  # noqa: E402
from gate_sh_impedance import PIT as PITCH  # noqa: E402
from gate_sh_impedance import PLANE, model  # noqa: E402

# Complex, for the same reason measure_sweep3d_cost.py insists on it.
OM = OM_REAL * (1.0 + 0.03j)
DZ = -PITCH  # one plane apart


def tensor_rule(kr_max: float, n_axis: int) -> tuple[np.ndarray, float]:
    """1-D node set and weight for a separable (k_x, k_y) tensor rule.

    Deliberately a tensor product, not the radially masked set used for the
    Dz = 0 case: separability is what makes the transform affordable, and the
    mask destroys it. At Dz != 0 the e^{-kappa |dz|} factor already suppresses
    the corners, so the mask buys much less here than it does at equal depth.
    """
    edge = np.linspace(-kr_max, kr_max, n_axis + 1)
    ctr = 0.5 * (edge[:-1] + edge[1:])
    return ctr, float(edge[1] - edge[0])


def dg0_on_grid(mod, kx: np.ndarray, ky: np.ndarray, ref: ReferenceMedium) -> np.ndarray:
    """DeltaG0 = layered - whole-space, at paired (kx, ky) nodes. Shape (n, 9, 9)."""
    lay = LC.corrected_layered_9x9(mod, OM, kx, ky, PLANE + 1, PLANE)
    ws = np.empty_like(lay)
    for j, (a, b) in enumerate(zip(kx, ky, strict=True)):
        ws[j] = vertical_kernel_9x9(np.array([a]), float(b), DZ, OM, ref)[:, :, 0]
    return lay - ws


def main() -> int:
    mod = model(1.0)
    s_p, s_s = mod.complex_slowness_p(), mod.complex_slowness_s()
    ref = ReferenceMedium(1 / s_p[PLANE], 1 / s_s[PLANE], mod.rho[PLANE])

    print("=" * 78)
    print("MEASUREMENT -- the class-A (Dz != 0) reverberation table")
    print(f"  plane {PLANE}, dz = {DZ} km, omega = {OM:.3f}")
    print("  NOTE: real-space table size carries no n_k. This measures BUILD, not fit.")
    print("=" * 78)

    print("\n[A] CONVERGENCE of DeltaG0 under transverse refinement")
    print("    self-convergence: relative change from the previous rule")
    print(f"    {'kr*pitch':>9} {'n/axis':>7} {'nodes':>9} {'rel change':>12} {'sec':>7}")
    prev = None
    for mult in (10.0, 20.0):
        for n_ax in (16, 32, 64, 128):
            k1, dk = tensor_rule(mult / PITCH, n_ax)
            kxg, kyg = np.meshgrid(k1, k1, indexing="ij")
            kx, ky = kxg.ravel(), kyg.ravel()
            t0 = time.perf_counter()
            d = dg0_on_grid(mod, kx, ky, ref)
            dt = time.perf_counter() - t0
            # real-space value at one pitch along x, zero along y
            ph = np.exp(1j * kx * PITCH)
            val = np.einsum("nab,n->ab", d, ph) * dk * dk / (2 * np.pi) ** 2
            chg = (
                float("nan")
                if prev is None
                else float(np.abs(val - prev).max() / max(np.abs(val).max(), 1e-300))
            )
            prev = val
            print(f"    {mult:9.0f} {n_ax:7d} {kx.size:9d} {chg:12.3e} {dt:7.2f}")
        prev = None
        print()

    print("    A change that STOPS falling means the rule is resolved. A change")
    print("    that never falls means the reverberation is as expensive as the")
    print("    full kernel -- in which case STOP and report, do not add nodes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
