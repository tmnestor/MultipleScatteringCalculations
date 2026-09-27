#!/usr/bin/env python3
"""MEASUREMENT: is the periodic thesis gate's dressing consistent with its kernel?

CORRECTION, 27 September 2026: every number below was measured while the gate's solve
ignored its zeroed observation-plane T-matrix (no ``T_local``), so the observation plane
scattered and every arm carried a spurious error of ~3.6e-4.  Fixed in the gate; rerun:
    [1] still holds (dressing on a uniform background changes nothing, 1.4e-17); [3] with
    the Ewald kernel [T1]/[T3] = 3.82x, [T2]/[T3] = 42165x -- the arms no longer exchange.

WHY.  With the exact Ewald lateral sum in place of the truncated one, the gate's
arms SWAP (``measure_thesis_gate_floor``): the thesis ordering goes from 0.94x
to 3.66x the floor while dress-after goes from 3.28x to 1.62x, although the
floor itself moves only ~20%.  A more accurate whole-space kernel should not do
that.  The dressing -- DeltaG0(k_par -> 0) / d^2 added to kernel_hat[dz][0, 0] --
is the same for both kernels, so the change must come from how it combines
with the kernel.

THE SIMPLIFICATION.  The medium is laterally uniform and the illumination is at
normal incidence, so only the k = 0 (specular) component kernel_hat[dz][0, 0]
acts: three 9x9 blocks, dz = -1, 0, +1.

CHECKS:
  [1] on a UNIFORM background the dressing must change nothing, with either kernel;
  [2] each kernel's specular block against the others, and against the point
      propagator's Weyl sum ws(dz)/d^2 that the dressing subtracts;
  [3] SWAP: the Ewald kernel with the truncated kernel's specular blocks, and the
      reverse.  If the arms follow the specular blocks, those blocks are the
      whole story.

Run:  conda run -n seismic python scripts/measure_thesis_gate_dressing.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from cubic_scattering.slab_scattering import SlabGeometry, build_slab_kernels  # noqa: E402
from cubic_scattering.sweep_kernels import same_depth_kernel_9x9, vertical_kernel_9x9  # noqa: E402
from scripts import gate_thesis_formulation_periodic as gate  # noqa: E402

TRUNC = dict(volume_averaged=True, periodic=True)
EWALD = dict(volume_averaged=True, periodic=True, lattice_ewald=True)


def specular(kw: dict) -> np.ndarray:
    """kernel_hat[:, 0, 0] (the k = 0 blocks) of the gate's geometry, per dz index."""
    geom = SlabGeometry(M=gate.M, N_z=gate.N_Z + 1, a=gate.A_HALF)
    ref = ReferenceMedium(gate.A0, gate.B0, gate.R0)
    kh = build_slab_kernels(geom, gate.OM, ref, **kw)
    return np.array(kh[:, 0, 0])


def weyl_point(n_z: int) -> np.ndarray:
    """ws(dz)/d^2: the point propagator's Weyl sum at k_par -> 0, as the dressing subtracts it."""
    ref = ReferenceMedium(gate.A0, gate.B0, gate.R0)
    out = []
    for k in range(2 * n_z - 1):
        dz = k - (n_z - 1)
        if dz == 0:
            ws = same_depth_kernel_9x9(np.array([gate.EPS]), 0.0, gate.OM, ref)[:, :, 0]
        else:
            ws = vertical_kernel_9x9(np.array([gate.EPS]), 0.0, dz * gate.D, gate.OM, ref)[:, :, 0]
        out.append(ws / gate.D**2)
    return np.array(out)


def with_specular(kw: dict, spec: np.ndarray) -> np.ndarray:
    """The kernel of kw, with its k = 0 blocks replaced by spec."""
    geom = SlabGeometry(M=gate.M, N_z=gate.N_Z + 1, a=gate.A_HALF)
    ref = ReferenceMedium(gate.A0, gate.B0, gate.R0)
    kh = build_slab_kernels(geom, gate.OM, ref, **kw).copy()
    kh[:, 0, 0] = spec
    return kh


def arms(kw: dict, kernel_override=None) -> tuple[float, float, float]:
    """(|T1|, |T2|, |T3|) absolute, with the gate's own runs; optional replacement kernel."""
    if kernel_override is not None:
        orig = gate.build_slab_kernels
        gate.build_slab_kernels = lambda *_args, **_kwargs: kernel_override.copy()  # type: ignore[assignment]
    try:
        t1, s1 = gate._run(dressed=True, uniform=False, kernel_kw=kw)
        t2, _ = gate._run(dressed=False, uniform=False, kernel_kw=kw)
        t3, s3 = gate._run(dressed=True, uniform=True, kernel_kw=kw)
    finally:
        if kernel_override is not None:
            gate.build_slab_kernels = orig
    return t1 * s1, t2 * s1, t3 * s3


def main() -> int:
    """Run the three checks.

    Returns:
        0.
    """
    print("=" * 78)
    print("PERIODIC THESIS GATE: DRESSING VERSUS KERNEL, SPECULAR (k = 0) BLOCKS")
    print("=" * 78)
    n_z = gate.N_Z + 1

    print("\n[1] dressing on a UNIFORM background must change nothing")
    for name, kw in (("truncated", TRUNC), ("Ewald", EWALD)):
        dressed, s = gate._run(dressed=True, uniform=True, kernel_kw=kw)
        plain, _ = gate._run(dressed=False, uniform=True, kernel_kw=kw)
        change = abs(dressed - plain) * s
        print(f"    {name:9s}: |dressed - undressed| error change {change:.2e} (floor {plain * s:.2e})")

    print("\n[2] specular blocks: kernels against each other and against the point Weyl sum ws/d^2")
    sp_t, sp_e, wp = specular(TRUNC), specular(EWALD), weyl_point(n_z)
    for k in range(2 * n_z - 1):
        dz = k - (n_z - 1)
        scale = np.abs(sp_e[k]).max()
        print(
            f"    dz = {dz:+d}: |trunc - Ewald| {np.abs(sp_t[k] - sp_e[k]).max() / scale:.2e}   "
            f"|Ewald - ws/d^2| {np.abs(sp_e[k] - wp[k]).max() / scale:.2e}   "
            f"|trunc - ws/d^2| {np.abs(sp_t[k] - wp[k]).max() / scale:.2e}   (of max|Ewald block|)"
        )

    print("\n[3] SWAP the specular blocks; the rest of each kernel is left as it is")
    rows = [
        ("truncated kernel", TRUNC, None),
        ("Ewald kernel", EWALD, None),
        ("Ewald kernel, truncated specular", EWALD, with_specular(EWALD, sp_t)),
        ("truncated kernel, Ewald specular", TRUNC, with_specular(TRUNC, sp_e)),
    ]
    for name, kw, override in rows:
        a1, a2, a3 = arms(kw, override)
        print(f"    {name:34s}  [T1]/[T3] {a1 / a3:5.2f}x   [T2]/[T3] {a2 / a3:5.2f}x   floor {a3:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
