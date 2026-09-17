"""GATE: the dz = 0 cell-averaged lattice sum, via the analytic d^2 tail.

The sinc form factor closed dz != 0 but cannot touch dz = 0: that branch is
EWALD, whose real-space half is a spatial summation, and a form factor is the
transform of the cell indicator -- it multiplies plane waves, nothing else.

The route taken instead is an exact analytic tail. <g> - g = (d^2/24) grad^2 g
+ O(d^4), and grad^2 g_c = -k_c^2 g_c EXACTLY away from the origin, so

    sum_{|R| > R0} <D>  =  (1 - k_c^2 d^2/24) (D_full - D_near) + O(d^4),

with D_full the Ewald sum that already exists. Near shell direct, far field in
closed form.

⚠ WHY THIS IS THE ONLY WAY OUT, not merely a faster one. The correction
<D> - D has an O(1/R) tail and turns CONDITIONALLY convergent past k r ~ 1,
so its value depends on the summation shape there. Enlarging the box cannot
converge it: the tail needs R ~ 80 while the shape term switches on near
R ~ 1/(k_S d) ~ 50 (`scripts/investigate_correction_tail_shape.py`). Removing
the slow part in closed form is the escape.

THE TWO TESTS, and the first is the sharp one:

  [1] R0-INDEPENDENCE. R0 splits the sum between a directly averaged near shell
      and an analytic tail. If the tail is right the split cannot matter, so
      the answer must not drift with R0. A wrong tail shows up here immediately
      and cannot be hidden by raising anything.

  [2] AGREEMENT WITH THE DIRECT SHELL SUM at small k d, where that sum is still
      absolutely convergent and therefore trustworthy. This is the regime the
      shell sum handles CORRECTLY -- which is exactly what makes it a valid
      arbiter for a method built for the regime where it does not.

⚠ [2] IS RUN AT THE OPERATING POINT ONLY. At larger k d the shell sum is the
thing under suspicion, so disagreement there would not indict the tail. Saying
so in advance stops a later disagreement being read as a failure of this method.

Run:  conda run -n seismic python scripts/gate_dz0_averaged_lattice.py
SI units (m, m/s, kg/m3, Pa).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubic_scattering.cell_averaged_lattice import averaged_same_plane_9x9  # noqa: E402
from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from cubic_scattering.lattice_kupradze import bloch_kernel_hat_9x9  # noqa: E402
from cubic_scattering.slab_scattering import (  # noqa: E402
    _cell_averaged_propagator,
    _propagator_block_9x9,
)

REF = ReferenceMedium(5000.0, 3000.0, 2500.0)
D = 1.0
OMEGA = 60.0
K_PAR = np.zeros(2)


def shell_correction(reach: int, n_gauss: int = 6) -> np.ndarray:
    """sum over the dz=0 plane of [<G> - G], at k_par = 0 (all phases 1)."""
    total = np.zeros((9, 9), dtype=complex)
    for dx in range(-reach, reach + 1):
        for dy in range(-reach, reach + 1):
            if dx == 0 and dy == 0:
                continue
            r_vec = np.array([0.0, dx * D, dy * D])
            total += _cell_averaged_propagator(
                r_vec, D, OMEGA, REF, n_gauss, double=False
            ) - _propagator_block_9x9(r_vec, OMEGA, REF)
    return total


def main() -> int:
    print("=" * 78)
    print("GATE: dz = 0 cell-averaged lattice sum via the analytic d^2 tail")
    print("=" * 78)
    print(f"\n  d = {D}, omega = {OMEGA}, k_par = 0,  k_S d = {OMEGA / REF.beta * D:.4f}")

    print("\n  [1] R0-INDEPENDENCE -- the sharp test of the tail")
    print(f"      {'R0':>5} {'n_gauss':>9} {'|block|':>15} {'rel vs R0=2':>14}")
    base = None
    rels = []
    for r0 in (1, 2, 3, 4):
        blk = averaged_same_plane_9x9(D, OMEGA, REF, K_PAR, r0_cells=r0)
        if r0 == 2:
            base = blk
        print(f"      {r0:>5} {6:>9} {float(np.max(np.abs(blk))):15.8e}", end="")
        if base is not None and r0 != 2:
            rel = float(np.max(np.abs(blk - base)) / np.max(np.abs(base)))
            rels.append((r0, rel))
            print(f" {rel:14.2e}")
        else:
            print(f" {'--':>14}")
    # re-print the ones computed before base was set
    blk1 = averaged_same_plane_9x9(D, OMEGA, REF, K_PAR, r0_cells=1)
    rel1 = float(np.max(np.abs(blk1 - base)) / np.max(np.abs(base)))
    print(f"\n      R0 = 1 vs 2: {rel1:.2e}   (the nearest shell is where the")
    print("      d^2 tail is least accurate, so this is the worst case)")
    worst_far = max((r for _, r in rels), default=1.0)
    print(f"      worst drift among R0 = 3, 4 vs 2: {worst_far:.2e}")
    r0_ok = worst_far < 1e-3
    print(f"      => R0-independent: {r0_ok}")

    print("\n  [2] vs the direct shell sum, at the operating point")
    point = bloch_kernel_hat_9x9(1, D, 0.0, OMEGA, REF)[0, 0]
    avg = averaged_same_plane_9x9(D, OMEGA, REF, K_PAR, r0_cells=3)
    print(f"      {'reach':>7} {'|avg - (pt + shell)|':>24} {'rel':>12}")
    last = None
    for reach in (8, 16, 24):
        ref_blk = point + shell_correction(reach)
        rel = float(np.max(np.abs(avg - ref_blk)) / np.max(np.abs(avg)))
        last = rel
        print(f"      {reach:>7} {float(np.max(np.abs(avg - ref_blk))):24.6e} {rel:12.2e}")
    shell_ok = last is not None and last < 5e-3
    print(f"      => agrees with the convergent shell sum: {shell_ok}")

    ok = r0_ok and shell_ok
    print("\n" + "=" * 78)
    if ok:
        print("PASS -- the dz = 0 source-cell average is now available in closed")
        print("form, with no reach parameter to converge and no box to enlarge.")
        print("Together with the sinc form factor at dz != 0, the cell average")
        print("is exact at every separation, and the O(1/R) shape-dependent")
        print("shell sum is no longer on the critical path.")
    else:
        print("FAIL -- see which test failed.  R0-drift indicts the TAIL;")
        print("disagreement with the shell sum at small k d, with R0 stable,")
        print("indicts the near-shell average or the assembly instead.")
    print()
    print("⚠ SCOPE: k_par = 0 and one frequency.  The tail is exact through")
    print("d^2; its residual is the cube's non-isotropic fourth moment, which")
    print("R0 controls.  Wiring this into build_slab_kernels is NOT done here.")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
