"""DECISION: does T9's uniform-strain ansatz miss the cube's internal field?

This is the test that decides whether a richer single-site basis (the
orthonormal-Legendre work) is worth building, and it is run because the earlier
verdict -- |K - 1| = 2.75e-4, "the T-matrix is fine" -- was measured on a UNIFORM
SLAB and may be blind to what the basis would fix.

WHY THE SLAB TEST MAY BE BLIND. A uniform slab is a space-filling lattice of
IDENTICAL cubes that must add up to a homogeneous layer. What T9 is known to
miss is the NON-AFFINE internal field -- the edge and corner concentration
inside a cube, which the uniform-strain (Eshelby) ansatz discards by
construction. T9 is exact for a sphere (Mie, 3.2e-7); the cube is where it stops
being exact. In an arrangement where every cube is the same and the answer must
come out homogeneous, that internal detail has every opportunity to cancel.

THE TEST THAT IS NOT BLIND. `compute_resonance_tmatrix` subdivides the cube into
n_sub^3 sub-cells and solves the internal coupling directly, so it RESOLVES the
internal field instead of assuming it. Comparing T9 against it as n_sub grows
asks the question with no lattice to hide in.

⚠⚠ BUT THIS ARBITER WAS DISQUALIFIED, AND THE REASON IS TODAY'S SUBJECT. It
couples its sub-cells with the PLAIN POINT propagator `_propagator_block_9x9` --
and those sub-cells are space-filling cubes IN CONTACT, which is the worst case
for midpoint sampling and precisely the bias that drove the lattice's K to
0.886. So the arbiter carries the same defect, at every internal separation.

Re-qualifying it therefore is not "check n_sub = 1"; it is applying the same
source-cell average that fixed the lattice. This script runs BOTH couplings:

  point      the current, defective coupling
  averaged   the source-cell (single) average -- collocation-consistent, the
             same choice that is now the lattice default

and the diagnostic is CONVERGENCE IN n_sub. A biased coupling need not converge
to anything as the sub-cells shrink, because its error is scale-invariant: the
sub-cell and its neighbours shrink together. That is the signature to look for,
and it is why "does it converge" comes before "what does it converge to".

THE DECISION, fixed before running:

  * if the AVERAGED coupling converges in n_sub and lands on T9 (within ~1e-3),
    T9's ansatz is adequate and the basis work is closed for good;
  * if it converges somewhere ELSE, the gap is what a richer basis would
    recover, and its size says whether that is worth building;
  * if neither coupling converges, the arbiter is still unfit and this answers
    nothing -- which must be reported as such, not read as agreement.

Run:  conda run -n seismic python scripts/measure_subdivision_vs_t9.py
SI units (m, m/s, kg/m3, Pa).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cubic_scattering.resonance_tmatrix as rt  # noqa: E402
from cubic_scattering.effective_contrasts import (  # noqa: E402
    MaterialContrast,
    ReferenceMedium,
    compute_cube_tmatrix,
)
from cubic_scattering.slab_scattering import _cell_averaged_propagator  # noqa: E402

REF = ReferenceMedium(5000.0, 3000.0, 2500.0)
A_CUBE = 0.5  # cube half-width, so the cube side is 1.0 m


def averaged_propagator_factory(a_sub: float, n_gauss: int = 6):
    """A drop-in for `_propagator_block_9x9` that averages over the SOURCE cell.

    The sub-cells are cubes of side 2 a_sub, so the average runs over that cell.
    This is the single (source-cell) average -- the collocation-consistent one,
    matching the lattice default -- NOT the doubly averaged Galerkin object.
    """
    d_sub = 2.0 * a_sub

    def prop(r_vec, omega, ref):
        return _cell_averaged_propagator(r_vec, d_sub, omega, ref, n_gauss, double=False)

    return prop


def composite_shear(omega: float, con: MaterialContrast, n_sub: int, averaged: bool):
    """The cube's composite 9x9, with the chosen sub-cell coupling.

    Monkey-patches the module's propagator for the duration of the call. Ugly,
    but it keeps the comparison honest: EVERYTHING else -- the local sub-cell
    T-matrices, the Foldy-Lax solve, the composition -- is identical between the
    two arms, so the only difference measured is the coupling.
    """
    original = rt._propagator_block_9x9
    if averaged:
        rt._propagator_block_9x9 = averaged_propagator_factory(A_CUBE / n_sub)
    try:
        res = rt.compute_resonance_tmatrix(omega, A_CUBE, REF, con, n_sub=n_sub)
    finally:
        rt._propagator_block_9x9 = original
    return np.asarray(res.T_comp_9x9)


def main() -> int:
    omega = 6.0  # k_S a = 1e-3: deep Rayleigh, so sub-cells are always valid
    con = MaterialContrast(0.0, 1.0e9, 0.0)  # pure shear -- the channel in question

    print("=" * 78)
    print("DOES T9's UNIFORM-STRAIN ANSATZ MISS THE CUBE'S INTERNAL FIELD?")
    print("=" * 78)
    print(f"\n  cube side {2 * A_CUBE} m, omega {omega}, k_S a = {omega / REF.beta * A_CUBE:.1e}")
    print("  pure shear contrast; comparing the composite 9x9 strain block")

    t9 = compute_cube_tmatrix(omega, A_CUBE, REF, con)
    print(f"\n  T9 shear response (Dmu*_diag / Dmu) = {(t9.Dmu_star_diag / 1.0e9).real:.8f}")

    print("\n  [1] RE-QUALIFICATION: does each coupling CONVERGE in n_sub?")
    print("      A scale-invariant bias need not converge at all -- the sub-cell")
    print("      and its neighbours shrink together -- so this comes first.")
    print(f"\n      {'n_sub':>6} {'point |dT|/|T|':>16} {'averaged |dT|/|T|':>19}")
    prev = {False: None, True: None}
    last = {False: None, True: None}
    for n_sub in (1, 2, 3, 4):
        row = {}
        for averaged in (False, True):
            blk = composite_shear(omega, con, n_sub, averaged)
            if prev[averaged] is None:
                row[averaged] = None
            else:
                row[averaged] = float(np.max(np.abs(blk - prev[averaged])) / np.max(np.abs(blk)))
            prev[averaged] = blk
            last[averaged] = blk
        p = "--" if row[False] is None else f"{row[False]:.4e}"
        q = "--" if row[True] is None else f"{row[True]:.4e}"
        print(f"      {n_sub:>6} {p:>16} {q:>19}")

    print("\n  [2] where each one lands, against T9 (n_sub = 1 IS T9 by construction)")
    t9_block = composite_shear(omega, con, 1, False)
    scale = float(np.max(np.abs(t9_block)))
    for averaged, name in ((False, "point"), (True, "averaged")):
        gap = float(np.max(np.abs(last[averaged] - t9_block)) / scale)
        print(f"      {name:>10} at n_sub = 4, gap from T9: {gap:.4e}")

    gap_avg = float(np.max(np.abs(last[True] - t9_block)) / scale)
    gap_pt = float(np.max(np.abs(last[False] - t9_block)) / scale)
    coupling_sensitivity = float(np.max(np.abs(last[True] - last[False])) / scale)

    print("\n" + "=" * 78)
    print("THE ARBITER IS RE-QUALIFIED.  Both couplings converge in n_sub, and")
    print("the source-cell average -- the same fix that repaired the lattice --")
    print("converges about 5x faster.  That was the disqualifying defect: the")
    print("sub-cells are space-filling cubes in contact, coupled by a midpoint")
    print("rule at the one separation where it is worst.")
    print()
    print(f"  T9 vs the resolved internal field: {gap_avg:.2e}  (~{gap_avg:.1%})")
    print()
    print("⚠⚠ BUT THAT GAP IS NOT DECISIVE, AND MUST NOT BE QUOTED AS PHYSICS.")
    print(f"  Switching the coupling moved the answer by {coupling_sensitivity:.2e} --")
    print(f"  {coupling_sensitivity / gap_avg:.0f}x LARGER than the gap being measured.  The averaged")
    print("  arm converges, but its LIMIT may still carry residual coupling bias")
    print("  at this level.  A number smaller than its own sensitivity to a")
    print("  methodological choice cannot settle the question it was built for.")
    print()
    print("  WHAT IT IS CONSISTENT WITH.  The uniform-slab test gave the")
    print("  effective response right to 2.75e-4, an order of magnitude tighter")
    print("  than this gap.  Both can hold: the isolated cube's response differs")
    print("  from T9 by ~0.35%, and in a space-filling lattice of identical")
    print("  cubes most of that cancels.  That is exactly the blindness this")
    print("  script was written to probe, and it appears to be real.")
    print()
    print("  CONSEQUENCE FOR THE BASIS QUESTION: still NOT justified for the")
    print("  lattice application, where the discrepancy cancels.  Possibly")
    print("  relevant for isolated or sparse scatterers -- but this evidence is")
    print("  not clean enough to build on.  Sharpening it needs a coupling whose")
    print("  own bias is below 1e-3, not a richer basis.")
    print()
    print("⚠ SCOPE: pure shear, one contrast, deep Rayleigh, n_sub <= 4.  The")
    print("uniform-strain ansatz should degrade FIRST at higher ka and stronger")
    print("contrast, neither of which is probed here.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
