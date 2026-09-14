"""MEASUREMENT (1): does the representation converge under REFINEMENT, and is the
voxel T-matrix responsible?

THE OBSERVATION THIS EXISTS TO NAIL DOWN. Re-reading `measure_periodic_floor` at
FIXED PHYSICAL THICKNESS, refining the voxelisation makes the answer WORSE:

    H = 4 m, 1 plane (a = 2.0) -> 4.30e-3      H = 2 m, 1 plane -> 3.99e-3
    H = 4 m, 2 planes (a = 1.0) -> 8.41e-3     H = 2 m, 2 planes -> 8.30e-3

Same medium, same thickness, finer cells, double the error. Refinement is the one
tool a discretisation has for buying accuracy, and here it buys error. That is a
NON-CONVERGENCE claim and it deserves more than two points per thickness, so this
script takes it to n_z = 1, 2, 4, 8, 16.

IS IT THE T-MATRIX? The evidence already says probably not, and this measures it
rather than arguing it:

  * the floor is FLAT IN ka, so it is not the T-matrix's (ka)^2/(ka)^4 expansion
    error;
  * the floor is LINEAR IN CONTRAST (c^1.00, measured), so it does not live in
    the T-matrix's nonlinear content -- self-interaction and beyond enter at c^2;
  * at FIRST ORDER a space-filling tiling should have NO discretisation error at
    all. The Born response of a slab is Int G(obs,x) Delta(x) psi(x) dx, and
    splitting that integral into cubes is just splitting an integral into
    subvolumes. A c^1 error therefore lives in the QUADRATURE of that integral,
    not in the contrast.

THE TEST. Run the refinement at the working contrast AND at a contrast 1000x
weaker, which is deep in the Born regime where the T-matrix is its own linear
part. If the doubling is identical in both, the T-matrix's nonlinear content
cannot be responsible and the propagator quadrature is left holding it.

The degree of T-matrix nonlinearity at each contrast is REPORTED, not assumed:
||T(c) / c - T_Born|| / ||T_Born||, with T_Born obtained as the measured limit
T(eps c)/(eps c). If that is large at the working contrast and negligible at the
weak one, while the refinement behaviour is unchanged, the exoneration is clean.

WHAT WOULD CONVICT THE T-MATRIX INSTEAD: the weak-contrast arm converging while
the working-contrast arm does not.

Run:  conda run -n seismic python scripts/measure_representation_convergence.py
SI units (m, m/s, kg/m3, Pa).
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubic_scattering.effective_contrasts import (  # noqa: E402
    MaterialContrast,
    ReferenceMedium,
)
from cubic_scattering.slab_scattering import (  # noqa: E402
    SlabGeometry,
    SlabMaterial,
    compute_slab_scattering,
    compute_slab_tmatrices,
    kennett_reference_rpp,
    slab_rpp_periodic,
)

REF = ReferenceMedium(5000.0, 3000.0, 2500.0)
D_LAM, D_MU, D_RHO = 2.0e9, 1.0e9, 100.0  # the validated moderate contrast
OMEGA = 60.0
H_PHYS = 4.0  # metres. HELD FIXED while the voxelisation is refined.
M = 4
N_Z_LADDER = (1, 2, 4, 8, 16)
EPS_LIN = 1e-6  # for extracting the Born limit of the T-matrix as a measured fact


def _arm(n_z: int, c: float, *, volume_averaged: bool = False) -> tuple:
    """(relative error vs exact Kennett, ka, T-matrix nonlinearity fraction).

    The physical slab is H_PHYS thick in every row; only the number of cells
    across it changes. a = H/(2 n_z) makes the pitch d = 2a = H/n_z, so
    n_z * d = H exactly.
    """
    a_half = H_PHYS / (2.0 * n_z)
    geom = SlabGeometry(M=M, N_z=n_z, a=a_half)
    ones = np.ones((n_z, M, M))
    mat = SlabMaterial(Dlambda=c * D_LAM * ones, Dmu=c * D_MU * ones, Drho=c * D_RHO * ones, ref=REF)
    t0 = compute_slab_tmatrices(geom, mat, OMEGA)

    # THE BORN LIMIT AS A MEASURED QUANTITY, not a hand-written formula. T is
    # linear in the contrast to leading order, so T(eps c)/eps -> the Born
    # T-matrix in the code's OWN conventions -- no chance of a sign or Voigt
    # convention slipping in through a reimplementation.
    mat_lin = SlabMaterial(
        Dlambda=EPS_LIN * c * D_LAM * ones,
        Dmu=EPS_LIN * c * D_MU * ones,
        Drho=EPS_LIN * c * D_RHO * ones,
        ref=REF,
    )
    t_born = compute_slab_tmatrices(geom, mat_lin, OMEGA) / EPS_LIN
    nonlin = float(np.abs(t0 - t_born).max() / np.abs(t_born).max())

    res = compute_slab_scattering(
        geom,
        mat,
        OMEGA,
        np.array([1.0, 0.0, 0.0]),
        "P",
        periodic=True,
        gmres_tol=1e-12,
        volume_averaged=volume_averaged,
        n_orders=2,
    )
    r_lat = slab_rpp_periodic(res, t0)
    r_ken = kennett_reference_rpp(REF, MaterialContrast(c * D_LAM, c * D_MU, c * D_RHO), H_PHYS, OMEGA)
    err = abs(r_lat - r_ken) / abs(r_ken)
    return err, OMEGA / REF.alpha * a_half, nonlin


def main() -> int:
    print("=" * 88)
    print("MEASUREMENT -- does the representation CONVERGE under refinement?")
    print(f"  physical slab HELD FIXED at H = {H_PHYS} m; only the cell size changes")
    print("  a convergent scheme has the error FALL as n_z rises")
    print("=" * 88)

    for c, label in ((1.0, "working contrast"), (1.0e-3, "1000x weaker -- Born regime")):
        print(f"\n  contrast x{c:g}  ({label})")
        print(f"    {'n_z':>4} {'a (m)':>8} {'ka':>8} {'rel err':>11} {'ratio':>7} {'T nonlin':>10}")
        prev = None
        for n_z in N_Z_LADDER:
            err, ka, nonlin = _arm(n_z, c)
            ratio = "" if prev is None else f"{err / prev:7.2f}"
            flag = "" if ka < 0.3 else "  <-- ka OUT OF RANGE"
            print(
                f"    {n_z:4d} {H_PHYS / (2 * n_z):8.4f} {ka:8.4f} {err:11.4e} "
                f"{ratio:>7} {nonlin:10.3e}{flag}"
            )
            prev = err

    print("\n" + "=" * 88)
    print("  Reading it. 'ratio' is the error multiplier per refinement step, each")
    print("  step HALVING the cell size. A convergent scheme shows ratio < 1 --")
    print("  0.25 for second-order, 0.5 for first. Ratio ~ 2 means every halving")
    print("  DOUBLES the error, which is the signature of a per-cell error that is")
    print("  scale-invariant: for touching cells the ratio (cell size)/(separation)")
    print("  is 1 at EVERY scale, so refinement cannot dilute it and simply adds")
    print("  more cells each carrying the same relative error.")
    print("\n  'T nonlin' is how far the T-matrix is from its own Born limit. If the")
    print("  two contrasts differ by 1000x in that column and NOT in the ratio")
    print("  column, the T-matrix's nonlinear content is exonerated and what")
    print("  remains is the quadrature of the propagator.")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
