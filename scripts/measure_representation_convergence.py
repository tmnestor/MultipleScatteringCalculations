"""MEASUREMENT (1): does the representation converge under REFINEMENT, and is the
voxel T-matrix responsible?

═══ ANSWERED, 2026-09-15 (evening). READ THIS BEFORE THE ORIGINAL TEXT BELOW ═══
IT DOES CONVERGE. The apparent non-convergence had two causes, neither of them
the T-matrix:

  1. the periodic lateral sum was TRUNCATED, and its artifact accumulated plane
     by plane -- which is why the error GREW with n_z rather than sitting at a
     floor. `lattice_ewald=True` sums it exactly.
  2. the contact propagator was a MIDPOINT rule. The Galerkin contact
     correction removes what is left of the growth.

With both: 7.73e-4, 8.35e-4, 8.51e-4, 8.58e-4 over n_z = 1, 2, 4, 8 -- flat.
Against 3.95e-3 -> 2.62e-2 originally. What survives is a scale-invariant
residual of ~8.6e-4, which is a genuine per-cell error and the only thing still
to explain.

The original reasoning is kept below because its EXONERATION of the T-matrix was
correct and independently established, and because the measurements it proposed
are the ones that settled the question. Its quantitative claims, however, were
all taken on the truncated kernel.
════════════════════════════════════════════════════════════════════════════════

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
    build_slab_kernels,
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


def _arm(
    n_z: int,
    c: float,
    *,
    volume_averaged: bool = False,
    va_radius: int = 1,
    lattice_ewald: bool = True,
) -> tuple:
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

    # The kernel is built explicitly so va_radius can be swept; passing it in
    # bypasses the solver's own build, which fixes the radius at 1.
    kh = build_slab_kernels(
        geom,
        OMEGA,
        REF,
        volume_averaged=volume_averaged,
        n_orders=2,
        periodic=True,
        va_radius=va_radius,
        lattice_ewald=lattice_ewald,
    )
    res = compute_slab_scattering(
        geom,
        mat,
        OMEGA,
        np.array([1.0, 0.0, 0.0]),
        "P",
        periodic=True,
        gmres_tol=1e-12,
        kernel_hat=kh,
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

    # THE KERNEL IS NOW A VARIABLE, because the original run of this campaign
    # used the TRUNCATED lateral sum. At M = 4 that truncation was 63% of the
    # measured one-plane floor, and it sits underneath every conclusion drawn
    # here. Both kernels are run side by side so the question "was the
    # non-convergence the truncation accumulating plane by plane?" is answered
    # by measurement rather than by re-reading old numbers.
    for c, label in ((1.0, "working contrast"), (1.0e-3, "1000x weaker -- Born regime")):
        print(f"\n  contrast x{c:g}  ({label})")
        print(
            f"    {'n_z':>4} {'a (m)':>8} {'ka':>8} {'TRUNCATED':>11} {'ratio':>7}"
            f" {'EWALD':>11} {'ratio':>7} {'T nonlin':>10}"
        )
        prev_old = prev_new = None
        for n_z in N_Z_LADDER:
            err_old, ka, nonlin = _arm(n_z, c, lattice_ewald=False)
            err_new, _, _ = _arm(n_z, c, lattice_ewald=True)
            r_old = "" if prev_old is None else f"{err_old / prev_old:7.2f}"
            r_new = "" if prev_new is None else f"{err_new / prev_new:7.2f}"
            flag = "" if ka < 0.3 else "  <-- ka OUT OF RANGE"
            print(
                f"    {n_z:4d} {H_PHYS / (2 * n_z):8.4f} {ka:8.4f} {err_old:11.4e} {r_old:>7}"
                f" {err_new:11.4e} {r_new:>7} {nonlin:10.3e}{flag}"
            )
            prev_old, prev_new = err_old, err_new

    # ---- (2) DOES GALERKIN AVERAGING FURTHER OUT RESTORE CONVERGENCE? -------
    # The point propagator is a MIDPOINT rule for what should be a doubly
    # volume-averaged (Galerkin) operator between cells. `build_slab_kernels`
    # applies the averaged object only on the nearest-neighbour shell; va_radius
    # extends it. If the non-convergence is the contact quadrature, pushing the
    # averaging outwards should bend the curve down.
    print("\n\n  (2) VOLUME-AVERAGED (GALERKIN) PROPAGATOR, radius swept")
    print("      point = midpoint rule; radius r = averaged out to Chebyshev r cells")
    print(f"    {'n_z':>4} {'point':>11} {'r = 1':>11} {'r = 2':>11} {'r = 3':>11}")
    for n_z in (1, 2, 4, 8):
        row = [f"    {n_z:4d}"]
        for va_r in (0, 1, 2, 3):
            try:
                err, _, _ = _arm(n_z, 1.0, volume_averaged=(va_r > 0), va_radius=max(va_r, 1))
                row.append(f" {err:11.4e}")
            except (RuntimeError, ValueError, TypeError):
                # inter_voxel_propagator_9x9 raises "is not a nearest neighbour":
                # the volume-averaged object EXISTS only on the contact shell.
                # Extending it is a new set of tables, not a parameter.
                row.append(f" {'n/a':>11}")
        print("".join(row))
    print("\n      r >= 2 is 'n/a' BY CONSTRUCTION, not by cost: the volume-averaged")
    print("      object exists only on the contact shell. That is not the limitation")
    print("      it first appears, because contact is exactly where the midpoint rule")
    print("      fails -- beyond it (cell size)/(separation) < 1 and falls, so the")
    print("      point propagator is already good there.")
    print("\n      WITHDRAWN: this section previously concluded that 'Galerkin")
    print("      averaging at contact LOWERS the error but does NOT restore")
    print("      convergence -- it still grows with refinement'. That was measured")
    print("      on the TRUNCATED lateral sum, whose own growth dominated and which")
    print("      no contact fix could have removed. On the exact lattice sum the")
    print("      r = 1 column is FLAT. The growth is gone.")

    print("\n" + "=" * 88)
    print("  WHAT THE RE-RUN SHOWS. The non-convergence had TWO causes, and the")
    print("  T-matrix was neither:")
    print()
    print("    1. the TRUNCATED lateral sum. Its artifact ACCUMULATED plane by")
    print("       plane, which is why the error grew with n_z instead of merely")
    print("       sitting at a floor -- the question left open when the defect was")
    print("       found. Exact summation cuts the n_z = 16 error from 2.62e-2 to")
    print("       5.06e-3 and turns the ratio from 1.29, still climbing, into 1.05.")
    print("    2. the CONTACT MIDPOINT RULE. Adding the Galerkin contact correction")
    print("       to the exact sum flattens it outright: 7.73e-4, 8.35e-4, 8.51e-4,")
    print("       8.58e-4 across n_z = 1, 2, 4, 8.")
    print()
    print("  SO THE SCHEME CONVERGES. What it converges TO is not zero: a residual")
    print("  of about 8.6e-4 survives refinement. That is the genuine scale-")
    print("  invariant per-cell error and it is now the only thing left to explain")
    print("  -- with the same two suspects as before, but facing a flat curve")
    print("  rather than a growing one:")
    print("    (a) the averaged object's dynamic content is truncated at O(w^4);")
    print("    (b) a space-filling lattice of ISOLATED-cube T-matrices may need a")
    print("        lattice renormalisation to reproduce the continuum -- the same")
    print("        class of correction as the sphere-packing Delta -> Delta/phi")
    print("        already established in this project.")
    print()
    print("  'ratio' is the error multiplier per refinement step, each step HALVING")
    print("  the cell size. Ratio ~ 1 is a scale-invariant per-cell error; ratio > 1")
    print("  means the error is also ACCUMULATING over cells, which is what the")
    print("  truncated column still shows and the exact one no longer does.")
    print()
    print("  'T nonlin' differs by 1000x between the two contrast arms while the")
    print("  ratio columns do not, so the T-matrix's nonlinear content is exonerated")
    print("  -- as before, but now without a confounded kernel underneath it.")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
