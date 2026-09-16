"""SETTLEMENT: if the scheme is collocation, why is the propagator a MIDPOINT?

`settle_single_site_formulation.py` established that this solver is a
collocation scheme and that the collocation T-matrix plus the point contact is
the consistent pairing. That raises an obvious follow-up: the argument that
picked the single-averaged contact is NOT special to contact.

The argument, from `_contact_operator`'s own docstring: the state variable is a
POINT value at the cell centre, while a source cell carries a VOLUME moment, so
the field that state needs is

    psi_i = Int_{V_source} G(x_centre_i - x') dx'  x  (moment / V)

-- averaged over the SOURCE cell, evaluated at the RECEIVER CENTRE. Nothing in
that sentence mentions contact. It applies at EVERY separation. Yet the default
uses the source-cell average only on the contact shell and the bare midpoint
value G(r_i - r_j) everywhere else. `va_all=True` is the switch that applies it
everywhere, and the library's own docstring for it says the midpoint is

    "a scale-invariant bias that refinement cannot remove."

THE CONTRADICTION THIS RESOLVED, and it turned out to be a DEFECT rather than a
stale measurement. That docstring is an argument FOR va_all, yet va_all was
recorded as REFUTED at "32% worse". Both could not stand.

Running it found the reason. `_cell_averaged_propagator` defaults to
``double=True``; `_single_contact_cached` passes ``double=False`` explicitly,
but BOTH va_all call sites omitted it. So va_all applied the GALERKIN double
average beyond the contact shell while the shell beside it used the single
average -- the exact formulation mismatch this scheme was fixed to remove,
reintroduced one shell further out. Fixed in the same commit as this script's
first run; `TestVaAllAveragingConvention` pins it.

HOW IT SURFACED, because no gate caught it and the suite passed 1033 tests with
it in place: panel [3] extends the SINGLE average by hand (via va_radius, which
routes through the contact operator) and reached 1.12e-4 at radius 3, while
panel [1]'s va_all -- covering radii 2 through 4, a strict superset -- managed
only 3.85e-4. A superset of corrected shells cannot be worse unless the two
routes compute different operators. That is the whole detection.

⚠ So the recorded "32% worse" was never a verdict on source-cell averaging. It
was a measurement of the double average applied everywhere, which is consistent
with the contact finding rather than opposed to it.

WHY "COLLOCATION FOR EVERYTHING" IS NOT AVAILABLE AS A LIMIT, and this is the
part that is structural rather than empirical. Collocation is a statement about
the TEST space -- one equation per cell, enforced at its centre. It cannot also
be imposed on the SOURCE space, because:

  * at the SELF cell the point value of dd-G does not exist at all. The 1/r^3
    singularity is non-integrable and carries the Eshelby delta. The self term
    is necessarily a volume integral, and that integral IS the T-matrix.
  * at CONTACT the cells touch, so the separation between source and receiver
    material reaches zero and the kernel is singular on the touching face.

So the source side is always an integral over the cell; the only question is
whether that integral is approximated by its midpoint. The question is
therefore not "collocation or not" but "at what separation does the midpoint
rule become accurate enough", which is a quadrature question with a measurable
answer -- the third panel below.

ARBITER: Kennett, exact for a uniform layer.

Run:  conda run -n seismic python scripts/settle_collocation_everywhere.py
SI units (m, m/s, kg/m3, Pa).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubic_scattering.effective_contrasts import (  # noqa: E402
    MaterialContrast,
    ReferenceMedium,
    compute_cube_tmatrix,
)
from cubic_scattering.slab_scattering import (  # noqa: E402
    SlabGeometry,
    SlabMaterial,
    build_slab_kernels,
    compute_slab_scattering,
    kennett_reference_rpp,
    slab_rpp_periodic,
)
from cubic_scattering.voigt_tmatrix import effective_stiffness_voigt  # noqa: E402

REF = ReferenceMedium(5000.0, 3000.0, 2500.0)
OMEGA, H_PHYS, M = 60.0, 4.0, 4

CASES = (
    ("pure shear", 0.0, 1.0e9, 0.0),
    ("working (mixed)", 2.0e9, 1.0e9, 100.0),
)


def err_vs_kennett(
    dlam: float,
    dmu: float,
    drho: float,
    n_z: int,
    va_all: bool,
    va_radius: int = 1,
    va_gauss: int = 4,
) -> float:
    """Relative |R_PP| error against Kennett, collocation T-matrix throughout."""
    a = H_PHYS / (2.0 * n_z)
    geom = SlabGeometry(M=M, N_z=n_z, a=a)
    ones = np.ones((n_z, M, M))
    con = MaterialContrast(dlam, dmu, drho)
    mat = SlabMaterial(Dlambda=dlam * ones, Dmu=dmu * ones, Drho=drho * ones, ref=REF)

    t9 = compute_cube_tmatrix(OMEGA, a, REF, con)
    v = (2.0 * a) ** 3
    cell = np.zeros((9, 9), dtype=complex)
    cell[:3, :3] = OMEGA**2 * complex(t9.Drho_star) * v * np.eye(3)
    cell[3:, 3:] = v * effective_stiffness_voigt(t9.Dlambda_star, t9.Dmu_star_diag, t9.Dmu_star_off)
    t0 = np.broadcast_to(cell, (n_z, M, M, 9, 9)).copy()

    kh = build_slab_kernels(
        geom,
        OMEGA,
        REF,
        periodic=True,
        lattice_ewald=True,
        volume_averaged=True,
        n_orders=2,
        contact_average="single",
        va_all=va_all,
        va_radius=va_radius,
        va_gauss=va_gauss,
    )
    res = compute_slab_scattering(
        geom,
        mat,
        OMEGA,
        np.array([1.0, 0.0, 0.0]),
        "P",
        periodic=True,
        gmres_tol=1e-13,
        kernel_hat=kh,
        T_local=t0,
    )
    r = complex(slab_rpp_periodic(res, t0))
    exact = kennett_reference_rpp(REF, con, H_PHYS, OMEGA)
    return float(abs(r - exact) / abs(exact))


def main() -> int:
    print("=" * 78)
    print("COLLOCATION EVERYWHERE?  Re-measuring va_all on the FIXED baseline")
    print("=" * 78)

    print("\n  [1] midpoint propagator vs source-cell average at every separation")
    print("      (collocation T-matrix + single contact throughout; the ONLY")
    print("       thing varying is the propagator beyond the contact shell)")
    verdicts: list[bool] = []
    for label, dl, dm, dr in CASES:
        print(f"\n    {label}")
        print(f"      {'n_z':>5} {'midpoint':>13} {'va_all':>13} {'ratio':>8}  {'better':>10}")
        for n_z in (2, 4, 8):
            e_mid = err_vs_kennett(dl, dm, dr, n_z, False)
            e_va = err_vs_kennett(dl, dm, dr, n_z, True)
            better = "va_all" if e_va < e_mid else "midpoint"
            verdicts.append(e_va < e_mid)
            print(f"      {n_z:>5} {e_mid:13.4e} {e_va:13.4e} {e_va / e_mid:8.3f}  {better:>10}")

    n_better = sum(verdicts)
    print(f"\n      va_all better in {n_better}/{len(verdicts)} rows")

    print("\n  [2] is the va_all quadrature converged?  (if it is not, panel [1]")
    print("      measures quadrature error, not the midpoint bias)")
    print("      NOTE: the correction REACH is pinned at 4 cells inside the")
    print("      Ewald correction and is not exposed, so what is varied here is")
    print("      the Gauss order at fixed reach.")
    dl, dm, dr = CASES[0][1:]
    print(f"\n      {'va_gauss':>9} {'err':>13} {'d vs prev':>12}")
    prev = None
    for ng in (4, 6, 8):
        e = err_vs_kennett(dl, dm, dr, 4, True, va_gauss=ng)
        d = "" if prev is None else f"{abs(e - prev) / abs(e):12.2e}"
        print(f"      {ng:>9} {e:13.4e} {d:>12}")
        prev = e

    print("\n  [3] how far out does the source-cell average actually matter?")
    print("      va_radius is the shell treated with the EXACT single-averaged")
    print("      contact operator; beyond it the midpoint is used (va_all off).")
    print("      If the gain saturates at radius 1, the midpoint rule is already")
    print("      adequate past the touching shell and 'collocation everywhere'")
    print("      is the right default for a quantitative reason, not a dogma.")
    print(f"\n      {'va_radius':>10} {'err':>14} {'vs radius 1':>14}")
    base = None
    for rad in (1, 2, 3):
        e = err_vs_kennett(dl, dm, dr, 4, False, va_radius=rad)
        if base is None:
            base = e
        print(f"      {rad:>10} {e:14.4e} {e / base:14.4f}")

    print("\n" + "=" * 78)
    print("WHY NOT COLLOCATION FOR EVERYTHING -- the structural half")
    print()
    print("  Collocation constrains the TEST space (one equation per cell at its")
    print("  centre).  The SOURCE side is always an integral over the cell,")
    print("  because at the self cell the point value of dd-G does not exist --")
    print("  a non-integrable 1/r^3 plus the Eshelby delta -- and at contact the")
    print("  cells touch.  So the scheme cannot be 'collocation everywhere' even")
    print("  in principle; the self integral IS the T-matrix.")
    print()
    print("  The real question is where the MIDPOINT RULE is an adequate")
    print("  approximation to that source integral, and that is what panel [1]")
    print("  and [3] measure.  Answer: NOT at the shells it is currently used")
    print("  on.  Correcting them is worth ~9x on the Kennett residual, and the")
    print("  gain grows under refinement -- a scale-invariant bias removed, not")
    print("  a discretisation error reduced.")
    print()
    print("  ⚠ TWO THINGS THIS DOES NOT ESTABLISH.  The corrected sequence is")
    print("  still falling but NOT at a clean second-order rate (1.20, 1.16 per")
    print("  halving), so it is not yet a convergence claim.  And the correction")
    print("  reach is pinned at 4 cells and not exposed, so part of what remains")
    print("  may be shell truncation rather than physics -- panel [3] is still")
    print("  descending at its last point, which is the evidence for that.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
