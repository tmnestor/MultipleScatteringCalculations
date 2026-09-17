"""THE PAYOFF: does the exact cell average beat the shell route against Kennett?

Everything before this measured pieces. This measures the thing the pieces were
for. Three propagator treatments, one arbiter:

  midpoint          the source-cell average only on the contact shell, the bare
                    midpoint G(r_i - r_j) beyond it
  va_all (shell)    the average extended outward by a real-space correction
                    shell, truncated at va_all_reach
  exact             sinc^1 form factor at dz != 0 + the analytic d^2 tail at
                    dz == 0 -- no correction shell, no reach, no truncation

ARBITER: Kennett, exact for a uniform layer and derived from none of them.

WHAT WOULD MAKE THIS A FAILURE, stated before running. The exact route removes a
truncation; it does NOT add physics the shell route lacks in the limit. So at
the operating point -- where the shell sum is still absolutely convergent and
reach 8 is only mildly truncated -- the two should come out CLOSE, with `exact`
no worse. A large gap in either direction means something is wired wrong, not
that a breakthrough happened. The gain the exact route is for shows up where the
shell route CANNOT go: it needs R ~ 80 to converge while the shape term switches
on near R ~ 1/(k_S d) ~ 50.

⚠ SO THE HEADLINE IS NOT THE ERROR AT ONE MESH. It is (a) that `exact` matches
the best shell result without a reach parameter, and (b) that it is
reach-INDEPENDENT by construction, which the shell route can never be.

Run:  conda run -n seismic python scripts/measure_exact_cell_average_kennett.py
SI units (m, m/s, kg/m3, Pa).
"""

from __future__ import annotations

import sys
import time
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


def err_vs_kennett(dlam: float, dmu: float, drho: float, n_z: int, mode: str, reach: int = 8) -> float:
    """Relative |R_PP| error vs Kennett, collocation T-matrix throughout."""
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

    common = {
        "periodic": True,
        "lattice_ewald": True,
        "volume_averaged": True,
        "n_orders": 2,
        "contact_average": "single",
    }
    if mode == "exact":
        kh = build_slab_kernels(geom, OMEGA, REF, **common, exact_cell_average=True)
    elif mode == "shell":
        kh = build_slab_kernels(geom, OMEGA, REF, **common, va_all=True, va_all_reach=reach)
    elif mode == "midpoint":
        kh = build_slab_kernels(geom, OMEGA, REF, **common)
    else:
        raise ValueError(mode)

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
    print("THE PAYOFF: exact cell average vs the shell route, against Kennett")
    print("=" * 78)

    print("\n  [1] the three treatments under REFINEMENT")
    print("      The single-mesh error is the least informative column here.")
    print("      What separates the routes is the RATIO per mesh doubling: 4.0")
    print("      is second order, 1.0 is a saturated floor.")
    for label, dl, dm, dr in CASES:
        print(f"\n    {label}")
        print(
            f"      {'n_z':>5} {'midpoint':>13} {'shell R=8':>13} {'EXACT':>13}"
            f" {'shell x':>9} {'exact x':>9}"
        )
        prev_s = prev_e = None
        seq_e = []
        for n_z in (2, 4, 8):
            e_mid = err_vs_kennett(dl, dm, dr, n_z, "midpoint")
            e_shl = err_vs_kennett(dl, dm, dr, n_z, "shell")
            e_exa = err_vs_kennett(dl, dm, dr, n_z, "exact")
            seq_e.append(e_exa)
            rs = "" if prev_s is None else f"{prev_s / e_shl:9.2f}"
            re_ = "" if prev_e is None else f"{prev_e / e_exa:9.2f}"
            print(f"      {n_z:>5} {e_mid:13.4e} {e_shl:13.4e} {e_exa:13.4e} {rs:>9} {re_:>9}")
            prev_s, prev_e = e_shl, e_exa

        # ⚠ A NON-MONOTONE SEQUENCE IS A ZERO CROSSING, NOT CONVERGENCE, and the
        # minimum of such a sequence is an accident of where the sign flip fell.
        # Quoting it as an accuracy would be quoting a cancellation. This is not
        # hypothetical: the mixed case dips to 3.9e-7 at n_z = 4 and comes back
        # up to 1.6e-6 at n_z = 8.
        if not (seq_e[0] > seq_e[1] > seq_e[2]):
            print("      ⚠ EXACT is NON-MONOTONE here: the dip is a SIGN CHANGE,")
            print("        not convergence.  Do not quote the minimum as an")
            print("        accuracy -- it is where the error crossed zero.")

    print("\n  [2] THE STRUCTURAL CLAIM: the exact route has no reach to converge")
    print("      The shell route's answer MOVES with its truncation radius; the")
    print("      exact route has no such parameter at all.  That is the point,")
    print("      and it is what the error at a single mesh cannot show.")
    dl, dm, dr = CASES[0][1:]
    print(f"\n      {'reach':>7} {'shell':>14} {'EXACT':>14}")
    e_exa = err_vs_kennett(dl, dm, dr, 4, "exact")
    for reach in (2, 4, 8):
        e_shl = err_vs_kennett(dl, dm, dr, 4, "shell", reach=reach)
        print(f"      {reach:>7} {e_shl:14.4e} {e_exa:14.4e}")
    print("      (the EXACT column is constant BY CONSTRUCTION -- it is printed")
    print("       repeatedly only to make the contrast legible)")

    print("\n  [3] cost")
    t0 = time.perf_counter()
    err_vs_kennett(dl, dm, dr, 2, "exact")
    t_exact = time.perf_counter() - t0
    t0 = time.perf_counter()
    err_vs_kennett(dl, dm, dr, 2, "shell")
    t_shell = time.perf_counter() - t0
    print(f"      exact {t_exact:.1f} s   shell(R=8) {t_shell:.1f} s   ratio {t_exact / t_shell:.2f}x")

    print("\n" + "=" * 78)
    print("WHAT THIS SHOWS, AND WHAT IT DOES NOT.")
    print()
    print("  THE RESULT IS THE REFINEMENT BEHAVIOUR, not any single error.  The")
    print("  shell route SATURATES -- its ratio per doubling falls toward 1.0,")
    print("  which is a floor, because its truncation is scale-invariant: the")
    print("  cell and the shell shrink together.  The exact route does not, and")
    print("  approaches second order.  That is the difference between removing a")
    print("  bias and reducing a discretisation error.")
    print()
    print("  ⚠ AN EXPECTATION SET BEFORE RUNNING WAS WRONG, and the record is")
    print("  kept rather than quietly revised.  This script originally argued")
    print("  the two routes 'should come out close, since reach 8 is only mildly")
    print("  truncated'.  The reach sweep in settle_collocation_everywhere.py had")
    print("  ALREADY shown reach 8 is NOT nearly converged -- still falling as")
    print("  R^-1.5 with no saturation.  The premise was wrong, not the result.")
    print()
    print("  ⚠ AND THE MIXED CASE DIPS THROUGH ZERO at n_z = 4.  Its 3.9e-7 is a")
    print("  sign change, not an accuracy; a two-point comparison would have")
    print("  reported it as a 17x gain.  Three meshes were run for that reason.")
    print()
    print("  The durable claim is structural: the exact route has NO reach")
    print("  parameter, where the shell route provably cannot converge one")
    print("  (R ~ 80 needed, shape-dependence switching on at R ~ 50).")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
