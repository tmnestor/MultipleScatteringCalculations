"""MEASUREMENT: does the single-site T27 remove the refinement bias?

THE CASE FOR TRYING IT, in one line: the single-site T27 - T9 difference is
1.82e-2 and SCALE-INVARIANT (1.02x across a 32x range of cell sizes,
`measure_t27_vs_t9_single_site`), and the unexplained refinement bias is 2.3e-2
and scale-invariant. Same order, same signature.

WHAT IS AND IS NOT BEING BUILT. The project's verdict against a 27-component
LATTICE solver stands and is respected: inter-voxel quadratic-mode coupling is
<=0.03% at our ka, so no 27x27 propagator appears here. Only the SINGLE-SITE
response changes, reduced to the same 9 components the lattice already couples.
The propagator, the kernel and the solver are untouched.

THE ONE SUBSTITUTION, stated plainly rather than buried. The 9x9 cell T-matrix
is built from four scalars: Drho_star, Dlambda_star, Dmu_star_diag,
Dmu_star_off. The Galerkin (T27) result supplies the three STRAIN scalars. It
does not expose Drho_star -- the density channel lives in its T1u block -- so
the density channel is kept at its T9 value. This therefore tests the strain
sector, which is exactly where the 18 quadratic trial functions act, and the
density channel is held fixed rather than silently swapped.

WHAT COUNTS. The bias GROWS with refinement and saturates. If the T27 strain
sector is the missing physics, the T27 ladder should stop growing. A smaller
constant with the same growth would mean T27 is merely another contributor, like
the propagator averaging before it.

Run:  conda run -n seismic python scripts/measure_t27_lattice_kennett.py
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
    compute_cube_tmatrix,
    compute_cube_tmatrix_galerkin,
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
from cubic_scattering.voigt_tmatrix import effective_stiffness_voigt  # noqa: E402

REF = ReferenceMedium(5000.0, 3000.0, 2500.0)
CONTRAST = MaterialContrast(2.0e9, 1.0e9, 100.0)
OMEGA, H_PHYS, M = 60.0, 4.0, 4
LADDER = (1, 2, 4, 8, 16, 32)


def _t9_from_galerkin(a: float) -> np.ndarray:
    """One 9x9 cell T-matrix with the T27 strain sector, T9 density channel."""
    t9 = compute_cube_tmatrix(OMEGA, a, REF, CONTRAST)
    t27 = compute_cube_tmatrix_galerkin(OMEGA, a, REF, CONTRAST)
    v = (2.0 * a) ** 3
    t = np.zeros((9, 9), dtype=complex)
    # Density channel from T9 -- the Galerkin result does not expose Drho_star.
    t[:3, :3] = OMEGA**2 * complex(t9.Drho_star) * v * np.eye(3)
    # Strain sector from T27: this is the whole point of the substitution.
    t[3:, 3:] = v * effective_stiffness_voigt(t27.Dlambda_star, t27.Dmu_star_diag, t27.Dmu_star_off)
    return t


def _err(n_z: int, *, use_t27: bool) -> float:
    a_half = H_PHYS / (2.0 * n_z)
    geom = SlabGeometry(M=M, N_z=n_z, a=a_half)
    ones = np.ones((n_z, M, M))
    mat = SlabMaterial(
        Dlambda=CONTRAST.Dlambda * ones,
        Dmu=CONTRAST.Dmu * ones,
        Drho=CONTRAST.Drho * ones,
        ref=REF,
    )
    if use_t27:
        cell = _t9_from_galerkin(a_half)
        t_local = np.broadcast_to(cell, (n_z, M, M, 9, 9)).copy()
    else:
        t_local = compute_slab_tmatrices(geom, mat, OMEGA)
    kh = build_slab_kernels(geom, OMEGA, REF, periodic=True)
    res = compute_slab_scattering(
        geom,
        mat,
        OMEGA,
        np.array([1.0, 0.0, 0.0]),
        "P",
        periodic=True,
        gmres_tol=1e-12,
        kernel_hat=kh,
        T_local=t_local,
    )
    r_lat = slab_rpp_periodic(res, t_local)
    r_ken = kennett_reference_rpp(REF, CONTRAST, H_PHYS, OMEGA)
    return abs(r_lat - r_ken) / abs(r_ken)


def main() -> int:
    print("=" * 84)
    print("MEASUREMENT -- does the single-site T27 strain sector remove the bias?")
    print(f"  physical slab fixed at H = {H_PHYS} m; only the cell size changes")
    print("=" * 84)
    print(f"\n  {'n_z':>4} {'T9':>12} {'ratio':>7} {'T27':>12} {'ratio':>7}")

    e9s, e27s, p9, p27 = [], [], None, None
    for n_z in LADDER:
        e9, e27 = _err(n_z, use_t27=False), _err(n_z, use_t27=True)
        r9 = "" if p9 is None else f"{e9 / p9:7.2f}"
        r27 = "" if p27 is None else f"{e27 / p27:7.2f}"
        print(f"  {n_z:4d} {e9:12.4e} {r9:>7} {e27:12.4e} {r27:>7}")
        e9s.append(e9)
        e27s.append(e27)
        p9, p27 = e9, e27

    grew9 = e9s[-1] / e9s[0]
    grew27 = e27s[-1] / e27s[0]
    print(f"\n  growth across the ladder:  T9 {grew9:.2f}x   T27 {grew27:.2f}x")
    print(f"  finest rung:               T9 {e9s[-1]:.3e}   T27 {e27s[-1]:.3e}")
    print("=" * 84)
    if e27s[-1] < 0.3 * e9s[-1] and grew27 < 1.5:
        print("  T27 REMOVES THE BIAS: the ladder stops growing and the finest rung")
        print("  improves several-fold. The 9-component single-site response was the")
        print("  missing physics -- the quadratic modes carry a scale-invariant part")
        print("  of the cell's response that (u, eps) at the centre cannot.")
    elif e27s[-1] < e9s[-1]:
        print("  T27 HELPS BUT DOES NOT FIX IT: the error is smaller but still grows.")
        print("  Like the propagator averaging before it, this is a contributor and")
        print("  not the cause. Report the new constant and keep looking.")
    else:
        print("  T27 DOES NOT HELP. The scale-invariant single-site difference is")
        print("  real but does not act in the direction of the bias, so the match in")
        print("  magnitude was a coincidence.")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
