"""MEASUREMENT: is the residual a SELF-ENERGY (local-field) discrepancy?

WHERE THIS SITS. The propagator-quadrature hypothesis is closed: replacing the
midpoint G(r) by the cell average <G>, and then by the full double (Galerkin)
average over both cells, each lowered the constant and NONE restored convergence
(`measure_lattice_saturation` [S5]). The j != 0 part of the lattice sum is now
the continuum integral to quadrature accuracy, so if the answer is still wrong
the remaining suspect is the SELF cell.

THE BOOKKEEPING THAT MUST BALANCE. The continuum response needs the whole
integral Int_all G. The lattice splits it: the self cell is carried inside T0 as
the isolated-cube Eshelby depolarisation Gamma_0, and every other cell is
carried by the kernel. If both halves are right, the total is right. The kernel
half now is. So the question is whether Gamma_0 -- the response of a cube ALONE
in the reference -- is the right self-energy for a cube EMBEDDED in a filled
lattice. That is the Clausius-Mossotti question, and this project has already
met its analogue in the sphere-packing Delta -> Delta/phi renormalisation.

HOW Gamma_0 IS REACHED WITHOUT REIMPLEMENTING IT. T0 = dC_V (I - Gamma_0 dC_V)^-1
with dC_V the Born (contrast x volume) T-matrix, so

    Gamma_0 = T_Born^-1 - T0^-1

and T_Born is obtained as the measured limit T(eps c)/eps -- the code's own
conventions, no hand-written formula. A SHIFTED self-energy is then simply

    T(gamma) = (T0^-1 - gamma I)^-1

WHY THIS IS NOT THE kappa TEST RELABELLED. kappa scaled T linearly. gamma enters
the self-energy, so T depends on it NONLINEARLY and its influence grows with
contrast. They are different one-parameter families, and the earlier result that
kappa drifts says nothing about whether gamma is constant.

THE PREDICTION, and it is sharp: a local-field constant is a property of the
LATTICE, not of how many cells happen to be stacked. If gamma solved for at each
rung is n_z-INDEPENDENT, the missing physics is identified and derivable. If it
drifts like kappa did, the self-energy is not the answer either.

Run:  conda run -n seismic python scripts/measure_lattice_self_energy.py
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
D_LAM, D_MU, D_RHO = 2.0e9, 1.0e9, 100.0
OMEGA, H_PHYS, M = 60.0, 4.0, 4
LADDER = (1, 2, 4, 8, 16)
EPS_LIN = 1e-6


def _setup(n_z: int):
    """(geometry, material, T0, T_Born, Gamma_0, kernel) for one rung."""
    a_half = H_PHYS / (2.0 * n_z)
    geom = SlabGeometry(M=M, N_z=n_z, a=a_half)
    ones = np.ones((n_z, M, M))
    mat = SlabMaterial(Dlambda=D_LAM * ones, Dmu=D_MU * ones, Drho=D_RHO * ones, ref=REF)
    t0 = compute_slab_tmatrices(geom, mat, OMEGA)
    mat_lin = SlabMaterial(
        Dlambda=EPS_LIN * D_LAM * ones,
        Dmu=EPS_LIN * D_MU * ones,
        Drho=EPS_LIN * D_RHO * ones,
        ref=REF,
    )
    t_born = compute_slab_tmatrices(geom, mat_lin, OMEGA) / EPS_LIN
    g0 = np.linalg.inv(t_born[0, 0, 0]) - np.linalg.inv(t0[0, 0, 0])
    # The kernel is built ONCE per rung and reused across the solve for gamma --
    # only T changes, and the cell-averaged kernel is expensive.
    kh = build_slab_kernels(geom, OMEGA, REF, periodic=True, volume_averaged=True, va_all=True)
    return geom, mat, t0, t_born, g0, kh


def _r_lattice(geom, mat, t0, kh, gamma: float) -> complex:
    """R_PP with the self-energy shifted by gamma: T = (T0^-1 - gamma I)^-1."""
    shape = t0.shape
    flat = t0.reshape(-1, 9, 9)
    shifted = np.empty_like(flat)
    for s in range(flat.shape[0]):
        shifted[s] = np.linalg.inv(np.linalg.inv(flat[s]) - gamma * np.eye(9))
    t = shifted.reshape(shape)
    res = compute_slab_scattering(
        geom,
        mat,
        OMEGA,
        np.array([1.0, 0.0, 0.0]),
        "P",
        periodic=True,
        gmres_tol=1e-12,
        kernel_hat=kh,
        T_local=t,
    )
    return slab_rpp_periodic(res, t)


def main() -> int:
    print("=" * 84)
    print("MEASUREMENT -- is the residual a SELF-ENERGY (local-field) discrepancy?")
    print("  gamma solved per rung; a LATTICE constant must not depend on n_z")
    print("=" * 84)

    r_ken = kennett_reference_rpp(REF, MaterialContrast(D_LAM, D_MU, D_RHO), H_PHYS, OMEGA)
    print(f"\n  {'n_z':>4} {'|Gamma_0|':>11} {'gamma':>13} {'gamma/|G_0|':>12} {'residual':>11}")
    ratios = []
    for n_z in LADDER:
        geom, mat, t0, t_born, g0, kh = _setup(n_z)
        scale = float(np.abs(g0).max())

        # Secant on the REAL gamma that matches the exact reflection coefficient.
        # Bracket from the scale of Gamma_0 itself, which is the only natural
        # unit in the problem.
        g_a, g_b = 0.0, 0.05 * scale
        f_a = abs(_r_lattice(geom, mat, t0, kh, g_a)) - abs(r_ken)
        f_b = abs(_r_lattice(geom, mat, t0, kh, g_b)) - abs(r_ken)
        for _ in range(40):
            if abs(f_b - f_a) < 1e-300:
                break
            g_c = g_b - f_b * (g_b - g_a) / (f_b - f_a)
            f_c = abs(_r_lattice(geom, mat, t0, kh, g_c)) - abs(r_ken)
            g_a, f_a, g_b, f_b = g_b, f_b, g_c, f_c
            if abs(f_c) < 1e-14 * abs(r_ken):
                break
        resid = abs(_r_lattice(geom, mat, t0, kh, g_b) - r_ken) / abs(r_ken)
        ratios.append(g_b / scale)
        print(f"  {n_z:4d} {scale:11.4e} {g_b:13.5e} {g_b / scale:12.5f} {resid:11.3e}")

    spread = (max(ratios) - min(ratios)) / abs(np.mean(ratios))
    print(f"\n  gamma/|Gamma_0| spread across the ladder: {spread:.1%}")
    print("=" * 84)
    if spread < 0.05:
        print("  SELF-ENERGY CONFIRMED: one gamma, independent of n_z, reproduces the")
        print("  exact answer at every rung. The missing physics is the local field of")
        print("  the lattice -- an isolated-cube Gamma_0 is not the self-energy of a")
        print("  cube embedded in a filled lattice. Derive it from the lattice sum.")
    else:
        print("  NOT A CONSTANT SELF-ENERGY: gamma drifts with n_z, so a scalar shift")
        print("  of the self-energy is not the missing physics either. Report the")
        print("  drift and look at the STRUCTURE -- a scalar on the 9x9 identity may")
        print("  simply be the wrong shape for a tensor local field.")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
