#!/usr/bin/env python3
"""Slab voxel refinement convergence study.

Validates that the Foldy-Lax solver converges to the Kennett reflectivity
as voxel size decreases, for a uniform elastic layer at normal P-wave incidence.

Two study modes:
  single-layer: N_z=1, varying a (each run has different physical thickness H=2a)
  fixed-H:      Fixed physical thickness H, refine mesh (N_z and M increase as a→0)

Usage:
    python scripts/slab_convergence_study.py
    python scripts/slab_convergence_study.py --study fixed-H --H 2.0
    python scripts/slab_convergence_study.py --volume-averaged
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from cubic_scattering.effective_contrasts import MaterialContrast, ReferenceMedium
from cubic_scattering.slab_scattering import (
    SlabGeometry,
    compute_slab_scattering,
    compute_slab_tmatrices,
    kennett_reference_rpp,
    slab_rpp_periodic,
    uniform_slab_material,
)

# =====================================================================
# Physical parameters
# =====================================================================

REF = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
CONTRAST = MaterialContrast(Dlambda=2e9, Dmu=1e9, Drho=100.0)
OMEGA = 150.0  # rad/s
K_HAT = np.array([1.0, 0.0, 0.0])  # normal P-wave incidence


# =====================================================================
# Single-layer study
# =====================================================================


def run_single_layer_study(
    a_values: list[float],
    M_values: list[int],
    volume_averaged: bool = False,
    gmres_tol: float = 1e-8,
    periodic: bool = False,
) -> None:
    """Single-layer study: N_z=1, varying a and M.

    Each run uses N_z=1 with half-width a.  Physical layer H=2a varies.
    Separate Kennett reference per a-value.
    """
    va_tag = " (volume-averaged)" if volume_averaged else ""
    p_tag = " [periodic]" if periodic else ""
    print(f"\nSlab Convergence Study: Single-Layer (N_z=1){va_tag}{p_tag}")
    print(
        f"Background: alpha={REF.alpha} m/s, beta={REF.beta} m/s, rho={REF.rho} kg/m^3"
    )
    print(
        f"Contrast: Dlambda={CONTRAST.Dlambda / 1e9:.1f} GPa, "
        f"Dmu={CONTRAST.Dmu / 1e9:.1f} GPa, Drho={CONTRAST.Drho:.0f} kg/m^3"
    )
    print(f"omega = {OMEGA:.1f} rad/s, normal P-wave incidence\n")
    print(
        f"{'a':>6}  {'M':>4}  {'ka_S':>6}  {'DOF':>7}"
        f"  {'|R_FL|':>11}  {'|R_K|':>11}  {'Rel Err':>9}"
        f"  {'GMRES':>5}  {'Resid':>9}  {'Time':>7}"
    )
    print("-" * 92)

    results = []
    for a, M in zip(a_values, M_values, strict=True):
        geom = SlabGeometry(M=M, N_z=1, a=a)
        mat = uniform_slab_material(geom, REF, CONTRAST)

        t0 = time.perf_counter()
        res = compute_slab_scattering(
            geom,
            mat,
            OMEGA,
            K_HAT,
            gmres_tol=gmres_tol,
            volume_averaged=volume_averaged,
            periodic=periodic,
        )
        T_local = compute_slab_tmatrices(geom, mat, OMEGA)
        elapsed = time.perf_counter() - t0

        R_FL = slab_rpp_periodic(res, T_local)
        R_K = kennett_reference_rpp(REF, CONTRAST, H=geom.d, omega=OMEGA)

        ka_S = OMEGA * a / REF.beta
        dof = geom.n_cubes * 9
        rel_err = abs(R_FL - R_K) / abs(R_K) if abs(R_K) > 0 else float("nan")

        results.append((a, M, ka_S, dof, R_FL, R_K, rel_err, res, elapsed))
        print(
            f"{a:6.3f}  {M:4d}  {ka_S:6.4f}  {dof:7d}"
            f"  {abs(R_FL):11.4e}  {abs(R_K):11.4e}  {rel_err:9.3e}"
            f"  {res.n_gmres_iter:5d}  {res.gmres_residual:9.2e}  {elapsed:6.1f}s"
        )

    # Summary
    if len(results) >= 2:
        errs = [r[6] for r in results if not np.isnan(r[6])]
        if len(errs) >= 2:
            a_arr = np.array([r[0] for r in results[: len(errs)]])
            e_arr = np.array(errs)
            log_a = np.log(a_arr[1:] / a_arr[:-1])
            log_e = np.log(e_arr[1:] / e_arr[:-1])
            slopes = log_e / log_a
            print("\nConvergence rates (log-log slope error vs a):")
            for i, s in enumerate(slopes):
                print(
                    f"  a={a_arr[i]:.3f} -> {a_arr[i + 1]:.3f}: "
                    f"slope = {s:.2f} (expect ~2 for O(ka^2))"
                )

    # Phase comparison
    print("\nPhase comparison (R_FL vs R_K):")
    for a, _M, _ka_S, _dof, R_FL, R_K, _rel_err, _res, _elapsed in results:
        phase_FL = np.degrees(np.angle(R_FL))
        phase_K = np.degrees(np.angle(R_K))
        print(f"  a={a:.3f}: phase_FL = {phase_FL:+7.2f}°, phase_K = {phase_K:+7.2f}°")

    # Complex values
    print("\nComplex reflection coefficients:")
    for a, _M, _ka_S, _dof, R_FL, R_K, _rel_err, _res, _elapsed in results:
        print(
            f"  a={a:.3f}: R_FL = {R_FL.real:+.6e} {R_FL.imag:+.6e}j"
            f"  |  R_K = {R_K.real:+.6e} {R_K.imag:+.6e}j"
        )


# =====================================================================
# Fixed-thickness study
# =====================================================================


def run_fixed_H_study(
    H: float,
    a_values: list[float],
    volume_averaged: bool = False,
    gmres_tol: float = 1e-8,
    periodic: bool = False,
) -> None:
    """Fixed-thickness study: H constant, mesh refined as a decreases."""
    va_tag = " (volume-averaged)" if volume_averaged else ""
    p_tag = " [periodic]" if periodic else ""
    print(f"\nSlab Convergence Study: Fixed Thickness H={H:.1f} m{va_tag}{p_tag}")
    print(
        f"Background: alpha={REF.alpha} m/s, beta={REF.beta} m/s, rho={REF.rho} kg/m^3"
    )
    print(
        f"Contrast: Dlambda={CONTRAST.Dlambda / 1e9:.1f} GPa, "
        f"Dmu={CONTRAST.Dmu / 1e9:.1f} GPa, Drho={CONTRAST.Drho:.0f} kg/m^3"
    )
    print(f"omega = {OMEGA:.1f} rad/s, normal P-wave incidence\n")

    # Kennett reference (same for all a-values)
    R_K = kennett_reference_rpp(REF, CONTRAST, H=H, omega=OMEGA)
    print(f"Kennett R_PP = {R_K.real:+.6e} {R_K.imag:+.6e}j,  |R_K| = {abs(R_K):.6e}\n")

    # Compute M and N_z for each a, keeping physical size fixed
    L = max(a * 8 for a in a_values)  # lateral extent from coarsest grid

    print(
        f"{'a':>6}  {'M':>4}  {'N_z':>4}  {'ka_S':>6}  {'DOF':>7}"
        f"  {'|R_FL|':>11}  {'Rel Err':>9}"
        f"  {'GMRES':>5}  {'Resid':>9}  {'Time':>7}"
    )
    print("-" * 88)

    results = []
    for a in a_values:
        d = 2.0 * a
        N_z = round(H / d)
        if abs(N_z * d - H) > 1e-10 * H:
            print(f"  SKIP a={a:.3f}: H/{d}={H / d:.2f} is not integer")
            continue
        M = round(L / d)

        geom = SlabGeometry(M=M, N_z=N_z, a=a)
        mat = uniform_slab_material(geom, REF, CONTRAST)

        t0 = time.perf_counter()
        res = compute_slab_scattering(
            geom,
            mat,
            OMEGA,
            K_HAT,
            gmres_tol=gmres_tol,
            volume_averaged=volume_averaged,
            periodic=periodic,
        )
        T_local = compute_slab_tmatrices(geom, mat, OMEGA)
        elapsed = time.perf_counter() - t0

        R_FL = slab_rpp_periodic(res, T_local)

        ka_S = OMEGA * a / REF.beta
        dof = geom.n_cubes * 9
        rel_err = abs(R_FL - R_K) / abs(R_K) if abs(R_K) > 0 else float("nan")

        results.append((a, M, N_z, ka_S, dof, R_FL, rel_err, res, elapsed))
        print(
            f"{a:6.3f}  {M:4d}  {N_z:4d}  {ka_S:6.4f}  {dof:7d}"
            f"  {abs(R_FL):11.4e}  {rel_err:9.3e}"
            f"  {res.n_gmres_iter:5d}  {res.gmres_residual:9.2e}  {elapsed:6.1f}s"
        )

    # Convergence rates
    if len(results) >= 2:
        errs = [r[6] for r in results if not np.isnan(r[6])]
        a_arr = np.array([r[0] for r in results[: len(errs)]])
        e_arr = np.array(errs)
        if len(errs) >= 2 and all(e > 0 for e in errs):
            log_a = np.log(a_arr[1:] / a_arr[:-1])
            log_e = np.log(e_arr[1:] / e_arr[:-1])
            slopes = log_e / log_a
            print("\nConvergence rates (log-log slope error vs a):")
            for i, s in enumerate(slopes):
                print(
                    f"  a={a_arr[i]:.3f} -> {a_arr[i + 1]:.3f}: "
                    f"slope = {s:.2f} (expect ~2 for O(ka^2))"
                )

    # Complex values
    print("\nComplex reflection coefficients:")
    for a, _M, _N_z, _ka_S, _dof, R_FL, _rel_err, _res, _elapsed in results:
        print(
            f"  a={a:.3f}: R_FL = {R_FL.real:+.6e} {R_FL.imag:+.6e}j"
            f"  |  R_K = {R_K.real:+.6e} {R_K.imag:+.6e}j"
        )


# =====================================================================
# CLI
# =====================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Slab voxel refinement convergence study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--study",
        default="single-layer",
        choices=["single-layer", "fixed-H", "both"],
        help="Which study to run (default: single-layer)",
    )
    parser.add_argument(
        "--H",
        default=2.0,
        type=float,
        help="Fixed layer thickness for fixed-H study (default: 2.0 m)",
    )
    parser.add_argument(
        "--volume-averaged",
        action="store_true",
        help="Use volume-averaged inter-voxel propagator for NN cubes",
    )
    parser.add_argument(
        "--gmres-tol",
        default=1e-8,
        type=float,
        help="GMRES relative tolerance (default: 1e-8)",
    )
    parser.add_argument(
        "--periodic",
        action="store_true",
        help="Use periodic (circular) convolution for infinite slab",
    )
    args = parser.parse_args()

    if args.study in ("single-layer", "both"):
        run_single_layer_study(
            a_values=[2.0, 1.0, 0.5, 0.25],
            M_values=[4, 8, 16, 32],
            volume_averaged=args.volume_averaged,
            gmres_tol=args.gmres_tol,
            periodic=args.periodic,
        )

    if args.study in ("fixed-H", "both"):
        run_fixed_H_study(
            H=args.H,
            a_values=[1.0, 0.5, 0.25],
            volume_averaged=args.volume_averaged,
            gmres_tol=args.gmres_tol,
            periodic=args.periodic,
        )


if __name__ == "__main__":
    main()
