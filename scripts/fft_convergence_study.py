#!/usr/bin/env python3
"""FFT Foldy-Lax vs Mie convergence study.

Runs the FFT-accelerated GMRES solver at increasing voxel counts and
compares forward and angular scattering amplitudes against exact Mie
theory.

Usage:
    python scripts/fft_convergence_study.py
    python scripts/fft_convergence_study.py --n-sub 3,7,13,21,27,33
    python scripts/fft_convergence_study.py --wave-type S --ka 0.3
    python scripts/fft_convergence_study.py --n-sub 3,7,13,21,27,33 --ka 0.5 --angular
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from cubic_scattering import (
    MaterialContrast,
    ReferenceMedium,
)
from cubic_scattering.sphere_scattering import (
    SphereDecompositionResult,
    compute_elastic_mie,
    foldy_lax_far_field,
    mie_far_field,
)
from cubic_scattering.sphere_scattering_fft import compute_sphere_foldy_lax_fft

# =====================================================================
# Physical parameters
# =====================================================================

REF = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
CONTRAST = MaterialContrast(Dlambda=2.0e9, Dmu=1.0e9, Drho=100.0)
RADIUS = 0.5


# =====================================================================
# Far-field extraction helpers
# =====================================================================


def foldy_lax_forward_amplitude(
    result: SphereDecompositionResult,
    wave_type: str,
) -> complex:
    """Forward scattering amplitude from Foldy-Lax result."""
    k_hat = np.array([1.0, 0.0, 0.0])
    r_hat_arr = np.array([[1.0, 0.0, 0.0]])
    r_distance = 1.0e6 * result.radius

    if wave_type == "P":
        pol = np.array([1.0, 0.0, 0.0])
        u_P, u_S = foldy_lax_far_field(
            result, r_hat_arr, r_distance, k_hat, pol, wave_type="P"
        )
        kP = result.omega / result.ref.alpha
        return complex(
            np.dot(u_P[0], r_hat_arr[0]) * r_distance * np.exp(-1j * kP * r_distance)
        )
    else:
        pol = np.array([0.0, 1.0, 0.0])
        u_P, u_S = foldy_lax_far_field(
            result, r_hat_arr, r_distance, k_hat, pol, wave_type="S"
        )
        kS = result.omega / result.ref.beta
        return complex(np.dot(u_S[0], pol) * r_distance * np.exp(-1j * kS * r_distance))


def mie_forward_amplitude(mie_result: object, wave_type: str) -> complex:
    """Forward scattering amplitude from Mie theory."""
    theta_fwd = np.array([1e-6])
    if wave_type == "P":
        f_P, f_SV, f_SH = mie_far_field(mie_result, theta_fwd, incident_type="P")
        return complex(f_P[0])
    else:
        f_P, f_SV, f_SH = mie_far_field(mie_result, theta_fwd, incident_type="SV")
        return complex(f_SV[0])


# =====================================================================
# Convergence study
# =====================================================================


def run_convergence(
    n_sub_list: list[int],
    wave_type: str,
    ka: float,
    gmres_tol: float,
) -> None:
    """Run forward-amplitude convergence study."""
    omega = ka * REF.alpha / RADIUS
    ka_P = omega * RADIUS / REF.alpha
    ka_S = omega * RADIUS / REF.beta

    mie = compute_elastic_mie(omega, RADIUS, REF, CONTRAST)
    f_mie = mie_forward_amplitude(mie, wave_type)

    print(f"\nConvergence Study: FFT Foldy-Lax vs Mie ({wave_type}-wave)")
    print(f"ka_P = {ka_P:.4f},  ka_S = {ka_S:.4f},  GMRES tol = {gmres_tol:.0e}")
    print(f"Mie f({wave_type}{wave_type}, 0) = {f_mie:.6e}")
    print(f"|f_Mie| = {abs(f_mie):.6e}")
    print()
    print(
        f"{'n_sub':>5}  {'N_cells':>7}  {'V_fill':>7}  {'DOF':>8}"
        f"  {'|f_LF|':>12}  {'Rel Err':>10}  {'Time':>8}"
    )
    print("-" * 73)

    V_sphere = (4.0 / 3.0) * np.pi * RADIUS**3
    results = []

    for n_sub in n_sub_list:
        a_sub = RADIUS / n_sub
        dd = 2.0 * a_sub

        t0 = time.perf_counter()
        result = compute_sphere_foldy_lax_fft(
            omega,
            RADIUS,
            REF,
            CONTRAST,
            n_sub,
            wave_type=wave_type,
            gmres_tol=gmres_tol,
        )
        elapsed = time.perf_counter() - t0

        nC = result.n_cells
        V_fill = nC * dd**3 / V_sphere
        dof = 9 * nC
        f_lf = foldy_lax_forward_amplitude(result, wave_type)
        rel_err = abs(f_lf - f_mie) / abs(f_mie) if abs(f_mie) > 0 else float("nan")

        results.append((n_sub, nC, V_fill, dof, f_lf, rel_err, elapsed))

        print(
            f"{n_sub:5d}  {nC:7d}  {V_fill:7.4f}  {dof:8d}"
            f"  {abs(f_lf):12.6e}  {rel_err:10.4e}  {elapsed:7.1f}s"
        )

    print()
    print("Complex values:")
    for n_sub, _nC, _V_fill, _dof, f_lf, _rel_err, _elapsed in results:
        print(
            f"  n_sub={n_sub:3d}: f_LF = {f_lf.real:+.6e} {f_lf.imag:+.6e}j"
            f"  |  f_Mie = {f_mie.real:+.6e} {f_mie.imag:+.6e}j"
        )


# =====================================================================
# Angular pattern comparison
# =====================================================================


def run_angular_pattern(
    n_sub: int,
    wave_type: str,
    ka: float,
    gmres_tol: float,
) -> None:
    """Compare angular scattering pattern: FFT Foldy-Lax vs Mie."""
    omega = ka * REF.alpha / RADIUS
    ka_P = omega * RADIUS / REF.alpha

    mie = compute_elastic_mie(omega, RADIUS, REF, CONTRAST)
    theta_arr = np.linspace(0.05, np.pi - 0.05, 19)

    incident_type_mie = "P" if wave_type == "P" else "SV"
    f_P_mie, f_SV_mie, f_SH_mie = mie_far_field(
        mie, theta_arr, incident_type=incident_type_mie
    )

    t0 = time.perf_counter()
    result = compute_sphere_foldy_lax_fft(
        omega,
        RADIUS,
        REF,
        CONTRAST,
        n_sub,
        wave_type=wave_type,
        gmres_tol=gmres_tol,
    )
    elapsed = time.perf_counter() - t0

    k_hat = np.array([1.0, 0.0, 0.0])
    r_distance = 1.0e6 * RADIUS
    M = len(theta_arr)
    r_hat_arr = np.zeros((M, 3))
    r_hat_arr[:, 0] = np.cos(theta_arr)
    r_hat_arr[:, 1] = np.sin(theta_arr)

    pol = np.array([1.0, 0.0, 0.0]) if wave_type == "P" else np.array([0.0, 1.0, 0.0])

    u_P_fl, u_S_fl = foldy_lax_far_field(
        result,
        r_hat_arr,
        r_distance,
        k_hat,
        pol,
        wave_type=wave_type,
    )

    kP = omega / REF.alpha
    kS = omega / REF.beta

    f_P_fl = np.array(
        [
            np.dot(u_P_fl[i], r_hat_arr[i]) * r_distance * np.exp(-1j * kP * r_distance)
            for i in range(M)
        ]
    )
    f_SV_fl = np.array(
        [
            np.dot(
                u_S_fl[i],
                np.array([-np.sin(theta_arr[i]), np.cos(theta_arr[i]), 0.0]),
            )
            * r_distance
            * np.exp(-1j * kS * r_distance)
            for i in range(M)
        ]
    )

    print(f"\nAngular pattern: {wave_type}-wave, ka_P={ka_P:.3f}, n_sub={n_sub}")
    print(
        f"N_cells={result.n_cells}, DOF={9 * result.n_cells}, solve time={elapsed:.1f}s"
    )
    print()
    print(
        f"{'theta':>8}  {'|f_P_Mie|':>12}  {'|f_P_LF|':>12}  {'P err':>10}"
        f"  {'|f_SV_Mie|':>12}  {'|f_SV_LF|':>12}  {'SV err':>10}"
    )
    print("-" * 85)

    for i, theta in enumerate(theta_arr):
        p_mie_abs = abs(f_P_mie[i])
        p_fl_abs = abs(f_P_fl[i])
        p_err = abs(f_P_fl[i] - f_P_mie[i]) / max(p_mie_abs, 1e-30)

        sv_mie_abs = abs(f_SV_mie[i])
        sv_fl_abs = abs(f_SV_fl[i])
        sv_err = abs(f_SV_fl[i] - f_SV_mie[i]) / max(sv_mie_abs, 1e-30)

        print(
            f"{np.degrees(theta):8.1f}deg"
            f"  {p_mie_abs:12.4e}  {p_fl_abs:12.4e}  {p_err:10.3e}"
            f"  {sv_mie_abs:12.4e}  {sv_fl_abs:12.4e}  {sv_err:10.3e}"
        )


# =====================================================================
# CLI
# =====================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FFT Foldy-Lax vs Mie convergence study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--n-sub",
        default="3,7,13,21",
        help="Comma-separated n_sub values (default: 3,7,13,21)",
    )
    parser.add_argument(
        "--wave-type",
        default="P",
        choices=["P", "S"],
        help="Incident wave type (default: P)",
    )
    parser.add_argument(
        "--ka",
        default=0.1,
        type=float,
        help="Target ka_P = omega*radius/alpha (default: 0.1)",
    )
    parser.add_argument(
        "--gmres-tol",
        default=1e-10,
        type=float,
        help="GMRES relative tolerance (default: 1e-10)",
    )
    parser.add_argument(
        "--angular",
        action="store_true",
        help="Also run angular pattern comparison (uses largest n_sub)",
    )
    args = parser.parse_args()

    n_sub_list = [int(x.strip()) for x in args.n_sub.split(",")]

    run_convergence(n_sub_list, args.wave_type, args.ka, args.gmres_tol)

    if args.angular:
        run_angular_pattern(
            max(n_sub_list),
            args.wave_type,
            args.ka,
            args.gmres_tol,
        )


if __name__ == "__main__":
    main()
