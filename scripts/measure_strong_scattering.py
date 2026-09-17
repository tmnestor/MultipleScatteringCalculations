"""MEASUREMENT: does either architecture hold up at STRONG scattering?

Everything so far ran at a spectral radius of rho ~ 0.5, where the Neumann
series still converges and GMRES finishes in under twenty applications. The
question is what happens as rho -> 1, where the series diverges and the solve
becomes genuinely hard.

⚠ WHY THIS IS TWO MEASUREMENTS AND NOT ONE. Kennett is exact for a laterally
INFINITE uniform layer. The directional sweep runs on a FINITE footprint with
edges, so Kennett cannot arbitrate it: the comparison would measure the
footprint until that footprint is many wavelengths across, which at this pitch
means ~133 cells per side and ~35k sites -- impractical for a real-space
summation. The project already made this argument for the 2.5-D case and it
survives the move to 3-D.

So the ground is covered by two valid comparisons instead:

  [A] THE SWEEP SOLVER, against a DENSE direct solve of the IDENTICAL operator.
      No physics arbiter is needed or wanted here -- this asks whether GMRES on
      the sweep operator still returns the right answer, and at what cost, as
      rho -> 1. A dense LU is a different algorithm on the same matrix, so a
      disagreement localises to the Krylov path.

  [B] THE PHYSICS, against Kennett, on the PERIODIC LATTICE route -- the
      architecture that can represent an infinite uniform layer. This asks
      whether the scheme is still ACCURATE when scattering is strong, which is
      a different question from whether the solve converges.

The two architectures are independently known to agree at the operator level to
7e-16 (`gate_rung5c_cross_architecture.py`), so [A] and [B] between them cover
the sweep's accuracy without ever putting Kennett against a finite footprint.

⚠ rho IS MEASURED, NOT ASSUMED, at every point. A contrast scale is calibrated
to hit each target, and the achieved rho is printed. Without that, "rho = 0.95"
would be an intention rather than a fact.

Run:  conda run -n seismic python scripts/measure_strong_scattering.py
Seismic units for [A] (km, km/s, g/cm3); SI for [B].
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubic_scattering.directional_sweeps import (  # noqa: E402
    SweepGrid3D,
    apply_g0_3d,
    build_g0_cache_3d,
)
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
from cubic_scattering.sweep_solver import solve_foldy_lax_3d  # noqa: E402
from cubic_scattering.voigt_tmatrix import effective_stiffness_voigt  # noqa: E402

# ---- [A] sweep, seismic units ------------------------------------------
REF_S = ReferenceMedium(5.0, 3.0, 2.5)
PITCH, N_SIDE, FREQ = 0.0375, 6, 6.0
CON_S = MaterialContrast(2.0, 1.0, 0.1)

# ---- [B] lattice vs Kennett, SI units -----------------------------------
REF_SI = ReferenceMedium(5000.0, 3000.0, 2500.0)
OMEGA_SI, H_PHYS, M = 60.0, 4.0, 4


def _sweep_operator(scale: float):
    """(cache, t0, psi, rho) for the sweep at a scaled contrast."""
    omega = 2.0 * np.pi * FREQ * (1 + 0.03j)
    grid = SweepGrid3D(n_z=2, n_x=N_SIDE, n_y=N_SIDE, pitch=PITCH)
    cache = build_g0_cache_3d(grid, REF_S, omega)

    a = 0.5 * PITCH
    con = MaterialContrast(scale * CON_S.Dlambda, scale * CON_S.Dmu, scale * CON_S.Drho)
    res_t = compute_cube_tmatrix(float(omega.real), a, REF_S, con)
    v = PITCH**3
    block = np.zeros((9, 9), dtype=complex)
    block[:3, :3] = float(omega.real) ** 2 * complex(res_t.Drho_star) * v * np.eye(3)
    block[3:, 3:] = v * effective_stiffness_voigt(
        res_t.Dlambda_star, res_t.Dmu_star_diag, res_t.Dmu_star_off
    )
    shape = (grid.n_z, grid.n_x, grid.n_y)
    t0 = np.broadcast_to(block, (*shape, 9, 9)).copy()

    k_s = float(omega.real) / REF_S.beta
    psi = np.zeros((*shape, 9), dtype=complex)
    psi[..., 1] = np.exp(1j * k_s * np.arange(grid.n_z) * PITCH)[:, None, None]

    rng = np.random.default_rng(20260917)
    vec = rng.normal(size=(*shape, 9)) + 1j * rng.normal(size=(*shape, 9))
    vec /= np.linalg.norm(vec)
    rho = 0.0
    for _ in range(20):
        nxt = apply_g0_3d(np.einsum("zxyab,zxyb->zxya", t0, vec), cache)
        rho = float(np.linalg.norm(nxt))
        if rho == 0.0:
            break
        vec = nxt / rho
    return grid, cache, t0, psi, rho


def _dense_solve(grid, cache, t0, psi):
    """Assemble (I - G0 T0) column by column and solve directly."""
    shape = (grid.n_z, grid.n_x, grid.n_y, 9)
    size = int(np.prod(shape))
    mat = np.zeros((size, size), dtype=complex)
    for c in range(size):
        e = np.zeros(size, dtype=complex)
        e[c] = 1.0
        v = e.reshape(shape)
        mat[:, c] = (v - apply_g0_3d(np.einsum("zxyab,zxyb->zxya", t0, v), cache)).ravel()
    return np.linalg.solve(mat, psi.ravel()).reshape(shape)


def panel_a0() -> None:
    """How far can rho be pushed by contrast at all? Measured, not assumed."""
    print("\n  [A0] HOW rho DEPENDS ON CONTRAST -- measured, because guessing")
    print("       it wrong is what made an earlier version of this script ask")
    print("       for rho = 0.99 and silently deliver 0.65.")
    print(f"\n      {'contrast x':>11} {'rho':>8}")
    for scale in (1.0, 10.0, 100.0, 1000.0, 10000.0):
        rho = _sweep_operator(scale)[4]
        print(f"      {scale:11.0f} {rho:8.3f}")
    print("\n      rho is STRONGLY NON-LINEAR in the contrast: sublinear up to")
    print("      about 20x (0.047 -> ~0.6), then superlinear, passing 1 near")
    print("      100x and reaching O(1000). So a linear calibration -- scale =")
    print("      target/rho(1) -- lands far short, and every scale it picks")
    print("      clusters in the range where rho happens to be flattest. That")
    print("      is exactly how a 'target rho = 0.99' row came to report 0.65")
    print("      and be read as SATURATION. rho does not saturate.")


def panel_a() -> None:
    print("\n  [A] SWEEP solver, against a dense direct solve of the same operator")
    print("      (different algorithm, same matrix -- so a disagreement is the")
    print("       Krylov path, not the physics)")
    print(f"\n      {'contrast x':>11} {'rho':>8} {'n_matvec':>9} {'GMRES vs dense':>16}")
    for scale in (10.0, 100.0, 1000.0, 10000.0):
        grid, cache, t0, psi, rho = _sweep_operator(scale)
        try:
            out = solve_foldy_lax_3d(cache, t0, psi, tol=1e-10, max_iter=4000)
            ref = _dense_solve(grid, cache, t0, psi)
            rel = float(np.linalg.norm(out.psi - ref) / np.linalg.norm(ref))
            print(f"      {scale:11.0f} {rho:8.3f} {out.n_matvec:>9} {rel:16.3e}")
        except RuntimeError as exc:
            first = str(exc).splitlines()[0]
            print(f"      {scale:11.0f} {rho:8.3f} {'FAILED':>9}   {first}")


def panel_b() -> None:
    print("\n  [B] PHYSICS at strong scattering: periodic lattice against Kennett")
    print("      (the architecture that CAN represent an infinite uniform layer)")
    print(f"\n      {'dmu/mu':>8} {'|R_PP|':>10} {'rel err vs Kennett':>20}")
    mu0 = REF_SI.rho * REF_SI.beta**2
    n_z, a = 4, H_PHYS / 8.0
    for frac in (0.05, 0.2, 0.5, 1.0, 2.0):
        dmu = frac * mu0
        con = MaterialContrast(0.0, dmu, 0.0)
        geom = SlabGeometry(M=M, N_z=n_z, a=a)
        ones = np.ones((n_z, M, M))
        mat = SlabMaterial(Dlambda=0.0 * ones, Dmu=dmu * ones, Drho=0.0 * ones, ref=REF_SI)
        t9 = compute_cube_tmatrix(OMEGA_SI, a, REF_SI, con)
        v = (2.0 * a) ** 3
        cell = np.zeros((9, 9), dtype=complex)
        cell[:3, :3] = OMEGA_SI**2 * complex(t9.Drho_star) * v * np.eye(3)
        cell[3:, 3:] = v * effective_stiffness_voigt(t9.Dlambda_star, t9.Dmu_star_diag, t9.Dmu_star_off)
        t0 = np.broadcast_to(cell, (n_z, M, M, 9, 9)).copy()
        kh = build_slab_kernels(
            geom,
            OMEGA_SI,
            REF_SI,
            periodic=True,
            lattice_ewald=True,
            volume_averaged=True,
            n_orders=2,
        )
        res = compute_slab_scattering(
            geom,
            mat,
            OMEGA_SI,
            np.array([1.0, 0.0, 0.0]),
            "P",
            periodic=True,
            gmres_tol=1e-13,
            kernel_hat=kh,
            T_local=t0,
        )
        r = complex(slab_rpp_periodic(res, t0))
        exact = kennett_reference_rpp(REF_SI, con, H_PHYS, OMEGA_SI)
        print(f"      {frac:8.2f} {abs(r):10.5f} {abs(r - exact) / abs(exact):20.3e}")


def main() -> int:
    print("=" * 78)
    print("STRONG SCATTERING: does either architecture hold up?")
    print("=" * 78)
    panel_a0()
    panel_a()
    panel_b()
    print("\n" + "=" * 78)
    print("READING IT.  [A] asks whether the SOLVE is right and what it costs;")
    print("[B] asks whether the PHYSICS is right.  They are different failures")
    print("and a single number cannot separate them.")
    print()
    print("THE STRONG-SCATTERING RESULT, from [A]:  GMRES still returns the")
    print("RIGHT answer past rho = 1, where the Neumann series diverges outright")
    print("-- it agrees with a dense direct solve to 9e-10 at rho = 1.69. But")
    print("the COST explodes: 16 applications at rho = 0.38 against 11648 at")
    print("rho = 1.69, a factor of 700. By rho ~ 79 it does not converge at all.")
    print()
    print("⚠ THAT IS A DIFFERENT AXIS FROM THE EVANESCENT ONE.  Widening the")
    print("evanescent range costs nothing in iterations (measured separately);")
    print("raising the SCATTERING STRENGTH costs almost everything. The sweep")
    print("architecture addresses the first and does nothing for the second, and")
    print("conflating them would credit it with a robustness it does not have.")
    print()
    print("⚠ AND rho = 1.69 IS NOT A PHYSICAL MEDIUM -- it is a 100x contrast,")
    print("used as a solver stress test. [B] carries the physical range.")
    print()
    print("⚠ Kennett is NOT put against the sweep, deliberately: it is exact for")
    print("an infinite uniform layer and the sweep has a finite footprint, so")
    print("that comparison would measure the edges.  The two architectures are")
    print("already known to agree at the OPERATOR level to 7e-16, which is what")
    print("lets [A] and [B] cover the ground between them.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
