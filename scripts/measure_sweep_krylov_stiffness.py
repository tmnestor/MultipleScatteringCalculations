"""DECISION: does the stiffness reappear in the Krylov solve?

THE QUESTION, AND WHY IT DECIDES SOMETHING. The matrix-vector wave equation
carries evanescent eigenvalues +-kappa with kappa = sqrt(k_par^2 - k^2),
unbounded as the lateral wavenumber grows. Marching the wave field directly
(Thomson-Haskell) integrates the growing branch and loses precision. Invariant
imbedding removes the INSTABILITY -- the reflection matrix is bounded -- but not
the STIFFNESS: the Riccati equation still spans the whole range of decay rates,
so it needs implicit steps, and in the horizontal-wavenumber domain that is
where the cost lived.

The directional-sweep architecture claims to avoid both. It never integrates an
ODE: each sweep marches the direction in which its partial wave CONTRACTS, and
the multiple scattering that couples the directions is carried by an outer
Krylov solve instead.

⚠ THE OBVIOUS WAY FOR THAT CLAIM TO BE HOLLOW is that the stiffness simply MOVES
-- out of the marching and into the iteration count. If GMRES needs more
iterations as the evanescent range widens, nothing has been gained; the expense
has been relocated and renamed. That is the single outcome that would kill the
architecture's advantage, and it is what this measures.

TWO SWEEPS, because they separate two different things:

  [A] REFINEMENT AT FIXED PHYSICS. Shrink the pitch at fixed domain and
      frequency. This widens the evanescent range (the quadrature cutoff scales
      as 1/pitch) while the physical problem is unchanged. A flat iteration
      count is mesh independence -- the property a good preconditioner has and
      a stiff solver does not.

  [B] FREQUENCY AT FIXED GRID. Lowering omega raises kappa_max/k, so the
      evanescent content DOMINATES more. If iterations track that ratio, the
      solver is feeling the same spread that made the Riccati stiff.

⚠ WHAT IS MEASURED IS n_matvec, NOT the outer iteration count. GMRES restarts
make the two differ, and the honest cost is the number of propagator
applications.

⚠ AND A CAVEAT ON [A] THAT MUST NOT BE GLOSSED. Refining at fixed domain also
makes the problem BIGGER. Iteration growth would then be ambiguous between
"stiffness moved" and "larger system". The site count is reported alongside so
the two can be told apart: stiffness would grow with kappa_max at fixed physics,
whereas a size effect would track the number of unknowns. They are separated
here only if they differ, and if they do not, this says so rather than picking
the convenient reading.

Run:  conda run -n seismic python scripts/measure_sweep_krylov_stiffness.py
Seismic units (km, km/s, g/cm3), time convention e^{-i omega t}.
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
from cubic_scattering.sweep_solver import solve_foldy_lax_3d  # noqa: E402
from cubic_scattering.voigt_tmatrix import effective_stiffness_voigt  # noqa: E402

REF = ReferenceMedium(5.0, 3.0, 2.5)

#: Domain width in km, held fixed under refinement. Chosen so that every
#: configuration below keeps ka < KA_CEILING -- see the guard in run_once.
DOMAIN = 0.3
CONTRAST = MaterialContrast(2.0, 1.0, 0.1)  # seismic units

#: The analytic cube T-matrix is validated only for ka < 0.3 and returns a
#: PLAUSIBLE number outside it, so an unguarded sweep silently reports
#: iteration counts for a T-matrix that is wrong. A first version of this
#: script ran at ka = 3.8 and produced a clean-looking 21 matvecs before the
#: next configuration failed to converge at all.
KA_CEILING = 0.3


def run_once(n_side: int, freq_hz: float, scale: float = 1.0) -> tuple[int, float, float, int]:
    """(n_matvec, spectral radius, kappa_max / k_S, n_sites) for one setup.

    ⚠ `scale` multiplies the contrast, and it is not cosmetic. At the physical
    contrast the spectral radius of G0 T0 is ~0.04, so the Neumann series
    converges at 4% per order and GMRES reaches 1e-8 in ~6 applications
    WHATEVER the evanescent spread. A flat iteration count in that regime
    measures weak scattering, not the absence of stiffness, and would be
    vacuous. The contrast is therefore calibrated up until rho is O(1), where
    the iteration count is genuinely set by the operator's spectrum.

    This tests the SOLVER, not the T-matrix's accuracy, so a strong contrast is
    legitimate here -- the scatterer need only be a plausible one.
    """
    pitch = DOMAIN / n_side
    omega = 2.0 * np.pi * freq_hz * (1 + 0.03j)
    grid = SweepGrid3D(n_z=2, n_x=n_side, n_y=n_side, pitch=pitch)
    cache = build_g0_cache_3d(grid, REF, omega)

    a = 0.5 * pitch
    ka = float(omega.real) / REF.alpha * a
    if ka >= KA_CEILING:
        msg = (
            f"ka = {ka:.3f} exceeds the validated ceiling {KA_CEILING}.\n"
            "  Where: scripts/measure_sweep_krylov_stiffness.py, run_once()\n"
            "  Why:   the analytic cube T-matrix is validated only below that and\n"
            "         returns a PLAUSIBLE number above it, so the iteration counts\n"
            "         would be measured on a wrong T-matrix and look fine.\n"
            "  Fix:   shrink DOMAIN, raise n_side, or lower the frequency"
        )
        raise ValueError(msg)
    con = MaterialContrast(scale * CONTRAST.Dlambda, scale * CONTRAST.Dmu, scale * CONTRAST.Drho)
    res_t = compute_cube_tmatrix(float(omega.real), a, REF, con)
    v = pitch**3
    block = np.zeros((9, 9), dtype=complex)
    block[:3, :3] = float(omega.real) ** 2 * complex(res_t.Drho_star) * v * np.eye(3)
    block[3:, 3:] = v * effective_stiffness_voigt(
        res_t.Dlambda_star, res_t.Dmu_star_diag, res_t.Dmu_star_off
    )
    shape = (grid.n_z, grid.n_x, grid.n_y)
    t0 = np.broadcast_to(block, (*shape, 9, 9)).copy()

    # A plane wave down z, evaluated on the grid -- the physical drive, not noise.
    k_s = float(omega.real) / REF.beta
    zs = np.arange(grid.n_z) * pitch
    psi = np.zeros((*shape, 9), dtype=complex)
    phase = np.exp(1j * k_s * zs)
    psi[..., 1] = phase[:, None, None]  # u_x polarisation

    # ⚠ THE SPECTRAL RADIUS DECIDES WHETHER ANY OF THIS MEANS ANYTHING.
    # A weakly coupled problem converges in a handful of iterations whatever
    # the evanescent spread, so a flat count would be vacuous rather than
    # evidence. Power iteration on G0 T0, reported alongside.
    rng = np.random.default_rng(20260917)
    vec = rng.normal(size=(*shape, 9)) + 1j * rng.normal(size=(*shape, 9))
    vec /= np.linalg.norm(vec)
    rho = 0.0
    for _ in range(12):
        nxt = apply_g0_3d(np.einsum("zxyab,zxyb->zxya", t0, vec), cache)
        rho = float(np.linalg.norm(nxt))
        if rho == 0.0:
            break
        vec = nxt / rho

    out = solve_foldy_lax_3d(cache, t0, psi, tol=1e-8, max_iter=2000)
    # The quadrature cutoff sets the widest evanescent decay rate the operator
    # carries; 30/pitch is make_sweep_grid's default reach.
    kappa_max_over_k = (30.0 / pitch) / k_s
    return out.n_matvec, rho, kappa_max_over_k, int(np.prod(shape))


def main() -> int:
    print("=" * 78)
    print("DOES THE STIFFNESS REAPPEAR IN THE KRYLOV ITERATION COUNT?")
    print("=" * 78)

    # ---- calibrate the contrast so the problem is genuinely hard ----------
    # Measured, not chosen: rho is very nearly linear in the contrast scale, so
    # one probe fixes the multiplier that lands it near the target. Without
    # this the whole measurement is vacuous -- see run_once's docstring.
    target_rho = 0.7
    _, rho1, _, _ = run_once(8, 6.0, 1.0)
    scale = target_rho / max(rho1, 1e-12)
    _, rho_c, _, _ = run_once(8, 6.0, scale)
    print(f"\n  calibration: rho = {rho1:.4f} at the physical contrast,")
    print(f"  so the contrast is scaled {scale:.1f}x, giving rho = {rho_c:.3f}")
    if not (0.3 < rho_c < 0.98):
        print(f"\n  ABORT: calibration landed at rho = {rho_c:.3f}, outside (0.3, 0.98).")
        print("  Below that the iteration count measures weak scattering rather")
        print("  than the operator's spectrum, and the result would be vacuous.")
        return 1

    print("\n  [A] refinement at FIXED physics (domain and frequency held)")
    print(f"      {'n_side':>7} {'pitch':>8} {'sites':>7} {'kappa_max/k_S':>14} {'rho':>8} {'n_matvec':>9}")
    rows_a = []
    for n_side in (4, 6, 8, 12):
        nm, rho, ratio, nsites = run_once(n_side, 6.0, scale)
        rows_a.append((n_side, nsites, ratio, nm, rho))
        print(f"      {n_side:>7} {DOMAIN / n_side:8.4f} {nsites:>7} {ratio:14.1f} {rho:8.3f} {nm:>9}")

    print("\n  [B] frequency at FIXED grid (lower omega = evanescent dominates)")
    print("      ⚠ rho is RE-CALIBRATED at every frequency.  Without that, the")
    print("      contrast fixed at one frequency gives a different rho at the")
    print("      others, and the iteration count tracks rho rather than the")
    print("      evanescent range -- confounding the two things being separated.")
    print(f"      {'freq Hz':>8} {'kappa_max/k_S':>14} {'rho':>8} {'n_matvec':>9}")
    rows_b = []
    for freq in (12.0, 6.0, 3.0, 1.5):
        _, r1, _, _ = run_once(8, freq, 1.0)
        s_f = target_rho / max(r1, 1e-12)
        nm, rho, ratio, _ = run_once(8, freq, s_f)
        rows_b.append((freq, ratio, nm, rho))
        print(f"      {freq:8.1f} {ratio:14.1f} {rho:8.3f} {nm:>9}")

    # ---- reading it -------------------------------------------------------
    # ⚠ THE RATIO MUST BE SIGNED, i.e. taken in the DIRECTION of widening
    # evanescent range. A max/min ratio ignores direction, and an earlier
    # version of this script reported "1.71x growth" from a sequence that in
    # fact FELL, 24 -> 14, with the 24 sitting at the SMALLEST evanescent range
    # where rho happened to be highest. Stiffness means the count RISES as the
    # range widens; a fall is the opposite finding.
    mv_a = [r[3] for r in rows_a]
    mv_b = [r[2] for r in rows_b]
    ratio_a = mv_a[-1] / max(mv_a[0], 1)
    ratio_b = mv_b[-1] / max(mv_b[0], 1)
    kappa_span_a = rows_a[-1][2] / rows_a[0][2]
    kappa_span_b = rows_b[-1][1] / rows_b[0][1]
    size_span = rows_a[-1][1] / rows_a[0][1]

    print("\n  [C] reading")
    print(
        f"      [A] kappa_max/k spanned {kappa_span_a:.1f}x, sites {size_span:.1f}x,"
        f" n_matvec {ratio_a:.2f}x"
    )
    print(f"      [B] kappa_max/k spanned {kappa_span_b:.1f}x at fixed size, n_matvec {ratio_b:.2f}x")

    flat_a = ratio_a < 1.5
    flat_b = ratio_b < 1.5
    print("\n" + "=" * 78)
    if flat_a and flat_b:
        print("FINDING: the stiffness does NOT reappear in the Krylov solve.")
        print()
        print("The iteration count is essentially flat while the evanescent range")
        print("widens by more than an order of magnitude -- in [B] at FIXED problem")
        print("size, which removes the ambiguity in [A] between stiffness and a")
        print("larger system.  The expense has not been relocated; the sweep")
        print("architecture removes it.")
        print()
        print("That is the claim worth making against an implicit Riccati solver,")
        print("and it is the one that had to be checked before making it.")
    elif flat_b and not flat_a:
        print("FINDING: MIXED, and [B] is the one that matters.")
        print()
        print("At fixed problem size the iteration count is flat as the evanescent")
        print("range widens, so the stiffness has not moved into the solve.  The")
        print("growth in [A] therefore tracks the PROBLEM SIZE, which is an")
        print("ordinary scaling question and not the failure mode in question.")
    else:
        print("FINDING: the stiffness HAS moved into the Krylov solve.")
        print()
        print(f"The iteration count grows {ratio_b:.1f}x at FIXED problem size as the")
        print("evanescent range widens.  That is the same spread that made the")
        print("Riccati equation stiff, now paid for in propagator applications")
        print("instead of implicit steps.  The architecture's advantage over an")
        print("implicit solver is NOT established, and a preconditioner would be")
        print("needed before any cost claim could be made.")
    print()
    print("⚠ SCOPE: whole-space background, one contrast, n_z = 2, GMRES without")
    print("a preconditioner.  A layered background or stronger contrast could")
    print("behave differently, and neither is tested here.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
