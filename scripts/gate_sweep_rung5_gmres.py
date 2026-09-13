#!/usr/bin/env python3
"""GATE, rung 5: the GMRES Foldy-Lax solve on the directional-sweep G0.

WHAT THIS GATE TESTS, AND WHAT IT DOES NOT
------------------------------------------
The operator G0 is gated against INDEPENDENT arbiters elsewhere: rung 1/1b
against the bundled kernel of horizontal_greens, rung 2b and rung 3 against the
closed-form Kupradze propagator exact_propagator_9x9. This gate tests the SOLVER
built around that operator -- the T0 embedding, the sign of (I - G0 T0), the
reshaping, and the Krylov convergence -- by two algorithms that are independent
of GMRES but share the operator:

  [5a] DENSE. Materialise (I - G0 T0) column by column and solve it directly
       with a dense LU. Different algorithm, same operator. Exact: 1e-10.

  [5b] NEUMANN. Sum psi = sum_n (G0 T0)^n psi_inc explicitly. A third algorithm,
       and the one that makes the physical claim concrete -- every order of
       multiple scattering is built by the iteration, not by the propagator.
       Valid only where the series converges; the spectral radius is reported.

WHY slab_scattering IS NOT USED AS THE ARBITER HERE
---------------------------------------------------
The design document named slab_scattering.compute_slab_scattering as the rung-5
arbiter. It is not a valid one for a 2.5-D solver, and the reason is geometric,
not numerical. SlabGeometry is M x M x N_z: a finite SQUARE footprint of extent
M*d in y, with the incident field evaluated cell by cell. A 2.5-D solver models
a medium that is infinite and INVARIANT in y. A finite strip cannot represent
that -- its y-edges do not vanish, they only decay with M -- so the two codes
solve different problems and any single number from the comparison would
measure the strip width, not the sweeps.

The cross-architecture comparison becomes available at stage 2, when the in-out
k_y sweep makes this solver 3-D and the two codes model the same thing. Until
then it is DEFERRED, not passed and not failed.

Run serially. Seismic units (km/s, g/cm3), time convention e^{-i omega t}.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cubic_scattering.directional_sweeps import (  # noqa: E402
    apply_g0,
    build_g0_cache,
    make_sweep_grid,
)
from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from cubic_scattering.sweep_solver import solve_sweep_foldy_lax  # noqa: E402

REF = ReferenceMedium(alpha=5.0, beta=3.0, rho=2.5)
OMEGA = 2 * np.pi * (1.0 + 0.03j)
PITCH = 0.25
N_Z, N_X = 2, 6


def _setup(scale: float):
    """A y-invariant model at k_y = 0, so no k_y integration is needed.

    With the medium invariant in y AND the incident field carrying k_y = 0, the
    whole problem lives at k_y = 0: the solution is y-invariant and the 2.5-D
    solver is complete, not an integrand.
    """
    grid = make_sweep_grid(N_Z, N_X, PITCH, ky=0.0, n_kz=256, n_kx=256)
    cache = build_g0_cache(grid, REF, OMEGA)
    rng = np.random.default_rng(20260913)
    t0 = scale * (rng.standard_normal((N_Z, N_X, 9, 9)) + 1j * rng.standard_normal((N_Z, N_X, 9, 9)))
    psi_inc = rng.standard_normal((N_Z, N_X, 9)) + 0j
    return cache, t0, psi_inc


def dense_operator(cache, t0):
    """Materialise (I - G0 T0) column by column."""
    size = N_Z * N_X * 9
    shape = (N_Z, N_X, 9)
    mat = np.zeros((size, size), dtype=complex)
    for c in range(size):
        e = np.zeros(size, dtype=complex)
        e[c] = 1.0
        psi = e.reshape(shape)
        col = psi - apply_g0(np.einsum("zxab,zxb->zxa", t0, psi), cache)
        mat[:, c] = col.ravel()
    return mat


def neumann_sum(cache, t0, psi_inc, n_terms: int):
    """Sum psi = sum_n (G0 T0)^n psi_inc explicitly."""
    term = psi_inc.copy()
    total = psi_inc.copy()
    for _ in range(n_terms):
        term = apply_g0(np.einsum("zxab,zxb->zxa", t0, term), cache)
        total = total + term
    return total, float(np.abs(term).max() / np.abs(total).max())


def main() -> int:
    print("=" * 74)
    print("GATE rung 5 -- GMRES Foldy-Lax on the directional-sweep G0")
    print(f"  alpha={REF.alpha} beta={REF.beta} rho={REF.rho} (km/s, g/cm3)")
    print(f"  omega={OMEGA:.4f}  pitch={PITCH} km  n_z={N_Z}  n_x={N_X}  k_y=0")
    print("=" * 74)

    ok = True
    print("\n[5a] DENSE -- GMRES vs a direct dense LU on the same operator")
    print(f"      {'T0 scale':>10} {'spec. radius':>13} {'n_matvec':>9}   {'relative diff':>14}")
    # The last two scales push the spectral radius of G0 T0 toward 1, where the
    # multiple scattering is strong and GMRES must actually work. A gate run
    # only at weak contrast passes because the problem is too easy -- two
    # iterations prove nothing about the solver.
    for scale in (1e-4, 1e-2, 1e-1, 1.0, 2.0):
        cache, t0, psi_inc = _setup(scale)
        mat = dense_operator(cache, t0)
        dense = np.linalg.solve(mat, psi_inc.ravel()).reshape(psi_inc.shape)

        res = solve_sweep_foldy_lax(cache, t0, psi_inc, tol=1e-12, max_iter=500)
        rel = float(np.abs(res.psi - dense).max() / np.abs(dense).max())

        # Spectral radius of G0 T0 = how hard the multiple scattering is.
        rho_gt = float(np.abs(np.linalg.eigvals(np.eye(mat.shape[0]) - mat)).max())
        good = rel < 1e-10
        ok = ok and good
        print(f"      {scale:10.0e} {rho_gt:13.3e} {res.n_matvec:9d}   {rel:14.3e}")
    print(f"      target < 1e-10 at every contrast  ->  {'PASS' if ok else 'FAIL'}")

    print("\n[5b] NEUMANN -- GMRES vs an explicit multiple-scattering series")
    print("      Valid only where the series converges; the last term is shown.")
    print(f"      {'T0 scale':>10} {'terms':>6} {'last term':>11}   {'relative diff':>14}")
    okn = True
    for scale, n_terms in ((1e-4, 6), (1e-2, 30), (1.0, 120)):
        cache, t0, psi_inc = _setup(scale)
        series, tail = neumann_sum(cache, t0, psi_inc, n_terms)
        res = solve_sweep_foldy_lax(cache, t0, psi_inc, tol=1e-12, max_iter=500)
        rel = float(np.abs(res.psi - series).max() / np.abs(series).max())
        good = rel < 1e-8
        okn = okn and good
        print(f"      {scale:10.0e} {n_terms:6d} {tail:11.3e}   {rel:14.3e}")
    ok = ok and okn
    print(f"      target < 1e-8 where the series has converged  ->  {'PASS' if okn else 'FAIL'}")

    print("\n[5c] CROSS-ARCHITECTURE vs slab_scattering -- DEFERRED, not passed.")
    print("      slab_scattering is 3-D on a finite M x M footprint and cannot")
    print("      represent a y-invariant medium; a 2.5-D solver models exactly that.")
    print("      The comparison becomes meaningful at stage 2, when the in-out k_y")
    print("      sweep makes this solver 3-D. See this file's module docstring.")

    print("\n" + "=" * 74)
    print(f"GATE rung 5: {'PASS' if ok else 'FAIL'}  (5c deferred)")
    print("=" * 74)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
