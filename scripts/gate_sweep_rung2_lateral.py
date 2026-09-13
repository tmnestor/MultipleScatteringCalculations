#!/usr/bin/env python3
"""GATE, rung 2: the Cartesian intra-plane lateral sweep.

Three checks, in increasing order of what they can prove:

  [2]  The running sweep reproduces the O(N^2) pairwise double sum, with a
       DISTINCT source at every site. Exact identity -- two algorithms, one
       physics. Target 1e-12 (round-off in the shared k_z quadrature).

  [2c] The mandatory CONTROL. Shows that the same comparison run with a uniform
       source cannot discriminate: an implementation that averaged the sites
       before sweeping would pass it. Printed so that a reader can see the weak
       version is weak, and cannot mistake it for evidence.

  [2b] The physics. Integrating the 2.5-D kernel over k_y must give the exact
       real-space propagator at the same separation. This is QUADRATURE-LIMITED,
       not exact, so it is reported at two k_y resolutions and passes by
       FALLING, not by clearing a fixed bar. A single number here would be
       indistinguishable from a converged wrong answer.

Run serially. Seismic units (km/s, g/cm3), time convention e^{-i omega t}.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cubic_scattering.directional_sweeps import make_sweep_grid, sweep_x  # noqa: E402
from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from cubic_scattering.horizontal_greens import (  # noqa: E402
    exact_propagator_9x9,
    horizontal_greens_direct,
)
from cubic_scattering.sweep_kernels import lateral_split_9x9  # noqa: E402

REF = ReferenceMedium(alpha=5.0, beta=3.0, rho=2.5)
OMEGA = 2 * np.pi * (1.0 + 0.03j)
PITCH = 0.25


def _splits(grid):
    right = lateral_split_9x9(grid.ky, grid.kz_nodes, grid.pitch, OMEGA, REF, direction="right")
    left = lateral_split_9x9(grid.ky, grid.kz_nodes, grid.pitch, OMEGA, REF, direction="left")
    return right, left


def _pairwise(sources, grid, split_right, split_left):
    """O(N^2) reference: a double loop over ordered pairs, not a recursion."""
    n_x = sources.shape[1]
    out = np.zeros_like(sources)
    w = grid.kz_weights
    for i in range(n_x):
        for j in range(n_x):
            if i == j:
                continue
            n = abs(i - j)
            split = split_right if j < i else split_left
            kern = np.einsum("abk,k->abk", split.amp_p, split.phase_p**n) + np.einsum(
                "abk,k->abk", split.amp_s, split.phase_s**n
            )
            block = np.einsum("k,abk->ab", w, kern)
            out[:, i, :] += sources[:, j, :] @ block.T
    return out


def check_resummation() -> float:
    """[2] Sweep == pairwise sum, distinct source at every site."""
    n_z, n_x = 2, 24
    grid = make_sweep_grid(n_z, n_x, PITCH, ky=0.6, n_kz=512)
    right, left = _splits(grid)
    rng = np.random.default_rng(20260913)
    sources = rng.standard_normal((n_z, n_x, 9)) + 1j * rng.standard_normal((n_z, n_x, 9))

    got = sweep_x(sources, grid, right, left)
    want = _pairwise(sources, grid, right, left)
    return float(np.abs(got - want).max() / np.abs(want).max())


def check_control() -> tuple[float, float]:
    """[2c] The vacuity control: uniform source cannot discriminate."""
    n_z, n_x = 1, 8
    grid = make_sweep_grid(n_z, n_x, PITCH, ky=0.6, n_kz=256)
    right, left = _splits(grid)

    uniform = np.ones((n_z, n_x, 9), dtype=complex)
    u_avg = np.broadcast_to(uniform.mean(axis=1, keepdims=True), uniform.shape).copy()
    weak = float(np.abs(sweep_x(uniform, grid, right, left) - sweep_x(u_avg, grid, right, left)).max())

    rng = np.random.default_rng(7)
    varied = rng.standard_normal((n_z, n_x, 9)) + 0j
    v_avg = np.broadcast_to(varied.mean(axis=1, keepdims=True), varied.shape).copy()
    a = sweep_x(varied, grid, right, left)
    b = sweep_x(v_avg, grid, right, left)
    strong = float(np.abs(a - b).max() / np.abs(a).max())
    return weak, strong


def summed_kernel(ky_max: float, n_ky: int, n_kz: int) -> np.ndarray:
    """Integrate the 2.5-D split kernel over k_y at one-pitch separation."""
    grid = make_sweep_grid(1, 2, PITCH, ky=0.0, n_kz=n_kz)
    ky_nodes = np.linspace(-ky_max, ky_max, n_ky)
    dky = ky_nodes[1] - ky_nodes[0]

    total = np.zeros((9, 9), dtype=complex)
    for idx, ky in enumerate(ky_nodes):
        split = lateral_split_9x9(ky, grid.kz_nodes, PITCH, OMEGA, REF, direction="right")
        kern = np.einsum("abk,k->abk", split.amp_p, split.phase_p) + np.einsum(
            "abk,k->abk", split.amp_s, split.phase_s
        )
        weight = 0.5 if idx in (0, n_ky - 1) else 1.0
        total += np.einsum("k,abk->ab", grid.kz_weights, kern) * weight * dky / (2 * np.pi)
    return total


def main() -> int:
    print("=" * 74)
    print("GATE rung 2 -- Cartesian intra-plane lateral sweep")
    print(f"  alpha={REF.alpha} beta={REF.beta} rho={REF.rho} (km/s, g/cm3)")
    print(f"  omega={OMEGA:.4f}  pitch={PITCH} km")
    print("=" * 74)

    print("\n[2] sweep == pairwise double sum, DISTINCT source per site")
    r2 = check_resummation()
    print(f"      n_z=2  n_x=24  n_kz=512     relative residual = {r2:.3e}")
    ok2 = r2 < 1e-12
    print(f"      target < 1e-12  ->  {'PASS' if ok2 else 'FAIL'}")

    print("\n[2c] CONTROL -- is the uniform-source version vacuous?")
    weak, strong = check_control()
    print(f"      uniform source, sweep(s) vs sweep(mean s): {weak:.3e}   <- cannot discriminate")
    print(f"      varied  source, sweep(s) vs sweep(mean s): {strong:.3e}   <- discriminates")
    okc = weak == 0.0 and strong > 1e-2
    print(f"      the weak test is vacuous, the real one is not  ->  {'PASS' if okc else 'FAIL'}")

    print("\n[2b] PHYSICS -- k_y-integrated kernel vs the exact real-space propagator")
    print("      QUADRATURE-LIMITED, not exact: pass = the residual FALLS.")
    want = exact_propagator_9x9(x=PITCH, y=0.0, z=0.0, omega=OMEGA, ref=REF)
    scale = np.abs(want).max()
    print(f"      {'ky_max':>8} {'n_ky':>6} {'n_kz':>6}   {'relative residual':>18}")
    residuals = []
    for ky_max, n_ky, n_kz in ((60.0, 1201, 1024), (120.0, 3201, 2048)):
        got = summed_kernel(ky_max, n_ky, n_kz)
        rel = float(np.abs(got - want).max() / scale)
        residuals.append(rel)
        print(f"      {ky_max:8.1f} {n_ky:6d} {n_kz:6d}   {rel:18.3e}")
    okb = residuals[-1] < residuals[0]
    print(f"      residual falls under refinement  ->  {'PASS' if okb else 'FAIL'}")

    print("\n[2b'] cross-check of the G block against the independent 2-D quadrature")
    print("      NOTE: horizontal_greens_direct returns the OLD (x, y, z) index")
    print("      ordering, not the seismological (z, x, y) used everywhere else in")
    print("      this package -- it builds kL = [kx, ky, kz] while")
    print("      post_kx_residue_kernel_9x9_vec builds [kz, kx, ky]. Without the")
    print("      permutation below the two disagree by 3.6e-1, a fixed offset that")
    print("      does NOT fall under refinement (measured at kmax 60..240, nk")
    print("      600..2400). That is a convention mismatch, not a physics error.")
    got_g = summed_kernel(120.0, 3201, 2048)[:3, :3]
    perm = [2, 0, 1]  # seismological (z, x, y) index <- direct's (x, y, z) index
    print(f"      {'kmax':>8} {'nk':>6}   {'relative residual':>18}")
    rels = []
    for kmax, nk in ((120.0, 1200), (120.0, 2400)):
        g_ref = horizontal_greens_direct(
            PITCH, 0.0, kmax, nk, omega=OMEGA, rho=REF.rho, alpha=REF.alpha, beta=REF.beta
        )
        g_ref = g_ref[np.ix_(perm, perm)]
        rel_g = float(np.abs(got_g - g_ref).max() / np.abs(g_ref).max())
        rels.append(rel_g)
        print(f"      {kmax:8.1f} {nk:6d}   {rel_g:18.3e}")
    print("      (a third code path; it too is quadrature-limited, and slower to")
    print("       converge than the split kernel, so this bounds rather than pins)")

    print("\n" + "=" * 74)
    verdict = ok2 and okc and okb
    print(f"GATE rung 2: {'PASS' if verdict else 'FAIL'}")
    print("=" * 74)
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
