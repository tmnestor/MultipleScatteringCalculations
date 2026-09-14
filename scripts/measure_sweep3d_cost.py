#!/usr/bin/env python3
"""MEASUREMENT, not a gate: the transverse quadrature for 3-D, and its cost.

Task 1 of docs/plans/2026-09-14-cartesian-directional-sweeps-stage2.md. This
script is the record of WHY that plan's architecture changed; it is kept rather
than deleted so the reasoning survives the decision.

THREE FINDINGS, in the order they were forced.

[F1] BRANCH POINTS ON THE CONTOUR. The post-k_y-residue kernel carries 1/ky_L
     and 1/ky_T, which vanish on the circles |k_perp| = kP and kS -- both INSIDE
     the integration domain. At REAL omega a midpoint rule straddles an
     integrable 1/sqrt singularity and does not converge: 1.9e-1 -> 1.1e-1 over
     an 8x refinement, non-monotone. Stage 1 types omega as COMPLEX throughout
     for exactly this reason. With damping 0.01-0.03, convergence is restored.
     A real omega here is not a simplification; it is a defect.

[F2] BOTH CONSTRAINTS BIND. The error is not a function of node spacing alone:
     at dk = 0.3125, kr_max = 40 gives 4.9e-3 while kr_max = 80 gives 5.8e-5.
     Cutoff and spacing must be refined together. The converged rule measured
     here is kr*pitch = 30, dk = 0.156 -> 3.2e-7 at 1.85M nodes; kr*pitch = 40
     gives an IDENTICAL error, which is how we know 30 is where truncation stops
     binding rather than merely where the table ends.

     Do NOT pick a rule by taking the smallest error in the table. An earlier
     version of this script did exactly that and selected a row from a
     non-converged sweep, understating the memory below by 16x.

     A CHEAPER OPERATING POINT, if 1e-6 suffices: kr*pitch = 20, n/axis = 1024
     gives 1.3e-6 at 823,592 nodes -- 45% of the converged rule's cost. It is
     mildly truncation-limited (kr*pitch = 30 at the same dk = 0.156 reaches
     3.2e-7), so use it only where the target is stated and met, never as the
     default. Node count is what Task 2's budget turns on, so the choice is
     worth making explicitly rather than by habit.

[F3] THE COST, which is what changed the plan. A 2-D transverse rule cannot be
     carried through a running sweep: the accumulator is (n_z, n_x, n_k, 9),
     which is 7.6 GB at the 5.8e-5 rule and 68.3 GB at 3.2e-7 on a 16^3 lattice,
     with two live at once. Storing the inter-plane stack spectrally,
     (n_z, n_z, 9, 9, n_k), is 615 GB. Stage 1 escapes this on two counts at
     once -- one fewer real-space axis AND a 1-D transverse rule.

     The same operator stored in REAL SPACE is 149 MB. That is the architecture
     the plan now uses: closed-form real-space tables for Dz = 0, and the
     spectral propagator transformed to real space once at build time for
     Dz != 0.

METHOD NOTE. `post_ky_residue_kernel_9x9_vec` is vectorised over k_x at ONE k_z,
so a polar (r, theta) rule would cost one Python call per node. A tensor product
in (k_z, k_x) with a RADIAL MASK keeps the radial economy -- the corners are what
the mask drops -- while allowing one vectorised call per k_z row.

Run:  conda run -n seismic python scripts/measure_sweep3d_cost.py
Seismic units (km/s, g/cm3), time convention e^{-i omega t}.
"""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from cubic_scattering.horizontal_greens import (  # noqa: E402
    exact_propagator_9x9,
    post_ky_residue_kernel_9x9_vec,
)

REF = ReferenceMedium(5.0, 3.0, 2.5)
PITCH = 0.25

# Complex, per [F1]. The imaginary part is not cosmetic: it moves the 1/k_y
# branch points off the integration contour. Never set this real.
DAMPING = 0.03
OM = 2 * np.pi * 6.0 * (1.0 + 1j * DAMPING)

# The converged rule, per [F2]. Flat in BOTH cutoff and spacing.
KR_MULT_CONVERGED = 30.0
N_AXIS_CONVERGED = 1536


def masked_rule(kr_max: float, n_per_axis: int) -> tuple[np.ndarray, list, list]:
    """Tensor product in (k_z, k_x), radially masked to |k_perp| <= kr_max.

    Args:
        kr_max: Radial cutoff, 1/km.
        n_per_axis: Nodes per axis across [-kr_max, kr_max].

    Returns:
        (kz_rows, kx_per_row, weight_per_row). For row j the nodes are
        kx_per_row[j] with weights weight_per_row[j]; weights include the
        1/(2 pi)^2.
    """
    edge = np.linspace(-kr_max, kr_max, n_per_axis + 1)
    ctr = 0.5 * (edge[:-1] + edge[1:])
    d = float(edge[1] - edge[0])
    w0 = d * d / (2 * np.pi) ** 2

    kz_rows, kx_rows, w_rows = [], [], []
    for kz in ctr:
        room = kr_max**2 - kz**2
        if room <= 0.0:
            continue
        keep = ctr[np.abs(ctr) <= np.sqrt(room)]
        if keep.size == 0:
            continue
        kz_rows.append(float(kz))
        kx_rows.append(keep)
        w_rows.append(np.full(keep.size, w0))
    return np.array(kz_rows), kx_rows, w_rows


def integrate(dy: float, kr_max: float, n_per_axis: int) -> tuple[np.ndarray, int, float]:
    """Integrate the post-k_y-residue kernel over the masked transverse plane."""
    kz_rows, kx_rows, w_rows = masked_rule(kr_max, n_per_axis)
    out = np.zeros((9, 9), dtype=complex)
    n_nodes = 0
    t0 = time.perf_counter()
    for kz, kx, w in zip(kz_rows, kx_rows, w_rows, strict=True):
        p = post_ky_residue_kernel_9x9_vec(
            kx, kz, abs(dy), omega=OM, rho=REF.rho, alpha=REF.alpha, beta=REF.beta
        )
        out += np.einsum("abk,k->ab", p, w)
        n_nodes += kx.size
    return out, n_nodes, time.perf_counter() - t0


def convergence_table(want: np.ndarray, scale: float) -> None:
    """Both constraints, scanned together -- see [F2]."""
    print(f"\n  {'kr*pitch':>9} {'n/axis':>7} {'dk':>8} {'nodes':>9} {'rel err':>11} {'sec':>6}")
    for mult in (10.0, 20.0, 30.0):
        for n_ax in (256, 512, 1024):
            got, n_nodes, dt = integrate(PITCH, mult / PITCH, n_ax)
            err = float(np.abs(got - want).max() / scale)
            dk = 2 * (mult / PITCH) / n_ax
            print(f"  {mult:9.0f} {n_ax:7d} {dk:8.4f} {n_nodes:9d} {err:11.3e} {dt:6.1f}")
        print()
    print("  Read DOWN a column for the cutoff and ACROSS a row for the spacing.")
    print("  A rule is converged only when flat in both; equal errors on the")
    print("  diagonal (same dk, different cutoff) are the tell that both bind.")


def cost_report(n_nodes: int) -> None:
    """[F3] -- why the spectral route was abandoned for Dz = 0."""
    print(f"\n  COST at the converged rule ({n_nodes:,} transverse nodes)")
    for n in (16, 32):
        acc = n * n * n_nodes * 9 * 16
        stack = n * n * 81 * n_nodes * 16
        dxy = (2 * n - 1) ** 2
        real_a = (n * (n - 1) // 2) * dxy * 81 * 16
        real_bc = dxy * 81 * 16
        print(f"    lattice {n}^3")
        print(f"      SPECTRAL sweep accumulator (n_z,n_x,n_k,9) : {acc / 1e9:9.1f} GB  x2 live")
        print(f"      SPECTRAL inter-plane stack (n_z,n_z,9,9,n_k): {stack / 1e9:9.1f} GB")
        print(f"      REAL-SPACE same-depth table                : {real_bc / 1e6:9.1f} MB")
        print(f"      REAL-SPACE inter-plane table               : {real_a / 1e6:9.1f} MB")
    print("\n  The real-space tables are the plan's architecture. The spectral")
    print("  figures are why: they are not a tuning problem, they are 3-4 orders.")


def main() -> int:
    # exact_propagator_9x9 takes CARTESIAN (x, y, z) in that order, while the
    # state vector is ordered (z, x, y). One pitch along y is (0, PITCH, 0);
    # reversing it gives a plausible wrong answer, not an error.
    want = exact_propagator_9x9(0.0, PITCH, 0.0, OM, REF)
    scale = float(np.abs(want).max())
    print("=" * 78)
    print("MEASUREMENT -- transverse quadrature for 3-D, and its cost")
    print(f"  background a={REF.alpha} b={REF.beta} rho={REF.rho}, pitch={PITCH} km")
    print(f"  omega = 2 pi * 6 Hz * (1 + {DAMPING}i)   <- COMPLEX, see [F1]")
    print(f"  kP = {abs(OM.real) / REF.alpha:.2f}, kS = {abs(OM.real) / REF.beta:.2f} rad/km")
    print("  arbiter: exact_propagator_9x9, closed form")
    print("=" * 78)

    convergence_table(want, scale)

    got, n_nodes, _ = integrate(PITCH, KR_MULT_CONVERGED / PITCH, N_AXIS_CONVERGED)
    err = float(np.abs(got - want).max() / scale)
    print(f"\n  CONVERGED RULE kr*pitch={KR_MULT_CONVERGED:.0f}, n/axis={N_AXIS_CONVERGED}")
    print(f"    nodes {n_nodes:,}   rel err {err:.3e}")

    cost_report(n_nodes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
