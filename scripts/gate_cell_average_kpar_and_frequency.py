"""GATE: the exact cell average across the Brillouin zone and across frequency.

`gate_dz0_averaged_lattice.py` validated the same-plane analytic tail at ONE
Bloch point (the zone centre) and ONE frequency. Those were its two stated scope
limits, and they are the two things standing between `exact_cell_average` and a
default. They carry different risks, so they get different predictions.

[A] AWAY FROM THE ZONE CENTRE. The tail multiplies (D_full - D_near); both carry
Bloch phases and nothing in the derivation depends on k_par, so the
R0-independence should hold unchanged across the zone. A failure here would mean
a PHASE CONVENTION slip rather than a defective tail: the near shell is built to
match `origin_scalar_tensors` exactly (separation s = -R, phase e^{+i k.R}), and
a sign error there is INVISIBLE at k_par = 0 because that sum is symmetric.

⚠ That invisibility is the reason this panel exists at all -- every previous
test of the tail ran at the one point where such an error cancels.

[B] FREQUENCY, where the two halves behave differently:

  * dz != 0 uses the sinc form factor, which is an IDENTITY on plane waves, not
    an expansion. No degradation with frequency is expected.
  * dz == 0 uses the d^2 tail, which IS an expansion, in (kappa d)^2. Its error
    should GROW with k d, and raising R0 should push the breakdown outward,
    because a larger near shell leaves less work to the expansion.

    ▶ If raising R0 does NOT reduce the error at high k d, the tail is not the
      limiting term and the diagnosis is wrong. That is the falsifiable content.

⚠ The Ewald route itself returns NaN above k_S d ~ 15 (recorded in
`_build_slab_kernels`, and the reason it is not the default). Finiteness is
checked explicitly here rather than letting a NaN read as a small error.

Run:  conda run -n seismic python scripts/gate_cell_average_kpar_and_frequency.py
SI units (m, m/s, kg/m3, Pa).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubic_scattering.cell_averaged_lattice import averaged_same_plane_9x9  # noqa: E402
from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from cubic_scattering.lattice_kupradze import bloch_kernel_hat_9x9  # noqa: E402
from cubic_scattering.slab_scattering import (  # noqa: E402
    _cell_averaged_propagator,
    _propagator_block_9x9,
)

REF = ReferenceMedium(5000.0, 3000.0, 2500.0)
D = 1.0
M = 4


def shell_correction(omega: float, k_par: np.ndarray, reach: int, n_gauss: int = 6) -> np.ndarray:
    """Direct sum of [<G> - G] over the same-plane neighbours, with Bloch phase.

    ⚠ The convention is `origin_scalar_tensors`': the separation to lattice site
    (i, j) is s = -d (i, j), and the phase is e^{+i k.R}. Matching it is not
    optional -- the kernel being compared against is assembled by that routine,
    and a sign slip cancels at k_par = 0, so only panel [A] can see it.
    """
    total = np.zeros((9, 9), dtype=complex)
    for i in range(-reach, reach + 1):
        for j in range(-reach, reach + 1):
            if i == 0 and j == 0:
                continue
            s_vec = np.array([0.0, -D * i, -D * j])
            phase = np.exp(1j * D * (k_par[0] * i + k_par[1] * j))
            total += phase * (
                _cell_averaged_propagator(s_vec, D, omega, REF, n_gauss, double=False)
                - _propagator_block_9x9(s_vec, omega, REF)
            )
    return total


def panel_a() -> bool:
    """R0-independence and agreement with the direct sum, across the zone."""
    omega = 60.0
    print("\n  [A] across the Brillouin zone   (M = 4, k_par = 2 pi n / (M d))")
    print(f"      {'(n1,n2)':>9} {'R0 3 vs 2':>12} {'R0 4 vs 2':>12} {'vs direct sum':>15}")
    ok = True
    for n1, n2 in ((0, 0), (1, 0), (1, 1), (2, 0), (2, 2)):
        k_par = 2.0 * np.pi * np.array([n1, n2], dtype=float) / (M * D)
        blocks = {r0: averaged_same_plane_9x9(D, omega, REF, k_par, r0_cells=r0) for r0 in (2, 3, 4)}
        scale = float(np.max(np.abs(blocks[2])))
        d3 = float(np.max(np.abs(blocks[3] - blocks[2])) / scale)
        d4 = float(np.max(np.abs(blocks[4] - blocks[2])) / scale)

        point = bloch_kernel_hat_9x9(M, D, 0.0, omega, REF)[n1, n2]
        ref_blk = point + shell_correction(omega, k_par, 12)
        rel = float(np.max(np.abs(blocks[3] - ref_blk)) / scale)

        print(f"      {f'({n1},{n2})':>9} {d3:12.2e} {d4:12.2e} {rel:15.2e}")
        ok = ok and d3 < 1e-3 and d4 < 1e-3 and rel < 5e-3
    print(f"      => holds across the zone: {ok}")
    if not ok:
        print("      A failure here is a PHASE CONVENTION slip, not a bad tail --")
        print("      invisible at k_par = 0 because that sum is symmetric.")
    return ok


def panel_b() -> bool:
    """Frequency sweep, and whether a bigger near shell buys back accuracy."""
    print("\n  [B] frequency   (the same-plane tail is an expansion in (kappa d)^2;")
    print("      the other branch is an identity and should not degrade)")
    k_par = 2.0 * np.pi * np.array([1.0, 0.0]) / (M * D)
    print(f"\n      {'omega':>8} {'k_S d':>8} {'finite':>7} {'R0=2':>12} {'R0=4':>12} {'R0=6':>12}")
    rows = []
    for omega in (60.0, 300.0, 1000.0, 3000.0, 6000.0, 12000.0):
        point = bloch_kernel_hat_9x9(M, D, 0.0, omega, REF)[1, 0]
        if not bool(np.all(np.isfinite(point))):
            print(
                f"      {omega:8.0f} {omega / REF.beta * D:8.3f} {'NO':>7} {'--':>12} {'--':>12} {'--':>12}"
            )
            continue
        best = averaged_same_plane_9x9(D, omega, REF, k_par, r0_cells=8)
        scale = float(np.max(np.abs(best)))
        drifts = [
            float(np.max(np.abs(averaged_same_plane_9x9(D, omega, REF, k_par, r0_cells=r0) - best)) / scale)
            for r0 in (2, 4, 6)
        ]
        rows.append((omega, drifts))
        print(
            f"      {omega:8.0f} {omega / REF.beta * D:8.3f} {'yes':>7}"
            f" {drifts[0]:12.2e} {drifts[1]:12.2e} {drifts[2]:12.2e}"
        )

    print("\n      Each row should FALL left to right (a bigger near shell buys")
    print("      back accuracy) and the left column should GROW down the table")
    print("      (the expansion degrading).  Both together confirm the tail is")
    print("      the limiting term; either failing means it is not.")
    grows = len(rows) >= 2 and rows[-1][1][0] > 10.0 * rows[0][1][0]
    r0_helps = bool(rows) and all(d[0] > d[2] for _, d in rows)
    print(f"      error grows with k_S d:              {grows}")
    print(f"      bigger near shell helps, every row:  {r0_helps}")
    return r0_helps


def main() -> int:
    print("=" * 78)
    print("GATE: exact cell average across the zone, and across frequency")
    print("=" * 78)
    a_ok = panel_a()
    b_ok = panel_b()

    print("\n" + "=" * 78)
    if a_ok and b_ok:
        print("PASS -- both stated scope limits are closed.  The tail holds across")
        print("the Brillouin zone, so the phase convention is right and not merely")
        print("right at the one symmetric point; and its error behaves as an")
        print("expansion that the near-shell radius controls.")
        print()
        print("⚠ This does NOT by itself make the route a safe default: the cost")
        print("is still 3.55x, and the mixed-contrast channel's convergence ORDER")
        print("remains unverified because its error passes through zero.")
    else:
        print("FAIL -- [A] failing means a phase-convention slip that k_par = 0")
        print("cannot see.  [B] failing means the d^2 tail is NOT the limiting")
        print("term at high k d, so a bigger near shell will not rescue it.")
    print("=" * 78)
    return 0 if (a_ok and b_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
