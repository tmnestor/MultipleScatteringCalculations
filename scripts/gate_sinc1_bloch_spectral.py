"""GATE: the sinc^1 form factor in the Bloch kernel, against the shell sum.

THE BUILD THIS GATES. `gate_sinc1_vs_gauss_9x9.py` verified the single-average
form factor at one offset. This puts it where it earns its keep: inside the
LATTICE kernel, as a multiplication on each plane-wave component, so the
source-cell average needs no real-space correction shell at all.

    <G>(R) = Int_cell G(R - u) du / V   ->   f0(k) Ghat(k),
    f0(k) = sinc(k_x h) sinc(k_y h) sinc(k_z h)      (SINGLE -- one power)

WHY THIS MATTERS: the correction it replaces has an O(1/R) tail and turns
SHAPE-DEPENDENT past k r ~ 1, so it cannot be converged by enlarging the box
(`scripts/investigate_correction_tail_shape.py`). A spectral multiplication has
no box to enlarge.

⚠ SCOPE, AND IT IS HALF THE PROBLEM. This covers dz != 0 ONLY, where the Bloch
kernel is a plane-wave sum. At dz = 0 the route is EWALD, whose real-space half
is not a plane-wave sum, so the form factor is not a multiplication there -- and
the plain reciprocal sum cannot be substituted, because with the SINGLE form
factor its terms decay as 1/|G|^2 against ~|G| growth and ~|G| states per shell,
which is log-divergent. (The DOUBLE form factor gains another 1/|G|^2 and does
converge; that is why the sinc^2 route exists.) `bloch_kernel_hat_9x9` now
REFUSES cell_half_width at dz = 0 rather than silently returning a wrong number.
The same-plane case still needs the contact correction, and that is the
remaining piece.

THE REFERENCE. At dz != 0 the existing real-space route -- point kernel plus a
shell sum of [<G> - G] -- converges EXPONENTIALLY (the interplane sum keeps its
exp(-kappa|dz|)), so it is trustworthy there and is the right thing to check
against. That is exactly the regime where the shell sum is NOT the problem,
which is what makes it a valid arbiter for a method aimed at the regime where it
is.

TWO TRAPS ALREADY PAID FOR, both live in this code path:
  * the form factor must be applied PER MODE -- P and S carry different k_z, so
    one scalar on the assembled 9x9 is wrong;
  * it HALVES the spectral decay rate, because sinc(i kappa h) =
    sinh(kappa h)/(kappa h) grows like e^{kappa h} against the kernel's
    e^{-kappa|dz|}. A cutoff tuned for the point kernel returns ~1e-8 where it
    used to return 1e-16, silently. `_spectral_bloch_block` doubles its floor.

Run:  conda run -n seismic python scripts/gate_sinc1_bloch_spectral.py
SI units (m, m/s, kg/m3, Pa).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from cubic_scattering.lattice_kupradze import bloch_kernel_hat_9x9  # noqa: E402
from cubic_scattering.slab_scattering import (  # noqa: E402
    _cell_averaged_propagator,
    _propagator_block_9x9,
)

REF = ReferenceMedium(5000.0, 3000.0, 2500.0)
D = 1.0
H = 0.5 * D
OMEGA = 60.0
M = 3


def shell_sum_reference(dz_vox: int, reach: int, n_gauss: int = 8) -> np.ndarray:
    """Bloch sum of [<G> - G] over the plane at this dz, at k_par = 0.

    At k_par = 0 every Bloch phase is 1, so this is the raw shell sum -- the
    quantity the spectral form factor is meant to make unnecessary.
    """
    total = np.zeros((9, 9), dtype=complex)
    dz = dz_vox * D
    for dx in range(-reach, reach + 1):
        for dy in range(-reach, reach + 1):
            r_vec = np.array([dz, dx * D, dy * D])
            total += _cell_averaged_propagator(
                r_vec, D, OMEGA, REF, n_gauss, double=False
            ) - _propagator_block_9x9(r_vec, OMEGA, REF)
    return total


def main() -> int:
    print("=" * 78)
    print("GATE: sinc^1 in the Bloch kernel vs the real-space shell sum")
    print("=" * 78)
    print(f"\n  d = {D}, h = {H}, omega = {OMEGA}, M = {M}, k_par = 0")

    # ---- [0] the default path must be untouched --------------------------
    a = bloch_kernel_hat_9x9(M, D, D, OMEGA, REF)
    b = bloch_kernel_hat_9x9(M, D, D, OMEGA, REF, cell_half_width=None)
    same = float(np.max(np.abs(a - b)))
    print(f"\n  [0] default path unchanged (cell_half_width=None): {same:.2e}")
    if same != 0.0:
        print("      FAIL -- the additive parameter changed the default path.")
        return 1

    # ---- [1] dz = 0 must be REFUSED, not silently wrong -------------------
    print("\n  [1] dz = 0 must refuse, since Ewald cannot take a form factor")
    try:
        bloch_kernel_hat_9x9(M, D, 0.0, OMEGA, REF, cell_half_width=H)
    except ValueError as exc:
        txt = str(exc)
        ok = "Where:" in txt and "Fix:" in txt and "log-divergent" in txt
        print(f"      refused with a diagnostic naming where/why/fix: {ok}")
        if not ok:
            return 1
    else:
        print("      FAIL -- dz = 0 accepted a form factor it cannot apply.")
        return 1

    # ---- [2] the measurement -------------------------------------------
    print("\n  [2] averaged spectral kernel vs point + shell sum, at k_par = 0")
    print(f"      {'dz':>5} {'reach':>7} {'|spectral - (pt + shell)|':>27} {'rel':>11}")
    verdicts = []
    for dz_vox in (1, 2):
        pt = bloch_kernel_hat_9x9(M, D, dz_vox * D, OMEGA, REF)[0, 0]
        avg = bloch_kernel_hat_9x9(M, D, dz_vox * D, OMEGA, REF, cell_half_width=H)[0, 0]
        prev = None
        for reach in (4, 8, 12):
            ref_block = pt + shell_sum_reference(dz_vox, reach)
            diff = float(np.max(np.abs(avg - ref_block)))
            rel = diff / float(np.max(np.abs(avg)))
            print(f"      {dz_vox:>5} {reach:>7} {diff:27.6e} {rel:11.2e}")
            prev = rel
        verdicts.append(prev is not None and prev < 1e-3)

    ok = all(verdicts)
    print("\n" + "=" * 78)
    if ok:
        print("PASS -- the spectral form factor reproduces the real-space shell")
        print("sum at dz != 0, to better than 1e-3 relative once the shell has")
        print("converged.  So for every non-equal-depth plane the source-cell")
        print("average is now EXACT and needs no correction shell, no reach")
        print("parameter, and no box to enlarge.")
        print()
        print("REMAINING: dz = 0.  That is where the O(1/R) tail and the")
        print("shape-dependence live, and it is the Ewald branch, so it needs")
        print("an analytic tail rather than this multiplication.")
    else:
        print("FAIL -- the spectral form factor does not reproduce the shell")
        print("sum at dz != 0.  Since the shell sum converges exponentially")
        print("there, the discrepancy is in the form-factor implementation --")
        print("check the PER-MODE application and the raised cutoff first.")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
