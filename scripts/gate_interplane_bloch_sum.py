"""GATE: the inter-plane Bloch lattice sum by reciprocal summation.

THE SPLIT THAT MAKES THIS TRACTABLE. The periodic lateral sum needs, for each
Bloch vector, sum_R G(dz, r - R) e^{i k.R} over the full lattice. By Poisson
summation that equals a RECIPROCAL sum,

    K(k_par; dz) = (1/d^2) sum_G  Ghat(k_par + G, dz)

with Ghat the 2-D lateral Fourier transform of the propagator -- which this
project already has, validated, as `sweep_kernels.vertical_kernel_9x9`. No
derivatives of scalar Ewald fields are needed; the spectral 9x9 is the object.

AND THE TWO CASES ARE NOT ALIKE, by the modules' own account:

  * dz != 0 -- `vertical_kernel_9x9` "keeps its e^{-kappa |dz|} convergence
    factor", so the reciprocal sum converges EXPONENTIALLY in |G|. That is this
    gate. A handful of reciprocal vectors suffices and the result is exact.
  * dz  = 0 -- `same_depth_kernel_9x9`'s "strain-strain block grows like |k_x|",
    so the reciprocal sum DIVERGES. That case needs the Ewald split in
    `planar_ewald` and is not attempted here.

So this closes the inter-plane half of the lateral-sum defect exactly, and
leaves the same-plane half, which is the genuinely hard one.

THE ARBITER IS A DIRECT REAL-SPACE SUM, sharing no code path: sum_R of the
closed-form `_propagator_block_9x9` with the Bloch phase. It converges only in a
damped medium, so the gate runs at finite Q -- which is available precisely
because the complex-medium fix landed earlier (12791fc). Real Earth has Q
anyway, so this is the physical case, not a contrivance.

Run:  conda run -n seismic python scripts/gate_interplane_bloch_sum.py
SI units (m, m/s, kg/m3, Pa).
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from cubic_scattering.resonance_tmatrix import _propagator_block_9x9  # noqa: E402
from cubic_scattering.sweep_kernels import vertical_kernel_9x9  # noqa: E402

REF = ReferenceMedium(5000.0, 3000.0, 2500.0)
D_PITCH = 2.0
# Damping chosen SO THE ARBITER CONVERGES, not for physical realism. The Poisson
# identity holds for any kappa, so the arbiter is free to use a medium no Earth
# would have. Q = 1 gives Im(k_P) = 0.06 /m, a 17 m decay length, so a 100-cell
# radius is ~12 decay lengths and the direct tail is ~1e-5.
#
# The first attempt used Q = 20, where the decay length is 333 m against a 120 m
# summation radius: the arbiter was nowhere near converged and "disagreed" by
# 20-77%. Its gap to the reciprocal sum then HALVED with every doubling of the
# radius -- converging TO it -- which is what exposed the arbiter rather than the
# sum. A slowly convergent reference must be shown converged before it can
# convict anything.
Q_FACTOR = 1.0
OMEGA = 600.0 * (1.0 + 0.5j / Q_FACTOR)


def bloch_reciprocal(k_par: np.ndarray, dz: float, n_g: int) -> np.ndarray:
    """(1/d^2) sum_G Ghat(k_par + G, dz) -- exponentially convergent for dz != 0."""
    b = 2.0 * np.pi / D_PITCH
    acc = np.zeros((9, 9), dtype=complex)
    for m in range(-n_g, n_g + 1):
        for n in range(-n_g, n_g + 1):
            kx = k_par[0] + b * m
            ky = k_par[1] + b * n
            acc += vertical_kernel_9x9(np.array([kx]), ky, dz, OMEGA, REF)[:, :, 0]
    return acc / D_PITCH**2


def bloch_direct(k_par: np.ndarray, dz: float, n_big: int) -> np.ndarray:
    """sum_R G(dz, R) e^{-i k.R} by brute force -- the independent arbiter.

    THE PHASE SIGN IS NOT FREE. Poisson gives
    sum_R f(r - R) e^{i k.R} = (1/A) sum_G fhat(k + G) e^{i(k+G).r}; at r = 0 the
    left side is sum_R f(-R) e^{i k.R}, and substituting R -> -R makes it
    sum_R f(R) e^{-i k.R}. Writing e^{+i k.R} here instead agrees at k_par = 0 --
    where the phase is 1 either way -- and fails everywhere else, which is
    exactly how this was caught.
    """
    acc = np.zeros((9, 9), dtype=complex)
    for m in range(-n_big, n_big + 1):
        for n in range(-n_big, n_big + 1):
            rx, ry = m * D_PITCH, n * D_PITCH
            phase = np.exp(-1j * (k_par[0] * rx + k_par[1] * ry))
            acc += _propagator_block_9x9(np.array([dz, rx, ry]), OMEGA, REF) * phase
    return acc


def main() -> int:
    print("=" * 84)
    print("GATE -- inter-plane Bloch lattice sum by reciprocal summation")
    print(f"  pitch = {D_PITCH} m, Q = {Q_FACTOR:.0f} (so the direct arbiter converges)")
    print("=" * 84)

    cases = [
        (np.array([0.0, 0.0]), 2.0),
        (np.array([0.11, 0.07]), 2.0),
        (np.array([0.11, 0.07]), 6.0),
    ]

    print(f"\n  [B1] reciprocal vs direct   {'k_par':>16} {'dz':>5} {'|K|':>12} {'rel':>10}")
    worst = 0.0
    for k_par, dz in cases:
        rec = bloch_reciprocal(k_par, dz, n_g=6)
        dirs = bloch_direct(k_par, dz, n_big=100)
        rel = np.abs(rec - dirs).max() / np.abs(dirs).max()
        worst = max(worst, rel)
        print(f"       {'':>22} {str(tuple(k_par)):>16} {dz:5.1f} {np.abs(rec).max():12.4e} {rel:10.2e}")
    # The arbiter must be SHOWN converged, not assumed: halving its radius must
    # not move it more than the agreement being claimed.
    half = bloch_direct(cases[0][0], cases[0][1], n_big=70)
    full = bloch_direct(cases[0][0], cases[0][1], n_big=100)
    drift = np.abs(full - half).max() / np.abs(full).max()
    print(f"\n       arbiter self-convergence (radius 70 -> 100): {drift:.2e}")

    # Exponential convergence is the claim; measure it rather than assert it.
    print("\n  [B2] convergence in the number of reciprocal vectors (dz = 2 m)")
    ref_val = bloch_reciprocal(np.array([0.11, 0.07]), 2.0, n_g=10)
    prev = None
    for n_g in (1, 2, 3, 4, 6):
        val = bloch_reciprocal(np.array([0.11, 0.07]), 2.0, n_g=n_g)
        rel = np.abs(val - ref_val).max() / np.abs(ref_val).max()
        gain = "" if prev is None else f"  x{prev / max(rel, 1e-300):8.1f}"
        print(f"       n_G = {n_g:2d}   rel to n_G = 10: {rel:10.3e}{gain}")
        prev = rel

    print("\n" + "=" * 84)
    if worst < 1e-4:
        print("  PASS: the reciprocal sum reproduces the direct lattice sum. The")
        print("  inter-plane half of the periodic lateral sum is now exact and")
        print("  cheap -- a handful of reciprocal vectors, no truncation artifact.")
        print("  What remains is the SAME-PLANE term, where the spectral kernel")
        print("  grows like |k| and the Ewald split of planar_ewald is required.")
    else:
        print(f"  FAIL: reciprocal and direct sums disagree by {worst:.2e}.")
        print("  Check the Poisson convention -- the 1/d^2 area factor and the")
        print("  sign of the Bloch phase are the two places this goes wrong.")
    print("=" * 84)
    return 0 if worst < 1e-4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
