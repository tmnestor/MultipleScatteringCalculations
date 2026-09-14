"""GATE: the ADP's same-depth operator, resummed exactly for an unbounded medium.

WHAT THIS CLOSES. The alternating-direction propagator is verified as a
COMPOSITION -- `gate_rung5c_cross_architecture` reproduces `slab_scattering`'s
G0 to 5.1e-16 -- but that gate runs `periodic=False`, and
`pair_propagators.same_depth_table` is a finite pairwise table over
(2 n_x - 1, 2 n_y - 1) separations. So the ADP has only ever been checked
against a FINITE lattice. For the laterally unbounded medium, which is the
physical case and the thesis's case, its same-depth table carries the same
truncation that `build_slab_kernels` carried until it was fixed.

THE TWO HALVES ARE NOT IN THE SAME POSITION, and it is worth being precise:

  * the INTER-PLANE half needs no new evidence. `bloch_kernel_hat_9x9` builds
    dz != 0 from `sweep_kernels.vertical_kernel_9x9` -- the ADP's OWN vertical
    operator -- summed over reciprocal vectors, and that route is already exact
    and gated (`gate_interplane_bloch_sum`, 6.6e-7 against a direct sum). The
    ADP's vertical sweep is the object being summed, not an alternative to it.
  * the SAME-DEPTH half is where the truncation lives, and it is what this gate
    tests: the ADP's own closed-form whole-space propagator, Bloch-summed over
    the lattice, must equal the Ewald resummation used by the exact kernel.

If it does, the ADP's same-depth OPERATOR is correct and only its SUMMATION was
truncated -- a resummation, not a reformulation. If it does not, the two
constructions disagree about the operator itself and that is a far more serious
finding.

WHY THE MEDIUM IS DAMPED, AND HEAVILY. At dz = 0 the undamped lattice sum of the
strain block is not absolutely convergent -- that is the whole reason Ewald is
needed there -- so a direct sum cannot be formed to compare against. With
Im(kappa) > 0 it converges, and the Poisson/Ewald identity holds for ANY kappa,
so a damped medium is a complete test of the identity and need not be physical.

The damping is chosen so the arbiter converges INSIDE its radius, and the gate
measures that rather than assuming it. A first run at Q = 1 gave a decay length
of 17 m against a 50 m radius: the arbiter still drifted by 8.7e-3 between
successive radii, which is the same order as the disagreement it was being asked
to judge. Reading [A1] then would have been reading the arbiter's own tail. That
failure mode -- a slowly convergent reference convicting the thing it is
converging towards -- has already cost this project one wrong accusation, so the
self-convergence check in [A0] gates the rest of the gate.

Run:  conda run -n seismic python scripts/gate_adp_same_depth_lattice.py
SI units (m, m/s, kg/m3, Pa).
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from cubic_scattering.lattice_kupradze import bloch_block_ewald_9x9  # noqa: E402
from cubic_scattering.resonance_tmatrix import _propagator_block_9x9  # noqa: E402

REF = ReferenceMedium(5000.0, 3000.0, 2500.0)
D_PITCH = 0.5
# Heavy damping so the direct arbiter is converged well inside its radius:
# Im(k_P) = 0.36 /m is a 2.8 m decay length, so the 60 m radius below is ~21
# decay lengths and the omitted tail is ~1e-9. The identity under test holds for
# any kappa, so this costs nothing.
OMEGA = 600.0 * (1.0 + 3.0j)
RADII = (40, 80, 120)
K_PARS = (
    np.array([0.0, 0.0]),
    np.array([2.0 * np.pi / (4.0 * D_PITCH), 0.0]),
    np.array([2.0 * np.pi / (4.0 * D_PITCH), 2.0 * np.pi * 3.0 / (4.0 * D_PITCH)]),
)


def adp_direct_bloch(k_par: np.ndarray, n_big: int) -> np.ndarray:
    """Bloch sum of the ADP's own same-depth closed form, over R != 0.

    `pair_propagators.same_depth_table` fills its entries with exactly this
    object at exactly these separations; summing it over the lattice is what the
    ADP would have to do to represent an unbounded medium.

    The phase is exp(-i k.R), matching the FFT convention the periodic kernel is
    written in (see `bloch_kernel_hat_9x9`). A sign slip here agrees at
    k_par = 0 and fails everywhere else, which is why k_par = 0 is only ONE of
    the three cases below.
    """
    acc = np.zeros((9, 9), dtype=complex)
    for m in range(-n_big, n_big + 1):
        for n in range(-n_big, n_big + 1):
            if m == 0 and n == 0:
                continue
            rx, ry = m * D_PITCH, n * D_PITCH
            phase = np.exp(-1j * (k_par[0] * rx + k_par[1] * ry))
            acc += _propagator_block_9x9(np.array([0.0, rx, ry]), OMEGA, REF) * phase
    return acc


def main() -> int:
    print("=" * 88)
    print("GATE -- the ADP same-depth operator vs its Ewald resummation")
    print(f"  pitch = {D_PITCH} m, omega = {OMEGA} (damped so the direct sum converges)")
    print("=" * 88)

    # The arbiter must be shown converged BEFORE it is allowed to convict.
    print("\n  [A0] arbiter self-convergence at k_par = 0")
    prev = None
    arbiter_drift = 1.0
    for n_big in RADII:
        val = adp_direct_bloch(K_PARS[0], n_big)
        if prev is not None:
            arbiter_drift = float(np.abs(val - prev).max() / np.abs(val).max())
            print(
                f"       radius -> {n_big:4d} cells ({n_big * D_PITCH:.0f} m):  drift {arbiter_drift:.2e}"
            )
        prev = val

    print("\n  [A1] ADP closed form, Bloch-summed, vs the Ewald resummation")
    print(f"       {'k_par (1/m)':>26} {'|K|':>13} {'rel diff':>11}")
    worst = 0.0
    for k_par in K_PARS:
        direct = adp_direct_bloch(k_par, RADII[-1])
        ewald = bloch_block_ewald_9x9(k_par, 0.0, D_PITCH, OMEGA, REF, cutoff=6)
        rel = float(np.abs(direct - ewald).max() / np.abs(direct).max())
        worst = max(worst, rel)
        label = f"({k_par[0]:.3f}, {k_par[1]:.3f})"
        print(f"       {label:>26} {np.abs(direct).max():13.5e} {rel:11.2e}")

    # How badly does the TRUNCATED version do? This is the size of the defect
    # the ADP still carries, expressed in the ADP's own terms.
    print("\n  [A2] what truncation costs the ADP, at its own table sizes")
    print(f"       {'table half-width':>18} {'rel error vs exact':>20}")
    exact = bloch_block_ewald_9x9(K_PARS[1], 0.0, D_PITCH, OMEGA, REF, cutoff=6)
    for n_x in (2, 4, 8, 16):
        trunc = adp_direct_bloch(K_PARS[1], n_x - 1)
        rel = float(np.abs(trunc - exact).max() / np.abs(exact).max())
        print(f"       {n_x - 1:18d} {rel:20.3e}")

    print("\n" + "=" * 88)
    ok = worst < 1e-6 and arbiter_drift < 1e-6
    print(f"  [A0] arbiter drift            : {arbiter_drift:.2e}")
    print(f"  [A1] ADP sum vs Ewald         : {worst:.2e}")
    if ok:
        print("\n  PASS: the ADP's same-depth OPERATOR is correct. Its closed form,")
        print("  summed over the lattice, is exactly what the Ewald route resums --")
        print("  two independent constructions of the same object, agreeing at three")
        print("  Bloch vectors including two away from k_par = 0 where a phase-sign")
        print("  slip would show.")
        print()
        print("  So what the ADP needs for an unbounded medium is a RESUMMATION, not")
        print("  a reformulation. The composition is already verified at 5.1e-16")
        print("  (gate_rung5c) and the inter-plane half is already exact by")
        print("  reciprocal summation of the ADP's own vertical kernel. [A2] shows")
        print("  the size of what truncation costs at the table widths actually used.")
    else:
        print("\n  FAIL:", end=" ")
        if arbiter_drift >= 1e-6:
            print("arbiter not converged -- fix it before reading [A1].", end=" ")
        elif worst >= 1e-6:
            print("the two constructions disagree about the OPERATOR, not just its", end=" ")
            print("summation. Check the phase sign and the R = 0 exclusion first.", end="")
        print()
    print("=" * 88)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
