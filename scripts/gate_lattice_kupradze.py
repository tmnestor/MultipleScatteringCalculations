"""GATE: Bloch-summed derivative tensors to fourth order, and the 9x9 from them.

THIS IS THE SECOND HALF OF THE LATERAL-SUM FIX. The first half -- the derivative
ladder itself -- is cleared by `gate_kupradze_derivatives` against the validated
point propagator, with no summation anywhere in it. Here the summation is added
and nothing else, so a failure here is the lattice sum, not the algebra.

WHAT IS CHECKED, in increasing order of how much they can convict:

  [L1] the ladder bridge. `ladder_from_plain` converts plain radial derivatives
       into (1/d d/dd)^k by an integer recursion. Applied to the Helmholtz
       kernel it must reproduce the closed-form Hankel ladder. This is the one
       piece shared by the screened and unscreened kernels, so it is checked on
       the kernel where an independent closed form exists.

  [L2] order 0 against `planar_ewald.ewald_total`. The n = 0 tensor IS the
       scalar lattice sum, already gated to 1.8e-15 at z = 0 and 8.7e-16 at
       z != 0. Any disagreement is the COORDINATE PERMUTATION between
       planar_ewald's (x, y, z) and this module's (z, x, y) -- which is the
       shape of a defect already on this project's record.

  [L3] ETA-INDEPENDENCE AT EVERY DERIVATIVE ORDER. The load-bearing test. eta
       only moves work between the two halves, so the total cannot depend on it
       -- and a wrong fourth-derivative recursion breaks eta-independence at
       order 4 while leaving the scalar perfect. Run at z != 0, because the
       reciprocal erfc pairing is invisible at z = 0, and at k_par != 0, because
       a wrong Bloch phase sign agrees exactly at k_par = 0.

  [L4] against a damped direct sum of the derivative tensors. Independent
       construction, no shared code path. Shown converged in its own radius
       first -- an unconverged arbiter has already cost this project a wrong
       accusation once.

  [L5] the z-parity invariant at the same-plane point. Every lattice vector is
       in-plane, so the summed scalar depends on z only through z^2; at z = 0
       every tensor with an ODD number of z indices must vanish identically.
       Nothing in the construction enforces this, and it is the only check here
       that is specific to the same-plane case the whole fix exists for.

Run:  conda run -n seismic python scripts/gate_lattice_kupradze.py
"""

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from cubic_scattering.kupradze_derivatives import (  # noqa: E402
    radial_ladder,
    scalar_derivative_tensors,
)
from cubic_scattering.lattice_kupradze import (  # noqa: E402
    direct_scalar_tensors,
    ladder_from_plain,
    lattice_block_9x9,
    lattice_scalar_tensors,
)
from cubic_scattering.planar_ewald import ewald_total  # noqa: E402

A_L = 2.0
K_PAR = np.array([0.2, 0.1])
ETA1, ETA2 = 0.7, 1.15
RC, GC = 8, 8
KAPPA_REAL = 1.5
KAPPA_DAMPED = 1.5 + 0.25j
# alpha/beta/rho chosen so kP = 0.9, kS = 1.5 at omega = 4.5 -- the same
# wavenumbers the Mathematica vector-lattice work uses, against a_L = 2.
REF = ReferenceMedium(5.0, 3.0, 2.5)
OMEGA = 4.5

# (z, x, y). Off-plane points for the pairing-sensitive tests; in-plane for [L5].
PTS_OFFPLANE = [
    np.array([0.31, 0.5, 0.0]),
    np.array([-0.24, 0.35, 0.28]),
    np.array([0.42, -0.6, 0.7]),
]
PTS_INPLANE = [
    np.array([0.0, 0.5, 0.0]),
    np.array([0.0, 0.35, 0.28]),
    np.array([0.0, -0.6, 0.7]),
]


def _to_xyz(r_zxy: np.ndarray) -> np.ndarray:
    """(z, x, y) -> (x, y, z), the ordering `planar_ewald` was ported in."""
    return np.array([r_zxy[1], r_zxy[2], r_zxy[0]])


def main() -> int:
    print("=" * 84)
    print("GATE -- Bloch-summed Kupradze derivative tensors and the same-plane 9x9")
    print(f"  a_L = {A_L}, k_par = {tuple(K_PAR)}, eta = {ETA1} vs {ETA2}, Rc = Gc = {RC}")
    print("=" * 84)

    # ---- [L1] the ladder bridge, against exact closed forms -----------------
    # Two independent exact checks, no finite differences: a monomial whose
    # ladder terminates, and the Helmholtz kernel, whose PLAIN derivatives have
    # a closed form (Leibniz on exp(i kappa d) times 1/d) entirely separate from
    # the Hankel recursion `radial_ladder` uses. Both are machine-precision.
    print("\n  [L1] ladder_from_plain vs exact closed forms")
    d, kap = 1.37, KAPPA_REAL

    # (a) f = d^6: (1/d d/dd)^k gives 6d^4, 24d^2, 48, 0.
    mono_plain = [d**6, 6 * d**5, 30 * d**4, 120 * d**3, 360 * d**2]
    mono_exact = [d**6, 6 * d**4, 24 * d**2, 48.0, 0.0]
    got = ladder_from_plain([complex(v) for v in mono_plain], d)
    l1 = max(abs(got[k] - mono_exact[k]) / max(abs(mono_exact[k]), 1.0) for k in range(5))
    print(f"       (a) monomial d^6, worst rel: {l1:.2e}")

    # (b) the Helmholtz kernel, plain derivatives in closed form.
    plain = [
        complex(
            sum(
                math.comb(j, m)
                * (1j * kap) ** (j - m)
                * np.exp(1j * kap * d)
                * (-1) ** m
                * math.factorial(m)
                / d ** (m + 1)
                for m in range(j + 1)
            )
            / (4.0 * np.pi)
        )
        for j in range(5)
    ]
    bridged = ladder_from_plain(plain, d)
    closed = radial_ladder(d, kap, 4)
    for k in range(5):
        rel = abs(bridged[k] - closed[k]) / abs(closed[k])
        l1 = max(l1, rel)
        print(f"       (b) k = {k}   Hankel {abs(closed[k]):12.5e}   rel {rel:9.2e}")

    # ---- [L2] order 0 against the gated scalar Ewald ------------------------
    print("\n  [L2] order-0 tensor vs planar_ewald.ewald_total (coordinate check)")
    print(f"       {'(z, x, y)':>22} {'|D0|':>13} {'rel':>11}")
    l2 = 0.0
    for r in PTS_OFFPLANE + PTS_INPLANE:
        mine = lattice_scalar_tensors(r, KAPPA_REAL, ETA1, RC, GC, A_L, K_PAR, order=0)[0]
        theirs = ewald_total(KAPPA_REAL, _to_xyz(r), ETA1, RC, GC, A_L, K_PAR)
        rel = abs(complex(mine) - theirs) / abs(theirs)
        l2 = max(l2, rel)
        label = f"({r[0]:.2f}, {r[1]:.2f}, {r[2]:.2f})"
        print(f"       {label:>22} {abs(complex(mine)):13.5e} {rel:11.2e}")

    # ---- [L3] eta-independence at every order -------------------------------
    print("\n  [L3] eta-independence by derivative order (z != 0, k_par != 0)")
    print(f"       {'order':>6} {'|D|':>13} {'rel diff':>11}")
    l3 = 0.0
    per_order = []
    for order in range(5):
        worst = 0.0
        mag = 0.0
        for r in PTS_OFFPLANE:
            a = lattice_scalar_tensors(r, KAPPA_REAL, ETA1, RC, GC, A_L, K_PAR, order)[order]
            b = lattice_scalar_tensors(r, KAPPA_REAL, ETA2, RC, GC, A_L, K_PAR, order)[order]
            scale = float(np.abs(a).max())
            worst = max(worst, float(np.abs(a - b).max()) / scale)
            mag = max(mag, scale)
        per_order.append(worst)
        l3 = max(l3, worst)
        print(f"       {order:6d} {mag:13.5e} {worst:11.2e}")

    # ---- [L4] against a converged damped direct sum --------------------------
    print(f"\n  [L4] damped kappa = {KAPPA_DAMPED}: Ewald vs direct tensor sum")
    r0 = PTS_OFFPLANE[0]
    near = direct_scalar_tensors(r0, KAPPA_DAMPED, 40, A_L, K_PAR)
    far = direct_scalar_tensors(r0, KAPPA_DAMPED, 60, A_L, K_PAR)
    arb = max(float(np.abs(far[n] - near[n]).max()) / float(np.abs(far[n]).max()) for n in range(5))
    print(f"       arbiter self-convergence (radius 40 -> 60): {arb:.2e}")
    print(f"       {'order':>6} {'rel diff':>11}")
    l4 = 0.0
    for order in range(5):
        e = lattice_scalar_tensors(r0, KAPPA_DAMPED, ETA1, RC, GC, A_L, K_PAR, order)[order]
        rel = float(np.abs(e - far[order]).max()) / float(np.abs(far[order]).max())
        l4 = max(l4, rel)
        print(f"       {order:6d} {rel:11.2e}")

    # ---- [L5] the z-parity invariant at the same-plane point -----------------
    print("\n  [L5] z-parity at z = 0: odd numbers of z indices must vanish")
    l5 = 0.0
    for r in PTS_INPLANE:
        tens = lattice_scalar_tensors(r, KAPPA_REAL, ETA1, RC, GC, A_L, K_PAR)
        for order in range(1, 5):
            t = tens[order]
            scale = float(np.abs(t).max())
            for idx in np.ndindex(*((3,) * order)):
                if idx.count(0) % 2 == 1:
                    l5 = max(l5, abs(complex(t[idx])) / scale)
    print(f"       worst odd-z entry, relative to the tensor scale: {l5:.2e}")

    # ---- the assembled 9x9, same plane --------------------------------------
    print("\n  [--] the same-plane 9x9, for scale (not a pass/fail test)")
    blk = lattice_block_9x9(PTS_INPLANE[1], OMEGA, REF, ETA1, RC, GC, A_L, K_PAR)
    free = scalar_derivative_tensors(PTS_INPLANE[1], OMEGA / REF.beta)[0]
    print(
        f"       |lattice 9x9|_max = {np.abs(blk).max():.5e}   (free-space scalar {abs(complex(free)):.3e})"
    )
    finite = bool(np.all(np.isfinite(blk)))

    print("\n" + "=" * 84)
    print(f"  [L1] ladder bridge            : {l1:.2e}")
    print(f"  [L2] vs gated scalar Ewald    : {l2:.2e}")
    print(f"  [L3] eta-independence, order 4: {per_order[4]:.2e}  (worst {l3:.2e})")
    print(f"  [L4] vs converged direct sum  : {l4:.2e}  (arbiter {arb:.2e})")
    print(f"  [L5] z-parity at z = 0        : {l5:.2e}")
    ok = l1 < 1e-12 and l2 < 1e-12 and l3 < 1e-9 and l4 < 1e-6 and arb < 1e-4 and l5 < 1e-12 and finite
    if ok:
        print("\n  PASS: the Bloch lattice sum is exact to fourth order. The scalar")
        print("  foundation, the derivative ladder and the summation now all carry")
        print("  their own evidence, so the same-plane 9x9 can be wired into")
        print("  build_slab_kernels -- whose acceptance test is the M-invariance of")
        print("  a laterally uniform medium, scripts/gate_lateral_sum_invariance.py.")
    else:
        print("\n  FAIL:", end=" ")
        if l1 >= 1e-12:
            print("ladder bridge wrong.", end=" ")
        if l2 >= 1e-12:
            print("order 0 disagrees with planar_ewald -- suspect the (z,x,y) order.", end=" ")
        if l3 >= 1e-9:
            print("eta-dependent -- a derivative recursion is wrong.", end=" ")
        if arb >= 1e-4:
            print("arbiter NOT converged; fix it before reading [L4].", end=" ")
        elif l4 >= 1e-6:
            print("disagrees with the direct sum.", end=" ")
        if l5 >= 1e-12:
            print("z-parity violated.", end=" ")
        print()
    print("=" * 84)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
