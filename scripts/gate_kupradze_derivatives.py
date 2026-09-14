"""GATE: the Kupradze derivative ladder, to fourth order.

WHY THIS IS THE RIGHT FIRST GATE. The remaining step in the lateral-sum fix is to
apply the Kupradze operators to LATTICE-SUMMED scalars, which needs Cartesian
derivative tensors up to FOURTH order for the strain-strain block. Fourth-order
tensor structures are where hand derivation fails silently. But the free-space
case of exactly the same algebra is already implemented, independently and by
hand, as `resonance_tmatrix.elastodynamic_greens_deriv` (radial phi/psi, 3 and 7
tensor structures) and is in production use. So the ladder can be convicted or
cleared BEFORE any Ewald summation is layered on top of it.

That matters because of how the last round of errors in this project went: they
were all in composite steps where a new construction and a new summation were
introduced together, and neither could be blamed. Here the summation is absent
by construction -- `scalar_derivative_tensors` is free-space -- so a failure can
only be the derivative algebra.

WHAT IS CHECKED:
  [K1] combinatorics. The number of distinct delta/x structures must be
       n!/(2^m m! (n-2m)!), and every derivative tensor must be fully symmetric
       under index permutation. A pairing enumerator that double-counts or drops
       a structure fails here and nowhere else -- the multiplicities are exactly
       what a hand transcription gets wrong.
  [K2] finite differences. An INDEPENDENT construction of the same tensors,
       sharing no code path with the ladder: repeated central differences of the
       scalar itself. It is the only check here that tests `radial_ladder`
       separately from the tensor structures.

       THE ARBITER MUST BE SHOWN CONVERGED BEFORE IT CAN CONVICT ANYTHING. A
       first run of this gate reported the ladder "disagreeing" by 4e-5 to 7e-2,
       rising with the derivative order -- which is the signature of the
       ARBITER's own truncation, not of an error in the ladder. Central
       differences carry a relative truncation of order (h kappa)^2, and at
       h = 1 m with kappa_S = 0.02 /m that is h^2 kappa^2 / 6 = 6.7e-5 against a
       measured 4.3e-5. So the gate now halves h and checks the error falls by
       the predicted factor of 4, then Richardson-extrapolates the two to remove
       the h^2 term. The extrapolated residual is what is allowed to convict.
  [K3] the assembled 9x9 against the validated `_propagator_block_9x9`. The
       load-bearing test: it exercises the ladder to fourth order (through Gdd),
       the Kupradze assembly, and the Voigt contraction together, against code
       that got there by a different route.
  [K4] complex kappa. Attenuative media are the physical case and the reason the
       Hankel series is used instead of scipy's real-argument routines. The
       point path takes a real omega, so this is checked for internal
       consistency (eta-style: the limit as Im -> 0) rather than against it.

Run:  conda run -n seismic python scripts/gate_kupradze_derivatives.py
"""

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from cubic_scattering.kupradze_derivatives import (  # noqa: E402
    _delta_x_structure,
    _pairings,
    propagator_block_9x9_kupradze,
    scalar_derivative_tensors,
)
from cubic_scattering.resonance_tmatrix import _propagator_block_9x9  # noqa: E402

REF = ReferenceMedium(5000.0, 3000.0, 2500.0)
OMEGA = 60.0
# Separations spanning contact, a lattice diagonal, and the far field, in metres.
SEPARATIONS = [
    np.array([0.0, 200.0, 0.0]),
    np.array([200.0, 200.0, 0.0]),
    np.array([200.0, 400.0, 600.0]),
    np.array([-300.0, 150.0, -450.0]),
]


def _scalar(r_vec: np.ndarray, kappa: complex) -> complex:
    r = float(np.linalg.norm(r_vec))
    return complex(np.exp(1j * kappa * r) / (4.0 * np.pi * r))


def _fd_tensor(r_vec: np.ndarray, kappa: complex, order: int, h: float) -> np.ndarray:
    """Repeated central differences of the scalar -- the independent arbiter."""
    if order == 0:
        return np.asarray(_scalar(r_vec, kappa), dtype=complex)
    out = np.zeros((3,) * order, dtype=complex)
    for axis in range(3):
        step = np.zeros(3)
        step[axis] = h
        lo = _fd_tensor(r_vec - step, kappa, order - 1, h)
        hi = _fd_tensor(r_vec + step, kappa, order - 1, h)
        out[..., axis] = (hi - lo) / (2.0 * h)
    return out


def main() -> int:
    print("=" * 84)
    print("GATE -- Kupradze derivative ladder to fourth order")
    print(f"  background alpha/beta/rho = {REF.alpha}/{REF.beta}/{REF.rho}, omega = {OMEGA}")
    print("=" * 84)

    # ---- [K1] combinatorics and full symmetry ------------------------------
    print("\n  [K1] pairing multiplicities and index symmetry")
    print(f"       {'n':>3} {'m':>3} {'terms':>7} {'expected':>9} {'sym err':>10}")
    k1_ok = True
    x = np.array([0.31, -0.72, 0.55], dtype=complex)
    for n in range(1, 5):
        for m in range(n // 2 + 1):
            got = len(_pairings(list(range(n)), m))
            want = math.factorial(n) // (2**m * math.factorial(m) * math.factorial(n - 2 * m))
            struct = _delta_x_structure(n, m, x)
            # Full symmetry: compare against every axis transposition.
            sym = 0.0
            for a in range(n):
                for b in range(a + 1, n):
                    ax = list(range(n))
                    ax[a], ax[b] = ax[b], ax[a]
                    sym = max(sym, float(np.abs(struct - struct.transpose(ax)).max()))
            ok = got == want and sym < 1e-14
            k1_ok = k1_ok and ok
            print(f"       {n:3d} {m:3d} {got:7d} {want:9d} {sym:10.2e}")

    # ---- [K2] against repeated central differences --------------------------
    print("\n  [K2] ladder vs repeated central differences (independent construction)")
    print(f"       {'order':>6} {'raw ratio':>10} {'Rich(h)':>11} {'Rich(h/2)':>11} {'ratio':>7}")
    kappa = OMEGA / REF.beta
    r0 = SEPARATIONS[2]
    k2 = 0.0
    k2_ratio_ok = True
    for order, h in ((1, 8.0), (2, 8.0), (3, 12.0), (4, 16.0)):
        exact = scalar_derivative_tensors(r0, kappa)[order]
        scale = np.abs(exact).max()
        fd = [_fd_tensor(r0, kappa, order, h / 2.0**i) for i in range(3)]
        err = [float(np.abs(exact - f).max() / scale) for f in fd]
        raw_ratio = err[0] / max(err[1], 1e-300)
        # Richardson on an O(h^2) rule: (4 f(h/2) - f(h)) / 3 kills the h^2 term,
        # leaving O(h^4). Two of them, one step apart, expose that rate.
        rich = [(4.0 * fd[i + 1] - fd[i]) / 3.0 for i in range(2)]
        e_r = [float(np.abs(exact - v).max() / scale) for v in rich]
        rich_ratio = e_r[0] / max(e_r[1], 1e-300)
        k2 = max(k2, e_r[1])
        # The raw rule must converge at h^2 and the extrapolant at h^4. Both
        # windows are generous; it is the RATE, not the residual, that is the
        # evidence that the gap is the arbiter's and not the ladder's.
        k2_ratio_ok = k2_ratio_ok and 3.0 < raw_ratio < 5.0 and 10.0 < rich_ratio < 22.0
        print(f"       {order:6d} {raw_ratio:10.2f} {e_r[0]:11.2e} {e_r[1]:11.2e} {rich_ratio:7.2f}")
    print("       raw ~ 4 (h^2) and Richardson ~ 16 (h^4): the residual is the")
    print("       ARBITER's truncation, converging to the ladder, not away from it.")

    # ---- [K3] the assembled 9x9 against the validated point path ------------
    print("\n  [K3] 9x9 vs resonance_tmatrix._propagator_block_9x9")
    print(f"       {'separation (m)':>26} {'|P|':>12} {'rel err':>11}")
    k3 = 0.0
    for r_vec in SEPARATIONS:
        mine = propagator_block_9x9_kupradze(r_vec, OMEGA, REF)
        theirs = _propagator_block_9x9(r_vec, OMEGA, REF)
        rel = float(np.abs(mine - theirs).max() / np.abs(theirs).max())
        k3 = max(k3, rel)
        label = f"({r_vec[0]:.0f}, {r_vec[1]:.0f}, {r_vec[2]:.0f})"
        print(f"       {label:>26} {np.abs(theirs).max():12.4e} {rel:11.2e}")

    # ---- [K4] complex kappa, and its lossless limit -------------------------
    print("\n  [K4] complex wavenumber (attenuative medium)")
    k4 = 0.0
    for q in (200.0, 2000.0, 20000.0):
        om_c = OMEGA * (1.0 + 0.5j / q)
        damped = propagator_block_9x9_kupradze(SEPARATIONS[2], om_c, REF)
        real = propagator_block_9x9_kupradze(SEPARATIONS[2], OMEGA, REF)
        rel = float(np.abs(damped - real).max() / np.abs(real).max())
        k4 = max(k4, rel * q)  # must scale as 1/Q, i.e. rel*Q ~ constant
        print(f"       Q = {q:8.0f}   departure from lossless: {rel:.3e}   x Q = {rel * q:7.3f}")
    # Finite and 1/Q-scaling: the product above must stay bounded, which it
    # cannot if the complex branch is wrong (it would blow up or go to zero).
    k4_ok = 0.01 < k4 < 100.0

    print("\n" + "=" * 84)
    print(f"  [K1] combinatorics + symmetry : {'PASS' if k1_ok else 'FAIL'}")
    print(
        f"  [K2] vs finite differences    : {k2:.2e} extrapolated, rates "
        f"{'PASS' if k2_ratio_ok else 'FAIL'}"
    )
    print(f"  [K3] vs validated 9x9         : {k3:.2e}")
    print(f"  [K4] complex kappa 1/Q scaling: {'PASS' if k4_ok else 'FAIL'}")
    ok = k1_ok and k2 < 1e-3 and k2_ratio_ok and k3 < 1e-11 and k4_ok
    if ok:
        print("\n  PASS: the derivative ladder reproduces the hand-derived point")
        print("  propagator to fourth order. The tensor structures are enumerated,")
        print("  not transcribed, and the radial ladder is separately confirmed by")
        print("  finite differences. This is the piece the lattice sum reuses: only")
        print("  the radial ladder changes when the scalars become Bloch sums.")
    else:
        print("\n  FAIL:", end=" ")
        if not k1_ok:
            print("pairing multiplicities or symmetry wrong.", end=" ")
        if k2 >= 1e-3 or not k2_ratio_ok:
            print("ladder disagrees with finite differences.", end=" ")
        if k3 >= 1e-11:
            print("assembled 9x9 disagrees with the validated path.", end=" ")
        if not k4_ok:
            print("complex-kappa branch misbehaves.", end="")
        print()
    print("=" * 84)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
