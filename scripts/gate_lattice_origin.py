"""GATE: the lattice sum AT r = 0 -- the diagonal term, where the split is singular.

WHY THIS NEEDS ITS OWN GATE. Everything gated so far is evaluated away from
lattice points. The term a periodic solver needs on the DIAGONAL is different in
kind: the field at a cube from all of its own in-plane images, which is the sum
over R != 0 evaluated at r = 0. There the real-space half's R = 0 term and the
subtracted free-space self-term are each singular, and only their difference is
finite. Computing that difference by evaluating both near r = 0 is catastrophic
cancellation, and it gets worse at every derivative order -- the fourth-order
strain term is the worst case.

`origin_scalar_tensors` instead takes the limit analytically: both singular
pieces are (1/(4 pi d)) times a function analytic at d = 0 whose values agree at
d = 0, so the poles cancel and the remainder is a Taylor series.

WHAT IS CHECKED:
  [O1] the EVENNESS the construction depends on. The remainder must be an even
       function of d, because every other piece of the split is smooth at r = 0
       and so is the total. `_regularised_self_ladder` raises on a non-vanishing
       odd coefficient, so this checks that it does not raise AND reports the
       margin -- a silent near-miss is what would make the origin value subtly
       wrong rather than obviously wrong.
  [O2] the r -> 0 limit. `lattice_scalar_tensors` at small but non-zero r is
       already gated (eta-independence to 1.3e-13 at order 4, and against a
       converged damped direct sum). It must converge to the origin value as r
       shrinks -- and it must do so at a RATE, not merely end up close, because
       "close" is also what a wrong constant plus cancellation noise looks like.
       The gate shows the approach and then the point where cancellation takes
       over, which is exactly why the analytic limit is needed.
  [O3] eta-independence of the origin value itself, at every order. The same
       load-bearing invariant as elsewhere: eta is bookkeeping and the total
       cannot depend on it.
  [O4] the symmetries that actually hold at r = 0. An earlier version of this
       gate asserted that every odd-order tensor vanishes and the even ones are
       isotropic, on the grounds that x = 0 kills all but the all-delta
       structures. THAT IS TRUE ONLY OF THE REGULARISED R = 0 PIECE. The rest of
       the sum is evaluated at x = -R for every other lattice site, and carries
       the full tensor structure of those; the reciprocal half likewise carries
       (i q)^a factors. What does hold is:
         (a) z-parity. Every lattice vector is in-plane, so the sum depends on z
             only through z^2 and any entry with an ODD number of z indices is
             zero. This holds for any Bloch vector.
         (b) at k_par = 0 only, R -> -R is a symmetry of the summand, so every
             ODD-ORDER tensor vanishes. It fails at k_par != 0, which is why
             this is checked at both and expected to differ.

Run:  conda run -n seismic python scripts/gate_lattice_origin.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubic_scattering.lattice_kupradze import (  # noqa: E402
    _regularised_self_ladder,
    lattice_scalar_tensors,
    origin_scalar_tensors,
)

A_L = 2.0
K_PAR = np.array([0.2, 0.1])
ETA1, ETA2 = 0.7, 1.15
RC, GC = 8, 8
KAPPA = 1.5


def main() -> int:
    print("=" * 84)
    print("GATE -- the Bloch lattice sum at r = 0 (the diagonal term)")
    print(f"  a_L = {A_L}, k_par = {tuple(K_PAR)}, kappa = {KAPPA}, eta = {ETA1} vs {ETA2}")
    print("=" * 84)

    # ---- [O1] evenness of the regularised self-term -------------------------
    print("\n  [O1] the regularised R = 0 term must be EVEN in d")
    o1_ok = True
    try:
        ladder = _regularised_self_ladder(KAPPA, ETA1, 4)
        print(f"       accepted; ladder f_0..f_2 = {', '.join(f'{abs(v):.4e}' for v in ladder[:3])}")
    except ValueError as exc:
        o1_ok = False
        print(f"       REJECTED: {exc}")

    # ---- [O2] the r -> 0 limit, as a rate -----------------------------------
    print("\n  [O2] lattice_scalar_tensors(r) -> origin value as r -> 0")
    origin = origin_scalar_tensors(KAPPA, ETA1, RC, GC, A_L, K_PAR)
    # The sum is smooth at r = 0, so the gap is O(|r|): each halving of |r| must
    # halve it. The RATE is the evidence -- merely being close is also what a
    # wrong constant plus cancellation noise looks like. Start well inside the
    # asymptotic regime; |r| comparable to the pitch is not in it.
    print(f"       {'|r|':>10} {'order 0':>11} {'order 2':>11} {'order 4':>11}  ratios")
    rows = []
    for frac in (0.05, 0.025, 0.0125, 0.00625):
        r_vec = np.array([0.0, frac * A_L, 0.0])
        near = lattice_scalar_tensors(r_vec, KAPPA, ETA1, RC, GC, A_L, K_PAR)
        rels = [
            float(np.abs(near[o] - origin[o]).max()) / float(np.abs(origin[o]).max()) for o in (0, 2, 4)
        ]
        ratio = ""
        if rows:
            ratio = "  " + " ".join(f"{rows[-1][k] / rels[k]:4.2f}" for k in range(3))
        rows.append(rels)
        print(f"       {frac * A_L:10.5f} {rels[0]:11.3e} {rels[1]:11.3e} {rels[2]:11.3e}{ratio}")
    ratios = [rows[i][k] / rows[i + 1][k] for i in range(len(rows) - 1) for k in range(3)]
    o2_ok = all(1.7 < r < 2.3 for r in ratios)
    print("       ratio ~ 2 per halving of |r| is first-order approach to the")
    print("       analytic limit -- the limit is right, not merely nearby.")

    # ---- [O3] eta-independence of the origin value --------------------------
    print("\n  [O3] eta-independence of the origin value, by order")
    other = origin_scalar_tensors(KAPPA, ETA2, RC, GC, A_L, K_PAR)
    o3 = 0.0
    for order in range(5):
        scale = float(np.abs(origin[order]).max())
        if scale == 0.0:
            print(f"       {order:6d}   (identically zero)")
            continue
        rel = float(np.abs(origin[order] - other[order]).max()) / scale
        o3 = max(o3, rel)
        print(f"       {order:6d} {scale:13.5e} {rel:11.2e}")

    # ---- [O4] the symmetries that actually hold -----------------------------
    print("\n  [O4] z-parity (any k_par), and odd orders vanishing at k_par = 0")
    o4 = 0.0
    for order in range(1, 5):
        t = origin[order]
        scale = float(np.abs(t).max())
        for idx in np.ndindex(*((3,) * order)):
            if idx.count(0) % 2 == 1:
                o4 = max(o4, abs(complex(t[idx])) / scale)
    print(f"       worst odd-z entry at k_par = {tuple(K_PAR)}: {o4:.2e}")

    at_gamma = origin_scalar_tensors(KAPPA, ETA1, RC, GC, A_L, np.zeros(2))
    o4_odd = max(float(np.abs(at_gamma[o]).max()) / float(np.abs(at_gamma[o - 1]).max()) for o in (1, 3))
    o4_bloch = max(float(np.abs(origin[o]).max()) / float(np.abs(origin[o - 1]).max()) for o in (1, 3))
    print(f"       odd-ORDER tensors at k_par = 0    : {o4_odd:.2e}  (must vanish)")
    print(f"       the same at k_par != 0            : {o4_bloch:.2e}  (must NOT)")
    print("       the contrast is the point: a gate that passed at k_par = 0 alone")
    print("       would also pass with the Bloch phase sign reversed.")

    print("\n" + "=" * 84)
    print(f"  [O1] evenness accepted        : {'PASS' if o1_ok else 'FAIL'}")
    print(f"  [O2] r -> 0 approach          : {'PASS' if o2_ok else 'FAIL'}")
    print(f"  [O3] eta-independence         : {o3:.2e}")
    print(f"  [O4] z-parity / odd at k=0    : {o4:.2e} / {o4_odd:.2e}")
    ok = o1_ok and o2_ok and o3 < 1e-9 and o4 < 1e-12 and o4_odd < 1e-12 and o4_bloch > 1e-3
    if ok:
        print("\n  PASS: the diagonal term is available analytically, so the periodic")
        print("  kernel no longer needs any value evaluated near a lattice point.")
        print("  With this the lattice sum is complete for every separation the")
        print("  solver asks for, including the self-image term at r = 0.")
    else:
        print("\n  FAIL:", end=" ")
        if not o1_ok:
            print("odd Taylor coefficient -- the two singular pieces do not match.", end=" ")
        if not o2_ok:
            print("the r -> 0 limit does not approach the analytic value.", end=" ")
        if o3 >= 1e-9:
            print("origin value is eta-dependent.", end=" ")
        if o4 >= 1e-12:
            print("z-parity violated at the origin.", end=" ")
        if o4_odd >= 1e-12:
            print("odd-order tensors do not vanish at k_par = 0.", end=" ")
        if o4_bloch <= 1e-3:
            print("odd orders vanish even at k_par != 0 -- the Bloch phase is lost.", end=" ")
        print()
    print("=" * 84)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
