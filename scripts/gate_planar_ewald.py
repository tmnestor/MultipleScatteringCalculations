"""GATE: the planar Ewald sum, ported from the validated Mathematica.

This reproduces the self-verification that `Mathematica/IntraPlaneKambe.wl`
applies to itself, on the same parameters, so the port is checked against the
work it came from rather than against fresh intuition:

    a_L = 2, k_par = (0.2, 0.1), eta1 = 0.7, eta2 = 1.15, Rc = Gc = 6,
    six random points at |r| = 0.5 with z != 0.

[W1] ETA-INDEPENDENCE at REAL kappa. The splitting parameter eta is a pure
     bookkeeping device: it decides how much of the sum is carried by the
     real-space half and how much by the reciprocal half, and the total cannot
     depend on it. This is the LOAD-BEARING test. The project's own record
     names the reciprocal erfc PAIRING as a root-cause bug that "z = 0 cannot
     detect; only eta-independence does" -- which is why the points below are
     chosen with z != 0 and why two well-separated etas are used.

[W2] AGREEMENT WITH A DAMPED DIRECT SUM. With Im(kappa) > 0 the plain lattice
     sum converges and is an independent construction of the same quantity. It
     shares no code path with the Ewald halves.

[W3] CONVERGENCE IN THE CUTOFFS. Raising Rc and Gc must not move the answer.
     A result that drifts is truncated, not converged, whatever [W1] says --
     eta-independence alone can be satisfied by two equally truncated sums.

Run:  conda run -n seismic python scripts/gate_planar_ewald.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubic_scattering.planar_ewald import direct_sum, ewald_total  # noqa: E402

A_L = 2.0
K_PAR = np.array([0.2, 0.1])
ETA1, ETA2 = 0.7, 1.15
RC, GC = 6, 6
KAPPA_REAL = 1.5
KAPPA_DAMPED = 1.5 + 0.25j
N_BIG = 40
TOL = 1e-10


def _points() -> list:
    """Six points at |r| = 0.5 with z != 0 -- the Mathematica's own choice."""
    rng = np.random.default_rng(20260623)
    pts = []
    while len(pts) < 6:
        u = rng.uniform(-1.0, 1.0)
        ph = rng.uniform(0.0, 2.0 * np.pi)
        if abs(u) < 0.15:  # keep z well away from 0, where the pairing hides
            continue
        st = np.sqrt(1.0 - u * u)
        pts.append(0.5 * np.array([st * np.cos(ph), st * np.sin(ph), u]))
    return pts


def main() -> int:
    print("=" * 84)
    print("GATE -- planar Ewald sum (port of Mathematica/IntraPlaneKambe.wl)")
    print(f"  a_L = {A_L}, k_par = {tuple(K_PAR)}, eta = {ETA1} vs {ETA2}, Rc = Gc = {RC}")
    print("=" * 84)

    pts = _points()

    # ---- [W1] eta-independence, real kappa --------------------------------
    print("\n  [W1] eta-independence at REAL kappa (the pairing test)")
    print(f"       {'z':>9} {'|G(eta1)|':>13} {'rel diff':>11}")
    w1 = 0.0
    for r in pts:
        a = ewald_total(KAPPA_REAL, r, ETA1, RC, GC, A_L, K_PAR)
        b = ewald_total(KAPPA_REAL, r, ETA2, RC, GC, A_L, K_PAR)
        rel = abs(a - b) / max(abs(a), 1e-300)
        w1 = max(w1, rel)
        print(f"       {r[2]:9.4f} {abs(a):13.6e} {rel:11.2e}")

    # ---- [W2] against a damped direct sum ---------------------------------
    print(f"\n  [W2] damped kappa = {KAPPA_DAMPED}: Ewald vs direct sum (L = {N_BIG})")
    print(f"       {'z':>9} {'|Ewald|':>13} {'rel diff':>11}")
    w2 = 0.0
    for r in pts:
        e = ewald_total(KAPPA_DAMPED, r, ETA1, RC, GC, A_L, K_PAR)
        d = direct_sum(KAPPA_DAMPED, r, N_BIG, A_L, K_PAR)
        rel = abs(e - d) / max(abs(d), 1e-300)
        w2 = max(w2, rel)
        print(f"       {r[2]:9.4f} {abs(e):13.6e} {rel:11.2e}")

    # ---- [W3] convergence in the cutoffs ----------------------------------
    print("\n  [W3] cutoff convergence -- Rc, Gc raised")
    w3 = 0.0
    for r in pts[:3]:
        a = ewald_total(KAPPA_REAL, r, ETA1, RC, GC, A_L, K_PAR)
        b = ewald_total(KAPPA_REAL, r, ETA1, RC + 4, GC + 4, A_L, K_PAR)
        rel = abs(a - b) / max(abs(a), 1e-300)
        w3 = max(w3, rel)
        print(f"       z = {r[2]:8.4f}   Rc,Gc {RC} -> {RC + 4}:  {rel:.2e}")

    print("\n" + "=" * 84)
    print(f"  [W1] eta-independence  {w1:.2e}")
    print(f"  [W2] vs direct sum     {w2:.2e}")
    print(f"  [W3] cutoff drift      {w3:.2e}")
    ok = w1 < TOL and w2 < 1e-6 and w3 < TOL
    if ok:
        print("\n  PASS: the port reproduces the Mathematica's own self-verification.")
        print("  eta-independence at z != 0 means the reciprocal erfc pairing is")
        print("  right -- the one thing the z = 0 case cannot check. This is the")
        print("  foundation for a correct periodic lateral sum.")
    else:
        print("\n  FAIL:", end=" ")
        print("eta-dependence => the erfc pairing is wrong." if w1 >= TOL else "", end="")
        print("disagrees with the direct sum." if w2 >= 1e-6 else "", end="")
        print("not converged in the cutoffs." if w3 >= TOL else "")
    print("=" * 84)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
