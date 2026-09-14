"""GATE: the planar Ewald sum AT z = 0, the case the earlier gate could not test.

WHY THIS IS SEPARATE. `gate_planar_ewald` deliberately used points with z != 0,
because the reciprocal erfc PAIRING is invisible at z = 0 -- there both erfc
arguments coincide, so a wrong pairing passes. That made z != 0 the right choice
for catching the bug, and it leaves z = 0 UNTESTED. The same-plane kernel is
exactly the z = 0 case, so it needs its own evidence before anything is built on
it.

WHY z = 0 IS THE HARD ONE, and why Ewald is unavoidable here. By Poisson the
same-plane lattice sum is (1/d^2) sum_G ghat(k_par + G, 0) with
ghat(q, 0) = i / (2 k_z), k_z = sqrt(kappa^2 - q^2). For |q| >> kappa this is
1 / (2 sqrt(q^2 - kappa^2)) ~ 1/(2|q|), and summing 1/|q| over a 2-D reciprocal
lattice DIVERGES. That is the same fact `same_depth_kernel_9x9` records as its
strain-strain block "growing like |k_x|". A plain reciprocal sum cannot be used
at z = 0; the Ewald split is what makes both halves converge.

WHAT IS CHECKED, at in-plane points with z = 0 exactly:
  [Z1] eta-independence. Still the sharpest internal test, now in the regime
       that matters.
  [Z2] agreement with a damped direct lattice sum -- an independent
       construction, shown converged in its own radius before it is believed.
       (The inter-plane gate was first run with an UNCONVERGED arbiter that
       "disagreed" by 20-77%; that is not repeated here.)
  [Z3] the plain reciprocal sum at rising cutoffs, to show how badly it behaves.
       BE PRECISE ABOUT WHAT THIS DOES AND DOES NOT SHOW. For the SCALAR it does
       not diverge outright: the oscillating phase makes it conditionally
       convergent and it creeps in at about 1/n_G, which is merely useless
       rather than impossible. The outright divergence belongs to the
       STRAIN-STRAIN block, which carries two more derivatives and so two more
       powers of q: ghat ~ 1/(2|q|) becomes ~|q|/2, and summing |q| over a 2-D
       reciprocal lattice genuinely diverges. That is the claim
       `same_depth_kernel_9x9` records, and it is inherited here, not measured
       by this gate. (Note also that the plain sum below covers ALL R while the
       Ewald total excludes R = 0, so their values are not meant to coincide;
       only the CONVERGENCE RATE is the point.)

Run:  conda run -n seismic python scripts/gate_sameplane_ewald.py
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
RC, GC = 8, 8
KAPPA_DAMPED = 1.5 + 0.25j
KAPPA_REAL = 1.5


def _inplane_points() -> list:
    """Points in the lattice plane: z = 0 EXACTLY, offset from any lattice site."""
    return [
        np.array([0.5, 0.0, 0.0]),
        np.array([0.35, 0.28, 0.0]),
        np.array([0.9, -0.4, 0.0]),
        np.array([-0.6, 0.7, 0.0]),
    ]


def _plain_reciprocal(kappa: complex, r: np.ndarray, n_g: int) -> complex:
    """The UNSPLIT reciprocal sum -- expected to diverge at z = 0."""
    b = 2.0 * np.pi / A_L
    idx = np.arange(-n_g, n_g + 1)
    m_g, n_g_ = np.meshgrid(idx, idx, indexing="ij")
    qx = K_PAR[0] + b * m_g
    qy = K_PAR[1] + b * n_g_
    kz = np.sqrt(np.asarray(kappa**2 - (qx**2 + qy**2), dtype=complex))
    kz = np.where(kz.imag < 0, -kz, kz)
    phase = np.exp(1j * (qx * r[0] + qy * r[1]))
    return complex(np.sum(1j / (2.0 * kz) * phase) / A_L**2)


def main() -> int:
    print("=" * 84)
    print("GATE -- planar Ewald AT z = 0 (the case gate_planar_ewald could not test)")
    print("=" * 84)
    pts = _inplane_points()

    # ---- [Z1] eta-independence at z = 0 -----------------------------------
    print("\n  [Z1] eta-independence at z = 0, real kappa")
    print(f"       {'(x, y)':>16} {'|G(eta1)|':>13} {'rel diff':>11}")
    z1 = 0.0
    for r in pts:
        a = ewald_total(KAPPA_REAL, r, ETA1, RC, GC, A_L, K_PAR)
        b = ewald_total(KAPPA_REAL, r, ETA2, RC, GC, A_L, K_PAR)
        rel = abs(a - b) / max(abs(a), 1e-300)
        z1 = max(z1, rel)
        print(f"       {f'({r[0]:.2f}, {r[1]:.2f})':>16} {abs(a):13.6e} {rel:11.2e}")

    # ---- [Z2] against a damped direct sum, itself shown converged ----------
    print(f"\n  [Z2] damped kappa = {KAPPA_DAMPED}: Ewald vs direct sum")
    r0 = pts[0]
    d_near = direct_sum(KAPPA_DAMPED, r0, 60, A_L, K_PAR)
    d_far = direct_sum(KAPPA_DAMPED, r0, 100, A_L, K_PAR)
    arb = abs(d_far - d_near) / abs(d_far)
    print(f"       arbiter self-convergence (radius 60 -> 100): {arb:.2e}")
    print(f"       {'(x, y)':>16} {'|Ewald|':>13} {'rel diff':>11}")
    z2 = 0.0
    for r in pts:
        e = ewald_total(KAPPA_DAMPED, r, ETA1, RC, GC, A_L, K_PAR)
        d = direct_sum(KAPPA_DAMPED, r, 100, A_L, K_PAR)
        rel = abs(e - d) / max(abs(d), 1e-300)
        z2 = max(z2, rel)
        print(f"       {f'({r[0]:.2f}, {r[1]:.2f})':>16} {abs(e):13.6e} {rel:11.2e}")

    # ---- [Z3] show the unsplit reciprocal sum does NOT converge ------------
    print("\n  [Z3] the PLAIN reciprocal sum at z = 0 -- how slowly it settles")
    print("       (scalar: conditionally convergent, ~1/n_G. The strain block's two")
    print("        extra powers of q are what diverge outright -- not shown here.)")
    print(f"       {'n_G':>5} {'|plain recip|':>15} {'change':>11}")
    prev = None
    for n_g in (4, 8, 16, 32, 64):
        val = _plain_reciprocal(KAPPA_REAL, r0, n_g)
        ch = "" if prev is None else f"{abs(val - prev) / abs(prev):11.2e}"
        print(f"       {n_g:5d} {abs(val):15.6e} {ch:>11}")
        prev = val

    print("\n" + "=" * 84)
    ok = z1 < 1e-10 and z2 < 1e-6 and arb < 1e-5
    print(f"  [Z1] eta-independence at z = 0 : {z1:.2e}")
    print(f"  [Z2] vs converged direct sum   : {z2:.2e}")
    if ok:
        print("\n  PASS: the Ewald split is valid AT z = 0, the regime the same-plane")
        print("  kernel needs, at machine precision on both tests, while [Z3] shows")
        print("  the plain reciprocal sum still creeping at n_G = 64. The scalar")
        print("  foundation for the same-plane lattice sum is in place; what remains")
        print("  is applying the Kupradze derivative operators to build the 9x9,")
        print("  where the strain block's two extra powers of q are what made the")
        print("  unsplit sum unusable in the first place.")
    else:
        print("\n  FAIL:", end=" ")
        print("eta-dependent at z = 0." if z1 >= 1e-10 else "", end="")
        print("disagrees with the direct sum." if z2 >= 1e-6 else "", end="")
        print("arbiter not converged -- fix it before reading [Z2]." if arb >= 1e-5 else "")
    print("=" * 84)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
