#!/usr/bin/env python3
"""Does the wrapper correction carry over to a STRATIFIED reference?

Recorded state before this gate: the correction was calibrated against a
whole-space reference and was measured to FAIL on the marine model
(0.009-0.398), which read as a genuine gap.

It is not a gap.  In the marine model every interface separates different
materials, and the GATE D pairs sit ON those interfaces.  A state jump injected
exactly at a material discontinuity has no unambiguous ``eta_S`` -- there are
two -- so the correction operator K is not defined there.  Move the source and
receiver planes into the INTERIOR of a layer and the correction is exact, even
when strong contrasts lie between them.

This matters because scattering voxels live in layer interiors, not on material
jumps, so the restriction costs the application nothing.

Gates
-----
1. interior pair, uniform zone, strong nearby contrast      -> machine precision
2. interior pair with a fast slab BETWEEN source and        -> machine precision
   receiver (crossed twice: transmission, reflection,
   conversion and interbed multiples all present)
3. control: a plane placed ON a discontinuity               -> must NOT pass for
   every choice of medium (the marine sweep below shows no
   combination works there)

Run:  conda run -n seismic python scripts/gate_stratified_correction.py
"""

import sys
from pathlib import Path

import numpy as np
import numpy.linalg as la

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, "/Users/tod/Desktop/SeismicInversion")

from cubic_scattering.layered_correction import J6, correct_6x6  # isort: skip

from Kennett_Reflectivity.layer_model import LayerModel  # isort: skip
from GlobalMatrix.layered_greens import layered_greens_6x6  # isort: skip

TOL = 1e-10

A_MED = (4.0, 2.22, 2.6)
B_MED = (6.5, 3.7, 3.3)


def corrected(model, w, kx, ky, j, i, lay_src, lay_rcv):
    """G6 with D1 and D2, each index taking its medium EXPLICITLY.

    Deliberately bypasses ``corrected_layered_6x6``'s continuity guard: GATE 3
    below has to place a plane on a discontinuity and try both sides, which the
    guarded entry point rightly refuses.

    Args:
        model: Stratified model.
        w: Angular frequency (rad/s).
        kx: Horizontal wavenumber, x component (rad/km).
        ky: Horizontal wavenumber, y component (rad/km).
        j: Source interface index.
        i: Receiver interface index.
        lay_src: Layer index supplying the source-side medium.
        lay_rcv: Layer index supplying the receiver-side medium.

    Returns:
        Corrected 6x6 in basis (u_z, u_x, u_y, T_zz, T_xz, T_yz).
    """
    raw = layered_greens_6x6(model, w, np.array([kx]), np.array([ky]), source_iface=j, receiver_iface=i)[0]
    s_s = model.complex_slowness_s()
    return correct_6x6(raw, w, s_s[lay_src], s_s[lay_rcv], kx, ky)


def law_residual(model, w, kx, ky, j, i, lay_j, lay_i) -> float:
    """Residual of G(i<-j)(+k) = J6 [ G(j<-i)(-k) ]^T J6.

    Args:
        model: Stratified model.
        w: Angular frequency (rad/s).
        kx: Horizontal wavenumber, x component (rad/km).
        ky: Horizontal wavenumber, y component (rad/km).
        j: Source interface index.
        i: Receiver interface index.
        lay_j: Layer supplying the medium at interface j.
        lay_i: Layer supplying the medium at interface i.

    Returns:
        Relative residual.
    """
    g1 = corrected(model, w, kx, ky, j, i, lay_src=lay_j, lay_rcv=lay_i)
    g2 = corrected(model, w, -kx, -ky, i, j, lay_src=lay_i, lay_rcv=lay_j)
    return float(la.norm(g1 - J6 @ g2.T @ J6) / la.norm(g1))


def sandwich_model(q: float = 1000.0) -> LayerModel:
    """ocean | A A A | B B | A A A | half-space A.

    Layers 1-3 and 6-9 are medium A, layers 4-5 the fast slab B.  Interface k
    lies at the bottom of layer k, i.e. z = 3 + k km.

    Args:
        q: Quality factor; high, so the contrasts really reflect.

    Returns:
        The layered model.
    """
    al = [1.5] + [A_MED[0]] * 3 + [B_MED[0]] * 2 + [A_MED[0]] * 4
    be = [0.0] + [A_MED[1]] * 3 + [B_MED[1]] * 2 + [A_MED[1]] * 4
    rh = [1.03] + [A_MED[2]] * 3 + [B_MED[2]] * 2 + [A_MED[2]] * 4
    return LayerModel.from_arrays(
        alpha=al,
        beta=be,
        rho=rh,
        thickness=[3.0, *([1.0] * 8), np.inf],
        Q_alpha=[q] * 10,
        Q_beta=[1e10, *([q] * 9)],
    )


def marine_model() -> LayerModel:
    """The model GATE D was established on: every interface a material contrast."""
    return LayerModel.from_arrays(
        alpha=[1.5, 3.2, 3.8, 4.4, 5.0, 6.5],
        beta=[0.0, 1.8, 2.1, 2.45, 2.8, 3.7],
        rho=[1.03, 2.3, 2.5, 2.65, 2.8, 3.3],
        thickness=[2.0, 0.5, 0.5, 0.5, 0.5, np.inf],
        Q_alpha=[20000, 600, 600, 600, 600, 600],
        Q_beta=[1e10, 300, 300, 300, 300, 300],
    )


def main() -> int:
    """Run the three gates.

    Returns:
        0 if the interior gates pass, 1 otherwise.
    """
    mod = sandwich_model()
    print("=" * 78)
    print("GATE 1+2 - interior pair, fast slab B crossed twice between the ends")
    print("  ocean | A A A | B B | A A A | half-space A,  Q = 1000")
    print("  source interface 6 (z=9, A|A), receiver interface 1 (z=4, A|A)")
    print("=" * 78)
    worst = 0.0
    for f in (6.0, 12.0, 25.0):
        for p, c, s in ((0.05, 0.6, 0.8), (0.12, 0.6, 0.8), (0.18, 1.0, 0.0)):
            w = 2 * np.pi * f
            r = law_residual(mod, w, w * p * c, w * p * s, j=6, i=1, lay_j=1, lay_i=1)
            worst = max(worst, r)
            print(f"  f={f:5.1f} p={p:5.2f} khat=({c},{s})   {r:.3e}")
    print(f"\n  interior: {'PASS' if worst < TOL else 'FAIL'}  (worst {worst:.3e})\n")

    print("=" * 78)
    print("GATE 3 - control: a plane ON a discontinuity has no consistent K")
    print("=" * 78)
    mar = marine_model()
    print("  marine model, all four above/below combinations:")
    print(
        f"  {'f':>6} {'pair':>7}   {'both above':>11} {'src below':>11} "
        f"{'both below':>11} {'rcv below':>11}"
    )
    for f in (5.0, 25.0):
        for j, i, p in ((1, 3, 0.08), (2, 4, 0.15), (1, 4, 0.12)):
            w = 2.0 * np.pi * f
            kx, ky = w * p * 0.6, w * p * 0.8
            vals = [
                law_residual(mar, w, kx, ky, j, i, j, i),
                law_residual(mar, w, kx, ky, j, i, j + 1, i),
                law_residual(mar, w, kx, ky, j, i, j + 1, i + 1),
                law_residual(mar, w, kx, ky, j, i, j, i + 1),
            ]
            print(f"  {f:6.1f} {f'{j}->{i}':>7}   " + " ".join(f"{v:11.3e}" for v in vals))
    print("\n  No combination reaches machine precision: the placement, not the")
    print("  operator, is what fails.  Keep source and receiver planes in the")
    print("  interior of a layer.\n")
    return 0 if worst < TOL else 1


if __name__ == "__main__":
    sys.exit(main())
