"""GATE: the 9x9 Bubnov-Galerkin inter-cell propagator.

WHAT IS BEING BUILT AND WHY. The lattice pairs a Galerkin T-matrix with a MOMENT
propagator. `T_0` is the first tier of the Bubnov-Galerkin hierarchy (3 constant
+ 6 linear trial functions over the cell); `inter_voxel_propagator` supplies
cell-averaged field and derivatives. The two agree only for a field that is
exactly linear across a cell, and a touching neighbour's near field has O(1)
curvature across the cell at every scale. That is why the refinement ladder
SATURATES -- it converges to a wrong limit -- and a consistent Galerkin scheme
converges at fixed polynomial degree, so saturation indicts consistency, not
basis order.

WHAT IS CHECKED, cheapest and most independent first:

  [G1] the autocorrelation kernels, against the THREE worked examples in the
       derivation. These are closed-form and exact, so agreement must be at
       machine precision. Nothing downstream is meaningful if these are wrong.

  [G2] the centre-of-mass reduction, against a DIRECT 6-D quadrature of
       Int Int phi_a G phi_b. The 6-D route shares no code with the reduced 3-D
       route -- different integrand, different domain, different rule -- so this
       tests the reduction itself, not its implementation twice. Run at a
       NON-TOUCHING separation where the kernel is smooth, so the arbiter is
       trustworthy; the face-contact quadrature bias on this project's record
       afflicts exactly the case being avoided here.

  [G3] quadrature convergence AT CONTACT, where the arbiter of [G2] cannot be
       trusted and so is not used. The apex-pyramid rule integrates the 1/rho
       singularity exactly in the radial variable, so the remaining error is
       ordinary smooth-function Gauss error and must fall with n_quad. A value
       that stalls would mean the singular handling is wrong.

  [G4] reciprocity, Gamma(-R) = Gamma(R)^T, with NO metric. A BUBNOV form uses
       the same family on both sides, so Int Int phi_a G phi_b is symmetric under
       (a,R) <-> (b,-R) directly.

       AN EARLIER VERSION OF THIS GATE DECORATED IT WITH THE 9-COMPONENT METRIC
       M = Sigma J and duly "failed" at 8e-2. That metric belongs to Foldy-Lax
       self-adjointness of G_0 T_0, not to the bare propagator, and the decorated
       statement cannot hold anyway: M Gamma(-R) = (M Gamma(R))^T would require M
       to commute with Gamma, which a diagonal M with unequal entries does not.
       The gate was failing a false statement. In the correct form the propagator
       satisfies it at 1e-16, face contact included.

       Reciprocity is homogeneous of degree one and so is blind to an overall
       scale -- it is paired here with [G2], which is an absolute comparison.

  [G5] the structural claim that motivated the build: the Galerkin operator must
       DIFFER from the moment operator by more than a per-block scale factor.
       If they agreed up to normalisation there would be nothing to fix.

Run:  conda run -n seismic python scripts/gate_galerkin_propagator.py
SI units (m, m/s, kg/m3, Pa).
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from cubic_scattering.galerkin_propagator import (  # noqa: E402
    autocorrelation,
    basis_terms,
    galerkin_block_9x9,
)
from cubic_scattering.resonance_tmatrix import elastodynamic_greens  # noqa: E402
from cubic_scattering.slab_scattering import _cell_averaged_propagator  # noqa: E402

REF = ReferenceMedium(5000.0, 3000.0, 2500.0)
OMEGA, D = 60.0, 1.0
A = 0.5 * D
# (no metric: see [G4] -- the Bubnov form is plainly transpose-symmetric)


def _direct_6d(r_vec: np.ndarray, n: int) -> np.ndarray:
    """Int Int phi_a(r) G(r-r') phi_b(r') by brute force -- the independent arbiter."""
    gx, gw = np.polynomial.legendre.leggauss(n)
    t, w = 0.5 * D * gx, 0.5 * D * gw
    grid = [
        (np.array([p, q, s]), wp * wq * ws)
        for p, wp in zip(t, w, strict=True)
        for q, wq in zip(t, w, strict=True)
        for s, ws in zip(t, w, strict=True)
    ]

    def phi(alpha: int, s: np.ndarray) -> np.ndarray:
        v = np.zeros(3)
        for c, e, dirn in basis_terms(alpha):
            v[dirn] += c * (s[0] ** e[0]) * (s[1] ** e[1]) * (s[2] ** e[2])
        return v

    out = np.zeros((9, 9), dtype=complex)
    for s, ws in grid:
        pa = np.array([phi(al, s) for al in range(9)])
        for u, wu in grid:
            g = elastodynamic_greens(r_vec + s - u, OMEGA, REF)
            pb = np.array([phi(be, u) for be in range(9)])
            out += ws * wu * (pa @ g @ pb.T)
    return out


def main() -> int:
    print("=" * 84)
    print("GATE -- the 9x9 Bubnov-Galerkin inter-cell propagator")
    print(f"  cell side d = {D} m, omega = {OMEGA} rad/s")
    print("=" * 84)

    # ---- [G1] autocorrelations vs the derivation's worked examples ---------
    print("\n  [G1] autocorrelation kernels vs the derivation's closed forms")
    rng = np.random.default_rng(20260915)
    u = rng.uniform(-2 * A, 2 * A, size=(400, 3))
    b = A - 0.5 * np.abs(u)
    g1 = 0.0
    want = np.prod(2.0 * A - np.abs(u), axis=-1)
    got = autocorrelation((0, 0, 0), (0, 0, 0), u, A)
    g1 = max(g1, float(np.abs(got - want).max() / np.abs(want).max()))
    print(f"       constant x constant  (tent)            {np.abs(got - want).max():.2e}")
    want = -u[:, 1] * b[:, 1] * (2 * b[:, 0]) * (2 * b[:, 2])
    got = autocorrelation((0, 0, 0), (0, 1, 0), u, A)
    g1 = max(g1, float(np.abs(got - want).max() / np.abs(want).max()))
    print(f"       displacement x axial strain            {np.abs(got - want).max():.2e}")
    want = (2 * b[:, 1] ** 3 / 3 - u[:, 1] ** 2 * b[:, 1] / 2) * (2 * b[:, 0]) * (2 * b[:, 2])
    got = autocorrelation((0, 1, 0), (0, 1, 0), u, A)
    g1 = max(g1, float(np.abs(got - want).max() / np.abs(want).max()))
    print(f"       axial strain x axial strain            {np.abs(got - want).max():.2e}")

    # ---- [G2] the CM reduction vs a direct 6-D quadrature -------------------
    print("\n  [G2] centre-of-mass reduction vs direct 6-D quadrature (non-touching)")
    g2 = 0.0
    for r_vec in (np.array([0.0, 2.0 * D, 0.0]), np.array([0.0, 2.0 * D, 1.0 * D])):
        red = galerkin_block_9x9(r_vec, D, OMEGA, REF, n_quad=12)
        ref6 = _direct_6d(r_vec, 8)
        rel = float(np.abs(red - ref6).max() / np.abs(ref6).max())
        g2 = max(g2, rel)
        print(f"       R = {str(tuple(r_vec)):>22}   rel diff {rel:.3e}")

    # ---- [G3] quadrature convergence at contact -----------------------------
    print("\n  [G3] convergence at FACE CONTACT (arbiter of [G2] not valid here)")
    print(f"       {'n_quad':>7} {'|Gamma|':>13} {'change':>11}")
    r_face = np.array([0.0, D, 0.0])
    prev, g3 = None, 1.0
    for n in (6, 9, 12, 16):
        blk = galerkin_block_9x9(r_face, D, OMEGA, REF, n_quad=n)
        ch = ""
        if prev is not None:
            g3 = float(np.abs(blk - prev).max() / np.abs(blk).max())
            ch = f"{g3:11.2e}"
        print(f"       {n:7d} {np.abs(blk).max():13.5e} {ch:>11}")
        prev = blk

    # ---- [G4] reciprocity under the project's 9-component metric ------------
    print("\n  [G4] reciprocity:  Gamma(-R) = Gamma(R)^T  (Bubnov -- no metric)")
    g4 = 0.0
    for r_vec in (
        np.array([0.0, 2.0 * D, 0.0]),
        np.array([D, 2.0 * D, 0.0]),
        np.array([0.0, D, 0.0]),  # face contact -- the hard case
    ):
        fwd = galerkin_block_9x9(r_vec, D, OMEGA, REF, n_quad=12)
        bwd = galerkin_block_9x9(-r_vec, D, OMEGA, REF, n_quad=12)
        rel = float(np.abs(bwd - fwd.T).max() / np.abs(fwd).max())
        g4 = max(g4, rel)
        print(f"       R = {str(tuple(r_vec)):>22}   rel {rel:.3e}")

    # ---- [G5] it must actually differ from the moment operator --------------
    print("\n  [G5] Galerkin vs the moment operator -- more than a per-block scale?")
    r_vec = np.array([0.0, 2.0 * D, 0.0])
    gal = galerkin_block_9x9(r_vec, D, OMEGA, REF, n_quad=12)
    mom = _cell_averaged_propagator(r_vec, D, OMEGA, REF, 8)
    print(f"       {'block':>6} {'best scalar':>13} {'resid after fit':>17}")
    g5 = 0.0
    for name, rr, cc in (
        ("G", slice(0, 3), slice(0, 3)),
        ("C", slice(0, 3), slice(3, 9)),
        ("H", slice(3, 9), slice(0, 3)),
        ("S", slice(3, 9), slice(3, 9)),
    ):
        a_, b_ = gal[rr, cc].ravel(), mom[rr, cc].ravel()
        c = complex(np.vdot(a_, b_) / np.vdot(a_, a_))
        resid = float(np.abs(a_ * c - b_).max() / max(np.abs(b_).max(), 1e-300))
        if name != "G":
            g5 = max(g5, resid)
        print(f"       {name:>6} {abs(c):13.5f} {resid:17.3e}")

    print("\n" + "=" * 84)
    print(f"  [G1] autocorrelations        : {g1:.2e}")
    print(f"  [G2] vs direct 6-D quadrature: {g2:.2e}")
    print(f"  [G3] contact convergence     : {g3:.2e}")
    print(f"  [G4] reciprocity             : {g4:.2e}")
    print(f"  [G5] differs from the moment operator: {g5:.2e}")
    ok = g1 < 1e-12 and g2 < 1e-4 and g3 < 1e-4 and g4 < 1e-12 and g5 > 1e-3
    if ok:
        print("\n  PASS: the Galerkin coupling is built and independently checked.")
        print("  [G2] validates the centre-of-mass reduction against brute force;")
        print("  [G3] shows the apex-pyramid rule handles the contact singularity;")
        print("  [G4] is an invariant the construction does not impose; and [G5]")
        print("  confirms there is a real difference to fix. Next: use it in the")
        print("  lattice and test whether the refinement ladder resumes converging.")
    else:
        print("\n  FAIL:", end=" ")
        if g1 >= 1e-12:
            print("autocorrelations wrong -- nothing downstream is meaningful.", end=" ")
        if g2 >= 1e-4:
            print("the CM reduction disagrees with brute force.", end=" ")
        if g3 >= 1e-4:
            print("contact quadrature not converged.", end=" ")
        if g4 >= 1e-12:
            print("reciprocity violated.", end=" ")
        if g5 <= 1e-3:
            print("no difference from the moment operator -- premise dead.", end=" ")
        print()
    print("=" * 84)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
