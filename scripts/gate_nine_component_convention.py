"""GATE: the 9-component source convention -- the mismatch that nothing gated.

THE LONG-STANDING GAP. The project's notes have carried this for months as an
open item with no test behind it: the 9-component SOURCE convention between the
T-matrix side and the propagator side is mismatched. It surfaced concretely when
`gate_variational_summation` could not run -- the thesis's summation chapter
needs a bilinear form in which H is self-adjoint (VariationalSum.tex, Htdef),
and ours is not, by 1.4e-3. This gate measures the convention itself instead of
its symptoms.

THE BASIS. (u_z, u_x, u_y, e_zz, e_xx, e_yy, 2e_xy, 2e_zy, 2e_zx) -- the last
three are ENGINEERING shear, carrying a factor 2. Define

    J = diag(1, 1, 1,  1, 1, 1,  1/2, 1/2, 1/2)

which is not an arbitrary fitting knob: it is the STRAIN CONTRACTION METRIC. The
physical pairing of two strains is e:e = sum_ii e_ii^2 + 2 sum_shear e_shear^2,
and in engineering components g = 2e that is exactly the form weighted by J.
Any object whose two indices are both "strain-like" must be symmetric under J.

WHAT IS MEASURED, and each is exact or it is a defect -- there is no tolerance
band in which a convention is "nearly" right:

  [C1] the closed-form propagator, the validated arbiter. J P(r) symmetric.
  [C2] the ASSEMBLED lattice G0, via a different code path
       (inter_voxel_propagator, FFT convolution). Same question.
  [C3] the cube T0. Plain symmetric, J-symmetric, or J T0 J symmetric?
  [C4] which weight W makes the Foldy-Lax operator self-adjoint, i.e. W H
       symmetric, as the thesis's bilinear form requires.

HOW G0 IS ISOLATED. Handing the solver IDENTITY T-matrices makes
H = I - G0 . I, so G0 = I - H exactly. No inversion, no extraction of a small
quantity from a large one -- the cancellation trap this project has already paid
for once.

WHAT A FAILURE MEANS. If [C1] and [C2] agree with each other and [C3] does not,
the T side is the one to change. If [C1] and [C2] disagree, the propagator has
two conventions internally and that is the more serious finding.

Run:  conda run -n seismic python scripts/gate_nine_component_convention.py
SI units (m, m/s, kg/m3, Pa).
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from cubic_scattering.horizontal_greens import exact_propagator_9x9  # noqa: E402
from cubic_scattering.slab_scattering import (  # noqa: E402
    SlabGeometry,
    SlabMaterial,
    _slab_matvec,
    build_slab_kernels,
    compute_slab_tmatrices,
)

REF = ReferenceMedium(5000.0, 3000.0, 2500.0)
OMEGA = 60.0
A_HALF, M, N_Z, PCT = 1.0, 3, 3, 0.05
EXACT = 1e-12  # a convention is exact or it is wrong; this is not a tolerance

J = np.diag(np.array([1, 1, 1, 1, 1, 1, 0.5, 0.5, 0.5], dtype=float))
# THE ASSEMBLED MATRIX NEEDS THE PARITY AS WELL AS THE METRIC, and conflating
# the two is why an exhaustive +-1 signature search found nothing earlier.
# J P(r) symmetric is a statement about ONE separation. The assembled G0 needs
# block (i,j) = P(r_ij) to match block (j,i) = P(-r_ij), i.e. M P(r) = P(-r)^T M.
# The blocks have definite parity -- G, S even (0, 2 derivatives), C, H odd (1) --
# so P(-r) = Sigma P(r) Sigma with Sigma = diag(I3, -I6). Combining that with
# P^T = J P J^-1 from [C1] gives M = Sigma J, and nothing simpler works:
# a pure sign pattern cannot supply the 1/2, and J alone cannot supply the parity.
SIGMA = np.diag(np.array([1, 1, 1, -1, -1, -1, -1, -1, -1], dtype=float))
MJ = SIGMA @ J  # diag(1,1,1, -1,-1,-1, -.5,-.5,-.5)


def _asym(a: np.ndarray) -> float:
    """Relative departure from plain symmetry."""
    scale = np.abs(a).max()
    return float(np.abs(a - a.T).max() / scale) if scale > 0 else 0.0


def _kron(metric: np.ndarray, n_sites: int) -> np.ndarray:
    return np.kron(np.eye(n_sites), metric)


def main() -> int:
    print("=" * 86)
    print("GATE -- the 9-component source convention")
    print("  J = diag(1,1,1, 1,1,1, .5,.5,.5) is the STRAIN CONTRACTION METRIC,")
    print("  not a fitted knob: it undoes the factor 2 of engineering shear.")
    print("=" * 86)

    # ---- [C1] the validated closed-form arbiter ---------------------------
    print("\n  [C1] closed-form propagator exact_propagator_9x9")
    print(f"       {'separation (z,x,y)':>20} {'plain P':>12} {'J P':>12}")
    c1_plain, c1_j = [], []
    for r in [(2.0, 0.0, 0.0), (2.0, 2.0, 0.0), (2.0, 4.0, 6.0), (4.0, -2.0, 2.0)]:
        z, x, y = r
        p = exact_propagator_9x9(x, y, z, OMEGA + 0.0j, REF)
        a_plain, a_j = _asym(p), _asym(J @ p)
        c1_plain.append(a_plain)
        c1_j.append(a_j)
        print(f"       {str(r):>20} {a_plain:12.3e} {a_j:12.3e}")

    # ---- build the lattice operators --------------------------------------
    geom = SlabGeometry(M=M, N_z=N_Z, a=A_HALF)
    lam0 = REF.rho * (REF.alpha**2 - 2.0 * REF.beta**2)
    mu0 = REF.rho * REF.beta**2
    ones = np.ones((N_Z, M, M))
    material = SlabMaterial(
        Dlambda=PCT * lam0 * ones, Dmu=PCT * mu0 * ones, Drho=PCT * REF.rho * ones, ref=REF
    )
    t0 = compute_slab_tmatrices(geom, material, OMEGA)
    kh = build_slab_kernels(geom, OMEGA, REF)
    n_sites = N_Z * M * M
    n = n_sites * 9
    jn, mn = _kron(J, n_sites), _kron(MJ, n_sites)

    def _dense(t_local: np.ndarray) -> np.ndarray:
        mat = np.zeros((n, n), dtype=complex)
        e = np.zeros(n, dtype=complex)
        for j in range(n):
            e[j] = 1.0
            mat[:, j] = _slab_matvec(e, t_local, kh, geom)
            e[j] = 0.0
        return mat

    # ---- [C2] the assembled lattice G0, isolated with identity T ----------
    t_id = np.tile(np.eye(9, dtype=complex), (N_Z, M, M, 1, 1))
    g0 = np.eye(n) - _dense(t_id)
    print("\n  [C2] assembled lattice G0 (identity T0, so G0 = I - H exactly)")
    print(f"       plain G0      : {_asym(g0):.3e}")
    print(f"       J G0          : {_asym(jn @ g0):.3e}")
    print(f"       (Sigma J) G0  : {_asym(mn @ g0):.3e}   <-- the predicted metric")
    c2 = _asym(mn @ g0)

    # ---- [C3] the cube T0 -------------------------------------------------
    w = np.zeros((n, n), dtype=complex)
    tf = t0.reshape(-1, 9, 9)
    for s in range(tf.shape[0]):
        w[s * 9 : (s + 1) * 9, s * 9 : (s + 1) * 9] = tf[s]
    print("\n  [C3] cube T0")
    print(f"       plain T0   : {_asym(w):.3e}")
    print(f"       J T0       : {_asym(jn @ w):.3e}")
    print(f"       J T0 J     : {_asym(jn @ w @ jn):.3e}")

    # ---- [C4] which weight makes the Foldy-Lax operator self-adjoint? -----
    h = _dense(t0)
    print("\n  [C4] W H symmetry -- what the thesis's bilinear form requires")
    cands = {
        "W = T0 (as used)": w,
        "W = J T0": jn @ w,
        "W = T0 J": w @ jn,
        "W = J T0 J": jn @ w @ jn,
        "W = (Sigma J) T0": mn @ w,
        "W = T0 (Sigma J)": w @ mn,
        "W = (Sigma J) T0 (Sigma J)": mn @ w @ mn,
    }
    best_name, best_val = None, 1.0
    for name, ww in cands.items():
        a = _asym(ww @ h)
        print(f"       {name:18s} : {a:.3e}")
        if a < best_val:
            best_name, best_val = name, a

    # ---- verdict ----------------------------------------------------------
    print("\n" + "=" * 86)
    prop_ok = max(c1_j) < EXACT and max(c1_plain) > EXACT
    lattice_ok = c2 < EXACT and _asym(g0) > EXACT
    t_plain = _asym(w) < EXACT
    ok = best_val < EXACT
    if prop_ok and lattice_ok and t_plain and not ok:
        print("  MISMATCH LOCATED, and it is on the T SIDE.")
        print("  Both propagators -- the closed form AND the assembled lattice, which")
        print("  share no code path -- are self-adjoint under J and NOT under the")
        print("  identity. They agree with each other and with the physical strain")
        print("  metric. T0 is self-adjoint under the IDENTITY instead, so T0 is the")
        print("  object out of convention, and no weight built from it repairs the")
        print(f"  Foldy-Lax operator (best: {best_name} at {best_val:.3e}).")
        print("  FIX: carry the engineering-shear halving into the T-matrix assembly,")
        print("  then re-run; [C4] must reach machine precision before the thesis's")
        print("  summation scheme can be tested at all.")
    elif prop_ok and not lattice_ok:
        print("  THE TWO PROPAGATOR PATHS DISAGREE. The closed form is J-symmetric")
        print("  but the assembled lattice is not, so the defect is in the lattice")
        print("  assembly, not in the T-matrix. That is the more serious finding and")
        print("  it must be settled before anything downstream is believed.")
    elif ok:
        print(f"  CONSISTENT: {best_name} makes the Foldy-Lax operator self-adjoint")
        print(f"  at {best_val:.3e}. The thesis's bilinear form is well defined and")
        print("  the variational summation can be tested on this operator.")
    else:
        print("  UNRESOLVED: the pattern matches none of the anticipated cases.")
        print("  Report the numbers and claim nothing.")
    print("=" * 86)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
