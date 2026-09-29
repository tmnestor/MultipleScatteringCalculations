#!/usr/bin/env python3
"""GATE: the reciprocity law of the 9x9 cube T0 -- the adjoint-state prerequisite.

WHY THIS GATE EXISTS
--------------------
The adjoint of the Foldy-Lax operator (I - G0 T0) is the SAME operator, taken at
-k_y and conjugated by the Voigt weight W, only if BOTH factors obey a matching
reciprocity law.  The propagator's half is measured elsewhere (W P symmetric,
gate_9x9_source_convention; the symplectic J6 law of the reverberation,
gate_sweep_dressed_equivalence [D3]).  T0's half had never been measured.  The
law it needs is

    T0^T = W T0 W^-1,   equivalently   T0 W^-1 symmetric,
    W = diag(1,1,1, 1,1,1, 1/2,1/2,1/2).

For an axis-aligned cubic cell with no normal-shear coupling, W commutes with
T0, so the law is the same as T0 = T0^T.  All three symmetries are printed,
along with the normal-shear block that decides whether they can differ.

[R] RAYLEIGH.  _sub_cell_tmatrix_9x9: block-diagonal, omega^2 Drho* V I3 (+)
    V Dc*_Voigt, with Dc*_Voigt a cubic stiffness.  Must pass to machine
    precision.  This is the T0 the directional-sweep solver carries.

[C] RESONANCE COMPOSITE.  compute_resonance_tmatrix(...).T_comp_9x9 =
    sum_n T_loc psi_exc[n], with psi_exc solved for the phase-free Taylor
    patterns about the cube centre, so it is a field-independent operator.
    Asserted to machine precision.  As first built its patterns also carried
    the incident plane-wave phase: it depended on k_hat and failed this law by
    1.6e-2 (ka_S 0.5) and 1.7e-1 (ka_S 1.2, oblique), the calibration that
    shows the check can fail.  Its output omits the first moment of the
    sub-cell forces, which this law cannot see (that term is symmetric).

Run: conda run -n seismic python scripts/gate_t0_reciprocity.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cubic_scattering.effective_contrasts import (  # noqa: E402
    MaterialContrast,
    ReferenceMedium,
    compute_cube_tmatrix,
)
from cubic_scattering.resonance_tmatrix import (  # noqa: E402
    _sub_cell_tmatrix_9x9,
    compute_resonance_tmatrix,
)

W = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
W_INV = np.linalg.inv(W)
TOL = 1e-14


def rel_asym(m: np.ndarray) -> float:
    """Relative antisymmetric part ||m - m^T|| / ||m||."""
    return float(np.linalg.norm(m - m.T) / np.linalg.norm(m))


def measure(t0: np.ndarray) -> dict[str, float]:
    """The three candidate reciprocity residuals and the normal-shear block."""
    return {
        "T0 symmetric": rel_asym(t0),
        "T0 W^-1 symmetric (required)": rel_asym(t0 @ W_INV),
        "W T0 symmetric": rel_asym(W @ t0),
        "normal-shear block": float(np.linalg.norm(t0[3:6, 6:9]) / np.linalg.norm(t0)),
    }


def report(tag: str, res: dict[str, float]) -> None:
    """Print one measurement block."""
    print(tag)
    for k, v in res.items():
        print(f"  {k:<32s} {v:.2e}")


def main() -> int:
    """Run [R] and [C], both asserted."""
    ref = ReferenceMedium(alpha=5.0, beta=3.0, rho=2.5)
    con = MaterialContrast(Dlambda=2.0, Dmu=1.0, Drho=0.1)
    a = 0.5
    ok = True

    for ka in (0.05, 0.1, 0.3):
        omega = ka * ref.beta / a
        t0 = _sub_cell_tmatrix_9x9(compute_cube_tmatrix(omega, a, ref, con), omega, a)
        res = measure(t0)
        report(f"[R] Rayleigh cube, ka_S = {ka}", res)
        ok &= res["T0 W^-1 symmetric (required)"] < TOL

    ok_c = True
    oblique = np.array([0.3, 0.5, 0.81])
    for ka, khat in ((0.5, None), (1.2, oblique / np.linalg.norm(oblique))):
        omega = ka * ref.beta / a
        res_c = compute_resonance_tmatrix(omega, a, ref, con, n_sub=3, k_hat=khat)
        direction = "z-hat" if khat is None else "oblique"
        tag = f"[C] Resonance composite, n_sub = 3, ka_S = {ka}, k_hat {direction}"
        res = measure(res_c.T_comp_9x9)
        report(tag, res)
        ok_c &= res["T0 W^-1 symmetric (required)"] < TOL

    print("\n[R] PASS" if ok else "\n[R] FAIL")
    print("[C] PASS" if ok_c else "[C] FAIL")
    return 0 if ok and ok_c else 1


if __name__ == "__main__":
    raise SystemExit(main())
