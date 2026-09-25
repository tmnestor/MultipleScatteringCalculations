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
    sum_n T_loc psi_exc[n] carries the incident plane-wave phase on its INPUT
    side and no phase on its OUTPUT side, and depends on k_hat.  It is not a
    field-independent operator, so it cannot be reciprocal.  REPORTED, not
    asserted: the adjoint must not be built on it as it stands.

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
    """Run [R] (asserted) and [C] (reported)."""
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

    oblique = np.array([0.3, 0.5, 0.81])
    for ka, khat in ((0.5, None), (1.2, oblique / np.linalg.norm(oblique))):
        omega = ka * ref.beta / a
        res_c = compute_resonance_tmatrix(omega, a, ref, con, n_sub=3, k_hat=khat)
        direction = "z-hat" if khat is None else "oblique"
        tag = f"[C] Resonance composite, n_sub = 3, ka_S = {ka}, k_hat {direction}"
        report(tag, measure(res_c.T_comp_9x9))

    print("\n[R] PASS" if ok else "\n[R] FAIL")
    print("[C] reported only: the composite is incidence-dependent and not an operator")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
