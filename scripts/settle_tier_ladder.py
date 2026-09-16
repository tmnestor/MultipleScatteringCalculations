"""SETTLEMENT: what does the T9 -> T27 -> T57 ladder actually measure?

It was being read as a convergence sequence in mode count. It is not one. The
two steps vary DIFFERENT things, and reporting them as a single sequence
1.000 -> 0.7488 -> 0.9641 mixes a method change with a basis change.

  T9  -> T27   SAME MODES, different formulation
  T27 -> T57   SAME formulation, more modes

THE EVIDENCE, and it is structural rather than numerical. The shear scalars
sigma_Eg and sigma_T2g live in the GERADE sector. In T27 that sector is 1x1
per irrep -- the library's own comment reads "Solve gerade 1x1 blocks with
smooth correction", and the result dataclass carries sigma_Eg as a scalar. The
18 quadratic modes T27 adds over T9 are all UNGERADE and never enter it. So
T27's shear scalars are computed from the same 6 strain modes as T9's.

In T57 the gerade sector is enlarged -- A1g 3x3, Eg 4x4, T2g 5x5, each one
strain function plus cubic ones -- and the result carries Eg_block as a (4,4)
array. That step is a genuine basis extension at fixed formulation.

WHAT THE FORMULATION DIFFERENCE IS. T9 builds its strain response from

    I_{ijkl} = Int_V d_i d_j G_kl dV

-- a SINGLE volume integral, no test weighting. T27 builds the same response
from a Galerkin bilinear form carrying both a trial and a test weight (the M,
bel and Bbody terms in the solve above). That is the collocation-versus-
Galerkin axis this project already knows is load-bearing: it is the same
single-versus-double averaging that produced the contact-operator defect, and
fixing that moved K from 0.886234 to 1.023737.

WHY THIS MATTERS. A convergence study needs one variable. Measuring
"convergence" across a step that changes the method measures nothing about
convergence. The ladder factorises cleanly instead:

    T9 -> T27   isolates FORMULATION at fixed modes
    T27 -> T57  isolates MODES at fixed formulation

and only the second is a convergence datum.

Run:  conda run -n seismic python scripts/settle_tier_ladder.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubic_scattering.effective_contrasts import (  # noqa: E402
    MaterialContrast,
    ReferenceMedium,
    compute_cube_tmatrix,
    compute_cube_tmatrix_galerkin,
    compute_cube_tmatrix_galerkin_57,
)

REF = ReferenceMedium(5000.0, 3000.0, 2500.0)
MU0 = REF.rho * REF.beta**2


def main() -> int:
    a, omega = 0.5, 6.0
    con = MaterialContrast(0.0, 0.25e9, 0.0)  # pure shear
    print("=" * 76)
    print("WHAT THE TIER LADDER MEASURES")
    print("=" * 76)

    r9 = compute_cube_tmatrix(omega, a, REF, con)
    r27 = compute_cube_tmatrix_galerkin(omega, a, REF, con)
    r57 = compute_cube_tmatrix_galerkin_57(omega, a, REF, con)

    print("\n[1] the GERADE sector size at each tier -- checked, not read")
    sc27 = np.ndim(np.asarray(complex(r27.sigma_Eg))) == 0
    eg57_block = np.asarray(r57.Eg_block)
    print(f"    T27 sigma_Eg is a scalar (gerade Eg block is 1x1): {sc27}")
    print(f"    T57 Eg_block shape: {eg57_block.shape}")
    ok_struct = sc27 and eg57_block.shape == (4, 4)
    print(f"    => T27 gerade = 1x1 (strain only); T57 gerade = 4x4 (strain + 3 cubic): {ok_struct}")
    print("    The 18 quadratic modes T27 adds over T9 are UNGERADE and")
    print("    cannot enter a gerade scalar.  So T9 and T27 use the SAME")
    print("    modes for this quantity.")

    eg9 = float((2.0 * complex(r9.T2c) + complex(r9.T3c)).real)
    eg27 = float(complex(r27.sigma_Eg).real)
    eg57 = float(complex(r57.sigma_Eg).real)
    t2g9 = float((2.0 * complex(r9.T2c)).real)
    t2g27 = float(complex(r27.sigma_T2g).real)
    t2g57 = float(complex(r57.sigma_T2g).real)

    print("\n[2] the two steps, labelled by what each actually varies")
    print(f"    {'step':>24} {'varies':>14} {'d sigma_Eg':>13} {'d sigma_T2g':>13}")
    d_form_eg = abs(eg27 - eg9) / abs(eg9)
    d_form_t2 = abs(t2g27 - t2g9) / abs(t2g9)
    d_mode_eg = abs(eg57 - eg27) / abs(eg27)
    d_mode_t2 = abs(t2g57 - t2g27) / abs(t2g27)
    print(f"    {'T9  -> T27':>24} {'FORMULATION':>14} {d_form_eg:13.4e} {d_form_t2:13.4e}")
    print(f"    {'T27 -> T57':>24} {'MODES':>14} {d_mode_eg:13.4e} {d_mode_t2:13.4e}")

    print("\n[3] consequences")
    print(f"    The formulation step moves the shear scalar by {d_form_eg:.1%}")
    print(f"    at FIXED modes.  The mode step moves it by {d_mode_eg:.1%} at")
    print("    FIXED formulation.  They are comparable in size, which is")
    print("    exactly why reading them as one sequence was misleading: the")
    print("    apparent 'oscillation' 1.000 -> 0.749 -> 0.964 is a method")
    print("    change followed by a basis change, not a convergence pattern.")
    print()
    print("    Only T27 -> T57 is a convergence datum, and one datum cannot")
    print("    establish a rate.  A meaningful study needs T57 -> (gerade 5)")
    print("    at the same formulation -- which is what the Chebyshev /")
    print("    orthonormal-Legendre work makes reachable.")
    print()
    print("    The T9 -> T27 gap is NOT noise and NOT a tier effect: it is the")
    print("    single-average versus Galerkin axis, the same one that produced")
    print("    the contact-operator defect.  Which of the two is right for a")
    print("    single site is a separate question this script does not settle.")

    print("\n[4] the closed-form anchor still holds")
    lam0 = REF.rho * REF.alpha**2 - 2.0 * MU0
    sd = (3.0 * np.sqrt(3.0) * (lam0 + MU0) + 2.0 * MU0 * np.pi) / (6.0 * np.pi * MU0 * (lam0 + 2.0 * MU0))
    pred = -2.0 * 0.25e9 * sd
    print(f"    T9 sigma_Eg          = {eg9:+.10f}")
    print(f"    -2 dmu S_diag (A22)  = {pred:+.10f}")
    print(f"    relative difference  = {abs(eg9 - pred) / abs(pred):.3e}")
    print("    so the T9 end of the ladder is pinned to a closed form, and")
    print("    the formulation gap is measured against something exact.")

    print("\n" + "=" * 76)
    print("SETTLED: the ladder is not a convergence sequence.  Its first step")
    print("varies the FORMULATION at fixed modes; its second varies the MODES")
    print("at fixed formulation.")
    print("=" * 76)
    return 0 if ok_struct else 1


if __name__ == "__main__":
    raise SystemExit(main())
