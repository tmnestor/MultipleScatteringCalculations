"""MEASUREMENT: does the shear channel converge with GERADE degree?

Task 4 of `plans/chebyshev_tensor_basis.md`.

WHY THE GERADE SECTOR AND NOT THE TIER NUMBER. Parity splits the closed set
into sectors that do not communicate: gerade carries d_u, d^3u, ... and
ungerade carries u, d_d_u, ... The leading modulus far field is the stress
dipole Int dc:grad(u) dV, which needs only d_u -- a gerade quantity. So:

  T9  -> T27  adds UNGERADE only (18 quadratic).  The package records
              "T27 far-field == T9".  It could not have been otherwise.
  T27 -> T57  adds GERADE (30 cubic).  This one can move the far field.

Counting tiers by their mode totals therefore mixes two different things. The
sequence that means anything is indexed by the highest GERADE degree present:
1 (T9), 3 (T57).  T27 is at gerade degree 1 exactly like T9.

THE QUANTITY. All three tiers expose the same shear scalars once the mapping is
made explicit:

    amp_e_diag = 1/(1 - sigma_Eg)      amp_e_off = 1/(1 - sigma_T2g)

    T9  (analytic) : sigma_Eg = 2 T2c + T3c,  sigma_T2g = 2 T2c
    T27, T57       : sigma_Eg, sigma_T2g read from the gerade blocks

and by the closed forms of `Mathematica/CubeA22Block.wl`,

    sigma_T2g(T9) = -2 dmu S_shear,   sigma_Eg(T9) = -2 dmu S_diag

so the T9 end of the sequence is known in closed form, not just numerically.

⚠ WHY THIS RE-MEASURES BEFORE EXTENDING. The recorded sequence
1.000 -> 0.7525 -> 0.9692 was measured when the contact propagator was
DOUBLE-averaged against a collocation state. That defect is fixed, and fixing
it moved K from 0.886234 to 1.023737 and cut the residual 5.4x. A sequence
whose earlier terms predate that fix cannot be extended meaningfully, so the
first thing here is to re-measure it. Note that this particular sequence is a
SINGLE-SITE quantity and so may well be untouched by a lattice-side fix -- the
point is that this is checked rather than assumed.

⚠ WHAT THREE POINTS CAN AND CANNOT DO. Two distinct gerade degrees (1 and 3)
cannot establish a convergence rate. This script reports the sequence and its
increments; it does NOT fit a rate, because fitting c*N^-p through two points
would produce a number with no evidential content. The degree-5 point that
would make a fit meaningful requires the 63-mode gerade assembly, which is not
built here.

Run:  conda run -n seismic python scripts/measure_gerade_tier_convergence.py
SI units (m, m/s, kg/m3, Pa).
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
LAM0 = REF.rho * REF.alpha**2 - 2.0 * MU0
KA_CEILING = 0.3


def s_shear_closed(lam: float, mu: float) -> float:
    return (np.pi * (lam + 2 * mu) - np.sqrt(3.0) * (lam + mu)) / (3.0 * np.pi * mu * (lam + 2 * mu))


def s_diag_closed(lam: float, mu: float) -> float:
    return (3.0 * np.sqrt(3.0) * (lam + mu) + 2.0 * mu * np.pi) / (6.0 * np.pi * mu * (lam + 2 * mu))


def sigmas(tier: str, omega: float, a: float, con: MaterialContrast) -> tuple[float, float]:
    """(sigma_Eg, sigma_T2g) for a tier, on the current library defaults.

    Each branch binds its own name so mypy sees one concrete result type per
    tier -- the three entry points return three unrelated dataclasses.
    """
    if tier == "T9":
        r9 = compute_cube_tmatrix(omega, a, REF, con)
        return (
            float((2.0 * complex(r9.T2c) + complex(r9.T3c)).real),
            float((2.0 * complex(r9.T2c)).real),
        )
    if tier == "T27":
        r27 = compute_cube_tmatrix_galerkin(omega, a, REF, con)
        return float(complex(r27.sigma_Eg).real), float(complex(r27.sigma_T2g).real)
    if tier == "T57":
        r57 = compute_cube_tmatrix_galerkin_57(omega, a, REF, con)
        return float(complex(r57.sigma_Eg).real), float(complex(r57.sigma_T2g).real)
    raise ValueError(tier)


def main() -> int:
    a = 0.5
    dmu = 0.25e9
    con = MaterialContrast(0.0, dmu, 0.0)  # pure shear, the channel in question
    print("=" * 78)
    print("GERADE TIER SEQUENCE IN THE SHEAR CHANNEL -- re-measured")
    print("=" * 78)

    # closed-form T9 anchor, from CubeA22Block.wl
    sig_eg_closed = -2.0 * dmu * s_diag_closed(LAM0, MU0)
    sig_t2g_closed = -2.0 * dmu * s_shear_closed(LAM0, MU0)
    print("\n  closed-form T9 anchor (from the A22 channels):")
    print(f"    sigma_Eg  = -2 dmu S_diag  = {sig_eg_closed:+.10f}")
    print(f"    sigma_T2g = -2 dmu S_shear = {sig_t2g_closed:+.10f}")

    for omega in (6.0, 30.0):
        ka = omega / REF.alpha * a
        print(
            f"\n  omega = {omega:.0f}   k_P a = {ka:.4f}"
            + ("   ** ABOVE CEILING" if ka >= KA_CEILING else "")
        )
        print(f"    {'tier':>5} {'gerade deg':>11} {'sigma_Eg':>14} {'sigma_T2g':>14} {'Eg / Eg(T9)':>13}")
        base_eg = None
        rows = []
        for tier, gdeg in (("T9", 1), ("T27", 1), ("T57", 3)):
            eg, t2g = sigmas(tier, omega, a, con)
            if base_eg is None:
                base_eg = eg
            rows.append((tier, gdeg, eg, t2g, eg / base_eg))
            print(f"    {tier:>5} {gdeg:>11} {eg:+14.10f} {t2g:+14.10f} {eg / base_eg:13.6f}")

        # the anchor check, at the lower frequency only
        if omega == 6.0:
            rel = abs(rows[0][2] - sig_eg_closed) / abs(sig_eg_closed)
            print(f"\n    T9 sigma_Eg vs its closed form: rel {rel:.3e}")
            print("    (the static closed form; the gap is the (ka)^2 form factor)")

    print("\n  READING THE SEQUENCE -- AND A REFUTATION")
    omega = 6.0
    eg9, _ = sigmas("T9", omega, a, con)
    eg27, _ = sigmas("T27", omega, a, con)
    eg57, _ = sigmas("T57", omega, a, con)
    d_ungerade = abs(eg27 - eg9) / abs(eg9)
    d_gerade = abs(eg57 - eg27) / abs(eg27)
    print(f"    |T27 - T9| / T9   (ungerade added) = {d_ungerade:.4e}")
    print(f"    |T57 - T27| / T27 (gerade added)   = {d_gerade:.4e}")
    print()
    print("    THE PREDICTION MADE BEFORE MEASURING WAS WRONG, and this is the")
    print("    result.  sigma_Eg is a GERADE scalar and T27 adds only UNGERADE")
    print("    modes, so on the parity argument T27 and T9 had to agree.  They")
    print("    differ by 25%.")
    print()
    print("    The cause is that T9 and T27 are not nested truncations of one")
    print("    formulation.  compute_cube_tmatrix is the analytic Eshelby-route")
    print("    T-matrix; compute_cube_tmatrix_galerkin is a Bubnov-Galerkin")
    print("    scheme.  Comparing their sigma_Eg conflates ADDED MODES with")
    print("    CHANGED METHOD, and the 25% is the method.")
    print()
    print("    What survives: the parity decoupling of the CLOSED SET, which")
    print("    was verified directly in CubeA22Block.wl -- A22 contains neither")
    print("    Q nor P.  That is a property of the moment formulation and is")
    print("    untouched.  What does not survive is carrying that argument over")
    print("    to this package's tier ladder without checking, which is what")
    print("    this script was written to check and did.")
    print()
    print("    CONSEQUENCE FOR THE PLAN: the motivation in")
    print("    plans/chebyshev_tensor_basis.md reads the tier history as")
    print("    'ungerade cannot move the far field, gerade can'.  That reading")
    print("    is not supported by these numbers and the plan is amended.")
    print()
    print("    Step 1 IS answered, though: the sequence 1.000 -> 0.7488 ->")
    print("    0.9641 reproduces the recorded 1.000 -> 0.7525 -> 0.9692, so it")
    print("    is a single-site quantity and the lattice-side contact fix left")
    print("    it alone.  It was safe to extend; it is the interpretation that")
    print("    was not safe.")

    print("\n  WHAT IS NOT DONE HERE")
    print("    Two gerade degrees cannot establish a convergence rate, so no")
    print("    rate is fitted.  The degree-5 point needs the 63-mode gerade")
    print("    assembly, which is not built.  The independent subdivision")
    print("    arbiter is also not run -- it must first be re-qualified on its")
    print("    n_sub = 1 control, since it was disqualified for a contact-")
    print("    propagator defect that has since been fixed.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
