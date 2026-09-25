#!/usr/bin/env python3
"""Mode-converted reciprocity of the exact Mie far field: P->SV against SV->P.

WHY THIS EXISTS
---------------
``LatexPDFs/psvsh_reciprocity.tex`` Eq. (4) states the far-field reciprocity
relation for elastic scattering amplitudes (Varatharajulu 1977),

    k_in^2  q.f_{out<-in}(x; d, p)  =  k_out^2  p.f_{in<-out}(-d; -x, q),

whose P<->S off-diagonal carries the weight (k_S/k_P)^2 = (alpha/beta)^2.  The
note derived it from Green's-tensor symmetry alone.  This gate scores it against
the exact elastic Mie solution, where the P- and SV-incident branches are solved
from different right-hand sides and assembled by different far-field code
(m = 0 Legendre functions versus the m = 1 renormalised N-type expansion).  A
normalisation fault in either branch breaks the relation.

THE GEOMETRY, AND WHERE THE MINUS SIGN COMES FROM
-------------------------------------------------
Forward: P incident along +z, SV observed at polar angle theta in the xz-plane
(phi = 0), projected on theta-hat.  That is ``mie_far_field(.., "P")[1]``.

Reverse: S incident along -x_hat(theta) polarised along theta-hat, P observed
along -z, projected on the forward polarisation z_hat.  Rotating about y by
pi - theta carries the reverse incidence onto +z with polarisation -x_hat, and
the observation direction onto polar angle theta at phi = pi.  For an
x_hat-polarised SV wave the radial P field goes as cos(phi), so the two sign
flips cancel and the rotated amplitude is ``mie_far_field(.., "SV")[0]``.  The
projection z_hat . (radially outgoing P along -z) contributes the one surviving
minus sign.  Hence, for a sphere,

    k_P^2 f_{P->SV}(theta)  =  - k_S^2 f_{SV->P}(theta).

TOLERANCE
---------
``mie_far_field`` evaluates the outgoing field at r = 1e6 a, not at infinity, so
each amplitude carries an O(n^2 / (k r)) near-field residue.  At ka_S = 0.1 that
is ~1e-4 relative to the peak amplitude.  The tolerance TOL sits an order above
it, and the negative controls below fail by O(1), so the margin between pass
and fail is three orders of magnitude.

NEGATIVE CONTROLS
-----------------
  - Unweighted symmetry f_{P->SV} = -f_{SV->P} (the scalar-wave form) must FAIL:
    it is wrong by the factor (alpha/beta)^2.
  - The opposite sign must FAIL.
If either control passed, the check would not be discriminating.

Run:  conda run -n seismic python scripts/gate_mie_psv_reciprocity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubic_scattering.effective_contrasts import (  # noqa: E402
    MaterialContrast,
    ReferenceMedium,
)
from cubic_scattering.sphere_scattering import (  # noqa: E402
    compute_elastic_mie,
    mie_far_field,
)

REF = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
CONTRASTS = {
    "moderate (dlam=+2, dmu=+1 GPa, drho=+100)": MaterialContrast(Dlambda=2.0e9, Dmu=1.0e9, Drho=100.0),
    "density only (drho=+250)": MaterialContrast(Dlambda=0.0, Dmu=0.0, Drho=250.0),
    "strong soft (dlam=-8, dmu=-10 GPa, drho=-500)": MaterialContrast(
        Dlambda=-8.0e9, Dmu=-10.0e9, Drho=-500.0
    ),
}
KA_S = (0.1, 0.5, 1.5, 4.0)
RADIUS = 1.0
THETA = np.linspace(0.05, np.pi - 0.05, 60)
TOL = 1.0e-3
CONTROL_FLOOR = 0.3

_PASS: list[tuple[str, bool]] = []


def report(label: str, ok: bool) -> None:
    """Record and print one check.

    Args:
        label: Human-readable description of the check.
        ok: Whether the check passed.
    """
    _PASS.append((label, bool(ok)))
    print(f"  {'PASS' if ok else '****FAIL****'}  {label}")


def peak_rel(a: NDArray, b: NDArray) -> float:
    """Max |a - b| over angle, relative to the peak |b|.

    Normalising by the peak rather than pointwise keeps angles near a zero of
    the amplitude from dominating the error.

    Args:
        a: Left-hand side, shape (M,).
        b: Right-hand side, shape (M,).

    Returns:
        The peak-normalised maximum deviation.
    """
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def amplitudes(contrast: MaterialContrast, ka_s: float) -> tuple[NDArray, NDArray, float, float]:
    """Far-field P->SV and SV->P amplitudes on the THETA grid.

    Args:
        contrast: Sphere material contrast.
        ka_s: Dimensionless S-wave frequency k_S a.

    Returns:
        (f_P->SV, f_SV->P, k_P, k_S).
    """
    omega = ka_s * REF.beta / RADIUS
    mie = compute_elastic_mie(omega, RADIUS, REF, contrast)
    _, f_p_to_sv, _ = mie_far_field(mie, THETA, "P")
    f_sv_to_p, _, _ = mie_far_field(mie, THETA, "SV")
    return f_p_to_sv, f_sv_to_p, omega / REF.alpha, omega / REF.beta


def main() -> int:
    """Run the reciprocity check and its controls over contrasts and frequencies.

    Returns:
        Process exit code: 0 if every check passed, 1 otherwise.
    """
    print("Mie P->SV versus SV->P reciprocity  [psvsh_reciprocity.tex Eq. (4)]")
    print(f"  alpha/beta = {REF.alpha / REF.beta:.4f},  (alpha/beta)^2 = {(REF.alpha / REF.beta) ** 2:.4f}")
    for name, contrast in CONTRASTS.items():
        print(f"\n  contrast: {name}")
        for ka_s in KA_S:
            f_psv, f_svp, k_p, k_s = amplitudes(contrast, ka_s)
            lhs = k_p**2 * f_psv
            rhs = -(k_s**2) * f_svp

            err = peak_rel(lhs, rhs)
            err_unweighted = peak_rel(f_psv, -f_svp)
            err_wrong_sign = peak_rel(lhs, -rhs)
            print(
                f"    ka_S={ka_s:<4}  weighted err={err:.2e}   "
                f"controls: unweighted={err_unweighted:.2f}, wrong sign={err_wrong_sign:.2f}"
            )
            report(f"{name}, ka_S={ka_s}: k_P^2 f_PSV = -k_S^2 f_SVP  (err < {TOL:g})", err < TOL)
            report(
                f"{name}, ka_S={ka_s}: controls fail (> {CONTROL_FLOOR})",
                err_unweighted > CONTROL_FLOOR and err_wrong_sign > CONTROL_FLOOR,
            )

    print()
    ok = sum(1 for _, p in _PASS if p)
    for label, passed in _PASS:
        if not passed:
            print(f"  FAILED: {label}")
    print(f"  {ok}/{len(_PASS)} checks passed")
    return 0 if ok == len(_PASS) else 1


if __name__ == "__main__":
    sys.exit(main())
