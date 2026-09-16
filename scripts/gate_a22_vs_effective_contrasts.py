"""GATE: the first-gradient block A22, derived from first principles, against
the production effective contrasts.

WHY THIS GATE EXISTS. The closed set's middle equation

    (d_pr d_ij + w^2 drho N^r_ij,p - M_in,pk dc_nkrj) d_r u_j = d_p u0_i

determines the first gradient on its own -- it couples neither to u nor to the
second gradients (the parity decoupling). The leading modulus far field is the
stress dipole Int dc:grad(u) dV, which needs only d_u, so A22 IS the leading
T-matrix. Mathematica/CubeA22Block.wl decomposes it under O_h and reads off the
two shear channels in closed form:

    S_shear (T2g, off-diagonal) = (pi(l+2m) - sqrt(3)(l+m)) / (3 pi m (l+2m))
    S_diag  (Eg,  diagonal)     = (3 sqrt(3)(l+m) + 2 m pi) / (6 pi m (l+2m))

`effective_contrasts.py` computes the same physics by a completely different
route -- tabulated master integrals with an explicit Eshelby delta correction,
assembled into A^c/B^c/C^c and then T1c/T2c/T3c. Two independent
implementations agreeing is this project's evidence standard; one agreeing
with itself is not.

THE IDENTIFICATION. The production amplification factors are

    amp_e_off  = 1/(1 - 2 T2)          amp_e_diag = 1/(1 - 2 T2 - T3)

and the A22 channel eigenvalues are 1 + 2 dmu S, whose inverse is the strain
concentration. Matching gives two predictions with no free parameters:

    T2c = -dmu * S_shear
    T3c =  2 dmu * (S_shear - S_diag)

WHY IT MATTERS BEYOND A CONSISTENCY CHECK. Dmu*_diag is exactly the quantity
the empirical K = 0.886234 multiplies. If the two routes disagree, the
discrepancy is a candidate mechanism for K; if they agree, K is not a
single-site error in the diagonal shear channel and must live in the lattice.
Either outcome is worth having, which is what makes this a gate rather than a
formality.

The moments are scale-free (M has grading (D,W) = (2,0), so it carries no
factor of the cube size), hence so are S_shear and S_diag -- the comparison
does not depend on the cube half-width.
"""

from __future__ import annotations

import numpy as np
from cubic_scattering.effective_contrasts import (
    MaterialContrast,
    ReferenceMedium,
    compute_cube_tmatrix,
)

# Validated background: alpha = 5 km/s, beta = 3 km/s, rho = 2.5 g/cm^3
ALPHA, BETA, RHO = 5000.0, 3000.0, 2500.0
MU = RHO * BETA**2
LAM = RHO * ALPHA**2 - 2.0 * MU

# Moderate contrast, from the validated set
DLAM, DMU, DRHO = 2.0e9, 1.0e9, 100.0

A_HALF = 50.0
KA_TARGET = 0.05  # well inside the ka < 0.3 validity ceiling
OMEGA = KA_TARGET * ALPHA / A_HALF


def s_shear(lam: float, mu: float) -> float:
    """Off-diagonal (T2g) shear self-term, from CubeA22Block.wl."""
    return (np.pi * (lam + 2 * mu) - np.sqrt(3.0) * (lam + mu)) / (3.0 * np.pi * mu * (lam + 2 * mu))


def s_diag(lam: float, mu: float) -> float:
    """Diagonal (Eg) shear self-term, from CubeA22Block.wl."""
    return (3.0 * np.sqrt(3.0) * (lam + mu) + 2.0 * mu * np.pi) / (6.0 * np.pi * mu * (lam + 2 * mu))


def main() -> int:
    ref = ReferenceMedium(alpha=ALPHA, beta=BETA, rho=RHO)
    contrast = MaterialContrast(Dlambda=DLAM, Dmu=DMU, Drho=DRHO)
    res = compute_cube_tmatrix(OMEGA, A_HALF, ref, contrast)

    ss, sd = s_shear(LAM, MU), s_diag(LAM, MU)
    t2_pred = -DMU * ss
    t3_pred = 2.0 * DMU * (ss - sd)

    t2_got = complex(res.T2c).real
    t3_got = complex(res.T3c).real

    print("=" * 62)
    print("GATE: A22 (first principles) vs effective_contrasts.py")
    print("=" * 62)
    print(f"  lam = {LAM:.6e}   mu = {MU:.6e}   ka = {KA_TARGET}")
    print(f"  S_shear (T2g) = {ss:.12e}")
    print(f"  S_diag  (Eg)  = {sd:.12e}")
    print(f"  ratio S_shear/S_diag = {ss / sd:.9f}")
    print()
    print("  predicted from A22        production code")
    print(f"  T2c = {t2_pred:+.12e}   {t2_got:+.12e}")
    print(f"  T3c = {t3_pred:+.12e}   {t3_got:+.12e}")

    def rel(a: float, b: float) -> float:
        scale = max(abs(a), abs(b), 1e-300)
        return abs(a - b) / scale

    r2, r3 = rel(t2_pred, t2_got), rel(t3_pred, t3_got)
    print()
    print(f"  relative difference  T2c: {r2:.3e}    T3c: {r3:.3e}")

    # the amplification factors themselves, which is what the contrasts use
    amp_off_pred = 1.0 / (1.0 + 2.0 * DMU * ss)
    amp_diag_pred = 1.0 / (1.0 + 2.0 * DMU * sd)
    amp_off_got = complex(res.amp_e_off).real
    amp_diag_got = complex(res.amp_e_diag).real
    print()
    print(f"  amp_e_off   predicted {amp_off_pred:.12f}   code {amp_off_got:.12f}")
    print(f"  amp_e_diag  predicted {amp_diag_pred:.12f}   code {amp_diag_got:.12f}")
    ro = rel(amp_off_pred, amp_off_got)
    rd = rel(amp_diag_pred, amp_diag_got)
    print(f"  relative difference  off: {ro:.3e}    diag: {rd:.3e}")

    print()
    print("  For reference, the empirical shear-channel factor is K = 0.886234,")
    print("  which multiplies Dmu*_diag.  If the two routes agree here, K is")
    print("  NOT a single-site error in this channel.")
    print(f"  amp_e_diag / amp_e_off = {amp_diag_got / amp_off_got:.9f}")

    tol = 1e-6
    ok = max(r2, r3, ro, rd) < tol
    print()
    print("=" * 62)
    print(
        "PASS -- the two independent routes agree."
        if ok
        else f"FAIL -- routes disagree by more than {tol:g}."
    )
    print("=" * 62)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
