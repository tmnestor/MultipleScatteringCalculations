#!/usr/bin/env python3
"""GATE A2: the derivative of the cube T-matrix with respect to its contrast.

WHY THIS GATE EXISTS
--------------------
The voxel gradient of note (IV), AdjointStateGradients, contracts the adjoint
and forward fields through dT0/dm_a, a in {Dlambda, Dmu, Drho}. T0 is the
Rayleigh cube (resonance_tmatrix._sub_cell_tmatrix_9x9 of compute_cube_tmatrix),
nonlinear in m through the Eshelby amplification factors and the contrast-
dependent form factors. No analytic derivative of it exists in the repository.

THE NOTE'S PLANNED ARBITER WAS WRONG
------------------------------------
The note named a complex-step derivative. Complex step, and its generalisation
the Cauchy contour integral, require T0 to be HOLOMORPHIC in m. It is not:
compute_cube_tmatrix applies its real form factors to the REAL parts of the
effective contrasts only (Dmu*.real * ff_mu + 1j * Dmu*.imag, and likewise for
kappa and rho), so the map m -> T0 has no complex derivative. [H] measures this.
Every contrast-dependent ingredient is otherwise a smooth rational function
(amplification factors 1/(1 - ...), linear effective contrasts, rational c2/c4
form-factor coefficients), so T0 IS smooth as a map of REAL m, away from the
resonance poles of the amplification factors. The derivative is therefore taken
by Richardson-extrapolated central differences along the real axis: three
parameters per voxel type, so it costs nine T0 evaluations and nothing else.

CHECKS
------
  [R]   Richardson central differences at finite contrast, for each parameter:
        steps h, h/2, h/4. The observed order of the raw differences must be ~2
        (a smooth function) -- unless the channel is linear to rounding (the
        density channel at small ka), where there is no truncation error to
        measure and the order is reported as "linear", not asserted. The
        extrapolated derivative's self-change is the error estimate, asserted
        < 1e-8 relative.
  [Lin] Directional consistency: the derivative along a random real direction u
        must equal sum_a u_a dT0/dm_a. A kink or branch would break it.
        Asserted < 1e-8.
  [B]   Born limit, the ABSOLUTE check: at zero contrast the amplification
        factors are exactly 1 and the form factors tend to 1 as ka -> 0, so
                dT0/dDrho     -> omega^2 V I3 (+) 0
                dT0/dDlambda  -> 0 (+) V C_Voigt(1, 0, 0)
                dT0/dDmu      -> 0 (+) V C_Voigt(0, 1, 1)
        with O((ka)^2) error from the form factor. Reported as convergence in
        ka with the fitted slope, not as a tolerance; asserted slope in [1.8, 2.2].
  [H]   CALIBRATION / FINDING: the derivative along an IMAGINARY step disagrees
        with the real one (T0 is not holomorphic), so complex step would be
        wrong. The disagreement grows as (ka)^2 -- ~5e-4 at ka = 0.05, ~2e-2 at
        0.3 -- so it is judged against the FD error it would replace: asserted
        to exceed it by 1e3 (the error floored at 1e-8). A first version used a
        fixed 1e-3 and failed at ka = 0.05 for that reason, not for any defect.

Test parameters are the project's validated set: alpha = 5, beta = 3, rho = 2.5,
contrast (2 GPa, 1 GPa, 0.1 g/cm3), cube half-width a = 0.5 km; both a real
frequency and the complex frequency the sweep-solver gates use.

Run:  conda run -n seismic python scripts/gate_a2_t0_derivative.py
Seismic units (km/s, g/cm3, GPa), time convention e^{-i omega t}.
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
from cubic_scattering.resonance_tmatrix import _sub_cell_tmatrix_9x9  # noqa: E402
from cubic_scattering.voigt_tmatrix import effective_stiffness_voigt  # noqa: E402

REF = ReferenceMedium(alpha=5.0, beta=3.0, rho=2.5)
A_HALF = 0.5
M0 = np.array([2.0, 1.0, 0.1])  # (Dlambda, Dmu, Drho)
SCALE = np.array([1.0, 1.0, 0.1])  # step scale per parameter
NAMES = ("Dlambda", "Dmu", "Drho")
TOL = 1e-8
ROUNDING_FLOOR = 1e-10  # below this relative change, raw differences are rounding
H_MARGIN = 1e3  # [H] must exceed the FD error (floored at TOL) by this factor


def t0(m: np.ndarray, omega: complex) -> np.ndarray:
    """The 9x9 Rayleigh cube T0 at contrast m = (Dlambda, Dmu, Drho)."""
    res = compute_cube_tmatrix(omega, A_HALF, REF, MaterialContrast(Dlambda=m[0], Dmu=m[1], Drho=m[2]))
    return _sub_cell_tmatrix_9x9(res, omega, A_HALF)


def central(m: np.ndarray, u: np.ndarray, h: float, omega: complex) -> np.ndarray:
    """Central difference of T0 along direction u with step h."""
    return (t0(m + h * u, omega) - t0(m - h * u, omega)) / (2.0 * h)


def richardson(m: np.ndarray, u: np.ndarray, h: float, omega: complex):
    """Raw differences at h, h/2, h/4; two extrapolations; observed order; error estimate."""
    d = [central(m, u, h / 2**j, omega) for j in range(3)]
    r1 = (4.0 * d[1] - d[0]) / 3.0
    r2 = (4.0 * d[2] - d[1]) / 3.0
    err = float(np.linalg.norm(r2 - r1) / np.linalg.norm(r2))
    # The observed order is meaningful only while the change between raw
    # differences is above rounding. A channel that is linear to rounding (the
    # density channel at small ka) has no truncation error to measure; its
    # order is then noise and is returned as nan rather than asserted.
    change = np.linalg.norm(d[1] - d[2]) / np.linalg.norm(d[2])
    if change < ROUNDING_FLOOR:
        return r2, float("nan"), err
    order = float(np.log2(np.linalg.norm(d[0] - d[1]) / np.linalg.norm(d[1] - d[2])))
    return r2, order, err


def rel(a: np.ndarray, b: np.ndarray) -> float:
    """||a - b|| / ||a||."""
    return float(np.linalg.norm(a - b) / np.linalg.norm(a))


def born_weights(omega: complex) -> list[np.ndarray]:
    """The zero-contrast, ka -> 0 derivatives: omega^2 V I3 and V C_Voigt."""
    vol = (2.0 * A_HALF) ** 3
    out = []
    for dl, dm, dr in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
        w = np.zeros((9, 9), dtype=complex)
        w[:3, :3] = omega**2 * dr * vol * np.eye(3)
        w[3:, 3:] = vol * effective_stiffness_voigt(dl, dm, dm)
        out.append(w)
    return out


def main() -> int:
    """[R], [Lin], [H] at finite contrast; [B] at zero contrast."""
    print("=" * 88)
    print("GATE A2 -- dT0/dm for the Rayleigh cube, m = (Dlambda, Dmu, Drho)")
    print(f"  ref (alpha, beta, rho) = ({REF.alpha}, {REF.beta}, {REF.rho}), a = {A_HALF} km")
    print(f"  m0 = {tuple(M0)}")
    print("=" * 88)
    ok = True
    rng = np.random.default_rng(20260926)

    for ka, damp in ((0.05, 0.0), (0.3, 0.0), (0.3, 0.03)):
        omega = ka * REF.beta / A_HALF * (1.0 + 1j * damp)
        print(f"\n  ka_S = {ka}, omega = {omega:.4f}")
        print(f"    {'param':>8} {'order':>7} {'[R] err':>10} {'[H] imag vs real':>18}")
        grads = []
        for a in range(3):
            u = np.zeros(3)
            u[a] = 1.0
            g, order, err = richardson(M0, u, 1e-2 * SCALE[a], omega)
            grads.append(g)
            # [H]: derivative along an imaginary step, (f(m+ih) - f(m-ih)) / (2ih).
            hi = 1e-4 * SCALE[a]
            g_imag = (t0(M0 + 1j * hi * u, omega) - t0(M0 - 1j * hi * u, omega)) / (2j * hi)
            r_h = rel(g, g_imag)
            order_s = "linear" if np.isnan(order) else f"{order:.2f}"
            print(f"    {NAMES[a]:>8} {order_s:>7} {err:10.2e} {r_h:18.2e}")
            ok &= err < TOL and (np.isnan(order) or 1.8 < order < 2.2)
            # [H] is judged against the FD error it would replace, not a fixed
            # number: the non-holomorphic part grows as (ka)^2, so an absolute
            # threshold is wrong at small ka.
            ok &= r_h > H_MARGIN * max(err, TOL)
        u = rng.standard_normal(3) * SCALE
        u /= np.linalg.norm(u / SCALE)
        g_dir, _, err_dir = richardson(M0, u, 1e-2, omega)
        r_lin = rel(g_dir, sum(u[a] * grads[a] for a in range(3)))
        print(f"    [Lin] random direction vs sum of partials: {r_lin:.2e}   ([R] err {err_dir:.1e})")
        ok &= r_lin < TOL

    print("\n  [B] Born limit at zero contrast: || dT0/dm - Born weight || / || Born weight ||")
    print(f"    {'ka_S':>6} " + " ".join(f"{n:>10}" for n in NAMES))
    kas = (0.4, 0.2, 0.1, 0.05)
    table = []
    for ka in kas:
        omega = ka * REF.beta / A_HALF
        born = born_weights(omega)
        row = []
        for a in range(3):
            u = np.zeros(3)
            u[a] = 1.0
            g, _, _ = richardson(np.zeros(3), u, 1e-2 * SCALE[a], omega)
            row.append(rel(born[a], g))
        table.append(row)
        print(f"    {ka:6.2f} " + " ".join(f"{x:10.2e}" for x in row))
    table_arr = np.array(table)
    slopes = [float(np.polyfit(np.log(kas), np.log(table_arr[:, a]), 1)[0]) for a in range(3)]
    print("    fitted slope in ka: " + " ".join(f"{NAMES[a]} {slopes[a]:.2f}" for a in range(3)))
    ok &= all(1.8 < s < 2.2 for s in slopes)

    print("\n" + ("GATE A2 PASS" if ok else "GATE A2 FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
