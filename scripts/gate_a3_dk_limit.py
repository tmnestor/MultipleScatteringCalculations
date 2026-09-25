#!/usr/bin/env python3
"""GATE A3: the Dietrich-Kormendi limit of the voxel gradient, ABSOLUTE.

WHY THIS GATE EXISTS
--------------------
Every earlier rung of note (IV)'s ladder is an invariant or a self-consistency
check -- reciprocity, a transpose, a derivative against its own differences --
and all of those are blind to an overall scale or sign error. This rung is the
first comparison against physics computed by an independent route.

At zero contrast the voxel gradient contracts the reference residual through
the Born vertex dT0/dm|_0 with the reference fields (note IV, eq. dkgrad). For a
laterally uniform plane of voxels hit by a plane wave, that is the D&K Frechet
kernel of a thin layer. In a uniform medium the Born PP reflection of a layer of
thickness h and contrast dm, referenced at its top, is EXACTLY

    dR_PP = R_lin(dm) (1 - e^{2 i omega eta_P h}),

with R_lin the Aki & Richards linearised PP coefficient,
    R_lin = 1/2 (1 - 4 beta^2 p^2) drho/rho + 1/(2 cos^2 i) dalpha/alpha - 4 beta^2 p^2 dbeta/beta.
That closed form shares no code with the voxel side.

VOXEL SIDE -- the gradient code's own pieces
--------------------------------------------
Cubes of pitch d on a d x d lattice form a sheet; per unit area its Born source
at the specular wavenumber k_par is

    tau = (1/d^2) (dT0/dm_a)|_0 psi_inc,     psi_inc = e^{i omega eta_P d/2} . [P down],

with dT0/dm from sweep_gradient.rayleigh_t0_and_derivative (the Richardson
derivative the gradient uses, gate A2). The field at the layer top is the
whole-space plane-to-plane kernel vertical_kernel_9x9 at dz = -d/2 -- the kernel
the receiver maps and the vertical sweep carry -- projected onto up-going P by
the mode bridge (sweep_modes.state_to_modes). Polarisations follow Aki &
Richards: each P displacement along its own wavevector.

WHAT IS COMPARED, AND WHAT IS EXPECTED
--------------------------------------
The sheet samples the layer at its midpoint and the cube T0 carries its form
factor (A2: coefficient ~0.12 in (k_S a)^2), so the two sides differ by O(d^2):
convergence in pitch, not a tolerance. Any O(1) defect -- a missing 2 pi or
1/d^2, a sign, a wrong vertex -- leaves a FLAT residual and fails the slope.

  [A3] relative residual per parameter (Dlambda, Dmu, Drho), at normal and two
       oblique slownesses (one split between k_x and k_y, so the k_y path is
       exercised); fitted slope in d asserted in [1.8, 2.2], and the finest
       residual asserted < 1e-2.

NOT COVERED: the finite-lattice assembly and the adjoint chain (the unit tests
check those against brute force), the lattice's non-specular orders (evanescent
for d < half a wavelength, and dropped here), and a stratified reference.

Run:  conda run -n seismic python scripts/gate_a3_dk_limit.py
Seismic units (km/s, g/cm3, GPa), time convention e^{-i omega t}, z down.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from cubic_scattering.sweep_gradient import rayleigh_t0_and_derivative  # noqa: E402
from cubic_scattering.sweep_kernels import vertical_kernel_9x9  # noqa: E402
from cubic_scattering.sweep_modes import modes_to_state, state_to_modes  # noqa: E402

REF = ReferenceMedium(alpha=5.0, beta=3.0, rho=2.5)
OMEGA = 2 * np.pi * 1.0
PITCHES = (0.4, 0.2, 0.1, 0.05)
NAMES = ("Dlambda", "Dmu", "Drho")
P_DOWN, P_UP = 0, 3  # sweep_modes.MODE_NAMES order


def r_lin_derivatives(p: float) -> np.ndarray:
    """d R_lin / d(lambda, mu, rho) for the Aki & Richards linearised PP coefficient."""
    al, be, rh = REF.alpha, REF.beta, REF.rho
    cos2 = 1.0 - (al * p) ** 2
    dr_dalpha = 1.0 / (2.0 * cos2 * al)
    dr_dbeta = -4.0 * be * p**2
    dr_drho = 0.5 * (1.0 - 4.0 * be**2 * p**2) / rh
    # alpha^2 = (lambda + 2 mu)/rho, beta^2 = mu/rho
    d_lam = dr_dalpha / (2.0 * rh * al)
    d_mu = dr_dalpha / (rh * al) + dr_dbeta / (2.0 * rh * be)
    d_rho = dr_drho + dr_dalpha * (-al / (2.0 * rh)) + dr_dbeta * (-be / (2.0 * rh))
    return np.array([d_lam, d_mu, d_rho])


def born_layer(p: float, d: float) -> np.ndarray:
    """Exact Born PP reflection derivative of a layer of thickness d, at its top."""
    eta = np.sqrt(1.0 / REF.alpha**2 - p**2)
    return r_lin_derivatives(p) * (1.0 - np.exp(2j * OMEGA * eta * d))


def voxel_sheet(kx: float, ky: float, d: float, *, sign: float = 1.0, area_power: int = 2) -> np.ndarray:
    """P-up amplitude at the layer top from a Born sheet of cubes, per parameter.

    ``sign`` and ``area_power`` exist only for the calibration legs: the true
    sheet has sign +1 and source density 1/d^2.
    """
    kz_p = np.sqrt((OMEGA / REF.alpha) ** 2 - kx**2 - ky**2)
    m2s = modes_to_state(kx, ky, OMEGA, REF)
    s2m = state_to_modes(kx, ky, OMEGA, REF)
    psi_inc = np.exp(1j * kz_p * d / 2.0) * m2s[:, P_DOWN]
    kernel = vertical_kernel_9x9(np.array([kx]), ky, -d / 2.0, OMEGA, REF)[:, :, 0]
    _, dt0 = rayleigh_t0_and_derivative(
        np.zeros((1, 1, 3)), [REF], OMEGA, d / 2.0, step=np.array([1e-2, 1e-2, 1e-3])
    )
    out = np.zeros(3, dtype=complex)
    for a in range(3):
        tau = sign * dt0[0, 0, a] @ psi_inc / d**area_power
        out[a] = (s2m @ (kernel @ tau))[P_UP]
    return out


def main() -> int:
    """Residual per parameter and slowness; slope in the pitch."""
    print("=" * 92)
    print("GATE A3 -- D&K limit, absolute: voxel Born sheet vs exact Born thin-layer PP reflection")
    print(f"  uniform medium (alpha, beta, rho) = ({REF.alpha}, {REF.beta}, {REF.rho})")
    print(f"  f = {OMEGA / (2 * np.pi):.1f} Hz")
    print("=" * 92)
    ok = True
    cases = (("normal", 0.0, 0.0), ("oblique, k_x", 0.10, 0.0), ("oblique, k_x & k_y", 0.12, 0.09))
    for label, px, py in cases:
        p = float(np.hypot(px, py))
        kx, ky = OMEGA * px, OMEGA * py
        print(f"\n  {label}: p = {p:.3f} s/km (incidence {np.degrees(np.arcsin(REF.alpha * p)):.1f} deg)")
        print(f"    {'pitch d':>8} " + " ".join(f"{n:>10}" for n in NAMES) + f"   {'dR_lam (voxel)':>24}")
        res = []
        for d in PITCHES:
            vox, ana = voxel_sheet(kx, ky, d), born_layer(p, d)
            rel = np.abs(vox - ana) / np.abs(ana)
            res.append(rel)
            print(f"    {d:8.3f} " + " ".join(f"{x:10.2e}" for x in rel) + f"   {vox[0]:24.6e}")
        arr = np.array(res)
        slopes = [float(np.polyfit(np.log(PITCHES), np.log(arr[:, a]), 1)[0]) for a in range(3)]
        print("    slope in d: " + "  ".join(f"{NAMES[a]} {slopes[a]:.2f}" for a in range(3)))
        good = all(1.8 < s < 2.2 for s in slopes) and float(arr[-1].max()) < 1e-2
        print(f"    -> {'PASS' if good else 'FAIL'}")
        ok &= good

    # Calibration: a sign or normalisation defect must leave a FLAT residual and
    # fail the slope criterion. If these passed, [A3] would prove nothing.
    print("\n  [c] calibration legs at 30 deg: must FAIL the slope criterion")
    kx, p = OMEGA * 0.10, 0.10
    for label, kw in (("vertex sign flipped", {"sign": -1.0}), ("1/d^2 omitted", {"area_power": 0})):
        rel = np.array(
            [
                np.abs(voxel_sheet(kx, 0.0, d, **kw) - born_layer(p, d)) / np.abs(born_layer(p, d))
                for d in PITCHES
            ]
        )
        slope = float(np.polyfit(np.log(PITCHES), np.log(rel[:, 0]), 1)[0])
        failed = not (1.8 < slope < 2.2 and rel[-1].max() < 1e-2)
        print(
            f"    {label:<22} residual {rel[0, 0]:.2e} -> {rel[-1, 0]:.2e}, slope {slope:5.2f}  "
            f"-> {'fails, as it must' if failed else 'PASSES: gate is blind'}"
        )
        ok &= failed
    print("\n" + ("GATE A3 PASS" if ok else "GATE A3 FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
