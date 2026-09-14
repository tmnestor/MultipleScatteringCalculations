#!/usr/bin/env python3
"""GATE: the ABSOLUTE MAGNITUDE of the ocean + free-surface reverberation.

`gate_free_surface_reverberation` establishes that the top reverberation exists,
attenuates, responds to the water properties, is periodic in the round-trip
phase, and reaches the 3-D table. Every one of those is satisfied by a WRONGLY
SCALED surface too. This gate fixes the scale.

THE OBJECT. Put a plane in a uniform elastic half-space under the ocean, with no
internal contrast anywhere. The only reflector above it is then the seabed, the
water column and the free surface. The same-plane diagonal of the layered table
is therefore a pure UPWARD look, and must factor as

    DeltaG0 = D_down . phase(H) . R_up . phase(H) . S_up,

with H the standoff from the plane to the seabed and R_up the total upward
reflection of the ocean system seen from just below the seabed.

THE ARBITER, built independently from the interface coefficients. Above the
seabed the water column is closed by a pressure release, so an up-going fluid
wave returns as M = -e0^2 with e0 = exp(i w eta_w h_w). Adding that stack onto
the seabed interface by the Kennett rule gives

    R_up = Ru + Tu . M . (I - Rd . M)^-1 . Td,

with (Rd, Ru, Td, Tu) the fluid-solid coefficients of `psv_fluid_solid` and M
carrying the fluid's single mode. Nothing in that expression comes from the
Green's function, so agreement is evidence rather than bookkeeping.

DIAGONALS ONLY, and that is not a dodge. `psv_fluid_solid` uses the flux
convention `sqrt(eta rho)`; the mode bridge uses unit displacement. Those differ
by a DIAGONAL similarity, which leaves diagonal entries exactly invariant --
the same argument as `gate_gmm_marine_stack`, and the factor itself is
[[displacement-vs-flux-basis-trap]] if the conversions are ever wanted.

  [A1] IN-SPAN. The part of DeltaG0 outside the rank-3 one-way span must be
       negligible, or the extraction below is reading a projection.
  [A2] MAGNITUDE, free surface ON. The extracted R_up must equal the arbiter.
  [A3] MAGNITUDE, free surface OFF. With the ocean a half-space, M = 0 and the
       arbiter collapses to Ru alone -- the bare seabed reflection. This fixes
       the scale of the DEFAULT background too, which was never pinned either.

Run:  conda run -n seismic python scripts/gate_free_surface_magnitude.py
Seismic units (km, km/s, g/cm3), time convention e^{-i omega t}.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "PhD_fortran_code"))
sys.path.insert(0, "/Users/tod/Desktop/SeismicInversion")

import cubic_scattering.layered_correction as LC  # noqa: E402
from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from cubic_scattering.kennett_layers import _vertical_slowness, psv_fluid_solid  # noqa: E402
from cubic_scattering.sweep_kernels import same_depth_kernel_9x9  # noqa: E402
from cubic_scattering.sweep_modes import vertical_factorisation  # noqa: E402

PITCH = 1.0
OM = 2 * np.pi * 6.0
W_ALPHA, W_RHO, W_THICK = 1.5, 1.03, 1.37
A, B, R = 5.0, 3.0, 2.5
PLANE = 8  # interfaces below the seabed, so H = PLANE * PITCH
Q = 1e6


def model(n_lay: int = 18):
    """Ocean over a UNIFORM elastic half-space. No internal contrast at all."""
    import Kennett_Reflectivity.layer_model as lm

    return lm.LayerModel.from_arrays(
        alpha=[W_ALPHA, *([A] * n_lay), A],
        beta=[0.0, *([B] * n_lay), B],
        rho=[W_RHO, *([R] * n_lay), R],
        thickness=[W_THICK, *([PITCH] * n_lay), np.inf],
        Q_alpha=[Q] * (n_lay + 2),
        Q_beta=[1e12, *([Q] * n_lay), Q],
    )


def arbiter_r_up(p: float, *, free_surface: bool) -> np.ndarray:
    """R_up = Ru + Tu M (I - Rd M)^-1 Td, from the interface coefficients alone."""
    mod = model()
    s_p, s_s = mod.complex_slowness_p(), mod.complex_slowness_s()
    eta_w = _vertical_slowness(s_p[0], p)
    eta_s = _vertical_slowness(s_p[1], p)
    neta_s = _vertical_slowness(s_s[1], p)
    c = psv_fluid_solid(p, eta_w, W_RHO, eta_s, neta_s, R, 1.0 / s_s[1])

    m = np.zeros((2, 2), dtype=complex)
    if free_surface:
        e0 = np.exp(1j * OM * eta_w * W_THICK)
        m[0, 0] = -(e0**2)  # pressure release, fluid's single mode

    # ORDER MATTERS, AND PP CANNOT SEE IT. An up-going SOLID wave transmits UP
    # into the fluid through Tu (rows = fluid modes, columns = solid),
    # reverberates in the water as M, and returns DOWN into the solid through
    # Td (rows = solid, columns = fluid). So Td stands on the LEFT:
    #
    #     R_up = Ru + Td M (I - Rd M)^-1 Tu.
    #
    # Written the other way round, Tu M (...) Td, the PP entry is UNCHANGED --
    # the two P-to-P factors simply commute as scalars -- while every entry
    # involving a conversion collapses to zero, because Tu's fluid-SV row and
    # Td's fluid-SV column are both identically zero. A gate on PP alone would
    # pass that. The SV diagonal is what catches it: physically an up-going SV
    # converts to P at the seabed, reverberates, and returns as SV.
    return c.Ru + c.Td @ m @ np.linalg.inv(np.eye(2) - c.Rd @ m) @ c.Tu


def extracted_r_up(kx: float, ky: float, *, free_surface: bool) -> tuple:
    """(R_up fitted from DeltaG0, out-of-span residual)."""
    mod = model()
    s_p, s_s = mod.complex_slowness_p(), mod.complex_slowness_s()
    ref = ReferenceMedium(1.0 / s_p[PLANE], 1.0 / s_s[PLANE], mod.rho[PLANE])

    lay = LC.corrected_layered_9x9(
        mod, OM, np.array([kx]), np.array([ky]), PLANE, PLANE, free_surface=free_surface
    )[0]
    d_g = lay - same_depth_kernel_9x9(np.array([kx]), ky, OM, ref)[:, :, 0]

    # Fitted for -dz: the path leaves UPWARD, reflects, returns DOWNWARD. So the
    # source rows are the up-going half and the receiver embedding the
    # down-going half -- the mirror of the layered-case extraction.
    fac = vertical_factorisation(kx, ky, -PITCH, OM, ref)
    ph = np.exp(1j * fac.kz[0:3] * (PLANE * PITCH))
    left = fac.receiver[:, 0:3] @ np.diag(ph)
    right = np.diag(ph) @ fac.source[3:6, :]

    r_fit = np.linalg.pinv(left) @ d_g @ np.linalg.pinv(right)
    out = float(np.abs(d_g - left @ r_fit @ right).max() / np.abs(d_g).max())
    return r_fit, out


def main() -> int:
    print("=" * 78)
    print("GATE -- absolute magnitude of the ocean + free-surface reverberation")
    print(f"  uniform half-space under {W_THICK} km of water; plane {PLANE} km below seabed")
    print("  arbiter: psv_fluid_solid coefficients + the pressure-release closure")
    print("  diagonals only -- convention-free under a diagonal similarity")
    print("=" * 78)

    ok_span = ok_fs = ok_hs = True
    for label, fs in (("free surface ON", True), ("free surface OFF (half-space)", False)):
        print(f"\n  {label}")
        print(
            f"    {'kh':>6} {'out-of-span':>12} {'PP fit':>11} {'PP arb':>11} {'SV fit':>11} {'SV arb':>11}"
        )
        good = True
        for kh in (0.3, 0.9, 1.8):
            kx, ky = kh * 0.8, kh * 0.6
            p = kh / OM
            r_fit, out = extracted_r_up(kx, ky, free_surface=fs)
            r_arb = arbiter_r_up(p, free_surface=fs)
            dpp = abs(r_fit[0, 0] - r_arb[0, 0]) / max(abs(r_arb[0, 0]), 1e-300)
            dsv = abs(r_fit[1, 1] - r_arb[1, 1]) / max(abs(r_arb[1, 1]), 1e-300)
            ok_span = ok_span and out < 1e-6
            good = good and dpp < 1e-4 and dsv < 1e-4
            print(
                f"    {kh:6.2f} {out:12.3e} {abs(r_fit[0, 0]):11.5f} {abs(r_arb[0, 0]):11.5f} "
                f"{abs(r_fit[1, 1]):11.5f} {abs(r_arb[1, 1]):11.5f}"
            )
            print(f"    {'':6} {'rel diff':>12} {dpp:11.3e} {'':11} {dsv:11.3e}")
        if fs:
            ok_fs = good
        else:
            ok_hs = good

    ok = ok_span and ok_fs and ok_hs
    print("\n" + "=" * 78)
    print(f"  [A1] DeltaG0 lies in the rank-3 one-way span : {'PASS' if ok_span else 'FAIL'}")
    print(f"  [A2] magnitude, free surface ON              : {'PASS' if ok_fs else 'FAIL'}")
    print(f"  [A3] magnitude, ocean as half-space          : {'PASS' if ok_hs else 'FAIL'}")
    print(f"\nGATE free-surface magnitude: {'PASS' if ok else 'FAIL'}")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
