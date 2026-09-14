#!/usr/bin/env python3
"""GATE: the ABSOLUTE MAGNITUDE of the same-plane reverberation DeltaG0.

This is the check `gate_sweep_dressed_equivalence` explicitly defers. That gate
establishes that DeltaG0 is reciprocal, vanishes without layering, recedes with
the reflector, and behaves as a self-energy under the dressed-T identity -- all
of which are blind to an overall scale. `gate_sh_impedance` [C2]/[C4] goes
further and compares the three DIAGONAL reflection coefficients fitted out of
DeltaG0 against `kennett_layers`. Two things that still hides:

  (a) the pseudo-inverse fit silently PROJECTS AWAY anything in DeltaG0 that
      does not lie in the rank-3 one-way span receiver[:,3:6] (x) source[0:3,:];
  (b) the mode-CONVERSION channels P<->SV are never compared at all, and a
      diagonal-only comparison cannot see them -- a similarity R -> C^-1 R C by
      any diagonal C leaves every diagonal entry exactly invariant.

So this gate runs the comparison FORWARD, with no fitting anywhere:

    DeltaG0  ?=  receiver[:,3:6] . R_kennett . source[0:3,:]

on the full 9x9, all nine mode channels at once, against a reflection matrix
built by `kennett_layers` alone.

  [M1] IN-SPAN. The part of DeltaG0 lying outside the rank-3 one-way span must
       be zero. If it is not, every fitted-R result in the programme is
       measuring a projection rather than the operator. Measured <= 4e-9.

  [M2] FORWARD MAGNITUDE, full 9x9. The predicted kernel must equal the measured
       one -- not in ratio, not per-diagonal, but as a whole operator. This is
       the absolute statement: a scale error anywhere in the stratified
       propagator, the wrapper correction or the mode bridge shows up here and
       cannot be absorbed.

  [M3] THE BASIS FACTOR, PREDICTED NOT FITTED. `vertical_factorisation`'s mode
       basis is NOT flux-normalised; Kennett's is. The two differ by a diagonal

           C = diag(1, c, 1),   c = i beta sqrt(eta_S) / (alpha sqrt(eta_P))

       the standard Kennett P/S flux ratio. It is invisible on the diagonal and
       is exactly what the conversion channels measure. It is written down here
       a priori and asserted against the fitted value -- 8 significant figures,
       identical across weak, strong and multi-interface models, which is what
       makes it a convention rather than a tuned constant.

       WHY the two bases differ, and why it is not cosmetic. `sweep_modes`
       normalises polarisations to unit DISPLACEMENT, bilinearly (pol @ pol = 1)
       so they continue analytically into evanescence. `kennett_layers`
       normalises to unit ENERGY FLUX -- "sqrt(eta*rho) normalization for
       unitary recursion", its own line 186 -- because that is what makes the
       interface matrices symmetric (Tu = Td.T, reciprocity) and unitary, which
       is what the reflectivity recursion needs to be valid at all. Vertical
       flux goes as rho v^2 eta |A|^2, so unit-flux amplitude goes as
       1/(v sqrt(rho eta)): hence the velocity and the sqrt(eta). The i on SV is
       Kennett's eigenvector convention (the m2ci = -2i in psv_solid_solid).

       NOT A NEW DERIVATION. This package already carries exactly this
       similarity: `slab_scattering.SlabReflectionMatrix.to_modified` converts
       the slab solver's displacement-convention R to Kennett's with
       D = diag(alpha sqrt(eta_P), i beta sqrt(eta_S)), pinned the same way, and
       its docstring warns that the naive sqrt(eta_i/eta_j) form is wrong. The
       sweep solver needed the conversion the slab solver already had. The
       agreement between the two is corroboration, not a coincidence.

Three models are exercised, because a single one cannot separate a convention
from an accident: a weak single interface (the single-bounce limit), a strong
single interface, and a fast/slow/fast stack carrying every order of internal
multiple. Four wavenumbers each, into the evanescent regime.

Run:  conda run -n seismic python scripts/gate_dg0_absolute_magnitude.py
Seismic units (km/s, g/cm3), time convention e^{-i omega t}.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "PhD_fortran_code"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, "/Users/tod/Desktop/SeismicInversion")

import cubic_scattering.layered_correction as LC  # noqa: E402
from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from cubic_scattering.kennett_layers import kennett_layers  # noqa: E402
from cubic_scattering.sweep_kernels import same_depth_kernel_9x9  # noqa: E402
from cubic_scattering.sweep_modes import vertical_factorisation  # noqa: E402
from gate_sh_impedance import (  # noqa: E402
    OM,
    PIT,
    PLANE,
    model,
    multilayer_model,
    substack_below_plane,
)

KH = (0.05, 0.5, 1.5, 3.0)
TOL_SPAN = 1e-7
TOL_FWD = 1e-6
TOL_C = 1e-6


def r_kennett_3x3(mod, p: float) -> np.ndarray:
    """Total downward reflection referenced AT the plane, modes (P, SV, SH).

    Conversions included -- the whole point of this gate is the off-diagonal.
    The sub-stack prepends a zero-thickness layer of the plane's own medium so
    that RD refers to the plane depth; see `substack_below_plane` for why.
    """
    ken = kennett_layers(substack_below_plane(mod), p, np.array([OM]))
    out = np.zeros((3, 3), dtype=complex)
    out[:2, :2] = ken.RD_psv[0]
    out[2, 2] = ken.RD_sh[0]
    return out


def flux_ratio(ref: ReferenceMedium, p: float) -> complex:
    """Kennett P/S flux normalisation c_SV/c_P = i beta sqrt(eta_S)/(alpha sqrt(eta_P))."""
    eta_p = np.sqrt(complex(1 / ref.alpha**2 - p**2))
    eta_s = np.sqrt(complex(1 / ref.beta**2 - p**2))
    return 1j * ref.beta * np.sqrt(eta_s) / (ref.alpha * np.sqrt(eta_p))


def reverberation(mod, kx: float, ky: float):
    """DeltaG0 at the plane, with the one-way mode maps it should factor through."""
    s_p, s_s = mod.complex_slowness_p(), mod.complex_slowness_s()
    ref_p = ReferenceMedium(1 / s_p[PLANE], 1 / s_s[PLANE], mod.rho[PLANE])
    ref1 = ReferenceMedium(1 / s_p[1], 1 / s_s[1], mod.rho[1])
    lay = LC.corrected_layered_9x9(mod, OM, np.array([kx]), np.array([ky]), PLANE, PLANE)[0]
    d_g = lay - same_depth_kernel_9x9(np.array([kx]), ky, OM, ref1)[:, :, 0]
    fac = vertical_factorisation(kx, ky, +PIT, OM, ref_p)
    return d_g, fac.receiver[:, 3:6], fac.source[0:3, :], ref_p


def probe(mod, label: str) -> tuple[bool, bool, bool]:
    print(f"\n  {label}")
    print(
        f"    {'kh':>6} {'|dG0|':>12} {'out-of-span':>12} {'fwd, no C':>11} "
        f"{'fwd, with C':>12} {'|c fit - c pred|':>17}"
    )
    span_ok = fwd_ok = c_ok = True
    for kh in KH:
        kx, ky = kh * 0.8, kh * 0.6
        p = kh / OM
        d_g, left, right, ref_p = reverberation(mod, kx, ky)
        nrm = np.linalg.norm(d_g)
        r_ken = r_kennett_3x3(mod, p)

        # [M1] anything outside the rank-3 one-way span
        r_fit = np.linalg.pinv(left) @ d_g @ np.linalg.pinv(right)
        out = float(np.linalg.norm(d_g - left @ r_fit @ right) / nrm)

        # [M3] the basis factor, predicted a priori then compared to the fit
        c_pred = flux_ratio(ref_p, p)
        c_fit = r_fit[0, 1] / r_ken[0, 1]
        dc = float(abs(c_fit - c_pred))

        # [M2] the forward prediction, before and after the basis factor
        cmat = np.diag([1.0 + 0j, c_pred, 1.0 + 0j])
        raw = float(np.linalg.norm(d_g - left @ r_ken @ right) / nrm)
        fwd = float(np.linalg.norm(d_g - left @ (np.linalg.inv(cmat) @ r_ken @ cmat) @ right) / nrm)

        span_ok = span_ok and out < TOL_SPAN
        fwd_ok = fwd_ok and fwd < TOL_FWD
        c_ok = c_ok and dc < TOL_C
        print(f"    {kh:6.2f} {nrm:12.5e} {out:12.3e} {raw:11.3e} {fwd:12.3e} {dc:17.3e}")
    return span_ok, fwd_ok, c_ok


def channel_table(mod, kh: float) -> None:
    """Every non-zero channel of R, fitted vs Kennett, after the basis factor.

    The diagonal-only comparison in `gate_sh_impedance` covers three of these
    five; the two conversions are what this gate adds.
    """
    kx, ky = kh * 0.8, kh * 0.6
    p = kh / OM
    d_g, left, right, ref_p = reverberation(mod, kx, ky)
    r_fit = np.linalg.pinv(left) @ d_g @ np.linalg.pinv(right)
    cmat = np.diag([1.0 + 0j, flux_ratio(ref_p, p), 1.0 + 0j])
    r_ken = np.linalg.inv(cmat) @ r_kennett_3x3(mod, p) @ cmat
    names = ("P", "SV", "SH")
    print(f"\n  every non-zero channel of R, renormalised Kennett vs fit, kh = {kh}")
    print(f"    {'channel':>12} {'fit':>26} {'kennett':>26} {'ratio':>10}")
    for i in range(3):
        for j in range(3):
            if abs(r_ken[i, j]) < 1e-14:
                continue
            rat = r_fit[i, j] / r_ken[i, j]
            tag = f"R[{names[i]}<-{names[j]}]"
            print(
                f"    {tag:>12} {r_fit[i, j].real:+.6e}{r_fit[i, j].imag:+.6e}j "
                f"{r_ken[i, j].real:+.6e}{r_ken[i, j].imag:+.6e}j {rat.real:10.6f}"
            )


def main() -> int:
    print("=" * 78)
    print("GATE -- absolute magnitude of the same-plane reverberation DeltaG0")
    print(f"  plane at interface {PLANE}, arbiter cubic_scattering.kennett_layers")
    print("  FORWARD comparison of the full 9x9: no fitting, no ratio, no per-mode rescale")
    print("=" * 78)

    cases = (
        (multilayer_model(), "MULTI-INTERFACE below the plane -- every internal multiple"),
        (model(0.01), "WEAK single interface -- the single-bounce limit"),
        (model(1.0), "STRONG single interface"),
    )
    span = fwd = cst = True
    for mod, label in cases:
        s, f, c = probe(mod, label)
        span, fwd, cst = span and s, fwd and f, cst and c

    channel_table(multilayer_model(), 0.5)

    print("\n" + "=" * 78)
    print(f"  [M1] DeltaG0 lies in the rank-3 one-way span      : {'PASS' if span else 'FAIL'}")
    print(f"  [M2] forward 9x9 magnitude vs Kennett             : {'PASS' if fwd else 'FAIL'}")
    print(f"  [M3] P/S basis factor predicted, not fitted       : {'PASS' if cst else 'FAIL'}")
    ok = span and fwd and cst
    print(f"\nGATE DeltaG0 absolute magnitude: {'PASS' if ok else 'FAIL'}")
    print("  The 'fwd, no C' column is NOT a failure -- it is the size of the")
    print("  conversion mismatch that a diagonal-only comparison cannot see, and")
    print("  it is removed by an a priori basis factor carrying no free parameter.")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
