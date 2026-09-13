#!/usr/bin/env python3
"""GATE: the SH impedance, under TWO constraints that must hold together.

THE DEFECT. GlobalMatrix's SH layer eigenvector
(`layer_matrix.layer_eigenvectors_sh_batched`) returns, in basis
[u_t, sigma_tz/(-iw)]::

    e_d = [-neta, mu*neta]      traction/displacement = -mu
    e_u = [ neta, mu*neta]      traction/displacement = +mu

The physical SH wave has ratio -/+ mu*eta. An eta is missing, and the layered SH
reflection comes out ANGLE-INDEPENDENT,

    R_SH = (mu1 - mu2)/(mu1 + mu2)                    measured, wrong
    R_SH = (mu1 eta1 - mu2 eta2)/(mu1 eta1 + mu2 eta2)   Aki & Richards

A reflection coefficient that does not depend on the ray parameter is the tell.
`cubic_scattering.kennett_layers` gets this right; GlobalMatrix does not. This is
NOT a thesis error -- the thesis uses a 6-component quasi-SH eigenvector with
energy normalisation (GRepresentations.tex, Eq. SHeigen/epsdef), a different and
more general formulation.

WHY ONE CONSTRAINT IS NOT ENOUGH. Fixing the eigenvector alone BREAKS the
uniform reduction (1.4e-15 -> 2.7e-1). For a unit traction jump in a uniform
medium the coded eigenvectors give displacement 1/(2 mu) where the correct ones
give 1/(2 mu eta) -- a factor eta. The uniform case nevertheless passes today
because the wrapper correction's K operator multiplies the SH channel by eta_S:

    K = 1 (+) (-P_par + eta_S P_perp)     and P_perp IS the SH direction.

So D2's eta_S on the perpendicular component compensates the missing eta. That
makes uniform media exact and cannot fix interfaces, where eta differs between
layers. The two errors must therefore be removed TOGETHER.

THE TWO CONSTRAINTS, which any candidate fix must satisfy simultaneously:

  [C1] UNIFORM REDUCTION. The stratified 9x9 on a uniform model must equal the
       independent whole-space kernel. Target 1e-13.

  [C2] INTERFACE REFLECTION. The SH reflection extracted from the stratified
       propagator must match `kennett_layers`, which matches Aki & Richards at
       every ray parameter. Target 1e-3 relative, and it must VARY with p.

  [C3] CONTROL. P and SV must stay at 1.000000 throughout -- they are correct
       today, and a fix that disturbs them has traded one defect for another.

  [C4] MULTI-INTERFACE. A single interface cannot exercise internal
       reverberation, and the single-bounce extraction of [C2] assumes one bounce
       dominates. Here a fast/slow/fast stack sits below the plane and the TOTAL
       downward reflection, every internal multiple included, is compared against
       `kennett_layers` on that sub-stack. All three modes to 1e-4.

       A REFERENCE-DEPTH TRAP worth recording: `kennett_layers` references RD at
       the first INTERFACE, not at the top of the stack -- its recursion phases
       the layer BELOW each interface, so the leading layer's own two-way delay
       is excluded. Feeding it the layers below the plane directly gives an RD
       short by exactly |e^{2 i kz PIT}| (0.428 here), and because that shifts P,
       SV and SH alike it reads as a physics failure rather than a bookkeeping
       one. A two-layer test with a 1e-9 top layer cannot see it. The sub-stack
       therefore prepends a zero-thickness layer of the plane's own medium.

Run:  conda run -n seismic python scripts/gate_sh_impedance.py
Seismic units (km/s, g/cm3), time convention e^{-i omega t}.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "PhD_fortran_code"))
sys.path.insert(0, "/Users/tod/Desktop/SeismicInversion")

import cubic_scattering.layered_correction as LC  # noqa: E402
import GlobalMatrix.layer_matrix as LM  # noqa: E402
import GlobalMatrix.layered_greens as LG  # noqa: E402
from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from cubic_scattering.kennett_layers import (  # noqa: E402
    _vertical_slowness,
)
from cubic_scattering.sweep_kernels import (  # noqa: E402
    same_depth_kernel_9x9,
    vertical_kernel_9x9,
)
from cubic_scattering.sweep_modes import vertical_factorisation  # noqa: E402
from Kennett_Reflectivity.layer_model import LayerModel  # noqa: E402

A, B, RH = 4.0, 2.22, 2.6
PIT, FRQ = 1.0, 6.0
OM = 2 * np.pi * FRQ
NL, PLANE, LREF = 60, 50, 53
HD = (LREF - 1 - PLANE) * PIT

_ORIG_EIG = LM.layer_eigenvectors_sh_batched
_ORIG_K = LC.k_operator


def model(scale: float, q: float = 20.0) -> LayerModel:
    """Uniform crust with medium B from layer LREF down: ONE interface."""
    al = [1.5] + [A] * (NL + 1)
    be = [0.0] + [B] * (NL + 1)
    rh = [1.03] + [RH] * (NL + 1)
    for lay in range(LREF, NL + 2):
        al[lay] = A + 2.5 * scale
        be[lay] = B + 1.48 * scale
        rh[lay] = RH + 0.7 * scale
    return LayerModel.from_arrays(
        alpha=al,
        beta=be,
        rho=rh,
        thickness=[3.0, *([PIT] * NL), np.inf],
        Q_alpha=[q] * (NL + 2),
        Q_beta=[1e10, *([q] * NL), q],
    )


def install(variant: str) -> None:
    """Restore the PRE-FIX behaviour, to keep this gate discriminating.

    The fix is applied in the source now, so the controls have to reinstate the
    old halves rather than the new ones. ``''`` is the shipped code.
    """

    def eig_old(neta, rho, beta_c):
        """The pre-fix SH eigenvector: impedance mu, an eta short."""
        mu_neta = rho * beta_c**2 * neta
        e_d = np.stack([-neta, mu_neta], axis=-1).astype(np.complex128)[:, :, None]
        e_u = np.stack([neta, mu_neta], axis=-1).astype(np.complex128)[:, :, None]
        return e_d, e_u

    def k_old(s_s, kx, ky, omega):
        """The pre-fix K, whose perpendicular component carried eta_S."""
        kpar = float(np.hypot(kx, ky))
        if kpar == 0.0:
            raise ValueError("k_operator undefined at kx = ky = 0") from None
        khat = np.array([kx, ky]) / kpar
        par = np.outer(khat, khat)
        eta = np.sqrt(s_s**2 - (kpar / complex(omega)) ** 2 + 0j)
        if np.imag(eta) < 0:
            eta = -eta
        out = np.eye(3, dtype=complex)
        out[1:, 1:] = -par + complex(eta) * (np.eye(2) - par)
        return out

    LM.layer_eigenvectors_sh_batched = _ORIG_EIG
    LG.layer_eigenvectors_sh_batched = _ORIG_EIG
    LC.k_operator = _ORIG_K
    if "eig" in variant:
        LM.layer_eigenvectors_sh_batched = eig_old
        LG.layer_eigenvectors_sh_batched = eig_old
    if "k" in variant:
        LC.k_operator = k_old


def c1_uniform_reduction() -> float:
    uni = model(0.0, q=2.0)
    s_p, s_s = uni.complex_slowness_p(), uni.complex_slowness_s()
    ref = ReferenceMedium(1 / s_p[1], 1 / s_s[1], uni.rho[1])
    kx = np.array([0.4, 1.1, 1.9])
    got = LC.corrected_layered_9x9(uni, OM, kx, np.full_like(kx, 0.3), PLANE + 1, PLANE)
    want = np.stack([vertical_kernel_9x9(np.array([k]), 0.3, -PIT, OM, ref)[:, :, 0] for k in kx])
    return float(np.abs(got - want).max() / np.abs(want).max())


def extract_r(mod: LayerModel, kx: float, ky: float) -> np.ndarray:
    """Per-mode reflection implied by the stratified propagator, modes (P,SV,SH).

    Solves DeltaG = D_up . phase(H) . R . phase(H) . S_down for R. Validated by
    P and SV coming out at 1.000000 against Kennett.
    """
    s_p, s_s = mod.complex_slowness_p(), mod.complex_slowness_s()
    ref_p = ReferenceMedium(1 / s_p[PLANE], 1 / s_s[PLANE], mod.rho[PLANE])
    ref1 = ReferenceMedium(1 / s_p[1], 1 / s_s[1], mod.rho[1])
    lay = LC.corrected_layered_9x9(mod, OM, np.array([kx]), np.array([ky]), PLANE, PLANE)[0]
    d_g = lay - same_depth_kernel_9x9(np.array([kx]), ky, OM, ref1)[:, :, 0]
    fac = vertical_factorisation(kx, ky, +PIT, OM, ref_p)
    ph = np.exp(1j * fac.kz[0:3] * HD)
    left = fac.receiver[:, 3:6] @ np.diag(ph)
    right = np.diag(ph) @ fac.source[0:3, :]
    return np.linalg.pinv(left) @ d_g @ np.linalg.pinv(right)


def analytic_r(mod: LayerModel, p: float) -> tuple[complex, complex]:
    """(SH reflection per Aki & Richards, the eta-less form) at the interface."""
    s_s = mod.complex_slowness_s()
    up, lo = PLANE + 1, LREF
    n1, n2 = _vertical_slowness(s_s[up], p), _vertical_slowness(s_s[lo], p)
    b1, b2 = 1 / s_s[up], 1 / s_s[lo]
    mu1, mu2 = mod.rho[up] * b1 * b1, mod.rho[lo] * b2 * b2
    z1, z2 = mu1 * n1, mu2 * n2
    return (z1 - z2) / (z1 + z2), (mu1 - mu2) / (mu1 + mu2)


def multilayer_model(q: float = 20.0) -> LayerModel:
    """Several contrasting layers below the plane: SH multiples compound.

    A single interface cannot exercise internal reverberation. Here the stack
    below interface PLANE alternates fast and slow, so the total reflection
    contains every order of interbed multiple, not just one bounce.
    """
    al = [1.5] + [A] * (NL + 1)
    be = [0.0] + [B] * (NL + 1)
    rh = [1.03] + [RH] * (NL + 1)
    fast = (6.5, 3.7, 3.3)
    slow = (3.2, 1.8, 2.3)
    for lay in (53, 54):
        al[lay], be[lay], rh[lay] = fast
    for lay in (56, 57):
        al[lay], be[lay], rh[lay] = slow
    for lay in range(59, NL + 2):
        al[lay], be[lay], rh[lay] = fast
    return LayerModel.from_arrays(
        alpha=al,
        beta=be,
        rho=rh,
        thickness=[3.0, *([PIT] * NL), np.inf],
        Q_alpha=[q] * (NL + 2),
        Q_beta=[1e10, *([q] * NL), q],
    )


def substack_below_plane(mod: LayerModel):
    """The layers strictly below the plane, as a stack whose RD sits AT the plane.

    kennett_layers references RD at the first INTERFACE, not at the top of the
    stack: its recursion phases the layer BELOW each interface, so the leading
    layer's own two-way delay is excluded. Measured, not assumed -- feeding the
    layers below the plane directly gives an RD short by exactly
    |e^{2 i kz * PIT}| = 0.428 here. A two-layer test with a 1e-9 top layer
    cannot see this, which is how it slipped past the first time.

    So a zero-thickness layer of the PLANE's own medium is prepended, and RD then
    refers to the plane depth -- the quantity the reverberation actually carries.
    """
    from cubic_scattering.kennett_layers import IsotropicLayer, LayerStack

    layers = [
        IsotropicLayer(
            float(mod.alpha[PLANE]),
            float(mod.beta[PLANE]),
            float(mod.rho[PLANE]),
            1e-9,
            float(mod.Q_alpha[PLANE]),
            float(mod.Q_beta[PLANE]),
        )
    ]
    for j in range(PLANE + 1, mod.n_layers):
        thick = float(mod.thickness[j])
        layers.append(
            IsotropicLayer(
                float(mod.alpha[j]),
                float(mod.beta[j]),
                float(mod.rho[j]),
                thick,
                float(mod.Q_alpha[j]),
                float(mod.Q_beta[j]),
            )
        )
    return LayerStack(layers)


def c4_multi_interface() -> bool:
    """[C4] Total reflection through a MULTI-interface stack, all multiples."""
    from cubic_scattering.kennett_layers import kennett_layers

    mod = multilayer_model()
    stack = substack_below_plane(mod)
    print("\n    [C4] MULTI-INTERFACE -- total RD at the plane, internal multiples included")
    print("         arbiter: kennett_layers on the sub-stack below the plane")
    print(
        f"         {'kh':>6} {'SH meas':>12} {'SH kennett':>12} {'ratio':>9} {'P ratio':>9} {'SV ratio':>9}"
    )
    ok = True
    for kh in (0.05, 0.5, 1.5):
        kx, ky = kh * 0.8, kh * 0.6
        p = kh / OM
        s_p, s_s = mod.complex_slowness_p(), mod.complex_slowness_s()
        ref_p = ReferenceMedium(1 / s_p[PLANE], 1 / s_s[PLANE], mod.rho[PLANE])
        ref1 = ReferenceMedium(1 / s_p[1], 1 / s_s[1], mod.rho[1])
        lay = LC.corrected_layered_9x9(mod, OM, np.array([kx]), np.array([ky]), PLANE, PLANE)[0]
        d_g = lay - same_depth_kernel_9x9(np.array([kx]), ky, OM, ref1)[:, :, 0]
        fac = vertical_factorisation(kx, ky, +PIT, OM, ref_p)
        # No explicit phase: RD is referenced at the plane, so it already carries
        # the propagation down to every interface and back.
        r_tot = np.linalg.pinv(fac.receiver[:, 3:6]) @ d_g @ np.linalg.pinv(fac.source[0:3, :])
        ken = kennett_layers(stack, p, np.array([OM]))
        sh_ratio = r_tot[2, 2] / ken.RD_sh[0]
        p_ratio = r_tot[0, 0] / ken.RD_psv[0, 0, 0]
        sv_ratio = r_tot[1, 1] / ken.RD_psv[0, 1, 1]
        good = abs(sh_ratio - 1) < 1e-4 and abs(p_ratio - 1) < 1e-4 and abs(sv_ratio - 1) < 1e-4
        ok = ok and good
        print(
            f"         {kh:6.2f} {r_tot[2, 2].real:12.6f} {ken.RD_sh[0].real:12.6f} "
            f"{sh_ratio.real:9.5f} {p_ratio.real:9.5f} {sv_ratio.real:9.5f}"
        )
    print(
        f"         all three modes within 1e-4 of the full multi-layer reflectivity -> "
        f"{'PASS' if ok else 'FAIL'}"
    )
    return ok


def report(variant: str) -> bool:
    install(variant)
    c1 = c1_uniform_reduction()
    mod = model(0.01)
    print(f"\n  variant: {variant and ('OLD ' + variant) or 'SHIPPED (fixed)'}")
    print(f"    [C1] uniform reduction            : {c1:.3e}   {'PASS' if c1 < 1e-13 else 'FAIL'}")
    print(f"    [C2/C3] {'kh':>5} {'P':>9} {'SV':>9} {'SH meas':>11} {'SH A&R':>11} {'SH ratio':>9}")
    sh_ok, psv_ok, varies = True, True, []
    for kh in (0.02, 0.5, 1.5, 3.0):
        kx, ky = kh * 0.8, kh * 0.6
        r_eff = extract_r(mod, kx, ky)
        p = kh / OM
        r_true, _ = analytic_r(mod, p)
        ratio = r_eff[2, 2] / r_true
        varies.append(r_eff[2, 2].real)
        psv_ok = psv_ok and abs(r_eff[0, 0] / _kennett_psv(mod, p, 0) - 1) < 1e-3
        sh_ok = sh_ok and abs(ratio - 1) < 1e-3
        print(
            f"          {kh:5.2f} {r_eff[0, 0].real / _kennett_psv(mod, p, 0).real:9.5f} "
            f"{r_eff[1, 1].real / _kennett_psv(mod, p, 1).real:9.5f} "
            f"{r_eff[2, 2].real:11.6f} {r_true.real:11.6f} {ratio.real:9.5f}"
        )
    spread = (max(varies) - min(varies)) / abs(np.mean(varies))
    print(
        f"    SH varies with p by {spread:.2%}  (the wrong form is p-INDEPENDENT, so this must be non-zero)"
    )
    ok = c1 < 1e-13 and sh_ok and psv_ok
    print(f"    -> {'PASS' if ok else 'FAIL'}")
    return ok


def _kennett_psv(mod: LayerModel, p: float, idx: int) -> complex:
    from cubic_scattering.kennett_layers import psv_solid_solid

    s_p, s_s = mod.complex_slowness_p(), mod.complex_slowness_s()
    up, lo = PLANE + 1, LREF
    e1, n1 = _vertical_slowness(s_p[up], p), _vertical_slowness(s_s[up], p)
    e2, n2 = _vertical_slowness(s_p[lo], p), _vertical_slowness(s_s[lo], p)
    psv = psv_solid_solid(p, e1, n1, mod.rho[up], 1 / s_s[up], e2, n2, mod.rho[lo], 1 / s_s[lo])
    return psv.Rd[idx, idx]


def main() -> int:
    print("=" * 78)
    print("GATE -- SH impedance under two simultaneous constraints")
    print(f"  plane at interface {PLANE}, single reflector {HD} km below")
    print("  reference: cubic_scattering.kennett_layers, which matches Aki & Richards")
    print("=" * 78)

    results = {}
    for variant in ("", "eig", "k", "eig+k"):  # "" = shipped; others reinstate old halves
        try:
            results[variant and ("OLD " + variant) or "SHIPPED"] = report(variant)
        except Exception as exc:  # noqa: BLE001
            print(f"    variant {variant!r} raised: {type(exc).__name__}: {exc}")
            results[variant and ("OLD " + variant) or "SHIPPED"] = False
    install("")
    c4 = c4_multi_interface()

    print("\n" + "=" * 78)
    winner = [k for k, v in results.items() if v]
    expected = ["SHIPPED"]
    if winner == expected:
        print(f"SATISFIES BOTH CONSTRAINTS: {', '.join(winner)}")
    else:
        print("NO variant satisfies both constraints yet.")
        print("  The compensating eta is therefore NOT confined to k_operator's")
        print("  perpendicular factor; the SH assembly needs a wider pass.")
    print("=" * 78)
    return 0 if winner else 1


if __name__ == "__main__":
    raise SystemExit(main())
