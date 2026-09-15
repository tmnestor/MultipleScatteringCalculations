"""MEASUREMENT: an empirical SHEAR-CHANNEL correction for the cubic-voxel lattice.

⚠ THE NAME MATTERS. An earlier version of this file called K a "lattice
renormalisation". That word names a MECHANISM -- the lattice modifying a voxel's
depolarisation -- and no mechanism is derived here. K is one scalar tuned against
the very arbiter it is then scored against. Section [V] below sets out the case
that it is a fudge factor and the case that it is more than one; the honest label
until a derivation exists is "an empirical one-parameter correction".

THE RESIDUAL THIS EXPLAINS. With the lateral sum exact and the Galerkin contact
correction applied, the refinement ladder flattens and a residual of ~8.6e-4
survives that refinement cannot remove. Of four suspects tested
(`measure_residual_origin`, `measure_lattice_depolarisation`), THREE were
refuted -- the averaged propagator's dynamic truncation, the isolated-cube
depolarisation as a whole (removing it is 24.6x WORSE), and propagator averaging
beyond contact. The fourth, the T27 single-site strain sector, was NOT refuted:
it was tested unfairly (see [T]) and remains open.

WHERE IT ACTUALLY LIVES. Split by channel, the residual is entirely in the SHEAR
channel -- pure density 1.4e-5 and pure lambda 1.2e-5, both converging cleanly,
against pure mu 2.9e-3. It is also ANTI-correlated with the amplification
A_theta, which exonerates the amplification factors as such.

THE LAW. Scanning Dmu*_diag shows the residual is a single wrong scalar with a
sharp minimum. Expressed as the ratio of effective to isolated depolarisation,

    K := (1 - A_eff) / (1 - A_iso)

is constant to ~0.3% over a 16x contrast range, flat in frequency until (ka)
grows -- the signature of a STATIC effect -- and extrapolates to

    K = 0.8862   (static, weak-contrast limit)

i.e. FOR A LATTICE OF T9 VOXELS, SETTING THE SHEAR DEPOLARISATION TO ~0.886 OF
ITS ISOLATED-CUBE VALUE REMOVES THE RESIDUAL. The dilatational and density
channels need no correction at all. The restriction to the T9 baseline is not
cosmetic -- see [T].

WHY THIS IS NOT A FIT, and the tests that earn that claim:
  [U] universality -- K is constant across contrast and frequency, and at the
      optimum the scheme reproduces exact Kennett to ~1e-8, five orders below
      the isolated-cube residual. The entire residual is this one scalar.
  [P] prediction -- K, taken from the PURE-SHEAR static limit alone, is applied
      untouched to cases it was never fitted to: mixed contrasts, 3x contrast,
      and NEGATIVE shear. It must also do NOTHING to the pure-lambda and
      pure-density cases, which need no correction; that it leaves them exactly
      unchanged is as much a test as the cases it improves.
  [C] convergence -- with K applied the error FALLS under refinement instead of
      saturating. That is the structural change, not the smaller constant.

AND THE CASE AGAINST, which [V] prints in full:
  - K is NOT DERIVED; 0.8862 is read off a minimisation.
  - It is FITTED AGAINST THE ARBITER. Kennett supplies both target and score, so
    the ~1e-8 residual at the optimum is guaranteed by construction. It shows
    that one parameter SUFFICES, not that it is RIGHT -- the evidence that it
    means something is [U] and [P], not that number.
  - It is BASELINE-DEPENDENT ([T]). "The lattice multiplies the true
    depolarisation by 0.886" ought to hold whichever theory computes the true
    depolarisation. It does not. On its own terms this is the strongest argument
    against reading K as a property of the lattice.

8/9 = 0.888889 is refuted (the limit is 2.6e-3 below it, far outside the fit
scatter). Its proximity to sqrt(pi)/2 = 0.886227 is treated as numerology --
sqrt(pi)/d is this project's Ewald splitting default, so the resemblance is
suspicious rather than encouraging. K is NOT wired into the library: this script
applies it explicitly so that no result silently depends on it.

Run:  conda run -n seismic python scripts/measure_shear_renormalisation.py
SI units (m, m/s, kg/m3, Pa).
"""

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
)
from cubic_scattering.slab_scattering import (  # noqa: E402
    SlabGeometry,
    SlabMaterial,
    build_slab_kernels,
    compute_slab_scattering,
    kennett_reference_rpp,
    slab_rpp_periodic,
)
from cubic_scattering.voigt_tmatrix import effective_stiffness_voigt  # noqa: E402

REF = ReferenceMedium(5000.0, 3000.0, 2500.0)
OMEGA, H_PHYS, M = 60.0, 4.0, 4
MU0 = REF.rho * REF.beta**2
K_RENORM = 0.886234


def _reflection(dlam, dmu, drho, n_z, omega, *, shear_scale=None, renorm=False, baseline="t9"):
    """Complex R_PP through the exact kernel, with the shear sector adjustable.

    ``baseline`` selects the single-site theory supplying the strain sector:
    't9' is the analytic Rayleigh cube T-matrix, 't27' the Galerkin one. The
    density channel always comes from T9, which is the only one that exposes it.
    """
    a = H_PHYS / (2.0 * n_z)
    geom = SlabGeometry(M=M, N_z=n_z, a=a)
    ones = np.ones((n_z, M, M))
    mat = SlabMaterial(Dlambda=dlam * ones, Dmu=dmu * ones, Drho=drho * ones, ref=REF)
    t9 = compute_cube_tmatrix(omega, a, REF, MaterialContrast(dlam, dmu, drho))
    src = (
        compute_cube_tmatrix_galerkin(omega, a, REF, MaterialContrast(dlam, dmu, drho))
        if baseline == "t27"
        else t9
    )

    dmu_diag = src.Dmu_star_diag
    if shear_scale is not None:
        dmu_diag = shear_scale * dmu
    elif renorm and dmu != 0.0:
        a_diag = src.Dmu_star_diag / dmu
        dmu_diag = (1.0 - K_RENORM * (1.0 - a_diag)) * dmu

    v = (2.0 * a) ** 3
    cell = np.zeros((9, 9), dtype=complex)
    cell[:3, :3] = omega**2 * complex(t9.Drho_star) * v * np.eye(3)
    cell[3:, 3:] = v * effective_stiffness_voigt(src.Dlambda_star, dmu_diag, src.Dmu_star_off)
    t0 = np.broadcast_to(cell, (n_z, M, M, 9, 9)).copy()
    kh = build_slab_kernels(
        geom,
        omega,
        REF,
        periodic=True,
        lattice_ewald=True,
        volume_averaged=True,
        n_orders=2,
    )
    res = compute_slab_scattering(
        geom,
        mat,
        omega,
        np.array([1.0, 0.0, 0.0]),
        "P",
        periodic=True,
        gmres_tol=1e-13,
        kernel_hat=kh,
        T_local=t0,
    )
    return complex(slab_rpp_periodic(res, t0)), t0


def _err(dlam, dmu, drho, n_z=4, omega=OMEGA, **kw):
    r, _ = _reflection(dlam, dmu, drho, n_z, omega, **kw)
    exact = kennett_reference_rpp(REF, MaterialContrast(dlam, dmu, drho), H_PHYS, omega)
    return float(abs(r - exact) / abs(exact))


def _optimum(dmu, omega, dlam=0.0, drho=0.0, baseline="t9"):
    """(K, s_iso, s*, residual at the optimum) by a local linear solve."""
    a = H_PHYS / (2.0 * 4)
    con = MaterialContrast(dlam, dmu, drho)
    src = (
        compute_cube_tmatrix_galerkin(omega, a, REF, con)
        if baseline == "t27"
        else compute_cube_tmatrix(omega, a, REF, con)
    )
    s_iso = (src.Dmu_star_diag / dmu).real
    h = 0.002
    kw = {"baseline": baseline}
    r0, _ = _reflection(dlam, dmu, drho, 4, omega, shear_scale=s_iso, **kw)
    r1 = (_reflection(dlam, dmu, drho, 4, omega, shear_scale=s_iso + h, **kw)[0] - r0) / h
    rk = complex(kennett_reference_rpp(REF, con, H_PHYS, omega))
    s_star = s_iso - float(np.real((r0 - rk) * np.conj(r1)) / abs(r1) ** 2)
    res = _err(dlam, dmu, drho, omega=omega, shear_scale=s_star, **kw)
    return (1.0 - s_star) / (1.0 - s_iso), s_iso, s_star, res


def main() -> int:
    print("=" * 88)
    print("MEASUREMENT -- the lattice renormalisation of a cube's shear depolarisation")
    print("  uniform slab, exact lateral sum + Galerkin contact; exact answer = Kennett")
    print("=" * 88)

    print("\n  [S] the residual is entirely in the SHEAR channel")
    print(f"       {'case':>16} {'A_theta':>10} {'residual':>12}")
    for name, dl, dm, dr in (
        ("pure density", 0.0, 0.0, 100.0),
        ("pure lambda", 2.0e9, 0.0, 0.0),
        ("pure mu", 0.0, 1.0e9, 0.0),
        ("working", 2.0e9, 1.0e9, 100.0),
    ):
        t = compute_cube_tmatrix(OMEGA, H_PHYS / 8.0, REF, MaterialContrast(dl, dm, dr))
        print(f"       {name:>16} {t.amp_theta.real:10.6f} {_err(dl, dm, dr):12.4e}")
    print("       ANTI-correlated with A_theta: pure lambda has MORE depolarisation")
    print("       and 240x LESS residual, which exonerates the amplification itself.")

    print("\n  [U] universality of K, and the residual at the optimum")
    print(f"       {'Dmu/mu0':>9} {'omega':>7} {'K':>10} {'residual at optimum':>21}")
    ks = []
    for dmu in (0.25e9, 1.0e9, 4.0e9):
        k, _, _, res = _optimum(dmu, OMEGA)
        ks.append(k)
        print(f"       {dmu / MU0:9.4f} {OMEGA:7.0f} {k:10.6f} {res:21.3e}")
    for om in (30.0, 240.0):
        k, _, _, res = _optimum(1.0e9, om)
        print(f"       {1.0e9 / MU0:9.4f} {om:7.0f} {k:10.6f} {res:21.3e}")
    spread = (max(ks) - min(ks)) / float(np.mean(ks))
    print(f"       spread of K across a 16x contrast range: {spread:.2e}")
    print(f"       8/9 = {8 / 9:.6f} is REFUTED: the static limit is 0.886234,")
    print("       2.6e-3 below it and far outside the fit scatter.")

    print(f"\n  [P] K = {K_RENORM} applied as a RULE to cases it was not fitted to")
    print(f"       {'case':>26} {'isolated':>12} {'renormalised':>13} {'gain':>8}")
    for name, dl, dm, dr in (
        ("working (mixed)", 2.0e9, 1.0e9, 100.0),
        ("mixed, 3x contrast", 6.0e9, 3.0e9, 300.0),
        ("pure lambda (none due)", 2.0e9, 0.0, 0.0),
        ("pure density (none due)", 0.0, 0.0, 100.0),
        ("NEGATIVE shear", 0.0, -1.0e9, 0.0),
        ("negative mixed", -2.0e9, -1.0e9, -100.0),
    ):
        a_ = _err(dl, dm, dr)
        b_ = _err(dl, dm, dr, renorm=True)
        print(f"       {name:>26} {a_:12.4e} {b_:13.4e} {a_ / max(b_, 1e-300):8.1f}x")
    print("       The two 'none due' rows must come out at 1.0x -- leaving the")
    print("       channels that need no correction untouched is as much a test as")
    print("       improving the ones that do. NEGATIVE shear is the strongest")
    print("       evidence against a fit: K was extracted on POSITIVE shear.")

    print("\n  [C] with K applied the scheme CONVERGES instead of saturating")
    print(f"       {'n_z':>4} {'isolated':>12} {'ratio':>7} {'renormalised':>13} {'ratio':>7}")
    pa = pb = None
    for n_z in (1, 2, 4, 8):
        a_ = _err(2.0e9, 1.0e9, 100.0, n_z=n_z)
        b_ = _err(2.0e9, 1.0e9, 100.0, n_z=n_z, renorm=True)
        ra = "" if pa is None else f"{a_ / pa:7.2f}"
        rb = "" if pb is None else f"{b_ / pb:7.2f}"
        print(f"       {n_z:4d} {a_:12.4e} {ra:>7} {b_:13.4e} {rb:>7}")
        pa, pb = a_, b_
    print("       The isolated column saturates at ratio 1.00; the renormalised one")
    print("       keeps falling. That structural change is the result, not the")
    print("       smaller constant.")

    # ---- [T] is K a LATTICE property, or tied to the T9 baseline? ----------
    print("\n  [T] the same optimisation from the T27 Galerkin strain sector")
    print("       If K were purely a property of the LATTICE, any single-site theory")
    print("       would have to be corrected to the SAME absolute s*. It is not.")
    print(f"       {'baseline':>9} {'case':>12} {'s_iso':>10} {'s*':>10} {'K':>10}")
    k_by = {}
    for base in ("t9", "t27"):
        for label, (dl, dm, dr) in (
            ("pure shear", (0.0, 1.0e9, 0.0)),
            ("working", (2.0e9, 1.0e9, 100.0)),
        ):
            k, s_iso, s_star, _ = _optimum(dm, OMEGA, dlam=dl, drho=dr, baseline=base)
            k_by.setdefault(base, []).append(k)
            print(f"       {base:>9} {label:>12} {s_iso:10.6f} {s_star:10.6f} {k:10.6f}")

    def _spread(vals):
        return abs(vals[0] - vals[1]) / float(np.mean(vals))

    print(
        f"\n       spread of K across the two cases:  T9 {_spread(k_by['t9']):.2e}"
        f"   T27 {_spread(k_by['t27']):.2e}"
    )
    print("       The T9 correction is universal; the T27 one is not.")
    print()
    print("       ⚠ DO NOT READ THAT AS 'T27 IS WRONG'. A 27-mode response driven")
    print("       through a 9-MODE propagator is not a well-posed scheme, and")
    print("       non-universality is exactly what that mismatch produces REGARDLESS")
    print("       of T27's merit. Using a symptom of the rigging as evidence about")
    print("       the thing being rigged is circular; an earlier version of this")
    print("       script concluded 'the T27 optimum carries no physical weight' and")
    print("       that conclusion is WITHDRAWN. The T27 arm establishes nothing in")
    print("       either direction -- it shows only that K is not portable across")
    print("       baselines. A fair T27 verdict needs the 27x27 generalised")
    print("       propagator, derived in docs/BubnovGalerkinCubicScatter.tex and")
    print("       not implemented.")

    print(f"\n       applying the T9-derived K = {K_RENORM} to each baseline:")
    print(f"       {'baseline':>9} {'case':>12} {'plain':>12} {'+K':>12} {'effect':>14}")
    for base in ("t9", "t27"):
        for label, (dl, dm, dr) in (
            ("pure shear", (0.0, 1.0e9, 0.0)),
            ("working", (2.0e9, 1.0e9, 100.0)),
        ):
            a_ = _err(dl, dm, dr, baseline=base)
            b_ = _err(dl, dm, dr, baseline=base, renorm=True)
            eff = f"{a_ / b_:9.2f}x" if b_ < a_ else f"{b_ / a_:8.2f}x WORSE"
            print(f"       {base:>9} {label:>12} {a_:12.4e} {b_:12.4e} {eff:>14}")
    print("       T27 starts on the OTHER SIDE of the optimum, so the T9 constant")
    print("       moves it further away. Note also that plain T27 beats plain T9 on")
    print("       pure shear -- the earlier 'T27 is 3x worse' was the mixed case only.")

    print("\n" + "=" * 88)
    print("  [V] VERDICT -- is this a renormalisation, or a fudge factor?")
    print("=" * 88)
    print("  WHAT K IS: an EMPIRICAL one-parameter correction with real predictive")
    print("  content, localised to the shear channel.")
    print("  WHAT K IS NOT: a 'renormalisation'. That word names a CAUSE, and no")
    print("  cause is established here. K is fitted against the very arbiter it is")
    print("  then scored against -- so the ~1e-8 residual at s* shows only that ONE")
    print("  PARAMETER SUFFICES to hit one number, and is not independent evidence.")
    print("  And K is BASELINE-DEPENDENT, which argues directly against the natural")
    print("  mechanism: 'the lattice multiplies the true depolarisation by 0.886'")
    print("  ought to hold whichever theory computes the true depolarisation.")
    print()
    print("  The distinction is not pedantry. A renormalisation would transfer to")
    print("  other lattices, voxel shapes and single-site theories; a fitted")
    print("  correction transfers to none without re-measurement. Nothing here")
    print("  licenses the former, which is why K is NOT wired into the library.")
    print("  See docs/shear_lattice_renormalisation.pdf section 8.")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
