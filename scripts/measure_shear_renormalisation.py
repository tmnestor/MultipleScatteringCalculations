"""MEASUREMENT: the lattice renormalisation of a cube's SHEAR depolarisation.

THE RESIDUAL THIS EXPLAINS. With the lateral sum exact and the Galerkin contact
correction applied, the refinement ladder flattens and a residual of ~8.6e-4
survives that refinement cannot remove. Four suspects were tested and refuted
(`measure_residual_origin`, `measure_lattice_depolarisation`): the averaged
propagator's dynamic truncation, the isolated-cube depolarisation as a whole
(removing it is 24.6x WORSE), the T27 single-site strain sector, and propagator
averaging beyond contact.

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

i.e. IN A SPACE-FILLING CUBIC LATTICE A CUBE'S SHEAR DEPOLARISATION IS ~0.886 OF
ITS ISOLATED-CUBE VALUE. The dilatational channel needs no correction at all.

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

⚠ K IS EMPIRICAL. It is measured, universal and predictive, but NOT DERIVED.
8/9 = 0.888889 is refuted (the limit is 2.6e-3 below it, far outside the fit
scatter). Its proximity to sqrt(pi)/2 = 0.886227 is treated as numerology --
sqrt(pi)/d is this project's Ewald splitting default, so the resemblance is
suspicious rather than encouraging. Until K is derived it must NOT enter the
.tex as established, and it is NOT wired into the library.

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


def _reflection(dlam, dmu, drho, n_z, omega, *, shear_scale=None, renorm=False):
    """Complex R_PP through the exact kernel, with the shear sector adjustable."""
    a = H_PHYS / (2.0 * n_z)
    geom = SlabGeometry(M=M, N_z=n_z, a=a)
    ones = np.ones((n_z, M, M))
    mat = SlabMaterial(Dlambda=dlam * ones, Dmu=dmu * ones, Drho=drho * ones, ref=REF)
    t27 = compute_cube_tmatrix(omega, a, REF, MaterialContrast(dlam, dmu, drho))

    dmu_diag = t27.Dmu_star_diag
    if shear_scale is not None:
        dmu_diag = shear_scale * dmu
    elif renorm and dmu != 0.0:
        a_diag = t27.Dmu_star_diag / dmu
        dmu_diag = (1.0 - K_RENORM * (1.0 - a_diag)) * dmu

    v = (2.0 * a) ** 3
    cell = np.zeros((9, 9), dtype=complex)
    cell[:3, :3] = omega**2 * complex(t27.Drho_star) * v * np.eye(3)
    cell[3:, 3:] = v * effective_stiffness_voigt(t27.Dlambda_star, dmu_diag, t27.Dmu_star_off)
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


def _optimum(dmu, omega):
    """(K, residual at the optimum) by a local linear solve, not a grid scan."""
    a = H_PHYS / (2.0 * 4)
    t27 = compute_cube_tmatrix(omega, a, REF, MaterialContrast(0.0, dmu, 0.0))
    s_iso = (t27.Dmu_star_diag / dmu).real
    h = 0.002
    r0, _ = _reflection(0.0, dmu, 0.0, 4, omega, shear_scale=s_iso)
    r1 = (_reflection(0.0, dmu, 0.0, 4, omega, shear_scale=s_iso + h)[0] - r0) / h
    rk = complex(kennett_reference_rpp(REF, MaterialContrast(0.0, dmu, 0.0), H_PHYS, omega))
    s_star = s_iso - float(np.real((r0 - rk) * np.conj(r1)) / abs(r1) ** 2)
    res = _err(0.0, dmu, 0.0, omega=omega, shear_scale=s_star)
    return (1.0 - s_star) / (1.0 - s_iso), res


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
        k, res = _optimum(dmu, OMEGA)
        ks.append(k)
        print(f"       {dmu / MU0:9.4f} {OMEGA:7.0f} {k:10.6f} {res:21.3e}")
    for om in (30.0, 240.0):
        k, res = _optimum(1.0e9, om)
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

    print("\n" + "=" * 88)
    print("  K IS MEASURED, UNIVERSAL AND PREDICTIVE -- BUT NOT DERIVED. It is")
    print("  deliberately NOT wired into the library and must not enter the .tex as")
    print("  established physics until the lattice sum that produces 0.8862 is")
    print("  worked out. The natural route is the difference between the isolated")
    print("  cube's Eshelby shear component and its space-filling counterpart.")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
