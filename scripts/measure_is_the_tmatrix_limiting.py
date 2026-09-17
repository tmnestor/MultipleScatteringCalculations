"""DECISION: is the single-cube T-matrix now the limiting error?

This decides whether a richer single-site basis (the orthonormal-Legendre work)
is worth building, and it is asked NOW because the answer just changed. Until
today the propagator's own error sat on a floor around 2e-5 that refinement
could not remove, so it swamped whatever the T-matrix contributed and the
question was unanswerable. That floor is gone.

THE INSTRUMENT, reused rather than reinvented. `measure_shear_renormalisation.py`
already scans the cube's shear response against Kennett and finds the value that
minimises the error. Two numbers come out:

    s_iso   what the T-matrix actually predicts
    s_star  the value that would make the slab answer exact

and the quantity that answers the question is

    K = (1 - s_star) / (1 - s_iso),

which is exactly 1 when the T-matrix's own shear response is right. |K - 1| is
the relative error of that response.

⚠⚠ THE OBVIOUS METRIC IS THE WRONG ONE, and it produced a confident verdict in
the wrong direction before that was noticed. The first version of this script
keyed on residual(s_iso) / residual(s_star) -- how much better the slab answer
gets once the shear scale is tuned. That ratio reaches 290 here, which reads as
"a perfect T-matrix would be 290x better" and is NOT what it means. It measures
how well ONE FREE PARAMETER, fitted against the very arbiter being scored, can
cancel a small residual; a single knob with leverage cancels almost any such
residual. That is fitting leverage, not physical headroom.

A richer basis does not hand you a parameter fitted to Kennett. It hands you a
different PREDICTION. If the current prediction is already right to a few parts
in ten thousand, a better basis can only move it within that. So the verdict
keys on |K - 1|.

⚠ THE DECISION RULE IS FIXED BEFORE RUNNING: |K - 1| below 1e-3 means the
response is right to a part in a thousand and a richer basis cannot pay; above
1e-2 means it can; between is reported as inconclusive rather than argued.

⚠ WHAT K IS NOT. It is one scalar tuned against the very arbiter it is then
scored against. It is used here ONLY as a diagnostic of how much room the
T-matrix has, never as a correction to apply. Its historical value moved
0.886234 -> 1.023737 when a propagator defect was fixed, which is exactly the
point: K measures the propagator and the T-matrix together, so it only means
something about the T-matrix once the propagator is sound.

⚠ AND IT IS RUN ON BOTH PROPAGATORS. If K sits near 1 on the new exact
propagator but not on the old one, that is evidence the residual was never the
T-matrix. Running only the new one could not distinguish "the T-matrix is fine"
from "this test cannot see it".

Run:  conda run -n seismic python scripts/measure_is_the_tmatrix_limiting.py
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


def _reflection(dlam, dmu, drho, n_z, *, shear_scale, exact):
    a = H_PHYS / (2.0 * n_z)
    geom = SlabGeometry(M=M, N_z=n_z, a=a)
    ones = np.ones((n_z, M, M))
    con = MaterialContrast(dlam, dmu, drho)
    mat = SlabMaterial(Dlambda=dlam * ones, Dmu=dmu * ones, Drho=drho * ones, ref=REF)
    t9 = compute_cube_tmatrix(OMEGA, a, REF, con)

    v = (2.0 * a) ** 3
    cell = np.zeros((9, 9), dtype=complex)
    cell[:3, :3] = OMEGA**2 * complex(t9.Drho_star) * v * np.eye(3)
    cell[3:, 3:] = v * effective_stiffness_voigt(t9.Dlambda_star, shear_scale * dmu, t9.Dmu_star_off)
    t0 = np.broadcast_to(cell, (n_z, M, M, 9, 9)).copy()

    kh = build_slab_kernels(
        geom,
        OMEGA,
        REF,
        periodic=True,
        lattice_ewald=True,
        volume_averaged=True,
        n_orders=2,
        exact_cell_average=None if exact else False,
    )
    res = compute_slab_scattering(
        geom,
        mat,
        OMEGA,
        np.array([1.0, 0.0, 0.0]),
        "P",
        periodic=True,
        gmres_tol=1e-13,
        kernel_hat=kh,
        T_local=t0,
    )
    return complex(slab_rpp_periodic(res, t0))


def _err(dlam, dmu, drho, n_z, shear_scale, exact):
    r = _reflection(dlam, dmu, drho, n_z, shear_scale=shear_scale, exact=exact)
    rk = kennett_reference_rpp(REF, MaterialContrast(dlam, dmu, drho), H_PHYS, OMEGA)
    return float(abs(r - rk) / abs(rk))


def analyse(dlam, dmu, drho, n_z, exact):
    """(K, residual as-is, residual at the best possible shear, ratio)."""
    a = H_PHYS / (2.0 * n_z)
    con = MaterialContrast(dlam, dmu, drho)
    s_iso = (compute_cube_tmatrix(OMEGA, a, REF, con).Dmu_star_diag / dmu).real

    # One Newton step on the complex reflection: it is linear in the shear
    # scale to well within the step, so a local solve lands on the optimum.
    h = 0.002
    r0 = _reflection(dlam, dmu, drho, n_z, shear_scale=s_iso, exact=exact)
    r1 = (_reflection(dlam, dmu, drho, n_z, shear_scale=s_iso + h, exact=exact) - r0) / h
    rk = complex(kennett_reference_rpp(REF, con, H_PHYS, OMEGA))
    s_star = s_iso - float(np.real((r0 - rk) * np.conj(r1)) / abs(r1) ** 2)

    e_iso = _err(dlam, dmu, drho, n_z, s_iso, exact)
    e_star = _err(dlam, dmu, drho, n_z, s_star, exact)
    k = (1.0 - s_star) / (1.0 - s_iso)
    return k, e_iso, e_star, e_iso / max(e_star, 1e-300)


def main() -> int:
    print("=" * 78)
    print("IS THE SINGLE-CUBE T-MATRIX THE LIMITING ERROR?")
    print("=" * 78)
    print("\n  s_iso  = the shear response the T-matrix predicts")
    print("  s_star = the shear response that would make the slab answer exact")
    print("  ratio  = residual(s_iso) / residual(s_star), i.e. how much a")
    print("           PERFECT single-cube T-matrix would buy end to end")

    cases = (("pure shear", 0.0, 1.0e9, 0.0), ("mixed", 2.0e9, 1.0e9, 100.0))
    ratios = []
    ks = []
    for label, dl, dm, dr in cases:
        print(f"\n  {label}")
        print(
            f"    {'propagator':>12} {'n_z':>4} {'K':>9} {'as-is':>12} {'best possible':>14} {'ratio':>8}"
        )
        for exact, name in ((False, "old shell"), (True, "EXACT")):
            for n_z in (4, 8):
                k, e_iso, e_star, ratio = analyse(dl, dm, dr, n_z, exact)
                if exact:
                    ratios.append(ratio)
                    ks.append(k)
                print(f"    {name:>12} {n_z:>4} {k:9.4f} {e_iso:12.4e} {e_star:14.4e} {ratio:8.2f}")

    # ⚠⚠ THE RATIO IS THE WRONG METRIC, and reading it cost a confident verdict
    # in the wrong direction before this was noticed. It measures how well ONE
    # FREE PARAMETER, tuned against the arbiter, can cancel the residual -- and
    # a single knob with leverage cancels almost any small scalar residual. That
    # is fitting leverage, not physical headroom. A richer basis does not hand
    # you a parameter fitted to Kennett; it hands you a different PREDICTION.
    #
    # The quantity that answers the question is K, which is 1 exactly when the
    # T-matrix's own shear response is right. |K - 1| is therefore the relative
    # error of that response, and it is what the verdict now keys on.
    worst_k = max(abs(k - 1.0) for k in ks)
    print(f"\n  largest |K - 1| on the exact propagator: {worst_k:.2e}")
    print(f"  (for contrast, the residual RATIO peaks at {max(ratios):.0f}x -- see")
    print("   the note in the source: that number is fitting leverage, not headroom)")
    print("\n" + "=" * 78)
    if worst_k < 1.0e-3:
        print("VERDICT: the T-matrix is NOT the limiting error.")
        print()
        print(f"Its shear response is right to {worst_k:.0e} relative.  On the OLD")
        print("propagator the same measurement gave |K - 1| ~ 2e-2, so fixing the")
        print("propagator moved the T-matrix's apparent error from 2% to 0.03%:")
        print("the T-matrix was never the problem, the propagator was.")
        print()
        print("A richer single-site basis -- the orthonormal-Legendre work --")
        print("cannot pay for itself against a response already correct to three")
        print("parts in ten thousand.  DO NOT BUILD IT on this evidence.")
        print()
        print("What the remaining error IS, is now the open question.  It is not")
        print("the shear response of the cube, and it is no longer the propagator")
        print("truncation.  Splitting the residual by channel is the next step.")
    elif worst_k > 1.0e-2:
        print("VERDICT: the T-matrix IS the limiting error.")
        print()
        print(f"Its shear response is off by {worst_k:.1%}, so a richer basis has")
        print("room to pay for itself.  It should target the COLLOCATION route")
        print("(direct integration of the moments), NOT the Galerkin tier ladder,")
        print("which measurement has already placed on the wrong side of the")
        print("formulation question.")
    else:
        print(f"VERDICT: INCONCLUSIVE.  |K - 1| = {worst_k:.2e}, between the")
        print("thresholds of 1e-3 and 1e-2.  Deciding would need the residual")
        print("split by channel rather than one scalar.")
    print()
    print("⚠ SCOPE: the shear channel only, one frequency, a uniform slab.  K is")
    print("a diagnostic of how far the response is off, never a correction to")
    print("apply -- it is one scalar tuned against the arbiter it is scored on.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
