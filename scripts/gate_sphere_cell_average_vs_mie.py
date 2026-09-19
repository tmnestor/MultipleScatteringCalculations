#!/usr/bin/env python3
"""Re-measure the voxel T-matrix against exact Mie, with the propagator fixed.

WHY THIS EXISTS
---------------
The sphere route's agreement with elastic Mie was last measured in June 2026:
about 4-5% at ka = 0.1, 19-23% at ka = 0.5, and 32-57% at ka = 1.5.  Those
numbers are stale.  The propagator defects they were measuring were found and
fixed in September -- the contact convention, the double-versus-single average,
and finally the exact cell average -- but every one of those commits landed in
the Bloch/slab route (``slab_scattering``, ``cell_averaged_lattice``,
``lattice_kupradze``, ``sweep_kernels``).  ``git log`` shows the sphere route
untouched since 2026-06-17.  So the sphere has been carrying the old propagator
for three months, and any comparison against it has been scoring a defect.

THE UNMATCHED PAIR
------------------
``scripts/settle_single_site_formulation.py`` settles that the single-site
closure is COLLOCATION: T9 closes on the cube-centre value through a SINGLE
volume integral.  The propagator it is used with must close on the same
quantity.  The sphere route pairs that T9 with a POINT propagator -- not the
double average, but no average at all.  ``cell_averaged_pair`` supplies the
matching half: the single (sinc^1) average over the receiver cell.

That is the same mismatch that made the shear correction K look like physics.
K measured 0.886234, then 1.023737 once the contact convention was fixed, then
1.0000 once the cell average was made exact -- it was never a correction, it was
a propagator defect measured twice over.  The prediction here is therefore
specific and falsifiable: matching the pair should move the sphere towards Mie,
and if it does not, the collocation settlement does not transfer to this
geometry and should be said so.

WHAT IS MEASURED, AND WHAT IS NOT
---------------------------------
A voxelised sphere is a STAIRCASE, and the staircase turns out to dominate
everything an n_sub ladder can see.  A cubic grid's coverage of a sphere is not
even monotonic in n_sub: the voxel volume is 26%, 4.7% and 17% too large at
n_sub = 3, 4, 6, and the RAW error tracks that dilution to within a factor 1.4
(0.35, 0.04, 0.21 against 0.256, 0.047, 0.168).  So a refinement ratio in n_sub
measures the geometry, not the propagator, and this gate's first version was
wrong to ask for one -- it duly failed its own check while the propagator
result underneath was clean.

What IS attributable to the propagator is the comparison AT FIXED GEOMETRY:
identical voxel set, identical staircase, one propagator swapped.  Part 5
reports that ratio at each resolution and requires it to be STABLE across them,
which is the evidence that the improvement is real rather than a coincidence at
one mesh.  Three independent resolutions agreeing is a stronger claim than a
refinement ratio, not a weaker one.

The volume correction is still reported, because it removes the dilution and so
brings the residual closer to what the propagator owns.  It does NOT remove the
shape error -- see the sphere-packing record for that separation.

Run:  conda run -n seismic python scripts/gate_sphere_cell_average_vs_mie.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubic_scattering.cell_averaged_pair import (  # noqa: E402
    auto_n_gauss,
    averaged_pair_block_9x9,
    tail_pair_block_9x9,
)
from cubic_scattering.effective_contrasts import (  # noqa: E402
    MaterialContrast,
    ReferenceMedium,
)
from cubic_scattering.kupradze_derivatives import (  # noqa: E402
    propagator_block_9x9_kupradze,
)
from cubic_scattering.resonance_tmatrix import _propagator_block_9x9  # noqa: E402
from cubic_scattering.sphere_scattering import (  # noqa: E402
    compute_elastic_mie,
    compute_sphere_foldy_lax,
    foldy_lax_far_field,
    mie_scattered_displacement,
)

#: The project's validated test parameters.
REF = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
CONTRAST = MaterialContrast(Dlambda=2.0e9, Dmu=1.0e9, Drho=100.0)

#: Observation distance, in sphere radii.  ``foldy_lax_far_field`` returns the
#: ASYMPTOTIC field and ``mie_scattered_displacement`` the exact one, so the
#: comparison is only meaningful where the neglected 1/(k r) term has died.
#: This gate's first run inherited 500 from the existing sphere tests, whose
#: only bar is a magnitude ratio in (0.1, 10), and duly reported a flat 5.6%
#: error that refused to move under refinement.  Measured at ka = 0.1,
#: n_sub = 4:
#:
#:     r/a = 100 -> 0.5052    500 -> 0.0562    5e3 -> 0.0088    5e4 -> 0.0088
#:
#: so 500 was reporting the harness, not the method, and the real error was
#: six times smaller.  Part 5 re-checks this convergence every run rather than
#: trusting the constant.
R_MULT = 5.0e4

_PASS: list[tuple[str, bool]] = []


def report(label: str, ok: bool) -> None:
    """Record and print one check.

    Args:
        label: Description.
        ok: Whether it passed.
    """
    _PASS.append((label, bool(ok)))
    print(f"  {'PASS' if ok else '****FAIL****'}  {label}")


def rel(a: NDArray, b: NDArray) -> float:
    """Relative difference of two arrays in the max norm.

    Args:
        a: First.
        b: Second.

    Returns:
        max|a-b| / max|b|.
    """
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-300))


def obs_points(r_distance: float, theta: NDArray) -> NDArray:
    """Observation points in the z-x plane, matching the existing sphere tests.

    Args:
        r_distance: Distance from the origin.
        theta: Scattering angles, radians.

    Returns:
        Shape (len(theta), 3), with z = index 0 and x = index 1.
    """
    pts = np.zeros((len(theta), 3))
    pts[:, 0] = r_distance * np.cos(theta)
    pts[:, 1] = r_distance * np.sin(theta)
    return pts


def pattern_error(
    omega: float,
    radius: float,
    n_sub: int,
    *,
    cell_average: bool,
    n_gauss: int | None = None,
    r_mult: float = R_MULT,
) -> tuple[float, float, int]:
    """Foldy-Lax far field against Mie, as a fraction of the pattern peak.

    Args:
        omega: Angular frequency.
        radius: Sphere radius.
        n_sub: Sub-cells per bounding-cube edge.
        cell_average: Use the single receiver-cell average.
        n_gauss: Gauss points per axis for that average.
        r_mult: Observation distance in sphere radii.  See ``R_MULT``.

    Returns:
        (raw error, volume-corrected error, cell count).
    """
    k_hat = np.array([1.0, 0.0, 0.0])
    pol = np.array([1.0, 0.0, 0.0])
    mie = compute_elastic_mie(omega, radius, REF, CONTRAST)
    fl = compute_sphere_foldy_lax(
        omega,
        radius,
        REF,
        CONTRAST,
        n_sub=n_sub,
        k_hat=k_hat,
        wave_type="P",
        cell_average=cell_average,
        n_gauss=n_gauss,
    )

    theta = np.linspace(0.2, np.pi - 0.2, 9)
    r_far = r_mult * radius
    pts = obs_points(r_far, theta)
    u_mie = mie_scattered_displacement(mie, pts)
    u_p, u_s = foldy_lax_far_field(fl, pts / r_far, r_far, k_hat, pol, wave_type="P")
    u_fl = u_p + u_s

    peak = float(np.max(np.abs(u_mie)))
    raw = float(np.max(np.abs(u_fl - u_mie))) / peak

    # The staircase: the voxelised sphere's volume is not the sphere's.  Scaling
    # by the ratio removes the leading geometric error so that what remains is
    # closer to the propagator's own contribution.  It does NOT remove the
    # shape error, only the dilution -- see the sphere-packing record.
    vol_voxel = fl.n_cells * (2.0 * fl.a_sub) ** 3
    vol_true = 4.0 / 3.0 * np.pi * radius**3
    corrected = float(np.max(np.abs(u_fl * (vol_true / vol_voxel) - u_mie))) / peak
    return raw, corrected, fl.n_cells


def part1() -> None:
    """The point propagator, by two independent routes."""
    print("\n[1] the base: two independent routes to the point propagator")
    omega = 300.0
    worst = 0.0
    for s in ([2.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [3.0, -2.0, 1.0]):
        s_vec = np.array(s)
        worst = max(
            worst,
            rel(
                propagator_block_9x9_kupradze(s_vec, omega, REF),
                _propagator_block_9x9(s_vec, omega, REF),
            ),
        )
    print(f"      Kupradze ladder vs hand-derived, worst of 4: {worst:.3e}")
    report("the derivative ladder reproduces the validated point propagator", worst < 1e-10)


def part2() -> None:
    """The single average is well-formed and regular where the double one is not."""
    print("\n[2] the single average: convergence and regularity")
    omega, d = 300.0, 0.25

    # The integrand is regular but PEAKED near the closest face, and Gauss
    # converges slowly against a nearby peak.  The three contact geometries put
    # that peak at 0.5 d, 0.707 d and 0.866 d respectively, and they converge at
    # correspondingly different rates -- which is why a single fixed order is
    # wrong here, and why ``auto_n_gauss`` exists.
    contacts = {
        "face": np.array([d, 0.0, 0.0]),
        "edge": np.array([d, d, 0.0]),
        "corner": np.array([d, d, d]),
    }
    gauss_orders = (4, 8, 12, 16)
    print("      convergence to a 24-point reference   " + "  ".join(f"g={g:<2d}" for g in gauss_orders))
    worst_auto = 0.0
    for name, s in contacts.items():
        ref_val = averaged_pair_block_9x9(s, omega, REF, d, n_gauss=24)
        errs = [rel(averaged_pair_block_9x9(s, omega, REF, d, n_gauss=g), ref_val) for g in gauss_orders]
        chosen = auto_n_gauss(s, d)
        auto_err = rel(averaged_pair_block_9x9(s, omega, REF, d, n_gauss=None), ref_val)
        worst_auto = max(worst_auto, auto_err)
        print(
            f"        {name:7s} "
            + "  ".join(f"{e:.1e}" for e in errs)
            + f"   auto picks g={chosen:2d} -> {auto_err:.1e}"
        )
        report(f"the average converges in n_gauss at {name} contact", errs[-1] < 1e-8)
        # FINITE at contact -- the property the double average lacks, because a
        # point source never reaches inside the receiver cell.
        report(f"the single average is finite at {name} contact", bool(np.all(np.isfinite(ref_val))))
    report("the adaptive order reaches the reference everywhere", worst_auto < 1e-8)

    # And it must reduce to the point propagator as the cell shrinks at fixed
    # separation, at SECOND order -- <g> - g = (d^2/24) grad^2 g.
    s_fix = np.array([1.0, 0.0, 0.0])
    errs = []
    for dd in (0.2, 0.1, 0.05):
        errs.append(
            rel(
                averaged_pair_block_9x9(s_fix, omega, REF, dd, n_gauss=8),
                _propagator_block_9x9(s_fix, omega, REF),
            )
        )
    orders = [float(np.log(errs[i] / errs[i + 1]) / np.log(2.0)) for i in range(len(errs) - 1)]
    print(f"      reduction to the point value: {'  '.join(f'{e:.2e}' for e in errs)}")
    print(f"      order: {'  '.join(f'{o:.2f}' for o in orders)}")
    report("the average reduces to the point propagator at second order", 1.7 < orders[-1] < 2.3)


def part3() -> None:
    """Direct quadrature against the closed-form d^2 tail."""
    print("\n[3] two routes to the same average: direct quadrature vs the d^2 tail")
    omega = 300.0
    s_fix = np.array([1.0, 0.0, 0.0])

    # Away from contact the two must agree to O(d^4), so halving d must cut the
    # disagreement by 16.  That is a joint test: it fails if EITHER route is
    # wrong, and it is the sharp check on the tail construction.
    diffs = []
    for dd in (0.2, 0.1, 0.05):
        diffs.append(
            rel(
                averaged_pair_block_9x9(s_fix, omega, REF, dd, n_gauss=8),
                tail_pair_block_9x9(s_fix, omega, REF, dd),
            )
        )
    orders = [float(np.log(diffs[i] / diffs[i + 1]) / np.log(2.0)) for i in range(len(diffs) - 1)]
    print(f"      |direct - tail| at d = 0.2, 0.1, 0.05: {'  '.join(f'{x:.2e}' for x in diffs)}")
    print(f"      order: {'  '.join(f'{o:.2f}' for o in orders)}")
    report("direct and tail agree at FOURTH order in the cell size", 3.5 < orders[-1] < 4.5)

    # Near contact the expansion stops being good.  Measuring where is the point
    # -- it says how large a near shell a tail-based implementation would need.
    print("      where the tail stops being usable (d = 0.25):")
    d = 0.25
    for cells in (1, 2, 3, 4, 6):
        s = np.array([cells * d, 0.0, 0.0])
        gap = rel(
            averaged_pair_block_9x9(s, omega, REF, d, n_gauss=8),
            tail_pair_block_9x9(s, omega, REF, d),
        )
        print(f"        separation {cells} cell(s): {gap:.3e}")
    near = rel(
        averaged_pair_block_9x9(np.array([d, 0.0, 0.0]), omega, REF, d, n_gauss=8),
        tail_pair_block_9x9(np.array([d, 0.0, 0.0]), omega, REF, d),
    )
    far = rel(
        averaged_pair_block_9x9(np.array([6 * d, 0.0, 0.0]), omega, REF, d, n_gauss=8),
        tail_pair_block_9x9(np.array([6 * d, 0.0, 0.0]), omega, REF, d),
    )
    report("the tail is much worse at contact than far away", near > 50.0 * far)


def part4() -> None:
    """The averaged propagator changes the answer, and the solve stays sane."""
    print("\n[4] the two propagators are genuinely different objects")
    omega, radius, n_sub = 300.0, 10.0, 4
    a_sub = radius / n_sub
    d = 2.0 * a_sub
    s = np.array([d, 0.0, 0.0])
    point = _propagator_block_9x9(s, omega, REF)
    avg = averaged_pair_block_9x9(s, omega, REF, d, n_gauss=None)
    diff = rel(avg, point)
    print(f"      at nearest-neighbour separation, |avg - point|/|point| = {diff:.3e}")
    report("the average differs from the point value where it matters", diff > 1e-3)

    fl_pt = compute_sphere_foldy_lax(
        omega, radius, REF, CONTRAST, n_sub=n_sub, wave_type="P", cell_average=False
    )
    fl_av = compute_sphere_foldy_lax(
        omega, radius, REF, CONTRAST, n_sub=n_sub, wave_type="P", cell_average=True
    )
    print(f"      cond(I - PT): point {fl_pt.condition_number:.3e}   averaged {fl_av.condition_number:.3e}")
    report(
        "the averaged solve is no worse conditioned", fl_av.condition_number < 10.0 * fl_pt.condition_number
    )
    report("the composite T-matrices differ", rel(fl_av.T3x3, fl_pt.T3x3) > 1e-4)


def part5() -> None:
    """The headline: against exact Mie, before and after."""
    print("\n[5] against exact elastic Mie -- the measurement this gate exists for")
    radius = 10.0

    # THE HARNESS FIRST.  None of the numbers below mean anything until the
    # comparison itself has converged: one side is asymptotic and the other
    # exact, so too small an observation distance measures the missing 1/(k r)
    # term and nothing else.  Checked every run, because that is exactly how
    # this gate's first version came to report a flat 5.6%.
    omega_chk = 0.1 * REF.beta / radius
    e_lo = pattern_error(omega_chk, radius, 4, cell_average=False, r_mult=R_MULT / 10.0)[1]
    e_hi = pattern_error(omega_chk, radius, 4, cell_average=False, r_mult=R_MULT)[1]
    print(
        f"      far-field convergence: r/a = {R_MULT / 10:.0e} -> {e_lo:.4f},  {R_MULT:.0e} -> {e_hi:.4f}"
    )
    report(
        "the far-field limit is reached, so this measures the method not the harness",
        abs(e_hi - e_lo) < 0.05 * max(e_hi, 1e-12),
    )

    for ka in (0.1, 0.5):
        omega = ka * REF.beta / radius
        print(f"\n      ka = {ka}")
        print(
            f"        {'n_sub':>5} {'N':>5} {'point raw':>11} {'avg raw':>11}"
            f" {'point corr':>11} {'avg corr':>11} {'dilution':>9}"
        )
        raw_pt, raw_av, corr_pt, corr_av, dilution = [], [], [], [], []
        subs = (3, 4, 6)
        for n_sub in subs:
            t0 = time.perf_counter()
            rp, cp, ncell = pattern_error(omega, radius, n_sub, cell_average=False)
            ra, ca, _ = pattern_error(omega, radius, n_sub, cell_average=True)
            raw_pt.append(rp)
            raw_av.append(ra)
            corr_pt.append(cp)
            corr_av.append(ca)
            vol_voxel = ncell * (2.0 * radius / n_sub) ** 3
            dilution.append(abs(1.0 - (4.0 / 3.0 * np.pi * radius**3) / vol_voxel))
            print(
                f"        {n_sub:5d} {ncell:5d} {rp:11.4f} {ra:11.4f} {cp:11.4f} {ca:11.4f}"
                f" {dilution[-1]:9.4f}   ({time.perf_counter() - t0:.1f} s)"
            )

        # The raw error is the STAIRCASE, not the method.  Showing that it
        # tracks the dilution is what licenses the volume correction and what
        # explains the otherwise alarming 0.35 -> 0.04 -> 0.21 excursion.
        track = [r / max(dd, 1e-30) for r, dd in zip(raw_pt, dilution, strict=True)]
        print(f"        raw error / dilution: {'  '.join(f'{t:.2f}' for t in track)}")
        report(
            f"the raw error is dominated by the voxel dilution, ka={ka}",
            all(0.5 < t < 2.0 for t in track),
        )

        # REFINEMENT IN n_sub IS NOT AVAILABLE HERE, and demanding it was this
        # gate's own mistake.  A cubic grid's coverage of a sphere is not
        # monotonic in n_sub: the voxel volume is 26%, 4.7% and 17% wrong at
        # n_sub = 3, 4, 6, and the raw error tracks that dilution to within a
        # factor 1.4 (0.35, 0.04, 0.21 against 0.256, 0.047, 0.168).  So the
        # geometry, not the propagator, sets how the error moves with n_sub,
        # and an n_sub ladder measures the staircase.
        #
        # What IS attributable to the propagator is the comparison AT FIXED
        # GEOMETRY -- identical voxel set, identical staircase, one propagator
        # swapped.  Its stability across three independent resolutions is the
        # evidence that the improvement is real rather than a coincidence at
        # one mesh, and it is a stronger statement than a refinement ratio
        # would have been, not a weaker one.
        ratios = [p / max(a, 1e-30) for p, a in zip(corr_pt, corr_av, strict=True)]
        print("        improvement (point / averaged) at fixed geometry, per n_sub:")
        print("          " + "   ".join(f"n_sub={s}: {r:.2f}x" for s, r in zip(subs, ratios, strict=True)))
        spread = (max(ratios) - min(ratios)) / max(float(np.mean(ratios)), 1e-30)
        print(f"        spread across resolutions: {spread:.1%}")
        report(
            f"the averaged propagator is closer to exact Mie at every resolution, ka={ka}",
            all(r > 1.0 for r in ratios),
        )
        report(
            f"and by a factor stable across resolutions, ka={ka}",
            spread < 0.25,
        )


def main() -> int:
    """Run every part and summarise.

    Returns:
        0 if all checks pass, 1 otherwise.
    """
    print("=" * 78)
    print("  The voxel T-matrix against exact Mie, with the single cell average")
    print("  SI units; time convention e^{-i omega t}")
    print("=" * 78)
    for part in (part1, part2, part3, part4, part5):
        part()
    ok = sum(1 for _, p in _PASS if p)
    print("\n" + "=" * 78)
    for label, passed in _PASS:
        if not passed:
            print(f"  FAILED: {label}")
    print(f"  {ok}/{len(_PASS)} checks passed")
    print("=" * 78)
    return 0 if ok == len(_PASS) else 1


if __name__ == "__main__":
    sys.exit(main())
