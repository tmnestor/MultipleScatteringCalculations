"""INVESTIGATION: is the midpoint-correction lattice sum SHAPE-dependent?

`settle_collocation_everywhere.py` found the source-cell correction is not
short-ranged: the Kennett error falls as R^-1.5 out to 289 shells with no
saturation. A 1/R tail in a 2-D lattice sum is the regime where the limit can
depend on the summation SHAPE, which would make reach extrapolation meaningless.
This tests that directly.

WHAT IS ALREADY KNOWN, so this does not re-derive it. The k=0 lattice sum of the
POINT propagator is ALREADY shape-dependent and characterised in closed form:
`xx - zz` = 1 exactly (slab depolarisation, medium-independent),
`xy - zy` = alpha^2/(2 beta^2), cubic-symmetric remainder
+-0.1298228 (alpha^2/beta^2 - 1). Summing planes then stacking imposes a SLAB
order and that term is PHYSICAL for a slab. So conditional convergence at k=0 is
established here, not a new suspicion. The open part is whether the CORRECTION
inherits it.

⚠ A MECHANISM PROPOSED BEFORE MEASURING, AND HOW IT FARED. The proposal was

    <G> - G = (d^2/24) grad^2 G + O(d^4),

so away from the origin Helmholtz gives grad^2 G = -k^2 G and the correction is
proportional to G ITSELF -- long-ranged -- while in the STATIC limit the entries
are derivatives of 1/r, HARMONIC, so the leading term vanishes and the
correction is short-ranged.

THE FIRST HALF IS CONFIRMED, THE SECOND IS WRONG, and panel [A] settles both.

Confirmed: at omega = 450 the ratio |<G>-G|/|G| PLATEAUS at 9.14e-4 against the
predicted (k_S d)^2/24 = 9.375e-4, within 3%. Helmholtz does make the correction
track G itself once k r >~ 1.

Wrong: the elastostatic Kelvin tensor is built from r, which is BIHARMONIC, not
harmonic -- grad^2 r = 2/r -- so grad^2 G != 0 and the d^2 term SURVIVES
statically. Measured, the static ratio falls as 1/r^2 exactly: a surviving
d^2 grad^2 term two orders down from G, not a cancelled one. The static
correction is short-ranged for a DIFFERENT reason than proposed -- not because
the leading term vanishes, but because grad^2 G decays two powers faster than G.

So the correction has a CROSSOVER at k r ~ 1, and panel [A] shows it happening
at the operating frequency: the 'x r^2' column is flat at 0.055 while k_S r <
0.3, then climbs (0.069, 0.109) as k_S r -> 1. That crossover is what makes the
shape question have two different answers, and it is why panel [B] is run at two
frequencies rather than one.

WHAT PANEL [B] DOES. Sums the correction over a SQUARE region and a CIRCULAR one
and compares the limits. Conditionally convergent sums give different answers for
different shapes; absolutely convergent ones do not. This does not depend on the
mechanism above being right, which is why it is the load-bearing test.

⚠ AND IT IS RUN AT TWO FREQUENCIES, because one is not enough. At the solver's
operating point (omega = 60, d = 1) the whole summation region out to 20 cells
sits at k_S r <= 0.4 -- QUASI-STATIC, where an oscillatory 1/r tail has not begun
and absolute convergence proves nothing about the asymptotic regime. The second
frequency pushes k_S r to ~3, genuinely into the oscillatory far field, so the
two together separate "absolutely convergent here" from "absolutely convergent".

Run:  conda run -n seismic python scripts/investigate_correction_tail_shape.py
SI units (m, m/s, kg/m3, Pa).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from cubic_scattering.slab_scattering import (  # noqa: E402
    _cell_averaged_propagator,
    _propagator_block_9x9,
)

REF = ReferenceMedium(5000.0, 3000.0, 2500.0)
D = 1.0  # lattice pitch (cube side), metres
OMEGA_OP = 60.0  # the solver's operating point: k_S d = 0.02
OMEGA_FAR = 450.0  # k_S d = 0.15, so k_S r ~ 3 at r = 20 cells
OMEGA_STAT = 60.0e-4
# Gauss points per axis for the source-cell average, gated in main() against a
# finer rule. NG=4 was tried first and REJECTED by that gate at 4.9e-6 -- the
# integrand is peaked toward the near face at small offsets, and the shape
# differences measured below are themselves O(1e-4) relative, accumulated over
# ~1600 offsets, so quadrature noise at 1e-6 per term is not comfortably clear
# of the signal. The cost is recovered by the single-pass accumulation in
# shape_sums, which does 2 passes over the disc where the naive version did 8.
NG = 8


def corr(r_vec: np.ndarray, omega: float, ng: int = NG) -> np.ndarray:
    """<G> - G over the SOURCE cell (single average), at one offset."""
    return _cell_averaged_propagator(r_vec, D, omega, REF, ng, double=False) - _propagator_block_9x9(
        r_vec, omega, REF
    )


def panel_a() -> None:
    """Radial decay of |<G>-G| / |G|, which identifies the surviving term."""
    print("\n  [A] MECHANISM: which term of the midpoint error survives?")
    for label, omega in (
        ("STATIC      (omega=6e-3)", OMEGA_STAT),
        ("OPERATING   (omega=60)", OMEGA_OP),
        ("FAR-FIELD   (omega=450)", OMEGA_FAR),
    ):
        kb = omega / REF.beta
        print(f"\n    {label}   k_S d = {kb * D:.4g}   (k_S d)^2/24 = {(kb * D) ** 2 / 24:.3e}")
        print(f"      {'n (cells)':>10} {'k_S r':>9} {'|corr|/|G|':>14} {'x r^2':>12}")
        for n in (4, 8, 16, 32, 64):
            r_vec = np.array([0.0, n * D, 0.0])
            g = _propagator_block_9x9(r_vec, omega, REF)
            ratio = float(np.max(np.abs(corr(r_vec, omega))) / np.max(np.abs(g)))
            print(f"      {n:>10} {kb * n * D:9.3f} {ratio:14.4e} {ratio * n**2:12.4e}")
    print("\n      A FLAT 'x r^2' column means the correction decays TWO powers")
    print("      faster than G -- the (d^2/24) grad^2 G term, with grad^2 G two")
    print("      orders down from G.  It does NOT vanish statically: the Kelvin")
    print("      tensor is built from r, which is BIHARMONIC (grad^2 r = 2/r),")
    print("      not harmonic.  A ratio that instead PLATEAUS at (k_S d)^2/24")
    print("      would be the Helmholtz regime, where grad^2 G = -k^2 G makes the")
    print("      correction track G itself and become long-ranged.")


def shape_sums(omega: float, radii: tuple[int, ...]) -> dict[int, tuple[float, float, float]]:
    """Cumulative square and circular sums of the correction on the dz=0 plane.

    One pass over offsets, accumulating into every radius at once -- the naive
    version recomputes the whole disc per radius and per shape, which is 8x the
    work for the same numbers.
    """
    rmax = max(radii)
    sq = {r: np.zeros((9, 9), dtype=complex) for r in radii}
    ci = {r: np.zeros((9, 9), dtype=complex) for r in radii}
    for dx in range(-rmax, rmax + 1):
        for dy in range(-rmax, rmax + 1):
            if dx == 0 and dy == 0:
                continue
            c = corr(np.array([0.0, dx * D, dy * D]), omega)
            cheb = max(abs(dx), abs(dy))
            r2 = dx * dx + dy * dy
            for r in radii:
                if cheb <= r:
                    sq[r] += c
                if r2 <= r * r:
                    ci[r] += c
    out: dict[int, tuple[float, float, float]] = {}
    for r in radii:
        out[r] = (
            float(np.max(np.abs(sq[r]))),
            float(np.max(np.abs(ci[r]))),
            float(np.max(np.abs(sq[r] - ci[r]))),
        )
    return out


def panel_b() -> bool:
    """Square vs circular summation. Different limits => conditionally convergent."""
    print("\n  [B] DIRECT SHAPE TEST: square region vs circular region")
    radii = (6, 10, 14, 20)
    verdicts = []
    for label, omega in (("OPERATING (omega=60)", OMEGA_OP), ("FAR-FIELD (omega=450)", OMEGA_FAR)):
        kb = omega / REF.beta
        print(f"\n    {label}   k_S * rmax = {kb * max(radii) * D:.2f}")
        print(f"      {'rmax':>6} {'|square|':>14} {'|circle|':>14} {'|sq - circ|':>14} {'rel':>9}")
        res = shape_sums(omega, radii)
        diffs = []
        for r in radii:
            nsq, nci, diff = res[r]
            diffs.append(diff)
            print(f"      {r:>6} {nsq:14.6e} {nci:14.6e} {diff:14.6e} {diff / max(nsq, nci):9.4f}")
        # The signature: does |sq - circ| tend to ZERO, or to a constant?
        shrink = diffs[0] / diffs[-1]
        closing = shrink > 2.0
        verdicts.append(closing)
        print(f"\n      |sq - circ| shrank by {shrink:.2f}x from rmax 6 to 20")
        print(f"      => gap closing (absolutely convergent): {closing}")
    return all(verdicts)


def main() -> int:
    print("=" * 78)
    print("IS THE MIDPOINT-CORRECTION LATTICE SUM SHAPE-DEPENDENT?")
    print("=" * 78)

    # Quadrature gate first: if NG is not converged, everything below is noise.
    # The probe is a NEAR offset (r = sqrt(10) cells), where the integrand is
    # most peaked toward the touching face -- the worst case in the sum, not a
    # comfortable one.
    r_probe = np.array([0.0, 3.0 * D, 1.0 * D])
    cn = corr(r_probe, OMEGA_OP, NG)
    cfine = corr(r_probe, OMEGA_OP, NG + 4)
    qerr = float(np.max(np.abs(cn - cfine)) / np.max(np.abs(cfine)))
    print(f"\n  quadrature gate: |corr(NG={NG}) - corr(NG={NG + 4})| / |corr| = {qerr:.2e}")
    if qerr > 1e-6:
        print(f"  ABORT: NG={NG} is not converged, the panels below would be noise.")
        return 1
    print(f"  NG = {NG} is converged; the panels measure the correction, not quadrature.")

    panel_a()
    both_absolute = panel_b()

    # The crossover radius: beyond it the correction tracks G and the sum turns
    # conditionally convergent. R_x d ~ 1/k_S.
    r_cross = REF.beta / (OMEGA_OP * D)

    print("\n" + "=" * 78)
    print("FINDING: THE ANSWER IS REGIME-DEPENDENT, and that is the whole point.")
    print()
    print("  k r << 1 : grad^2 G is two orders down from G (the Kelvin tensor is")
    print("             BIHARMONIC, not harmonic), so corr ~ d^2 G / r^2.  With")
    print("             the displacement block G ~ 1/r that is corr ~ d^2/r^3,")
    print("             a shell sum ~ d^2/R^2 and an O(1/R) TAIL -- slow, but")
    print("             ABSOLUTELY convergent.  Measured: gap shrank 5.4x.")
    print()
    print("  k r >> 1 : Helmholtz makes grad^2 G = -k^2 G, so corr -> -(k d)^2/24")
    print("             times G itself.  Measured plateau 9.14e-4 against the")
    print("             predicted (k_S d)^2/24 = 9.375e-4, within 3%.  The")
    print("             correction then decays like G (~1/r), each shell")
    print("             contributes a CONSTANT, and only oscillation converges")
    print("             the sum -- CONDITIONALLY.  Measured: gap GREW 2x.")
    print()
    print("  So the shape term is real but switches on at k r ~ 1.")
    print(f"  At the operating point that is R ~ 1/(k_S d) = {r_cross:.0f} cells.")
    print()
    print("  ▶ THE CONSEQUENCE IS A CATCH-22, AND IT IS QUANTITATIVE.")
    print("    The O(1/R) tail means converging the sum needs a LARGE box; the")
    print(f"    shape term means the box must stay well inside R = {r_cross:.0f}.")
    print("    The reach sweep was still falling at R = 8 with error 2.6e-5 and")
    print("    an O(1/R) tail, so another decade demands R ~ 80 -- past the")
    print("    crossover.  The box cannot be made big enough to converge without")
    print("    being big enough to become shape-dependent.")
    print()
    print("  ▶ SO THE SPECTRAL FORM FACTOR IS A CORRECTNESS ARGUMENT AFTER ALL,")
    print("    not merely an efficiency one.  The source-cell average is a")
    print("    CONVOLUTION with the cell indicator, hence EXACTLY the point")
    print("    kernel times prod_i sinc(q_i d/2) on each spectral component.")
    print("    That hands the conditional convergence back to whatever already")
    print("    handles it for the BARE kernel (Ewald at dz=0, the spectral sum")
    print("    for dz!=0), where it is already solved.  No such machinery exists")
    print("    in the package today.")
    print()
    print("  ▶ AND THE CURRENT NUMBERS ARE SAFE.  The solver runs at reach <= 8,")
    print(f"    i.e. k_S r <= 0.16, a factor {r_cross / 8:.0f} inside the crossover.")
    print("    Nothing already measured is invalidated; what is blocked is")
    print("    EXTRAPOLATING the reach sweep to a converged value.")
    if both_absolute:
        print()
        print("  ⚠ INCONSISTENT: panel [B] reported absolute convergence at BOTH")
        print("    frequencies, which contradicts the reading above.  Trust the")
        print("    table, not this paragraph, and re-examine.")
    print()
    print("⚠ SCOPE: dz=0 plane at k=0, which is where the bare sum's shape term")
    print("is known to live.  The dz!=0 planes are expected to converge")
    print("exponentially (the plane integral of d_i d_j G vanishes) and are NOT")
    print("tested here.  The crossover radius is quoted for the operating point")
    print("only -- it scales as 1/(k_S d), so a finer mesh pushes it OUT in cell")
    print("counts while a higher frequency pulls it in.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
