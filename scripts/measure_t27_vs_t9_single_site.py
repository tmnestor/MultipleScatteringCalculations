"""MEASUREMENT: how big is the single-site T27 - T9 difference at OUR parameters?

WHY THIS COMES BEFORE ANY BUILD. The project's recorded verdict
(`t27-lattice-verdict`) says: do NOT build the 27-component LATTICE solver,
because inter-voxel quadratic-mode coupling is <=0.03% (ka<=0.3) of a
touching-pair response. Our ka runs 0.024 down to 0.0015, squarely inside that
range, so that verdict stands and no 27x27 propagator is contemplated here.

But the same verdict adds: "Single-site T27 effects are 10-100x larger and
already implemented." Ten to a hundred times 0.03% is 0.3%-3%, and the
unexplained refinement bias is 2.3%. That is a coincidence worth testing, and it
needs no new propagator -- only the single-site response changes, reduced back
to the 9 components the lattice actually couples.

WHAT IS COMPARED. `_sub_cell_tmatrix_9x9` needs exactly four scalars:
Drho_star, Dlambda_star, Dmu_star_diag, Dmu_star_off. The Galerkin (T27) result
supplies the last three. It does NOT expose Drho_star -- the density channel
lives in its T1u block -- so this measures the STRAIN SECTOR, which is precisely
where the 18 quadratic trial functions act. The density channel is held at its
T9 value and flagged, not quietly substituted.

WHAT WOULD MAKE T27 THE ANSWER: a difference of order 1% that does NOT shrink
with the cell size. The bias we are chasing is scale-invariant, so a T27-T9 gap
that vanishes as a -> 0 cannot explain it, however large it is at coarse cells.
THAT SCALING IS THE REAL TEST -- not the magnitude at any single a.

Run:  conda run -n seismic python scripts/measure_t27_vs_t9_single_site.py
SI units (m, m/s, kg/m3, Pa).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubic_scattering.effective_contrasts import (  # noqa: E402
    MaterialContrast,
    ReferenceMedium,
    compute_cube_tmatrix,
    compute_cube_tmatrix_galerkin,
)

REF = ReferenceMedium(5000.0, 3000.0, 2500.0)
CONTRAST = MaterialContrast(2.0e9, 1.0e9, 100.0)
OMEGA = 60.0
# The same ladder the refinement study walks: a = H/(2 n_z) with H = 4 m.
HALF_WIDTHS = (2.0, 1.0, 0.5, 0.25, 0.125, 0.0625)


def main() -> int:
    print("=" * 92)
    print("MEASUREMENT -- single-site T27 vs T9, strain sector, at the refinement ladder's cells")
    print("  the bias being chased is SCALE-INVARIANT, so watch the trend, not the magnitude")
    print("=" * 92)
    print(
        f"\n  {'a (m)':>8} {'ka':>8} {'dLambda* rel':>13} {'dMu_diag rel':>13} "
        f"{'dMu_off rel':>13} {'max rel':>10}"
    )

    trend = []
    for a in HALF_WIDTHS:
        ka = OMEGA / REF.beta * a
        t9 = compute_cube_tmatrix(OMEGA, a, REF, CONTRAST)
        t27 = compute_cube_tmatrix_galerkin(OMEGA, a, REF, CONTRAST)

        def _rel(x9: complex, x27: complex) -> float:
            return abs(complex(x27) - complex(x9)) / max(abs(complex(x9)), 1e-300)

        r_lam = _rel(t9.Dlambda_star, t27.Dlambda_star)
        r_dia = _rel(t9.Dmu_star_diag, t27.Dmu_star_diag)
        r_off = _rel(t9.Dmu_star_off, t27.Dmu_star_off)
        worst = max(r_lam, r_dia, r_off)
        trend.append(worst)
        print(f"  {a:8.4f} {ka:8.4f} {r_lam:13.4e} {r_dia:13.4e} {r_off:13.4e} {worst:10.3e}")

    spread = max(trend) / max(min(trend), 1e-300)
    print(f"\n  max-rel across the ladder varies by {spread:.2f}x ({min(trend):.2e} .. {max(trend):.2e})")
    print("=" * 92)
    if max(trend) < 1e-4:
        print("  TOO SMALL. The single-site T27 correction is under 0.01% at every")
        print("  cell size, two orders below the 2.3% bias. Swapping T9 for T27")
        print("  cannot explain the refinement error, and the '10-100x larger than")
        print("  0.03%' line in the verdict does not hold at THESE parameters.")
    elif spread < 2.0:
        print("  SCALE-INVARIANT and of the right order: the correction does not")
        print("  shrink with the cell, which is the signature the bias has. This is")
        print("  a live candidate -- wire it into the lattice and re-run Kennett.")
    else:
        print("  RIGHT ORDER BUT NOT SCALE-INVARIANT: the correction shrinks with the")
        print("  cell, so it cannot produce a bias that survives refinement. It may")
        print("  still matter at coarse cells, but it is not the thing being chased.")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
