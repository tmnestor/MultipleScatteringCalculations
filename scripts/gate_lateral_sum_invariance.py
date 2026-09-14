"""GATE: the periodic lateral sum must be supercell-invariant.

THE REQUIREMENT IS AN IDENTITY, NOT A TOLERANCE. At n_z = 1 a laterally periodic
plane of identical cubes IS a uniform layer. M = 2, 4, 6, 8 describe the SAME
physical layer -- they differ only in how many identical cubes are called one
repeating unit. So the computed reflection coefficient must not depend on M at
all. Any dependence is a defect in the lattice sum.

WHAT WAS WRONG. `build_slab_kernels(periodic=True)` assembled the spatial kernel
only over dx, dy in [-(M-1), M-1] -- one (2M-1)^2 patch -- and WRAPPED it into
M x M. That is not a lattice sum. The omitted tail of the 1/r^3 strain block
beyond radius ~M d falls as 1/(M d), and the measured error obeyed
1.1185e-3 + 1.1508e-2 / M over M = 2..12 with 1-5% residuals. At M = 4 the
truncation accounted for 74% of what had been reported as the discretisation
floor.

TWO FIXES ARE UNDER TEST, and the contrast between them is the point.

  * `lattice_images=n` adds the images the patch leaves out, to |nx|, |ny| <= n
    supercells. Every added term is at least M cells away, well outside the
    contact shell, so the point propagator is the right object for all of them.
    This is the PARTIAL fix. The 1/r^3 strain block falls off fast enough, but
    the 1/r displacement block is only conditionally convergent in two
    dimensions, so the ladder converges as 1/n_img and stalls.
  * `lattice_ewald=True` replaces the real-space assembly outright, writing the
    kernel at the M^2 Bloch points as an exact Ewald lattice sum. There is no
    truncation left to shrink.

The image ladder is kept rather than deleted because it is the MEASUREMENT that
made Ewald necessary. Without it, "we need an Ewald sum" would be an assumption;
with it, the 1/n_img stall is on the record next to the exact answer.

Run:  conda run -n seismic python scripts/gate_lateral_sum_invariance.py
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
)
from cubic_scattering.slab_scattering import (  # noqa: E402
    SlabGeometry,
    SlabMaterial,
    build_slab_kernels,
    compute_slab_scattering,
    compute_slab_tmatrices,
    kennett_reference_rpp,
    slab_rpp_periodic,
)

REF = ReferenceMedium(5000.0, 3000.0, 2500.0)
CONTRAST = MaterialContrast(2.0e9, 1.0e9, 100.0)
OMEGA, A_HALF = 60.0, 0.25
MS = (2, 3, 4, 6, 8)
IMAGE_LADDER = (0, 1, 2, 4, 8, 16)


def _err(m: int, n_img: int, ewald: bool = False) -> float:
    geom = SlabGeometry(M=m, N_z=1, a=A_HALF)
    ones = np.ones((1, m, m))
    mat = SlabMaterial(
        Dlambda=CONTRAST.Dlambda * ones,
        Dmu=CONTRAST.Dmu * ones,
        Drho=CONTRAST.Drho * ones,
        ref=REF,
    )
    t0 = compute_slab_tmatrices(geom, mat, OMEGA)
    kh = build_slab_kernels(geom, OMEGA, REF, periodic=True, lattice_images=n_img, lattice_ewald=ewald)
    res = compute_slab_scattering(
        geom,
        mat,
        OMEGA,
        np.array([1.0, 0.0, 0.0]),
        "P",
        periodic=True,
        gmres_tol=1e-12,
        kernel_hat=kh,
    )
    r_lat = slab_rpp_periodic(res, t0)
    r_ken = kennett_reference_rpp(REF, CONTRAST, geom.d, OMEGA)
    return abs(r_lat - r_ken) / abs(r_ken)


def main() -> int:
    print("=" * 88)
    print("GATE -- supercell invariance of the periodic lateral sum")
    print(f"  one plane, a = {A_HALF} m: every M below is the SAME physical layer")
    print("=" * 88)

    header = "".join(f"{'M=' + str(m):>11}" for m in MS)
    print(f"\n  {'images':>7}{header}{'spread':>11}{'floor':>11}")
    spreads = []
    baseline: list[float] = []
    for n_img in IMAGE_LADDER:
        vals = [_err(m, n_img) for m in MS]
        if n_img == 0:
            baseline = vals
        spread = (max(vals) - min(vals)) / max(np.mean(vals), 1e-300)
        spreads.append(spread)
        row = "".join(f"{v:11.4e}" for v in vals)
        print(f"  {n_img:7d}{row}{spread:11.2e}{np.mean(vals):11.4e}")

    print(
        f"\n  spread fell {spreads[0] / max(spreads[-1], 1e-300):.1f}x "
        f"({spreads[0]:.2e} -> {spreads[-1]:.2e})"
    )

    # The EXACT sum. Images are a partial fix -- they converge only as 1/n_img,
    # which is what the ladder above measures. Ewald removes the truncation
    # rather than shrinking it, so it is not another rung on that ladder.
    ew = [_err(m, 0, ewald=True) for m in MS]
    ew_spread = (max(ew) - min(ew)) / max(np.mean(ew), 1e-300)
    row = "".join(f"{v:11.4e}" for v in ew)
    print(f"\n  {'EWALD':>7}{row}{ew_spread:11.2e}{np.mean(ew):11.4e}")
    spreads.append(ew_spread)

    img_spread = spreads[-2]
    print("\n" + "=" * 88)
    if ew_spread < 1e-9:
        print("  PASS: the Ewald kernel is supercell-invariant to machine precision.")
        print("  Every M describes the same physical layer and now returns the same")
        print("  number, which is an IDENTITY rather than a tolerance being met.")
        print()
        print("  Read the two rows together, because they say different things. The")
        print(f"  image ladder stalls at a spread of {img_spread:.2e} after 16 shells -- it")
        print("  converges only as 1/n_img, so it shrinks the truncation without")
        print("  removing it. Ewald removes it outright, in one shot, at cutoff 4.")
        print()
        old_at_m4 = baseline[MS.index(4)]
        print(f"  THE TRUE FLOOR IS {np.mean(ew):.4e}. At M = 4, the value used")
        print(f"  throughout the convergence campaign, the old kernel gave {old_at_m4:.4e},")
        pct = 100 * (1 - np.mean(ew) / old_at_m4)
        print(f"  so {pct:.0f}% of what was reported as the discretisation floor")
        print("  was this artifact. Every quantitative result from that campaign needs")
        print("  RE-RUNNING, not merely reinterpreting.")
        print()
        print("  Note the 1/M fit's extrapolation to M -> infinity was 1.1185e-3; the")
        print(f"  exact answer is {np.mean(ew):.4e}. The fit described the truncation well")
        print("  but its intercept was not the floor -- another reason to re-run rather")
        print("  than to correct old numbers arithmetically.")
    elif img_spread < 0.25 * spreads[0]:
        print("  PARTIAL: images help but the Ewald sum is not invariant either --")
        print("  which means the defect is NOT only the truncation. Check the Bloch")
        print("  phase sign and the R = 0 convention (included at dz != 0, excluded")
        print("  at dz = 0) before looking anywhere else.")
    else:
        print("  NO IMPROVEMENT: the supercell dependence is not the image tail.")
        print("  Re-examine the diagnosis before writing any more summation code.")
    print("=" * 88)
    return 0 if ew_spread < 1e-9 else 1


if __name__ == "__main__":
    raise SystemExit(main())
