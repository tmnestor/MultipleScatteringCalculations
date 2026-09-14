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

THE FIX UNDER TEST. `lattice_images=n` adds the images the patch leaves out, out
to |nx|, |ny| <= n supercells. Every added term is at least M cells away, well
outside the contact shell, so the point propagator is the correct object for all
of them.

WHAT THIS GATE CAN ALSO SHOW, and it is the reason it sweeps rather than checks
one number: whether a DIRECT image sum converges at all. The 1/r^3 strain block
falls off fast enough, but the 1/r displacement block converges only
conditionally in two dimensions. If the residual M-dependence stalls instead of
falling as images are added, a direct sum is not enough and an Ewald split is
required -- and that will be a measured conclusion rather than an assumption.

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


def _err(m: int, n_img: int) -> float:
    geom = SlabGeometry(M=m, N_z=1, a=A_HALF)
    ones = np.ones((1, m, m))
    mat = SlabMaterial(
        Dlambda=CONTRAST.Dlambda * ones,
        Dmu=CONTRAST.Dmu * ones,
        Drho=CONTRAST.Drho * ones,
        ref=REF,
    )
    t0 = compute_slab_tmatrices(geom, mat, OMEGA)
    kh = build_slab_kernels(geom, OMEGA, REF, periodic=True, lattice_images=n_img)
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
    for n_img in IMAGE_LADDER:
        vals = [_err(m, n_img) for m in MS]
        spread = (max(vals) - min(vals)) / max(np.mean(vals), 1e-300)
        spreads.append(spread)
        row = "".join(f"{v:11.4e}" for v in vals)
        print(f"  {n_img:7d}{row}{spread:11.2e}{np.mean(vals):11.4e}")

    print(
        f"\n  spread fell {spreads[0] / max(spreads[-1], 1e-300):.1f}x "
        f"({spreads[0]:.2e} -> {spreads[-1]:.2e})"
    )

    print("\n" + "=" * 88)
    if spreads[-1] < 1e-3:
        print("  PASS: the answer is supercell-invariant once the images are")
        print("  summed. The lateral sum was truncated, not wrong in principle, and")
        print("  a DIRECT image sum suffices -- no Ewald split is needed. The floor")
        print("  that remains is the genuine discretisation error, free of the")
        print("  1/M artifact that contaminated every earlier measurement.")
    elif spreads[-1] < 0.25 * spreads[0]:
        print("  PARTIAL: adding images reduces the supercell dependence but does")
        print("  not remove it. That is the signature of the conditionally")
        print("  convergent 1/r block -- a direct sum cannot finish the job and an")
        print("  EWALD SPLIT is required. Now a measured conclusion, not a guess.")
    else:
        print("  NO IMPROVEMENT: the supercell dependence is not the image tail.")
        print("  Re-examine the diagnosis before writing any more summation code.")
    print("=" * 88)
    return 0 if spreads[-1] < 1e-3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
