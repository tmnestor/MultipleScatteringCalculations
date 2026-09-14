"""MEASUREMENT: what discretisation floor does the PERIODIC route reach?

`gate_thesis_formulation` cannot conclude because its discretisation residual
(1.6-2.6%) is the same size as the ordering effect it is trying to resolve. The
cause is documented: a finite patch of scatterers is not a layer. The periodic
route removes that -- `compute_slab_scattering(periodic=True)` with
`slab_rpp_periodic` uses the Weyl lattice sum, replacing exp(ikr)/4pi r with
i/(2 k_z d^2) exp(i k_z |z|), which is an INFINITE layer exactly.

THIS SCRIPT ESTABLISHES THE FLOOR AND NOTHING ELSE. Before dressing that
machinery with a layered background, the question is whether it reaches ~1% AT
THE PARAMETERS THIS TEST NEEDS -- low ka, the contrast of interest, the lattice
sizes that are affordable. The slab convergence study reached 1.1%-0.31% at ITS
parameters; that is not the same claim.

If the floor here is ~1%, the ordering effect (1-3%) becomes resolvable and the
thesis question is back in reach. If it is not, the periodic route does not help
either and the remaining option is volume-averaged coupling for touching cubes.

ka IS PRINTED AND CHECKED. The analytic cube T-matrix is validated only for
ka < 0.3, and a previous run of this programme spent three attempts at ka = 0.94
before noticing.

Run:  conda run -n seismic python scripts/measure_periodic_floor.py
SI units (m, m/s, kg/m3, Pa) -- the slab machinery's own convention.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubic_scattering.effective_contrasts import MaterialContrast, ReferenceMedium  # noqa: E402
from cubic_scattering.slab_scattering import (  # noqa: E402
    SlabGeometry,
    SlabMaterial,
    compute_slab_scattering,
    compute_slab_tmatrices,
    kennett_reference_rpp,
    slab_rpp_periodic,
)

REF = ReferenceMedium(5000.0, 3000.0, 2500.0)
D_LAM, D_MU, D_RHO = 2.0e9, 1.0e9, 100.0  # the validated moderate contrast


def run(
    a_half: float, n_z: int, m: int, omega: float, *, volume_averaged: bool = False, n_orders: int = 2
) -> tuple:
    """(periodic R_PP, Kennett R_PP, relative error, ka).

    `volume_averaged` swaps the POINT propagator for the volume-averaged
    inter-voxel one at nearest-neighbour separations. Space-filling cubes touch,
    and the point propagator is known to differ from the volume-averaged object
    there, so this is the candidate cause of the per-plane floor -- which the
    measurement shows is flat in ka, i.e. not a Rayleigh-regime error and not
    fixed by refining the cube.
    """
    geom = SlabGeometry(M=m, N_z=n_z, a=a_half)
    ones = np.ones((n_z, m, m))
    material = SlabMaterial(Dlambda=D_LAM * ones, Dmu=D_MU * ones, Drho=D_RHO * ones, ref=REF)
    t0 = compute_slab_tmatrices(geom, material, omega)
    res = compute_slab_scattering(
        geom,
        material,
        omega,
        np.array([1.0, 0.0, 0.0]),
        "P",
        periodic=True,
        gmres_tol=1e-12,
        volume_averaged=volume_averaged,
        n_orders=n_orders,
    )
    r_per = slab_rpp_periodic(res, t0)

    # The exact answer: a uniform layer of thickness N_z * d with this contrast.
    contrast = MaterialContrast(D_LAM, D_MU, D_RHO)
    r_ken = kennett_reference_rpp(REF, contrast, n_z * geom.d, omega)

    ka = omega / REF.alpha * a_half
    return r_per, r_ken, abs(r_per - r_ken) / abs(r_ken), ka


def main() -> int:
    print("=" * 78)
    print("MEASUREMENT -- the discretisation floor of the PERIODIC route")
    print("  Weyl lattice sum = an INFINITE layer; compared against exact Kennett")
    print("=" * 78)
    print(
        f"\n  {'a (m)':>7} {'N_z':>4} {'M':>3} {'f (Hz)':>7} {'ka':>7} "
        f"{'|R| per':>11} {'|R| ken':>11} {'rel err':>10}"
    )

    best = 1.0
    for omega in (150.0, 60.0):
        for a_half in (2.0, 1.0, 0.5):
            for n_z, m in ((1, 4), (2, 4)):
                try:
                    r_p, r_k, err, ka = run(a_half, n_z, m, omega)
                except (RuntimeError, ValueError) as exc:
                    print(
                        f"  {a_half:7.2f} {n_z:4d} {m:3d} {omega / (2 * np.pi):7.1f} "
                        f"{'':>7}   {str(exc).splitlines()[0][:40]}"
                    )
                    continue
                flag = "" if ka < 0.3 else "  <-- ka OUT OF RANGE"
                best = min(best, err) if ka < 0.3 else best
                print(
                    f"  {a_half:7.2f} {n_z:4d} {m:3d} {omega / (2 * np.pi):7.1f} {ka:7.3f} "
                    f"{abs(r_p):11.4e} {abs(r_k):11.4e} {err:10.3e}{flag}"
                )

    print(f"\n  best relative error inside the validated ka range: {best:.3e}")

    # DOES THE VOLUME-AVERAGED COUPLING LOWER THE FLOOR?
    #
    # The floor above is flat in ka and roughly doubles with N_z -- a fixed
    # error PER PLANE that refining the cube does not touch. That is the
    # signature of the touching-face difference the repository documents, not of
    # the Rayleigh approximation. `volume_averaged=True` swaps the point
    # propagator for the volume-averaged one at nearest-neighbour separations,
    # which is the direct test of that diagnosis.
    #
    # The project notes carry two statements in tension about
    # inter_voxel_propagator -- an early verdict listing defects, and a later
    # finding that overturned the face-S part as an arbiter artifact. This
    # measures the outcome rather than adjudicating the record.
    print("\n  VOLUME-AVERAGED NEAREST NEIGHBOURS -- does the floor drop?")
    print(f"    {'a (m)':>7} {'N_z':>4} {'ka':>7} {'point':>11} {'vol-avg n=0':>12} {'vol-avg n=2':>12}")
    for a_half in (2.0, 1.0):
        for n_z in (1, 2):
            omega = 150.0
            _, _, e_pt, ka = run(a_half, n_z, 4, omega)
            out = [e_pt]
            for n_ord in (0, 2):
                try:
                    _, _, e_va, _ = run(a_half, n_z, 4, omega, volume_averaged=True, n_orders=n_ord)
                except (RuntimeError, ValueError) as exc:
                    out.append(float("nan"))
                    print(f"      n_orders={n_ord}: {str(exc).splitlines()[0][:50]}")
                    continue
                out.append(e_va)
            print(f"    {a_half:7.2f} {n_z:4d} {ka:7.3f} {out[0]:11.3e} {out[1]:12.3e} {out[2]:12.3e}")
    print("\n    A drop confirms the touching-face diagnosis and makes the thesis")
    print("    test viable. No drop means the per-plane floor is the single-site")
    print("    T-matrix's own limit, and neither lattice route reaches the 10x")
    print("    separation this question needs.")
    print("\n  Reading it. This is the FLOOR the thesis-formulation test would")
    print("  inherit. The ordering effect to be resolved is 1-3%, so a floor near")
    print("  1e-2 is not enough -- it needs to be nearer 1e-3 for the comparison")
    print("  to mean anything. Reported, not gated: this is a capability")
    print("  measurement, and its job is to say whether the route is worth taking.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
