"""MEASUREMENT: does the omitted coupling COMPOUND as scattering strengthens?

`measure_ordering_difference` found the dress-after omission to be first-order
in the weak regime: the Born estimate tracked the measured difference within
~10% and the difference was linear in T. That is the regime where dress-after
can in principle be corrected perturbatively. This asks where it stops being
so.

THE AXIS IS THE SPECTRAL RADIUS, not the contrast. rho(G0 T) is what says how
strong the multiple scattering actually is -- it is the quantity the Neumann
series converges on and the one GMRES feels. Reporting against "T scale" would
make the answer depend on an arbitrary normalisation.

CONTRASTS STAY PHYSICAL. The single-site renormalisation has a validity floor
around |Delta| < 52% of the background, and the background lambda here is
rho(alpha^2 - 2 beta^2) = 2.5(25 - 18) = 17.5 GPa. T scale 3.0 puts Dlambda at
6 GPa, i.e. 34%, inside that. Scattering strength is raised instead by ENLARGING
THE LATTICE, which is physical: more scatterers couple more, at fixed contrast.

WHAT IT FOUND, and what it does not support.

ROBUST: the ABSOLUTE difference grows from ~10% of the field at rho < 1 to
70-90% at rho > 2.5, in every geometry tried. In strong multiple scattering,
dressing afterwards is not an approximation to the stratified solve -- it is a
different answer. That is the measured case for the thesis's ordering.

NOT ROBUST, and recorded because it was briefly believed: the diff/Born RATIO.
Moving the deep reflector by ONE layer took the rho = 2 row from 1.99 to 0.63.
Both are correct computations of different geometries; the ratio is simply too
configuration-sensitive to carry a claim about compounding or saturation. Near
rho < 1 it is consistently 0.88-1.02 across geometries, which does support the
weaker statement that the omission is first-order THERE -- and that is all it is
used for.

REPORTED, not gated.

Run:  conda run -n seismic python scripts/measure_ordering_strong_scattering.py
Seismic units (km, km/s, g/cm3), time convention e^{-i omega t}.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "PhD_fortran_code"))
sys.path.insert(0, "/Users/tod/Desktop/SeismicInversion")

from cubic_scattering.directional_sweeps import (  # noqa: E402
    SweepGrid3D,
    apply_g0_3d,
    build_g0_cache_3d,
)
from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from cubic_scattering.pair_propagators import TransverseRule, layered_incident_field  # noqa: E402
from cubic_scattering.slab_scattering import (  # noqa: E402
    SlabGeometry,
    SlabMaterial,
    compute_slab_tmatrices,
)
from cubic_scattering.sweep_solver import solve_foldy_lax_3d  # noqa: E402

PITCH = 0.25
OM = 2 * np.pi * 6.0
SOURCE_IFACE = 14
RULE = TransverseRule(kr_max=10.0 / PITCH, n_axis=48)
SRC_VEC = np.array([1.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0], dtype=complex)
BG_LAMBDA = 2.5 * (25.0 - 2 * 9.0)  # 17.5 GPa, the validity yardstick


def model(n_lay: int = 24, q: float = 1e4):
    """Ocean over elastic layers with a strong contrast below the voxels."""
    import Kennett_Reflectivity.layer_model as lm

    a, b, r = 5.0, 3.0, 2.5
    al, be, rh = [1.5, *([a] * n_lay), a], [0.0, *([b] * n_lay), b], [1.03, *([r] * n_lay), r]
    # Contrast starts at layer 22, not 21: the deepest scattering plane used
    # here is interface 20, and a plane sitting ON a material discontinuity is
    # rejected by assert_interface_continuous -- K is built from the local
    # eta_S, which is two-valued there. Starting at 21 makes the 3-plane case
    # fail with exactly that diagnostic.
    for j in range(22, n_lay + 2):
        al[j], be[j], rh[j] = a + 2.0, b + 1.2, r + 0.6
    return lm.LayerModel.from_arrays(
        alpha=al,
        beta=be,
        rho=rh,
        thickness=[3.0, *([PITCH] * n_lay), np.inf],
        Q_alpha=[q] * (n_lay + 2),
        Q_beta=[1e12, *([q] * n_lay), q],
    )


def spectral_radius(cache, t0, shape) -> float:
    """rho(G0 T) by assembling the operator column by column."""
    size = int(np.prod(shape))
    mat = np.zeros((size, size), dtype=complex)
    for c in range(size):
        e = np.zeros(size, dtype=complex)
        e[c] = 1.0
        v = e.reshape(shape)
        mat[:, c] = apply_g0_3d(np.einsum("zxyab,zxyb->zxya", t0, v), cache).ravel()
    return float(np.max(np.abs(np.linalg.eigvals(mat))))


def main() -> int:
    mod = model()
    s_p, s_s = mod.complex_slowness_p(), mod.complex_slowness_s()

    print("=" * 78)
    print("MEASUREMENT -- does the dress-after omission compound with rho(G0 T)?")
    print(f"  background lambda = {BG_LAMBDA} GPa; contrasts kept inside the 52% floor")
    print("  scattering strength raised by ENLARGING THE LATTICE, at fixed contrast")
    print("=" * 78)
    print(
        f"\n  {'lattice':>10} {'T':>5} {'Dlam/lam':>9} {'rho(G0T)':>10} "
        f"{'diff':>11} {'Born':>11} {'diff/Born':>10}"
    )

    for n_z, n_x in ((2, 4), (2, 6), (3, 6)):
        planes = tuple(18 + i for i in range(n_z))
        j = planes[0]
        ref = ReferenceMedium(1.0 / s_p[j], 1.0 / s_s[j], mod.rho[j])
        dz_planes = tuple(float(planes[i] - SOURCE_IFACE) * PITCH for i in range(n_z))
        shape = (n_z, n_x, n_x, 9)

        psi_inc = layered_incident_field(
            n_z,
            n_x,
            n_x,
            PITCH,
            OM,
            ref,
            source_xy=(0, 0),
            source_vec=SRC_VEC,
            dz_planes=dz_planes,
            model=mod,
            plane_ifaces=planes,
            source_iface=SOURCE_IFACE,
            transverse=RULE,
        )
        grid = SweepGrid3D(n_z=n_z, n_x=n_x, n_y=n_x, pitch=PITCH)
        c_thesis = build_g0_cache_3d(grid, ref, OM, model=mod, plane_ifaces=planes, transverse=RULE)
        c_dress = build_g0_cache_3d(grid, ref, OM)

        geom = SlabGeometry(M=n_x, N_z=n_z, a=0.5 * PITCH)
        for t_scale in (1.0, 3.0):
            rng = np.random.default_rng(20260914)
            s = t_scale * (0.5 + rng.random((n_z, n_x, n_x)))
            material = SlabMaterial(Dlambda=2.0 * s, Dmu=1.0 * s, Drho=0.1 * s, ref=ref)
            t0 = compute_slab_tmatrices(geom, material, OM)

            rho = spectral_radius(c_thesis, t0, shape)
            try:
                a = solve_foldy_lax_3d(c_thesis, t0, psi_inc, tol=1e-12).psi
                b = solve_foldy_lax_3d(c_dress, t0, psi_inc, tol=1e-12).psi
            except RuntimeError as exc:
                # Reported, never hidden: solve_foldy_lax_3d raises rather than
                # returning a partially converged field, and the rho at which it
                # happens is itself the useful number.
                print(
                    f"  {f'{n_z}x{n_x}x{n_x}':>10} {t_scale:5.1f} {'':>9} "
                    f"{rho:10.4f}   GMRES did not converge: {str(exc).splitlines()[0]}"
                )
                continue

            diff = float(np.abs(a - b).max() / np.abs(a).max())
            tb = np.einsum("zxyab,zxyb->zxya", t0, b)
            born = float(
                np.abs(apply_g0_3d(tb, c_thesis) - apply_g0_3d(tb, c_dress)).max() / np.abs(a).max()
            )
            frac = float((2.0 * s).max() / BG_LAMBDA)
            print(
                f"  {f'{n_z}x{n_x}x{n_x}':>10} {t_scale:5.1f} {frac:9.2f} {rho:10.4f} "
                f"{diff:11.3e} {born:11.3e} {diff / max(born, 1e-300):10.3f}"
            )

    print("\n  Reading it. THE ABSOLUTE DIFFERENCE is the robust column: it grows")
    print("  from ~10% of the field at rho < 1 to 70-90% at rho > 2.5. In strong")
    print("  multiple scattering, dressing afterwards is not an approximation to")
    print("  the stratified solve, it is a different answer. That holds across")
    print("  every configuration tried.")
    print("\n  DO NOT READ THE diff/Born RATIO AS A TREND. It is not robust:")
    print("  moving the deep reflector by ONE layer (21 -> 22) took the")
    print("  rho = 2 row from 1.99 to 0.63. Both are correct computations of")
    print("  different geometries, but the ratio is too configuration-sensitive")
    print("  to support a claim about compounding or saturation. It is kept")
    print("  because near rho < 1 it is consistently ~0.9-1.0 across geometries,")
    print("  which does support the weaker claim that the omission is")
    print("  first-order THERE.")
    print("\n  The Dlam/lam column is the validity check, not decoration: past")
    print("  ~0.52 the single-site renormalisation is outside its measured range")
    print("  and any trend beyond that says more about the T-matrix than the")
    print("  ordering.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
