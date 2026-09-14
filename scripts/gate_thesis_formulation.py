"""GATE: is the thesis's two-potential formulation CORRECT?

Not "do the two orderings differ" -- that was measured, and they do, by up to
110% of the field. This asks which one is right, which needs an exact answer and
therefore a configuration in which one exists.

THE CONSTRUCTION. A laterally uniform, space-filling layer of cubes -- pitch
equal to the cube side -- IS a homogeneous layer. So one physical medium can be
written two ways:

  REFERENCE + CONTRAST : the background WITHOUT the contrast, the contrast
                         carried by a T-matrix at every voxel, solved by
                         Foldy-Lax. This is the thesis's two-potential split.
  ALL IN THE BACKGROUND: the contrast folded into the layered model, and no
                         scattering at all -- just the stratified propagator,
                         which is exact and already validated against Kennett
                         at 1e-15.

If the formulation is correct those must agree. That is its correctness
condition, and it is what this gate measures. Nothing here is a tolerance chosen
to be passed: the second route is the exact answer for this medium.

WHAT MAKES IT DECISIVE. The same comparison is run with the layer reverberation
PRESENT in G0 (the thesis ordering) and ABSENT (dress-after). The two differ by
70-110% at these contrasts, so the arbiter separates them by two orders of
magnitude. A tolerance argument is not needed.

THE TWO CONDITIONS, both of which must be respected or the equivalence fails:

  [1] SPACE-FILLING AND LATERALLY UNIFORM. Every voxel carries the same
      contrast, and the cubes tile the layer exactly. Layers are half a pitch
      thick here so that each voxel spans exactly two of them, contiguously and
      with no gap.
  [2] LATERALLY INFINITE. A finite n_x by n_y patch of scatterers is NOT a
      layer, and the truncation is a real error, not noise. It is handled by
      REFINEMENT: n_x is swept and the agreement must improve. A single lattice
      size would prove nothing, which is the same discipline rung 4 demanded.

A FAILURE HERE IS A REFUTATION, and would point at the stratified propagator or
the composition rather than at the ordering. Said plainly so the result is not
quietly reinterpreted if it comes out wrong.

Run:  conda run -n seismic python scripts/gate_thesis_formulation.py
Seismic units (km, km/s, g/cm3), time convention e^{-i omega t}.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "PhD_fortran_code"))
sys.path.insert(0, "/Users/tod/Desktop/SeismicInversion")

from cubic_scattering.directional_sweeps import SweepGrid3D, build_g0_cache_3d  # noqa: E402
from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from cubic_scattering.pair_propagators import TransverseRule, layered_incident_field  # noqa: E402
from cubic_scattering.slab_scattering import (  # noqa: E402
    SlabGeometry,
    SlabMaterial,
    compute_slab_tmatrices,
)
from cubic_scattering.sweep_solver import solve_foldy_lax_3d  # noqa: E402

PITCH = 0.25
HALF = 0.5 * PITCH  # layer thickness: a voxel spans exactly two layers
# ka MUST STAY IN THE RAYLEIGH REGIME. compute_slab_tmatrices uses the analytic
# cube T-matrix, which this project's regime table validates only for ka < 0.3;
# 0.3-1.0 needs the subdivided resonance treatment instead. A first run of this
# gate used 6 Hz, giving ka = 0.94 -- every T-matrix outside its validated range
# -- and the discretisation error then swamped the quantity being measured, at
# 0.19-0.42 and saturating with lattice size. 0.6 Hz puts ka at 0.094.
OM = 2 * np.pi * 0.6
N_LAY = 40
SOURCE_IFACE = 10
PLANES = (22, 24, 26)  # obs (T0 = 0), then two scattering planes, one pitch apart
SCAT_LAYERS = (24, 25, 26, 27)  # the layers those two voxels occupy
REFLECTOR_LAYER = 30  # background reflector, clear of every scattering plane
RULE = TransverseRule(kr_max=10.0 / PITCH, n_axis=48)
SRC_VEC = np.array([1.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0], dtype=complex)

A0, B0, R0 = 5.0, 3.0, 2.5
D_LAM, D_MU, D_RHO = 2.0, 1.0, 0.1  # GPa, GPa, g/cm3


def _model(with_contrast: bool, q: float = 1e4, *, uniform: bool = False):
    """The stack, with the voxel layers either background or background+Delta.

    Half-pitch layers so each voxel spans exactly two, contiguously: the voxel
    at interface j covers [d_j - p/2, d_j + p/2] = layers j and j+1, and the
    next voxel one pitch below starts exactly where it ends.
    """
    import Kennett_Reflectivity.layer_model as lm

    # `uniform` removes the ocean contrast too, so the top boundary is as close
    # to a whole space as this model allows and the two orderings coincide.
    w_a, w_r = (A0, R0) if uniform else (1.5, 1.03)
    al = [w_a, *([A0] * N_LAY), A0]
    be = [0.0, *([B0] * N_LAY), B0]
    rh = [w_r, *([R0] * N_LAY), R0]
    if not uniform:
        # A STRONG REFLECTOR CLOSE BELOW THE VOXELS, in the BACKGROUND of both
        # models. Without it DeltaG0 comes only from a distant seabed, the two
        # orderings agree to 2-3%, and the test has almost no signal to resolve
        # -- measured, and the reason this is here. Layer 30 is three half-pitches
        # below the deepest voxel layer (27) and well inside a wavelength at
        # 0.6 Hz, so the layer-mediated coupling is strong.
        for j in range(REFLECTOR_LAYER, N_LAY + 2):
            al[j], be[j], rh[j] = A0 + 2.0, B0 + 1.2, R0 + 0.6
    if with_contrast:
        # Delta lambda / mu / rho -> the perturbed layer's velocities.
        lam0, mu0 = R0 * (A0**2 - 2 * B0**2), R0 * B0**2
        r1 = R0 + D_RHO
        a1 = float(np.sqrt((lam0 + D_LAM + 2 * (mu0 + D_MU)) / r1))
        b1 = float(np.sqrt((mu0 + D_MU) / r1))
        for j in SCAT_LAYERS:
            al[j], be[j], rh[j] = a1, b1, r1
    return lm.LayerModel.from_arrays(
        alpha=al,
        beta=be,
        rho=rh,
        thickness=[3.0, *([HALF] * N_LAY), np.inf],
        Q_alpha=[q] * (N_LAY + 2),
        Q_beta=[1e12, *([q] * N_LAY), q],
    )


def _field_at_obs(n_x: int, *, layered_g0: bool, uniform: bool = False) -> tuple:
    """(total field at the observation plane by both routes) for one lattice size.

    With ``uniform`` the background carries no ocean contrast and no deep
    reflector, so the layered and whole-space G0 coincide and the FORMULATION
    question drops out -- what remains is the discretisation alone.
    """
    m_ref, m_full = _model(False, uniform=uniform), _model(True, uniform=uniform)
    s_p, s_s = m_ref.complex_slowness_p(), m_ref.complex_slowness_s()
    j = PLANES[0]
    ref = ReferenceMedium(1.0 / s_p[j], 1.0 / s_s[j], m_ref.rho[j])
    n_z = len(PLANES)
    dz_planes = tuple(float(PLANES[i] - SOURCE_IFACE) * HALF for i in range(n_z))

    kw = dict(
        source_xy=(n_x // 2, n_x // 2),
        source_vec=SRC_VEC,
        dz_planes=dz_planes,
        plane_ifaces=PLANES,
        source_iface=SOURCE_IFACE,
        transverse=RULE,
    )

    # ROUTE 2, the exact answer: the contrast lives in the background and there
    # is no scattering to do.
    exact = layered_incident_field(n_z, n_x, n_x, PITCH, OM, ref, model=m_full, **kw)

    # ROUTE 1: reference + contrast, solved.
    psi_inc = layered_incident_field(n_z, n_x, n_x, PITCH, OM, ref, model=m_ref, **kw)
    grid = SweepGrid3D(n_z=n_z, n_x=n_x, n_y=n_x, pitch=PITCH)
    cache = (
        build_g0_cache_3d(grid, ref, OM, model=m_ref, plane_ifaces=PLANES, transverse=RULE)
        if layered_g0
        else build_g0_cache_3d(grid, ref, OM)
    )

    geom = SlabGeometry(M=n_x, N_z=n_z, a=HALF)
    ones = np.ones((n_z, n_x, n_x))
    material = SlabMaterial(Dlambda=D_LAM * ones, Dmu=D_MU * ones, Drho=D_RHO * ones, ref=ref)
    t0 = compute_slab_tmatrices(geom, material, OM)
    t0[0] = 0.0  # the observation plane scatters nothing

    got = solve_foldy_lax_3d(cache, t0, psi_inc, tol=1e-11).psi
    # psi at a site with T = 0 IS the total field there.
    return got[0], exact[0]


def main() -> int:
    print("=" * 78)
    print("GATE -- is the thesis's two-potential formulation correct?")
    print("  reference + contrast (solved)  vs  contrast folded into the background")
    print("  the second is exact for this medium; agreement is the correctness test")
    print(f"  contrast Dlam={D_LAM} Dmu={D_MU} GPa, Drho={D_RHO} g/cm3, laterally uniform")
    ka = float(np.real(OM) / A0 * HALF)
    print(f"  ka = {ka:.3f}  (the cube T-matrix is validated for ka < 0.3)")
    if ka >= 0.3:
        print("  *** ka IS OUTSIDE THE VALIDATED RANGE -- the T-matrices are wrong")
        print("  *** before any of the physics below is reached. Fix ka first.")
    print("=" * 78)

    print(f"\n  {'n_x':>5} {'sites':>7} {'thesis ordering':>17} {'dress-after':>14}")
    thesis, dress = [], []
    for n_x in (6, 10, 14):
        a_th, exact = _field_at_obs(n_x, layered_g0=True)
        a_dr, _ = _field_at_obs(n_x, layered_g0=False)
        scale = np.abs(exact).max()
        e_th = float(np.abs(a_th - exact).max() / scale)
        e_dr = float(np.abs(a_dr - exact).max() / scale)
        thesis.append(e_th)
        dress.append(e_dr)
        print(f"  {n_x:5d} {n_x * n_x:7d} {e_th:17.4e} {e_dr:14.4e}")

    # THE CONTROL THAT DECIDES WHETHER ANY OF THE ABOVE MEANS ANYTHING.
    #
    # The comparison above conflates two questions: whether reference+contrast
    # equals all-in-background (the FORMULATION), and whether a space-filling
    # cube array with point-propagator coupling represents a layer at all (the
    # DISCRETISATION). Running it with a UNIFORM background removes the first --
    # with no layering the two orderings coincide by construction -- so whatever
    # is left is pure discretisation.
    #
    # If that residual is the same size as the numbers above, this gate cannot
    # speak to the formulation and says so rather than claiming a refutation.
    print("\n  DISCRETISATION CONTROL -- uniform background, formulation removed")
    print(f"    {'n_x':>5} {'residual':>12}")
    disc = []
    for n_x in (6, 10, 14):
        a_u, exact_u = _field_at_obs(n_x, layered_g0=True, uniform=True)
        d = float(np.abs(a_u - exact_u).max() / np.abs(exact_u).max())
        disc.append(d)
        print(f"    {n_x:5d} {d:12.4e}")

    improving = thesis[-1] < thesis[0]
    separated = dress[-1] > 10.0 * thesis[-1]
    conclusive = disc[-1] < 0.1 * thesis[-1]
    print(f"\n  discretisation residual is small enough to conclude: {'YES' if conclusive else 'NO'}")
    if not conclusive:
        print("    The discretisation error is comparable to the quantity being")
        print("    measured, so NOTHING above is evidence about the formulation.")
        print("    Two known causes, both documented in this repository: a finite")
        print("    patch is not a layer (the slab convergence study reached ~1%")
        print("    only with a periodic Weyl sum, an INFINITE layer), and")
        print("    space-filling cubes touch, where the point propagator is known")
        print("    to differ from the volume-averaged one.")

    print("\n" + "=" * 78)
    print(f"  thesis ordering improves with lattice size : {'YES' if improving else 'NO'}")
    print(f"  dress-after separated by >10x at the widest: {'YES' if separated else 'NO'}")
    print("\n  The lateral truncation is a real error, not noise: a finite patch of")
    print("  scatterers is not a layer. Refinement is the only honest reading, and")
    print("  a single lattice size would prove nothing.")
    print("\n  A thesis-ordering error that does NOT fall under refinement is a")
    print("  REFUTATION, and would point at the stratified propagator or the")
    print("  composition rather than at the ordering.")
    print("=" * 78)
    return 0 if improving else 1


if __name__ == "__main__":
    raise SystemExit(main())
