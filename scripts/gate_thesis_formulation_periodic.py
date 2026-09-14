"""GATE: the thesis two-potential formulation, on a PERIODIC lattice.

`gate_thesis_formulation` could not conclude: its finite-patch discretisation
residual (1.6-2.6%) was the size of the ordering effect it was trying to resolve.
`measure_periodic_floor` then showed the periodic route with volume-averaged
nearest neighbours reaches 1.8e-3 -- 6-17x below that effect. This is the same
test on that footing.

THE CORRECTNESS CONDITION, unchanged. A laterally uniform, space-filling layer
of cubes IS a homogeneous layer, so one medium can be written two ways:
reference + contrast carried by T-matrices and solved (the thesis's split), or
the contrast folded into the background with no scattering at all (exact). They
must agree.

EVERYTHING COLLAPSES TO k_par = 0, and that is what makes this tractable. A
laterally uniform medium at normal incidence has only the specular component, so

  * the ILLUMINATION is the layered propagator at k_par -> 0, uniform across the
    lattice -- no real-space transform, and no need for the internal mode
    amplitudes of the stack that no function returns;
  * the LAYER REVERBERATION enters the existing periodic kernel as a single
    addition, DeltaG0(k_par -> 0; dz) / d^2 into kernel_hat[dz][0, 0], which is
    the k = 0 Fourier component and therefore exactly the Weyl lattice sum of
    the reverberation;
  * the OBSERVABLE is the field at a plane carrying T0 = 0, where psi from the
    Foldy-Lax solve IS the total field -- no far-field projection.

k_par IS SMALL BUT NOT ZERO. `k_operator` is undefined at kx = ky = 0, where the
horizontal direction degenerates, so eps = 1e-6 stands in for normal incidence,
the same device the intra-plane R/T work uses.

THREE ARMS, the third being what makes the first two mean anything:
  [T1] thesis ordering  -- kernel dressed with DeltaG0
  [T2] dress-after      -- kernel left whole-space
  [T3] DISCRETISATION CONTROL -- uniform background, where the two coincide by
       construction, so the residual is the discretisation alone.

A [T1] error at the [T3] floor, with [T2] well above it, confirms the
formulation. [T1] above the floor is a REFUTATION and would point at the
stratified propagator or the composition. [T1] and [T2] both at the floor means
this configuration still cannot separate them.

Run:  conda run -n seismic python scripts/gate_thesis_formulation_periodic.py
SI units (m, m/s, kg/m3, Pa) -- the slab machinery's own convention.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "PhD_fortran_code"))
sys.path.insert(0, "/Users/tod/Desktop/SeismicInversion")

import cubic_scattering.layered_correction as LC  # noqa: E402
from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from cubic_scattering.slab_scattering import (  # noqa: E402
    SlabGeometry,
    SlabMaterial,
    build_slab_kernels,
    compute_slab_scattering,
    compute_slab_tmatrices,
)
from cubic_scattering.sweep_kernels import (  # noqa: E402
    same_depth_kernel_9x9,
    vertical_kernel_9x9,
)

A0, B0, R0 = 5000.0, 3000.0, 2500.0
D_LAM, D_MU, D_RHO = 2.0e9, 1.0e9, 100.0
OM = 60.0
A_HALF = 1.0  # ka = 0.012, deep in the validated Rayleigh range
D = 2.0 * A_HALF
EPS = 1e-6  # stands in for k_par = 0; k_operator is undefined exactly there
M, N_Z = 4, 1  # N_z = 1: the floor is 1.8e-3 there, 4.2e-3 at N_z = 2
N_LAY = 40
# PLANE SPACING MUST MATCH THE LATTICE. SlabGeometry places its planes
# uniformly d apart, so the observation plane must be exactly one pitch -- two
# half-layers -- above the scattering plane. An earlier version used 14 and 20,
# i.e. 3d apart, and _dressed_kernel compounded it by computing depths as
# SCAT_IFACE + 2*lz: the dressing addressed the wrong plane depths entirely.
SRC_IFACE, OBS_IFACE, SCAT_IFACE = 8, 18, 20
SCAT_LAYERS = (20, 21)  # the single voxel plane spans two half-pitch layers
# The reflector sits immediately below the voxel layers (20, 21). At layer 26
# the separation [T2]/[T3] was 2.95x, short of the 5x this gate demands, with
# [T1] already AT the floor. The legitimate response is to raise the signal, not
# to lower the bar: [T2] scales with DeltaG0 and [T3] is measured on a uniform
# background, so moving the reflector closer separates them without touching the
# floor. Interface 21 is then a discontinuity, but no plane sits there -- the
# planes are at 14 and 20.
REFL_LAYER = 22
PLANE_IFACES = (OBS_IFACE, SCAT_IFACE)


def _model(with_contrast: bool, *, uniform: bool):
    """Half-pitch layers; the voxel plane at SCAT_IFACE spans two of them."""
    import Kennett_Reflectivity.layer_model as lm

    half = 0.5 * D
    al = [1500.0, *([A0] * N_LAY), A0]
    be = [0.0, *([B0] * N_LAY), B0]
    rh = [1030.0, *([R0] * N_LAY), R0]
    if uniform:
        al[0], rh[0] = A0, R0
    else:
        for j in range(REFL_LAYER, N_LAY + 2):
            al[j], be[j], rh[j] = A0 + 2000.0, B0 + 1200.0, R0 + 600.0
    if with_contrast:
        lam0, mu0 = R0 * (A0**2 - 2 * B0**2), R0 * B0**2
        r1 = R0 + D_RHO
        for j in SCAT_LAYERS:
            al[j] = float(np.sqrt((lam0 + D_LAM + 2 * (mu0 + D_MU)) / r1))
            be[j] = float(np.sqrt((mu0 + D_MU) / r1))
            rh[j] = r1
    return lm.LayerModel.from_arrays(
        alpha=al,
        beta=be,
        rho=rh,
        thickness=[3000.0, *([half] * N_LAY), np.inf],
        Q_alpha=[1e4] * (N_LAY + 2),
        Q_beta=[1e12, *([1e4] * N_LAY), 1e4],
    )


def _p_tilde(model, src: int, rcv: int) -> np.ndarray:
    """The layered propagator at k_par -> 0, one 9x9."""
    return LC.corrected_layered_9x9(model, OM, np.array([EPS]), np.array([0.0]), src, rcv)[0]


def _dressed_kernel(model, ref: ReferenceMedium, ref_c: ReferenceMedium, geom: SlabGeometry) -> np.ndarray:
    """The periodic volume-averaged kernel, plus the layer reverberation at k = 0.

    kernel_hat[dz][0, 0] is the sum over every spatial entry, i.e. the k = 0
    Fourier component -- exactly the Weyl lattice sum. Adding
    DeltaG0(k_par -> 0)/d^2 there dresses the specular channel, which for a
    laterally uniform medium is the only one that acts.
    """
    kh = build_slab_kernels(geom, OM, ref, volume_averaged=True, periodic=True).copy()
    n_z = geom.N_z
    for k in range(2 * n_z - 1):
        dz_vox = k - (n_z - 1)
        # Plane lz is at PLANE_IFACES[lz]; index 0 is the observation plane.
        lz, mz = (dz_vox, 0) if dz_vox >= 0 else (0, -dz_vox)
        rcv, src = PLANE_IFACES[lz], PLANE_IFACES[mz]
        lay = _p_tilde(model, src, rcv)
        # The whole-space SUBTRACTION must use the COMPLEX medium, to match the
        # layered propagator it is subtracted from; only the kernel builder
        # needs the real one.
        if dz_vox == 0:
            ws = same_depth_kernel_9x9(np.array([EPS]), 0.0, OM, ref_c)[:, :, 0]
        else:
            ws = vertical_kernel_9x9(np.array([EPS]), 0.0, dz_vox * D, OM, ref_c)[:, :, 0]
        kh[k, 0, 0] += (lay - ws) / D**2
    return kh


def _run(*, dressed: bool, uniform: bool) -> float:
    m_ref, m_full = _model(False, uniform=uniform), _model(True, uniform=uniform)
    s_p, s_s = m_ref.complex_slowness_p(), m_ref.complex_slowness_s()
    # TWO REFERENCES, and the split is forced rather than chosen.
    # `inter_voxel_propagator` -- the volume-averaged nearest-neighbour object
    # that halves the discretisation floor -- allocates REAL arrays and raises
    # on a complex medium ("Cannot cast ufunc 'add' output from complex128 to
    # float64"). It has no radiation part. So the kernel builder gets the REAL
    # medium while corrected_layered_9x9 keeps the complex attenuative one it
    # needs for attenuation consistency.
    #
    # The inconsistency that introduces is 1/(2Q) = 5e-5 at Q = 1e4, more than
    # an order below the 1.8e-3 floor, so it cannot affect the verdict. Stated
    # rather than hidden: it is a real approximation, just a harmless one here.
    ref_c = ReferenceMedium(1.0 / s_p[SCAT_IFACE], 1.0 / s_s[SCAT_IFACE], m_ref.rho[SCAT_IFACE])
    ref = ReferenceMedium(A0, B0, R0)

    geom = SlabGeometry(M=M, N_z=N_Z + 1, a=A_HALF)  # +1 for the observation plane
    ones = np.ones((N_Z + 1, M, M))
    material = SlabMaterial(Dlambda=D_LAM * ones, Dmu=D_MU * ones, Drho=D_RHO * ones, ref=ref)
    t0 = compute_slab_tmatrices(geom, material, OM)
    t0[0] = 0.0  # the observation plane scatters nothing

    src_vec = np.zeros(9, dtype=complex)
    src_vec[0] = 1.0
    planes = PLANE_IFACES
    psi0 = np.zeros((N_Z + 1, M, M, 9), dtype=complex)
    exact = np.zeros(9, dtype=complex)
    for lz, iface in enumerate(planes):
        psi0[lz, :, :, :] = _p_tilde(m_ref, SRC_IFACE, iface) @ src_vec
    exact = _p_tilde(m_full, SRC_IFACE, planes[0]) @ src_vec

    kh = (
        _dressed_kernel(m_ref, ref, ref_c, geom)
        if dressed
        else build_slab_kernels(geom, OM, ref, volume_averaged=True, periodic=True)
    )
    res = compute_slab_scattering(
        geom,
        material,
        OM,
        np.array([1.0, 0.0, 0.0]),
        "P",
        periodic=True,
        gmres_tol=1e-12,
        psi0=psi0,
        kernel_hat=kh,
    )
    got = res.psi[0, M // 2, M // 2]
    return float(np.abs(got - exact).max() / np.abs(exact).max())


def main() -> int:
    ka = OM / A0 * A_HALF
    print("=" * 78)
    print("GATE -- the thesis two-potential formulation, periodic lattice")
    print(f"  ka = {ka:.4f} (validated range is ka < 0.3); N_z = {N_Z}, M = {M}")
    print("  periodic Weyl lattice sum + volume-averaged nearest neighbours")
    print("  floor measured at 1.8e-3 by scripts/measure_periodic_floor.py")
    print("=" * 78)

    t1 = _run(dressed=True, uniform=False)
    t2 = _run(dressed=False, uniform=False)
    t3 = _run(dressed=True, uniform=True)

    print(f"\n  [T1] thesis ordering (kernel dressed with DeltaG0) : {t1:.4e}")
    print(f"  [T2] dress-after     (kernel left whole-space)    : {t2:.4e}")
    print(f"  [T3] discretisation control (uniform background)  : {t3:.4e}")

    conclusive = t2 > 5.0 * t3
    confirmed = conclusive and t1 < 2.0 * t3
    refuted = conclusive and t1 > 5.0 * t3

    print("\n" + "=" * 78)
    if not conclusive:
        print("  INCONCLUSIVE: dress-after is not separated from the discretisation")
        print("  floor, so this configuration cannot tell the orderings apart and")
        print("  nothing here is evidence about the formulation.")
    elif confirmed:
        print("  CONFIRMED: the thesis ordering sits at the discretisation floor")
        print("  while dress-after does not. Reference + contrast reproduces the")
        print("  medium with the contrast folded into the background, which is the")
        print("  formulation's correctness condition.")
    elif refuted:
        print("  REFUTED: dress-after IS separated, so the test discriminates, and")
        print("  the thesis ordering still misses the exact answer. Look at the")
        print("  stratified propagator and the composition, not the ordering.")
    else:
        print("  AMBIGUOUS: between the thresholds. Report the numbers, claim")
        print("  nothing.")
    print("=" * 78)
    return 0 if confirmed else 1


if __name__ == "__main__":
    raise SystemExit(main())
