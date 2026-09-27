#!/usr/bin/env python3
"""MEASUREMENT: the thesis ordering under REFINEMENT at fixed physics.

WHY.  The periodic thesis gate compares the two orderings with a uniform-
background floor [T3], and with the exact lateral sum it is inconclusive with
the thesis arm WORSE than dress-after (``gate_thesis_formulation_periodic``).
A floor measured on a different (uniform) problem may not be the discretisation
error of the stratified runs.  Refinement needs no floor: the stratified-
reference Lippmann-Schwinger identity is EXACT, so the thesis ordering carries
discretisation error only and must converge to ZERO as the voxels shrink,
while dress-after carries a genuine ordering error and must converge to a
NONZERO limit.  A thesis arm that plateaus instead is a refutation.

FIXED PHYSICS.  A slab of thickness D = 2 m carrying the contrast, lying directly
on the gate's reflector half-space; source 12 m above the slab; water above.
Refinement n: the slab is n planes of cubes of side d = D/n, on half-pitch
sublayers near the slab (thick layers elsewhere), with the observation plane
(T0 = 0) one pitch above the top scattering plane.  Only the specular block
acts, so the lateral lattice is minimal (M = 2).  The kernel is the gate's:
exact Ewald lateral sum with the source-cell average.  The dressing is the
layered reverberation at k_par -> 0 per depth separation, taken at the
SCATTERING planes' depths (the Toeplitz kernel holds one block per dz).

THE FIRST QUESTION is whether the uniform floor [T3] falls with n at all: a
scale-invariant discretisation bias would make refinement useless here too.

RESULT (after passing ``T_local=t0``: before that the solve rebuilt T from the
material and the observation/empty planes scattered, which produced a spurious
n-independent "floor" of 3.565e-4):
    n   [T1] thesis  Toeplitz   [T2] dress-after  [T3] control  T1/T3
    1    1.199e-5    1.199e-5      1.163e-3        7.000e-6     1.71
    2    5.720e-6    1.969e-5      1.156e-3        3.371e-6     1.70
    3    3.666e-6    1.615e-5      1.154e-3        2.184e-6     1.68
    4    2.641e-6    1.321e-5      1.153e-3        1.590e-6     1.66
    6    1.620e-6    9.482e-6      1.152e-3        9.969e-7     1.63
The control converges as 1/n; dress-after carries an n-independent ordering
error; the thesis ordering converges as 1/n at ~1.65x the control -- once the
reverberation is added PAIR BY PAIR (``run_pairwise``).  The Toeplitz column,
one reverberation block per depth separation as ``kernel_hat`` must store it,
refines only slowly: the reverberation off the reflector depends on z_i + z_j,
not z_i - z_j.  That is a limitation of the Toeplitz kernel, not of the
formulation; ``apply_g0_3d`` stores every pair.

Run:  conda run -n seismic python scripts/measure_thesis_refinement.py
SI units, as the gate.
"""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "PhD_fortran_code"))
sys.path.insert(0, "/Users/tod/Desktop/SeismicInversion")

import cubic_scattering.layered_correction as LC  # noqa: E402
from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from cubic_scattering.pair_propagators import plane_reference_medium  # noqa: E402
from cubic_scattering.slab_scattering import (  # noqa: E402
    SlabGeometry,
    SlabMaterial,
    build_slab_kernels,
    compute_slab_scattering,
    compute_slab_tmatrices,
)
from cubic_scattering.sweep_kernels import same_depth_kernel_9x9, vertical_kernel_9x9  # noqa: E402
from scripts import gate_thesis_formulation_periodic as gate  # noqa: E402

A0, B0, R0 = gate.A0, gate.B0, gate.R0
OM, EPS = gate.OM, gate.EPS
D_SLAB = 2.0
Z_SRC = -12.0  # source depth relative to the slab top (z down)
M = 2
KERNEL_KW = gate.KERNEL_KW
T_START = time.perf_counter()


class Geometry:
    """Layers and plane interfaces for refinement n (depth z down, slab top at z = 0)."""

    def __init__(self, n: int, gap: int = 1) -> None:
        self.n = n
        self.gap = gap  # observation plane this many pitches above the slab; gap - 1 empty planes between
        self.d = D_SLAB / n
        self.h = self.d / 2  # sublayer thickness
        self.n_fine = int(round((D_SLAB - Z_SRC) / self.h))

    def iface(self, z: float) -> int:
        """Interface index at depth z (interface j lies between layers j and j+1)."""
        return 1 + int(round((z - Z_SRC) / self.h))

    def z_iface(self, j: int) -> float:
        return Z_SRC + (j - 1) * self.h

    @property
    def planes(self) -> tuple[int, ...]:
        """Observation plane, gap - 1 empty planes, then the n scattering planes at the voxel centres."""
        zs = [(i - self.gap + 0.5) * self.d for i in range(self.gap)] + [
            (k + 0.5) * self.d for k in range(self.n)
        ]
        return tuple(self.iface(z) for z in zs)

    def model(self, *, contrast: bool, uniform: bool):
        """Water, background down to the source, fine sublayers to the slab bottom, reflector below."""
        import Kennett_Reflectivity.layer_model as lm

        n_l = 2 + self.n_fine  # layer 0 water, 1 background to the source, then the fine layers
        al = [1500.0, A0, *([A0] * self.n_fine), A0]
        be = [0.0, B0, *([B0] * self.n_fine), B0]
        rh = [1030.0, R0, *([R0] * self.n_fine), R0]
        th = [3000.0, 300.0, *([self.h] * self.n_fine), np.inf]
        if uniform:
            # a fluid matched to the background: transparent to P at normal incidence. Layer 0 is
            # ALWAYS a fluid to the layered solver, so S reflects totally off interface 0 whatever
            # beta[0] is (Mathematica/OceanBoundary.wl); P-only studies (this one) are unaffected.
            al[0], rh[0] = A0, R0
        else:
            d_al, d_be, d_rh = gate.REFL_JUMP
            al[-1], be[-1], rh[-1] = A0 + d_al, B0 + d_be, R0 + d_rh
        if contrast:
            lam0, mu0 = R0 * (A0**2 - 2 * B0**2), R0 * B0**2
            r1 = R0 + gate.D_RHO
            a1 = float(np.sqrt((lam0 + gate.D_LAM + 2 * (mu0 + gate.D_MU)) / r1))
            b1 = float(np.sqrt((mu0 + gate.D_MU) / r1))
            for lay in range(2, n_l):  # layer lay spans interfaces lay-1 .. lay
                top, bot = self.z_iface(lay - 1), self.z_iface(lay)
                if top >= -1e-9 and bot <= D_SLAB + 1e-9:
                    al[lay], be[lay], rh[lay] = a1, b1, r1
        return lm.LayerModel.from_arrays(
            alpha=al,
            beta=be,
            rho=rh,
            thickness=th,
            Q_alpha=[1e4] * (n_l + 1),
            Q_beta=[1e10, *([1e4] * n_l)],
        )


def p_tilde(model, src: int, rcv: int) -> np.ndarray:
    """The layered propagator at k_par -> 0, one 9x9."""
    return LC.corrected_layered_9x9(model, OM, np.array([EPS]), np.array([0.0]), src, rcv)[0]


def run(geo: Geometry, *, dressed: bool, uniform: bool) -> tuple[float, float]:
    """(absolute error at the observation plane, |exact|) for one arm."""
    m_ref = geo.model(contrast=False, uniform=uniform)
    m_full = geo.model(contrast=True, uniform=uniform)
    planes = geo.planes
    n_z = len(planes)
    ref = ReferenceMedium(A0, B0, R0)
    s0 = geo.gap  # index of the first scattering plane
    ref_c = plane_reference_medium(m_ref, planes[s0])  # the scattering planes' own (complex) medium
    geom = SlabGeometry(M=M, N_z=n_z, a=geo.d / 2)
    ones = np.ones((n_z, M, M))
    material = SlabMaterial(
        Dlambda=gate.D_LAM * ones, Dmu=gate.D_MU * ones, Drho=gate.D_RHO * ones, ref=ref
    )
    t0 = compute_slab_tmatrices(geom, material, OM)
    t0[:s0] = 0.0  # the observation plane and the empty planes scatter nothing
    src_vec = np.zeros(9, dtype=complex)
    src_vec[0] = 1.0
    src_iface = geo.iface(Z_SRC)
    psi0 = np.zeros((n_z, M, M, 9), dtype=complex)
    for lz, j in enumerate(planes):
        psi0[lz] = p_tilde(m_ref, src_iface, j) @ src_vec
    exact = p_tilde(m_full, src_iface, planes[0]) @ src_vec
    kh = build_slab_kernels(geom, OM, ref, **KERNEL_KW).copy()
    if dressed:
        for k in range(2 * n_z - 1):
            dz = k - (n_z - 1)
            # one Toeplitz block per dz: take it between SCATTERING planes where one exists
            if dz >= 0:
                rcv, src = (planes[s0 + dz], planes[s0]) if s0 + dz < n_z else (planes[dz], planes[0])
            else:
                rcv, src = (planes[s0], planes[s0 - dz]) if s0 - dz < n_z else (planes[0], planes[-dz])
            ws = (
                same_depth_kernel_9x9(np.array([EPS]), 0.0, OM, ref_c)[:, :, 0]
                if dz == 0
                else vertical_kernel_9x9(np.array([EPS]), 0.0, dz * geo.d, OM, ref_c)[:, :, 0]
            )
            kh[k, 0, 0] += (p_tilde(m_ref, src, rcv) - ws) / geo.d**2
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
        T_local=t0,  # without it the solve rebuilds T from the material: the observation plane scatters
    )
    got = res.psi[0, 0, 0]
    return float(np.abs(got - exact).max()), float(np.abs(exact).max())


def run_pairwise(geo: Geometry, *, dressed: bool) -> tuple[float, float, np.ndarray]:
    """The stratified arm solved as the specular chain, dressing each plane PAIR exactly.

    At normal incidence on a laterally uniform slab only the k_par = 0 block acts, so the
    Foldy-Lax system is exactly a chain of n_z planes, psi_i = psi0_i + sum_j K_ij T_j psi_j.
    The Toeplitz kernel of ``run`` holds one reverberation block per depth separation, but
    the reverberation off the reflector depends on BOTH depths (its path is z_src + z_rcv),
    so here K_ij = kernel(i - j) + [P~(j -> i) - W(i - j)] / d^2 for every pair.

    Returns:
        (absolute error, |exact|, the observation-plane field).
    """
    m_ref = geo.model(contrast=False, uniform=False)
    m_full = geo.model(contrast=True, uniform=False)
    planes = geo.planes
    n_z = len(planes)
    ref = ReferenceMedium(A0, B0, R0)
    ref_c = plane_reference_medium(m_ref, planes[geo.gap])
    geom = SlabGeometry(M=M, N_z=n_z, a=geo.d / 2)
    ones = np.ones((n_z, M, M))
    material = SlabMaterial(
        Dlambda=gate.D_LAM * ones, Dmu=gate.D_MU * ones, Drho=gate.D_RHO * ones, ref=ref
    )
    t0 = compute_slab_tmatrices(geom, material, OM)[:, 0, 0]
    t0[: geo.gap] = 0.0
    src_vec = np.zeros(9, dtype=complex)
    src_vec[0] = 1.0
    src_iface = geo.iface(Z_SRC)
    psi0 = np.array([p_tilde(m_ref, src_iface, j) @ src_vec for j in planes])
    exact = p_tilde(m_full, src_iface, planes[0]) @ src_vec
    kh = build_slab_kernels(geom, OM, ref, **KERNEL_KW)[:, 0, 0]
    eps = np.array([EPS])
    big = np.eye(9 * n_z, dtype=complex)
    for i in range(n_z):
        for j in range(n_z):
            dz = i - j
            k = kh[dz + n_z - 1].copy()
            if dressed:
                ws = (
                    same_depth_kernel_9x9(eps, 0.0, OM, ref_c)[:, :, 0]
                    if dz == 0
                    else vertical_kernel_9x9(eps, 0.0, dz * geo.d, OM, ref_c)[:, :, 0]
                )
                k += (p_tilde(m_ref, planes[j], planes[i]) - ws) / geo.d**2
            big[9 * i : 9 * i + 9, 9 * j : 9 * j + 9] -= k @ t0[j]
    got = np.linalg.solve(big, psi0.reshape(-1)).reshape(n_z, 9)[0]
    return float(np.abs(got - exact).max()), float(np.abs(exact).max()), got


def main() -> int:
    """Arms per refinement; the trend of [T3] with n decides whether refinement can separate them.

    Returns:
        0.
    """
    print("=" * 78)
    print("THESIS ORDERING UNDER REFINEMENT: slab D = 2 m in n planes of cubes of side D/n")
    print("=" * 78)
    cols = (
        ("n", 2),
        ("ka", 8),
        ("[T1] thesis", 12),
        ("Toeplitz", 10),
        ("[T2] dress-after", 17),
        ("[T3] control", 13),
    )
    head = " ".join(f"{c:>{w}}" for c, w in cols)
    print(f"  {head} {'T1/T3':>6} {'T2/T3':>6} {'chain check':>12}")
    for n in (1, 2, 3, 4, 6):
        geo = Geometry(n)
        a1, s1, _ = run_pairwise(geo, dressed=True)
        a1t, _ = run(geo, dressed=True, uniform=False)
        a2, _ = run(geo, dressed=False, uniform=False)
        a2c, _, _ = run_pairwise(geo, dressed=False)  # the chain against the package solve
        a3, s3 = run(geo, dressed=True, uniform=True)
        r1, r1t, r2, r3 = a1 / s1, a1t / s1, a2 / s1, a3 / s3
        ka = OM / A0 * geo.d / 2
        print(
            f"  {n:2d} {ka:8.4f} {r1:12.3e} {r1t:10.3e} {r2:17.3e} {r3:13.3e} {r1 / r3:6.2f} {r2 / r3:6.2f}"
            f" {abs(a2c - a2) / s1:12.1e}   [{time.perf_counter() - T_START:5.0f} s]",
            flush=True,
        )
    print("\n  (relative errors, each arm by its own exact field; [T1] dresses every plane pair,")
    print("   'Toeplitz' one pair per depth separation as the package kernel must; 'chain check'")
    print("   is the undressed chain against the package's own solve)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
