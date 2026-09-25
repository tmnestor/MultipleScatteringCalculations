#!/usr/bin/env python3
"""GATE A1b: the transpose of the receiver map, R^T, in the stratified background.

WHY THIS GATE EXISTS
--------------------
The adjoint system of note (IV), AdjointStateGradients, is driven by R^T r-bar:
the data residual carried from the receivers back to the voxels. R maps the
voxel contrast sources tau = T0 psi to the field at a receiver, so its blocks are
the layered propagator G(r <- v). The note claimed, but had not measured, that
R^T is the reference field radiated FROM the receiver, by the diagonal law A1
established for the sweep operator. In 3-D there is no k_y to flip, so the law
is a pure source-receiver exchange:

    G(r <- v)^T = W_pm G(v <- r) W_pm^-1,    W_pm = I_pm W = diag(1,1,1, -1,-1,-1, -1/2,-1/2,-1/2).   (A1b)

For a receiver recording DISPLACEMENT, R = Pi_u G(r <- v) with Pi_u selecting rows
0..2, and W_pm^-1 Pi_u^T = Pi_u^T, so

    R^T r-bar = W_pm . [field at the voxels from a point FORCE r-bar at the receiver].   (P)

The residual is re-injected as a force at the receiver, propagated through the
reference by the existing incident-field code, and weighted by W_pm. No new
operator.

WHAT WAS FOUND (26 September 2026)
----------------------------------
The law holds, in the wavenumber domain to 5e-16 for every plane pair tried
(same and different materials, free surface off and on, Q = 2 and 1000), and in
real space to the quadrature floor. BUT at the documented transverse rule,
kr_max = 10/pitch, a receiver INSIDE the fast slab -- a different material from
the voxel planes -- fails at ~2e-2, flat in the node spacing. The cause is the
rule's CUTOFF, not the physics: refining kr_max at fixed dk takes the same pair
from 2.4e-2 (10/pitch) to 3.3e-10 (20/pitch) and 2.0e-10 (30/pitch).
layered_incident_field subtracts the whole-space kernel of the RECEIVER plane's
medium; when source and receiver media differ, what is left to transform is not
reverberation alone but also the difference between the direct paths, and the
10/pitch cutoff was measured (measure_dg0_transverse_rule) for reverberation
only. Same-material pairs are unaffected. layered_stack_table uses the same
construction and rule, so its cross-material entries are expected to carry the
same truncation; that is recorded, not tested, here.

CHECKS
------
  [K]   k-domain: corrected_layered_9x9 obeys G(r<-v)(k)^T = W_pm G(v<-r)(-k) W_pm^-1
        for same- and cross-material pairs, free surface off/on, Q = 2 and 1000.
        Cheap and exact; the free surface is only discriminating at high Q.
  [L]   real space: G(r<-v)^T vs W_pm G(v<-r) W_pm^-1 from layered_incident_field.
  [P]   R^T r-bar from the dense forward R, vs W_pm . (field of a force at the receiver).
  [c1]  CALIBRATION W in place of W_pm                      (must FAIL)
  [c2]  CALIBRATION no weight at all                     (must FAIL)

Real-space rules: 10/pitch REPORTED (the documented rule, and the deficiency);
20/pitch ASSERTED (< 1e-8, the transform's quadrature floor, not node-to-node
exact as A1 was); 30/pitch shows the residual has converged.

Run:  conda run -n seismic python scripts/gate_a1b_receiver_transpose.py   (~10 min)
Seismic units (km/s, g/cm3), time convention e^{-i omega t}.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/Users/tod/Desktop/SeismicInversion")

from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from cubic_scattering.layered_correction import corrected_layered_9x9  # noqa: E402
from cubic_scattering.pair_propagators import TransverseRule, layered_incident_field  # noqa: E402

W9 = np.array([1, 1, 1, 1, 1, 1, 0.5, 0.5, 0.5], dtype=float)
I_PM = np.array([1, 1, 1, -1, -1, -1, -1, -1, -1], dtype=float)
W_SIGNED = I_PM * W9
TOL_K = 1e-13
TOL_REAL = 1e-8

PITCH = 1.0
OMEGA = 2 * np.pi * 6.0
VOXEL_PLANES = (7, 11)
N_X, N_Y = 2, 2


def crust_model(q: float = 2.0):
    """The fast-slab crust of the sweep-solver tests: slab in layers 9 and 10."""
    from Kennett_Reflectivity.layer_model import LayerModel

    n_lay = 16
    al, be, rh = 4.0, 2.22, 2.6
    a = [1.5, *([al] * n_lay), al]
    b = [0.0, *([be] * n_lay), be]
    r = [1.03, *([rh] * n_lay), rh]
    for lay in (9, 10):
        a[lay], b[lay], r[lay] = 6.5, 3.7, 3.3
    return LayerModel.from_arrays(
        alpha=a,
        beta=b,
        rho=r,
        thickness=[3.0, *([PITCH] * n_lay), np.inf],
        Q_alpha=[q] * (n_lay + 2),
        Q_beta=[1e10, *([q] * n_lay), q],
    )


def local_ref(model, iface: int) -> ReferenceMedium:
    """The medium at a plane (layer max(iface, 1)), with its complex slowness."""
    s_p, s_s = model.complex_slowness_p(), model.complex_slowness_s()
    j = max(iface, 1)
    return ReferenceMedium(1.0 / s_p[j], 1.0 / s_s[j], model.rho[j])


def field(model, rule, free_surface, src_iface, src_xy, src_vec, planes):
    """layered_incident_field on a (len(planes), N_X, N_Y) lattice; every layer is one pitch."""
    return layered_incident_field(
        len(planes),
        N_X,
        N_Y,
        PITCH,
        OMEGA,
        local_ref(model, src_iface),
        source_xy=src_xy,
        source_vec=src_vec,
        dz_planes=tuple(float(p - src_iface) * PITCH for p in planes),
        model=model,
        plane_ifaces=tuple(planes),
        source_iface=src_iface,
        transverse=rule,
        free_surface=free_surface,
    )


def blocks(model, rule, free_surface, rcv_iface, rcv_xy):
    """G(r<-v) and G(v<-r) as (n_vox, 9, 9) arrays, voxels in (plane, ix, iy) order."""
    eye = np.eye(9, dtype=complex)
    voxels = [(p, ix, iy) for p in VOXEL_PLANES for ix in range(N_X) for iy in range(N_Y)]
    g_rv = np.zeros((len(voxels), 9, 9), dtype=complex)
    for n, (p, ix, iy) in enumerate(voxels):
        for b in range(9):
            f = field(model, rule, free_surface, p, (ix, iy), eye[b], (rcv_iface,))
            g_rv[n, :, b] = f[0, rcv_xy[0], rcv_xy[1]]
    g_vr = np.zeros((len(voxels), 9, 9), dtype=complex)
    for b in range(9):
        f = field(model, rule, free_surface, rcv_iface, rcv_xy, eye[b], VOXEL_PLANES)
        for n, (p, ix, iy) in enumerate(voxels):
            g_vr[n, :, b] = f[VOXEL_PLANES.index(p), ix, iy]
    return g_rv, g_vr


def rel(a: np.ndarray, b: np.ndarray) -> float:
    """||a - b|| / ||a||."""
    return float(np.linalg.norm(a - b) / np.linalg.norm(a))


def conj_by(g: np.ndarray, d: np.ndarray) -> np.ndarray:
    """diag(d) g diag(d)^-1 on a stack of 9x9 blocks."""
    return d[None, :, None] * g / d[None, None, :]


def k_domain() -> bool:
    """[K] the law on the wavenumber-domain kernel, where it should be exact."""
    print("\n[K] k-domain, corrected_layered_9x9:  G(r<-v)(k)^T = W_pm G(v<-r)(-k) W_pm^-1")
    kx = np.array([0.3, 0.8, 1.7, 3.0])
    ky = np.array([0.1, -0.5, 0.9, 2.0])
    cols = f"{'[K] W_pm':>10} {'[c1] W':>10}"
    print(f"    {'Q':>6} {'free surf':>9} {'pair (rcv<-src)':>16} {'material':>10} {cols}")
    ok = True
    for q in (2.0, 1000.0):
        model = crust_model(q)
        for fs in (False, True):
            for rcv, src in ((4, 7), (9, 7), (9, 11)):
                a = corrected_layered_9x9(model, OMEGA, kx, ky, src, rcv, free_surface=fs)
                b = corrected_layered_9x9(model, OMEGA, -kx, -ky, rcv, src, free_surface=fs)
                lhs = np.transpose(a, (0, 2, 1))
                r_d, r_w = rel(lhs, conj_by(b, W_SIGNED)), rel(lhs, conj_by(b, W9))
                mat = "cross" if 9 <= rcv <= 10 else "same"
                fs_s = "ON" if fs else "off"
                print(f"    {q:6.0f} {fs_s:>9} {f'{rcv} <- {src}':>16} {mat:>10} {r_d:10.2e} {r_w:10.2e}")
                ok &= r_d < TOL_K and r_w > 1e-3
    print(f"    -> {'PASS' if ok else 'FAIL'}")
    return ok


def real_space(model, rcv_iface, rcv_xy, free_surface, rules) -> bool:
    """[L], [P], [c1], [c2] for one receiver geometry, at each rule."""
    medium = "fast slab: CROSS-material" if 9 <= max(rcv_iface, 1) <= 10 else "background: same material"
    fs = "ON" if free_surface else "off"
    print(f"\n  receiver iface {rcv_iface} ({medium}), xy {rcv_xy}, free surface {fs}")
    cols = f"{'[L] W_pm':>10} {'[c1] W':>10} {'[c2] none':>10} {'[P] force':>10}"
    print(f"    {'rule (kr_max, n)':>18} {'role':>9} {cols}")
    ok = True
    for rule, role in rules:
        g_rv, g_vr = blocks(model, rule, free_surface, rcv_iface, rcv_xy)
        lhs = np.transpose(g_rv, (0, 2, 1))
        r_l, r_c1, r_c2 = rel(lhs, conj_by(g_vr, W_SIGNED)), rel(lhs, conj_by(g_vr, W9)), rel(lhs, g_vr)

        rng = np.random.default_rng(7)
        r_bar = rng.standard_normal(3) + 1j * rng.standard_normal(3)
        rt_dense = np.einsum("nab,a->nb", g_rv[:, :3, :], r_bar)
        force = np.concatenate([r_bar, np.zeros(6)])
        f = field(model, rule, free_surface, rcv_iface, rcv_xy, force, VOXEL_PLANES)
        r_p = rel(rt_dense, W_SIGNED[None, :] * f.reshape(-1, 9))

        tag = f"({rule.kr_max:.0f}, {rule.n_axis})"
        print(f"    {tag:>18} {role:>9} {r_l:10.2e} {r_c1:10.2e} {r_c2:10.2e} {r_p:10.2e}")
        if role == "asserted":
            ok &= r_l < TOL_REAL and r_p < TOL_REAL and min(r_c1, r_c2) > 1e-3
    print(f"    -> {'PASS' if ok else 'FAIL'}  (asserted rule, < {TOL_REAL:.0e})")
    return ok


def main() -> int:
    """k-domain first, then real space for a same- and a cross-material receiver."""
    print("=" * 96)
    print("GATE A1b -- receiver transpose:  G(r<-v)^T = W_pm G(v<-r) W_pm^-1,  R^T r = W_pm (force at rcv)")
    print(f"  fast-slab crust, omega/2pi = {OMEGA / (2 * np.pi):.1f}, voxel planes {VOXEL_PLANES},")
    print(f"  {N_X}x{N_Y} voxels per plane, pitch {PITCH} km; slab occupies layers 9-10")
    print("=" * 96)
    ok = k_domain()

    print("\n[L]/[P] real space, layered_incident_field, Q = 2")
    rules = (
        (TransverseRule(kr_max=10.0 / PITCH, n_axis=64), "reported"),
        (TransverseRule(kr_max=20.0 / PITCH, n_axis=128), "asserted"),
        (TransverseRule(kr_max=30.0 / PITCH, n_axis=192), "converged"),
    )
    model = crust_model(2.0)
    ok &= real_space(model, 4, (1, 0), False, rules)
    ok &= real_space(model, 9, (1, 1), False, rules)
    print("\n" + ("GATE A1b PASS" if ok else "GATE A1b FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
