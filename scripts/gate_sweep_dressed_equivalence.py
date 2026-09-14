#!/usr/bin/env python3
"""GATE: the same-plane self-energy, checked three ways.

The intra-plane closure puts a layer-reverberation self-energy on the diagonal of
G0 -- the path that leaves a voxel, reflects off a layer boundary and returns to
that same voxel. T0 is the whole-space single-site T-matrix and does not contain
it. That is a THEORY argument; these gates are what can be measured about it.

  [D1] DRESSED-T EQUIVALENCE. A self-energy may equivalently be absorbed into the
       T-matrix. Writing G0 = G_off + D with D block-diagonal,

           A:  b = T0 psi,      (I - (G_off + D) T0) psi  = psi_inc
           B:  b = Td psi',     (I - G_off Td)      psi'  = psi_inc,
                                 Td = T0 (I - D T0)^-1

       The two SOURCES b must be identical -- the fields psi and psi' are not,
       they differ by the self-return, so comparing psi would be wrong. This is
       an exact algebraic identity, so it is a 1e-12 test. It catches a
       self-energy applied on the wrong side, double-counted, or sign-flipped;
       it does NOT independently confirm the VALUE of D.

       Note which part of the vertical stack's diagonal is the self-energy: the
       block carries Dx = 0 AND Dx != 0 (same-plane inter-voxel reverberation).
       Only Dx = 0 is a self-energy. Working with the dense operator makes that
       split exact by definition -- D is literally the block-diagonal of G0.

  [D2] RECESSION LIMIT. Move the reflecting contrast away and the self-energy
       must vanish, returning the whole-space answer. A magnitude statement, not
       just a structural one, and it is made at SOLVE level rather than on the
       kernel.

  [D3] RECIPROCITY. The reverberation must satisfy the symplectic law
       G(i<-j)(+k) = J6 [G(j<-i)(-k)]^T J6 -- the invariant
       gate_stratified_correction validated for STRATIFIED media.

       NOT W-symmetry. W.M symmetric (GATE F) was established on the WHOLE-SPACE
       closed form, and DeltaG does not satisfy it: measured 0.33 to 1.06 across
       k_x, at every separation, not only at dz = 0. That is recorded here rather
       than quietly dropped, because the reason is informative. At coincident
       interfaces the full layered kernel fails the symplectic law itself
       (9.5e-3) while DeltaG passes it (8.1e-14): the non-reciprocal part is the
       one-sided jump carried by the DIRECT term, and removing that term is
       exactly what the subtraction does. So the reverberation is the clean
       reciprocal object and the whole-space W-symmetry is simply the wrong
       invariant to demand of it.

       Reciprocity is homogeneous of degree one and therefore blind to an overall
       scale error -- which is why it is reported with D2 and never alone.

Run:  conda run -n seismic python scripts/gate_sweep_dressed_equivalence.py
Seismic units (km/s, g/cm3), time convention e^{-i omega t}.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "PhD_fortran_code"))
sys.path.insert(0, "/Users/tod/Desktop/SeismicInversion")

from cubic_scattering.directional_sweeps import (  # noqa: E402
    LayeredBackground,
    apply_g0,
    build_g0_cache,
    make_sweep_grid,
)
from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from cubic_scattering.layered_correction import W9  # noqa: E402
from Kennett_Reflectivity.layer_model import LayerModel  # noqa: E402

ALPHA, BETA, RHO, Q = 4.0, 2.22, 2.6, 2.0
PITCH, N_LAY, FREQ = 1.0, 16, 6.0
N_Z, N_X = 1, 4
PLANE_IFACE = 8


def model(contrast_layers: tuple[int, ...] = ()) -> LayerModel:
    """Ocean over N_LAY crust layers; a fast slab at the given layer indices.

    The plane sits at interface 8, so layers 8 and 9 must match: a material jump
    on the plane makes the correction operator K two-valued and is refused.
    """
    al = [1.5, *([ALPHA] * N_LAY), ALPHA]
    be = [0.0, *([BETA] * N_LAY), BETA]
    rh = [1.03, *([RHO] * N_LAY), RHO]
    for lay in contrast_layers:
        al[lay], be[lay], rh[lay] = 6.5, 3.7, 3.3
    return LayerModel.from_arrays(
        alpha=al,
        beta=be,
        rho=rh,
        thickness=[3.0, *([PITCH] * N_LAY), np.inf],
        Q_alpha=[Q] * (N_LAY + 2),
        Q_beta=[1e10, *([Q] * N_LAY), Q],
    )


def reference_for(mod: LayerModel) -> ReferenceMedium:
    s_p, s_s = mod.complex_slowness_p(), mod.complex_slowness_s()
    return ReferenceMedium(1.0 / s_p[1], 1.0 / s_s[1], mod.rho[1])


def dense_g0(cache) -> np.ndarray:
    """Materialise G0 column by column. Small lattice, so this is cheap."""
    size = N_Z * N_X * 9
    shape = (N_Z, N_X, 9)
    out = np.zeros((size, size), dtype=complex)
    for c in range(size):
        e = np.zeros(size, dtype=complex)
        e[c] = 1.0
        out[:, c] = apply_g0(e.reshape(shape), cache).ravel()
    return out


def split_self_energy(g0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split G0 into its block-diagonal self-energy D and the remainder G_off."""
    d = np.zeros_like(g0)
    n_site = N_Z * N_X
    for s in range(n_site):
        sl = slice(9 * s, 9 * s + 9)
        d[sl, sl] = g0[sl, sl]
    return d, g0 - d


def solve_direct(g0: np.ndarray, t0: np.ndarray, psi_inc: np.ndarray) -> np.ndarray:
    """Formulation A: sources b = T0 psi with the self-energy left inside G0."""
    size = g0.shape[0]
    psi = np.linalg.solve(np.eye(size) - g0 @ t0, psi_inc)
    return t0 @ psi


def solve_dressed(g_off: np.ndarray, d: np.ndarray, t0: np.ndarray, psi_inc: np.ndarray) -> np.ndarray:
    """Formulation B: sources b = Td psi' with the self-energy absorbed into T."""
    size = g_off.shape[0]
    t_d = t0 @ np.linalg.inv(np.eye(size) - d @ t0)
    psi_p = np.linalg.solve(np.eye(size) - g_off @ t_d, psi_inc)
    return t_d @ psi_p


def block_t0(rng, scale: float) -> np.ndarray:
    """Site-block-diagonal T0. Random: no accidental structure to hide behind."""
    size = N_Z * N_X * 9
    out = np.zeros((size, size), dtype=complex)
    for s in range(N_Z * N_X):
        sl = slice(9 * s, 9 * s + 9)
        out[sl, sl] = scale * (rng.standard_normal((9, 9)) + 1j * rng.standard_normal((9, 9)))
    return out


def main() -> int:
    omega = 2 * np.pi * FREQ
    grid = make_sweep_grid(N_Z, N_X, PITCH, ky=0.3, n_kz=64, n_kx=128, kx_max=24.0)
    ok = True

    print("=" * 78)
    print("GATE -- same-plane self-energy: dressed-T equivalence, recession, symmetry")
    print(f"  crust alpha={ALPHA} beta={BETA} rho={RHO}, Q={Q}, pitch={PITCH} km")
    print(f"  freq={FREQ} Hz, plane at interface {PLANE_IFACE}, n_x={N_X}")
    print("=" * 78)

    mod = model(contrast_layers=(10, 11))
    cache = build_g0_cache(
        grid,
        reference_for(mod),
        omega,
        background=LayeredBackground(model=mod, plane_ifaces=(PLANE_IFACE,)),
    )
    g0 = dense_g0(cache)
    d_self, g_off = split_self_energy(g0)

    print("\n[D1] DRESSED-T EQUIVALENCE -- the two formulations must give one b")
    print(f"      {'T0 scale':>10} {'|D|/|G0|':>10} {'|b_A|':>11}   {'relative diff':>14}")
    rng = np.random.default_rng(20260913)
    for scale in (1e-3, 1e-1, 1.0):
        t0 = block_t0(rng, scale)
        psi_inc = rng.standard_normal(g0.shape[0]) + 0j
        b_a = solve_direct(g0, t0, psi_inc)
        b_b = solve_dressed(g_off, d_self, t0, psi_inc)
        rel = float(np.abs(b_a - b_b).max() / np.abs(b_a).max())
        good = rel < 1e-12
        ok = ok and good
        print(
            f"      {scale:10.0e} {np.abs(d_self).max() / np.abs(g0).max():10.3e} "
            f"{np.abs(b_a).max():11.3e}   {rel:14.3e}"
        )
    print(f"      target < 1e-12  ->  {'PASS' if ok else 'FAIL'}")

    print("\n[D1c] CONTROL -- dropping the self-energy must CHANGE the answer")
    t0 = block_t0(np.random.default_rng(7), 1.0)
    psi_inc = np.random.default_rng(8).standard_normal(g0.shape[0]) + 0j
    b_full = solve_direct(g0, t0, psi_inc)
    b_drop = solve_direct(g_off, t0, psi_inc)
    rel_drop = float(np.abs(b_full - b_drop).max() / np.abs(b_full).max())
    b_eq = solve_dressed(g_off, d_self, t0, psi_inc)
    rel_eq = float(np.abs(b_full - b_eq).max() / np.abs(b_full).max())
    # The bar is a RATIO, not an absolute size. How much the self-energy matters
    # depends on how close the reflector is and how hard the damping is; what
    # makes the equivalence non-vacuous is that the identity holds far more
    # tightly than the term it moves.
    margin = rel_drop / max(rel_eq, 1e-300)
    okc = margin > 1e6
    print(f"      zeroing D changes the sources by:      {rel_drop:.3e}")
    print(f"      moving D into T changes them by:       {rel_eq:.3e}")
    print(f"      margin (the identity beats the term):  {margin:.1e}x")
    print(f"      ->  {'PASS' if okc else 'FAIL'}")
    ok = ok and okc

    print("\n[D2] RECESSION -- move the reflector away, the self-energy must vanish")
    print("      and the layered solve must return the whole-space answer.")
    print(f"      {'contrast at layers':>20} {'|D|':>11} {'|b_lay - b_ws|/|b_ws|':>22}")
    ws_cache = build_g0_cache(grid, reference_for(model()), omega)
    ws_g0 = dense_g0(ws_cache)
    t0r = block_t0(np.random.default_rng(11), 1.0)
    psi_r = np.random.default_rng(12).standard_normal(g0.shape[0]) + 0j
    b_ws = solve_direct(ws_g0, t0r, psi_r)
    mags, diffs = [], []
    for layers in ((10, 11), (12, 13), (14, 15)):
        m = model(contrast_layers=layers)
        c = build_g0_cache(
            grid,
            reference_for(m),
            omega,
            background=LayeredBackground(model=m, plane_ifaces=(PLANE_IFACE,)),
        )
        gg = dense_g0(c)
        dd, _ = split_self_energy(gg)
        b_l = solve_direct(gg, t0r, psi_r)
        mag = float(np.abs(dd).max())
        dif = float(np.abs(b_l - b_ws).max() / np.abs(b_ws).max())
        mags.append(mag)
        diffs.append(dif)
        print(f"      {str(layers):>20} {mag:11.3e} {dif:22.3e}")
    okr = mags == sorted(mags, reverse=True) and diffs == sorted(diffs, reverse=True)
    print(f"      both fall monotonically as the reflector recedes  ->  {'PASS' if okr else 'FAIL'}")
    ok = ok and okr

    print("\n[D3] RECIPROCITY of the reverberation (symplectic law, 6x6)")
    from cubic_scattering.layered_correction import J6, correct_6x6
    from GlobalMatrix.layered_greens import layered_greens_6x6

    def g6(mod_, kx, ky, j, i):
        ss = mod_.complex_slowness_s()
        raw = layered_greens_6x6(
            mod_, omega, np.array([kx]), np.array([ky]), source_iface=j, receiver_iface=i
        )[0]
        return correct_6x6(raw, omega, ss[max(j, 1)], ss[max(i, 1)], kx, ky)

    uni = model()
    print(f"      {'pair':>8} {'k':>14}   {'on G_lay':>11} {'on dG':>11}")
    worst = 0.0
    # Only pairs whose LOCAL medium is unchanged by the contrast: subtracting a
    # uniform-model kernel at a plane whose own material differs is not a
    # reverberation, and reads as a spurious failure.
    for j, i, lbl in ((9, 8, "9->8"), (8, 8, "8->8")):
        for kx, ky in ((0.4, 0.3), (1.6, -0.5)):
            a, b = g6(mod, kx, ky, j, i), g6(mod, -kx, -ky, i, j)
            da = a - g6(uni, kx, ky, j, i)
            db = b - g6(uni, -kx, -ky, i, j)
            r_full = float(np.linalg.norm(a - J6 @ b.T @ J6) / np.linalg.norm(a))
            r_rev = float(np.linalg.norm(da - J6 @ db.T @ J6) / np.linalg.norm(da))
            worst = max(worst, r_rev)
            print(f"      {lbl:>8} ({kx:+.1f},{ky:+.1f})   {r_full:11.3e} {r_rev:11.3e}")
    oks = worst < 1e-11
    print(f"      worst on the reverberation: {worst:.3e}  ->  {'PASS' if oks else 'FAIL'}")
    print("      Note the 8->8 row: the FULL kernel fails this law while the")
    print("      reverberation passes it. The one-sided jump in the direct term is")
    print("      the non-reciprocal part, and subtracting it is what isolates a")
    print("      clean reciprocal object.")
    ok = ok and oks

    print("\n[D3b] W-symmetry, REPORTED not gated -- DeltaG does NOT satisfy it")
    worst_w = 0.0
    for s in range(N_Z * N_X):
        sl = slice(9 * s, 9 * s + 9)
        wd = W9 @ d_self[sl, sl]
        worst_w = max(worst_w, float(np.linalg.norm(wd - wd.T) / np.linalg.norm(wd)))
    print(f"      worst ||W D - (W D)^T|| / ||W D||: {worst_w:.3e}")
    print("      GATE F (W M symmetric) was established on the WHOLE-SPACE closed")
    print("      form. It is not an invariant of the layer reverberation, and")
    print("      demanding it here would be chasing a false gate -- the same trap")
    print("      GATE E turned out to be in the wrapper work.")

    print("\n" + "=" * 78)
    print(f"GATE dressed-T equivalence: {'PASS' if ok else 'FAIL'}")
    print("  Not established HERE -- every gate above is homogeneous of degree one")
    print("  -- is the absolute magnitude of D. That is now carried by")
    print("  scripts/gate_dg0_absolute_magnitude.py, which predicts the full 9x9")
    print("  forward from the Kennett reflection matrix with nothing fitted:")
    print("  2e-10 at weak, strong and multi-interface contrast.")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
