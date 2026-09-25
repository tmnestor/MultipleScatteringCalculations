#!/usr/bin/env python3
"""GATE A1: the reciprocity law of the ASSEMBLED directional-sweep G0.

WHY THIS GATE EXISTS
--------------------
Note (IV), AdjointStateGradients, reduces the adjoint Foldy-Lax solve to the
FORWARD solve at -k_y, provided G0 and T0 obey one matching diagonal similarity
law. The pieces of G0 had been checked separately (whole-space symmetry,
gate_9x9_source_convention; the reverberation's J6 law, gate_sweep_dressed_
equivalence [D3]); the ASSEMBLED discrete operator -- lateral sweep plus
vertical sweep, with the same-plane reverberation on the diagonal of the
vertical stack -- had not been. That is the object the adjoint solve applies.

WHAT WAS FOUND (25 September 2026)
----------------------------------
The note first stated the law with the Voigt weight alone,
G0(k_y)^T = W G0(-k_y) W^-1. MEASURED, that fails at 0.29-0.74, flat under
quadrature refinement -- a convention error, not a quadrature one. A free fit
of the per-component ratio [G0(+k_y)^T]_ab / [G0(-k_y)]_ab is an exact outer
product d_a / d_b (spread 6e-14) with

    d = (1,1,1, -1,-1,-1, -1/2,-1/2,-1/2)  =  I_pm W,
    I_pm = diag(1,1,1, -1,-1,-1, -1,-1,-1),   W = diag(1,1,1, 1,1,1, 1/2,1/2,1/2).

I_pm is the parity between the displacement and strain halves of the state: the
two off-diagonal blocks of the 9x9 carry ONE spatial derivative, taken at the
receiver in one and at the source in the other, so exchanging source and
receiver flips their sign. W alone cannot see it. Hence the law is

    G0(k_y)^T = W_pm G0(-k_y) W_pm^-1,     W_pm = I_pm W.                           (A1)

T0 has no displacement-strain block, so I_pm T0 I_pm = T0, and with T0^T = W T0 W^-1
(gate_t0_reciprocity) T0^T = W_pm T0 W_pm^-1 as well. The adjoint system of the
gradient is the transpose of (I - T0 G0) -- NOT of (I - G0 T0) -- and

    (I - T0 G0(k_y))^T = I - G0^T T0^T = W_pm (I - G0(-k_y) T0) W_pm^-1,      (F)

the forward operator at -k_y. The note's first draft wrote the left side as
(I - G0 T0)^T; that is a different operator, and [c3] shows it does not obey
the law (2e-3 whole-space, 9e-2 stratified).

CHECKS
------
  [fit] free rank-one fit of the ratio: prints d and its spread
  [L]   REQUIRED    G^T(+k_y)  vs  W_pm G(-k_y) W_pm^-1        (asserted < 1e-10)
  [c1]  CALIBRATION G^T(+k_y)  vs  W G(-k_y) W^-1        (the note's first law; must FAIL)
  [c2]  CALIBRATION G^T(+k_y)  vs  W_pm G(+k_y) W_pm^-1        (no k_y flip; must FAIL)
  [F]   the Foldy-Lax identity (F) with physical Rayleigh T0 blocks, a different
        contrast in every voxel                           (asserted < 1e-10)
  [c3]  CALIBRATION (I - G0 T0)^T vs the same right side  (wrong order; must FAIL)

k_y != 0 is required: at k_y = 0 the SH channel decouples and the k_y flip is
invisible (the trap recorded in StratifiedComposition). The sweeps are
quadratures on grids symmetric about zero, so a transposition maps a node onto
a node; both resolutions are reported, and a residual that is FLAT under
refinement is a convention defect rather than a quadrature one.

Run:  conda run -n seismic python scripts/gate_a1_sweep_reciprocity.py
Seismic units (km/s, g/cm3), time convention e^{-i omega t}.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/Users/tod/Desktop/SeismicInversion")

from cubic_scattering.directional_sweeps import (  # noqa: E402
    LayeredBackground,
    build_g0_cache,
    make_sweep_grid,
    sweep_x,
    sweep_z,
)
from cubic_scattering.effective_contrasts import (  # noqa: E402
    MaterialContrast,
    ReferenceMedium,
    compute_cube_tmatrix,
)
from cubic_scattering.resonance_tmatrix import _sub_cell_tmatrix_9x9  # noqa: E402

W9 = np.array([1, 1, 1, 1, 1, 1, 0.5, 0.5, 0.5], dtype=float)
I_PM = np.array([1, 1, 1, -1, -1, -1, -1, -1, -1], dtype=float)
W_SIGNED = I_PM * W9
TOL = 1e-10


def dense(apply, n_z: int, n_x: int) -> np.ndarray:
    """Materialise a linear map on (n_z, n_x, 9) states, column by column."""
    size = n_z * n_x * 9
    mat = np.zeros((size, size), dtype=complex)
    for c in range(size):
        e = np.zeros(size, dtype=complex)
        e[c] = 1.0
        mat[:, c] = apply(e.reshape(n_z, n_x, 9)).ravel()
    return mat


def rel(a: np.ndarray, b: np.ndarray) -> float:
    """||a - b|| / ||a||."""
    return float(np.linalg.norm(a - b) / np.linalg.norm(a))


def similar(g: np.ndarray, d: np.ndarray) -> np.ndarray:
    """diag(d) g diag(d)^-1."""
    return (d[:, None] * g) / d[None, :]


def fit_ratio(gp: np.ndarray, gm: np.ndarray, n_sites: int) -> tuple[np.ndarray, float]:
    """Fit [gp^T]_ab / [gm]_ab = d_a / d_b per component pair; return d and the spread."""
    comp = np.tile(np.arange(9), n_sites)
    a_mat, b_mat = gp.T, gm
    ratio = np.full((9, 9), np.nan, dtype=complex)
    spread = 0.0
    floor_a, floor_b = 1e-8 * np.abs(a_mat).max(), 1e-8 * np.abs(b_mat).max()
    for a in range(9):
        for b in range(9):
            m = (comp[:, None] == a) & (comp[None, :] == b)
            av, bv = a_mat[m], b_mat[m]
            keep = (np.abs(av) > floor_a) & (np.abs(bv) > floor_b)
            if keep.any():
                r = av[keep] / bv[keep]
                ratio[a, b] = np.median(r.real) + 1j * np.median(r.imag)
                spread = max(spread, float(np.abs(r - ratio[a, b]).max() / abs(ratio[a, b])))
    d = ratio[:, 0] / ratio[0, 0]  # d_a / d_0, normalised to d_0 = 1
    return d, spread


def rayleigh_t0(n_z: int, n_x: int, pitch: float, omega: complex, ref: ReferenceMedium) -> np.ndarray:
    """Physical Rayleigh cube T0 blocks, a different contrast in every voxel."""
    rng = np.random.default_rng(20260925)
    a = 0.5 * pitch
    t0 = np.zeros((n_z, n_x, 9, 9), dtype=complex)
    for iz in range(n_z):
        for ix in range(n_x):
            dl, dm, dr = rng.uniform(-1.0, 2.0), rng.uniform(-0.5, 1.0), rng.uniform(-0.1, 0.2)
            res = compute_cube_tmatrix(omega, a, ref, MaterialContrast(Dlambda=dl, Dmu=dm, Drho=dr))
            t0[iz, ix] = _sub_cell_tmatrix_9x9(res, omega, a)
    return t0


def block_diag_t0(t0: np.ndarray) -> np.ndarray:
    """Dense block-diagonal T0 from (n_z, n_x, 9, 9) blocks."""
    blocks = t0.reshape(-1, 9, 9)
    n = 9 * blocks.shape[0]
    t_full = np.zeros((n, n), dtype=complex)
    for k, blk in enumerate(blocks):
        t_full[9 * k : 9 * k + 9, 9 * k : 9 * k + 9] = blk
    return t_full


def run_case(name, n_z, n_x, pitch, ky, omega, ref, background, quads) -> bool:
    """Measure the laws at each quadrature resolution; assert at the finest."""
    print(f"\n{name}")
    print(f"  n_z={n_z} n_x={n_x} pitch={pitch} k_y={ky} omega/2pi={omega / (2 * np.pi):.3f}")
    n_sites = n_z * n_x
    d_full, w_full = np.tile(W_SIGNED, n_sites), np.tile(W9, n_sites)
    z = np.repeat(np.arange(n_z), n_x * 9)
    same = z[:, None] == z[None, :]
    cols = f"{'[L] W_pm, flip':>12} {'[c1] W, flip':>13} {'[c2] W_pm, no flip':>16}"
    print(f"  {'(n_kz, n_kx)':>14} {'piece':<22} {cols}")
    t0 = rayleigh_t0(n_z, n_x, pitch, omega, ref)
    last: dict[str, float] = {}
    for n_kz, n_kx, kx_max in quads:
        ops = {}
        for sgn in (+1, -1):
            grid = make_sweep_grid(n_z, n_x, pitch, ky=sgn * ky, n_kz=n_kz, n_kx=n_kx, kx_max=kx_max)
            cache = build_g0_cache(grid, ref, omega, background=background)
            gx = dense(lambda s, c=cache: sweep_x(s, c.grid, c.split_right, c.split_left), n_z, n_x)
            gz = dense(lambda s, c=cache: sweep_z(s, c.grid, c.vertical), n_z, n_x)
            ops[sgn] = (gx, gz)
        (gxp, gzp), (gxm, gzm) = ops[+1], ops[-1]
        pieces = {
            "lateral (sweep_x)": (gxp, gxm),
            "vertical, same plane": (gzp * same, gzm * same),
            "vertical, inter-plane": (gzp * ~same, gzm * ~same),
            "ASSEMBLED G0": (gxp + gzp, gxm + gzm),
        }
        for pname, (gp, gm) in pieces.items():
            if np.linalg.norm(gp) == 0:
                continue
            r_l = rel(gp.T, similar(gm, d_full))
            r_c1 = rel(gp.T, similar(gm, w_full))
            r_c2 = rel(gp.T, similar(gp, d_full))
            print(f"  {f'({n_kz}, {n_kx})':>14} {pname:<22} {r_l:12.2e} {r_c1:13.2e} {r_c2:16.2e}")
            last[pname] = r_l
            last[pname + "|c1"], last[pname + "|c2"] = r_c1, r_c2
        gp, gm = gxp + gzp, gxm + gzm
        eye, t_full = np.eye(gp.shape[0]), block_diag_t0(t0)
        rhs = similar(eye - gm @ t_full, d_full)
        last["F"] = rel((eye - t_full @ gp).T, rhs)
        last["c3"] = rel((eye - gp @ t_full).T, rhs)
        print(f"  {f'({n_kz}, {n_kx})':>14} {'[F] (I - T0 G0)^T':<22} {last['F']:12.2e}")
        print(f"  {f'({n_kz}, {n_kx})':>14} {'[c3] (I - G0 T0)^T':<22} {'':12} {last['c3']:13.2e}")

    d_fit, spread = fit_ratio(gp, gm, n_sites)
    with np.printoptions(precision=3, suppress=True):
        im_d = np.abs(d_fit.imag).max()
        print(f"  [fit] d = {d_fit.real}   (max |Im d| {im_d:.1e}, spread {spread:.1e})")
    ok_l = last["ASSEMBLED G0"] < TOL and last["F"] < TOL
    ok_c = min(last["ASSEMBLED G0|c1"], last["ASSEMBLED G0|c2"], last["c3"]) > 1e-3
    print(f"  [L] + [F] at the finest quadrature < {TOL:.0e}  ->  {'PASS' if ok_l else 'FAIL'}")
    print(f"  [c1-3] calibration legs fail (> 1e-3)     ->  {'PASS' if ok_c else 'FAIL (gate blind)'}")
    return ok_l and ok_c


def layered_model():
    """The fast-slab crust of test_sweep_solver: 16 pitch-thick layers, planes 7 and 11."""
    from Kennett_Reflectivity.layer_model import LayerModel

    n_lay, pitch, q = 16, 1.0, 2.0
    al, be, rh = 4.0, 2.22, 2.6
    a = [1.5, *([al] * n_lay), al]
    b = [0.0, *([be] * n_lay), be]
    r = [1.03, *([rh] * n_lay), rh]
    for lay in (9, 10):
        a[lay], b[lay], r[lay] = 6.5, 3.7, 3.3
    model = LayerModel.from_arrays(
        alpha=a,
        beta=b,
        rho=r,
        thickness=[3.0, *([pitch] * n_lay), np.inf],
        Q_alpha=[q] * (n_lay + 2),
        Q_beta=[1e10, *([q] * n_lay), q],
    )
    s_p, s_s = model.complex_slowness_p(), model.complex_slowness_s()
    ref = ReferenceMedium(1.0 / s_p[1], 1.0 / s_s[1], model.rho[1])
    return model, ref


def main() -> int:
    """Whole-space case, then the stratified case."""
    print("=" * 96)
    print("GATE A1 -- assembled directional-sweep G0:  G(k_y)^T = W_pm G(-k_y) W_pm^-1,  W_pm = I_pm W")
    print("=" * 96)

    ok = run_case(
        "[W] WHOLE-SPACE background",
        n_z=2,
        n_x=4,
        pitch=0.25,
        ky=0.6,
        omega=2 * np.pi * (1.0 + 0.03j),
        ref=ReferenceMedium(alpha=5.0, beta=3.0, rho=2.5),
        background=None,
        quads=((128, 128, None), (512, 512, None)),
    )

    model, ref = layered_model()
    ok &= run_case(
        "[S] STRATIFIED background, fast slab between planes 7 and 11",
        n_z=2,
        n_x=4,
        pitch=1.0,
        ky=0.3,
        omega=2 * np.pi * 6.0,
        ref=ref,
        background=LayeredBackground(model=model, plane_ifaces=(7, 11)),
        quads=((64, 64, 3.0), (256, 256, 3.0)),
    )

    print("\n" + ("GATE A1 PASS" if ok else "GATE A1 FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
