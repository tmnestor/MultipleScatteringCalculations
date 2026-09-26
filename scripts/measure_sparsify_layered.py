#!/usr/bin/env python3
"""MEASUREMENT, not a gate: the sparsifying preconditioner in a LAYERED background.

WHY.  In whole space, centre-anchored translation-invariant stencils cut GMRES
iterations ~20x at strong contrast (``measure_sparsify_preconditioner.py``).
The project's problem is stratified: G0 carries the layer reverberation, the
stencils must be fitted per DEPTH, and the question left open there is whether
the fit survives a material contrast inside the domain -- where the layered
table's own transverse rule already had to be tightened (``TransverseRule``).

GEOMETRY.  Six depth planes, one pitch apart, on half-pitch sublayers (planes at
even interfaces 10..20).  A material change at interface 15, HALF A PITCH from
planes 14 and 16: planes 10, 12, 14 in medium A (5, 3, 2.5), planes 16, 18, 20
in medium B (6.5, 3.7, 3.0).  Water above, Q = 50.  Pitch 0.25 km, 1.5 Hz: 8
points per S wavelength in A.  Transverse rule kr_max = 20/pitch at
dk pitch = 0.156 -- the rule measure_stack_table_cross_material.py found
necessary half a pitch from a contrast.

THE OPERATOR IS ASSEMBLED HERE, NOT BY build_g0_cache_3d.  That function adds a
single same-depth whole-space table built from one caller ``ref``, while
layered_stack_table subtracts each plane's OWN medium from its diagonal.  With
planes in two materials no single ref is right for every plane: a mismatched
plane's same-plane coupling is off by W(ref) - W(local).  No existing caller
straddles a contrast, so nothing is broken today; this script sidesteps it by
building one cache per material (same layered table, own same-depth table) and
taking each receiver plane's rows from the cache of its own medium.

STENCILS.  Per centre plane (6) and lateral type (9): near = the 3x3x3
neighbourhood clipped to the domain; far = every domain plane, lateral offsets
within |d| <= R, minus near.  Centre-anchored, lam = 1e-3.  CONTROL: the same
fit on the WHOLE-SPACE kernel of medium A -- what ignoring the layering costs.

RESULT (2026-09-27).  GMRES iterations, none / 27-voxel truncation /
whole-space stencils / layered per-depth stencils:
    rho 0.05: 6 / 5 / 4 / 4        rho 0.45: 15 / 12 / 7 / 7
    rho 1.52: 43 / 27 / 11 / 11    rho 3.68: 167 / 83 / 17 / 16
The stencils matter (5x over truncation at the strongest contrast), but the
layering in them does NOT: stencils fitted to medium A's whole-space kernel do
as well as per-depth layered ones, although the layered operator differs from
that kernel by 43-97% in medium B and 2-6% in A.  H is built from the TRUE local
blocks either way; only the far-annihilating combinations come from the fit.
Worst far leakage per plane 5e-2 to 1e-1, largest on the domain faces (10, 20),
not at the contrast (14, 16).

Run:  conda run -n seismic python scripts/measure_sparsify_layered.py
Seismic units (km, km/s, g/cm3, GPa); T0 from each plane's real velocities.
"""

import itertools
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.sparse.linalg import eigs, splu

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/Users/tod/Desktop/SeismicInversion")

from cubic_scattering.directional_sweeps import G0Cache3D, SweepGrid3D, apply_g0_3d  # noqa: E402
from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from cubic_scattering.pair_propagators import (  # noqa: E402
    TransverseRule,
    layered_stack_table,
    same_depth_table,
)
from cubic_scattering.slab_scattering import (  # noqa: E402
    SlabGeometry,
    SlabMaterial,
    compute_slab_tmatrices,
)
from scripts.measure_sparsify_preconditioner import assemble, dofs, run_gmres  # noqa: E402

PITCH = 0.25
HALF = PITCH / 2
N_LAY = 40
PLANES = (10, 12, 14, 16, 18, 20)
# First layer of medium B.  Interface j lies between layers j and j+1, so the
# jump is on interface 15 -- half a pitch from planes 14 (layers 14-15, A) and
# 16 (layers 16-17, B).  Layer 15 would put it ON plane 14, which
# assert_interface_continuous rightly refuses.
CONTRAST_LAYER = 16
MED_A, MED_B = (5.0, 3.0, 2.5), (6.5, 3.7, 3.0)
Q = 50.0
OM = 2 * np.pi * 1.5
N = 6  # domain: N planes x N x N
R = 5  # lateral far window
C = R + 1  # window centre; near minus far lateral offsets reach R + 1
W = 2 * C + 1
RULE = TransverseRule(kr_max=20.0 / PITCH, n_axis=256)
D9 = np.r_[np.ones(3), np.full(6, PITCH)]
T0_START = time.perf_counter()


def stage(label: str) -> None:
    """Print one timed stage."""
    print(f"  [{time.perf_counter() - T0_START:6.0f} s] {label}", flush=True)


def layer_model():
    """Water over half-pitch sublayers, medium A above interface 15, B below."""
    from Kennett_Reflectivity.layer_model import LayerModel

    a = [1.5, *([MED_A[0]] * N_LAY), MED_B[0]]
    b = [0.0, *([MED_A[1]] * N_LAY), MED_B[1]]
    r = [1.03, *([MED_A[2]] * N_LAY), MED_B[2]]
    for j in range(CONTRAST_LAYER, N_LAY + 2):
        a[j], b[j], r[j] = MED_B
    return LayerModel.from_arrays(
        alpha=a,
        beta=b,
        rho=r,
        thickness=[3.0, *([HALF] * N_LAY), np.inf],
        Q_alpha=[Q] * (N_LAY + 2),
        Q_beta=[1e10, *([Q] * N_LAY), Q],
    )


def plane_media(model) -> list[ReferenceMedium]:
    """Each plane's own (complex, attenuated) medium -- what layered_stack_table subtracts."""
    s_p, s_s = model.complex_slowness_p(), model.complex_slowness_s()
    return [ReferenceMedium(1.0 / s_p[j], 1.0 / s_s[j], model.rho[j]) for j in PLANES]


def plane_caches(model, media: list[ReferenceMedium]) -> list[G0Cache3D]:
    """One cache per plane: the shared layered table, that plane's own same-depth table."""
    grid = SweepGrid3D(n_z=N, n_x=W, n_y=W, pitch=PITCH)
    lay = layered_stack_table(
        N, W, W, PITCH, OM, media[0], model=model, plane_ifaces=PLANES, transverse=RULE
    )
    stage("layered table built")
    tables: dict[tuple, NDArray] = {}
    out = []
    for m in media:
        key = (complex(m.alpha), complex(m.beta), float(m.rho))
        if key not in tables:
            tables[key] = same_depth_table(W, W, PITCH, OM, m)
        out.append(G0Cache3D(grid=grid, same_depth=tables[key], layered=lay))
    return out


def apply_layered(src: NDArray, caches: list[G0Cache3D]) -> NDArray:
    """G0 with each receiver plane's same-depth part from ITS OWN medium."""
    out = np.zeros_like(src, dtype=complex)
    done: dict[int, NDArray] = {}
    for z, cache in enumerate(caches):
        k = id(cache.same_depth)
        if k not in done:
            done[k] = apply_g0_3d(src, cache)
        out[z] = done[k][z]
    return out


def whole_space_caches(medium: ReferenceMedium) -> list[G0Cache3D]:
    """CONTROL kernel: whole space of one medium for every plane (no layering)."""
    from cubic_scattering.directional_sweeps import build_g0_cache_3d

    cache = build_g0_cache_3d(SweepGrid3D(n_z=N, n_x=W, n_y=W, pitch=PITCH), medium, OM)
    return [cache] * N


def kernel(caches: list[G0Cache3D]) -> NDArray:
    """ker[z_r, z_s, dx + C, dy + C] = the 9x9 block, sources at the window centre."""
    ker = np.zeros((N, N, W, W, 9, 9), dtype=complex)
    for zs in range(N):
        for c in range(9):
            e = np.zeros((N, W, W, 9), dtype=complex)
            e[zs, C, C, c] = 1.0
            ker[:, zs, :, :, :, c] = apply_layered(e, caches)
    return ker


def dense_domain(caches: list[G0Cache3D]) -> NDArray:
    """The N^3 domain operator: the window's lateral sub-square 0..N-1."""
    size = 9 * N**3
    g = np.zeros((size, size), dtype=complex)
    col = 0
    for z, x, y, c in itertools.product(range(N), range(N), range(N), range(9)):
        e = np.zeros((N, W, W, 9), dtype=complex)
        e[z, x, y, c] = 1.0
        g[:, col] = apply_layered(e, caches)[:, :N, :N, :].ravel()
        col += 1
    return g


def domain_from_kernel(ker: NDArray) -> NDArray:
    """The N^3 domain operator from the kernel: laterally translation-invariant.

    G[(zr, x, y), (zs, x', y')] = ker[zr, zs, x - x' + C, y - y' + C]; lateral
    offsets within the domain are at most N - 1 <= C.  Seconds, where
    ``dense_domain`` (one apply per column) takes ~16 min; checked against it.

    Args:
        ker: ``kernel`` output.

    Returns:
        Shape (9 N^3, 9 N^3).
    """
    idx = np.arange(N)
    dx = idx[:, None] - idx[None, :] + C  # (x, x')
    blocks = ker[:, :, dx[:, :, None, None], dx[None, None, :, :]]  # (zr, zs, x, x', y, y', 9, 9)
    g = blocks.transpose(0, 2, 4, 6, 1, 3, 5, 7)  # (zr, x, y, a, zs, x', y', b)
    return g.reshape(9 * N**3, 9 * N**3)


def fit_types(ker: NDArray, lam: float = 1e-3) -> dict:
    """Centre-anchored stencils per (centre plane, lateral type), fitted once each."""
    out = {}
    for zc in range(N):
        for sx, sy in itertools.product((-1, 0, 1), repeat=2):

            def lat_ok(d: int, s: int, lim: int) -> bool:
                return (-lim if s >= 0 else 0) <= d <= (lim if s <= 0 else 0)

            near = [
                (a, bx, by)
                for a, bx, by in itertools.product((-1, 0, 1), repeat=3)
                if 0 <= zc + a < N and lat_ok(bx, sx, 1) and lat_ok(by, sy, 1)
            ]
            near.sort(key=lambda o: o != (0, 0, 0))
            near_set = set(near)
            far = [
                (z - zc, bx, by)
                for z in range(N)
                for bx in range(-R, R + 1)
                for by in range(-R, R + 1)
                if lat_ok(bx, sx, R) and lat_ok(by, sy, R) and (z - zc, bx, by) not in near_set
            ]
            k = np.zeros((9 * len(near), 9 * len(far)), dtype=complex)
            for i, (a, bx, by) in enumerate(near):
                for j, (fz, fx, fy) in enumerate(far):
                    blk = ker[zc + a, zc + fz, bx - fx + C, by - fy + C]
                    k[9 * i : 9 * i + 9, 9 * j : 9 * j + 9] = D9[:, None] * blk
            k_c, k_o = k[:9], k[9:]
            gram = k_o @ k_o.conj().T
            reg = lam * float(np.linalg.eigvalsh(gram)[-1]) * np.eye(gram.shape[0])
            xh = -np.linalg.solve((gram + reg).T, (k_c @ k_o.conj().T).T).T
            alpha = np.vstack([np.eye(9), xh.conj().T])
            leak = float(np.linalg.norm(alpha.conj().T @ k, 2) / np.linalg.norm(k_c, 2))
            out[zc, sx, sy] = (near, alpha, leak)
    return out


def place(types: dict) -> list:
    """Per domain voxel: (mu dofs, alpha), from its type."""
    out = []
    for z, x, y in itertools.product(range(N), repeat=3):
        sx = -1 if x == 0 else (1 if x == N - 1 else 0)
        sy = -1 if y == 0 else (1 if y == N - 1 else 0)
        near, alpha, _ = types[z, sx, sy]
        vox = np.array([((z + a) * N + (x + bx)) * N + (y + by) for a, bx, by in near])
        out.append((dofs(vox), alpha))
    return out


def t_blocks(strength: float) -> NDArray:
    """Cube T-matrices, contrast (2, 1 GPa, 0.1 g/cm3) x strength x U(0.5, 1.5), per plane's medium."""
    rng = np.random.default_rng(20260927)
    scale = strength * (0.5 + rng.random((N, N, N)))
    out = np.zeros((N, N, N, 9, 9), dtype=complex)
    for z, j in enumerate(PLANES):
        med = MED_A if j < CONTRAST_LAYER else MED_B
        ref = ReferenceMedium(*med)
        s = scale[z : z + 1]
        mat = SlabMaterial(Dlambda=2.0 * s, Dmu=1.0 * s, Drho=0.1 * s, ref=ref)
        out[z] = compute_slab_tmatrices(SlabGeometry(M=N, N_z=1, a=HALF), mat, OM)[0]
    return out


def main() -> int:
    """Iterations: none / whole-space stencils / layered per-depth stencils, over contrast.

    Returns:
        0.
    """
    print("=" * 78)
    print(
        f"SPARSIFYING PRECONDITIONER, LAYERED BACKGROUND: {N}^3 voxels, contrast between planes 14 and 16"
    )
    print("=" * 78)
    model = layer_model()
    media = plane_media(model)
    caches = plane_caches(model, media)
    ws = whole_space_caches(media[0])
    stage("caches built")
    ker_lay = kernel(caches)
    g = domain_from_kernel(ker_lay)
    # Checked against the slow, independent path (one apply per column).
    rng_c = np.random.default_rng(7)
    worst_col = 0.0
    for col in rng_c.choice(g.shape[0], 5, replace=False):
        z, rem = divmod(int(col), 9 * N * N)
        x, rem = divmod(rem, 9 * N)
        y, c = divmod(rem, 9)
        e = np.zeros((N, W, W, 9), dtype=complex)
        e[z, x, y, c] = 1.0
        want = apply_layered(e, caches)[:, :N, :N, :].ravel()
        worst_col = max(worst_col, float(np.abs(g[:, col] - want).max() / np.abs(want).max()))
    if not worst_col < 1e-12:
        msg = (
            f"domain_from_kernel disagrees with the per-column apply by {worst_col:.2e}.\n"
            "  Where: scripts/measure_sparsify_layered.py, domain_from_kernel\n"
            "  Valid: agreement to round-off (< 1e-12)\n"
            "  Fix:   check the (receiver - source) offset orientation and the axis transpose."
        )
        raise RuntimeError(msg)
    stage(f"domain operator from the kernel; 5 columns vs direct apply: {worst_col:.1e}")
    lay_types = fit_types(ker_lay)
    ws_types = fit_types(kernel(ws))
    stage("stencil types fitted (layered and whole-space control)")
    print("  worst far leakage per centre plane (layered fit):")
    for zc in range(N):
        worst = max(v[2] for k, v in lay_types.items() if k[0] == zc)
        print(f"    plane {PLANES[zc]} ({'A' if PLANES[zc] < CONTRAST_LAYER else 'B'}): {worst:.2e}")
    row_scale = np.tile(D9, N**3)
    lay, ws_st = place(lay_types), place(ws_types)
    size = g.shape[0]
    rng = np.random.default_rng(1)
    b: NDArray = np.asarray(rng.standard_normal(size) + 1j * rng.standard_normal(size))
    cols = ("rho", "none", "truncation", "whole-space st.", "layered st.", "|PA-H|/|PA| lay")
    widths = (7, 6, 11, 15, 12, 16)
    print(f"\n  {'strength':>8} " + " ".join(f"{c:>{w}}" for c, w in zip(cols, widths, strict=True)))
    trunc_mask = np.zeros((size, size), dtype=bool)
    for i, (mu, _) in enumerate(lay):
        trunc_mask[np.ix_(np.arange(9 * i, 9 * i + 9), mu)] = True
    for strength in (1.0, 10.0, 30.0, 60.0):
        tb = t_blocks(strength)
        t = sp.block_diag([tb[z, x, y] for z, x, y in itertools.product(range(N), repeat=3)]).toarray()
        a = np.eye(size) - g @ t
        rho = float(np.abs(eigs(g @ t, k=1, which="LM", return_eigenvectors=False))[0])
        row = [run_gmres(a, b, None)]
        lu_tr = splu(sp.csc_matrix(np.where(trunc_mask, a, 0.0)))
        row.append(run_gmres(a, b, lu_tr.solve))
        for sten in (ws_st, lay):
            h, p = assemble(g, t, sten, row_scale)
            lu_h = splu(h)
            row.append(run_gmres(a, b, lambda v, lu=lu_h, pp=p: lu.solve(pp @ v)))
        pa = p @ a
        err = float(np.linalg.norm(pa - h.toarray()) / np.linalg.norm(pa))
        print(
            f"  {strength:8g} {rho:7.3f} {row[0]:6d} {row[1]:11d} {row[2]:15d} {row[3]:12d} {err:16.2e}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
