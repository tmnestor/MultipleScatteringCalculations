"""Cross-check of the two-way march (the preconditioner of Algorithm 6.4 on the true contrast, DownUp).

One pass from zero, as printed in Section 6.4 (and the module's docstring), with the planes solved exactly:
    down, i from the top:    x'_i  = A_i^-1 (b_i + sum_{j<i} L_ij E x'_j)
    up, i from the bottom:   x''_i = A_i^-1 (b_i + (R E x')_i + (L E x')_i + sum_{j>i} U_ij E x''_j)
with A_i = M - K_0,ii E_i.  Every block is formed here as an explicit dense matrix (columns by unit vectors
through the plane's sampling and the stored kernel of that part), and the pass is evaluated by dense solves;
DownUp, with its inner GMRES at 1e-13, must reproduce it.  Also: the parts add up to the whole coupling of
LateralOperator, applied independently.
"""

import numpy as np
from cubic_scattering import interface_downup as idu
from cubic_scattering import interface_solver as iso
from cubic_scattering.effective_contrasts import MaterialContrast, ReferenceMedium
from cubic_scattering.graded_voxel.basis import source_expansion

A = ReferenceMedium(5.0, 3.0, 2.5)
B = ReferenceMedium(6.5, 3.6, 2.8)
H = 0.05
OMEGA = A.beta / (2 * H)

cells = [(iz, ix, iy) for iz in (-2, -1, 0, 1) for ix in range(2) for iy in range(2)]
cells.remove((0, 1, 1))
g = iso.InterfaceGrid(np.array(cells), H, A, B)
c = iso.InterfaceCouplings(g, workers=1)
op = iso.LateralOperator(c, OMEGA, p=1, r=0, parts=True)
f = np.random.default_rng(11).uniform(0.0, 0.5, len(cells))
ctr: list[MaterialContrast | None] = [
    MaterialContrast(fm * g.medium(m).lam, fm * g.medium(m).mu, fm * g.medium(m).rho)
    for m, fm in enumerate(f)
]
delta = iso.cell_contrasts(g, OMEGA, ctr, 0)
e = np.array([source_expansion(d, op.n_field)[: op.n_source, : op.na] for d in delta])
du = idu.DownUp(op, e, inner_tol=1e-13, passes=1)
layers, rows, na = du.layers, du.rows, op.na
size = {z: len(du.cells[z]) * rows for z in layers}


def dense(kernel, zr, zs):
    """The block K_{zr zs} E_zs as a dense (size zr) x (size zs) matrix, column by column."""
    out = np.zeros((size[zr], size[zs]), complex)
    for col in range(size[zs]):
        u = np.zeros(size[zs], complex)
        u[col] = 1.0
        acc = np.einsum("rcf,cf->rf", kernel, du._spectrum(zs, u.reshape(-1, na, 9)))
        out[:, col] = du._field(zr, acc).ravel()
    return out


m_plane = {z: np.kron(np.eye(len(du.cells[z])), np.kron(op.m9, np.eye(9))) for z in layers}
a_plane = {z: m_plane[z] - dense(du.k0[z], z, z) for z in layers}
lo = {k: dense(v, *k) for k, v in du.lo.items()}
up = {k: dense(v, *k) for k, v in du.up.items()}
refl = {k: dense(v, *k) for k, v in du.refl.items()}

# the parts add up to the whole coupling (LateralOperator's own product)
v = np.random.default_rng(5).standard_normal(len(cells) * na * 9) + 0j
full = v.reshape(len(cells), na, 9)
xz = {z: full[du.cells[z]].ravel() for z in layers}
src = np.einsum("ncbij,nbj->nci", e, full)
whole = op._convolve(src.reshape(len(cells), -1)) if hasattr(op, "_convolve") else None
split_err = 0.0
if whole is not None:
    for zr in layers:
        acc = dense(du.k0[zr], zr, zr) @ xz[zr]
        for parts in (lo, up, refl):
            for (r, s), mat in parts.items():
                if r == zr:
                    acc = acc + mat @ xz[s]
        got = acc.reshape(-1, na, 9)
        want = whole.reshape(len(cells), na, 9)[du.cells[zr]]
        split_err = max(split_err, np.abs(got - want).max() / np.abs(whole).max())

# one pass from zero, by dense solves
b = {z: full[du.cells[z]].ravel() for z in layers}
down: dict[int, np.ndarray] = {}
for i, z in enumerate(layers):
    rhs = b[z].copy()
    for zs in layers[:i]:
        rhs += lo[z, zs] @ down[zs]
    down[z] = np.linalg.solve(a_plane[z], rhs)
upx: dict[int, np.ndarray] = {}
for i, z in reversed(list(enumerate(layers))):
    rhs = b[z].copy()
    for (r, s), mat in refl.items():
        if r == z:
            rhs += mat @ down[s]
    for (r, s), mat in lo.items():
        if r == z:
            rhs += mat @ down[s]
    for zs in layers[i + 1 :]:
        rhs += up[z, zs] @ upx[zs]
    upx[z] = np.linalg.solve(a_plane[z], rhs)
want = np.zeros_like(full)
for z in layers:
    want[du.cells[z]] = upx[z].reshape(-1, na, 9)
got = du(v).reshape(full.shape)
err = np.abs(got - want).max() / np.abs(want).max()

print(
    f"{len(layers)} planes, {len(cells)} cells; parts of the split: K0 {len(du.k0)}, L {len(lo)}, "
    f"U {len(up)}, R {len(refl)}"
)
if whole is not None:
    print(f"K0 + L + U + R against LateralOperator's coupling: relative error {split_err:.2e}")
print(f"one DownUp pass against the dense two-way march: relative error {err:.2e}")
ok = err < 1e-10 and (whole is None or split_err < 1e-12)
print("PASS" if ok else "FAIL")
