"""Cross-check of Algorithm 5.1 in 3-D: the depth factorisation of the plane-averaged system.

(i)  The block LDU that DtNSweep builds at every lateral wavenumber, applied with its pivots inverted
     exactly (x = U^-1 S^-1 L^-1 b), must reproduce AveragedSweep's dense inverse. (ii) Each pivot S_z
     must be the Schur complement A_zz - A_z,<z (A_<z,<z)^-1 A_<z,z, computed independently: the
     discrete Dirichlet-to-Neumann map of the half-domain above plane z. Four planes (two each side of
     the interface) of 2 x 2 cells with a gap, contrast drawn on [0, 50] %.
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
op = iso.LateralOperator(c, OMEGA, p=1, r=0)
f = np.random.default_rng(11).uniform(0.0, 0.5, len(cells))
ctr: list[MaterialContrast | None] = [
    MaterialContrast(fm * g.medium(m).lam, fm * g.medium(m).mu, fm * g.medium(m).rho)
    for m, fm in enumerate(f)
]
delta = iso.cell_contrasts(g, OMEGA, ctr, 0)
e = np.array([source_expansion(d, op.n_field)[: op.n_source, : op.na] for d in delta])

avg = idu.AveragedSweep(op, e)
dtn = idu.DtNSweep(op, e)
n_l, rows = len(dtn.layers), dtn.rows

# (i) the exact LDU, applied at every wavenumber, against AveragedSweep
v = np.random.default_rng(5).standard_normal(len(cells) * op.na * 9) + 0j
full = v.reshape(len(cells), op.na, 9)
bh = [dtn._to_grid(z, full[dtn.cells[z]].reshape(len(dtn.cells[z]), -1)) for z in dtn.layers]
w: list[np.ndarray] = []
for i in range(n_l):
    wi = bh[i].copy()
    for k in range(i):
        wi -= np.einsum("frs,sf->rf", dtn.lo[i, k], w[k])
    w.append(wi)
y = [np.einsum("frs,sf->rf", dtn.s_inv[i], w[i]) for i in range(n_l)]
x: list[np.ndarray] = [np.empty(0)] * n_l
for k in reversed(range(n_l)):
    xk = y[k].copy()
    for j in range(k + 1, n_l):
        xk -= np.einsum("frs,sf->rf", dtn.up[k, j], x[j])
    x[k] = xk
out = np.zeros_like(full)
for i, z in enumerate(dtn.layers):
    out[dtn.cells[z]] = dtn._at_cells(z, x[i]).reshape(len(dtn.cells[z]), op.na, 9)
want = avg(v).reshape(full.shape)
err_i = np.abs(out - want).max() / np.abs(want).max()

# (ii) each pivot against the Schur complement from the averaged system, by an independent dense solve
a, _ = idu._averaged_system(op, e)
err_ii = 0.0
for k in range(n_l):
    akk = a[:, k * rows : (k + 1) * rows, k * rows : (k + 1) * rows]
    if k == 0:
        schur = akk
    else:
        lead = a[:, : k * rows, : k * rows]
        schur = akk - a[:, k * rows : (k + 1) * rows, : k * rows] @ np.linalg.solve(
            lead, a[:, : k * rows, k * rows : (k + 1) * rows]
        )
    err_ii = max(err_ii, np.abs(dtn.s[k] - schur).max() / np.abs(schur).max())

print(f"{n_l} planes, {a.shape[0]} lateral wavenumbers, {rows} unknowns per plane and wavenumber")
print(f"(i)  exact LDU vs AveragedSweep's dense inverse: relative error {err_i:.2e}")
print(f"(ii) pivots vs independent Schur complements (DtN maps): relative error {err_ii:.2e}")
print("PASS" if err_i < 1e-10 and err_ii < 1e-10 else "FAIL")
