"""The dense sphere system on its own: compute_sphere_foldy_lax solves what
assemble_sphere_foldy_lax_system assembles.

A study that solves the same system another way (the GMRES comparison of gate_imbedding_vs_foldy_lax.py)
takes the matrix from the assembly, so the two must not drift apart.
"""

import numpy as np

from cubic_scattering.effective_contrasts import MaterialContrast, ReferenceMedium
from cubic_scattering.sphere_scattering import assemble_sphere_foldy_lax_system, compute_sphere_foldy_lax

REF = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
CONTRAST = MaterialContrast(Dlambda=2e9, Dmu=1e9, Drho=100.0)


def test_solving_the_assembled_system_is_the_solver() -> None:
    args = (0.3 * REF.beta / 10.0, 10.0, REF, CONTRAST, 4, np.array([1.0, 0.0, 0.0]), "P")
    system = assemble_sphere_foldy_lax_system(*args, cell_average=True)
    solved = np.linalg.solve(system.A_mat, np.hstack([system.psi_inc, system.pw_inc]))
    res = compute_sphere_foldy_lax(*args, cell_average=True)
    assert system.A_mat.shape == (9 * res.n_cells, 9 * res.n_cells)
    assert np.max(np.abs(solved[:, :9] - res.psi_exc)) <= 1e-12 * np.max(np.abs(res.psi_exc))
    assert np.max(np.abs(solved[:, 9:] - res.psi_pw)) <= 1e-12 * np.max(np.abs(res.psi_pw))
