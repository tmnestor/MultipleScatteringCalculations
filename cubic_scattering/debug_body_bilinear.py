"""Quick check: body bilinear C2 commutation after Mp fix."""

import numpy as np

from cubic_scattering.compute_gerade_blocks import (
    GERADE_INDICES,
    _build_basis_components,
    _extract_irrep_block,
    compute_body_bilinear,
    compute_mass_matrix,
)
from cubic_scattering.tmatrix_assembly import (
    _build_c2_matrix_57,
    _build_c3_matrix_57,
    _build_usym_57,
)

basis = _build_basis_components()
g_idx = np.array(GERADE_INDICES)

M_ger = compute_mass_matrix(basis, GERADE_INDICES)
BA_ger, BB_ger = compute_body_bilinear(basis, GERADE_INDICES)

# C3 check
R_C3 = _build_c3_matrix_57()[np.ix_(g_idx, g_idx)]
comm_BA_C3 = R_C3.T @ BA_ger @ R_C3 - BA_ger
comm_BB_C3 = R_C3.T @ BB_ger @ R_C3 - BB_ger
print(f"BA C3 commutator: {np.max(np.abs(comm_BA_C3)):.2e}")
print(f"BB C3 commutator: {np.max(np.abs(comm_BB_C3)):.2e}")

# C2(0) check
R_C2 = _build_c2_matrix_57(0)[np.ix_(g_idx, g_idx)]
comm_BA_C2 = R_C2.T @ BA_ger @ R_C2 - BA_ger
comm_BB_C2 = R_C2.T @ BB_ger @ R_C2 - BB_ger
print(f"BA C2(0) commutator: {np.max(np.abs(comm_BA_C2)):.2e}")
print(f"BB C2(0) commutator: {np.max(np.abs(comm_BB_C2)):.2e}")

# Usym projection block-diagonality
Usym = _build_usym_57()
Usym_g = Usym[np.ix_(g_idx, np.arange(21, 57))]
M_proj = Usym_g.T @ M_ger @ Usym_g
BA_proj = Usym_g.T @ BA_ger @ Usym_g
BB_proj = Usym_g.T @ BB_ger @ Usym_g

boundaries = [0, 3, 4, 12, 21, 36]
for name, mat in [("M", M_proj), ("BA", BA_proj), ("BB", BB_proj)]:
    max_offblock = 0.0
    for bi in range(len(boundaries) - 1):
        for bj in range(bi + 1, len(boundaries) - 1):
            block = mat[
                boundaries[bi] : boundaries[bi + 1], boundaries[bj] : boundaries[bj + 1]
            ]
            max_offblock = max(max_offblock, np.max(np.abs(block)))
    print(f"Off-block max {name}: {max_offblock:.2e}")

# Cross-validation vs T₂₇
strain_ev_A_ref = 1.755317619825370
for irrep, ref_B in [
    ("A1g", -1.611924358631187),
    ("Eg", 0.043992031944107),
    ("T2g", 0.579676728845940),
]:
    Mb = _extract_irrep_block(M_proj, irrep)
    BAb = _extract_irrep_block(BA_proj, irrep)
    BBb = _extract_irrep_block(BB_proj, irrep)
    ratA = BAb[0, 0] / Mb[0, 0]
    ratB = BBb[0, 0] / Mb[0, 0]
    print(
        f"{irrep}: BA/M[0,0]={ratA:.10f} (ref {strain_ev_A_ref:.10f})  BB/M[0,0]={ratB:.10f} (ref {ref_B:.10f})"
    )
