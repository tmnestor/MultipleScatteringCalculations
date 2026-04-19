"""
tmatrix_assembly.py
Assemble the full T-matrix from per-irrep blocks (27×27 or 57×57).

The 27-component Galerkin basis (3 displacement + 6 strain + 18 quadratic)
decomposes under O_h into 7 irreducible representations:
  Ungerade: 4×T1u (12D) + 2×T2u (6D) + A2u (1D) + Eu (2D)
  Gerade:   A1g (1D) + Eg (2D) + T2g (3D)

The 57-component basis adds 30 cubic modes (10 monomials × 3 directions),
all gerade, extending to 9 irreps:
  Ungerade: 4×T1u (12D) + 2×T2u (6D) + A2u (1D) + Eu (2D)  [unchanged]
  Gerade:   3×A1g (3D) + A2g (1D) + 4×Eg (8D) + 3×T1g (9D) + 5×T2g (15D)

Basis ordering (0-indexed):
  0-2:   constant displacement: e_1, e_2, e_3
  3-5:   axial strain: r_1 e_1, r_2 e_2, r_3 e_3
  6-8:   shear strain: (r_3 e_2 + r_2 e_3)/2, (r_3 e_1 + r_1 e_3)/2, (r_2 e_1 + r_1 e_2)/2
  9-14:  quadratic × e_1: r1², r2², r3², r2r3, r1r3, r1r2
  15-20: quadratic × e_2: r1², r2², r3², r2r3, r1r3, r1r2
  21-26: quadratic × e_3: r1², r2², r3², r2r3, r1r3, r1r2
  27-36: cubic × e_1: r1³, r2³, r3³, r1²r2, r1²r3, r2²r1, r2²r3, r3²r1, r3²r2, r1r2r3
  37-46: cubic × e_2: (same 10 monomials)
  47-56: cubic × e_3: (same 10 monomials)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import block_diag

if TYPE_CHECKING:
    from .effective_contrasts import GalerkinTMatrixResult, GalerkinTMatrixResult57

# ================================================================
# C3 permutation matrix (27×27)
# ================================================================
# The C3 rotation cycles coordinates: r1→r2→r3→r1 and e1→e2→e3→e1.
# This is a pure permutation of the 27 basis functions (no sign flips).


def _c3_permutation_index(i: int) -> int:
    """Map basis index i under C3: (1→2→3→1) cycle on both shape and direction.

    For displacement (0-2): e_k → e_{k+1 mod 3}
    For strain (3-5): r_k e_k → r_{k+1} e_{k+1}
    For shear (6-8): same cyclic on the pair indices
    For quadratic (9-26): cycle both monomial indices and direction
    """
    if i < 3:
        # displacement: e_1→e_2→e_3→e_1
        return (i + 1) % 3
    if i < 6:
        # axial strain: r_k e_k cycles
        return 3 + (i - 3 + 1) % 3
    if i < 9:
        # shear strain: (r_3 e_2+r_2 e_3)/2 → ... cycles
        # index 6 = {23} pair, 7 = {13} pair, 8 = {12} pair
        # Under C3: {23}→{31}→{12}→{23}, i.e. 6→7→8→6
        return 6 + (i - 6 + 1) % 3
    # Quadratic block: indices 9-26
    # Layout: 6 monomials × 3 directions
    # Direction block: dir = (i-9) // 6, monomial = (i-9) % 6
    rel = i - 9
    direction = rel // 6  # 0,1,2 → e_1,e_2,e_3
    mono_idx = rel % 6  # 0:r1², 1:r2², 2:r3², 3:r2r3, 4:r1r3, 5:r1r2

    # New direction: cycle
    new_dir = (direction + 1) % 3

    # Monomial permutation under r1→r2, r2→r3, r3→r1:
    # r1² → r2², r2² → r3², r3² → r1²
    # r2r3 → r3r1, r1r3 → r2r1=r1r2, r1r2 → r2r3
    mono_map = {0: 1, 1: 2, 2: 0, 3: 4, 4: 5, 5: 3}
    new_mono = mono_map[mono_idx]

    return 9 + new_dir * 6 + new_mono


def _build_c3_matrix() -> np.ndarray:
    """Build the 27×27 C3 permutation matrix."""
    R = np.zeros((27, 27))
    for i in range(27):
        j = _c3_permutation_index(i)
        R[j, i] = 1.0  # column i maps to column j
    return R


# ================================================================
# C2 rotation matrices (27×27 diagonal)
# ================================================================
# C2 about axis k negates the two coordinates ≠ k.
# Each basis function picks up a sign = (parity of r-polynomial) × (parity of e-direction).


def _basis_parity(i: int, axis: int) -> int:
    """Sign of basis function i under C2 rotation about `axis` (0,1,2).

    C2(axis) negates the two coordinates ≠ axis.
    Sign = product of (sign of each coordinate power under negation) × (sign of e-direction).
    """
    # Displacement (0-2): e_k has parity +1 if k==axis, -1 otherwise
    if i < 3:
        return 1 if i == axis else -1

    # Axial strain (3-5): r_k e_k → sign(r_k) × sign(e_k)
    # r_k under C2(axis): negated if k ≠ axis, so sign(r_k) = +1 if k==axis, -1 otherwise
    # e_k: same. Product: always +1 (both flip or both don't).
    if i < 6:
        return 1

    # Shear strain (6-8):
    # 6: (r_3 e_2 + r_2 e_3)/2 — pair {2,3}, direction components mixed
    # 7: (r_3 e_1 + r_1 e_3)/2 — pair {0,2}
    # 8: (r_2 e_1 + r_1 e_2)/2 — pair {0,1}
    # For shear basis j (j=6,7,8), the pair of axes involved:
    if i < 9:
        pairs = {6: (1, 2), 7: (0, 2), 8: (0, 1)}
        a, b = pairs[i]
        # Under C2(axis): r_a e_b + r_b e_a
        # sign(r_a) = +1 if a==axis else -1; sign(e_b) = +1 if b==axis else -1
        # Both terms have same total sign:
        sign_r_a = 1 if a == axis else -1
        sign_e_b = 1 if b == axis else -1
        return sign_r_a * sign_e_b

    # Quadratic (9-26): q_m · e_k
    rel = i - 9
    direction = rel // 6  # e_k direction
    mono_idx = rel % 6  # monomial index

    # Direction parity
    dir_sign = 1 if direction == axis else -1

    # Monomial parity under C2(axis):
    # Monomials: 0:r1², 1:r2², 2:r3², 3:r2r3, 4:r1r3, 5:r1r2
    # r_j² is always even under any C2.
    # r_j r_k: even if both j,k == axis or both ≠ axis; odd if exactly one == axis.
    mono_axes = {0: (0, 0), 1: (1, 1), 2: (2, 2), 3: (1, 2), 4: (0, 2), 5: (0, 1)}
    j, k = mono_axes[mono_idx]
    # Under C2(axis), r_j → (+1 if j==axis else -1) × r_j
    sign_j = 1 if j == axis else -1
    sign_k = 1 if k == axis else -1
    mono_sign = sign_j * sign_k

    return mono_sign * dir_sign


def _build_c2_matrix(axis: int) -> np.ndarray:
    """Build diagonal 27×27 C2 matrix for rotation about given axis."""
    signs = np.array([_basis_parity(i, axis) for i in range(27)], dtype=float)
    return np.diag(signs)


# ================================================================
# Usym construction from stored C3-invariant columns
# ================================================================
# The UsymData from CubeT27StiffnessLS_Results.wl stores the C3-invariant
# seed columns for each irrep. These are 1-indexed in Mathematica;
# we convert to 0-indexed here.
#
# For d=1 irreps: the stored column IS the Usym column
# For d=2 irreps: generate partner via C3 rotation
# For d=3 irreps: extract 3 Cartesian copies via C2 projections


def _stored_columns() -> dict[str, np.ndarray]:
    """Return the C3-invariant seed columns from CubeT27StiffnessLS_Results.wl.

    Each value is shape (27, m) where m is the multiplicity of the irrep.
    """
    cols = {}

    # A1g: rows 3,4,5 = [1,1,1] (0-indexed)
    v = np.zeros(27)
    v[3] = v[4] = v[5] = 1.0
    cols["A1g"] = v.reshape(27, 1)

    # Eg: rows 3,4,5 = [1, -1/2, -1/2]
    v = np.zeros(27)
    v[3] = 1.0
    v[4] = v[5] = -0.5
    cols["Eg"] = v.reshape(27, 1)

    # T2g: rows 6,7,8 = [1,1,1]
    v = np.zeros(27)
    v[6] = v[7] = v[8] = 1.0
    cols["T2g"] = v.reshape(27, 1)

    # A2u: rows 12,19,26 = [1,1,1]
    v = np.zeros(27)
    v[12] = v[19] = v[26] = 1.0
    cols["A2u"] = v.reshape(27, 1)

    # Eu: rows 19,26 = [1,-1]
    v = np.zeros(27)
    v[19] = 1.0
    v[26] = -1.0
    cols["Eu"] = v.reshape(27, 1)

    # T1u: 4 columns (0-indexed rows from Mathematica data)
    T1u = np.zeros((27, 4))
    # Col 0: rows 0,1,2 = [1,1,1]
    T1u[0, 0] = T1u[1, 0] = T1u[2, 0] = 1.0
    # Col 1: rows 9,16,23 = [1,1,1] (S-same: r_k² e_k direction)
    T1u[9, 1] = T1u[16, 1] = T1u[23, 1] = 1.0
    # Col 2: rows 10,11,15,17,21,22 = [1,1,1,1,1,1] (S-cross: r_j² e_k, j≠k)
    for r in [10, 11, 15, 17, 21, 22]:
        T1u[r, 2] = 1.0
    # Col 3: rows 13,14,18,20,24,25 = [1,1,1,1,1,1] (X-type: r_jr_k e_l)
    for r in [13, 14, 18, 20, 24, 25]:
        T1u[r, 3] = 1.0
    cols["T1u"] = T1u

    # T2u: 2 columns
    T2u = np.zeros((27, 2))
    # Col 0: rows 10,11,15,17,21,22 = [+1,-1,-1,+1,+1,-1]
    T2u[10, 0] = 1.0
    T2u[11, 0] = -1.0
    T2u[15, 0] = -1.0
    T2u[17, 0] = 1.0
    T2u[21, 0] = 1.0
    T2u[22, 0] = -1.0
    # Col 1: rows 13,14,18,20,24,25 = [+1,-1,-1,+1,+1,-1]
    T2u[13, 1] = 1.0
    T2u[14, 1] = -1.0
    T2u[18, 1] = -1.0
    T2u[20, 1] = 1.0
    T2u[24, 1] = 1.0
    T2u[25, 1] = -1.0
    cols["T2u"] = T2u

    return cols


def _expand_d3_columns(seed_col: np.ndarray, R_C3: np.ndarray) -> np.ndarray:
    """Expand a C3-invariant seed column into 3 Cartesian copies for d=3 irreps.

    For a d=3 irrep (T-type), the C3-invariant column v satisfies R_C3 @ v = v.
    The three Cartesian copies are obtained by C2 projection:
      v_k = (v + R_C2k @ v) / 2  for each axis k

    But since v is C3-invariant, we can equivalently use the C3 orbit:
      v_1 = v (projected to axis-1 content)
    Actually the cleanest approach: use C2 projections to separate axes.

    Returns (27, 3) with orthogonal columns.
    """
    R_C2 = [_build_c2_matrix(k) for k in range(3)]

    copies = np.zeros((27, 3))
    for k in range(3):
        # C2(k) has eigenvalue +1 for basis functions "aligned with axis k"
        # and -1 for functions with odd parity in the other two axes.
        # The projection (I + C2_k)/2 picks out the axis-k content.
        copies[:, k] = 0.5 * (seed_col + R_C2[k] @ seed_col)

    # Verify orthogonality and normalize
    for k in range(3):
        nrm = np.linalg.norm(copies[:, k])
        if nrm > 1e-14:
            copies[:, k] /= nrm
        else:
            msg = f"Zero norm for C2-projected copy axis={k}"
            raise ValueError(msg)

    return copies


def _expand_d2_columns(seed_col: np.ndarray, R_C3: np.ndarray) -> np.ndarray:
    """Expand a C3-invariant seed column into 2 E-type partners.

    For d=2 irreps, the E representation under C3 has matrix
    D_E(C3) = [[-1/2, -sqrt(3)/2], [sqrt(3)/2, -1/2]].

    The seed v is C3-invariant (fixed by C3 projector = (I+C3+C3²)/3).
    The partner w satisfies: R_C3 @ v = -v/2 + sqrt(3)/2 * w.
    => w = (2/sqrt(3)) (R_C3 @ v + v/2)

    Returns (27, 2) with orthonormal columns.
    """
    v = seed_col.copy()
    w = (2.0 / np.sqrt(3.0)) * (R_C3 @ v + 0.5 * v)

    # Normalize
    nv = np.linalg.norm(v)
    nw = np.linalg.norm(w)
    result = np.zeros((27, 2))
    result[:, 0] = v / nv
    result[:, 1] = w / nw

    return result


def _build_usym_27() -> np.ndarray:
    """Build the 27×27 orthogonal change-of-basis matrix Usym.

    Column ordering matches the T_irrep block ordering:
      T1u (12 cols) | T2u (6 cols) | A2u (1) | Eu (2) | A1g (1) | Eg (2) | T2g (3)

    For d=3 irreps with m multiplicity copies, each copy gets 3 Cartesian
    partners, giving d×m columns total. The Cartesian partners cycle as
    (z, x, y) corresponding to axes (0, 1, 2).
    """
    stored = _stored_columns()
    R_C3 = _build_c3_matrix()

    all_cols = []

    # T1u: d=3, m=4 → 12 columns
    # For each of 4 seed columns, generate 3 Cartesian copies
    for col_idx in range(4):
        seed = stored["T1u"][:, col_idx]
        copies = _expand_d3_columns(seed, R_C3)
        for k in range(3):
            all_cols.append(copies[:, k])

    # T2u: d=3, m=2 → 6 columns
    for col_idx in range(2):
        seed = stored["T2u"][:, col_idx]
        copies = _expand_d3_columns(seed, R_C3)
        for k in range(3):
            all_cols.append(copies[:, k])

    # A2u: d=1, m=1 → 1 column
    v = stored["A2u"][:, 0]
    all_cols.append(v / np.linalg.norm(v))

    # Eu: d=2, m=1 → 2 columns
    seed = stored["Eu"][:, 0]
    eu_cols = _expand_d2_columns(seed, R_C3)
    for k in range(2):
        all_cols.append(eu_cols[:, k])

    # A1g: d=1, m=1 → 1 column
    v = stored["A1g"][:, 0]
    all_cols.append(v / np.linalg.norm(v))

    # Eg: d=2, m=1 → 2 columns
    seed = stored["Eg"][:, 0]
    eg_cols = _expand_d2_columns(seed, R_C3)
    for k in range(2):
        all_cols.append(eg_cols[:, k])

    # T2g: d=3, m=1 → 3 columns
    seed = stored["T2g"][:, 0]
    t2g_copies = _expand_d3_columns(seed, R_C3)
    for k in range(3):
        all_cols.append(t2g_copies[:, k])

    Usym = np.column_stack(all_cols)
    assert Usym.shape == (27, 27), f"Usym shape {Usym.shape}, expected (27, 27)"

    return Usym


# ================================================================
# T27 assembly
# ================================================================


def assemble_tmatrix_27(galerkin: GalerkinTMatrixResult) -> np.ndarray:
    """Assemble the full 27×27 T-matrix from per-irrep blocks.

    The T-matrix in the irrep basis is block-diagonal:
      T_irrep = diag(T1u⊗I3, T2u⊗I3, A2u, Eu⊗I2, A1g, Eg⊗I2, T2g⊗I3)

    where ⊗I_d means Kronecker product with d×d identity (d copies of same block).

    The physical T27 in the original basis is:
      T27 = Usym @ T_irrep @ Usym.T
    """
    Usym = _build_usym_27()

    # Build block-diagonal T_irrep
    T_irrep = block_diag(
        np.kron(np.eye(3), galerkin.T1u_block),  # 12×12
        np.kron(np.eye(3), galerkin.T2u_block),  # 6×6
        np.array([[galerkin.sigma_A2u]]),  # 1×1
        galerkin.sigma_Eu * np.eye(2),  # 2×2
        np.array([[galerkin.sigma_A1g]]),  # 1×1
        galerkin.sigma_Eg * np.eye(2),  # 2×2
        galerkin.sigma_T2g * np.eye(3),  # 3×3
    )

    return Usym @ T_irrep @ Usym.T


# ================================================================
# Validation: extract strain block
# ================================================================


def tmatrix_27_to_voigt_6x6(T27: np.ndarray) -> np.ndarray:
    """Extract the strain-strain block (rows 3:9, cols 3:9) of T27.

    This should match voigt_tmatrix_6x6(T1c, T2c, T3c) from voigt_tmatrix.py.
    """
    return T27[3:9, 3:9]


# ================================================================
# 57D extension: C3 permutation for cubic modes
# ================================================================

# Cubic monomial permutation under C3 (r1→r2, r2→r3, r3→r1):
# Monomials (0-indexed): 0:r1³, 1:r2³, 2:r3³, 3:r1²r2, 4:r1²r3,
#   5:r2²r1, 6:r2²r3, 7:r3²r1, 8:r3²r2, 9:r1r2r3
# Under C3:
#   r1³→r2³, r2³→r3³, r3³→r1³
#   r1²r2→r2²r3, r1²r3→r2²r1, r2²r1→r3²r2, r2²r3→r3²r1,
#   r3²r1→r1²r2, r3²r2→r1²r3
#   r1r2r3→r1r2r3 (invariant)
_CUBIC_MONO_C3 = {0: 1, 1: 2, 2: 0, 3: 6, 4: 5, 5: 8, 6: 7, 7: 3, 8: 4, 9: 9}

# Cubic monomial axes for C2 parity: (p, q, r) exponents for r1^p r2^q r3^r
_CUBIC_MONO_AXES = {
    0: (3, 0, 0),  # r1³
    1: (0, 3, 0),  # r2³
    2: (0, 0, 3),  # r3³
    3: (2, 1, 0),  # r1²r2
    4: (2, 0, 1),  # r1²r3
    5: (1, 2, 0),  # r2²r1
    6: (0, 2, 1),  # r2²r3
    7: (1, 0, 2),  # r3²r1
    8: (0, 1, 2),  # r3²r2
    9: (1, 1, 1),  # r1r2r3
}


def _c3_permutation_index_57(i: int) -> int:
    """Map basis index i (0-56) under C3: (1→2→3→1) cycle."""
    if i < 27:
        return _c3_permutation_index(i)
    # Cubic block: indices 27-56
    rel = i - 27
    direction = rel // 10  # 0,1,2 → e_1,e_2,e_3
    mono_idx = rel % 10  # 0-9
    new_dir = (direction + 1) % 3
    new_mono = _CUBIC_MONO_C3[mono_idx]
    return 27 + new_dir * 10 + new_mono


def _basis_parity_57(i: int, axis: int) -> int:
    """Sign of basis function i (0-56) under C2 rotation about `axis`."""
    if i < 27:
        return _basis_parity(i, axis)
    # Cubic (27-56): c_m(r) · e_k
    rel = i - 27
    direction = rel // 10
    mono_idx = rel % 10
    # Direction parity
    dir_sign = 1 if direction == axis else -1
    # Monomial parity under C2(axis): r_j → (-1 if j≠axis) r_j
    p, q, r = _CUBIC_MONO_AXES[mono_idx]
    exps = [p, q, r]
    # Count how many exponents correspond to axes ≠ rotation axis
    odd_sum = sum(exps[j] for j in range(3) if j != axis)
    mono_sign = 1 if odd_sum % 2 == 0 else -1
    return mono_sign * dir_sign


def _build_c3_matrix_57() -> np.ndarray:
    """Build the 57×57 C3 permutation matrix."""
    R = np.zeros((57, 57))
    for i in range(57):
        j = _c3_permutation_index_57(i)
        R[j, i] = 1.0
    return R


def _build_c2_matrix_57(axis: int) -> np.ndarray:
    """Build diagonal 57×57 C2 matrix for rotation about given axis."""
    signs = np.array([_basis_parity_57(i, axis) for i in range(57)], dtype=float)
    return np.diag(signs)


# ================================================================
# O_h character projection for Usym₅₇
# ================================================================

# Monomial exponent tables for basis lookup (must match incident_field layout)
_QUAD_MONO_EXPONENTS = [
    (2, 0, 0),
    (0, 2, 0),
    (0, 0, 2),
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 0),
]

_CUBIC_MONO_EXPONENTS = [
    (3, 0, 0),
    (0, 3, 0),
    (0, 0, 3),
    (2, 1, 0),
    (2, 0, 1),
    (1, 2, 0),
    (0, 2, 1),
    (1, 0, 2),
    (0, 1, 2),
    (1, 1, 1),
]

# O_h character table: 10 conjugacy classes × 10 irreps
# Class order: E, 8C₃, 3C₂, 6C₄, 6C₂', i, 8S₆, 3σ_h, 6S₄, 6σ_d
_OH_CLASS_ORDER = [
    "E",
    "8C3",
    "3C2",
    "6C4",
    "6C2p",
    "i",
    "8S6",
    "3sh",
    "6S4",
    "6sd",
]

_OH_CHAR_TABLE = {
    "A1g": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    "A2g": [1, 1, 1, -1, -1, 1, 1, 1, -1, -1],
    "Eg": [2, -1, 2, 0, 0, 2, -1, 2, 0, 0],
    "T1g": [3, 0, -1, 1, -1, 3, 0, -1, 1, -1],
    "T2g": [3, 0, -1, -1, 1, 3, 0, -1, -1, 1],
    "A1u": [1, 1, 1, 1, 1, -1, -1, -1, -1, -1],
    "A2u": [1, 1, 1, -1, -1, -1, -1, -1, 1, 1],
    "Eu": [2, -1, 2, 0, 0, -2, 1, -2, 0, 0],
    "T1u": [3, 0, -1, 1, -1, -3, 0, 1, -1, 1],
    "T2u": [3, 0, -1, -1, 1, -3, 0, 1, 1, -1],
}

_OH_IRREP_DIMS = {
    "A1g": 1,
    "A2g": 1,
    "Eg": 2,
    "T1g": 3,
    "T2g": 3,
    "A1u": 1,
    "A2u": 1,
    "Eu": 2,
    "T1u": 3,
    "T2u": 3,
}


def _classify_oh_element(sigma: list[int], signs: list[int]) -> str:
    """Classify O_h element into conjugacy class via (tr, det, nfixed) invariants."""
    M = np.zeros((3, 3))
    for j in range(3):
        M[sigma[j], j] = signs[j]
    tr = int(round(np.trace(M)))
    det = int(round(np.linalg.det(M)))
    nfixed = sum(1 for j in range(3) if sigma[j] == j and signs[j] == 1)

    if tr == 3:
        return "E"
    if tr == 0 and det == 1:
        return "8C3"
    if tr == -1 and det == 1 and nfixed == 1:
        return "3C2"
    if tr == 1 and det == 1:
        return "6C4"
    if tr == -1 and det == 1 and nfixed == 0:
        return "6C2p"
    if tr == -3:
        return "i"
    if tr == 0 and det == -1:
        return "8S6"
    if tr == 1 and det == -1 and nfixed == 2:
        return "3sh"
    if tr == -1 and det == -1:
        return "6S4"
    if tr == 1 and det == -1 and nfixed == 1:
        return "6sd"
    msg = f"Unknown O_h element: tr={tr}, det={det}, nfixed={nfixed}"
    raise ValueError(msg)


def _build_oh_rep_57() -> tuple[list[np.ndarray], list[str]]:
    """Build all 48 O_h representation matrices (57×57) and class labels.

    Each O_h element is a 3×3 signed permutation M = (sigma, signs) with
    M_{sigma(j),j} = signs[j].  Action on vector polynomial phi(r) = c_m(r) e_k:
      (g phi)(r) = M phi(M^T r)
    Coordinate sub: r_j -> s_j r_{sigma(j)}.  Direction: e_k -> s_k e_{sigma(k)}.
    Each 57×57 rep matrix is itself a signed permutation (sparse, exact).
    """
    from itertools import permutations
    from itertools import product as iproduct

    n = 57

    # (exponent_tuple, direction) -> basis_index lookup
    exp_dir_to_idx: dict[tuple[tuple[int, ...], int], int] = {}
    idx_to_exp_dir: dict[int, tuple[tuple[int, ...], int]] = {}

    # Displacement (0-2)
    for k in range(3):
        key = ((0, 0, 0), k)
        exp_dir_to_idx[key] = k
        idx_to_exp_dir[k] = key

    # Axial strain (3-5)
    for k in range(3):
        exp_list = [0, 0, 0]
        exp_list[k] = 1
        exp_t: tuple[int, int, int] = (exp_list[0], exp_list[1], exp_list[2])
        key = (exp_t, k)
        exp_dir_to_idx[key] = 3 + k
        idx_to_exp_dir[3 + k] = key

    # Quadratic (9-26)
    for d in range(3):
        for mi, exp in enumerate(_QUAD_MONO_EXPONENTS):
            idx = 9 + d * 6 + mi
            key = (exp, d)
            exp_dir_to_idx[key] = idx
            idx_to_exp_dir[idx] = key

    # Cubic (27-56)
    for d in range(3):
        for mi, exp in enumerate(_CUBIC_MONO_EXPONENTS):
            idx = 27 + d * 10 + mi
            key = (exp, d)
            exp_dir_to_idx[key] = idx
            idx_to_exp_dir[idx] = key

    # Shear (6-8): pair -> index
    shear_pairs = [(1, 2), (0, 2), (0, 1)]
    pair_to_shear = {p: 6 + i for i, p in enumerate(shear_pairs)}

    matrices: list[np.ndarray] = []
    labels: list[str] = []

    for perm in permutations(range(3)):
        for stuple in iproduct([1, -1], repeat=3):
            sigma = list(perm)
            signs = list(stuple)
            sigma_inv = [0, 0, 0]
            for j in range(3):
                sigma_inv[sigma[j]] = j

            R = np.zeros((n, n))

            for i in range(n):
                if 6 <= i <= 8:
                    # Shear: (r_a e_b + r_b e_a)/2 -> s_a s_b (r_{σa} e_{σb} + ...)/2
                    a, b = shear_pairs[i - 6]
                    sa, sb = sorted([sigma[a], sigma[b]])
                    sign = signs[a] * signs[b]
                    R[pair_to_shear[(sa, sb)], i] = sign
                else:
                    exp_raw, dk = idx_to_exp_dir[i]
                    new_dir = sigma[dk]
                    new_exp = (
                        exp_raw[sigma_inv[0]],
                        exp_raw[sigma_inv[1]],
                        exp_raw[sigma_inv[2]],
                    )
                    sign = signs[dk]
                    for j in range(3):
                        if exp_raw[j] % 2 == 1:
                            sign *= signs[j]
                    R[exp_dir_to_idx[(new_exp, new_dir)], i] = sign

            matrices.append(R)
            labels.append(_classify_oh_element(sigma, signs))

    return matrices, labels


# ================================================================
# Usym₅₇ construction
# ================================================================


def _expand_d3_columns_57(seed_col: np.ndarray, R_C3: np.ndarray) -> np.ndarray:
    """Expand a C3-invariant seed column (57D) into 3 Cartesian copies for d=3 irreps."""
    R_C2 = [_build_c2_matrix_57(k) for k in range(3)]
    copies = np.zeros((57, 3))
    for k in range(3):
        copies[:, k] = 0.5 * (seed_col + R_C2[k] @ seed_col)
    for k in range(3):
        nrm = np.linalg.norm(copies[:, k])
        if nrm > 1e-14:
            copies[:, k] /= nrm
        else:
            msg = f"Zero norm for C2-projected copy axis={k}"
            raise ValueError(msg)
    return copies


def _expand_d2_columns_57(seed_col: np.ndarray, R_C3: np.ndarray) -> np.ndarray:
    """Expand a C3-invariant seed column (57D) into 2 E-type partners."""
    v = seed_col.copy()
    w = (2.0 / np.sqrt(3.0)) * (R_C3 @ v + 0.5 * v)
    nv = np.linalg.norm(v)
    nw = np.linalg.norm(w)
    result = np.zeros((57, 2))
    result[:, 0] = v / nv
    result[:, 1] = w / nw
    return result


def _stored_columns_57() -> dict[str, np.ndarray]:
    """Compute seed columns for T₅₇ irreps via O_h character projection.

    Ungerade: reuses validated T₂₇ seeds (embedded in 57D).
    Gerade: character projectors + fused reduction on full 57D O_h rep,
    mirroring CubeT27Assemble.wl Section 9b.

    For gerade irreps with a strain component (A1g, Eg, T2g),
    the strain basis function is placed first in the seed ordering
    so that the [0,0] block entry gives the physical T1c, T2c, T3c.

    Returns dict keyed by irrep name with shape (57, m) arrays.
    """
    n = 57
    cols: dict[str, np.ndarray] = {}

    # ── Ungerade sector: identical to T₂₇ ──
    stored_27 = _stored_columns()
    for name in ["A2u", "Eu", "T1u", "T2u"]:
        v27 = stored_27[name]
        v57 = np.zeros((n, v27.shape[1]))
        v57[:27, :] = v27
        cols[name] = v57

    # ── Gerade sector: O_h character projection ──
    rep_matrices, class_labels = _build_oh_rep_57()

    class_to_idx = {name: i for i, name in enumerate(_OH_CLASS_ORDER)}

    def char_projector(irrep: str) -> np.ndarray:
        chars = _OH_CHAR_TABLE[irrep]
        d = _OH_IRREP_DIMS[irrep]
        P = np.zeros((n, n))
        for k in range(48):
            chi = chars[class_to_idx[class_labels[k]]]
            P += chi * rep_matrices[k]
        return P * (d / 48.0)

    # C3: must match _build_c3_matrix_57() for expand compatibility
    R_C3_ref = _build_c3_matrix_57()
    c3_idx = next(k for k in range(48) if np.allclose(rep_matrices[k], R_C3_ref))

    # C2': any face-diagonal C2 (class "6C2p")
    c2p_idx = next(k for k in range(48) if class_labels[k] == "6C2p")

    I_n = np.eye(n)
    R_C3 = rep_matrices[c3_idx]
    R_C2p = rep_matrices[c2p_idx]
    P_fix_C3 = (I_n + R_C3 + R_C3 @ R_C3) / 3.0
    P_fix_C2p = (I_n + R_C2p) / 2.0

    # Strain reference vectors for ordering gerade seeds
    v_strain_refs: dict[str, np.ndarray] = {}
    v = np.zeros(n)
    v[3] = v[4] = v[5] = 1.0
    v_strain_refs["A1g"] = v  # volumetric
    v = np.zeros(n)
    v[3] = 1.0
    v[4] = v[5] = -0.5
    v_strain_refs["Eg"] = v  # deviatoric
    v = np.zeros(n)
    v[6] = v[7] = v[8] = 1.0
    v_strain_refs["T2g"] = v  # shear

    expected_mults = {"A1g": 3, "A2g": 1, "Eg": 4, "T1g": 3, "T2g": 5}

    for irrep in ["A1g", "A2g", "Eg", "T1g", "T2g"]:
        d = _OH_IRREP_DIMS[irrep]
        m = expected_mults[irrep]
        P = char_projector(irrep)

        # Verify multiplicity via trace
        rank = int(round(np.trace(P)))
        assert rank == d * m, f"{irrep}: Tr(P)={rank}, expected d*m={d * m}"

        # Fused projector: reduces d-dimensional irrep copies to 1D each
        if d == 1:
            P_fused = P
        elif d == 2:
            P_fused = P @ P_fix_C2p
        else:
            P_fused = P @ P_fix_C3

        strain_ref = v_strain_refs.get(irrep)
        if strain_ref is not None:
            # Strain-first ordering: project strain through fused projector
            v0 = P_fused @ strain_ref
            v0_norm = np.linalg.norm(v0)
            assert v0_norm > 1e-10, f"{irrep}: strain projection vanishes"
            v0 = v0 / v0_norm

            if m == 1:
                cols[irrep] = v0.reshape(n, 1)
            else:
                seed_mat = np.zeros((n, m))
                seed_mat[:, 0] = v0
                # Deflate: remove strain from fused projector image
                P_def = (I_n - np.outer(v0, v0)) @ P_fused
                U, S, _ = np.linalg.svd(P_def, full_matrices=True)
                for col in range(m - 1):
                    u = U[:, col].copy()
                    # Re-orthogonalize against all previous seeds
                    for prev in range(col + 1):
                        u -= np.dot(u, seed_mat[:, prev]) * seed_mat[:, prev]
                    seed_mat[:, col + 1] = u / np.linalg.norm(u)
                cols[irrep] = seed_mat
        else:
            # Purely cubic irreps (A2g, T1g): SVD directly
            U, S, _ = np.linalg.svd(P_fused, full_matrices=True)
            cols[irrep] = U[:, :m]

    return cols


def _build_usym_57() -> np.ndarray:
    """Build the 57×57 orthogonal change-of-basis matrix Usym₅₇.

    Column ordering:
      T1u (12) | T2u (6) | A2u (1) | Eu (2) | A1g (3) | A2g (1) | Eg (8) | T1g (9) | T2g (15)
    """
    stored = _stored_columns_57()
    R_C3 = _build_c3_matrix_57()

    all_cols: list[np.ndarray] = []

    # T1u: d=3, m=4 → 12 columns
    for col_idx in range(4):
        seed = stored["T1u"][:, col_idx]
        copies = _expand_d3_columns_57(seed, R_C3)
        for k in range(3):
            all_cols.append(copies[:, k])

    # T2u: d=3, m=2 → 6 columns
    for col_idx in range(2):
        seed = stored["T2u"][:, col_idx]
        copies = _expand_d3_columns_57(seed, R_C3)
        for k in range(3):
            all_cols.append(copies[:, k])

    # A2u: d=1, m=1 → 1 column
    v = stored["A2u"][:, 0]
    all_cols.append(v / np.linalg.norm(v))

    # Eu: d=2, m=1 → 2 columns
    seed = stored["Eu"][:, 0]
    eu_cols = _expand_d2_columns_57(seed, R_C3)
    for k in range(2):
        all_cols.append(eu_cols[:, k])

    # A1g: d=1, m=3 → 3 columns
    for col_idx in range(3):
        v = stored["A1g"][:, col_idx]
        all_cols.append(v / np.linalg.norm(v))

    # A2g: d=1, m=1 → 1 column
    v = stored["A2g"][:, 0]
    all_cols.append(v / np.linalg.norm(v))

    # Eg: d=2, m=4 → 8 columns
    for col_idx in range(4):
        seed = stored["Eg"][:, col_idx]
        eg_cols = _expand_d2_columns_57(seed, R_C3)
        for k in range(2):
            all_cols.append(eg_cols[:, k])

    # T1g: d=3, m=3 → 9 columns
    for col_idx in range(3):
        seed = stored["T1g"][:, col_idx]
        copies = _expand_d3_columns_57(seed, R_C3)
        for k in range(3):
            all_cols.append(copies[:, k])

    # T2g: d=3, m=5 → 15 columns
    for col_idx in range(5):
        seed = stored["T2g"][:, col_idx]
        copies = _expand_d3_columns_57(seed, R_C3)
        for k in range(3):
            all_cols.append(copies[:, k])

    Usym = np.column_stack(all_cols)
    assert Usym.shape == (57, 57), f"Usym shape {Usym.shape}, expected (57, 57)"

    return Usym


# ================================================================
# T57 assembly
# ================================================================


def assemble_tmatrix_57(galerkin: GalerkinTMatrixResult57) -> np.ndarray:
    """Assemble the full 57×57 T-matrix from per-irrep blocks.

    The T-matrix in the irrep basis is block-diagonal:
      T_irrep = diag(
        T1u⊗I3,      # 12×12 (unchanged from T₂₇)
        T2u⊗I3,      # 6×6 (unchanged)
        A2u,          # 1×1 (unchanged)
        Eu⊗I2,        # 2×2 (unchanged)
        A1g_block⊗I1, # 3×3 (enlarged: 1 strain + 2 cubic)
        A2g,          # 1×1 (NEW)
        Eg_block⊗I2,  # 8×8 (enlarged: 1 strain + 3 cubic)
        T1g_block⊗I3, # 9×9 (NEW)
        T2g_block⊗I3, # 15×15 (enlarged: 1 strain + 4 cubic)
      )
    """
    Usym = _build_usym_57()

    T_irrep = block_diag(
        np.kron(np.eye(3), galerkin.T1u_block),  # 12×12
        np.kron(np.eye(3), galerkin.T2u_block),  # 6×6
        np.array([[galerkin.sigma_A2u]]),  # 1×1
        galerkin.sigma_Eu * np.eye(2),  # 2×2
        galerkin.A1g_block,  # 3×3
        np.array([[galerkin.sigma_A2g]]),  # 1×1
        np.kron(np.eye(2), galerkin.Eg_block),  # 8×8
        np.kron(np.eye(3), galerkin.T1g_block),  # 9×9
        np.kron(np.eye(3), galerkin.T2g_block),  # 15×15
    )

    return Usym @ T_irrep @ Usym.T


def tmatrix_57_to_voigt_6x6(T57: np.ndarray) -> np.ndarray:
    """Extract the strain-strain block (rows 3:9, cols 3:9) of T57.

    The strain basis functions (indices 3-8) are the same in T₂₇ and T₅₇.
    """
    return T57[3:9, 3:9]
