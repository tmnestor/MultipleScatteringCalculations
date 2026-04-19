"""
test_tmatrix_assembly.py
Tests for the 27×27 T-matrix assembly from per-irrep blocks.
"""

import numpy as np

from cubic_scattering import (
    MaterialContrast,
    ReferenceMedium,
    compute_cube_tmatrix_galerkin,
    voigt_tmatrix_6x6,
)
from cubic_scattering.tmatrix_assembly import (
    _build_c2_matrix,
    _build_c3_matrix,
    _build_usym_27,
    assemble_tmatrix_27,
    tmatrix_27_to_voigt_6x6,
)

# Standard test parameters
REF = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
CONTRAST = MaterialContrast(Dlambda=2e9, Dmu=1e9, Drho=100.0)
WEAK_CONTRAST = MaterialContrast(
    Dlambda=REF.mu * 1e-4, Dmu=REF.mu * 1e-4, Drho=REF.rho * 1e-4
)


def _galerkin_at_ka(ka: float, a: float = 10.0):
    omega = ka * REF.beta / a
    return compute_cube_tmatrix_galerkin(omega, a, REF, CONTRAST)


def test_usym_orthogonality():
    """Usym.T @ Usym = I_27."""
    Usym = _build_usym_27()
    UtU = Usym.T @ Usym
    np.testing.assert_allclose(UtU, np.eye(27), atol=1e-14)


def test_c2_involution():
    """R_C2^2 = I for each axis."""
    for axis in range(3):
        R = _build_c2_matrix(axis)
        np.testing.assert_allclose(R @ R, np.eye(27), atol=1e-15)


def test_c3_period():
    """R_C3^3 = I."""
    R = _build_c3_matrix()
    np.testing.assert_allclose(R @ R @ R, np.eye(27), atol=1e-15)


def test_c3_is_permutation():
    """C3 matrix should be a permutation (each row/col has exactly one 1)."""
    R = _build_c3_matrix()
    assert np.allclose(np.sort(np.abs(R), axis=1)[:, -1], 1.0)
    assert np.allclose(np.sum(np.abs(R), axis=1), 1.0)


def test_t27_strain_block_matches_voigt():
    """T27[3:9, 3:9] == voigt_tmatrix_6x6(T1c, T2c, T3c)."""
    g = _galerkin_at_ka(0.05)
    T27 = assemble_tmatrix_27(g)
    T_strain = tmatrix_27_to_voigt_6x6(T27)
    T_voigt = voigt_tmatrix_6x6(g.T1c, g.T2c, g.T3c)
    np.testing.assert_allclose(T_strain, T_voigt, rtol=1e-10)


def test_t27_eigenvalues_match_irrep():
    """Eigenvalues of T27 reproduce all per-irrep values with correct multiplicities."""
    g = _galerkin_at_ka(0.05)
    T27 = assemble_tmatrix_27(g)

    eigs_T27 = np.sort(np.real(np.linalg.eigvals(T27)))

    # Build expected eigenvalues with multiplicities
    # Note: eigenvalues may be complex (conjugate pairs from non-symmetric
    # Tblock, or from smooth body bilinear). Compare real parts since the
    # assembled T27 eigenvalues are computed via np.real().
    expected = []
    for ev in g.T1u_eigenvalues:
        expected.extend([np.real(ev)] * 3)  # d=3
    for ev in g.T2u_eigenvalues:
        expected.extend([np.real(ev)] * 3)  # d=3
    expected.append(np.real(g.sigma_A2u))  # d=1, m=1
    expected.extend([np.real(g.sigma_Eu)] * 2)  # d=2
    expected.append(np.real(g.sigma_A1g))  # d=1
    expected.extend([np.real(g.sigma_Eg)] * 2)  # d=2
    expected.extend([np.real(g.sigma_T2g)] * 3)  # d=3
    expected = np.sort(expected)

    np.testing.assert_allclose(eigs_T27, expected, rtol=1e-10)


def test_t27_born_limit():
    """Weak contrast → T27 small, nearly proportional to contrast."""
    omega = 0.05 * REF.beta / 10.0
    g = compute_cube_tmatrix_galerkin(omega, 10.0, REF, WEAK_CONTRAST)
    T27 = assemble_tmatrix_27(g)

    # All eigenvalues should be small
    eigs = np.abs(np.linalg.eigvals(T27))
    assert np.max(eigs) < 0.01, (
        f"Max eigenvalue {np.max(eigs)} too large for weak contrast"
    )


def test_t27_shape():
    """T27 is 27×27."""
    g = _galerkin_at_ka(0.05)
    T27 = assemble_tmatrix_27(g)
    assert T27.shape == (27, 27)


def test_usym_column_count():
    """Usym has exactly 27 columns summing to correct irrep dimensions."""
    Usym = _build_usym_27()
    assert Usym.shape == (27, 27)
    # 12 (T1u) + 6 (T2u) + 1 (A2u) + 2 (Eu) + 1 (A1g) + 2 (Eg) + 3 (T2g) = 27
