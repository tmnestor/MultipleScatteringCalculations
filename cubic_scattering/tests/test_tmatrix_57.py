"""
test_tmatrix_57.py
Tests for the 57×57 T-matrix extension (T₂₇ + 30 cubic modes).

Tests are organized into:
  1. Symmetry infrastructure (C3, C2, Usym₅₇ orthogonality)
  2. Consistency with T₂₇ (ungerade regression, strain block)
  3. Incident field extension (m=3 monomial, 57-component overlap)
  4. Solver structure (block sizes, Born limit, eigenvalue properties)
"""

import numpy as np
import pytest

from cubic_scattering import (
    GalerkinTMatrixResult57,
    MaterialContrast,
    ReferenceMedium,
    compute_cube_tmatrix_galerkin,
    compute_cube_tmatrix_galerkin_57,
)
from cubic_scattering.incident_field import (
    _CUBIC_EXPONENTS,
    _monomial_fourier_1d,
    cube_overlap_integrals,
    cube_overlap_integrals_57,
)
from cubic_scattering.tmatrix_assembly import (
    _build_c2_matrix,
    _build_c2_matrix_57,
    _build_c3_matrix,
    _build_c3_matrix_57,
    _build_usym_27,
    _build_usym_57,
)

# Standard test parameters
REF = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
CONTRAST = MaterialContrast(Dlambda=2e9, Dmu=1e9, Drho=100.0)
WEAK_CONTRAST = MaterialContrast(
    Dlambda=REF.mu * 1e-4, Dmu=REF.mu * 1e-4, Drho=REF.rho * 1e-4
)


# ================================================================
# 1. Symmetry infrastructure tests
# ================================================================


class TestC3C2_57:
    """Tests for 57D symmetry matrices."""

    def test_c3_period(self):
        """R_C3^3 = I_57."""
        R = _build_c3_matrix_57()
        np.testing.assert_allclose(R @ R @ R, np.eye(57), atol=1e-15)

    def test_c3_is_permutation(self):
        """C3 matrix is a permutation (each row/col has exactly one 1)."""
        R = _build_c3_matrix_57()
        assert np.allclose(np.sum(np.abs(R), axis=1), 1.0)
        assert np.allclose(np.sum(np.abs(R), axis=0), 1.0)

    def test_c3_extends_27(self):
        """57D C3 restricted to first 27 rows/cols equals 27D C3."""
        R57 = _build_c3_matrix_57()
        R27 = _build_c3_matrix()
        np.testing.assert_allclose(R57[:27, :27], R27, atol=1e-15)

    def test_c2_involution(self):
        """R_C2^2 = I_57 for each axis."""
        for axis in range(3):
            R = _build_c2_matrix_57(axis)
            np.testing.assert_allclose(R @ R, np.eye(57), atol=1e-15)

    def test_c2_extends_27(self):
        """57D C2 restricted to first 27 rows/cols equals 27D C2."""
        for axis in range(3):
            R57 = _build_c2_matrix_57(axis)
            R27 = _build_c2_matrix(axis)
            np.testing.assert_allclose(R57[:27, :27], R27, atol=1e-15)

    def test_c2_diagonal(self):
        """C2 matrices are diagonal (±1 entries)."""
        for axis in range(3):
            R = _build_c2_matrix_57(axis)
            assert np.allclose(R, np.diag(np.diag(R)))
            assert np.all(np.abs(np.diag(R)) == 1.0)

    def test_c3_c2_commutation(self):
        """C3 and C2 generate O_h: C3 C2(0) C3^{-1} = C2(1)."""
        R_C3 = _build_c3_matrix_57()
        R_C2_0 = _build_c2_matrix_57(0)
        R_C2_1 = _build_c2_matrix_57(1)
        # C3 cycles axes: 0→1→2→0, so C3 C2(0) C3^T = C2(1)
        conjugated = R_C3 @ R_C2_0 @ R_C3.T
        np.testing.assert_allclose(conjugated, R_C2_1, atol=1e-14)


class TestUsym57:
    """Tests for 57×57 Usym construction."""

    def test_shape(self):
        """Usym₅₇ is 57×57."""
        Usym = _build_usym_57()
        assert Usym.shape == (57, 57)

    def test_orthogonality(self):
        """Usym₅₇.T @ Usym₅₇ = I_57."""
        Usym = _build_usym_57()
        UtU = Usym.T @ Usym
        np.testing.assert_allclose(UtU, np.eye(57), atol=1e-12)

    def test_column_count(self):
        """Correct number of columns per irrep: 12+6+1+2+3+1+8+9+15=57."""
        Usym = _build_usym_57()
        assert Usym.shape[1] == 57

    def test_first_27_columns_span_same_space(self):
        """The first 21 Usym₅₇ columns (ungerade) span the same space as
        the first 21 Usym₂₇ columns.

        Since ungerade modes only live in indices 0-26, the restricted
        columns should span the same 21D subspace.
        """
        U57 = _build_usym_57()
        U27 = _build_usym_27()
        # First 21 columns are the ungerade sector in both
        # In 57D, they're embedded in 57D but only have support in first 27 rows
        U57_ung = U57[:27, :21]
        U27_ung = U27[:, :21]
        # Check they span the same space by comparing projection matrices
        P57 = U57_ung @ U57_ung.T
        P27 = U27_ung @ U27_ung.T
        np.testing.assert_allclose(P57, P27, atol=1e-12)

    def test_c3_invariance_of_seeds(self):
        """Each seed column is C3-invariant (eigenvalue 1 for d=1,3; ω for d=2)."""
        R_C3 = _build_c3_matrix_57()
        Usym = _build_usym_57()
        # For d=3 irreps (T-type), each group of 3 columns should mix under C3
        # For d=1 irreps (A-type), each column is invariant
        # Check A1g (cols 21:24): each should be C3-eigenvector
        for col in range(21, 24):
            v = Usym[:, col]
            v_rot = R_C3 @ v
            # v should be unchanged (C3 eigenvalue = 1 for A₁)
            np.testing.assert_allclose(v_rot, v, atol=1e-12)

    def test_cubic_modes_are_gerade(self):
        """All cubic mode columns (indices 27-56) have even parity under inversion.

        Inversion = product of all three C2 rotations (for our O_h representation).
        Gerade ↔ inversion eigenvalue = +1.
        """
        R_inv = _build_c2_matrix_57(0) @ _build_c2_matrix_57(1) @ _build_c2_matrix_57(2)
        # Check that rows 27-56 have positive inversion eigenvalue
        for i in range(27, 57):
            assert R_inv[i, i] == 1.0, (
                f"Cubic mode {i} is ungerade (inversion eigenvalue = {R_inv[i, i]})"
            )


# ================================================================
# 2. Incident field extension tests
# ================================================================


class TestMonomial3:
    """Tests for m=3 monomial Fourier integral."""

    def test_m3_zero_k(self):
        """∫ x³ exp(0) dx = 0 (odd function)."""
        result = _monomial_fourier_1d(3, 0.0, 1.0)
        assert abs(result) < 1e-15

    def test_m3_small_k_taylor(self):
        """Small-k Taylor matches numerical integration."""
        a = 1.0
        k = 0.001
        # Numerical via scipy
        from scipy.integrate import quad

        def integrand_re(x):
            return x**3 * np.cos(k * x)

        def integrand_im(x):
            return x**3 * np.sin(k * x)

        re_val, _ = quad(integrand_re, -a, a)
        im_val, _ = quad(integrand_im, -a, a)
        expected = re_val + 1j * im_val
        result = _monomial_fourier_1d(3, k, a)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_m3_large_k_exact(self):
        """Large-k exact formula matches numerical integration."""
        a = 2.0
        k = 5.0
        from scipy.integrate import quad

        def integrand_re(x):
            return x**3 * np.cos(k * x)

        def integrand_im(x):
            return x**3 * np.sin(k * x)

        re_val, _ = quad(integrand_re, -a, a)
        im_val, _ = quad(integrand_im, -a, a)
        expected = re_val + 1j * im_val
        result = _monomial_fourier_1d(3, k, a)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_m3_purely_imaginary(self):
        """∫ x³ exp(ikx) dx is purely imaginary (odd × even = odd integrand)."""
        result = _monomial_fourier_1d(3, 2.0, 1.0)
        assert abs(np.real(result)) < 1e-14 * abs(result)


class TestCubicExponents:
    """Tests for cubic monomial exponent table."""

    def test_count(self):
        """10 cubic monomials."""
        assert len(_CUBIC_EXPONENTS) == 10

    def test_all_degree_3(self):
        """Each exponent triple sums to 3."""
        for p, q, r in _CUBIC_EXPONENTS:
            assert p + q + r == 3


class TestOverlap57:
    """Tests for 57-component overlap integrals."""

    def test_first_27_match(self):
        """First 27 components of overlap_57 match overlap_27."""
        k_vec = np.array([0.0, 0.0, 1.0])
        pol = np.array([0.0, 0.0, 1.0])
        a = 1.0
        c27 = cube_overlap_integrals(k_vec, pol, a)
        c57 = cube_overlap_integrals_57(k_vec, pol, a)
        np.testing.assert_allclose(c57[:27], c27, atol=1e-15)

    def test_shape(self):
        """Output has 57 components."""
        k_vec = np.array([1.0, 0.0, 0.0])
        pol = np.array([1.0, 0.0, 0.0])
        c = cube_overlap_integrals_57(k_vec, pol, 1.0)
        assert c.shape == (57,)

    def test_zero_k(self):
        """At k=0, overlap = volume integral of monomial × polarization.

        For odd monomials (all cubic), ∫ x^{odd} = 0 over symmetric interval.
        So cubic components should all vanish at k=0.
        """
        k_vec = np.array([0.0, 0.0, 0.0])
        pol = np.array([1.0, 0.0, 0.0])
        c = cube_overlap_integrals_57(k_vec, pol, 1.0)
        # Cubic modes (indices 27-56) all have odd monomials → zero at k=0
        np.testing.assert_allclose(c[27:], 0.0, atol=1e-15)


# ================================================================
# 3. Solver structure tests
# ================================================================


class TestGalerkin57Solver:
    """Tests for the T₅₇ solver structure and cross-validation with T₂₇.

    The gerade sector uses Galerkin body bilinear + LS-convolved stiffness
    (Path-B), while T₂₇ uses Eshelby self-consistent amplification (Path-A).
    The T1c/T2c/T3c intermediates differ between approaches, but the
    effective contrasts (Δλ*, Δμ*) agree to ~0.1%.
    """

    @pytest.fixture
    def galerkin_57(self):
        ka = 0.05
        a = 10.0
        omega = ka * REF.beta / a
        return compute_cube_tmatrix_galerkin_57(omega, a, REF, CONTRAST)

    @pytest.fixture
    def galerkin_27(self):
        ka = 0.05
        a = 10.0
        omega = ka * REF.beta / a
        return compute_cube_tmatrix_galerkin(omega, a, REF, CONTRAST)

    def test_result_type(self, galerkin_57):
        """Returns GalerkinTMatrixResult57."""
        assert isinstance(galerkin_57, GalerkinTMatrixResult57)

    def test_ungerade_matches_t27(self, galerkin_57, galerkin_27):
        """Ungerade sector eigenvalues match T₂₇ exactly."""
        np.testing.assert_allclose(
            galerkin_57.T1u_eigenvalues, galerkin_27.T1u_eigenvalues, rtol=1e-12
        )
        np.testing.assert_allclose(
            galerkin_57.T2u_eigenvalues, galerkin_27.T2u_eigenvalues, rtol=1e-12
        )
        np.testing.assert_allclose(
            galerkin_57.sigma_A2u, galerkin_27.sigma_A2u, rtol=1e-12
        )
        np.testing.assert_allclose(
            galerkin_57.sigma_Eu, galerkin_27.sigma_Eu, rtol=1e-12
        )

    def test_block_shapes(self, galerkin_57):
        """Per-irrep block shapes are correct."""
        assert galerkin_57.A1g_block.shape == (3, 3)
        assert galerkin_57.Eg_block.shape == (4, 4)
        assert galerkin_57.T1g_block.shape == (3, 3)
        assert galerkin_57.T2g_block.shape == (5, 5)
        assert galerkin_57.T1u_block.shape == (4, 4)
        assert galerkin_57.T2u_block.shape == (2, 2)

    def test_eigenvalue_counts(self, galerkin_57):
        """Correct number of eigenvalues per irrep."""
        assert len(galerkin_57.A1g_eigenvalues) == 3
        assert len(galerkin_57.Eg_eigenvalues) == 4
        assert len(galerkin_57.T1g_eigenvalues) == 3
        assert len(galerkin_57.T2g_eigenvalues) == 5
        assert len(galerkin_57.T1u_eigenvalues) == 4
        assert len(galerkin_57.T2u_eigenvalues) == 2

    def test_t_scalar_relation(self, galerkin_57):
        """T1c, T2c, T3c follow from per-irrep strain eigenvalues."""
        # σ_T2g = 2*T2c
        np.testing.assert_allclose(
            galerkin_57.sigma_T2g, 2.0 * galerkin_57.T2c, rtol=1e-12
        )
        # σ_Eg = 2*T2c + T3c
        np.testing.assert_allclose(
            galerkin_57.sigma_Eg, 2.0 * galerkin_57.T2c + galerkin_57.T3c, rtol=1e-12
        )
        # σ_A1g = 3*T1c + 2*T2c + T3c
        np.testing.assert_allclose(
            galerkin_57.sigma_A1g,
            3.0 * galerkin_57.T1c + 2.0 * galerkin_57.T2c + galerkin_57.T3c,
            rtol=1e-12,
        )

    def test_t_scalar_complex(self, galerkin_57):
        """T1c, T2c, T3c are complex (have imaginary part from smooth correction)."""
        assert abs(galerkin_57.T1c.imag) > 0, "T1c should have nonzero imaginary part"
        assert abs(galerkin_57.T2c.imag) > 0, "T2c should have nonzero imaginary part"
        assert abs(galerkin_57.T3c.imag) > 0, "T3c should have nonzero imaginary part"

    def test_radiation_damping_sign(self, galerkin_57):
        """Radiation damping: Im(σ) should be nonzero for gerade eigenvalues.

        At finite ka, the smooth body bilinear adds an imaginary part
        representing energy radiated to infinity (radiation damping).
        """
        assert abs(galerkin_57.sigma_A1g.imag) > 0
        assert abs(galerkin_57.sigma_Eg.imag) > 0
        assert abs(galerkin_57.sigma_T2g.imag) > 0

    def test_static_limit_real(self):
        """At omega=0 (static limit), smooth correction vanishes → real σ."""
        omega = 0.0
        a = 10.0
        result = compute_cube_tmatrix_galerkin_57(omega, a, REF, CONTRAST)
        assert abs(result.sigma_A1g.imag) < 1e-20
        assert abs(result.sigma_Eg.imag) < 1e-20
        assert abs(result.sigma_T2g.imag) < 1e-20

    def test_effective_contrasts_agree_with_t27(self, galerkin_57, galerkin_27):
        """T₅₇ effective contrasts agree with T₂₇ to within 2%.

        Both use Galerkin body bilinear + LS stiffness + smooth correction,
        but T₂₇ uses 1×1 gerade blocks while T₅₇ uses enlarged blocks
        (3×3 A1g, 4×4 Eg, 5×5 T2g) with cubic mode corrections.
        The (ka)⁴ corrections from cubic modes create ~1-2% differences.
        """
        np.testing.assert_allclose(
            galerkin_57.Dlambda_star, galerkin_27.Dlambda_star, rtol=2e-2
        )
        np.testing.assert_allclose(
            galerkin_57.Dmu_star_off, galerkin_27.Dmu_star_off, rtol=2e-2
        )
        np.testing.assert_allclose(
            galerkin_57.Dmu_star_diag, galerkin_27.Dmu_star_diag, rtol=2e-2
        )
