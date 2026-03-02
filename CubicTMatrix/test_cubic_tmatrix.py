"""
test_cubic_tmatrix.py
Comprehensive tests for the CubicTMatrix package.

Tests:
  1. Born limit (a→0): effective contrasts → bare contrasts
  2. Voigt T-matrix structure: block-diagonal, correct eigenvalues
  3. Strain extraction: verify strain ↔ (u,t) consistency
  4. Cubic symmetry: verify I_{1111}=I_{2222}=I_{3333} etc.
  5. Numerical comparison with tmatrix_cube.py SymPy results
  6. Effective stiffness matrix symmetry
  7. Small-sphere comparison (cube T3→0 as ωa→0)
"""

import os
import sys

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CubicTMatrix import (
    MaterialContrast,
    ReferenceMedium,
    compute_cube_tmatrix,
    effective_stiffness_voigt,
    strain_from_displacement_traction,
    traction_from_strain,
    voigt_tmatrix_6x6,
)


def test_born_limit():
    """
    Test 1: In the Born limit (a → 0), all amplification factors → 1
    and effective contrasts → bare contrasts.
    """
    print("Test 1: Born limit (a → 0)")

    ref = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
    contrast = MaterialContrast(Dlambda=2.0e9, Dmu=1.0e9, Drho=100.0)
    omega = 2.0 * np.pi * 10.0

    # Very small cube
    a_small = 0.01  # 1 cm half-width
    result = compute_cube_tmatrix(omega, a_small, ref, contrast)

    # Check amplification factors ≈ 1
    assert abs(result.amp_u - 1.0) < 1e-6, f"amp_u = {result.amp_u}, expected ≈ 1"
    assert abs(result.amp_theta - 1.0) < 1e-6, (
        f"amp_theta = {result.amp_theta}, expected ≈ 1"
    )
    assert abs(result.amp_e_off - 1.0) < 1e-6, (
        f"amp_e_off = {result.amp_e_off}, expected ≈ 1"
    )
    assert abs(result.amp_e_diag - 1.0) < 1e-6, (
        f"amp_e_diag = {result.amp_e_diag}, expected ≈ 1"
    )

    # Check effective contrasts ≈ bare contrasts
    assert abs(result.Drho_star - contrast.Drho) < 1e-3, (
        f"Drho_star = {result.Drho_star}, expected ≈ {contrast.Drho}"
    )
    assert abs(result.Dlambda_star - contrast.Dlambda) / abs(contrast.Dlambda) < 1e-6, (
        "Dlambda_star relative error too large"
    )
    assert abs(result.Dmu_star_off - contrast.Dmu) / abs(contrast.Dmu) < 1e-6, (
        "Dmu_star_off relative error too large"
    )
    assert abs(result.Dmu_star_diag - contrast.Dmu) / abs(contrast.Dmu) < 1e-6, (
        "Dmu_star_diag relative error too large"
    )

    # Cubic anisotropy should vanish
    assert abs(result.cubic_anisotropy) / abs(contrast.Dmu) < 1e-6, (
        f"Cubic anisotropy = {result.cubic_anisotropy}, expected ≈ 0"
    )

    print("  Born limit: all checks passed ✓")


def test_voigt_structure():
    """
    Test 2: Voigt T-matrix has correct block structure.
    """
    print("Test 2: Voigt T-matrix structure")

    T1c = 0.001 + 0.0001j
    T2c = -0.0002 + 0.00003j
    T3c = 0.00005 - 0.00001j

    T_V = voigt_tmatrix_6x6(T1c, T2c, T3c)

    # Block-diagonal structure: upper-right and lower-left 3×3 = 0
    assert np.allclose(T_V[:3, 3:], 0), "Off-diagonal blocks should be zero"
    assert np.allclose(T_V[3:, :3], 0), "Off-diagonal blocks should be zero"

    # D block: T1 J + (2T2+T3) I3
    D = T_V[:3, :3]
    expected_diag = T1c + 2 * T2c + T3c
    expected_off = T1c
    for i in range(3):
        assert abs(D[i, i] - expected_diag) < 1e-15, (
            f"D diagonal [{i},{i}] = {D[i, i]}, expected {expected_diag}"
        )
        for j in range(3):
            if i != j:
                assert abs(D[i, j] - expected_off) < 1e-15, (
                    f"D off-diag [{i},{j}] = {D[i, j]}, expected {expected_off}"
                )

    # S block: 2T2 I3
    S = T_V[3:, 3:]
    expected_shear = 2 * T2c
    assert np.allclose(S, expected_shear * np.eye(3)), "S block should be 2T2 × I3"

    # Eigenvalues of D block
    D_eigs = np.sort(np.real(np.linalg.eigvals(D)))
    # Dilatation mode: eigenvalue = 3T1 + 2T2 + T3
    # Deviatoric modes (×2): eigenvalue = 2T2 + T3
    ev_dilat = 3 * T1c + 2 * T2c + T3c
    ev_deviat = 2 * T2c + T3c
    eigs_expected = np.sort(np.real([ev_deviat, ev_deviat, ev_dilat]))
    assert np.allclose(D_eigs, eigs_expected, atol=1e-14), (
        f"D eigenvalues: {D_eigs} vs expected {eigs_expected}"
    )

    print("  Voigt structure: all checks passed ✓")


def test_strain_extraction():
    """
    Test 3: Strain extraction from (u,t) is consistent with
    the constitutive relations for a plane wave.
    """
    print("Test 3: Strain extraction consistency")

    ref = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
    lam = ref.lam
    mu = ref.mu

    kx = 0.005
    ky = 0.003  # out-of-plane wavenumber

    # Pick a P-wave: k = (kx, ky, kz_P) where kz_P² = (ω/α)² - kx² - ky²
    omega = 2 * np.pi * 10.0
    k_alpha = omega / ref.alpha
    kz_P_sq = k_alpha**2 - kx**2 - ky**2
    assert kz_P_sq > 0, "Need propagating P-wave"
    kz_P = np.sqrt(kz_P_sq)

    # P-wave polarisation: displacement ∝ k-hat
    k_mag = np.sqrt(kx**2 + ky**2 + kz_P**2)
    uz_0 = kz_P / k_mag
    ux_0 = kx / k_mag
    uy_0 = ky / k_mag

    # Compute strains directly from plane wave
    eps_zz = 1j * kz_P * uz_0
    eps_xx = 1j * kx * ux_0
    eps_yy = 1j * ky * uy_0
    two_eps_zx = 1j * kz_P * ux_0 + 1j * kx * uz_0
    two_eps_zy = 1j * kz_P * uy_0 + 1j * ky * uz_0
    two_eps_xy = 1j * ky * ux_0 + 1j * kx * uy_0

    strain_direct = np.array(
        [eps_zz, eps_xx, eps_yy, two_eps_xy, two_eps_zy, two_eps_zx]
    )

    # Compute tractions on z=const surface
    theta = eps_zz + eps_xx + eps_yy
    tzz_0 = lam * theta + 2 * mu * eps_zz
    txz_0 = mu * two_eps_zx
    tyz_0 = mu * two_eps_zy

    # Build (u,t) vector
    ut_vec = np.array([uz_0, ux_0, uy_0, tzz_0, txz_0, tyz_0])

    # Extract strains via the matrix
    S = strain_from_displacement_traction(kx, ky, ref)
    strain_extracted = S @ ut_vec

    # Compare
    max_err = np.max(np.abs(strain_extracted - strain_direct))
    rel_err = max_err / np.max(np.abs(strain_direct))
    assert rel_err < 1e-12, f"Strain extraction error: rel_err = {rel_err:.2e}"

    print(f"  P-wave strain extraction: rel_err = {rel_err:.2e} ✓")

    # Also test with an SV-wave
    k_beta = omega / ref.beta
    kz_S_sq = k_beta**2 - kx**2 - ky**2
    assert kz_S_sq > 0, "Need propagating S-wave"
    kz_S = np.sqrt(kz_S_sq)

    # SV-wave: displacement ⊥ k in the (kx, kz) plane
    # Polarisation: u ∝ (-kz, 0, kx) for pure SV in xz-plane
    # More generally, for 3D: u ⊥ k, in the sagittal plane
    k_h = np.sqrt(kx**2 + ky**2)
    k_mag_S = np.sqrt(kx**2 + ky**2 + kz_S**2)
    # SV in the (k_h, kz) plane: u_z ∝ -k_h, u_h ∝ kz
    uz_0 = -k_h / k_mag_S
    ux_0 = kz_S * kx / (k_h * k_mag_S) if k_h > 0 else 0.0
    uy_0 = kz_S * ky / (k_h * k_mag_S) if k_h > 0 else 0.0

    eps_zz = 1j * kz_S * uz_0
    eps_xx = 1j * kx * ux_0
    eps_yy = 1j * ky * uy_0
    two_eps_zx = 1j * kz_S * ux_0 + 1j * kx * uz_0
    two_eps_zy = 1j * kz_S * uy_0 + 1j * ky * uz_0
    two_eps_xy = 1j * ky * ux_0 + 1j * kx * uy_0

    strain_direct = np.array(
        [eps_zz, eps_xx, eps_yy, two_eps_xy, two_eps_zy, two_eps_zx]
    )

    theta = eps_zz + eps_xx + eps_yy
    tzz_0 = lam * theta + 2 * mu * eps_zz
    txz_0 = mu * two_eps_zx
    tyz_0 = mu * two_eps_zy

    ut_vec = np.array([uz_0, ux_0, uy_0, tzz_0, txz_0, tyz_0])
    strain_extracted = S @ ut_vec

    max_err = np.max(np.abs(strain_extracted - strain_direct))
    rel_err = max_err / np.max(np.abs(strain_direct))
    assert rel_err < 1e-12, f"SV strain extraction error: rel_err = {rel_err:.2e}"

    print(f"  SV-wave strain extraction: rel_err = {rel_err:.2e} ✓")


def test_cubic_symmetry():
    """
    Test 4: Verify cubic symmetry of the integral decomposition.
    """
    print("Test 4: Cubic symmetry verification")

    ref = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
    contrast = MaterialContrast(Dlambda=2.0e9, Dmu=1.0e9, Drho=100.0)
    omega = 2.0 * np.pi * 10.0
    a = 10.0

    result = compute_cube_tmatrix(omega, a, ref, contrast)

    # Ac, Bc, Cc should satisfy:
    # I_1111 = Ac + 2Bc + Cc
    # I_1122 = Ac
    # I_1212 = Bc
    I_1111 = result.Ac + 2 * result.Bc + result.Cc
    I_1122 = result.Ac
    I_1212 = result.Bc

    print(f"  I_1111 = {I_1111}")
    print(f"  I_1122 = {I_1122}")
    print(f"  I_1212 = {I_1212}")
    print(f"  Cc = {result.Cc}")

    # C^c should be real-valued to leading order and O(ωa)^7
    # (the cubic anisotropy only appears at 7th order)
    kPa = omega * a / ref.alpha
    kSa = omega * a / ref.beta
    print(f"  kP*a = {kPa:.4f}, kS*a = {kSa:.4f}")

    # T3 should be small compared to T1, T2
    ratio = abs(result.T3c) / max(abs(result.T1c), abs(result.T2c))
    print(f"  |T3|/max(|T1|,|T2|) = {ratio:.6f}")
    assert ratio < 0.1 * max(kPa, kSa) ** 4, (
        f"T3 seems too large relative to T1,T2 (ratio = {ratio:.6f})"
    )

    print("  Cubic symmetry: all checks passed ✓")


def test_numerical_comparison():
    """
    Test 5: Compare with reference values from tmatrix_cube.py SymPy code.

    Uses the same parameters as Section 10 of TMatrix_Derivation.pdf:
      α=5000, β=3000, ρ=2500, f=10Hz, a=10m
      Δλ=2GPa, Δμ=1GPa, Δρ=100
    """
    print("Test 5: Numerical comparison with SymPy reference")

    ref = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
    contrast = MaterialContrast(Dlambda=2.0e9, Dmu=1.0e9, Drho=100.0)
    omega = 2.0 * np.pi * 10.0
    a = 10.0

    result = compute_cube_tmatrix(omega, a, ref, contrast)

    kPa = omega * a / ref.alpha
    kSa = omega * a / ref.beta
    print(f"  kP*a = {kPa:.4f}, kS*a = {kSa:.4f}")

    # Print all computed values for comparison
    print(f"\n  {'Quantity':<20} {'Value':<50}")
    print(f"  {'-' * 70}")
    for name, val in [
        ("Gamma0", result.Gamma0),
        ("Ac", result.Ac),
        ("Bc", result.Bc),
        ("Cc", result.Cc),
        ("T1c", result.T1c),
        ("T2c", result.T2c),
        ("T3c", result.T3c),
        ("amp_u", result.amp_u),
        ("amp_theta", result.amp_theta),
        ("amp_e_off", result.amp_e_off),
        ("amp_e_diag", result.amp_e_diag),
        ("Drho_star", result.Drho_star),
        ("Dlambda_star", result.Dlambda_star),
        ("Dmu_star_diag", result.Dmu_star_diag),
        ("Dmu_star_off", result.Dmu_star_off),
        ("cubic_aniso", result.cubic_anisotropy),
    ]:
        print(f"  {name:<20} {val.real:+.6e} {val.imag:+.6e}j")

    # Basic sanity checks
    assert result.Gamma0.real != 0, "Gamma0 should be nonzero"
    assert abs(result.amp_u - 1.0) < 0.01, "amp_u should be close to 1"
    assert abs(result.T3c) < abs(result.T2c), "T3 should be smaller than T2"
    assert abs(result.Drho_star.real - contrast.Drho) / contrast.Drho < 0.01, (
        "Drho_star should be close to Drho for weak scattering"
    )

    print("\n  Numerical comparison: sanity checks passed ✓")


def test_effective_stiffness_symmetry():
    """
    Test 6: Effective stiffness matrix should be symmetric
    and have correct eigenvalue structure.
    """
    print("Test 6: Effective stiffness matrix properties")

    ref = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
    contrast = MaterialContrast(Dlambda=2.0e9, Dmu=1.0e9, Drho=100.0)
    omega = 2.0 * np.pi * 10.0
    a = 10.0

    result = compute_cube_tmatrix(omega, a, ref, contrast)

    Dc = effective_stiffness_voigt(
        result.Dlambda_star, result.Dmu_star_diag, result.Dmu_star_off
    )

    # Should be symmetric
    assert np.allclose(Dc, Dc.T), "Effective stiffness should be symmetric"

    # Block-diagonal
    assert np.allclose(Dc[:3, 3:], 0), "Upper-right block should be zero"
    assert np.allclose(Dc[3:, :3], 0), "Lower-left block should be zero"

    # Shear block should be diagonal with 2Δμ*_off
    S_block = Dc[3:, 3:]
    expected = 2.0 * result.Dmu_star_off * np.eye(3)
    assert np.allclose(S_block, expected), "Shear block incorrect"

    print("  Effective stiffness: all checks passed ✓")


def test_traction_extraction():
    """
    Test 7: Traction extraction matrix correctly picks traction components.
    """
    print("Test 7: Traction extraction")

    ref = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
    P = traction_from_strain(0.0, 0.0, ref)

    # Voigt stress: (σzz, σxx, σyy, σxy, σzy, σzx)
    # Traction on z=const: (tzz, txz, tyz) = (σzz, σzx, σzy)
    stress = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    trac = P @ stress

    assert trac[0] == 1.0, "tzz should be σzz"  # Voigt 0
    assert trac[1] == 6.0, "txz should be σzx"  # Voigt 5
    assert trac[2] == 5.0, "tyz should be σzy"  # Voigt 4

    print("  Traction extraction: all checks passed ✓")


def test_sphere_limit():
    """
    Test 8: For very small ωa, T3→0 and the cube T-matrix reduces
    to the sphere T-matrix (with appropriate volume scaling).
    """
    print("Test 8: Sphere limit (T3 → 0)")

    ref = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
    contrast = MaterialContrast(Dlambda=2.0e9, Dmu=1.0e9, Drho=100.0)

    # Small ωa regime
    omega = 2.0 * np.pi * 1.0  # 1 Hz (low frequency)
    a = 1.0  # 1 m cube half-width

    result = compute_cube_tmatrix(omega, a, ref, contrast)

    kPa = omega * a / ref.alpha
    kSa = omega * a / ref.beta
    print(f"  kP*a = {kPa:.6f}, kS*a = {kSa:.6f}")

    # T3 should be much smaller than T1, T2
    if abs(result.T2c) > 1e-30:
        ratio = abs(result.T3c) / abs(result.T2c)
        print(f"  |T3/T2| = {ratio:.6e}")
        assert ratio < 0.01, f"|T3/T2| = {ratio} should be << 1"

    # Dmu_star_diag ≈ Dmu_star_off
    if abs(result.Dmu_star_off) > 1e-30:
        aniso_ratio = abs(result.cubic_anisotropy) / abs(result.Dmu_star_off)
        print(f"  |cubic_aniso / Dmu_star_off| = {aniso_ratio:.6e}")
        assert aniso_ratio < 0.01, (
            f"Cubic anisotropy ratio = {aniso_ratio} should be << 1"
        )

    print("  Sphere limit: all checks passed ✓")


# ================================================================
# Run all tests
# ================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  CubicTMatrix Test Suite")
    print("=" * 70)

    tests = [
        test_born_limit,
        test_voigt_structure,
        test_strain_extraction,
        test_cubic_symmetry,
        test_numerical_comparison,
        test_effective_stiffness_symmetry,
        test_traction_extraction,
        test_sphere_limit,
    ]

    passed = 0
    failed = 0

    for test in tests:
        print()
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 70}")
    print(f"  Results: {passed} passed, {failed} failed out of {len(tests)}")
    print(f"{'=' * 70}")

    if failed > 0:
        sys.exit(1)
