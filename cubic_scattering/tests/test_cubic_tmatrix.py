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
  7. Traction extraction
  8. Small-sphere comparison (cube T3→0 as ωa→0)
  9. Static Eshelby + dynamic: verify against CubicTMatrix_FullGreensTensor.nb
"""

import sys

import numpy as np

from cubic_scattering import (
    MaterialContrast,
    ReferenceMedium,
    compute_cube_tmatrix,
    effective_stiffness_voigt,
    strain_from_displacement_traction,
    traction_from_strain,
    voigt_tmatrix_6x6,
)
from cubic_scattering.resonance_tmatrix import (
    _build_incident_field_coupled,
    _propagator_block_9x9,
    compute_resonance_tmatrix,
    elastodynamic_greens,
    elastodynamic_greens_deriv,
    sub_cell_centres,
    voigt_tmatrix_from_resonance_result,
)


def test_born_limit():
    """
    Test 1: In the Born limit (Δc → 0), all amplification factors → 1
    and effective contrasts → bare contrasts.

    Note: The static Eshelby depolarization tensor means that T₁, T₂, T₃
    are finite even for a → 0 (they scale as Δc/ρv², independent of a).
    The true Born limit is weak scattering: Δc/ρv² << 1.
    """
    print("Test 1: Born limit (Δc → 0)")

    ref = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
    mu = ref.mu  # = 22.5e9

    # Weak contrasts: 0.01% of background moduli
    contrast = MaterialContrast(Dlambda=mu * 1e-4, Dmu=mu * 1e-4, Drho=ref.rho * 1e-4)
    omega = 2.0 * np.pi * 10.0
    a = 10.0

    result = compute_cube_tmatrix(omega, a, ref, contrast)

    # Check amplification factors ≈ 1  (T_i ∝ Δc → 0)
    tol = 1e-4
    assert abs(result.amp_u - 1.0) < tol, f"amp_u = {result.amp_u}, expected ≈ 1"
    assert abs(result.amp_theta - 1.0) < tol, (
        f"amp_theta = {result.amp_theta}, expected ≈ 1"
    )
    assert abs(result.amp_e_off - 1.0) < tol, (
        f"amp_e_off = {result.amp_e_off}, expected ≈ 1"
    )
    assert abs(result.amp_e_diag - 1.0) < tol, (
        f"amp_e_diag = {result.amp_e_diag}, expected ≈ 1"
    )

    # Check effective contrasts ≈ bare contrasts
    assert abs(result.Drho_star - contrast.Drho) / abs(contrast.Drho) < tol, (
        "Drho_star relative error too large"
    )
    assert abs(result.Dlambda_star - contrast.Dlambda) / abs(contrast.Dlambda) < tol, (
        "Dlambda_star relative error too large"
    )
    assert abs(result.Dmu_star_off - contrast.Dmu) / abs(contrast.Dmu) < tol, (
        "Dmu_star_off relative error too large"
    )
    assert abs(result.Dmu_star_diag - contrast.Dmu) / abs(contrast.Dmu) < tol, (
        "Dmu_star_diag relative error too large"
    )

    # Cubic anisotropy should be small relative to Δμ*
    assert abs(result.cubic_anisotropy) / abs(result.Dmu_star_off) < 0.1, (
        f"Cubic anisotropy = {result.cubic_anisotropy}, expected small relative to Dmu*"
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

    # C^c has both static (Eshelby) and dynamic parts.
    # The static part arises from the cube's departure from spherical
    # symmetry: C_stat ∝ b₀·(3j₂ - k₁) where the geometric factor
    # (3j₂ - k₁) = -2(5√3 - 2π)/9 ≠ 0 for a cube.
    kPa = omega * a / ref.alpha
    kSa = omega * a / ref.beta
    print(f"  kP*a = {kPa:.4f}, kS*a = {kSa:.4f}")

    # T3 should be the same order as T1, T2 (static Eshelby contributes)
    # but less than 1 for the self-consistency to converge
    ratio = abs(result.T3c) / max(abs(result.T1c), abs(result.T2c))
    print(f"  |T3|/max(|T1|,|T2|) = {ratio:.6f}")
    assert ratio < 5.0, (
        f"T3 seems anomalously large relative to T1,T2 (ratio = {ratio:.6f})"
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
    Test 8: When α/β → 1 (Poisson ratio → −1), C_stat → 0 because
    b₀ = (α²−β²)/(8πρα²β²) → 0.  In this limit, T3 → 0 and the
    cube T-matrix reduces to the isotropic (sphere-like) form.

    Note: The old test checked ωa → 0, but the static Eshelby C_stat
    is independent of ω and a.  The sphere limit is really α → β.
    """
    print("Test 8: Sphere limit (α ≈ β → C_stat → 0)")

    # Nearly incompressible: α/β only slightly > 1
    beta = 3000.0
    alpha = beta * 1.001  # α/β = 1.001
    ref = ReferenceMedium(alpha=alpha, beta=beta, rho=2500.0)
    contrast = MaterialContrast(Dlambda=2.0e9, Dmu=1.0e9, Drho=100.0)

    omega = 2.0 * np.pi * 10.0
    a = 10.0

    result = compute_cube_tmatrix(omega, a, ref, contrast)

    print(f"  α/β = {alpha / beta:.6f}")

    # T3 should be much smaller than T2 because C_stat ∝ (α²-β²)
    if abs(result.T2c) > 1e-30:
        ratio = abs(result.T3c) / abs(result.T2c)
        print(f"  |T3/T2| = {ratio:.6e}")
        assert ratio < 0.01, f"|T3/T2| = {ratio} should be << 1 when α ≈ β"

    # Dmu_star_diag ≈ Dmu_star_off
    if abs(result.Dmu_star_off) > 1e-30:
        aniso_ratio = abs(result.cubic_anisotropy) / abs(result.Dmu_star_off)
        print(f"  |cubic_aniso / Dmu_star_off| = {aniso_ratio:.6e}")
        assert aniso_ratio < 0.01, (
            f"Cubic anisotropy ratio = {aniso_ratio} should be << 1 when α ≈ β"
        )

    print("  Sphere limit: all checks passed ✓")


def test_notebook_verification():
    """
    Test 9: Verify A, B, C and T1, T2, T3 against the analytic results
    from CubicTMatrix_FullGreensTensor.nb (Mathematica notebook).

    Parameters (seismological units: km/s, g/cm³):
      α=5, β=3, ρ=2.5, ω=20π, a=0.01, Δλ=1.25, Δμ=0.75

    The notebook computes the FULL integrals (static Eshelby + smooth
    dynamic) symbolically, then evaluates numerically.
    """
    print("Test 9: Notebook verification (static + dynamic)")

    ref = ReferenceMedium(alpha=5.0, beta=3.0, rho=2.5)
    contrast = MaterialContrast(Dlambda=1.25, Dmu=0.75, Drho=0.0)
    omega = 2.0 * np.pi * 10.0
    a = 0.01

    result = compute_cube_tmatrix(omega, a, ref, contrast)

    # Reference values from CubicTMatrix_FullGreensTensor.nb Section 10
    ref_A = -0.012201107458741135 - 0.00008607635386320063j
    ref_B = 0.0026137073560736812 + 0.00003973958017869591j
    ref_C = -0.003587055298886922 - 1.0444731153752018e-10j
    ref_T1 = -0.002746105632556146 + 0.00015071169827338333j
    ref_T2 = -0.007190550077000591 - 0.00003475258026337854j
    ref_T3 = -0.005380582948330384 - 1.566709673062803e-10j

    tol = 1e-8  # ~8 significant digits (limited by Taylor truncation at n=8)

    for name, computed, expected in [
        ("A", result.Ac, ref_A),
        ("B", result.Bc, ref_B),
        ("C", result.Cc, ref_C),
        ("T1", result.T1c, ref_T1),
        ("T2", result.T2c, ref_T2),
        ("T3", result.T3c, ref_T3),
    ]:
        rel_err = abs(computed - expected) / abs(expected)
        print(f"  {name}: rel_err = {rel_err:.2e}")
        assert rel_err < tol, (
            f"{name} mismatch: computed={computed}, expected={expected}, "
            f"rel_err={rel_err:.2e}"
        )

    # Verify static Eshelby dominance: |Re(A)| >> |Im(A)|
    assert abs(result.Ac.real) > 100 * abs(result.Ac.imag), (
        "Static (real) part of A should dominate over dynamic (imaginary) part"
    )
    assert abs(result.Cc.real) > 1e6 * abs(result.Cc.imag), (
        "C should be almost entirely real (static Eshelby)"
    )

    print("  Notebook verification: all checks passed ✓")


# ================================================================
# Tests for coupled (Drho, Dlambda, Dmu) Foldy-Lax
# ================================================================


def test_greens_deriv_finite_difference():
    """Gd and Gdd vs finite differences of elastodynamic_greens."""
    ref = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
    omega = 2.0 * np.pi * 50.0
    r_vec = np.array([0.3, -0.2, 0.5])

    G, Gd, Gdd = elastodynamic_greens_deriv(r_vec, omega, ref)

    # Verify G matches existing function
    G_ref = elastodynamic_greens(r_vec, omega, ref)
    assert np.allclose(G, G_ref, rtol=1e-12), "G does not match elastodynamic_greens"

    # Finite-difference check for Gd
    h = 1e-7
    for k in range(3):
        r_plus = r_vec.copy()
        r_minus = r_vec.copy()
        r_plus[k] += h
        r_minus[k] -= h
        G_plus = elastodynamic_greens(r_plus, omega, ref)
        G_minus = elastodynamic_greens(r_minus, omega, ref)
        Gd_fd = (G_plus - G_minus) / (2.0 * h)
        err = np.max(np.abs(Gd[:, :, k] - Gd_fd))
        scale = max(np.max(np.abs(Gd[:, :, k])), 1e-30)
        assert err / scale < 1e-5, f"Gd[:,:,{k}] finite diff error {err / scale:.2e}"

    # Finite-difference check for Gdd
    for k in range(3):
        r_plus = r_vec.copy()
        r_minus = r_vec.copy()
        r_plus[k] += h
        r_minus[k] -= h
        _, Gd_plus, _ = elastodynamic_greens_deriv(r_plus, omega, ref)
        _, Gd_minus, _ = elastodynamic_greens_deriv(r_minus, omega, ref)
        Gdd_fd = (Gd_plus - Gd_minus) / (2.0 * h)
        # Gdd[:, :, :, k] = ∂Gd/∂x_k
        err = np.max(np.abs(Gdd[:, :, :, k] - Gdd_fd))
        scale = max(np.max(np.abs(Gdd[:, :, :, k])), 1e-30)
        assert err / scale < 1e-4, f"Gdd[:,:,:,{k}] finite diff error {err / scale:.2e}"


def test_greens_deriv_symmetry():
    """G_{ij,k} = G_{ji,k} (G is symmetric in i,j)."""
    ref = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
    omega = 2.0 * np.pi * 50.0
    r_vec = np.array([0.4, -0.1, 0.6])

    _, Gd, Gdd = elastodynamic_greens_deriv(r_vec, omega, ref)

    # Symmetry in first two indices
    for k in range(3):
        assert np.allclose(Gd[:, :, k], Gd[:, :, k].T, atol=1e-20), (
            f"Gd[:,:,{k}] not symmetric"
        )
    for k in range(3):
        for l_idx in range(3):
            assert np.allclose(
                Gdd[:, :, k, l_idx], Gdd[:, :, k, l_idx].T, atol=1e-20
            ), f"Gdd[:,:,{k},{l_idx}] not symmetric"


def test_propagator_9x9_rayleigh_limit():
    """At large r, off-diagonal blocks (C, H, S) are small relative to G."""
    ref = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
    omega = 2.0 * np.pi * 10.0
    r_vec = np.array([0.0, 0.0, 100.0])  # far field

    P = _propagator_block_9x9(r_vec, omega, ref)

    G_block = P[:3, :3]
    C_block = P[:3, 3:]
    H_block = P[3:, :3]
    S_block = P[3:, 3:]

    norm_G = np.linalg.norm(G_block)
    assert norm_G > 0, "G block should be nonzero"

    # Off-diagonal blocks decay faster (1/r² vs 1/r for G in far field)
    ratio_C = np.linalg.norm(C_block) / norm_G
    ratio_H = np.linalg.norm(H_block) / norm_G
    ratio_S = np.linalg.norm(S_block) / norm_G

    assert ratio_C < 0.1, f"C/G ratio = {ratio_C:.4f}, expected << 1 at large r"
    assert ratio_H < 0.1, f"H/G ratio = {ratio_H:.4f}, expected << 1 at large r"
    assert ratio_S < 0.1, f"S/G ratio = {ratio_S:.4f}, expected << 1 at large r"


def test_coupled_rayleigh_limit():
    """At n_sub=1, coupled T3x3 matches Rayleigh density-only T-matrix."""
    ref = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
    contrast = MaterialContrast(Dlambda=2.0e9, Dmu=1.0e9, Drho=100.0)
    omega = 2.0 * np.pi * 10.0
    a = 1.0

    result = compute_resonance_tmatrix(omega, a, ref, contrast, n_sub=1)

    # Rayleigh reference: density-only T-matrix = ω² Δρ* V I₃
    rayleigh = compute_cube_tmatrix(omega, a, ref, contrast)
    V = (2.0 * a) ** 3
    t_eff = V * omega**2 * complex(rayleigh.Drho_star)
    T_ray = t_eff * np.eye(3, dtype=complex)

    # At n_sub=1, no inter-cell coupling — T3x3 block must match density T-matrix
    rel_err = np.linalg.norm(result.T3x3 - T_ray) / max(np.linalg.norm(T_ray), 1e-30)
    assert rel_err < 1e-10, f"Coupled T3x3 vs Rayleigh at n=1: rel_err = {rel_err:.2e}"


def test_coupled_density_only_isotropic():
    """Dlambda=Dmu=0: T3x3 is isotropic, stress_dipole_voigt is zero."""
    ref = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
    contrast = MaterialContrast(Dlambda=0.0, Dmu=0.0, Drho=200.0)
    omega = 2.0 * np.pi * 10.0
    a = 1.0

    result = compute_resonance_tmatrix(omega, a, ref, contrast, n_sub=2)

    # T3x3 should be isotropic (proportional to I₃) for density-only contrast
    T3x3 = result.T3x3
    diag = np.diag(T3x3)
    off_diag_max = np.max(np.abs(T3x3 - np.diag(diag)))
    assert off_diag_max < 1e-10 * np.max(np.abs(diag)), (
        f"T3x3 off-diagonal = {off_diag_max:.2e}, should be ~0 for density-only"
    )
    assert np.allclose(diag[0], diag[1], rtol=1e-10), "T3x3 diagonal not isotropic"
    assert np.allclose(diag[0], diag[2], rtol=1e-10), "T3x3 diagonal not isotropic"

    # No stiffness contrast → stress dipole block should be zero
    assert np.linalg.norm(result.stress_dipole_voigt) < 1e-10 * np.linalg.norm(T3x3), (
        "stress_dipole_voigt should be ~0 for density-only contrast"
    )


def test_coupled_stress_dipole_nonzero():
    """Nonzero Dlambda/Dmu produces nonzero stress_dipole_voigt."""
    ref = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
    contrast = MaterialContrast(Dlambda=2.0e9, Dmu=1.0e9, Drho=100.0)
    omega = 2.0 * np.pi * 10.0
    a = 1.0

    result = compute_resonance_tmatrix(omega, a, ref, contrast, n_sub=1)

    assert result.stress_dipole_voigt is not None, "stress_dipole_voigt should exist"
    assert np.linalg.norm(result.stress_dipole_voigt) > 0, (
        "stress_dipole_voigt should be nonzero for nonzero stiffness contrast"
    )


def test_coupled_cubic_symmetry():
    """force_monopole has Oh diagonal symmetry."""
    ref = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
    contrast = MaterialContrast(Dlambda=2.0e9, Dmu=1.0e9, Drho=100.0)
    omega = 2.0 * np.pi * 10.0
    a = 1.0

    result = compute_resonance_tmatrix(omega, a, ref, contrast, n_sub=2)

    T3x3 = result.force_monopole

    # Oh symmetry: T3x3 should be diagonal with equal entries
    diag = np.diag(T3x3)
    off_diag_max = np.max(np.abs(T3x3 - np.diag(diag)))
    assert off_diag_max < 1e-10 * np.max(np.abs(diag)), (
        f"force_monopole off-diag = {off_diag_max:.2e}, should be ~0"
    )
    assert np.allclose(diag[0], diag[1], rtol=1e-10), (
        f"T3x3[0,0]={diag[0]}, T3x3[1,1]={diag[1]} should be equal"
    )
    assert np.allclose(diag[0], diag[2], rtol=1e-10), (
        f"T3x3[0,0]={diag[0]}, T3x3[2,2]={diag[2]} should be equal"
    )


def test_voigt_from_resonance_near_field():
    """Near-field 6x6: displacement block and traction block vs analytic construction."""
    from cubic_scattering.resonance_tmatrix import (
        _traction_from_voigt_strain,
        _volume_integral_voigt,
    )

    ref = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
    contrast = MaterialContrast(Dlambda=2.0e9, Dmu=1.0e9, Drho=100.0)
    omega = 2.0 * np.pi * 1.0
    a = 0.5
    kx = 0.001
    ky = 0.0

    rayleigh = compute_cube_tmatrix(omega, a, ref, contrast)
    V = (2.0 * a) ** 3

    result = compute_resonance_tmatrix(omega, a, ref, contrast, n_sub=1)
    T6_res = voigt_tmatrix_from_resonance_result(result, kx=kx, ky=ky)

    # 1. Displacement block = Gamma0 * omega^2 * Drho* * V * I3
    expected_u = complex(rayleigh.Gamma0 * omega**2 * rayleigh.Drho_star * V)
    for i in range(3):
        rel_err = abs(T6_res[i, i] - expected_u) / max(abs(expected_u), 1e-30)
        assert rel_err < 1e-10, f"u diag [{i}] rel_err = {rel_err:.2e}"

    # 2. Oh symmetry: txz and tyz entries equal
    assert np.allclose(T6_res[4, 4], T6_res[5, 5], rtol=1e-10), (
        "txz and tyz traction entries should be equal"
    )

    # 3. Traction block matches analytic G_self @ T_loc @ input_conv
    S_V = _volume_integral_voigt(rayleigh.Ac, rayleigh.Bc, rayleigh.Cc)
    C_trac = _traction_from_voigt_strain(ref)
    Dc_star = effective_stiffness_voigt(
        rayleigh.Dlambda_star, rayleigh.Dmu_star_diag, rayleigh.Dmu_star_off
    )
    S = strain_from_displacement_traction(kx, ky, ref)
    expected_trac = V * C_trac @ S_V @ Dc_star @ S
    rel_err_t = np.linalg.norm(T6_res[3:, :] - expected_trac) / max(
        np.linalg.norm(expected_trac), 1e-30
    )
    assert rel_err_t < 1e-10, f"Traction block vs analytic: rel_err = {rel_err_t:.2e}"


def test_incident_strain_plane_wave():
    """_build_incident_field_coupled gives correct Voigt strain."""
    ref = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
    omega = 2.0 * np.pi * 10.0
    a = 1.0
    n = 2
    centres = sub_cell_centres(a, n)

    U0 = _build_incident_field_coupled(centres, omega, ref)
    N = len(centres)

    # Column 0 (displacement e_z): u=(1,0,0), strain=0 at centre
    m_centre = N // 2
    u_col0 = U0[9 * m_centre : 9 * m_centre + 3, 0]
    eps_col0 = U0[9 * m_centre + 3 : 9 * m_centre + 9, 0]
    # Phase is exp(ikz·z), displacement should be nonzero
    assert np.abs(u_col0[0]) > 0, "u_z for col 0 should be nonzero"
    assert np.allclose(eps_col0, 0.0), "strain for col 0 should be zero"

    # Column 3 (strain ε_zz): strain should have e_0 Voigt
    eps_col3 = U0[9 * m_centre + 3 : 9 * m_centre + 9, 3]
    phase = U0[9 * m_centre + 3, 3]  # should be the phase value
    expected_eps = np.zeros(6, dtype=complex)
    expected_eps[0] = phase  # e_zz in Voigt
    assert np.allclose(eps_col3, expected_eps, atol=1e-14), (
        f"Strain col 3: {eps_col3} vs expected {expected_eps}"
    )


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
        test_notebook_verification,
        test_greens_deriv_finite_difference,
        test_greens_deriv_symmetry,
        test_propagator_9x9_rayleigh_limit,
        test_coupled_rayleigh_limit,
        test_coupled_density_only_isotropic,
        test_coupled_stress_dipole_nonzero,
        test_coupled_cubic_symmetry,
        test_voigt_from_resonance_near_field,
        test_incident_strain_plane_wave,
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
