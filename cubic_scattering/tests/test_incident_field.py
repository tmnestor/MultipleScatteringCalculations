"""
test_incident_field.py
Tests for plane-wave projection onto the 27-component Galerkin basis.
"""

import numpy as np

from cubic_scattering import ReferenceMedium
from cubic_scattering.incident_field import (
    _monomial_fourier_1d,
    cube_overlap_integrals,
    plane_wave_PSV_SH,
)

REF = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)


def test_constant_field():
    """k=0, pol=e_z → only displacement and even-power monomials nonzero."""
    c = cube_overlap_integrals(np.zeros(3), np.array([0.0, 0.0, 1.0]), 1.0)

    # Displacement: c[2] = pol_z * V = 1 * 8 = 8
    assert c[0] == 0.0
    assert c[1] == 0.0
    np.testing.assert_allclose(c[2], 8.0, atol=1e-14)

    # Strain: all zero (odd monomial × even k=0 → nonzero only for even power)
    # Actually axial strain has r_k which is odd → integral = 0 at k=0
    np.testing.assert_allclose(c[3:9], 0.0, atol=1e-14)


def test_monomial_limits():
    """k→0 limits match moment integrals."""
    a = 1.0
    # m=0: ∫_{-1}^{1} 1 dx = 2
    np.testing.assert_allclose(_monomial_fourier_1d(0, 0.0, a), 2.0, atol=1e-14)

    # m=1: ∫_{-1}^{1} x dx = 0 (odd function)
    np.testing.assert_allclose(_monomial_fourier_1d(1, 0.0, a), 0.0, atol=1e-14)

    # m=2: ∫_{-1}^{1} x² dx = 2/3
    np.testing.assert_allclose(_monomial_fourier_1d(2, 0.0, a), 2.0 / 3.0, atol=1e-14)

    # Check small-k continuity: compare Taylor with exact at small ka
    k_small = 0.01
    for m in range(3):
        val = _monomial_fourier_1d(m, k_small, a)
        # Verify it's close to the k=0 limit
        val0 = _monomial_fourier_1d(m, 0.0, a)
        if abs(val0) > 0:
            assert abs(val - val0) / abs(val0) < 0.01


def test_monomial_exact_values():
    """Check exact formulas at moderate ka."""
    a = 1.0
    k = 2.0
    ka = k * a

    # m=0: 2 sin(ka)/ka = 2 sinc(ka/π)
    expected = 2.0 * np.sin(ka) / ka
    np.testing.assert_allclose(_monomial_fourier_1d(0, k, a), expected, rtol=1e-14)

    # m=1: (2i/k²)[sin(ka) - ka cos(ka)]
    expected = 2j / k**2 * (np.sin(ka) - ka * np.cos(ka))
    np.testing.assert_allclose(_monomial_fourier_1d(1, k, a), expected, rtol=1e-14)

    # m=2: (2/k³)[2ka cos(ka) + (k²a²-2) sin(ka)]
    expected = 2.0 / k**3 * (2.0 * ka * np.cos(ka) + (ka**2 - 2.0) * np.sin(ka))
    np.testing.assert_allclose(_monomial_fourier_1d(2, k, a), expected, rtol=1e-14)


def test_p_wave_strain():
    """P-wave along z → only z-related strain coupling."""
    omega = 100.0
    kP = omega / REF.alpha
    k_vec = np.array([0.0, 0.0, kP])
    pol = np.array([0.0, 0.0, 1.0])  # P-wave

    c = cube_overlap_integrals(k_vec, pol, 1.0)

    # Displacement: only z-component
    assert abs(c[0]) < 1e-14
    assert abs(c[1]) < 1e-14
    assert abs(c[2]) > 0

    # Axial strain: only ε_zz (r_3 e_3, index 5) should be nonzero
    # r_1 e_1: pol_1=0 → c[3]=0
    # r_2 e_2: pol_2=0 → c[4]=0
    assert abs(c[3]) < 1e-14
    assert abs(c[4]) < 1e-14
    assert abs(c[5]) > 0  # ε_zz from P-wave


def test_psv_sh_generation():
    """Check PSV/SH waves are orthogonal and have correct wavenumbers."""
    k_hat = np.array([0.0, 0.0, 1.0])
    omega = 1000.0
    waves = plane_wave_PSV_SH(k_hat, omega, REF)

    assert len(waves) == 3

    k_P, pol_P, label_P = waves[0]
    k_SV, pol_SV, label_SV = waves[1]
    k_SH, pol_SH, label_SH = waves[2]

    assert label_P == "P"
    assert label_SV == "SV"
    assert label_SH == "SH"

    # Wavenumber magnitudes
    np.testing.assert_allclose(np.linalg.norm(k_P), omega / REF.alpha, rtol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(k_SV), omega / REF.beta, rtol=1e-14)

    # P-wave polarization parallel to k_hat
    np.testing.assert_allclose(abs(np.dot(pol_P, k_hat)), 1.0, atol=1e-14)

    # SV perpendicular to k_hat
    np.testing.assert_allclose(np.dot(pol_SV, k_hat), 0.0, atol=1e-14)

    # SH perpendicular to both k_hat and SV
    np.testing.assert_allclose(np.dot(pol_SH, k_hat), 0.0, atol=1e-14)
    np.testing.assert_allclose(np.dot(pol_SH, pol_SV), 0.0, atol=1e-14)
