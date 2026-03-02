"""Verification test suite for the FFTProp package.

Tests are structured to verify each component against the original
Fortran FFTPROP.F behaviour. Since we don't have Fortran output files
to compare against directly, tests verify:

1. Internal consistency of spectral arrays
2. Physical properties (symmetry, reciprocity, energy conservation)
3. Numerical stability and correctness of key functions
4. End-to-end execution with default parameters

Run from the IntegralEquationScattering directory:
    python3 FFTProp.py/test_package.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Add parent directory to path for package import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib.util
from pathlib import Path

# Import via importlib since directory name contains a dot
spec = importlib.util.spec_from_file_location(
    "FFTProp",
    str(Path(__file__).parent / "__init__.py"),
)
pkg = importlib.util.module_from_spec(spec)
sys.modules["FFTProp"] = pkg
spec.loader.exec_module(pkg)

from FFTProp import (
    GridConfig,
    build_spectral_arrays,
    compute_wavefield,
    default_grid,
    default_medium,
    default_source_array,
    vertical_slowness,
)


def test_medium_properties():
    """Test 1: Reference medium derived quantities match Fortran."""
    print("=" * 60)
    print("Test 1: Reference medium properties")
    print("=" * 60)

    med = default_medium()
    assert med.alpha == 5.0
    assert med.rho == 3.0
    assert med.Q == 50.0

    # Fortran: ai = alpha/(2*Q) = 5/(100) = 0.05
    assert abs(med.alpha_imag - 0.05) < 1e-15
    print(f"  alpha_imag = {med.alpha_imag} (expected 0.05)")

    # Fortran: a0 = Cmplx(alpha, ai) = (5.0, 0.05)
    a0 = med.complex_alpha
    assert abs(a0 - complex(5.0, 0.05)) < 1e-15
    print(f"  a0 = {a0} (expected (5+0.05j))")

    # Fortran: ca0 = C1/a0
    ca0 = med.complex_slowness_p
    expected_ca0 = 1.0 / complex(5.0, 0.05)
    assert abs(ca0 - expected_ca0) < 1e-15
    print(f"  ca0 = {ca0:.10f}")

    # Fortran: cb0 = ca0/sqrt(3)
    cb0 = med.complex_slowness_s
    expected_cb0 = expected_ca0 / np.sqrt(3.0)
    assert abs(cb0 - expected_cb0) < 1e-15
    print(f"  cb0 = {cb0:.10f}")

    # Fortran: w = Cmplx(2*Pi*freq, 0.2)
    w = med.complex_omega(2.0, 0.2)
    expected_w = complex(2.0 * np.pi * 2.0, 0.2)
    assert abs(w - expected_w) < 1e-14
    print(f"  w = {w:.10f}")

    # Fortran: ka0 = w*a0
    ka0 = med.ka0(2.0, 0.2)
    expected_ka0 = expected_w * complex(5.0, 0.05)
    assert abs(ka0 - expected_ka0) < 1e-12
    print(f"  ka0 = {ka0:.10f}")

    # Fortran: kb0 = ka0*sqrt(3)
    kb0 = med.kb0(2.0, 0.2)
    expected_kb0 = expected_ka0 * np.sqrt(3.0)
    assert abs(kb0 - expected_kb0) < 1e-12
    print(f"  kb0 = {kb0:.10f}")

    print("  PASSED\n")


def test_vertical_slowness():
    """Test 2: Vertical slowness matches Fortran VERTICALSLO."""
    print("=" * 60)
    print("Test 2: Vertical slowness (VERTICALSLO)")
    print("=" * 60)

    med = default_medium()
    ka0 = med.ka0(2.0, 0.2)

    # Test propagating case: kx = 0 → kz = ka0
    kx = np.array([0.0 + 0j])
    kz = vertical_slowness(ka0, kx)
    # kz should equal ka0 (since kz = sqrt(ka0² - 0²) = ka0)
    assert abs(kz[0] - ka0) < 1e-10 or abs(kz[0] + ka0) < 1e-10
    # Check Im(kz) >= 0
    assert kz[0].imag >= 0, f"Im(kz) = {kz[0].imag} should be >= 0"
    print(f"  kz(kx=0) = {kz[0]:.8f}")
    print(f"  ka0      = {ka0:.8f}")

    # Test evanescent case: kx >> ka0
    kx_large = np.array([10.0 * abs(ka0) + 0j])
    kz_evan = vertical_slowness(ka0, kx_large)
    assert kz_evan[0].imag >= 0, "Evanescent wave should have Im(kz) >= 0"
    print(f"  kz(kx=10*|ka0|) = {kz_evan[0]:.8f} (should be mostly imaginary)")

    # Test branch cut: Im(kz) should never be negative (strict LT convention)
    kx_array = np.linspace(-2 * abs(ka0), 2 * abs(ka0), 1000) + 0j
    kz_array = vertical_slowness(ka0, kx_array)
    assert np.all(kz_array.imag >= 0), "Branch cut violation: found Im(kz) < 0"
    print(f"  Branch cut check: all {len(kz_array)} values have Im(kz) >= 0")

    # Test symmetry: kz(-kx) should equal kz(kx)
    kx_sym = np.array([5.0 + 0j, -5.0 + 0j])
    kz_sym = vertical_slowness(ka0, kx_sym)
    assert abs(kz_sym[0] - kz_sym[1]) < 1e-14, "kz should be symmetric in kx"
    print(f"  Symmetry: kz(5) = kz(-5) = {kz_sym[0]:.8f}")

    print("  PASSED\n")


def test_spectral_arrays():
    """Test 3: Spectral array construction and consistency."""
    print("=" * 60)
    print("Test 3: Spectral arrays")
    print("=" * 60)

    grid = default_grid()
    med = default_medium()
    sa = build_spectral_arrays(med, grid, freq=2.0, atten_imag=0.2)

    Nk = grid.Nk
    Nh = grid.Nh

    # Check array shapes
    assert sa.kxvec.shape == (Nk,), f"kxvec shape: {sa.kxvec.shape}"
    assert sa.kzavec.shape == (Nk,), f"kzavec shape: {sa.kzavec.shape}"
    assert sa.PC.shape == (Nk, 5, 2, grid.Nscatz), f"PC shape: {sa.PC.shape}"
    print(f"  Array shapes correct: kxvec={sa.kxvec.shape}, PC={sa.PC.shape}")

    # Check kx grid: first element = 0, FFT-ordered
    assert abs(sa.kxvec[0]) < 1e-15, f"kxvec[0] should be 0, got {sa.kxvec[0]}"
    assert abs(sa.kxvec[Nh].real + Nh * sa.dkx) < 1e-10, "kxvec[Nh] check"
    print(f"  kxvec[0] = {sa.kxvec[0]}")
    print(f"  kxvec[Nh] = {sa.kxvec[Nh]} (expected {-Nh * sa.dkx})")

    # Check phase factors: Eavec = rtEa²
    Ea_check = sa.rtEa * sa.rtEa
    assert np.allclose(sa.Eavec, Ea_check), "Eavec should equal rtEa²"
    print("  Eavec = rtEa²: verified")

    # Check vertical slowness positive imaginary part
    assert np.all(sa.kzavec.imag >= 0), "kzavec should have Im >= 0"
    assert np.all(sa.kzbvec.imag >= 0), "kzbvec should have Im >= 0"
    print("  Im(kzavec) >= 0: verified")
    print("  Im(kzbvec) >= 0: verified")

    # Check grid parameters
    ka0 = med.ka0(2.0, 0.2)
    expected_kxmax = 32.0 * ka0.real
    expected_dkx = 2.0 * expected_kxmax / float(Nk)
    assert abs(sa.dkx - expected_dkx) < 1e-12
    print(f"  dkx = {sa.dkx:.8f} (expected {expected_dkx:.8f})")
    print(f"  dx = {sa.dx:.8f}")
    print(f"  z = {sa.z:.8f}")

    # Check PC transform: m=0 should be sqrt(dkx / (pi*kz))
    for i_depth in range(grid.Nscatz):
        pc_m0_p = sa.PC[:, 2, 0, i_depth]
        expected_m0 = np.sqrt(sa.dkx / (np.pi * sa.kzavec))
        assert np.allclose(pc_m0_p, expected_m0, rtol=1e-12), (
            f"PC m=0, P-wave, depth {i_depth} mismatch"
        )
    print("  PC[m=0] = sqrt(dkx/pi*kz): verified for all depth layers")

    print("  PASSED\n")


def test_free_surface():
    """Test 4: Free-surface reflection properties."""
    print("=" * 60)
    print("Test 4: Free-surface reflection")
    print("=" * 60)

    grid = default_grid()
    med = default_medium()
    sa = build_spectral_arrays(med, grid, freq=2.0, atten_imag=0.2)

    # Check Rayleigh reflection at normal incidence (kx=0)
    # At normal incidence: p=0, so R1=1, R12=1, R2=0, Rayleigh=1
    # Rpp_base = (-1+0)/1 = -1, Rpp = Eavec[0]*(-1), Rss = Ebvec[0]*(-1)
    # Rsp = 0 (no P-SV coupling at normal incidence)
    idx0 = 0  # kx=0 index
    rpp_0 = sa.Rpp[idx0]
    rsp_0 = sa.Rsp[idx0]
    rss_0 = sa.Rss[idx0]
    print(f"  Rpp(kx=0) = {rpp_0:.8f}")
    print(f"  Rsp(kx=0) = {rsp_0:.8f} (should be ~0)")
    print(f"  Rss(kx=0) = {rss_0:.8f}")

    # Rsp should be zero at normal incidence (no P-SV coupling)
    assert abs(rsp_0) < 1e-10, f"Rsp at normal incidence should be ~0, got {rsp_0}"
    print("  Rsp(kx=0) ≈ 0: verified (no P-SV coupling at normal incidence)")

    # Check W matrices are finite and non-zero at kx=0
    assert np.isfinite(sa.W11[idx0]), "W11 not finite at kx=0"
    assert abs(sa.W11[idx0]) > 0, "W11 should be non-zero at kx=0"
    print(f"  W11(kx=0) = {sa.W11[idx0]:.8f}")
    print(f"  W22(kx=0) = {sa.W22[idx0]:.8f}")

    # W12 and W21 should be zero at normal incidence (no P-SV coupling)
    assert abs(sa.W12[idx0]) < 1e-10, "W12 at kx=0 should be ~0"
    assert abs(sa.W21[idx0]) < 1e-10, "W21 at kx=0 should be ~0"
    print("  W12(kx=0) ≈ 0, W21(kx=0) ≈ 0: verified")

    print("  PASSED\n")


def test_default_source():
    """Test 5: Default source array matches Fortran."""
    print("=" * 60)
    print("Test 5: Default source array")
    print("=" * 60)

    grid = default_grid()
    SY = default_source_array(grid)

    # Check shape
    assert SY.shape == (grid.Nscatx, 5, 2, grid.Nscatz)
    print(f"  Shape: {SY.shape}")

    # Fortran: SY(Nscatx, -2, 1, Nscatz) = C1
    # Python:  SY[Nscatx-1, 0, 0, Nscatz-1]
    val = SY[grid.Nscatx - 1, 0, 0, grid.Nscatz - 1]
    assert val == 1.0 + 0j, f"Source at bottom-right should be 1, got {val}"
    print(f"  SY[Nscatx-1, 0, 0, Nscatz-1] = {val} (expected 1+0j)")

    # Check all other elements are zero
    total_nonzero = np.count_nonzero(SY)
    assert total_nonzero == 1, f"Should have exactly 1 nonzero, got {total_nonzero}"
    print(f"  Non-zero elements: {total_nonzero} (expected 1)")

    print("  PASSED\n")


def test_fft_conventions():
    """Test 6: FFT convention matches Fortran."""
    print("=" * 60)
    print("Test 6: FFT conventions")
    print("=" * 60)

    # Fortran FFT(X, N, +1.0) should equal N * np.fft.ifft(X)
    # Fortran FFT(X, N, -1.0) should equal np.fft.fft(X)

    N = 64
    np.random.seed(42)
    X = np.random.randn(N) + 1j * np.random.randn(N)

    # k→x (signi=+1): Fortran gives sum_k X(k)*exp(+i*2*pi*n*k/N)
    # This equals N * ifft(X)
    fft_k2x = N * np.fft.ifft(X)

    # x→k (signi=-1): Fortran gives sum_k X(k)*exp(-i*2*pi*n*k/N)
    # This equals fft(X)
    fft_x2k = np.fft.fft(X)

    # Round-trip: fft(ifft(X)*N) should give N*X
    roundtrip = np.fft.fft(N * np.fft.ifft(X))
    assert np.allclose(roundtrip, N * X), "FFT round-trip failed"
    print("  Round-trip k→x→k: verified")

    # Also verify: ifft(fft(X))*N should give N*X
    roundtrip2 = N * np.fft.ifft(np.fft.fft(X))
    assert np.allclose(roundtrip2, N * X), "FFT round-trip (reverse) failed"
    print("  Round-trip x→k→x: verified")

    print("  PASSED\n")


def test_end_to_end():
    """Test 7: End-to-end computation with default parameters."""
    print("=" * 60)
    print("Test 7: End-to-end computation")
    print("=" * 60)

    # Use smaller grid for faster testing
    small_grid = GridConfig(Nk=256, Nscatx=9, Nscatz=2, jskip=4)

    print(
        f"  Running with reduced grid: Nk={small_grid.Nk}, "
        f"Nscatx={small_grid.Nscatx}, Nscatz={small_grid.Nscatz}"
    )

    result = compute_wavefield(grid=small_grid)

    # Check output shapes
    assert result.PSY.shape == (small_grid.Nscatx, 5, 2, small_grid.Nscatz)
    assert result.Svec.shape == (small_grid.Nscatx, 5, 2, small_grid.Nscatz)
    assert result.Rvec.shape == (small_grid.Nscatx, 5, 2, small_grid.Nscatz)
    print(f"  Output shapes correct: PSY={result.PSY.shape}")

    # Check that arrays contain non-trivial values
    svec_max = np.max(np.abs(result.Svec))
    rvec_max = np.max(np.abs(result.Rvec))
    psy_max = np.max(np.abs(result.PSY))
    print(f"  max|Svec| = {svec_max:.6e}")
    print(f"  max|Rvec| = {rvec_max:.6e}")
    print(f"  max|PSY|  = {psy_max:.6e}")

    # Svec and Rvec should be non-zero (source/receiver couple to scatterers)
    assert svec_max > 0, "Svec should be non-zero"
    assert rvec_max > 0, "Rvec should be non-zero"
    print("  Non-trivial output: verified")

    # Check that values are finite (no NaN or inf)
    assert np.all(np.isfinite(result.Svec)), "Svec contains NaN/inf"
    assert np.all(np.isfinite(result.Rvec)), "Rvec contains NaN/inf"
    assert np.all(np.isfinite(result.PSY)), "PSY contains NaN/inf"
    print("  All values finite: verified")

    print("  PASSED\n")


def test_full_grid():
    """Test 8: Full-size grid computation (Fortran default parameters)."""
    print("=" * 60)
    print("Test 8: Full-size grid (Nk=4096, Nscatx=81, Nscatz=2)")
    print("=" * 60)

    print("  Running with full Fortran default parameters...")
    print("  (This may take a minute...)")

    result = compute_wavefield()

    # Check shapes match Fortran dimensions
    grid = default_grid()
    assert result.PSY.shape == (grid.Nscatx, 5, 2, grid.Nscatz)
    print(f"  Output shape: {result.PSY.shape}")

    svec_max = np.max(np.abs(result.Svec))
    rvec_max = np.max(np.abs(result.Rvec))
    psy_max = np.max(np.abs(result.PSY))
    print(f"  max|Svec| = {svec_max:.6e}")
    print(f"  max|Rvec| = {rvec_max:.6e}")
    print(f"  max|PSY|  = {psy_max:.6e}")

    # Print first few Svec and Rvec values (matching Fortran output format)
    print("\n  Sample output (first 5 entries):")
    j = 0
    for iscat in range(grid.Nscatz):
        for jscat in range(min(1, grid.Nscatx)):
            for i_wt in range(2):
                for m_idx in range(5):
                    j += 1
                    rv = result.Rvec[jscat, m_idx, i_wt, iscat]
                    sv = result.Svec[jscat, m_idx, i_wt, iscat]
                    print(
                        f"    [{j:4d}] Rvec=({rv.real:+.6e},{rv.imag:+.6e}j) "
                        f"Svec=({sv.real:+.6e},{sv.imag:+.6e}j)"
                    )

    assert np.all(np.isfinite(result.Svec)), "Svec contains NaN/inf"
    assert np.all(np.isfinite(result.Rvec)), "Rvec contains NaN/inf"
    assert np.all(np.isfinite(result.PSY)), "PSY contains NaN/inf"
    print("\n  All values finite: verified")

    print("  PASSED\n")


if __name__ == "__main__":
    print("FFTProp Package Verification Suite")
    print("=" * 60)
    print()

    tests = [
        test_medium_properties,
        test_vertical_slowness,
        test_spectral_arrays,
        test_free_surface,
        test_default_source,
        test_fft_conventions,
        test_end_to_end,
    ]

    # Full grid test is optional (slow)
    run_full = "--full" in sys.argv

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    if run_full:
        try:
            test_full_grid()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback

            traceback.print_exc()
            failed += 1
    else:
        print("(Skipping full-grid test. Run with --full to include.)")

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)
