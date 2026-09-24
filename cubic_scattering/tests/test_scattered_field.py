"""
test_scattered_field.py
Tests for far-field scattering amplitudes from the T27 cube T-matrix.
"""

import numpy as np

from cubic_scattering import (
    MaterialContrast,
    ReferenceMedium,
    compute_cube_tmatrix_galerkin,
    compute_elastic_mie,
    mie_far_field,
)
from cubic_scattering.incident_field import cube_overlap_integrals
from cubic_scattering.scattered_field import (
    cube_far_field,
    optical_theorem_check,
    optical_theorem_from_amplitudes,
    scattering_cross_section,
)
from cubic_scattering.tmatrix_assembly import assemble_tmatrix_27

REF = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
CONTRAST = MaterialContrast(Dlambda=2e9, Dmu=1e9, Drho=100.0)
WEAK_CONTRAST = MaterialContrast(Dlambda=REF.mu * 1e-4, Dmu=REF.mu * 1e-4, Drho=REF.rho * 1e-4)


def _setup(ka: float, a: float = 10.0, contrast: MaterialContrast = CONTRAST):
    omega = ka * REF.beta / a
    g = compute_cube_tmatrix_galerkin(omega, a, REF, contrast)
    T27 = assemble_tmatrix_27(g)
    kP = omega / REF.alpha
    k_vec = np.array([0.0, 0.0, kP])
    pol = np.array([0.0, 0.0, 1.0])
    c_inc = cube_overlap_integrals(k_vec, pol, a)
    c_sc = T27 @ c_inc
    return omega, g, T27, k_vec, pol, c_inc, c_sc


def test_dipole_pattern():
    """Pure density contrast → cos(θ) dipole P-wave pattern."""
    density_only = MaterialContrast(Dlambda=0.0, Dmu=0.0, Drho=100.0)
    omega, g, T27, k_vec, pol, c_inc, c_sc = _setup(0.05, contrast=density_only)

    theta = np.linspace(0, np.pi, 100)
    f_P, f_SV, f_SH = cube_far_field(c_inc, c_sc, theta, REF, g, density_only, omega, 10.0, k_vec, pol)

    # For pure density contrast, f_P should be dominated by cos(θ) pattern
    # from the dipole term F·r̂ = |F| cos(θ)
    f_P_real = np.real(f_P)
    # Check cos(θ) proportionality: f_P(0)/f_P(π) should be negative
    # (dipole reverses sign)
    assert f_P_real[0] * f_P_real[-1] < 0, "Dipole pattern should reverse at θ=π"

    # f_P(π/2) should be near zero for pure cos(θ) pattern
    mid_idx = len(theta) // 2
    assert abs(f_P_real[mid_idx]) < 0.1 * abs(f_P_real[0])


def test_stiffness_monopole():
    """Pure stiffness contrast → monopole + quadrupole pattern (no cos(θ) dipole)."""
    stiffness_only = MaterialContrast(Dlambda=2e9, Dmu=1e9, Drho=0.0)
    omega, g, T27, k_vec, pol, c_inc, c_sc = _setup(0.05, contrast=stiffness_only)

    theta = np.array([0.0, np.pi / 2, np.pi])
    f_P, f_SV, f_SH = cube_far_field(c_inc, c_sc, theta, REF, g, stiffness_only, omega, 10.0, k_vec, pol)

    # With Δρ=0, the density force monopole is zero.
    # The stiffness stress dipole gives l=0 (monopole) + l=2 (quadrupole).
    # The monopole is isotropic, so f_P should be similar at all angles.
    # Check that f_P(0) and f_P(π) have the same sign (monopole dominates).
    assert np.real(f_P[0]) * np.real(f_P[2]) > 0, "Stiffness monopole should have same sign at 0 and π"


def test_cube_vs_mie_rayleigh():
    """ka=0.05 far-field matches equal-volume Mie sphere to ~10%."""
    omega, g, T27, k_vec, pol, c_inc, c_sc = _setup(0.05)
    a = 10.0

    theta = np.array([0.0, np.pi / 4, np.pi / 2])
    f_P, f_SV, f_SH = cube_far_field(c_inc, c_sc, theta, REF, g, CONTRAST, omega, a, k_vec, pol)

    # Equal-volume sphere
    V_cube = (2 * a) ** 3
    a_sphere = (3 * V_cube / (4 * np.pi)) ** (1.0 / 3.0)
    mie = compute_elastic_mie(omega, a_sphere, REF, CONTRAST, n_max=10)
    f_P_mie, f_SV_mie, f_SH_mie = mie_far_field(mie, theta, "P")

    # In the Rayleigh limit, cube and sphere far-field should agree
    # to within shape-dependent corrections (~5% for cube vs sphere)
    for i in range(len(theta)):
        ratio = abs(f_P[i]) / abs(f_P_mie[i])
        assert 0.85 < ratio < 1.25, (
            f"theta={np.degrees(theta[i]):.0f}°: |cube/mie|={ratio:.3f} out of range"
        )


def test_cube_vs_mie_weak_contrast():
    """Weak contrast: cube matches Mie more closely (Born limit)."""
    omega, g, T27, k_vec, pol, c_inc, c_sc = _setup(0.05, contrast=WEAK_CONTRAST)
    a = 10.0

    f_P, _, _ = cube_far_field(c_inc, c_sc, np.array([0.0]), REF, g, WEAK_CONTRAST, omega, a, k_vec, pol)

    V_cube = (2 * a) ** 3
    a_sphere = (3 * V_cube / (4 * np.pi)) ** (1.0 / 3.0)
    mie = compute_elastic_mie(omega, a_sphere, REF, WEAK_CONTRAST, n_max=10)
    f_P_mie, _, _ = mie_far_field(mie, np.array([0.0]), "P")

    # At weak contrast, the Born approximation is accurate and
    # cube/sphere differ only by geometric shape factors
    ratio = abs(f_P[0]) / abs(f_P_mie[0])
    assert 0.90 < ratio < 1.15, f"|cube/mie|={ratio:.4f} at weak contrast"


def _mie_forward_pp(mie) -> complex:
    """Forward (θ=0) P→P Mie amplitude, summed analytically from the coefficients.

    The forward extinction is a tiny O((ka)³) imaginary part riding on an O(1)
    real part, so a single θ=0 Hankel evaluation of ``mie_far_field`` is
    numerically unreliable at low ka.  Asymptotically ``k h_n'(kr) → (-i)^n
    e^{ikr}/r`` and ``P_n(1) = 1``, so

        f_PP(0) = Σ a_n (-i)^n,

    in the standard e^{-iωt} convention used by ``optical_theorem_from_amplitudes``
    (σ_ext = +(4π/k) Im f(0)).  The stored ``a_n`` no longer carry the old (-1)^n
    rotation (removed in ``sphere_scattering.py``), and this sum is no longer
    negated: that negation only compensated the former -4π/k checker sign.
    """
    return sum(mie.a_n[n] * (-1j) ** n for n in range(mie.n_max + 1))


def test_optical_theorem_mie_gate():
    """Ground-truth energy-conservation gate: exact-Mie σ_ext = σ_sc ≈ 1.0.

    The elastic optical theorem (``LatexPDFs/psvsh_optical_theorem.tex``):

        σ_ext = +(4π/k_P) Im[f_PP(0)],
        σ_sc  = 2π ∫ [ |f_P|² + (β/α)(|f_SV|²+|f_SH|²) ] sinθ dθ,

    from energy flux: a unit-amplitude scattered wave of mode c carries radial
    flux ½ρcω², so the S channels carry weight β/α = k_P/k_S.

    ``compute_elastic_mie`` is unitary, so the ratio must be 1.0 and
    ka-independent.  Regression guard: the inverse α/β S-flux weight shifts the
    level off 1.0, and a wrong forward phase makes it drift with ka; both FAIL
    the bands below.
    """
    radius = 10.0
    ka_vals = [0.05, 0.1, 0.3, 0.5]
    ratios = []
    for ka in ka_vals:
        omega = ka * REF.beta / radius
        kP = omega / REF.alpha
        mie = compute_elastic_mie(omega, radius, REF, CONTRAST, n_max=12)
        theta = np.linspace(0.0, np.pi, 2000)
        f_P, f_SV, f_SH = mie_far_field(mie, theta, "P")
        f_pp_fwd = _mie_forward_pp(mie)
        sigma_ext, sigma_sc = optical_theorem_from_amplitudes(
            f_pp_fwd, theta, f_P, f_SV, f_SH, kP, REF.alpha, REF.alpha, REF.beta
        )
        assert sigma_sc > 0, f"ka={ka}: σ_sc={sigma_sc:.3e} must be positive"
        assert sigma_ext > 0, f"ka={ka}: σ_ext={sigma_ext:.3e} must be positive"
        ratios.append(sigma_ext / sigma_sc)

    ratios = np.array(ratios)
    # (1) Frequency-independence: a phase-corrupted forward would drift with ka
    #     (the old (-1)^n-rotated sum gave the stable-but-wrong ≈0.229).
    spread = float(ratios.max() - ratios.min())
    assert spread < 0.01, (
        f"exact-Mie σ_ext/σ_sc must be ka-independent with the phase-consistent "
        f"forward sum; got {ratios} (spread {spread:.4f})."
    )
    # (2) Energy conservation: σ_ext = σ_sc = 1.0 for the unitary exact-Mie
    #     sphere.  The (-1)^n-rotated forward gives ≈0.229; the wrong α/β S-flux
    #     weight shifts the level off 1.0.  Both FAIL this band.
    assert np.all(np.abs(ratios - 1.0) < 0.02), (
        f"exact-Mie σ_ext/σ_sc = {ratios}; expected 1.0 (energy conservation). "
        "The (-1)^n-rotated forward sum gives ≈0.229 (phase artifact, not a Mie "
        "defect); the wrong α/β S-flux weight also fails (see docstring)."
    )


def test_optical_theorem_cube():
    """Cube T27 closes the elastic optical theorem: σ_ext/σ_sc = 1 (lossless).

    σ_ext = (4π/k_P) Im[f_P(0)] and σ_sc with the β/α S-flux weight, both from
    optical_theorem_check.  The density channel's forward extinction comes
    from Im[Γ₀] through the complex amp_u; the modulus channel's from the
    radiation reaction Im[Δc*] through the stress dipole.  At forward incidence
    the incident strain i k_P d̂d̂ is imaginary, so Im[Δc*] lands in Im[f_P(0)].

    Before the fix the far field used Re[Δc*] only, and the ratio sat at 0.61
    with the modulus extinction missing entirely.
    """
    for ka, tol in ((0.05, 0.003), (0.1, 0.003), (0.3, 0.005)):
        omega, g, T27, k_vec, pol, c_inc, c_sc = _setup(ka)
        sigma_ext, sigma_sc = optical_theorem_check(T27, REF, g, CONTRAST, omega, 10.0, k_vec, pol)
        ratio = sigma_ext / sigma_sc
        assert abs(ratio - 1.0) < tol, f"ka={ka}: cube σ_ext/σ_sc = {ratio:.5f}, expected 1"


def test_optical_theorem_cube_modulus_only():
    """A pure-modulus cube has non-zero extinction equal to its scattering.

    Regression guard: with Re[Δc*] only, σ_ext was exactly 0 here while σ_sc > 0.
    """
    modulus_only = MaterialContrast(Dlambda=2e9, Dmu=1e9, Drho=0.0)
    for ka in (0.05, 0.1, 0.3):
        omega, g, T27, k_vec, pol, c_inc, c_sc = _setup(ka, contrast=modulus_only)
        sigma_ext, sigma_sc = optical_theorem_check(T27, REF, g, modulus_only, omega, 10.0, k_vec, pol)
        ratio = sigma_ext / sigma_sc
        assert abs(ratio - 1.0) < 0.003, f"ka={ka}: modulus-only σ_ext/σ_sc = {ratio:.5f}"


def test_optical_theorem_checkers_share_sign_convention():
    """Both checkers compute σ_ext from the SAME forward amplitude identically.

    optical_theorem_from_amplitudes and optical_theorem_check must use one
    convention, σ_ext = +(4π/k) Im[f(0)] with e^{-iωt}; a passive scatterer gives
    σ_ext > 0 in both.
    """
    omega, g, T27, k_vec, pol, c_inc, c_sc = _setup(0.1)
    kP = omega / REF.alpha
    theta = np.linspace(0.0, np.pi, 721)
    f_P, f_SV, f_SH = cube_far_field(c_inc, c_sc, theta, REF, g, CONTRAST, omega, 10.0, k_vec, pol)
    ext_amp, _ = optical_theorem_from_amplitudes(
        f_P[0], theta, f_P, f_SV, f_SH, kP, REF.alpha, REF.alpha, REF.beta
    )
    ext_chk, _ = optical_theorem_check(T27, REF, g, CONTRAST, omega, 10.0, k_vec, pol)
    assert ext_chk > 0.0
    assert abs(ext_amp - ext_chk) < 1e-12 * abs(ext_chk), f"{ext_amp} vs {ext_chk}"


def test_cross_section_scales_with_contrast():
    """Scattering cross-section scales as contrast² in Born limit."""
    a = 10.0
    ka = 0.05

    # Compute at two different weak contrasts
    contrasts = [
        MaterialContrast(Dlambda=REF.mu * 1e-4, Dmu=REF.mu * 1e-4, Drho=REF.rho * 1e-4),
        MaterialContrast(Dlambda=REF.mu * 2e-4, Dmu=REF.mu * 2e-4, Drho=REF.rho * 2e-4),
    ]
    sigmas = []
    for c in contrasts:
        omega, g, T27, k_vec, pol, c_inc, c_sc = _setup(ka, a=a, contrast=c)
        sigma = scattering_cross_section(c_inc, c_sc, REF, g, c, omega, a, k_vec, pol)
        sigmas.append(sigma)

    # σ_sc ~ contrast² in Born limit, doubling contrast → 4× cross-section
    ratio = sigmas[1] / sigmas[0]
    assert 3.5 < ratio < 4.5, f"σ_sc ratio for 2× contrast = {ratio:.2f}, expected ~4"


def test_cross_section_positive():
    """Scattering cross-section should be positive."""
    omega, g, T27, k_vec, pol, c_inc, c_sc = _setup(0.05)

    sigma_sc = scattering_cross_section(c_inc, c_sc, REF, g, CONTRAST, omega, 10.0, k_vec, pol)
    assert sigma_sc > 0


def test_sv_scattering_nonzero():
    """P-wave incidence should produce nonzero SV scattering at off-axis angles."""
    omega, g, T27, k_vec, pol, c_inc, c_sc = _setup(0.05)

    theta = np.array([np.pi / 4])
    f_P, f_SV, f_SH = cube_far_field(c_inc, c_sc, theta, REF, g, CONTRAST, omega, 10.0, k_vec, pol)

    # For P-wave along z with stiffness contrast, mode conversion to SV
    # should be nonzero at oblique angles
    assert abs(f_SV[0]) > 0

    # SH is small relative to P-wave scattering (cube has cubic symmetry,
    # not cylindrical, so SH need not be exactly zero in xz plane)
    assert abs(f_SH[0]) < 0.25 * abs(f_P[0])


def test_far_field_scales_with_frequency():
    """In Rayleigh limit, f_P ~ (ka)³ for stiffness, (ka)³ for density."""
    a = 10.0
    f_vals = []
    ka_vals = [0.02, 0.05, 0.1]

    for ka in ka_vals:
        omega, g, T27, k_vec, pol, c_inc, c_sc = _setup(ka, a=a)
        f_P, _, _ = cube_far_field(c_inc, c_sc, np.array([0.0]), REF, g, CONTRAST, omega, a, k_vec, pol)
        f_vals.append(abs(f_P[0]))

    # f_P(0) ~ (ka)^2 in the Rayleigh limit (monopole + dipole both ~ k²)
    # Check that doubling ka roughly quadruples |f_P|
    ratio_1 = f_vals[1] / f_vals[0]
    ratio_2 = f_vals[2] / f_vals[1]
    ka_ratio_1 = (ka_vals[1] / ka_vals[0]) ** 2
    ka_ratio_2 = (ka_vals[2] / ka_vals[1]) ** 2

    assert abs(ratio_1 / ka_ratio_1 - 1.0) < 0.15, (
        f"f_P scaling: got {ratio_1:.3f}, expected {ka_ratio_1:.3f}"
    )
    assert abs(ratio_2 / ka_ratio_2 - 1.0) < 0.15, (
        f"f_P scaling: got {ratio_2:.3f}, expected {ka_ratio_2:.3f}"
    )
