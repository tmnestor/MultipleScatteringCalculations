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
WEAK_CONTRAST = MaterialContrast(
    Dlambda=REF.mu * 1e-4, Dmu=REF.mu * 1e-4, Drho=REF.rho * 1e-4
)


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
    f_P, f_SV, f_SH = cube_far_field(
        c_inc, c_sc, theta, REF, g, density_only, omega, 10.0, k_vec, pol
    )

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
    f_P, f_SV, f_SH = cube_far_field(
        c_inc, c_sc, theta, REF, g, stiffness_only, omega, 10.0, k_vec, pol
    )

    # With Δρ=0, the density force monopole is zero.
    # The stiffness stress dipole gives l=0 (monopole) + l=2 (quadrupole).
    # The monopole is isotropic, so f_P should be similar at all angles.
    # Check that f_P(0) and f_P(π) have the same sign (monopole dominates).
    assert np.real(f_P[0]) * np.real(f_P[2]) > 0, (
        "Stiffness monopole should have same sign at 0 and π"
    )


def test_cube_vs_mie_rayleigh():
    """ka=0.05 far-field matches equal-volume Mie sphere to ~10%."""
    omega, g, T27, k_vec, pol, c_inc, c_sc = _setup(0.05)
    a = 10.0

    theta = np.array([0.0, np.pi / 4, np.pi / 2])
    f_P, f_SV, f_SH = cube_far_field(
        c_inc, c_sc, theta, REF, g, CONTRAST, omega, a, k_vec, pol
    )

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

    f_P, _, _ = cube_far_field(
        c_inc, c_sc, np.array([0.0]), REF, g, WEAK_CONTRAST, omega, a, k_vec, pol
    )

    V_cube = (2 * a) ** 3
    a_sphere = (3 * V_cube / (4 * np.pi)) ** (1.0 / 3.0)
    mie = compute_elastic_mie(omega, a_sphere, REF, WEAK_CONTRAST, n_max=10)
    f_P_mie, _, _ = mie_far_field(mie, np.array([0.0]), "P")

    # At weak contrast, the Born approximation is accurate and
    # cube/sphere differ only by geometric shape factors
    ratio = abs(f_P[0]) / abs(f_P_mie[0])
    assert 0.90 < ratio < 1.15, f"|cube/mie|={ratio:.4f} at weak contrast"


def _mie_forward_pp(mie) -> complex:
    """Phase-consistent forward (θ=0) P→P amplitude for the optical theorem.

    The forward extinction is a tiny O((ka)³) imaginary part riding on an
    O(1) real part, so a single θ=0 Hankel evaluation of ``mie_far_field`` is
    numerically unreliable at low ka.  We therefore build the forward sum
    analytically from the coefficients.

    The crucial subtlety: the stored coefficients ``mie.a_n`` carry a deliberate
    ``(-1)^n`` phase rotation (``sphere_scattering.py:659``) whose only purpose
    is to make the *angular displacement pattern* forward-peaked, matching the
    Rayleigh / Foldy-Lax convention.  That rotation is harmless for ``|f|²`` (it
    cancels in σ_sc) but it CORRUPTS the coherent forward-interference sum that
    drives the optical theorem.  We therefore UNDO it here.

    Asymptotically ``k h_n'(kr) → (-i)^n e^{ikr}/r`` and ``P_n(1)=1``, so the
    forward sum with the rotation removed is::

        f_PP(0) = Σ a_n · (-1)^n · (-i)^n = Σ a_n · i^n,

    negated to match the ``σ_ext = -(4π/k) Im[f(0)]`` sign convention used by
    ``optical_theorem_from_amplitudes`` (the same ``-4π/k`` prefactor the cube
    far-field obeys).  With this phase-consistent forward amplitude the exact
    Mie sphere closes the optical theorem to σ_ext = σ_sc = 1.0 to 5 digits,
    ka-independent — confirming ``compute_elastic_mie`` is unitary.
    """
    return -sum(mie.a_n[n] * (1j) ** n for n in range(mie.n_max + 1))


def test_optical_theorem_mie_gate():
    """Ground-truth energy-conservation gate: exact-Mie σ_ext = σ_sc ≈ 1.0.

    The elastic optical theorem (``cube_tmatrix_closedform.tex``):

        σ_ext = -(4π/k_P) Im[f_PP(0)],
        σ_sc  = 2π ∫ [ |f_P|² + (β/α)(|f_SV|²+|f_SH|²) ] sinθ dθ,

    derived from energy flux: a unit-amplitude scattered wave of mode c carries
    radial flux ½ρcω², so the per-channel weight in σ_sc is the scattered speed
    over the incident speed — β/α = k_P/k_S for the S channels (eq:dsigma), the
    energy-CONSERVING weight.  The previous code used its inverse k_S/k_P = α/β.

    ``compute_elastic_mie`` is UNITARY to 10 significant figures: per order, the
    extinction |ext_n| equals the scattered power (P→P plus P→S converted).
    There is NO Mie-solver defect and NO radiation-reaction deficit — the BC
    solve and coefficients are exact and must not be touched.

    The only obstacle to closing σ_ext = σ_sc = 1.0 was the COHERENT FORWARD
    SUM.  The stored coefficients ``mie.a_n`` carry a deliberate ``(-1)^n`` phase
    rotation (``sphere_scattering.py:659``) that makes the *angular displacement
    pattern* forward-peaked — a real convention the Rayleigh / Foldy-Lax code
    relies on, kept for the angular amplitudes.  It cancels in ``|f|²`` (σ_sc is
    unaffected) but corrupts the coherent forward-interference sum, which is the
    sole reason the naive forward amplitude gave the stable ≈0.229 artifact
    (NOT a missing P→S converted power, NOT a solver defect).

    The fix (``_mie_forward_pp``) forms the forward amplitude with the rotation
    UNDONE: f_PP(0) = Σ a_n·(-1)^n·(-i)^n = Σ a_n·i^n (negated for the -4π/k
    sign convention).  With the energy-conserving β/α S-flux weight in σ_sc this
    closes the optical theorem to σ_ext = σ_sc = 1.0, ka-INDEPENDENT — the true
    energy-conservation gate, asserted here across ka_β ∈ {0.05,0.1,0.3,0.5}.
    Verified per-order to 10 digits and total to ~5 digits.

    Regression guard: the wrong α/β S-flux weight would shift the level off 1.0,
    and the old (-1)^n-rotated forward sum would give ≈0.229 — both FAIL the
    band below.
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
    """Cube T27 σ_ext/σ_sc through the corrected (β/α) checker.

    With the energy-conserving β/α S-flux weight the cube ratio is finite,
    positive and frequency-stable (≈0.61, drifting only with ka²).

    Unlike the Mie sphere, the cube far-field (``cube_far_field``) does NOT use
    the Mie ``(-1)^n`` coefficient rotation — its forward amplitude obeys the
    same ``-Q_P/(4πρα²)`` convention as the ``-4π/k`` optical-theorem prefactor
    in ``optical_theorem_check``.  The cube forward is therefore already
    phase-consistent for the optical theorem and needs NO analogous undo; the
    checker is used unmodified here.

    The exact-Mie sphere closes the optical theorem to σ_ext = σ_sc = 1.0
    (``test_optical_theorem_mie_gate``), so the σ_sc side and the β/α weight are
    independently validated.  The cube sits below 1.0 because of the FORWARD
    amplitude, not the checker:

      * the cube's DENSITY radiation damping is now CORRECT — the corrected
        Im[Γ₀] = (1/3)P + (2/3)S diagonal radiation reaction closes the
        density-only optical theorem to 1.0 to ~0.02 across ka (see
        ``test_gamma0_radiation_reaction.test_density_optical_theorem_closes_to_one``);
      * the cube's MODULUS forward damping is a structural 2nd-order optical
        term: the linearized Galerkin far-field maps Im(Δσ) → Re(f_P), so the
        modulus contribution to Im[f_P(0)] is absent at first order.

    The ≈0.61 deficit is precisely this missing MODULUS extinction in the linear
    forward amplitude (a known T-matrix limitation, NOT a normalization bug).
    NOTE: the previous ≈0.82 pin (and its "density radiation damping is correct"
    claim) was an artifact of the OLD Im[Γ₀] overshooting the density channel by
    ~1.354×: the inflated density σ_ext partially masked the absent modulus
    extinction.  Fixing Im[Γ₀] (density σ_ext/σ_sc 1.354→1.0) drops the FULL-cube
    ratio to ≈0.61, exposing the modulus deficit honestly; the modulus channel
    is still STRUCTURALLY absent from the linear forward amplitude.  The textbook
    1.0 would require both the corrected density damping (done) AND the 2nd-order
    modulus forward term (not built).
    """
    ratios = []
    for ka in (0.05, 0.1, 0.3):
        omega, g, T27, k_vec, pol, c_inc, c_sc = _setup(ka)
        sigma_ext, sigma_sc = optical_theorem_check(
            T27, REF, g, CONTRAST, omega, 10.0, k_vec, pol
        )
        assert sigma_sc > 0, f"ka={ka}: sigma_sc={sigma_sc:.4e} should be positive"
        assert sigma_ext > 0, (
            f"ka={ka}: σ_ext={sigma_ext:.4e} should be positive "
            "(forward scattering present)"
        )
        ratios.append(sigma_ext / sigma_sc)

    ratios = np.array(ratios)
    # Achieved with the corrected β/α weight AND the corrected density Im[Γ₀]
    # (radiation reaction); frequency-stable ≈0.61.  Residual = missing MODULUS
    # forward damping (2nd-order optical term the linearized forward amplitude
    # omits); NOT a checker normalization bug and NOT a density-damping error.
    assert np.all((ratios > 0.58) & (ratios < 0.64)), (
        f"cube σ_ext/σ_sc = {ratios}; expected ≈0.61 (frequency-stable). "
        "Residual from absent modulus forward damping (2nd-order optical term); "
        "density radiation damping is now correct (Im[Γ₀] fix)."
    )


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

    sigma_sc = scattering_cross_section(
        c_inc, c_sc, REF, g, CONTRAST, omega, 10.0, k_vec, pol
    )
    assert sigma_sc > 0


def test_sv_scattering_nonzero():
    """P-wave incidence should produce nonzero SV scattering at off-axis angles."""
    omega, g, T27, k_vec, pol, c_inc, c_sc = _setup(0.05)

    theta = np.array([np.pi / 4])
    f_P, f_SV, f_SH = cube_far_field(
        c_inc, c_sc, theta, REF, g, CONTRAST, omega, 10.0, k_vec, pol
    )

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
        f_P, _, _ = cube_far_field(
            c_inc, c_sc, np.array([0.0]), REF, g, CONTRAST, omega, a, k_vec, pol
        )
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
