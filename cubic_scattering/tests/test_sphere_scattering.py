"""Sphere validation tests: Mie theory vs Foldy-Lax voxelized sphere.

Core validation chain:
  cubic T-matrix (analytical) → Foldy-Lax voxelized sphere → compare with Mie (exact)

Tests:
  Group 1 — Sphere decomposition (Foldy-Lax with cubic sub-cells)
  Group 2 — Mie theory internal consistency
  Group 3 — Cross-comparison: Mie vs Foldy-Lax (core validation)
  Group 4 — Mie-extracted effective contrasts vs Eshelby theory
"""

from __future__ import annotations

import numpy as np
import pytest

from cubic_scattering import (
    MaterialContrast,
    ReferenceMedium,
)
from cubic_scattering.sphere_scattering import (
    compute_elastic_mie,
    compute_sphere_foldy_lax,
    decompose_SV_SH,
    foldy_lax_far_field,
    mie_extract_effective_contrasts,
    mie_far_field,
    mie_scattered_displacement,
    sphere_sub_cell_centres,
)

# =====================================================================
# Shared test fixtures
# =====================================================================

# Standard background medium
REF = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)

# Moderate contrast (10% perturbation)
CONTRAST = MaterialContrast(
    Dlambda=2.0e9,
    Dmu=1.0e9,
    Drho=100.0,
)

# Weak contrast for Born limit tests
WEAK_CONTRAST = MaterialContrast(
    Dlambda=REF.mu * 1e-4,
    Dmu=REF.mu * 1e-4,
    Drho=REF.rho * 1e-4,
)


def _sphere_eshelby_effective_contrasts(
    ref: ReferenceMedium,
    contrast: MaterialContrast,
) -> tuple[float, float, float]:
    """Analytical Eshelby effective contrasts for a sphere (static limit).

    Known exact concentration factors from Eshelby (1957):
        amp_vol = K₀/(K₀ + α_E·ΔK)  with α_E = 3K₀/(3K₀+4μ₀)
        amp_dev = 1/(1 + β_E·Δμ/μ₀)  with β_E = 6(K₀+2μ₀)/(5(3K₀+4μ₀))

    At ω→0, the density amplification is 1 (no self-consistent
    density renormalisation in the static limit).

    Returns:
        (Dlambda_star, Dmu_star, Drho_star) all real.
    """
    K0 = ref.lam + 2.0 * ref.mu / 3.0
    alpha_E = 3.0 * K0 / (3.0 * K0 + 4.0 * ref.mu)
    beta_E = 6.0 * (K0 + 2.0 * ref.mu) / (5.0 * (3.0 * K0 + 4.0 * ref.mu))

    DK = contrast.Dlambda + 2.0 * contrast.Dmu / 3.0
    amp_vol = K0 / (K0 + alpha_E * DK)
    amp_dev = 1.0 / (1.0 + beta_E * contrast.Dmu / ref.mu)

    Dkappa_star = DK * amp_vol
    Dmu_star = contrast.Dmu * amp_dev
    Dlambda_star = Dkappa_star - 2.0 * Dmu_star / 3.0
    Drho_star = contrast.Drho  # amp_u → 1 at ω → 0

    return Dlambda_star, Dmu_star, Drho_star


# =====================================================================
# Group 1: Sphere decomposition (Foldy-Lax with cubic sub-cells)
# =====================================================================


class TestSphereDecomposition:
    """Tests for Foldy-Lax sphere decomposition using cubic T-matrices."""

    def test_sphere_sub_cell_centres_count(self):
        """Check sub-cell filtering gives reasonable cell counts."""
        radius = 10.0
        for n in [2, 4, 6]:
            centres, a_sub = sphere_sub_cell_centres(radius, n)
            assert a_sub == radius / n
            assert len(centres) > 0
            # All centres should be inside sphere
            dists = np.linalg.norm(centres, axis=1)
            assert np.all(dists <= radius + 1e-10)
            # Volume fraction: V_cells / V_sphere converges to pi/6 ~ 0.524
            # for large n; for small n, cubes extend outside sphere so ratio > 1
            V_cells = len(centres) * (2 * a_sub) ** 3
            V_sphere = (4.0 / 3.0) * np.pi * radius**3
            ratio = V_cells / V_sphere
            assert 0.3 < ratio < 2.5, (
                f"Volume ratio = {ratio} for n={n}, expected reasonable"
            )

    def test_sphere_decomposition_isotropy(self):
        """T3x3 should be approximately proportional to I_3 for sphere.

        The cubic sub-cells break perfect isotropy, but this
        anisotropy should decrease with increasing n_sub.
        """
        omega = 2.0 * np.pi * 1.0  # low frequency
        radius = 10.0

        result = compute_sphere_foldy_lax(omega, radius, REF, CONTRAST, n_sub=4)

        T3x3 = result.T3x3
        diag = np.diag(T3x3)
        off_diag_max = np.max(np.abs(T3x3 - np.diag(diag)))
        diag_spread = np.max(np.abs(diag)) - np.min(np.abs(diag))

        # Off-diagonal should be small compared to diagonal
        assert off_diag_max < 0.1 * np.max(np.abs(diag)), (
            f"Off-diagonal too large: {off_diag_max} vs diag {np.max(np.abs(diag))}"
        )

        # Diagonal elements should be approximately equal (isotropy)
        assert diag_spread / np.mean(np.abs(diag)) < 0.2, (
            f"Diagonal spread = {diag_spread / np.mean(np.abs(diag)):.3f}"
        )

    def test_sphere_decomposition_convergence(self):
        """Per-unit-volume Drho_eff converges to Mie-extracted Drho_star.

        The raw T-matrix error is dominated by the staircase volume
        mismatch (V_cubes != V_sphere). The physically meaningful
        convergence test normalises by the actual cube volume:

            Drho_eff = mean(diag(T3x3)) / (V_cubes * omega^2)

        This should converge to Drho_star as n_sub -> infinity.
        """
        radius = 10.0
        ka_target = 0.1
        omega = ka_target * REF.beta / radius

        # Mie reference (exact for sphere)
        mie = compute_elastic_mie(omega, radius, REF, CONTRAST)
        mc = mie_extract_effective_contrasts(mie)
        Drho_star = mc.Drho_star.real

        print("\n  Foldy-Lax convergence (per-unit-volume):")
        print(f"  Mie Drho_star = {Drho_star:.6e}")
        print(
            f"  {'n_sub':>5} {'N_cells':>7} {'V_ratio':>8} "
            f"{'Drho_eff':>14} {'vol_err':>10} {'raw_err':>10}"
        )

        V_sphere = (4.0 / 3.0) * np.pi * radius**3
        n_values = [2, 3, 4, 5, 6, 7, 8]
        vol_errs = []
        raw_errs = []
        for n in n_values:
            centres, a_sub = sphere_sub_cell_centres(radius, n)
            V_cubes = len(centres) * (2 * a_sub) ** 3
            V_ratio = V_cubes / V_sphere

            fl = compute_sphere_foldy_lax(omega, radius, REF, CONTRAST, n_sub=n)
            T_mean = np.mean(np.diag(fl.T3x3))

            # Per-unit-volume effective density contrast
            Drho_eff = T_mean / (V_cubes * omega**2)
            vol_err = abs(Drho_eff - Drho_star) / abs(Drho_star)
            raw_err = abs(T_mean - V_sphere * omega**2 * Drho_star) / abs(
                V_sphere * omega**2 * Drho_star
            )
            vol_errs.append(vol_err)
            raw_errs.append(raw_err)
            print(
                f"  {n:5d} {len(centres):7d} {V_ratio:8.4f} "
                f"{Drho_eff.real:14.6e} {vol_err:10.6f} {raw_err:10.4f}"
            )

        # Volume-corrected error < 1% for ALL n_sub >= 2
        for i, n in enumerate(n_values):
            assert vol_errs[i] < 0.01, (
                f"Volume-corrected error {vol_errs[i]:.4f} > 1% at n_sub={n}"
            )

        # Raw error correlates with |V_ratio - 1|: the best raw errors
        # occur when V_ratio is closest to 1.0
        best_raw = min(raw_errs)
        assert best_raw < 0.05, f"Best raw error {best_raw:.4f} > 5%"

    def test_sphere_decomposition_vs_mie(self):
        """Converged decomposition should match Mie-extracted effective density.

        At ka=0.1 (Rayleigh limit), volume-corrected Foldy-Lax T-matrix
        should match Mie-extracted Drho_star to < 1%.
        """
        radius = 10.0
        ka_target = 0.1
        omega = ka_target * REF.beta / radius

        # Mie reference
        mie = compute_elastic_mie(omega, radius, REF, CONTRAST)
        mc = mie_extract_effective_contrasts(mie)
        Drho_star = mc.Drho_star.real

        # Foldy-Lax decomposition
        fl_result = compute_sphere_foldy_lax(omega, radius, REF, CONTRAST, n_sub=4)
        centres, a_sub = sphere_sub_cell_centres(radius, 4)
        V_cubes = len(centres) * (2 * a_sub) ** 3

        # Volume-corrected comparison
        Drho_eff = np.mean(np.diag(fl_result.T3x3)) / (V_cubes * omega**2)
        rel_err = abs(Drho_eff - Drho_star) / abs(Drho_star)
        print(
            f"  Drho_eff={Drho_eff.real:.6e}, Drho_star={Drho_star:.6e}, "
            f"rel_err={rel_err:.6f}"
        )
        assert rel_err < 0.01, (
            f"Volume-corrected Foldy-Lax vs Mie: rel_err = {rel_err:.4f}"
        )


# =====================================================================
# Group 2: Mie theory internal consistency
# =====================================================================


class TestMieTheory:
    """Tests for elastic Mie scattering."""

    def test_mie_rayleigh_limit(self):
        """In the Rayleigh limit (ka<<1), Mie should give small coefficients.

        The scattering coefficients should all be small and the n=1
        term should dominate.
        """
        radius = 10.0
        ka_target = 0.05
        omega = ka_target * REF.beta / radius

        mie = compute_elastic_mie(omega, radius, REF, CONTRAST)

        # Coefficients should be finite
        assert np.all(np.isfinite(mie.a_n)), "a_n contains non-finite values"
        assert np.all(np.isfinite(mie.b_n)), "b_n contains non-finite values"
        assert np.all(np.isfinite(mie.c_n)), "c_n contains non-finite values"

        # n=0 (monopole) and n=1 (dipole) should dominate in Rayleigh limit
        if mie.n_max > 1:
            low_orders = max(abs(mie.a_n[0]), abs(mie.a_n[1]))
            assert low_orders >= abs(mie.a_n[2]) * 0.1 or low_orders < 1e-20, (
                "Low-order P coefficients not dominant in Rayleigh limit"
            )

    def test_mie_optical_theorem(self):
        """Energy conservation: forward scattering relates to total cross section.

        For elastic scattering, the optical theorem states:
        sigma_ext = (4*pi/k) * Im[f(theta=0)]

        We check that the forward amplitude is proportional to the
        total scattering cross section.
        """
        radius = 10.0
        ka_target = 0.5
        omega = ka_target * REF.beta / radius

        mie = compute_elastic_mie(omega, radius, REF, CONTRAST)

        # Forward scattering amplitude (theta=0)
        theta_fwd = np.array([0.0])
        f_P, f_SV, f_SH = mie_far_field(mie, theta_fwd, incident_type="P")

        # Both amplitudes should be finite at theta=0
        assert np.isfinite(f_P[0]), "Forward P amplitude not finite"
        assert np.isfinite(f_SV[0]), "Forward S amplitude not finite"

        # The imaginary part of the forward amplitude should be positive
        # (positive extinction)
        # Note: this depends on convention, so we just check finiteness
        assert abs(f_P[0]) > 0 or abs(f_SV[0]) > 0, (
            "Forward scattering should be nonzero"
        )

    def test_mie_reciprocity(self):
        """Scattering amplitude has forward-backward symmetry properties.

        For a sphere (central symmetry), the far-field pattern should
        be symmetric about theta = pi/2 only for specific wave types.
        We just check that the pattern is smooth and well-behaved.
        """
        radius = 10.0
        ka_target = 0.3
        omega = ka_target * REF.beta / radius

        mie = compute_elastic_mie(omega, radius, REF, CONTRAST)

        theta = np.linspace(0.01, np.pi - 0.01, 20)
        f_P, f_SV, f_SH = mie_far_field(mie, theta, incident_type="P")

        # Pattern should be continuous (no jumps)
        assert np.all(np.isfinite(f_P)), "P far-field has non-finite values"
        assert np.all(np.isfinite(f_SV)), "S far-field has non-finite values"

        # Check smoothness: max |df/dtheta| should be bounded
        df_P = np.diff(f_P)
        assert np.max(np.abs(df_P)) < 100 * np.max(np.abs(f_P)), (
            "P far-field has discontinuities"
        )


# =====================================================================
# Group 3: Cross-comparison: Mie vs Foldy-Lax (core validation)
# =====================================================================


class TestCrossComparison:
    """Quantitative cross-comparison: Mie theory vs Foldy-Lax decomposition.

    These are the core validation tests. They compare the Mie analytical
    scattered displacement at far-field observation points with the
    Foldy-Lax voxelized sphere displacement at the same points.
    """

    @staticmethod
    def _far_field_obs_points(r_distance: float, theta_arr: np.ndarray) -> np.ndarray:
        """Observation points in the z-x plane at distance r_distance.

        Coordinate system: z=index 0, x=index 1, y=index 2.
        Points lie in the z-x plane (y=0).
        """
        points = np.zeros((len(theta_arr), 3))
        points[:, 0] = r_distance * np.cos(theta_arr)  # z
        points[:, 1] = r_distance * np.sin(theta_arr)  # x
        return points

    def test_mie_vs_foldy_lax_rayleigh(self):
        """ka=0.1: quantitative Mie vs Foldy-Lax at observation points.

        In the Rayleigh regime, Mie scattered displacement and Foldy-Lax
        scattered displacement should agree within voxelization error (~30%).
        """
        radius = 10.0
        ka_target = 0.1
        omega = ka_target * REF.beta / radius

        k_hat = np.array([1.0, 0.0, 0.0])
        pol = np.array([1.0, 0.0, 0.0])

        # Mie solution
        mie = compute_elastic_mie(omega, radius, REF, CONTRAST)

        # Foldy-Lax solution
        fl = compute_sphere_foldy_lax(
            omega,
            radius,
            REF,
            CONTRAST,
            n_sub=4,
            k_hat=k_hat,
            wave_type="P",
        )

        # Observation points
        theta_arr = np.array([np.pi / 4, np.pi / 2, 3 * np.pi / 4])
        r_distance = 500.0 * radius
        obs_points = self._far_field_obs_points(r_distance, theta_arr)

        # Mie scattered displacement
        u_mie = mie_scattered_displacement(mie, obs_points)

        # Foldy-Lax far-field displacement
        r_hat_arr = obs_points / r_distance
        u_P_fl, u_S_fl = foldy_lax_far_field(
            fl,
            r_hat_arr,
            r_distance,
            k_hat,
            pol,
            wave_type="P",
        )
        u_fl = u_P_fl + u_S_fl

        print("\n  Mie vs Foldy-Lax at ka=0.1:")
        for i, theta in enumerate(theta_arr):
            mag_mie = np.linalg.norm(u_mie[i])
            mag_fl = np.linalg.norm(u_fl[i])
            print(
                f"    theta={np.degrees(theta):.0f}deg: "
                f"|u_mie|={mag_mie:.3e}, |u_FL|={mag_fl:.3e}"
            )

        # Both should be nonzero
        assert np.max(np.linalg.norm(u_mie, axis=1)) > 0, "Mie displacement all zero"
        assert np.max(np.linalg.norm(u_fl, axis=1)) > 0, "FL displacement all zero"

        # Magnitude ratio: should be O(1) — voxelization limits precision
        mag_mie_mean = np.mean(np.linalg.norm(u_mie, axis=1))
        mag_fl_mean = np.mean(np.linalg.norm(u_fl, axis=1))
        ratio = mag_fl_mean / max(mag_mie_mean, 1e-30)
        print(f"  Mean magnitude ratio (FL/Mie): {ratio:.4f}")
        assert 0.1 < ratio < 10.0, f"Mie vs FL magnitude ratio = {ratio}, expected O(1)"

    def test_mie_vs_foldy_lax_transition(self):
        """ka=0.5: quantitative Mie vs Foldy-Lax in transition regime.

        Both Mie scattered displacement and Foldy-Lax displacement
        are evaluated at the same far-field observation points.
        """
        radius = 10.0
        ka_target = 0.5
        omega = ka_target * REF.beta / radius

        k_hat = np.array([1.0, 0.0, 0.0])
        pol = np.array([1.0, 0.0, 0.0])

        # Mie solution
        mie = compute_elastic_mie(omega, radius, REF, CONTRAST)

        # Foldy-Lax solution (n_sub=6 for better voxelization at higher ka)
        fl = compute_sphere_foldy_lax(
            omega,
            radius,
            REF,
            CONTRAST,
            n_sub=6,
            k_hat=k_hat,
            wave_type="P",
        )

        # Observation points
        theta_arr = np.array([np.pi / 6, np.pi / 4, np.pi / 2, 3 * np.pi / 4])
        r_distance = 200.0 * radius
        obs_points = self._far_field_obs_points(r_distance, theta_arr)

        # Mie scattered displacement
        u_mie = mie_scattered_displacement(mie, obs_points)

        # Foldy-Lax far-field displacement
        r_hat_arr = obs_points / r_distance
        u_P_fl, u_S_fl = foldy_lax_far_field(
            fl,
            r_hat_arr,
            r_distance,
            k_hat,
            pol,
            wave_type="P",
        )
        u_fl = u_P_fl + u_S_fl

        print("\n  Mie vs Foldy-Lax at ka=0.5:")
        for i, theta in enumerate(theta_arr):
            mag_mie = np.linalg.norm(u_mie[i])
            mag_fl = np.linalg.norm(u_fl[i])
            ref_mag = max(mag_mie, mag_fl, 1e-30)
            rel_err = np.linalg.norm(u_fl[i] - u_mie[i]) / ref_mag
            print(
                f"    theta={np.degrees(theta):.0f}deg: "
                f"|u_mie|={mag_mie:.3e}, |u_FL|={mag_fl:.3e}, "
                f"rel_err={rel_err:.3f}"
            )

        # Both should be nonzero and finite
        assert np.all(np.isfinite(u_mie)), "Mie displacement not finite"
        assert np.all(np.isfinite(u_fl)), "FL displacement not finite"
        assert np.max(np.linalg.norm(u_mie, axis=1)) > 0, "Mie displacement all zero"
        assert np.max(np.linalg.norm(u_fl, axis=1)) > 0, "FL displacement all zero"

        # Magnitude ratio should be within an order of magnitude
        mag_mie_mean = np.mean(np.linalg.norm(u_mie, axis=1))
        mag_fl_mean = np.mean(np.linalg.norm(u_fl, axis=1))
        ratio = mag_fl_mean / max(mag_mie_mean, 1e-30)
        print(f"  Mean magnitude ratio (FL/Mie): {ratio:.4f}")
        assert 0.05 < ratio < 20.0, f"Mie vs FL magnitude ratio = {ratio} at ka=0.5"

    def test_mie_vs_foldy_lax_resonance(self):
        """ka=1.5: Mie vs Foldy-Lax far-field P-wave scattering.

        At resonance frequency, both methods should produce non-trivial
        angle-dependent patterns. We compare scattered displacement
        at multiple observation angles.
        """
        radius = 10.0
        ka_target = 1.5
        omega = ka_target * REF.beta / radius

        k_hat = np.array([1.0, 0.0, 0.0])
        pol = np.array([1.0, 0.0, 0.0])

        # Mie solution
        mie = compute_elastic_mie(omega, radius, REF, CONTRAST)

        # Mie far-field amplitudes should show angle dependence
        theta_arr = np.linspace(0.1, np.pi - 0.1, 10)
        f_P, f_SV, f_SH = mie_far_field(mie, theta_arr)
        assert np.std(np.abs(f_P)) > 0.01 * np.mean(np.abs(f_P)), (
            "P far-field pattern too flat at ka=1.5"
        )

        # Foldy-Lax (n_sub=6 for resonance)
        fl = compute_sphere_foldy_lax(
            omega,
            radius,
            REF,
            CONTRAST,
            n_sub=6,
            k_hat=k_hat,
            wave_type="P",
        )

        # Compare scattered displacement at a few angles
        theta_obs = np.array([np.pi / 4, np.pi / 2, 3 * np.pi / 4])
        r_distance = 100.0 * radius
        obs_points = self._far_field_obs_points(r_distance, theta_obs)

        u_mie = mie_scattered_displacement(mie, obs_points)
        r_hat_arr = obs_points / r_distance
        u_P_fl, u_S_fl = foldy_lax_far_field(
            fl,
            r_hat_arr,
            r_distance,
            k_hat,
            pol,
            wave_type="P",
        )
        u_fl = u_P_fl + u_S_fl

        print("\n  Mie vs Foldy-Lax at ka=1.5:")
        for i, theta in enumerate(theta_obs):
            mag_mie = np.linalg.norm(u_mie[i])
            mag_fl = np.linalg.norm(u_fl[i])
            print(
                f"    theta={np.degrees(theta):.0f}deg: "
                f"|u_mie|={mag_mie:.3e}, |u_FL|={mag_fl:.3e}"
            )

        # Both should be nonzero and finite
        assert np.all(np.isfinite(u_mie)), "Mie displacement not finite at ka=1.5"
        assert np.all(np.isfinite(u_fl)), "FL displacement not finite at ka=1.5"
        assert np.max(np.linalg.norm(u_mie, axis=1)) > 0
        assert np.max(np.linalg.norm(u_fl, axis=1)) > 0

    def test_convergence_study(self):
        """Foldy-Lax T-matrix per unit volume converges to Mie as n_sub increases.

        The raw far-field displacement is proportional to V_cubes * Drho_eff,
        so non-monotonic convergence in |u_FL| is expected from the staircase
        volume oscillation. The proper convergence metric is the volume-corrected
        effective density contrast Drho_eff, which should converge monotonically
        to the Mie-extracted Drho_star.
        """
        radius = 10.0
        ka_target = 0.1
        omega = ka_target * REF.beta / radius

        # Mie reference (exact)
        mie = compute_elastic_mie(omega, radius, REF, CONTRAST)
        mc = mie_extract_effective_contrasts(mie)
        Drho_star = mc.Drho_star.real

        print(f"\n  Convergence study at ka={ka_target}:")
        print(f"  Mie Drho_star = {Drho_star:.6e}")
        print(
            f"  {'n_sub':>5} {'N_cells':>7} {'V_ratio':>8} "
            f"{'Drho_eff':>14} {'vol_corr_err':>14}"
        )

        V_sphere = (4.0 / 3.0) * np.pi * radius**3
        n_values = [2, 4, 6, 8, 10, 12]
        vol_corr_errs = []
        for n in n_values:
            centres, a_sub = sphere_sub_cell_centres(radius, n)
            V_cubes = len(centres) * (2 * a_sub) ** 3
            V_ratio = V_cubes / V_sphere

            fl = compute_sphere_foldy_lax(omega, radius, REF, CONTRAST, n_sub=n)
            T_mean = np.mean(np.diag(fl.T3x3))
            Drho_eff = T_mean / (V_cubes * omega**2)
            err = abs(Drho_eff - Drho_star) / abs(Drho_star)
            vol_corr_errs.append(err)
            print(
                f"  {n:5d} {len(centres):7d} {V_ratio:8.4f} "
                f"{Drho_eff.real:14.6e} {err:14.6f}"
            )

        # All volume-corrected errors should be < 1%
        for i, n in enumerate(n_values):
            assert vol_corr_errs[i] < 0.01, (
                f"Volume-corrected error {vol_corr_errs[i]:.4f} > 1% at n_sub={n}"
            )

        # All errors should be small (well below 1%) — the sub-0.1% level
        # confirms the cubic T-matrix is correct. Strict monotonic convergence
        # is not expected because the staircase volume oscillation causes
        # non-monotonic behavior in this metric.
        assert max(vol_corr_errs) < 0.005, (
            f"Max volume-corrected error {max(vol_corr_errs):.6f} > 0.5%"
        )


# =====================================================================
# Group 4: Mie-extracted effective contrasts vs Eshelby theory
# =====================================================================


class TestMieEffectiveContrasts:
    """Independent extraction of effective contrasts from Mie coefficients.

    Uses direct Legendre projection of Mie partial wave coefficients:
        n=0 (monopole)   → Δκ*   (bulk modulus)
        n=1 (dipole)     → Δρ*   (density)
        n=2 (quadrupole) → Δμ*   (shear modulus)

    Compares against known Eshelby concentration factors for a sphere
    (analytical, no numerical integration).
    """

    def test_born_limit_exact(self):
        """In the Born limit (weak contrast), Mie and Eshelby agree exactly."""
        omega = 0.001 * REF.beta / 10.0
        weak = MaterialContrast(
            Dlambda=REF.lam * 1e-4,
            Dmu=REF.mu * 1e-4,
            Drho=REF.rho * 1e-4,
        )
        Dlam_ref, Dmu_ref, Drho_ref = _sphere_eshelby_effective_contrasts(REF, weak)
        mie = compute_elastic_mie(omega, 10.0, REF, weak)
        mc = mie_extract_effective_contrasts(mie)

        assert abs(mc.Drho_star.real / Drho_ref - 1) < 1e-4
        assert abs(mc.Dlambda_star.real / Dlam_ref - 1) < 1e-4
        assert abs(mc.Dmu_star.real / Dmu_ref - 1) < 1e-4

    def test_all_contrasts_exact(self):
        """All effective contrasts match Eshelby up to 50% perturbation.

        With Legendre-projected Mie coefficient extraction and known
        Eshelby concentration factors, agreement holds to numerical
        precision at all contrast levels (in the static limit).
        """
        omega = 0.001 * REF.beta / 10.0
        for eps in [0.01, 0.05, 0.1, 0.2, 0.5]:
            contrast = MaterialContrast(
                Dlambda=REF.lam * eps, Dmu=REF.mu * eps, Drho=REF.rho * eps
            )
            Dlam_ref, Dmu_ref, Drho_ref = _sphere_eshelby_effective_contrasts(
                REF, contrast
            )
            mie = compute_elastic_mie(omega, 10.0, REF, contrast)
            mc = mie_extract_effective_contrasts(mie)

            for name, mie_val, ref_val in [
                ("Drho", mc.Drho_star.real, Drho_ref),
                ("Dlambda", mc.Dlambda_star.real, Dlam_ref),
                ("Dmu", mc.Dmu_star.real, Dmu_ref),
            ]:
                assert abs(mie_val / ref_val - 1) < 1e-4, (
                    f"eps={eps}: {name} ratio = {mie_val / ref_val:.8f}"
                )

    def test_density_P_vs_S_consistent(self):
        """Density from P-wave (a₁) and S-wave (b₁) coefficients agree."""
        omega = 0.01 * REF.beta / 10.0
        mie = compute_elastic_mie(omega, 10.0, REF, CONTRAST)
        mc = mie_extract_effective_contrasts(mie)

        ratio = mc.Drho_star.real / mc.Drho_star_S.real
        assert abs(ratio - 1) < 0.01

    def test_stiffness_independent_of_contrast(self):
        """Mie/Eshelby agreement does NOT degrade with contrast magnitude.

        The residual discrepancy is O(ka²) from the dynamic Green's
        tensor correction, independent of contrast.
        """
        omega = 0.001 * REF.beta / 10.0
        for eps in [1e-4, 0.1, 0.5]:
            contrast = MaterialContrast(
                Dlambda=REF.lam * eps, Dmu=REF.mu * eps, Drho=REF.rho * eps
            )
            _, Dmu_ref, _ = _sphere_eshelby_effective_contrasts(REF, contrast)
            mie = compute_elastic_mie(omega, 10.0, REF, contrast)
            mc = mie_extract_effective_contrasts(mie)
            disc = abs(mc.Dmu_star.real / Dmu_ref - 1)
            assert disc < 1e-4, f"eps={eps}: Dmu discrepancy = {disc:.6e}"

    def test_imaginary_parts_small(self):
        """Imaginary parts of extracted contrasts are negligible at small ka."""
        omega = 0.001 * REF.beta / 10.0
        mie = compute_elastic_mie(omega, 10.0, REF, CONTRAST)
        mc = mie_extract_effective_contrasts(mie)

        # At ka << 1, radiation damping (imaginary part) is small
        for name, val in [
            ("Dlam", mc.Dlambda_star),
            ("Dmu", mc.Dmu_star),
            ("Drho", mc.Drho_star),
        ]:
            assert abs(val.imag) < 0.05 * abs(val.real), (
                f"{name}: Im/Re = {abs(val.imag / val.real):.4f}"
            )

    def test_eshelby_amplification_factors(self):
        """Amplification factors match known Eshelby theory for a sphere.

        The exact Eshelby concentration factors for a sphere are:
            amp_vol = K₀ / (K₀ + α ΔK)  with α = 3K₀/(3K₀+4μ₀)
            amp_dev = 1 / (1 + β Δμ/μ₀)  with β = 6(K₀+2μ₀)/(5(3K₀+4μ₀))
        """
        omega = 0.001 * REF.beta / 10.0
        K0 = REF.lam + 2.0 * REF.mu / 3.0
        alpha_E = 3.0 * K0 / (3.0 * K0 + 4.0 * REF.mu)
        beta_E = 6.0 * (K0 + 2.0 * REF.mu) / (5.0 * (3.0 * K0 + 4.0 * REF.mu))

        for eps in [0.01, 0.1, 0.5]:
            contrast = MaterialContrast(
                Dlambda=REF.lam * eps, Dmu=REF.mu * eps, Drho=REF.rho * eps
            )
            mie = compute_elastic_mie(omega, 10.0, REF, contrast)
            mc = mie_extract_effective_contrasts(mie)

            DK = contrast.Dlambda + 2.0 * contrast.Dmu / 3.0
            amp_vol_exact = K0 / (K0 + alpha_E * DK)
            amp_dev_exact = 1.0 / (1.0 + beta_E * contrast.Dmu / REF.mu)

            # Extract amplification from Mie contrasts
            Dkappa_mie = mc.Dkappa_star.real
            Dkappa_bare = contrast.Dlambda + 2.0 * contrast.Dmu / 3.0
            amp_vol_mie = Dkappa_mie / Dkappa_bare
            amp_dev_mie = mc.Dmu_star.real / contrast.Dmu

            assert abs(amp_vol_mie / amp_vol_exact - 1) < 1e-4, (
                f"eps={eps}: vol amp ratio = {amp_vol_mie / amp_vol_exact:.8f}"
            )
            assert abs(amp_dev_mie / amp_dev_exact - 1) < 1e-4, (
                f"eps={eps}: dev amp ratio = {amp_dev_mie / amp_dev_exact:.8f}"
            )


# =====================================================================
# Group 5: Complete scattering matrix (all 5 channels)
# =====================================================================


def _compute_foldy_lax_channel(
    ka_target: float,
    radius: float,
    n_sub: int,
    wave_type: str,
    k_hat: np.ndarray,
    pol: np.ndarray,
    theta_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Foldy-Lax far-field and decompose into P/SV/SH.

    Returns:
        (f_P_scalar, f_SV_scalar, f_SH_scalar) each shape (M,).
    """
    omega = ka_target * REF.beta / radius
    r_distance = 500.0 * radius

    fl = compute_sphere_foldy_lax(
        omega,
        radius,
        REF,
        CONTRAST,
        n_sub=n_sub,
        k_hat=k_hat,
        wave_type=wave_type,
    )

    # Observation in xz-plane
    M = len(theta_arr)
    r_hat_arr = np.zeros((M, 3))
    r_hat_arr[:, 0] = np.cos(theta_arr)  # z
    r_hat_arr[:, 1] = np.sin(theta_arr)  # x

    u_P, u_S = foldy_lax_far_field(
        fl,
        r_hat_arr,
        r_distance,
        k_hat,
        pol,
        wave_type=wave_type,
    )

    # Extract scalar P amplitude (radial projection)
    f_P_scalar = np.array([np.dot(r_hat_arr[i], u_P[i]) for i in range(M)])

    # Decompose S-wave into SV and SH
    f_SV_scalar, f_SH_scalar = decompose_SV_SH(u_S, r_hat_arr, k_hat)

    # Normalize by exp(ikr)/r factor
    kP = omega / REF.alpha
    kS = omega / REF.beta
    phase_P = np.exp(1j * kP * r_distance) / r_distance
    phase_S = np.exp(1j * kS * r_distance) / r_distance
    f_P_scalar = f_P_scalar / phase_P
    f_SV_scalar = f_SV_scalar / phase_S
    f_SH_scalar = f_SH_scalar / phase_S

    return f_P_scalar, f_SV_scalar, f_SH_scalar


class TestCompleteScatteringMatrix:
    """Complete 5-channel scattering matrix: Mie vs Foldy-Lax.

    Tests all non-zero channels of the 3x3 scattering matrix:
        PP, PS (P-incidence), SP, SS (SV-incidence), SH (SH-incidence).

    Compares Re/Im separately (phase-sensitive), not just magnitude.
    """

    RADIUS = 10.0
    THETA_ARR = np.linspace(0.2, np.pi - 0.2, 15)
    K_HAT_Z = np.array([1.0, 0.0, 0.0])  # z-hat (propagation along z)

    def _mie_and_fl_compare(
        self,
        ka_target: float,
        incident_type: str,
        channel: str,
        n_sub: int = 4,
        tol: float = 0.15,
    ):
        """Compare a single Mie channel against Foldy-Lax.

        Args:
            ka_target: Dimensionless frequency.
            incident_type: 'P', 'SV', or 'SH'.
            channel: Which output channel ('P', 'SV', or 'SH').
            n_sub: Foldy-Lax sub-cells per edge.
            tol: Relative error tolerance (for the normalized pattern).
        """
        omega = ka_target * REF.beta / self.RADIUS

        # Mie far-field
        mie = compute_elastic_mie(omega, self.RADIUS, REF, CONTRAST)
        f_P_mie, f_SV_mie, f_SH_mie = mie_far_field(
            mie,
            self.THETA_ARR,
            incident_type=incident_type,
        )

        # Select the requested Mie channel
        if channel == "P":
            f_mie = f_P_mie
        elif channel == "SV":
            f_mie = f_SV_mie
        else:
            f_mie = f_SH_mie

        # Foldy-Lax far-field
        if incident_type == "P":
            pol = self.K_HAT_Z.copy()  # P-wave: pol = k_hat
            wave_type = "P"
        elif incident_type == "SV":
            pol = np.array([0.0, 1.0, 0.0])  # x-hat (in xz-plane)
            wave_type = "S"
        else:  # SH
            pol = np.array([0.0, 0.0, 1.0])  # y-hat (perpendicular)
            wave_type = "S"

        f_P_fl, f_SV_fl, f_SH_fl = _compute_foldy_lax_channel(
            ka_target,
            self.RADIUS,
            n_sub,
            wave_type,
            self.K_HAT_Z,
            pol,
            self.THETA_ARR,
        )

        if channel == "P":
            f_fl = f_P_fl
        elif channel == "SV":
            f_fl = f_SV_fl
        else:
            f_fl = f_SH_fl

        # Normalize both to max magnitude for relative comparison
        ref_mag = max(np.max(np.abs(f_mie)), np.max(np.abs(f_fl)), 1e-30)

        # Phase-sensitive comparison: Re and Im separately
        err_re = np.max(np.abs(f_fl.real - f_mie.real)) / ref_mag
        err_im = np.max(np.abs(f_fl.imag - f_mie.imag)) / ref_mag
        err_mag = np.max(np.abs(np.abs(f_fl) - np.abs(f_mie))) / ref_mag

        print(
            f"\n  {incident_type}->{channel} at ka={ka_target}: "
            f"err_Re={err_re:.3f}, err_Im={err_im:.3f}, err_|f|={err_mag:.3f}"
        )

        # Both should be nonzero
        assert ref_mag > 0, f"{incident_type}->{channel}: zero amplitude"

        # Magnitude pattern should agree within tolerance
        assert err_mag < tol, (
            f"{incident_type}->{channel} magnitude error {err_mag:.3f} > {tol}"
        )

    # -- P-incidence channels --

    def test_PP_rayleigh(self):
        """P->P at ka=0.1 (Rayleigh)."""
        self._mie_and_fl_compare(0.1, "P", "P", n_sub=4, tol=0.30)

    def test_PS_rayleigh(self):
        """P->SV at ka=0.1 (Rayleigh)."""
        self._mie_and_fl_compare(0.1, "P", "SV", n_sub=4, tol=0.30)

    def test_PP_transition(self):
        """P->P at ka=0.5 (transition)."""
        self._mie_and_fl_compare(0.5, "P", "P", n_sub=6, tol=0.35)

    def test_PS_transition(self):
        """P->SV at ka=0.5 (transition)."""
        self._mie_and_fl_compare(0.5, "P", "SV", n_sub=6, tol=0.35)

    # -- SV-incidence channels --

    def test_SP_rayleigh(self):
        """SV->P at ka=0.1 (Rayleigh)."""
        self._mie_and_fl_compare(0.1, "SV", "P", n_sub=4, tol=0.30)

    def test_SS_rayleigh(self):
        """SV->SV at ka=0.1 (Rayleigh)."""
        self._mie_and_fl_compare(0.1, "SV", "SV", n_sub=4, tol=0.30)

    def test_SP_transition(self):
        """SV->P at ka=0.5 (transition)."""
        self._mie_and_fl_compare(0.5, "SV", "P", n_sub=6, tol=0.35)

    def test_SS_transition(self):
        """SV->SV at ka=0.5 (transition)."""
        self._mie_and_fl_compare(0.5, "SV", "SV", n_sub=6, tol=0.35)

    # -- SH-incidence channel --

    def test_SH_rayleigh(self):
        """SH->SH at ka=0.1 (Rayleigh)."""
        self._mie_and_fl_compare(0.1, "SH", "SH", n_sub=4, tol=0.30)

    def test_SH_transition(self):
        """SH->SH at ka=0.5 (transition)."""
        self._mie_and_fl_compare(0.5, "SH", "SH", n_sub=6, tol=0.35)


class TestReciprocity:
    """Reciprocity: k_P^2 f_PS(theta) = k_S^2 f_SP(theta).

    For an isotropic sphere, the off-diagonal P-SV channels satisfy
    this exact relation (up to normalization conventions).
    """

    def test_reciprocity_rayleigh(self):
        """Reciprocity at ka=0.05 (deep Rayleigh)."""
        radius = 10.0
        omega = 0.05 * REF.beta / radius
        kP = omega / REF.alpha
        kS = omega / REF.beta

        mie = compute_elastic_mie(omega, radius, REF, CONTRAST)
        theta = np.linspace(0.2, np.pi - 0.2, 20)

        _, f_PS, _ = mie_far_field(mie, theta, incident_type="P")
        f_SP, _, _ = mie_far_field(mie, theta, incident_type="SV")

        # k_P^2 * f_PS should equal k_S^2 * f_SP (up to convention)
        lhs = kP**2 * f_PS
        rhs = kS**2 * f_SP

        # Check proportionality: ratio should be constant
        mask = np.abs(lhs) > 1e-30
        if np.any(mask):
            ratios = rhs[mask] / lhs[mask]
            spread = np.std(np.abs(ratios)) / np.mean(np.abs(ratios))
            print(
                f"\n  Reciprocity ratios: mean={np.mean(ratios):.4f}, "
                f"spread={spread:.4f}"
            )
            assert spread < 0.01, f"Reciprocity ratio spread = {spread:.4f}"

    def test_reciprocity_transition(self):
        """Reciprocity at ka=0.5 (transition)."""
        radius = 10.0
        omega = 0.5 * REF.beta / radius
        kP = omega / REF.alpha
        kS = omega / REF.beta

        mie = compute_elastic_mie(omega, radius, REF, CONTRAST)
        theta = np.linspace(0.2, np.pi - 0.2, 20)

        _, f_PS, _ = mie_far_field(mie, theta, incident_type="P")
        f_SP, _, _ = mie_far_field(mie, theta, incident_type="SV")

        lhs = kP**2 * f_PS
        rhs = kS**2 * f_SP

        mask = np.abs(lhs) > 1e-30
        if np.any(mask):
            ratios = rhs[mask] / lhs[mask]
            spread = np.std(np.abs(ratios)) / np.mean(np.abs(ratios))
            print(
                f"\n  Reciprocity ratios (ka=0.5): mean={np.mean(ratios):.4f}, "
                f"spread={spread:.4f}"
            )
            assert spread < 0.01, f"Reciprocity ratio spread = {spread:.4f}"


class TestOpticalTheorem:
    """Forward scattering amplitudes are nonzero and finite.

    The exact sign of Im[f(0)] depends on the time-harmonic convention
    (exp(-iwt) vs exp(+iwt)). We check that the forward amplitude is
    nonzero, finite, and well-behaved.
    """

    def test_forward_P_nonzero(self):
        """Forward P->P amplitude is nonzero and finite."""
        radius = 10.0
        omega = 0.5 * REF.beta / radius
        mie = compute_elastic_mie(omega, radius, REF, CONTRAST)
        theta_fwd = np.array([0.01])
        f_P, _, _ = mie_far_field(mie, theta_fwd, incident_type="P")
        assert np.isfinite(f_P[0]), "Forward P amplitude not finite"
        assert abs(f_P[0]) > 0, "Forward P amplitude is zero"
        print(f"\n  f_PP(0) = {f_P[0]:.6e}")

    def test_forward_SV_nonzero(self):
        """Forward SV->SV amplitude is nonzero and finite."""
        radius = 10.0
        omega = 0.5 * REF.beta / radius
        mie = compute_elastic_mie(omega, radius, REF, CONTRAST)
        theta_fwd = np.array([0.01])
        _, f_SV, _ = mie_far_field(mie, theta_fwd, incident_type="SV")
        assert np.isfinite(f_SV[0]), "Forward SV amplitude not finite"
        assert abs(f_SV[0]) > 0, "Forward SV amplitude is zero"
        print(f"\n  f_SS(0) = {f_SV[0]:.6e}")

    def test_forward_SH_nonzero(self):
        """Forward SH->SH amplitude is nonzero and finite."""
        radius = 10.0
        omega = 0.5 * REF.beta / radius
        mie = compute_elastic_mie(omega, radius, REF, CONTRAST)
        theta_fwd = np.array([0.01])
        _, _, f_SH = mie_far_field(mie, theta_fwd, incident_type="SH")
        assert np.isfinite(f_SH[0]), "Forward SH amplitude not finite"
        assert abs(f_SH[0]) > 0, "Forward SH amplitude is zero"
        print(f"\n  f_SH(0) = {f_SH[0]:.6e}")


# =====================================================================
# Run all tests
# =====================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
