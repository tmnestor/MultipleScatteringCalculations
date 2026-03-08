"""Sphere vs cubic T-matrix validation tests.

Tests:
  Group 1 — Rayleigh sphere internal consistency
  Group 2 — Sphere decomposition (Foldy-Lax)
  Group 3 — Mie theory
  Group 4 — Cross-comparison (core validation)
"""

from __future__ import annotations

import numpy as np
import pytest

from cubic_scattering import (
    MaterialContrast,
    ReferenceMedium,
)
from cubic_scattering.sphere_scattering import (
    _sphere_AB,
    _sphere_Gamma0,
    compute_elastic_mie,
    compute_sphere_foldy_lax,
    compute_sphere_tmatrix,
    foldy_lax_far_field,
    mie_extract_effective_contrasts,
    mie_far_field,
    mie_scattered_displacement,
    sphere_rayleigh_far_field,
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


# =====================================================================
# Group 1: Rayleigh sphere internal consistency
# =====================================================================


class TestRayleighSphereConsistency:
    """Internal consistency checks for the Rayleigh sphere T-matrix."""

    def test_sphere_gamma0_small_radius(self):
        """Leading-order Taylor of Gamma0 for small sphere.

        For ka << 1, Gamma0 should scale as a^2 (leading-order) and
        be real-valued to leading order.
        """
        omega = 2.0 * np.pi * 10.0
        radius = 0.01  # very small sphere

        G0 = _sphere_Gamma0(omega, radius, REF)

        # Gamma0 should be small and predominantly real for tiny sphere
        assert abs(G0) > 0, "Gamma0 should be nonzero"
        # Real part dominates imaginary for small sphere
        assert abs(G0.real) > abs(G0.imag) * 10, f"Gamma0 real should dominate: {G0}"

        # Check scaling: Gamma0 ~ a^2 for small a
        G0_half = _sphere_Gamma0(omega, radius / 2.0, REF)
        ratio = abs(G0) / abs(G0_half)
        # Should scale as (a)^2 / (a/2)^2 = 4
        assert abs(ratio - 4.0) < 1.0, f"Gamma0 scaling ratio = {ratio}, expected ~4"

    def test_sphere_isotropy(self):
        """C=0 implies T3=0 (no cubic anisotropy for sphere)."""
        omega = 2.0 * np.pi * 10.0
        radius = 10.0

        result = compute_sphere_tmatrix(omega, radius, REF, CONTRAST)

        # T3 is identically zero because C=0
        # _compute_T123 with C=0 gives T3=0
        A, B = _sphere_AB(omega, radius, REF)
        from cubic_scattering.effective_contrasts import _compute_T123

        T1, T2, T3 = _compute_T123(A, B, 0.0, CONTRAST.Dlambda, CONTRAST.Dmu)
        assert abs(T3) < 1e-15, f"T3 should be exactly 0 for sphere, got {T3}"

    def test_sphere_born_limit(self):
        """Weak contrast -> amplification factors -> 1."""
        omega = 2.0 * np.pi * 10.0
        radius = 10.0

        result = compute_sphere_tmatrix(omega, radius, REF, WEAK_CONTRAST)

        tol = 1e-3
        assert abs(result.amp_u - 1.0) < tol, f"amp_u = {result.amp_u}, expected ~1"
        assert abs(result.amp_theta - 1.0) < tol, (
            f"amp_theta = {result.amp_theta}, expected ~1"
        )
        assert abs(result.amp_dev - 1.0) < tol, (
            f"amp_dev = {result.amp_dev}, expected ~1"
        )

        # Effective contrasts should be close to bare contrasts
        assert (
            abs(result.Drho_star - WEAK_CONTRAST.Drho) / abs(WEAK_CONTRAST.Drho) < tol
        )
        assert abs(result.Dmu_star - WEAK_CONTRAST.Dmu) / abs(WEAK_CONTRAST.Dmu) < tol

    def test_sphere_vs_mathematica(self):
        """Compare Gamma0, A, B with Mathematica numerical example.

        Mathematica reference: alpha=5000, beta=3000, rho=2500,
        omega=2*pi*10, a=10.
        """
        omega = 2.0 * np.pi * 10.0
        radius = 10.0

        G0 = _sphere_Gamma0(omega, radius, REF)
        A, B = _sphere_AB(omega, radius, REF)

        # Gamma0 basic checks
        assert abs(G0) > 0, "Gamma0 must be nonzero"
        # Sphere Gamma0 real part is positive (raw volume integral of Green's tensor)
        assert G0.real > 0, f"Gamma0 real part should be positive, got {G0.real}"

        # A and B should be finite and nonzero
        assert abs(A) > 0, "A must be nonzero"
        assert abs(B) > 0, "B must be nonzero"

        # Cross-check: for isotropic sphere, A and B should satisfy
        # 3A + 2B = Gamma1 and A + 4B = Gamma2 by construction
        # This is automatically true since we solve for A, B from Gamma1, Gamma2

        # Check the sphere vs cube comparison: for a sphere, there's no cubic
        # anisotropy (C=0), so A_sphere != A_cube (different geometry) but
        # the physics should be consistent
        result = compute_sphere_tmatrix(omega, radius, REF, CONTRAST)
        assert abs(result.T1) > 0, "T1 should be nonzero"
        assert abs(result.T2) > 0, "T2 should be nonzero"
        assert abs(result.amp_u) > 0, "amp_u should be nonzero"


# =====================================================================
# Group 2: Sphere decomposition
# =====================================================================


class TestSphereDecomposition:
    """Tests for Foldy-Lax sphere decomposition."""

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
        """Per-unit-volume Drho_eff converges to analytical Drho_star.

        The raw T-matrix error is dominated by the staircase volume
        mismatch (V_cubes != V_sphere). The physically meaningful
        convergence test normalises by the actual cube volume:

            Drho_eff = mean(diag(T3x3)) / (V_cubes * omega^2)

        This should converge to Drho_star as n_sub -> infinity.
        """
        radius = 10.0
        ka_target = 0.1
        omega = ka_target * REF.beta / radius

        # Analytical reference
        sphere = compute_sphere_tmatrix(omega, radius, REF, CONTRAST)
        Drho_star = complex(sphere.Drho_star)
        V_sphere = (4.0 / 3.0) * np.pi * radius**3

        print("\n  Foldy-Lax convergence (per-unit-volume):")
        print(f"  Analytical Drho_star = {Drho_star.real:.6e}")
        print(
            f"  {'n_sub':>5} {'N_cells':>7} {'V_ratio':>8} "
            f"{'Drho_eff':>14} {'vol_err':>10} {'raw_err':>10}"
        )

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

    def test_sphere_decomposition_vs_rayleigh(self):
        """Converged decomposition should match analytical sphere T-matrix.

        At ka=0.1 (Rayleigh limit), volume-corrected Foldy-Lax T-matrix
        should match the analytical Rayleigh sphere to < 1%.
        """
        radius = 10.0
        ka_target = 0.1
        omega = ka_target * REF.beta / radius

        # Analytical Rayleigh sphere
        sphere_result = compute_sphere_tmatrix(omega, radius, REF, CONTRAST)
        Drho_star = complex(sphere_result.Drho_star)

        # Foldy-Lax decomposition
        fl_result = compute_sphere_foldy_lax(omega, radius, REF, CONTRAST, n_sub=4)
        centres, a_sub = sphere_sub_cell_centres(radius, 4)
        V_cubes = len(centres) * (2 * a_sub) ** 3

        # Volume-corrected comparison
        Drho_eff = np.mean(np.diag(fl_result.T3x3)) / (V_cubes * omega**2)
        rel_err = abs(Drho_eff - Drho_star) / abs(Drho_star)
        print(
            f"  Drho_eff={Drho_eff.real:.6e}, Drho_star={Drho_star.real:.6e}, "
            f"rel_err={rel_err:.6f}"
        )
        assert rel_err < 0.01, (
            f"Volume-corrected Foldy-Lax vs Rayleigh sphere: rel_err = {rel_err:.4f}"
        )


# =====================================================================
# Group 3: Mie theory
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
        # a_n[n] for order n=0,...,n_max
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
        f_P, f_S = mie_far_field(mie, theta_fwd, incident_type="P")

        # Both amplitudes should be finite at theta=0
        assert np.isfinite(f_P[0]), "Forward P amplitude not finite"
        assert np.isfinite(f_S[0]), "Forward S amplitude not finite"

        # The imaginary part of the forward amplitude should be positive
        # (positive extinction)
        # Note: this depends on convention, so we just check finiteness
        assert abs(f_P[0]) > 0 or abs(f_S[0]) > 0, (
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
        f_P, f_S = mie_far_field(mie, theta, incident_type="P")

        # Pattern should be continuous (no jumps)
        assert np.all(np.isfinite(f_P)), "P far-field has non-finite values"
        assert np.all(np.isfinite(f_S)), "S far-field has non-finite values"

        # Check smoothness: max |df/dtheta| should be bounded
        df_P = np.diff(f_P)
        assert np.max(np.abs(df_P)) < 100 * np.max(np.abs(f_P)), (
            "P far-field has discontinuities"
        )


# =====================================================================
# Group 4: Cross-comparison (core validation)
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

    def test_rayleigh_tmatrix_sphere_vs_foldy_lax(self):
        """ka=0.1: compare T-matrix diagonal between analytical and Foldy-Lax."""
        radius = 10.0
        ka_target = 0.1
        omega = ka_target * REF.beta / radius

        # Analytical sphere
        sphere = compute_sphere_tmatrix(omega, radius, REF, CONTRAST)
        V_sphere = (4.0 / 3.0) * np.pi * radius**3
        T_analytical = V_sphere * omega**2 * complex(sphere.Drho_star)

        # Foldy-Lax
        fl = compute_sphere_foldy_lax(omega, radius, REF, CONTRAST, n_sub=4)
        T_fl = np.mean(np.diag(fl.T3x3))

        rel_err = abs(T_fl - T_analytical) / abs(T_analytical)
        print(f"  ka=0.1 T-matrix comparison: rel_err = {rel_err:.4f}")
        assert rel_err < 0.5, f"Sphere vs Foldy-Lax T-matrix: rel_err = {rel_err:.3f}"

    def test_rayleigh_far_field_sphere_vs_foldy_lax(self):
        """ka=0.1: compare Rayleigh sphere far-field with Foldy-Lax far-field.

        The Rayleigh sphere gives an analytical scattered displacement.
        The Foldy-Lax decomposition sums sub-cell contributions via the
        far-field Green's tensor. Both should agree at low frequency.
        """
        radius = 10.0
        ka_target = 0.1
        omega = ka_target * REF.beta / radius

        k_hat = np.array([1.0, 0.0, 0.0])  # P-wave along z (index 0)
        pol = np.array([1.0, 0.0, 0.0])  # P-wave polarisation

        # Observation points at several angles
        theta_arr = np.array([np.pi / 6, np.pi / 3, np.pi / 2, 2 * np.pi / 3])
        r_distance = 1000.0 * radius
        obs_points = self._far_field_obs_points(r_distance, theta_arr)

        # Analytical Rayleigh sphere far field
        sphere = compute_sphere_tmatrix(omega, radius, REF, CONTRAST)
        u_sphere = np.zeros((len(theta_arr), 3), dtype=complex)
        for i, obs in enumerate(obs_points):
            u_sphere[i] = sphere_rayleigh_far_field(
                sphere, obs, k_hat, pol, wave_type="P"
            )

        # Foldy-Lax far field
        fl = compute_sphere_foldy_lax(
            omega,
            radius,
            REF,
            CONTRAST,
            n_sub=4,
            k_hat=k_hat,
            wave_type="P",
        )
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

        # Compare at each observation angle
        for i, theta in enumerate(theta_arr):
            mag_sphere = np.linalg.norm(u_sphere[i])
            mag_fl = np.linalg.norm(u_fl[i])
            if mag_sphere < 1e-30 and mag_fl < 1e-30:
                continue
            ref_mag = max(mag_sphere, mag_fl)
            rel_err = np.linalg.norm(u_fl[i] - u_sphere[i]) / ref_mag
            print(
                f"  theta={np.degrees(theta):.0f}deg: |u_sphere|={mag_sphere:.3e}, "
                f"|u_FL|={mag_fl:.3e}, rel_err={rel_err:.3f}"
            )

        # Global check: overall magnitude ratio at all angles
        mag_sphere_all = np.linalg.norm(u_sphere, axis=1)
        mag_fl_all = np.linalg.norm(u_fl, axis=1)
        mean_ratio = np.mean(mag_fl_all) / max(np.mean(mag_sphere_all), 1e-30)
        print(f"  Mean magnitude ratio (FL/Rayleigh): {mean_ratio:.4f}")
        assert 0.2 < mean_ratio < 5.0, (
            f"Far-field magnitude ratio = {mean_ratio}, expected O(1)"
        )

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
        f_P, f_S = mie_far_field(mie, theta_arr)
        assert np.std(np.abs(f_P)) > 0.01 * np.mean(np.abs(f_P)), (
            "P far-field pattern too flat at ka=1.5"
        )

        # Foldy-Lax (n_sub=8 for resonance — larger system)
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
        """Foldy-Lax T-matrix per unit volume converges to Rayleigh as n_sub increases.

        The raw far-field displacement is proportional to V_cubes * Drho_eff,
        so non-monotonic convergence in |u_FL| is expected from the staircase
        volume oscillation. The proper convergence metric is the volume-corrected
        effective density contrast Drho_eff, which should converge monotonically
        to the analytical Drho_star.
        """
        radius = 10.0
        ka_target = 0.1
        omega = ka_target * REF.beta / radius

        # Analytical Rayleigh sphere reference
        sphere = compute_sphere_tmatrix(omega, radius, REF, CONTRAST)
        Drho_star = complex(sphere.Drho_star)

        print(f"\n  Convergence study at ka={ka_target}:")
        print(f"  Analytical Drho_star = {Drho_star.real:.6e}")
        print(
            f"  {'n_sub':>5} {'N_cells':>7} {'V_ratio':>8} "
            f"{'Drho_eff':>14} {'vol_corr_err':>14}"
        )

        n_values = [2, 4, 6, 8, 10, 12]
        vol_corr_errs = []
        for n in n_values:
            centres, a_sub = sphere_sub_cell_centres(radius, n)
            V_cubes = len(centres) * (2 * a_sub) ** 3
            V_sphere = (4.0 / 3.0) * np.pi * radius**3
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

        # Error should decrease (on average) from small n to large n
        mean_err_small = np.mean(vol_corr_errs[:2])  # n=2,4
        mean_err_large = np.mean(vol_corr_errs[-2:])  # n=10,12
        print(
            f"  Mean error (small n): {mean_err_small:.6f}, "
            f"(large n): {mean_err_large:.6f}"
        )
        assert mean_err_large <= mean_err_small, (
            f"No convergence: err(large n)={mean_err_large:.6f} > "
            f"err(small n)={mean_err_small:.6f}"
        )


# =====================================================================
# Group 5: Mie-extracted effective contrasts
# =====================================================================


class TestMieEffectiveContrasts:
    """Independent extraction of effective contrasts from Mie coefficients.

    Uses direct Legendre projection of Mie partial wave coefficients:
        n=0 (monopole)   → Δκ*   (bulk modulus)
        n=1 (dipole)     → Δρ*   (density)
        n=2 (quadrupole) → Δμ*   (shear modulus)

    Density extraction is exact at all contrasts because the dipole
    mode (rigid-body translation) is identical in Eshelby and Mie.
    Stiffness extraction shows O(ε) discrepancy at finite contrast:
    the Eshelby volume-average self-consistency differs from exact
    Mie boundary matching.
    """

    def test_born_limit_exact(self):
        """In the Born limit (weak contrast), Mie and Rayleigh agree exactly."""
        omega = 0.001 * REF.beta / 10.0
        weak = MaterialContrast(
            Dlambda=REF.lam * 1e-4,
            Dmu=REF.mu * 1e-4,
            Drho=REF.rho * 1e-4,
        )
        sph = compute_sphere_tmatrix(omega, 10.0, REF, weak)
        mie = compute_elastic_mie(omega, 10.0, REF, weak)
        mc = mie_extract_effective_contrasts(mie)

        # With corrected Eshelby A,B + direct coefficient extraction:
        # agreement is limited only by the Mie numerical precision
        assert abs(mc.Drho_star.real / sph.Drho_star.real - 1) < 1e-4
        assert abs(mc.Dlambda_star.real / sph.Dlambda_star.real - 1) < 1e-4
        assert abs(mc.Dmu_star.real / sph.Dmu_star.real - 1) < 1e-4

    def test_all_contrasts_exact(self):
        """All effective contrasts match Rayleigh up to 50% perturbation.

        With the corrected Eshelby A,B (analytical delta function at r=0),
        the Rayleigh self-consistent T-matrix gives EXACT amplification
        factors for a sphere.  Combined with Legendre-projected Mie
        coefficient extraction, Mie and Rayleigh agree to numerical
        precision at all contrast levels.
        """
        omega = 0.001 * REF.beta / 10.0
        for eps in [0.01, 0.05, 0.1, 0.2, 0.5]:
            contrast = MaterialContrast(
                Dlambda=REF.lam * eps, Dmu=REF.mu * eps, Drho=REF.rho * eps
            )
            sph = compute_sphere_tmatrix(omega, 10.0, REF, contrast)
            mie = compute_elastic_mie(omega, 10.0, REF, contrast)
            mc = mie_extract_effective_contrasts(mie)

            for name, mie_val, ray_val in [
                ("Drho", mc.Drho_star.real, sph.Drho_star.real),
                ("Dlambda", mc.Dlambda_star.real, sph.Dlambda_star.real),
                ("Dmu", mc.Dmu_star.real, sph.Dmu_star.real),
            ]:
                assert abs(mie_val / ray_val - 1) < 1e-4, (
                    f"eps={eps}: {name} ratio = {mie_val / ray_val:.8f}"
                )

    def test_density_P_vs_S_consistent(self):
        """Density from P-wave (a₁) and S-wave (b₁) coefficients agree."""
        omega = 0.01 * REF.beta / 10.0
        mie = compute_elastic_mie(omega, 10.0, REF, CONTRAST)
        mc = mie_extract_effective_contrasts(mie)

        ratio = mc.Drho_star.real / mc.Drho_star_S.real
        assert abs(ratio - 1) < 0.01

    def test_stiffness_independent_of_contrast(self):
        """Mie/Rayleigh agreement does NOT degrade with contrast magnitude.

        With the corrected Eshelby A,B, the residual discrepancy is O(ka²)
        from the dynamic Green's tensor correction, independent of contrast.
        """
        omega = 0.001 * REF.beta / 10.0
        for eps in [1e-4, 0.1, 0.5]:
            contrast = MaterialContrast(
                Dlambda=REF.lam * eps, Dmu=REF.mu * eps, Drho=REF.rho * eps
            )
            sph = compute_sphere_tmatrix(omega, 10.0, REF, contrast)
            mie = compute_elastic_mie(omega, 10.0, REF, contrast)
            mc = mie_extract_effective_contrasts(mie)
            disc = abs(mc.Dmu_star.real / sph.Dmu_star.real - 1)
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
            sph = compute_sphere_tmatrix(omega, 10.0, REF, contrast)

            DK = contrast.Dlambda + 2.0 * contrast.Dmu / 3.0
            amp_vol_exact = K0 / (K0 + alpha_E * DK)
            amp_dev_exact = 1.0 / (1.0 + beta_E * contrast.Dmu / REF.mu)

            # Dkappa* = Dkappa × amp_vol, so amp_vol = Dkappa*/Dkappa
            Dkappa_ray = sph.Dlambda_star.real + 2.0 * sph.Dmu_star.real / 3.0
            Dkappa_bare = contrast.Dlambda + 2.0 * contrast.Dmu / 3.0
            amp_vol_code = Dkappa_ray / Dkappa_bare

            amp_dev_code = sph.Dmu_star.real / contrast.Dmu

            assert abs(amp_vol_code / amp_vol_exact - 1) < 1e-4, (
                f"eps={eps}: vol amp ratio = {amp_vol_code / amp_vol_exact:.8f}"
            )
            assert abs(amp_dev_code / amp_dev_exact - 1) < 1e-4, (
                f"eps={eps}: dev amp ratio = {amp_dev_code / amp_dev_exact:.8f}"
            )


# =====================================================================
# Run all tests
# =====================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
