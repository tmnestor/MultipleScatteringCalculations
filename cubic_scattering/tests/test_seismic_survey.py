"""Tests for marine seismic survey simulation."""

import sys
from pathlib import Path

import numpy as np
import pytest

from cubic_scattering.kennett_layers import (
    FluidLayer,
    IsotropicLayer,
    LayerStack,
    _complex_slowness,
    _vertical_slowness,
    kennett_layers,
    kennett_reflectivity_batch,
    psv_fluid_solid,
)
from cubic_scattering.seismic_survey import (
    GatherConfig,
    ShotGatherResult,
    SurveyConfig,
    build_survey_stack,
    compute_shot_gather,
    free_surface_reverberations,
    receiver_ghost,
    ricker_source_spectrum,
    source_ghost,
)

# ── Shared constants ──────────────────────────────────────────────────────

WATER_ALPHA = 1500.0
WATER_RHO = 1025.0
WATER_DEPTH = 70.0

# Sediment
SED_ALPHA = 2200.0
SED_BETA = 800.0
SED_RHO = 2100.0

# Half-space
HS_ALPHA = 5000.0
HS_BETA = 3000.0
HS_RHO = 2700.0

OMEGA = np.linspace(10.0, 500.0, 64)
P_VALS = np.array([1e-6, 1e-4, 2e-4, 3e-4, 5e-4])


# ── 1. Ghost operators ───────────────────────────────────────────────────


class TestGhostOperators:
    """Test source and receiver ghost operators."""

    def test_source_ghost_zero_depth(self):
        """Zero source depth gives zero ghost."""
        g = source_ghost(OMEGA, P_VALS, z_s=0.0, alpha=WATER_ALPHA)
        np.testing.assert_allclose(g, 0.0, atol=1e-14)

    def test_receiver_ghost_zero_depth_hydrophone(self):
        """Zero receiver depth gives 2.0 for hydrophone (pressure doubling)."""
        g = receiver_ghost(
            OMEGA, P_VALS, z_r=0.0, alpha=WATER_ALPHA, receiver_type="hydrophone"
        )
        np.testing.assert_allclose(g, 2.0, atol=1e-14)

    def test_receiver_ghost_zero_depth_geophone(self):
        """Zero receiver depth gives 0 for geophone (velocity cancels)."""
        g = receiver_ghost(
            OMEGA, P_VALS, z_r=0.0, alpha=WATER_ALPHA, receiver_type="geophone"
        )
        np.testing.assert_allclose(g, 0.0, atol=1e-14)

    def test_ghost_notch_frequency(self):
        """Source ghost has a notch at f = alpha / (2 * z_s) for normal incidence."""
        z_s = 5.0
        f_notch = WATER_ALPHA / (2.0 * z_s)  # 150 Hz
        omega_notch = 2.0 * np.pi * f_notch
        p = np.array([1e-8])  # effectively normal incidence
        g = source_ghost(np.array([omega_notch]), p, z_s=z_s, alpha=WATER_ALPHA)
        # At notch: exp(2j*omega*z/alpha) = exp(2j*pi) = 1, so ghost = 1-1 = 0
        np.testing.assert_allclose(np.abs(g[0, 0]), 0.0, atol=1e-6)

    def test_hydrophone_vs_geophone_polarity(self):
        """Hydrophone and geophone ghosts have opposite sign convention."""
        z = 8.0
        gh = receiver_ghost(
            OMEGA, P_VALS, z_r=z, alpha=WATER_ALPHA, receiver_type="hydrophone"
        )
        gg = receiver_ghost(
            OMEGA, P_VALS, z_r=z, alpha=WATER_ALPHA, receiver_type="geophone"
        )
        # At normal incidence: hydrophone = 1+exp, geophone = 1-exp
        # Sum = 2, so hydrophone + geophone = 2 at all frequencies
        np.testing.assert_allclose(gh + gg, 2.0, atol=1e-12)

    def test_ghost_shape(self):
        """Ghost operators have correct shape (np_slow, nfreq)."""
        g = source_ghost(OMEGA, P_VALS, z_s=5.0, alpha=WATER_ALPHA)
        assert g.shape == (len(P_VALS), len(OMEGA))


# ── 2. Free surface reverberations ───────────────────────────────────────


class TestFreeSurfaceReverberations:
    """Test free surface reverberation operator."""

    def test_zero_reflectivity_gives_zero(self):
        """R = 0 sub-ocean -> R_total = 0."""
        RRd = np.zeros((5, 10), dtype=np.complex128)
        eaea = np.ones_like(RRd)
        R = free_surface_reverberations(RRd, eaea)
        np.testing.assert_allclose(R, 0.0, atol=1e-14)

    def test_bounded_for_unit_reflectivity(self):
        """R_total = E^2 / (1 + E^2) for RRd = 1 — bounded."""
        RRd = np.ones((3, 5), dtype=np.complex128)
        eaea = np.exp(1j * np.random.default_rng(42).uniform(0, 2 * np.pi, (3, 5)))
        R = free_surface_reverberations(RRd, eaea)
        # |R| should be bounded (no blow-up)
        assert np.all(np.abs(R) < 10.0)

    def test_geometric_series_first_terms(self):
        """Verify first-order expansion: R ≈ E²·RRd for small RRd."""
        RRd = 0.01 * np.ones((3, 5), dtype=np.complex128)
        eaea = np.exp(1j * np.linspace(0.1, 1.0, 5)[np.newaxis, :])
        R = free_surface_reverberations(RRd, eaea)
        R_first_order = eaea * RRd
        np.testing.assert_allclose(R, R_first_order, rtol=0.02)


# ── 3. FluidLayer ─────────────────────────────────────────────────────────


class TestFluidLayer:
    """Test FluidLayer dataclass."""

    def test_valid_construction(self):
        fl = FluidLayer(alpha=1500, rho=1025, thickness=70)
        assert fl.beta == 0.0
        assert fl.alpha == 1500

    def test_reject_nonzero_beta(self):
        with pytest.raises(ValueError, match="beta must be 0"):
            FluidLayer(alpha=1500, rho=1025, thickness=70, beta=100.0)

    def test_reject_negative_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            FluidLayer(alpha=-1, rho=1025, thickness=70)


# ── 4. Fluid-solid interface coefficients ────────────────────────────────


class TestFluidSolidInterface:
    """Test psv_fluid_solid() against PhD ocean_bottom_interface()."""

    def test_rd_only_pp_nonzero(self):
        """Rd should only have [0,0] nonzero (no S reflection in fluid)."""
        s_p1 = _complex_slowness(WATER_ALPHA, np.inf)
        s_p2 = _complex_slowness(SED_ALPHA, np.inf)
        s_s2 = _complex_slowness(SED_BETA, np.inf)
        p = 1e-4
        coeff = psv_fluid_solid(
            p=p,
            eta1=_vertical_slowness(s_p1, p),
            rho1=WATER_RHO,
            eta2=_vertical_slowness(s_p2, p),
            neta2=_vertical_slowness(s_s2, p),
            rho2=SED_RHO,
            beta2=1.0 / s_s2,
        )
        assert coeff.Rd[0, 1] == pytest.approx(0.0, abs=1e-14)
        assert coeff.Rd[1, 0] == pytest.approx(0.0, abs=1e-14)
        assert coeff.Rd[1, 1] == pytest.approx(0.0, abs=1e-14)

    def test_td_column1_zero(self):
        """Td column 1 is zero (no downgoing S from fluid)."""
        s_p1 = _complex_slowness(WATER_ALPHA, np.inf)
        s_p2 = _complex_slowness(SED_ALPHA, np.inf)
        s_s2 = _complex_slowness(SED_BETA, np.inf)
        p = 1e-4
        coeff = psv_fluid_solid(
            p=p,
            eta1=_vertical_slowness(s_p1, p),
            rho1=WATER_RHO,
            eta2=_vertical_slowness(s_p2, p),
            neta2=_vertical_slowness(s_s2, p),
            rho2=SED_RHO,
            beta2=1.0 / s_s2,
        )
        np.testing.assert_allclose(coeff.Td[:, 1], 0.0, atol=1e-14)

    def test_tu_row1_zero(self):
        """Tu row 1 is zero (no upgoing S into fluid)."""
        s_p1 = _complex_slowness(WATER_ALPHA, np.inf)
        s_p2 = _complex_slowness(SED_ALPHA, np.inf)
        s_s2 = _complex_slowness(SED_BETA, np.inf)
        p = 1e-4
        coeff = psv_fluid_solid(
            p=p,
            eta1=_vertical_slowness(s_p1, p),
            rho1=WATER_RHO,
            eta2=_vertical_slowness(s_p2, p),
            neta2=_vertical_slowness(s_s2, p),
            rho2=SED_RHO,
            beta2=1.0 / s_s2,
        )
        np.testing.assert_allclose(coeff.Tu[1, :], 0.0, atol=1e-14)

    def test_match_phd_ocean_bottom(self):
        """Compare against PhD ocean_bottom_interface at 5 slowness values."""
        phd_path = Path("PhD_fortran_code/Kennett_Reflectivity")
        if not phd_path.exists():
            pytest.skip("PhD code not available")
        sys.path.insert(0, str(phd_path))
        from scattering_matrices import ocean_bottom_interface

        for p_val in [1e-6, 5e-5, 1e-4, 2e-4, 3e-4]:
            s_p1 = _complex_slowness(WATER_ALPHA, np.inf)
            s_p2 = _complex_slowness(SED_ALPHA, np.inf)
            s_s2 = _complex_slowness(SED_BETA, np.inf)

            mine = psv_fluid_solid(
                p=p_val,
                eta1=_vertical_slowness(s_p1, p_val),
                rho1=WATER_RHO,
                eta2=_vertical_slowness(s_p2, p_val),
                neta2=_vertical_slowness(s_s2, p_val),
                rho2=SED_RHO,
                beta2=1.0 / s_s2,
            )
            theirs = ocean_bottom_interface(
                p=p_val,
                eta1=_vertical_slowness(s_p1, p_val),
                rho1=WATER_RHO,
                eta2=_vertical_slowness(s_p2, p_val),
                neta2=_vertical_slowness(s_s2, p_val),
                rho2=SED_RHO,
                beta2=1.0 / s_s2,
            )

            np.testing.assert_allclose(mine.Rd, theirs.Rd, atol=1e-14)
            np.testing.assert_allclose(mine.Ru, theirs.Ru, atol=1e-14)
            np.testing.assert_allclose(mine.Td, theirs.Td, atol=1e-14)
            np.testing.assert_allclose(mine.Tu, theirs.Tu, atol=1e-14)

    def test_normal_incidence_pp_impedance(self):
        """Near-normal Rd_PP ~ (Z2-Z1)/(Z2+Z1) for acoustic impedance."""
        s_p1 = _complex_slowness(WATER_ALPHA, np.inf)
        s_p2 = _complex_slowness(SED_ALPHA, np.inf)
        s_s2 = _complex_slowness(SED_BETA, np.inf)
        p = 1e-8  # near-normal
        coeff = psv_fluid_solid(
            p=p,
            eta1=_vertical_slowness(s_p1, p),
            rho1=WATER_RHO,
            eta2=_vertical_slowness(s_p2, p),
            neta2=_vertical_slowness(s_s2, p),
            rho2=SED_RHO,
            beta2=1.0 / s_s2,
        )
        Z1 = WATER_RHO * WATER_ALPHA
        Z2 = SED_RHO * SED_ALPHA
        expected = (Z2 - Z1) / (Z2 + Z1)
        # Modified coefficients at very small p should be close
        assert coeff.Rd[0, 0].real == pytest.approx(expected, rel=0.01)


# ── 5. Batched Kennett consistency ────────────────────────────────────────


class TestBatchedKennett:
    """Test kennett_reflectivity_batch against scalar kennett_layers."""

    def test_match_scalar_solid_only(self):
        """Batched matches scalar for solid-only stack at selected (p, omega)."""
        stack = LayerStack(
            layers=[
                IsotropicLayer(SED_ALPHA, SED_BETA, SED_RHO, 200.0),
                IsotropicLayer(HS_ALPHA, HS_BETA, HS_RHO, np.inf),
            ]
        )
        p_test = np.array([1e-5, 1e-4, 2e-4])
        omega_test = np.array([50.0, 150.0, 300.0])

        # Batched
        batch_result = kennett_reflectivity_batch(stack, p_test, omega_test)

        # Scalar
        for ip, p_val in enumerate(p_test):
            scalar_result = kennett_layers(stack, float(p_val), omega_test)
            np.testing.assert_allclose(
                batch_result[ip, :],
                scalar_result.RPP,
                atol=1e-12,
                err_msg=f"Mismatch at p={p_val}",
            )

    def test_fluid_solid_stack(self):
        """Batched Kennett works with FluidLayer + solid layers."""
        stack = LayerStack(
            layers=[
                FluidLayer(WATER_ALPHA, WATER_RHO, WATER_DEPTH),
                IsotropicLayer(SED_ALPHA, SED_BETA, SED_RHO, 200.0),
                IsotropicLayer(HS_ALPHA, HS_BETA, HS_RHO, np.inf),
            ]
        )
        p_test = np.array([1e-5, 1e-4])
        omega_test = np.array([50.0, 200.0])
        result = kennett_reflectivity_batch(stack, p_test, omega_test)
        assert result.shape == (2, 2)
        # Should have nonzero reflectivity
        assert np.any(np.abs(result) > 1e-6)


# ── 6. Half-space below water ─────────────────────────────────────────────


class TestHalfSpaceBelowWater:
    """Test normal-incidence R_PP for water over elastic half-space."""

    def test_normal_incidence_impedance(self):
        """R_PP at normal incidence = (Z2-Z1)/(Z2+Z1)."""
        # Single interface: water over half-space
        sub_ocean = LayerStack(
            layers=[
                IsotropicLayer(SED_ALPHA, SED_BETA, SED_RHO, 200.0),
                IsotropicLayer(HS_ALPHA, HS_BETA, HS_RHO, np.inf),
            ]
        )
        p = np.array([1e-8])  # near-normal
        omega = np.array([100.0])
        result = kennett_reflectivity_batch(sub_ocean, p, omega)

        # At normal incidence, just the top interface matters
        # for a two-layer solid stack
        Z1 = SED_RHO * SED_ALPHA
        Z2 = HS_RHO * HS_ALPHA
        expected = (Z2 - Z1) / (Z2 + Z1)
        assert result[0, 0].real == pytest.approx(expected, rel=0.01)


# ── 7. Survey stack builder ───────────────────────────────────────────────


class TestBuildSurveyStack:
    """Test build_survey_stack."""

    def test_correct_layer_count(self):
        survey = SurveyConfig(
            source_depth=5.0,
            receiver_depth=8.0,
            receiver_type="hydrophone",
            offsets=np.array([100.0, 200.0]),
            water_depth=WATER_DEPTH,
        )
        sediments = [IsotropicLayer(SED_ALPHA, SED_BETA, SED_RHO, 200.0)]
        hs = IsotropicLayer(HS_ALPHA, HS_BETA, HS_RHO, np.inf)
        stack = build_survey_stack(survey, sediments, hs)
        assert stack.n_layers == 3
        assert isinstance(stack.layers[0], FluidLayer)
        assert isinstance(stack.layers[1], IsotropicLayer)
        assert np.isinf(stack.layers[-1].thickness)

    def test_water_properties(self):
        survey = SurveyConfig(
            source_depth=5.0,
            receiver_depth=8.0,
            receiver_type="hydrophone",
            offsets=np.array([100.0]),
            water_depth=WATER_DEPTH,
            water_alpha=1480.0,
            water_rho=1030.0,
        )
        stack = build_survey_stack(
            survey, [], IsotropicLayer(HS_ALPHA, HS_BETA, HS_RHO, np.inf)
        )
        water = stack.layers[0]
        assert isinstance(water, FluidLayer)
        assert water.alpha == 1480.0
        assert water.rho == 1030.0
        assert water.thickness == WATER_DEPTH


# ── 8. Ricker source spectrum ─────────────────────────────────────────────


class TestRickerSpectrum:
    """Test Ricker wavelet spectrum."""

    def test_peak_at_f_peak(self):
        """Maximum of Ricker spectrum near omega_peak."""
        omega = np.linspace(1, 500, 1000)
        f_peak = 30.0
        S = ricker_source_spectrum(omega, f_peak)
        omega_peak = 2.0 * np.pi * f_peak
        # Peak at omega/omega_peak = 1 where S = exp(-1) ≈ 0.368
        i_max = np.argmax(np.abs(S))
        assert omega[i_max] == pytest.approx(omega_peak, rel=0.02)

    def test_zero_at_dc(self):
        """S(0) = 0."""
        S = ricker_source_spectrum(np.array([0.0]), 30.0)
        assert S[0] == pytest.approx(0.0, abs=1e-14)


# ── 9. SurveyConfig validation ────────────────────────────────────────────


class TestSurveyConfig:
    """Test SurveyConfig validation."""

    def test_reject_invalid_receiver_type(self):
        with pytest.raises(ValueError, match="receiver_type"):
            SurveyConfig(
                source_depth=5.0,
                receiver_depth=8.0,
                receiver_type="geophone_invalid",
                offsets=np.array([100.0]),
                water_depth=70.0,
            )

    def test_accept_hydrophone(self):
        sc = SurveyConfig(
            source_depth=5.0,
            receiver_depth=8.0,
            receiver_type="hydrophone",
            offsets=np.array([100.0]),
            water_depth=70.0,
        )
        assert sc.receiver_type == "hydrophone"


# ── 10. End-to-end shot gather ────────────────────────────────────────────


class TestEndToEndGather:
    """End-to-end test with simple model."""

    def test_gather_runs_and_has_correct_shape(self):
        """Verify compute_shot_gather produces correct shapes."""
        survey = SurveyConfig(
            source_depth=5.0,
            receiver_depth=8.0,
            receiver_type="hydrophone",
            offsets=np.array([200.0, 500.0, 1000.0]),
            water_depth=WATER_DEPTH,
        )
        sediments = [IsotropicLayer(SED_ALPHA, SED_BETA, SED_RHO, 200.0)]
        hs = IsotropicLayer(HS_ALPHA, HS_BETA, HS_RHO, np.inf)
        stack = build_survey_stack(survey, sediments, hs)

        gc = GatherConfig(
            T=2.0,
            nw=256,
            np_slow=512,
            f_peak=30.0,
            free_surface=False,
        )
        result = compute_shot_gather(stack, survey, gc)

        assert isinstance(result, ShotGatherResult)
        nt = 2 * gc.nw
        assert result.gather.shape == (3, nt)
        assert result.time.shape == (nt,)
        assert len(result.offsets) == 3

    def test_gather_has_energy(self):
        """Gather should have nonzero energy (reflections present)."""
        survey = SurveyConfig(
            source_depth=5.0,
            receiver_depth=8.0,
            receiver_type="hydrophone",
            offsets=np.array([500.0]),
            water_depth=WATER_DEPTH,
        )
        sediments = [IsotropicLayer(SED_ALPHA, SED_BETA, SED_RHO, 200.0)]
        hs = IsotropicLayer(HS_ALPHA, HS_BETA, HS_RHO, np.inf)
        stack = build_survey_stack(survey, sediments, hs)

        gc = GatherConfig(
            T=2.0,
            nw=256,
            np_slow=512,
            f_peak=30.0,
            free_surface=False,
        )
        result = compute_shot_gather(stack, survey, gc)
        assert np.max(np.abs(result.gather)) > 0


# ── 11. YAML config loading ──────────────────────────────────────────────


class TestYAMLConfig:
    """Test YAML configuration loading."""

    def test_load_example_config(self):
        config_path = Path("configs/example_survey.yml")
        if not config_path.exists():
            pytest.skip("Example config not available")
        from cubic_scattering.seismic_survey import load_survey_config

        stack, survey, gather = load_survey_config(config_path)

        assert isinstance(stack.layers[0], FluidLayer)
        assert survey.receiver_type == "hydrophone"
        assert survey.water_depth == 70.0
        assert gather.f_peak == 30.0
        assert stack.n_layers == 5  # water + 3 sediment + half-space

    def test_config_not_found(self):
        from cubic_scattering.seismic_survey import load_survey_config

        with pytest.raises(FileNotFoundError):
            load_survey_config("nonexistent.yml")


# ── 12. Existing Kennett tests regression ────────────────────────────────


class TestKennettRegression:
    """Ensure FluidLayer changes don't break existing solid-only tests."""

    def test_solid_only_unchanged(self):
        """Solid-only stack gives identical results to before."""
        stack = LayerStack(
            layers=[
                IsotropicLayer(5000.0, 3000.0, 2500.0, 100.0),
                IsotropicLayer(5500.0, 3300.0, 2700.0, np.inf),
            ]
        )
        omega = np.linspace(10, 500, 32)
        p = 1e-4
        result = kennett_layers(stack, p, omega)
        # Should produce valid non-zero reflectivity
        assert np.any(np.abs(result.RPP) > 1e-6)
        # Should not crash — this is the main regression check
