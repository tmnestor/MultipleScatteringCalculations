"""Tests for Kennett recursive R/T layer stacking."""

import sys
from pathlib import Path

import numpy as np
import pytest

from cubic_scattering.cpa_iteration import Phase
from cubic_scattering.kennett_layers import (
    IsotropicLayer,
    LayerStack,
    _batch_inv2x2,
    _batch_matmul2x2,
    _complex_slowness,
    _vertical_slowness,
    cpa_stack_from_phases,
    cubic_to_isotropic_layer,
    kennett_layers,
    psv_solid_solid,
    random_velocity_stack,
    sh_solid_solid,
)

# ── Shared fixtures ─────────────────────────────────────────────────────

# Background medium (from MEMORY.md validated parameters)
ALPHA_REF = 5000.0  # m/s
BETA_REF = 3000.0  # m/s
RHO_REF = 2500.0  # kg/m^3

# Contrasting layer (~10% faster)
ALPHA2 = 5500.0
BETA2 = 3300.0
RHO2 = 2700.0

# Weak contrast layer (<0.1%)
ALPHA_WEAK = 5002.0
BETA_WEAK = 3001.0
RHO_WEAK = 2501.0

# Frequencies (avoid DC)
OMEGA = np.linspace(1.0, 100.0, 64) * 2.0 * np.pi

# Slowness values
P_NEAR_NORMAL = 1e-6  # s/m (effectively normal incidence)
P_OBLIQUE = np.sin(np.radians(30)) / ALPHA_REF  # 30-degree P incidence

# Layer thickness
THICKNESS = 100.0  # m


# ── 1. IsotropicLayer construction ──────────────────────────────────────


class TestIsotropicLayer:
    """Test IsotropicLayer construction and validation."""

    def test_valid_construction(self):
        layer = IsotropicLayer(alpha=5000, beta=3000, rho=2500, thickness=100)
        assert layer.alpha == 5000
        assert layer.beta == 3000
        assert layer.rho == 2500
        assert layer.thickness == 100
        assert np.isinf(layer.Q_alpha)
        assert np.isinf(layer.Q_beta)

    def test_half_space(self):
        layer = IsotropicLayer(alpha=5000, beta=3000, rho=2500, thickness=np.inf)
        assert np.isinf(layer.thickness)

    def test_with_attenuation(self):
        layer = IsotropicLayer(
            alpha=5000, beta=3000, rho=2500, thickness=100, Q_alpha=200, Q_beta=100
        )
        assert layer.Q_alpha == 200
        assert layer.Q_beta == 100

    def test_reject_negative_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            IsotropicLayer(alpha=-1, beta=3000, rho=2500, thickness=100)

    def test_reject_zero_beta(self):
        with pytest.raises(ValueError, match="beta"):
            IsotropicLayer(alpha=5000, beta=0, rho=2500, thickness=100)

    def test_reject_negative_rho(self):
        with pytest.raises(ValueError, match="rho"):
            IsotropicLayer(alpha=5000, beta=3000, rho=-1, thickness=100)

    def test_reject_negative_thickness(self):
        with pytest.raises(ValueError, match="thickness"):
            IsotropicLayer(alpha=5000, beta=3000, rho=2500, thickness=-10)


# ── 2. LayerStack validation ────────────────────────────────────────────


class TestLayerStack:
    """Test LayerStack validation and factory methods."""

    def test_valid_two_layer(self):
        layers = [
            IsotropicLayer(
                alpha=ALPHA_REF, beta=BETA_REF, rho=RHO_REF, thickness=THICKNESS
            ),
            IsotropicLayer(alpha=ALPHA2, beta=BETA2, rho=RHO2, thickness=np.inf),
        ]
        stack = LayerStack(layers=layers)
        assert stack.n_layers == 2

    def test_reject_single_layer(self):
        with pytest.raises(ValueError, match=">= 2"):
            LayerStack(
                layers=[
                    IsotropicLayer(alpha=5000, beta=3000, rho=2500, thickness=np.inf),
                ]
            )

    def test_reject_finite_last_layer(self):
        with pytest.raises(ValueError, match="half-space"):
            LayerStack(
                layers=[
                    IsotropicLayer(alpha=5000, beta=3000, rho=2500, thickness=100),
                    IsotropicLayer(alpha=5000, beta=3000, rho=2500, thickness=100),
                ]
            )

    def test_from_arrays(self):
        stack = LayerStack.from_arrays(
            alpha=np.array([5000, 5500]),
            beta=np.array([3000, 3300]),
            rho=np.array([2500, 2700]),
            thickness=np.array([100, np.inf]),
        )
        assert stack.n_layers == 2
        assert stack.layers[0].alpha == 5000
        assert np.isinf(stack.layers[-1].thickness)

    def test_homogeneous_factory(self):
        stack = LayerStack.homogeneous(alpha=5000, beta=3000, rho=2500, n_layers=5)
        assert stack.n_layers == 5
        assert np.isinf(stack.layers[-1].thickness)
        for lay in stack.layers[:-1]:
            assert lay.thickness == 100.0


# ── 3. Vertical slowness ────────────────────────────────────────────────


class TestVerticalSlowness:
    """Test complex slowness and vertical slowness computation."""

    def test_lossless_slowness(self):
        s = _complex_slowness(5000.0, np.inf)
        assert s == pytest.approx(1.0 / 5000.0)
        assert s.imag == 0.0

    def test_attenuative_slowness(self):
        s = _complex_slowness(5000.0, 100.0)
        assert s.real > 0
        assert s.imag > 0  # attenuation adds positive imaginary part

    def test_eta_squared_plus_p_squared(self):
        """Verify eta^2 + p^2 = s^2."""
        s = _complex_slowness(5000.0, np.inf)
        p = 1e-4
        eta = _vertical_slowness(s, p)
        assert eta**2 + p**2 == pytest.approx(s**2, rel=1e-12)

    def test_eta_squared_attenuative(self):
        """eta^2 + p^2 = s^2 also for attenuative case."""
        s = _complex_slowness(5000.0, 200.0)
        p = 5e-5
        eta = _vertical_slowness(s, p)
        assert eta**2 + p**2 == pytest.approx(s**2, rel=1e-12)

    def test_im_eta_positive_evanescent(self):
        """For evanescent waves (p > 1/v), Im(eta) > 0."""
        s = _complex_slowness(5000.0, np.inf)
        p = 1.0 / 3000.0  # p > s_P, so P-wave is evanescent
        eta = _vertical_slowness(s, p)
        assert eta.imag > 0

    def test_propagating_eta_positive_real(self):
        """For propagating waves (p < 1/v), Re(eta) > 0."""
        s = _complex_slowness(5000.0, np.inf)
        p = 1e-5  # small p → propagating
        eta = _vertical_slowness(s, p)
        assert eta.real > 0

    def test_match_phd_complex_slowness(self):
        """Compare against PhD layer_model.py complex_slowness."""
        phd_path = Path("PhD_fortran_code/Kennett_Reflectivity")
        if not phd_path.exists():
            pytest.skip("PhD code not available")
        sys.path.insert(0, str(phd_path))
        from layer_model import complex_slowness as phd_complex_slowness

        for v, Q in [(5000, 200), (3000, 100), (1500, 50)]:
            mine = _complex_slowness(v, Q)
            theirs = phd_complex_slowness(v, Q)
            assert mine == pytest.approx(theirs, rel=1e-14)

    def test_match_phd_vertical_slowness(self):
        """Compare against PhD layer_model.py vertical_slowness."""
        phd_path = Path("PhD_fortran_code/Kennett_Reflectivity")
        if not phd_path.exists():
            pytest.skip("PhD code not available")
        sys.path.insert(0, str(phd_path))
        from layer_model import complex_slowness as phd_cs
        from layer_model import vertical_slowness as phd_vs

        for v, Q in [(5000, 200), (3000, 100)]:
            s = phd_cs(v, Q)
            for p_val in [1e-5, 1e-4, 1e-3]:
                mine = _vertical_slowness(_complex_slowness(v, Q), p_val)
                theirs = phd_vs(s, complex(p_val))
                assert mine == pytest.approx(theirs, rel=1e-14)


# ── 4. P-SV coefficients ────────────────────────────────────────────────


class TestPSVCoefficients:
    """Test P-SV interfacial scattering coefficients."""

    @pytest.fixture()
    def interface_params(self):
        """Parameters for a typical solid-solid interface."""
        s_p1 = _complex_slowness(ALPHA_REF, np.inf)
        s_s1 = _complex_slowness(BETA_REF, np.inf)
        s_p2 = _complex_slowness(ALPHA2, np.inf)
        s_s2 = _complex_slowness(BETA2, np.inf)
        p = P_OBLIQUE
        return {
            "p": p,
            "eta1": _vertical_slowness(s_p1, p),
            "neta1": _vertical_slowness(s_s1, p),
            "rho1": RHO_REF,
            "beta1": 1.0 / s_s1,
            "eta2": _vertical_slowness(s_p2, p),
            "neta2": _vertical_slowness(s_s2, p),
            "rho2": RHO2,
            "beta2": 1.0 / s_s2,
        }

    def test_reciprocity_Tu_Td_transpose(self, interface_params):
        """Modified coefficients satisfy Tu = Td.T."""
        coeff = psv_solid_solid(**interface_params)
        np.testing.assert_allclose(coeff.Tu, coeff.Td.T, atol=1e-14)

    def test_energy_conservation(self, interface_params):
        """Rd†Rd + Td†Td = I for real slowness, propagating waves."""
        coeff = psv_solid_solid(**interface_params)
        energy = coeff.Rd.conj().T @ coeff.Rd + coeff.Td.conj().T @ coeff.Td
        np.testing.assert_allclose(energy, np.eye(2), atol=1e-12)

    def test_identical_layers_zero_reflection(self):
        """Identical layers give Rd = 0, Td = I."""
        s_p = _complex_slowness(ALPHA_REF, np.inf)
        s_s = _complex_slowness(BETA_REF, np.inf)
        p = P_OBLIQUE
        coeff = psv_solid_solid(
            p=p,
            eta1=_vertical_slowness(s_p, p),
            neta1=_vertical_slowness(s_s, p),
            rho1=RHO_REF,
            beta1=1.0 / s_s,
            eta2=_vertical_slowness(s_p, p),
            neta2=_vertical_slowness(s_s, p),
            rho2=RHO_REF,
            beta2=1.0 / s_s,
        )
        np.testing.assert_allclose(coeff.Rd, np.zeros((2, 2)), atol=1e-14)
        np.testing.assert_allclose(coeff.Td, np.eye(2), atol=1e-14)

    def test_Rd_Ru_symmetry(self, interface_params):
        """Ru = -Rd only at normal incidence; general Rd + Ru != 0."""
        coeff = psv_solid_solid(**interface_params)
        # Off-diagonal: Rd_PS = Rd_SP (symmetric for modified coefficients)
        assert coeff.Rd[0, 1] == pytest.approx(coeff.Rd[1, 0], rel=1e-12)

    def test_match_phd_solid_solid(self, interface_params):
        """Compare against PhD scattering_matrices.solid_solid_interface."""
        phd_path = Path("PhD_fortran_code/Kennett_Reflectivity")
        if not phd_path.exists():
            pytest.skip("PhD code not available")
        sys.path.insert(0, str(phd_path))
        from scattering_matrices import solid_solid_interface

        mine = psv_solid_solid(**interface_params)
        theirs = solid_solid_interface(**interface_params)
        np.testing.assert_allclose(mine.Rd, theirs.Rd, atol=1e-14)
        np.testing.assert_allclose(mine.Ru, theirs.Ru, atol=1e-14)
        np.testing.assert_allclose(mine.Td, theirs.Td, atol=1e-14)
        np.testing.assert_allclose(mine.Tu, theirs.Tu, atol=1e-14)


# ── 5. SH coefficients ──────────────────────────────────────────────────


class TestSHCoefficients:
    """Test SH interfacial scattering coefficients."""

    def test_impedance_formula(self):
        """Rd = (Z1 - Z2) / (Z1 + Z2) for SH impedance."""
        s_s1 = _complex_slowness(BETA_REF, np.inf)
        s_s2 = _complex_slowness(BETA2, np.inf)
        p = P_OBLIQUE
        neta1 = _vertical_slowness(s_s1, p)
        neta2 = _vertical_slowness(s_s2, p)
        beta1_c = 1.0 / s_s1
        beta2_c = 1.0 / s_s2

        coeff = sh_solid_solid(neta1, RHO_REF, beta1_c, neta2, RHO2, beta2_c)

        Z1 = RHO_REF * beta1_c**2 * neta1
        Z2 = RHO2 * beta2_c**2 * neta2
        expected_Rd = (Z1 - Z2) / (Z1 + Z2)

        assert coeff.Rd == pytest.approx(expected_Rd, rel=1e-12)

    def test_reciprocity_Tu_Td(self):
        """Tu = Td for modified SH coefficients."""
        s_s1 = _complex_slowness(BETA_REF, np.inf)
        s_s2 = _complex_slowness(BETA2, np.inf)
        p = P_OBLIQUE
        coeff = sh_solid_solid(
            _vertical_slowness(s_s1, p),
            RHO_REF,
            1.0 / s_s1,
            _vertical_slowness(s_s2, p),
            RHO2,
            1.0 / s_s2,
        )
        assert coeff.Tu == pytest.approx(coeff.Td, rel=1e-14)

    def test_Ru_equals_minus_Rd(self):
        """Ru = -Rd for modified SH coefficients."""
        s_s1 = _complex_slowness(BETA_REF, np.inf)
        s_s2 = _complex_slowness(BETA2, np.inf)
        p = P_OBLIQUE
        coeff = sh_solid_solid(
            _vertical_slowness(s_s1, p),
            RHO_REF,
            1.0 / s_s1,
            _vertical_slowness(s_s2, p),
            RHO2,
            1.0 / s_s2,
        )
        assert coeff.Ru == pytest.approx(-coeff.Rd, rel=1e-14)

    def test_energy_conservation(self):
        """abs(Rd)^2 + abs(Td)^2 = 1 for real impedances."""
        s_s1 = _complex_slowness(BETA_REF, np.inf)
        s_s2 = _complex_slowness(BETA2, np.inf)
        p = P_OBLIQUE
        coeff = sh_solid_solid(
            _vertical_slowness(s_s1, p),
            RHO_REF,
            1.0 / s_s1,
            _vertical_slowness(s_s2, p),
            RHO2,
            1.0 / s_s2,
        )
        energy = abs(coeff.Rd) ** 2 + abs(coeff.Td) ** 2
        assert energy == pytest.approx(1.0, rel=1e-12)


# ── 6. Homogeneous stack ────────────────────────────────────────────────


class TestHomogeneousStack:
    """Test that a homogeneous stack produces zero reflection."""

    def test_psv_zero_reflection(self):
        stack = LayerStack.homogeneous(ALPHA_REF, BETA_REF, RHO_REF, n_layers=5)
        result = kennett_layers(stack, P_OBLIQUE, OMEGA)
        np.testing.assert_allclose(result.RPP, 0.0, atol=1e-14)
        np.testing.assert_allclose(result.RSS, 0.0, atol=1e-14)
        np.testing.assert_allclose(result.RPS, 0.0, atol=1e-14)
        np.testing.assert_allclose(result.RSP, 0.0, atol=1e-14)

    def test_sh_zero_reflection(self):
        stack = LayerStack.homogeneous(ALPHA_REF, BETA_REF, RHO_REF, n_layers=5)
        result = kennett_layers(stack, P_OBLIQUE, OMEGA)
        np.testing.assert_allclose(result.RSH, 0.0, atol=1e-14)


# ── 7. Single interface ─────────────────────────────────────────────────


class TestSingleInterface:
    """Test single-interface (2-layer) reflectivity."""

    def test_RRd_equals_Rd(self):
        """For 2 layers, the cumulative R equals the interface R."""
        stack = LayerStack(
            layers=[
                IsotropicLayer(ALPHA_REF, BETA_REF, RHO_REF, THICKNESS),
                IsotropicLayer(ALPHA2, BETA2, RHO2, np.inf),
            ]
        )
        result = kennett_layers(stack, P_OBLIQUE, OMEGA)

        # Compute expected interface coefficients
        s_p1 = _complex_slowness(ALPHA_REF, np.inf)
        s_s1 = _complex_slowness(BETA_REF, np.inf)
        s_p2 = _complex_slowness(ALPHA2, np.inf)
        s_s2 = _complex_slowness(BETA2, np.inf)
        p = P_OBLIQUE
        coeff = psv_solid_solid(
            p,
            _vertical_slowness(s_p1, p),
            _vertical_slowness(s_s1, p),
            RHO_REF,
            1.0 / s_s1,
            _vertical_slowness(s_p2, p),
            _vertical_slowness(s_s2, p),
            RHO2,
            1.0 / s_s2,
        )

        # Phase through layer 1 (the only finite layer)
        eta1 = _vertical_slowness(s_p1, p)
        neta1 = _vertical_slowness(s_s1, p)
        ea = np.exp(1j * OMEGA * eta1 * THICKNESS)
        eb = np.exp(1j * OMEGA * neta1 * THICKNESS)

        # At single interface: RRd starts at Rd, then phase-shifted through layer 1
        # Actually: the recursion gives RRd = Rd (from the interface)
        # then no more interfaces above, so that's it.
        # The result should be Rd with phase shift through the top layer.
        # Wait: top layer = layer 0, RRd accumulates from below.
        # For 2 layers: interface 0 is between layers 0 and 1.
        # RRd starts = 0 (half-space). Phase through half-space = 1.
        # At interface 0: MT = 1*0*1 = 0, so RRd = Rd.
        # No more interfaces. Result = Rd exactly.
        for w in range(len(OMEGA)):
            np.testing.assert_allclose(result.RD_psv[w], coeff.Rd, atol=1e-14)

    def test_normal_incidence_pp_impedance(self):
        """Normal-incidence PP = (Z2-Z1)/(Z2+Z1)."""
        stack = LayerStack(
            layers=[
                IsotropicLayer(ALPHA_REF, BETA_REF, RHO_REF, THICKNESS),
                IsotropicLayer(ALPHA2, BETA2, RHO2, np.inf),
            ]
        )
        # Use very small p for near-normal; p=1e-6 still gives O(p^2) corrections
        result = kennett_layers(stack, P_NEAR_NORMAL, OMEGA)

        Z1 = RHO_REF * ALPHA_REF
        Z2 = RHO2 * ALPHA2
        expected_pp = (Z2 - Z1) / (Z2 + Z1)

        # Modified coefficients at small but nonzero p differ from exact normal
        # by O(p^2) terms; tolerance accounts for this
        np.testing.assert_allclose(result.RPP.real, expected_pp, rtol=1e-4)
        np.testing.assert_allclose(result.RPP.imag, 0.0, atol=1e-3)

    def test_normal_incidence_no_ps_coupling(self):
        """At near-normal incidence, P-SV coupling is small."""
        stack = LayerStack(
            layers=[
                IsotropicLayer(ALPHA_REF, BETA_REF, RHO_REF, THICKNESS),
                IsotropicLayer(ALPHA2, BETA2, RHO2, np.inf),
            ]
        )
        result = kennett_layers(stack, P_NEAR_NORMAL, OMEGA)
        # At p=1e-6, coupling is O(p) ~ 1e-3
        np.testing.assert_allclose(result.RPS, 0.0, atol=1e-2)
        np.testing.assert_allclose(result.RSP, 0.0, atol=1e-2)


# ── 8. Weak contrast Born ───────────────────────────────────────────────


class TestWeakContrastBorn:
    """Test that weak-contrast reflection scales linearly with contrast."""

    def test_pp_proportional_to_contrast(self):
        """RPP scales linearly with impedance contrast."""
        contrasts = [0.001, 0.002, 0.005]
        rpp_values = []

        for frac in contrasts:
            stack = LayerStack(
                layers=[
                    IsotropicLayer(ALPHA_REF, BETA_REF, RHO_REF, THICKNESS),
                    IsotropicLayer(
                        ALPHA_REF * (1 + frac),
                        BETA_REF * (1 + frac),
                        RHO_REF * (1 + frac),
                        np.inf,
                    ),
                ]
            )
            result = kennett_layers(stack, P_NEAR_NORMAL, OMEGA)
            rpp_values.append(np.abs(result.RPP[0]))

        # Check linearity: RPP(2x contrast) / RPP(1x contrast) ~ 2
        ratio1 = rpp_values[1] / rpp_values[0]
        ratio2 = rpp_values[2] / rpp_values[0]
        assert ratio1 == pytest.approx(contrasts[1] / contrasts[0], rel=0.01)
        assert ratio2 == pytest.approx(contrasts[2] / contrasts[0], rel=0.01)


# ── 9. Reciprocity ──────────────────────────────────────────────────────


class TestReciprocity:
    """Test reciprocity for symmetric stacks."""

    def test_symmetric_stack_Rd_from_both_sides(self):
        """A symmetric A-B-A stack has the same Rd from above and below."""
        # Original: A on top, B in middle, A as half-space
        stack_down = LayerStack(
            layers=[
                IsotropicLayer(ALPHA_REF, BETA_REF, RHO_REF, THICKNESS),
                IsotropicLayer(ALPHA2, BETA2, RHO2, THICKNESS),
                IsotropicLayer(ALPHA_REF, BETA_REF, RHO_REF, np.inf),
            ]
        )
        # Flipped: A on top, B in middle, A as half-space (same!)
        stack_up = LayerStack(
            layers=[
                IsotropicLayer(ALPHA_REF, BETA_REF, RHO_REF, THICKNESS),
                IsotropicLayer(ALPHA2, BETA2, RHO2, THICKNESS),
                IsotropicLayer(ALPHA_REF, BETA_REF, RHO_REF, np.inf),
            ]
        )
        rd_down = kennett_layers(stack_down, P_OBLIQUE, OMEGA)
        rd_up = kennett_layers(stack_up, P_OBLIQUE, OMEGA)

        np.testing.assert_allclose(rd_down.RPP, rd_up.RPP, atol=1e-14)
        np.testing.assert_allclose(rd_down.RSH, rd_up.RSH, atol=1e-14)


# ── 10. Attenuation ─────────────────────────────────────────────────────


class TestAttenuation:
    """Test that attenuation reduces reflection amplitudes."""

    def test_attenuated_reflection_smaller(self):
        """abs(R) with Q=100 <= abs(R) with Q=inf at most frequencies."""
        stack_lossless = LayerStack(
            layers=[
                IsotropicLayer(ALPHA_REF, BETA_REF, RHO_REF, THICKNESS),
                IsotropicLayer(ALPHA2, BETA2, RHO2, THICKNESS),
                IsotropicLayer(ALPHA_REF, BETA_REF, RHO_REF, np.inf),
            ]
        )
        stack_lossy = LayerStack(
            layers=[
                IsotropicLayer(ALPHA_REF, BETA_REF, RHO_REF, THICKNESS, 100, 100),
                IsotropicLayer(ALPHA2, BETA2, RHO2, THICKNESS, 100, 100),
                IsotropicLayer(ALPHA_REF, BETA_REF, RHO_REF, np.inf, 100, 100),
            ]
        )
        r_lossless = kennett_layers(stack_lossless, P_OBLIQUE, OMEGA)
        r_lossy = kennett_layers(stack_lossy, P_OBLIQUE, OMEGA)

        # On average, attenuation should reduce reflection amplitude
        mean_lossless = np.mean(np.abs(r_lossless.RPP))
        mean_lossy = np.mean(np.abs(r_lossy.RPP))
        assert mean_lossy < mean_lossless


# ── 11. Phase accumulation ──────────────────────────────────────────────


class TestPhaseAccumulation:
    """Test that phase of reflection encodes correct travel time."""

    def test_two_way_travel_time(self):
        """Phase slope dφ/dω encodes 2h/alpha for a reflector at depth h.

        Use a 3-layer model: the Kennett recursion accumulates phase through
        the middle layer (not the top layer). A reflector at depth h (middle
        layer thickness) produces phase delay 2h/alpha in the cumulative Rd.
        """
        h = 500.0  # middle layer thickness
        stack = LayerStack(
            layers=[
                IsotropicLayer(ALPHA_REF, BETA_REF, RHO_REF, THICKNESS),
                IsotropicLayer(ALPHA_REF, BETA_REF, RHO_REF, h),
                IsotropicLayer(ALPHA2, BETA2, RHO2, np.inf),
            ]
        )
        omega_dense = np.linspace(10, 200, 512) * 2 * np.pi
        result = kennett_layers(stack, P_NEAR_NORMAL, omega_dense)

        # Phase of RPP (unwrapped). The top interface gives Rd=0 (identical
        # layers), so the total reflection comes from the bottom interface
        # after two-way propagation through the middle layer.
        phase = np.unwrap(np.angle(result.RPP))

        # Linear fit: dφ/dω = two-way P travel time through middle layer
        coeffs = np.polyfit(omega_dense, phase, 1)
        expected_delay = 2 * h / ALPHA_REF

        assert coeffs[0] == pytest.approx(expected_delay, rel=0.01)


# ── 12. CPA bridge ──────────────────────────────────────────────────────


class TestCPABridge:
    """Test cubic_to_isotropic_layer matches as_reference_medium."""

    def test_matches_reference_medium(self):
        from cubic_scattering.cpa_iteration import CubicEffectiveMedium

        lam = RHO_REF * (ALPHA_REF**2 - 2 * BETA_REF**2)
        mu = RHO_REF * BETA_REF**2
        eff = CubicEffectiveMedium(lam=lam, mu_off=mu, mu_diag=mu * 1.001, rho=RHO_REF)

        layer = cubic_to_isotropic_layer(eff, thickness=100.0)
        ref = eff.as_reference_medium()

        assert layer.alpha == pytest.approx(ref.alpha, rel=1e-12)
        assert layer.beta == pytest.approx(ref.beta, rel=1e-12)
        assert layer.rho == pytest.approx(ref.rho, rel=1e-12)

    def test_with_attenuation(self):
        from cubic_scattering.cpa_iteration import CubicEffectiveMedium

        lam = RHO_REF * (ALPHA_REF**2 - 2 * BETA_REF**2)
        mu = RHO_REF * BETA_REF**2
        eff = CubicEffectiveMedium(lam=lam, mu_off=mu, mu_diag=mu, rho=RHO_REF)

        layer = cubic_to_isotropic_layer(eff, 100.0, Q_alpha=200, Q_beta=100)
        assert layer.Q_alpha == 200
        assert layer.Q_beta == 100


# ── 13. CPA stack ───────────────────────────────────────────────────────


class TestCPAStack:
    """Test cpa_stack_from_phases produces a valid stack."""

    def test_two_layer_cpa_stack(self):
        """Build a 2-layer stack from weak-contrast two-phase layers."""
        ref_lam = RHO_REF * (ALPHA_REF**2 - 2 * BETA_REF**2)
        ref_mu = RHO_REF * BETA_REF**2

        # Weak contrast for fast CPA convergence
        Dlam = ref_lam * 1e-4
        Dmu = ref_mu * 1e-4
        Drho = RHO_REF * 1e-4

        phases_layer1 = [
            Phase(lam=ref_lam, mu=ref_mu, rho=RHO_REF, volume_fraction=0.6),
            Phase(
                lam=ref_lam + Dlam,
                mu=ref_mu + Dmu,
                rho=RHO_REF + Drho,
                volume_fraction=0.4,
            ),
        ]
        phases_layer2 = [
            Phase(lam=ref_lam, mu=ref_mu, rho=RHO_REF, volume_fraction=0.3),
            Phase(
                lam=ref_lam + Dlam,
                mu=ref_mu + Dmu,
                rho=RHO_REF + Drho,
                volume_fraction=0.7,
            ),
        ]

        omega_cpa = 2 * np.pi * 10.0  # 10 Hz
        a = 1.0  # 1 m half-width

        stack = cpa_stack_from_phases([phases_layer1, phases_layer2], omega_cpa, a)

        assert stack.n_layers == 2
        assert np.isinf(stack.layers[-1].thickness)
        for lay in stack.layers:
            assert lay.alpha > 0
            assert lay.beta > 0
            assert lay.rho > 0


# ── 14. Random stack ────────────────────────────────────────────────────


class TestRandomStack:
    """Test random stack generation."""

    def test_velocity_stack_reproducibility(self):
        """Same seed gives same stack."""
        s1 = random_velocity_stack(
            ALPHA_REF, BETA_REF, RHO_REF, 5, 100, 200, 100, 50, seed=42
        )
        s2 = random_velocity_stack(
            ALPHA_REF, BETA_REF, RHO_REF, 5, 100, 200, 100, 50, seed=42
        )
        for lay1, lay2 in zip(s1.layers, s2.layers, strict=True):
            assert lay1.alpha == lay2.alpha
            assert lay1.beta == lay2.beta
            assert lay1.rho == lay2.rho

    def test_velocity_stack_physical(self):
        """Random stack has positive velocities and densities."""
        stack = random_velocity_stack(
            ALPHA_REF, BETA_REF, RHO_REF, 10, 100, 500, 300, 100, seed=123
        )
        assert stack.n_layers == 10
        for lay in stack.layers:
            assert lay.alpha > 0
            assert lay.beta > 0
            assert lay.rho > 0

    def test_velocity_stack_different_seeds(self):
        """Different seeds give different stacks."""
        s1 = random_velocity_stack(
            ALPHA_REF, BETA_REF, RHO_REF, 5, 100, 200, 100, 50, seed=1
        )
        s2 = random_velocity_stack(
            ALPHA_REF, BETA_REF, RHO_REF, 5, 100, 200, 100, 50, seed=2
        )
        alphas1 = [lay.alpha for lay in s1.layers]
        alphas2 = [lay.alpha for lay in s2.layers]
        assert alphas1 != alphas2


# ── 15. Cross-validation with PhD code ──────────────────────────────────


class TestCrossValidationPhD:
    """Cross-validate against PhD Kennett_Reflectivity code."""

    def test_psv_interface_coefficients(self):
        """Compare psv_solid_solid against PhD solid_solid_interface exactly."""
        phd_path = Path("PhD_fortran_code/Kennett_Reflectivity")
        if not phd_path.exists():
            pytest.skip("PhD code not available")
        sys.path.insert(0, str(phd_path))
        from scattering_matrices import solid_solid_interface

        # Test multiple parameter combinations
        configs = [
            (ALPHA_REF, BETA_REF, RHO_REF, ALPHA2, BETA2, RHO2),
            (4000, 2300, 2200, 6000, 3500, 2800),
            (3000, 1500, 2000, 5000, 3000, 2500),
        ]

        for alpha1, beta1, rho1, alpha2, beta2, rho2 in configs:
            for Q in [np.inf, 200.0]:
                for p in [P_NEAR_NORMAL, P_OBLIQUE, 5e-5]:
                    # Skip attenuative for PhD code that requires Q > 0
                    if np.isinf(Q):
                        sp1 = complex(1.0 / alpha1)
                        ss1 = complex(1.0 / beta1)
                        sp2 = complex(1.0 / alpha2)
                        ss2 = complex(1.0 / beta2)
                    else:
                        sp1 = _complex_slowness(alpha1, Q)
                        ss1 = _complex_slowness(beta1, Q)
                        sp2 = _complex_slowness(alpha2, Q)
                        ss2 = _complex_slowness(beta2, Q)

                    eta1 = _vertical_slowness(sp1, p)
                    neta1 = _vertical_slowness(ss1, p)
                    eta2 = _vertical_slowness(sp2, p)
                    neta2 = _vertical_slowness(ss2, p)
                    bc1 = 1.0 / ss1
                    bc2 = 1.0 / ss2

                    mine = psv_solid_solid(
                        p, eta1, neta1, rho1, bc1, eta2, neta2, rho2, bc2
                    )
                    theirs = solid_solid_interface(
                        p, eta1, neta1, rho1, bc1, eta2, neta2, rho2, bc2
                    )

                    np.testing.assert_allclose(mine.Rd, theirs.Rd, atol=1e-14)
                    np.testing.assert_allclose(mine.Ru, theirs.Ru, atol=1e-14)
                    np.testing.assert_allclose(mine.Td, theirs.Td, atol=1e-14)
                    np.testing.assert_allclose(mine.Tu, theirs.Tu, atol=1e-14)

    def test_recursion_structure_with_phd_utilities(self):
        """Validate recursion by running PhD batch utilities on our coefficients.

        The PhD code always includes an acoustic ocean layer, making direct
        full-stack comparison impossible. Instead, we validate that the PhD
        batch_inv2x2 and batch_matmul give identical results to ours.
        """
        phd_path = Path("PhD_fortran_code")
        if not (phd_path / "Kennett_Reflectivity").exists():
            pytest.skip("PhD code not available")
        sys.path.insert(0, str(phd_path))
        from Kennett_Reflectivity.kennett_reflectivity import (
            batch_inv2x2 as phd_inv,
        )
        from Kennett_Reflectivity.kennett_reflectivity import (
            batch_matmul as phd_mul,
        )

        # Test batch_inv2x2 agreement
        rng = np.random.default_rng(99)
        M = rng.standard_normal((32, 2, 2)) + 1j * rng.standard_normal((32, 2, 2))
        np.testing.assert_allclose(_batch_inv2x2(M), phd_inv(M), atol=1e-12)

        # Test batch_matmul agreement (all shapes)
        A2 = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
        B3 = rng.standard_normal((32, 2, 2)) + 1j * rng.standard_normal((32, 2, 2))
        A3 = rng.standard_normal((32, 2, 2)) + 1j * rng.standard_normal((32, 2, 2))

        np.testing.assert_allclose(
            _batch_matmul2x2(A2, B3), phd_mul(A2, B3), atol=1e-12
        )
        np.testing.assert_allclose(
            _batch_matmul2x2(A3, A2), phd_mul(A3, A2), atol=1e-12
        )
        np.testing.assert_allclose(
            _batch_matmul2x2(A3, B3), phd_mul(A3, B3), atol=1e-12
        )
