"""Tests for ocean-bottom reflection with heterogeneous sediment."""

from pathlib import Path

import numpy as np
import yaml

from cubic_scattering.effective_contrasts import MaterialContrast, ReferenceMedium
from cubic_scattering.ocean_bottom import (
    OceanBottomConfig,
    compute_ocean_bottom_reflection,
    load_ocean_bottom_config,
)
from cubic_scattering.slab_scattering import (
    SlabGeometry,
    uniform_slab_material,
)

# ── Small, fast test parameters ──────────────────────────────────────────

# Sediment background
SED_REF = ReferenceMedium(alpha=2000.0, beta=800.0, rho=1800.0)

# Halfspace
HS_ALPHA, HS_BETA, HS_RHO = 3000.0, 1700.0, 2200.0

# Water
WATER_ALPHA, WATER_RHO, WATER_DEPTH = 1500.0, 1025.0, 70.0

# Moderate contrast (~10% velocity perturbation in sediment)
# Δα/α ≈ 10%, Δβ/β ≈ 10%, Δρ/ρ = 5%
# Perturbed: α = 2200 m/s, β = 880 m/s, ρ = 1890 kg/m³
CONTRAST = MaterialContrast(Dlambda=1.3244e9, Dmu=0.3116e9, Drho=90.0)

# Weak contrast (for Born scaling test) — 1/1000 of moderate
WEAK_CONTRAST = MaterialContrast(Dlambda=1.3244e6, Dmu=0.3116e6, Drho=0.09)


def _make_config(
    geometry: SlabGeometry,
    material,
    *,
    nw: int = 64,
    T: float = 0.5,
    f_min: float = 10.0,
    f_max: float = 60.0,
    f_peak: float = 30.0,
    free_surface: bool = False,
    p: float = 0.0,
) -> OceanBottomConfig:
    return OceanBottomConfig(
        water_alpha=WATER_ALPHA,
        water_rho=WATER_RHO,
        water_depth=WATER_DEPTH,
        sed_ref=SED_REF,
        hs_alpha=HS_ALPHA,
        hs_beta=HS_BETA,
        hs_rho=HS_RHO,
        geometry=geometry,
        material=material,
        f_peak=f_peak,
        T=T,
        nw=nw,
        f_min=f_min,
        f_max=f_max,
        free_surface=free_surface,
        p=p,
    )


class TestOceanBottom:
    """Tests for compute_ocean_bottom_reflection."""

    def test_zero_contrast_traces_match(self) -> None:
        """Zero slab contrast → trace_total == trace_homogeneous."""
        geom = SlabGeometry(M=2, N_z=1, a=1.0)
        zero_contrast = MaterialContrast(Dlambda=0.0, Dmu=0.0, Drho=0.0)
        mat = uniform_slab_material(geom, SED_REF, zero_contrast)
        cfg = _make_config(geom, mat)

        result = compute_ocean_bottom_reflection(cfg, progress=False)

        np.testing.assert_allclose(
            result.trace_total,
            result.trace_homogeneous,
            atol=1e-15,
            err_msg="Zero contrast should give identical traces",
        )
        # R_slab should be zero
        np.testing.assert_allclose(
            result.R_slab,
            0.0,
            atol=1e-15,
            err_msg="R_slab should be zero for zero contrast",
        )

    def test_weak_heterogeneity_perturbation_small(self) -> None:
        """Weak contrast → small difference between total and homogeneous."""
        geom = SlabGeometry(M=2, N_z=1, a=1.0)
        mat = uniform_slab_material(geom, SED_REF, WEAK_CONTRAST)
        cfg = _make_config(geom, mat)

        result = compute_ocean_bottom_reflection(cfg, progress=False)

        diff = np.abs(result.trace_total - result.trace_homogeneous)
        peak = np.max(np.abs(result.trace_homogeneous))
        if peak > 0:
            relative_diff = np.max(diff) / peak
            assert relative_diff < 0.01, (
                f"Weak contrast perturbation too large: {relative_diff:.4f}"
            )

    def test_born_scaling_frequency_domain(self) -> None:
        """Doubling weak contrast roughly doubles |R_slab|."""
        geom = SlabGeometry(M=2, N_z=1, a=1.0)

        mat1 = uniform_slab_material(geom, SED_REF, WEAK_CONTRAST)
        mat2 = uniform_slab_material(
            geom,
            SED_REF,
            MaterialContrast(
                Dlambda=2 * WEAK_CONTRAST.Dlambda,
                Dmu=2 * WEAK_CONTRAST.Dmu,
                Drho=2 * WEAK_CONTRAST.Drho,
            ),
        )

        cfg1 = _make_config(geom, mat1)
        cfg2 = _make_config(geom, mat2)

        r1 = compute_ocean_bottom_reflection(cfg1, progress=False)
        r2 = compute_ocean_bottom_reflection(cfg2, progress=False)

        # Compare only at active frequencies where R_slab is nonzero
        active = np.abs(r1.R_slab) > 1e-20
        if np.any(active):
            ratio = np.abs(r2.R_slab[active]) / np.abs(r1.R_slab[active])
            mean_ratio = np.mean(ratio)
            assert 1.5 < mean_ratio < 2.5, (
                f"Born scaling ratio should be ~2, got {mean_ratio:.2f}"
            )

    def test_trace_is_causal(self) -> None:
        """Trace should be ~0 well before the two-way water travel time.

        Uses a generous margin (50% of t_water) because band-limited IFFT
        with the Ricker wavelet produces side lobes near the arrival.
        """
        geom = SlabGeometry(M=2, N_z=1, a=1.0)
        mat = uniform_slab_material(geom, SED_REF, CONTRAST)
        cfg = _make_config(geom, mat, nw=128, T=1.0, f_min=5.0, f_max=80.0)

        result = compute_ocean_bottom_reflection(cfg, progress=False)

        # Two-way travel time through water
        t_water = 2.0 * cfg.water_depth / cfg.water_alpha
        early = result.time < t_water * 0.5  # generous margin for wavelet sidelobes
        if np.any(early):
            early_energy = np.max(np.abs(result.trace_total[early]))
            peak_energy = np.max(np.abs(result.trace_total))
            if peak_energy > 0:
                ratio = early_energy / peak_energy
                assert ratio < 0.15, f"Pre-arrival energy too high: {ratio:.4f}"

    def test_energy_bound(self) -> None:
        """|R_total(ω)| < 1 for all active frequencies."""
        geom = SlabGeometry(M=2, N_z=1, a=1.0)
        mat = uniform_slab_material(geom, SED_REF, CONTRAST)
        cfg = _make_config(geom, mat)

        result = compute_ocean_bottom_reflection(cfg, progress=False)

        # Check only active band where physics is resolved
        freq_hz = result.omega_real / (2.0 * np.pi)
        active = (freq_hz >= cfg.f_min) & (freq_hz <= cfg.f_max)
        if np.any(active):
            max_R = np.max(np.abs(result.R_total[active]))
            assert max_R < 1.0, f"|R_total| exceeds 1: max = {max_R:.4f}"

    def test_result_shapes(self) -> None:
        """All output arrays have correct shapes and dtypes."""
        geom = SlabGeometry(M=2, N_z=1, a=1.0)
        mat = uniform_slab_material(geom, SED_REF, WEAK_CONTRAST)
        cfg = _make_config(geom, mat, nw=32, T=0.5, f_min=10.0, f_max=50.0)

        result = compute_ocean_bottom_reflection(cfg, progress=False)

        nt = 2 * cfg.nw
        nwm = cfg.nw - 1

        assert result.time.shape == (nt,)
        assert result.trace_total.shape == (nt,)
        assert result.trace_homogeneous.shape == (nt,)
        assert result.R_bg.shape == (nwm,)
        assert result.R_slab.shape == (nwm,)
        assert result.R_total.shape == (nwm,)
        assert result.omega_real.shape == (nwm,)

        assert np.isrealobj(result.time)
        assert np.isrealobj(result.trace_total)
        assert np.isrealobj(result.trace_homogeneous)
        assert np.iscomplexobj(result.R_bg)
        assert np.iscomplexobj(result.R_slab)
        assert np.iscomplexobj(result.R_total)

        assert result.elapsed_seconds > 0
        assert result.config is cfg

    # ── New tests: Kennett embedding + free surface ──────────────────────

    def test_homogeneous_matches_kennett(self) -> None:
        """R_total (decomposed path) matches R_bg (kennett_layers) at zero contrast.

        R_bg comes from the full kennett_layers([water|sed|hs]) recursion.
        R_total comes from the decomposed path: kennett_layers([sed|hs]) +
        sediment phase + _kennett_water_step. With zero slab contrast these
        must agree, validating the decomposition.
        """
        geom = SlabGeometry(M=2, N_z=1, a=1.0)
        zero_contrast = MaterialContrast(Dlambda=0.0, Dmu=0.0, Drho=0.0)
        mat = uniform_slab_material(geom, SED_REF, zero_contrast)
        cfg = _make_config(geom, mat)

        result = compute_ocean_bottom_reflection(cfg, progress=False)

        # At active frequencies, the decomposed R_total should match R_bg
        # from the proven kennett_layers recursion
        freq_hz = result.omega_real / (2.0 * np.pi)
        active = (freq_hz >= cfg.f_min) & (freq_hz <= cfg.f_max)

        np.testing.assert_allclose(
            result.R_total[active],
            result.R_bg[active],
            rtol=1e-10,
            err_msg="Decomposed path should match kennett_layers at zero contrast",
        )

    def test_coupling_reduces_slab(self) -> None:
        """Interface coupling reduces slab contribution (T_d·T_u < 1)."""
        geom = SlabGeometry(M=2, N_z=1, a=1.0)
        mat = uniform_slab_material(geom, SED_REF, CONTRAST)
        cfg = _make_config(geom, mat)

        result = compute_ocean_bottom_reflection(cfg, progress=False)

        # R_total - R_bg = coupled slab; R_slab = raw (uncoupled)
        active = np.abs(result.R_slab) > 1e-15
        if np.any(active):
            coupled_slab = np.abs(result.R_total[active] - result.R_bg[active])
            raw_slab = np.abs(result.R_slab[active])
            # T_d · T_u < 1 at fluid-solid → coupling reduces contribution
            ratio = np.mean(coupled_slab / raw_slab)
            assert ratio < 1.0, (
                f"Coupled slab should be smaller than raw: ratio = {ratio:.4f}"
            )

    def test_free_surface_creates_multiples(self) -> None:
        """free_surface=True creates water-column multiples at late times."""
        geom = SlabGeometry(M=2, N_z=1, a=1.0)
        mat = uniform_slab_material(geom, SED_REF, CONTRAST)

        cfg_no_fs = _make_config(
            geom, mat, nw=128, T=1.0, f_min=5.0, f_max=80.0, free_surface=False
        )
        cfg_fs = _make_config(
            geom, mat, nw=128, T=1.0, f_min=5.0, f_max=80.0, free_surface=True
        )

        result_no_fs = compute_ocean_bottom_reflection(cfg_no_fs, progress=False)
        result_fs = compute_ocean_bottom_reflection(cfg_fs, progress=False)

        # First water-column multiple arrives at ~3× two-way water time
        t_water_2way = 2.0 * WATER_DEPTH / WATER_ALPHA
        t_mult = 3.0 * t_water_2way
        late = (result_fs.time > t_mult * 0.7) & (result_fs.time < t_mult * 1.3)

        if np.any(late):
            energy_fs = np.sum(result_fs.trace_total[late] ** 2)
            energy_no_fs = np.sum(result_no_fs.trace_total[late] ** 2)
            assert energy_fs > 2.0 * energy_no_fs, (
                f"Free surface should create significant multiples: "
                f"E_fs={energy_fs:.4e}, E_no_fs={energy_no_fs:.4e}"
            )

    def test_free_surface_off_no_multiples(self) -> None:
        """free_surface=False: negligible energy at water-column multiple time."""
        geom = SlabGeometry(M=2, N_z=1, a=1.0)
        mat = uniform_slab_material(geom, SED_REF, CONTRAST)
        cfg = _make_config(geom, mat, nw=128, T=1.0, f_min=5.0, f_max=80.0)

        result = compute_ocean_bottom_reflection(cfg, progress=False)

        # At ~3× two-way water time, should have negligible energy
        t_water_2way = 2.0 * WATER_DEPTH / WATER_ALPHA
        t_mult = 3.0 * t_water_2way
        late = (result.time > t_mult * 0.8) & (result.time < t_mult * 1.2)

        if np.any(late):
            late_energy = np.max(np.abs(result.trace_total[late]))
            peak_energy = np.max(np.abs(result.trace_total))
            if peak_energy > 0:
                ratio = late_energy / peak_energy
                assert ratio < 0.15, (
                    f"Without free surface, late energy should be small: {ratio:.4f}"
                )


class TestObliqueIncidence:
    """Tests for p>0 oblique incidence extension."""

    def test_p0_unchanged(self) -> None:
        """Explicit p=0.0 gives identical results to default (regression)."""
        geom = SlabGeometry(M=2, N_z=1, a=1.0)
        mat = uniform_slab_material(geom, SED_REF, CONTRAST)

        cfg_default = _make_config(geom, mat)
        cfg_p0 = _make_config(geom, mat, p=0.0)

        r_default = compute_ocean_bottom_reflection(cfg_default, progress=False)
        r_p0 = compute_ocean_bottom_reflection(cfg_p0, progress=False)

        np.testing.assert_allclose(
            r_default.R_total,
            r_p0.R_total,
            atol=1e-15,
            err_msg="Explicit p=0 should match default",
        )
        np.testing.assert_allclose(
            r_default.trace_total,
            r_p0.trace_total,
            atol=1e-15,
            err_msg="Explicit p=0 traces should match default",
        )

    def test_small_p_perturbation(self) -> None:
        """Small p gives R_PP close to p=0 (continuity in p)."""
        geom = SlabGeometry(M=2, N_z=1, a=1.0)
        mat = uniform_slab_material(geom, SED_REF, CONTRAST)

        # p=0 baseline
        cfg_p0 = _make_config(geom, mat, p=0.0)
        r_p0 = compute_ocean_bottom_reflection(cfg_p0, progress=False)

        # Small p: well below critical angle for all layers
        p_small = 1e-5  # s/m — very small angle
        cfg_ps = _make_config(geom, mat, p=p_small)
        r_ps = compute_ocean_bottom_reflection(cfg_ps, progress=False)

        # R_total should be close at active frequencies
        freq_hz = r_p0.omega_real / (2.0 * np.pi)
        active = (freq_hz >= cfg_p0.f_min) & (freq_hz <= cfg_p0.f_max)
        if np.any(active):
            np.testing.assert_allclose(
                np.abs(r_ps.R_total[active]),
                np.abs(r_p0.R_total[active]),
                rtol=0.01,
                err_msg=f"Small p={p_small} should give R_PP close to p=0",
            )

    def test_critical_angle_bound(self) -> None:
        """p >= 1/alpha_water raises ValueError (evanescent in water)."""
        geom = SlabGeometry(M=2, N_z=1, a=1.0)
        mat = uniform_slab_material(geom, SED_REF, WEAK_CONTRAST)

        # Exactly at critical angle
        p_critical = 1.0 / WATER_ALPHA
        cfg = _make_config(geom, mat, p=p_critical)

        import pytest

        with pytest.raises(ValueError, match="evanescent"):
            compute_ocean_bottom_reflection(cfg, progress=False)

    def test_oblique_energy_bound(self) -> None:
        """|R_total(ω)| < 1 for moderate oblique incidence."""
        geom = SlabGeometry(M=2, N_z=1, a=1.0)
        mat = uniform_slab_material(geom, SED_REF, CONTRAST)

        # ~10° incidence in water
        p = np.sin(np.radians(10.0)) / WATER_ALPHA
        cfg = _make_config(geom, mat, p=p)
        result = compute_ocean_bottom_reflection(cfg, progress=False)

        freq_hz = result.omega_real / (2.0 * np.pi)
        active = (freq_hz >= cfg.f_min) & (freq_hz <= cfg.f_max)
        if np.any(active):
            max_R = np.max(np.abs(result.R_total[active]))
            assert max_R < 1.0, f"|R_total| exceeds 1 at p>0: max = {max_R:.4f}"


class TestYAMLConfig:
    """Tests for YAML config loading."""

    def _write_yaml(self, tmp_path: Path, data: dict) -> Path:
        """Write a YAML dict to a temp file and return the path."""
        p = tmp_path / "config.yml"
        with p.open("w") as f:
            yaml.dump(data, f)
        return p

    def _valid_config_dict(self) -> dict:
        """Valid config in seismic units (km/s, g/cm³, GPa, km)."""
        return {
            "ocean": {
                "water_alpha": 1.5,  # km/s
                "water_rho": 1.025,  # g/cm³
                "water_depth": 0.07,  # km
                "free_surface": False,
            },
            "sediment": {"alpha": 2.0, "beta": 0.8, "rho": 1.8},
            "halfspace": {"alpha": 3.0, "beta": 1.7, "rho": 2.2},
            "slab": {
                "M": 2,
                "N_z": 1,
                "a": 0.001,  # km = 1 m
                "material": "uniform",
                "mean": {
                    "Dlambda": 1.3244,
                    "Dmu": 0.3116,
                    "Drho": 0.09,
                },  # GPa, g/cm³
            },
            "recording": {
                "f_peak": 30.0,
                "T": 0.5,
                "nw": 64,
                "f_min": 10.0,
                "f_max": 60.0,
            },
            "slowness": {"p": 0.0},  # s/km
        }

    def test_yaml_config_load(self, tmp_path: Path) -> None:
        """Round-trip: write YAML → load → converts seismic→SI correctly."""
        data = self._valid_config_dict()
        yml_path = self._write_yaml(tmp_path, data)

        cfg = load_ocean_bottom_config(yml_path)

        # Velocities: km/s → m/s
        assert cfg.water_alpha == 1500.0
        assert cfg.water_rho == 1025.0
        assert cfg.water_depth == 70.0
        assert cfg.sed_ref.alpha == 2000.0
        assert cfg.sed_ref.beta == 800.0
        assert cfg.sed_ref.rho == 1800.0
        assert cfg.hs_alpha == 3000.0
        assert cfg.hs_beta == 1700.0
        assert cfg.hs_rho == 2200.0
        # Length: km → m
        assert cfg.geometry.M == 2
        assert cfg.geometry.N_z == 1
        assert cfg.geometry.a == 1.0
        # Recording params unchanged
        assert cfg.f_peak == 30.0
        assert cfg.p == 0.0
        assert cfg.free_surface is False

    def test_yaml_config_with_p(self, tmp_path: Path) -> None:
        """YAML with p>0 (s/km) loads and converts to SI (s/m)."""
        data = self._valid_config_dict()
        data["slowness"]["p"] = 0.2  # s/km
        yml_path = self._write_yaml(tmp_path, data)

        cfg = load_ocean_bottom_config(yml_path)
        assert cfg.p == 0.0002  # s/m

    def test_yaml_config_missing_section(self, tmp_path: Path) -> None:
        """Missing required section raises ValueError with diagnostic."""
        data = self._valid_config_dict()
        del data["sediment"]
        yml_path = self._write_yaml(tmp_path, data)

        import pytest

        with pytest.raises(ValueError, match="Missing required config sections"):
            load_ocean_bottom_config(yml_path)

    def test_yaml_config_file_not_found(self) -> None:
        """Non-existent file raises FileNotFoundError."""
        import pytest

        with pytest.raises(FileNotFoundError):
            load_ocean_bottom_config("/nonexistent/path.yml")

    def test_yaml_mean_random_computes_inclusion(self, tmp_path: Path) -> None:
        """mean + random: inclusion contrast = mean / phi."""
        data = self._valid_config_dict()
        data["slab"]["material"] = "random"
        data["slab"]["phi"] = 0.4
        data["slab"]["seed"] = 7
        # mean Δλ = 0.5 GPa → inclusion Δλ = 0.5/0.4 = 1.25 GPa
        data["slab"]["mean"] = {"Dlambda": 0.5, "Dmu": 0.1, "Drho": 0.02}
        yml_path = self._write_yaml(tmp_path, data)

        cfg = load_ocean_bottom_config(yml_path)

        # Inclusion cubes should have Δλ = 0.5/0.4 = 1.25 GPa = 1.25e9 Pa
        inclusions = cfg.material.Dlambda[cfg.material.Dlambda > 0]
        if len(inclusions) > 0:
            np.testing.assert_allclose(inclusions[0], 0.5e9 / 0.4, rtol=1e-10)

    def test_yaml_contrast_still_works(self, tmp_path: Path) -> None:
        """Raw 'contrast' parameterization still accepted."""
        data = self._valid_config_dict()
        del data["slab"]["mean"]
        data["slab"]["contrast"] = {"Dlambda": 1.0, "Dmu": 0.5, "Drho": 0.05}
        yml_path = self._write_yaml(tmp_path, data)

        cfg = load_ocean_bottom_config(yml_path)
        # Uniform → every cube has the raw contrast
        np.testing.assert_allclose(cfg.material.Dlambda[0, 0, 0], 1.0e9)
        np.testing.assert_allclose(cfg.material.Dmu[0, 0, 0], 0.5e9)

    def test_yaml_both_mean_and_contrast_raises(self, tmp_path: Path) -> None:
        """Specifying both 'mean' and 'contrast' raises ValueError."""
        data = self._valid_config_dict()
        data["slab"]["contrast"] = {"Dlambda": 1.0, "Dmu": 0.5, "Drho": 0.05}
        # data already has "mean" from _valid_config_dict
        yml_path = self._write_yaml(tmp_path, data)

        import pytest

        with pytest.raises(ValueError, match="exactly one of"):
            load_ocean_bottom_config(yml_path)

    def test_yaml_neither_mean_nor_contrast_raises(self, tmp_path: Path) -> None:
        """Missing both 'mean' and 'contrast' raises ValueError."""
        data = self._valid_config_dict()
        del data["slab"]["mean"]
        yml_path = self._write_yaml(tmp_path, data)

        import pytest

        with pytest.raises(ValueError, match="exactly one of"):
            load_ocean_bottom_config(yml_path)
