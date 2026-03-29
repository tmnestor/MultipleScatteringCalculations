"""Tests for YAML solver configuration."""

from pathlib import Path
from textwrap import dedent

import pytest

from cubic_scattering.solver_config import (
    DeviceConfig,
    load_config,
)


@pytest.fixture
def tmp_config(tmp_path):
    """Write a valid slab config to a temp file and return its path."""

    def _write(content: str) -> Path:
        p = tmp_path / "config.yml"
        p.write_text(dedent(content))
        return p

    return _write


VALID_SLAB = """\
device:
  backend: cpu
  dtype: complex128
  memory_limit_gb: 4.0

solver:
  gmres_tol: 1.0e-6
  max_iter: 500
  initial_guess: born

problem:
  type: slab
  geometry:
    M: 4
    N_z: 2
    a: 1.0
  reference:
    alpha: 5000.0
    beta: 3000.0
    rho: 2500.0
  contrast:
    Dlambda: 2.0e9
    Dmu: 1.0e9
    Drho: 100.0
  frequency:
    omega: 150.0
  incident:
    k_hat: [1.0, 0.0, 0.0]
    wave_type: P
  slab:
    material_type: uniform
"""

VALID_SPHERE = """\
device:
  backend: cpu
  dtype: complex128

solver:
  gmres_tol: 1.0e-8
  max_iter: 200
  initial_guess: born

problem:
  type: sphere
  geometry:
    M: 1
    N_z: 1
    a: 0.5
  reference:
    alpha: 5000.0
    beta: 3000.0
    rho: 2500.0
  contrast:
    Dlambda: 2.0e9
    Dmu: 1.0e9
    Drho: 100.0
  frequency:
    omega: 150.0
  incident:
    k_hat: [1.0, 0.0, 0.0]
    wave_type: P
  sphere:
    radius: 0.5
    n_sub: 3
"""


class TestLoadConfig:
    def test_load_valid_slab(self, tmp_config):
        path = tmp_config(VALID_SLAB)
        config = load_config(path)
        assert config.problem.type == "slab"
        assert config.device.backend == "cpu"
        assert config.solver.gmres_tol == 1e-6
        assert config.solver.max_iter == 500

    def test_load_valid_sphere(self, tmp_config):
        path = tmp_config(VALID_SPHERE)
        config = load_config(path)
        assert config.problem.type == "sphere"
        assert config.problem.sphere["radius"] == 0.5
        assert config.problem.sphere["n_sub"] == 3

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config(Path("/nonexistent/config.yml"))

    def test_defaults_applied(self, tmp_config):
        minimal = """\
        problem:
          type: slab
          geometry:
            M: 2
            N_z: 1
            a: 1.0
          reference:
            alpha: 5000.0
            beta: 3000.0
            rho: 2500.0
          contrast:
            Dlambda: 1.0e9
            Dmu: 1.0e9
            Drho: 100.0
          frequency:
            omega: 100.0
          incident:
            k_hat: [1.0, 0.0, 0.0]
            wave_type: P
        """
        path = tmp_config(minimal)
        config = load_config(path)
        assert config.device.backend == "auto"
        assert config.device.dtype == "auto"
        assert config.solver.initial_guess == "born"


class TestValidation:
    def test_invalid_problem_type(self, tmp_config):
        bad = VALID_SLAB.replace("type: slab", "type: cube")
        path = tmp_config(bad)
        with pytest.raises(ValueError, match="Invalid problem type"):
            load_config(path)

    def test_missing_reference_field(self, tmp_config):
        bad = VALID_SLAB.replace("    alpha: 5000.0\n", "")
        path = tmp_config(bad)
        with pytest.raises(ValueError, match="problem.reference.alpha"):
            load_config(path)

    def test_negative_omega(self, tmp_config):
        bad = VALID_SLAB.replace("omega: 150.0", "omega: -1.0")
        path = tmp_config(bad)
        with pytest.raises(ValueError, match="omega must be positive"):
            load_config(path)

    def test_invalid_wave_type(self, tmp_config):
        bad = VALID_SLAB.replace("wave_type: P", "wave_type: L")
        path = tmp_config(bad)
        with pytest.raises(ValueError, match="Invalid wave_type"):
            load_config(path)

    def test_invalid_backend(self, tmp_config):
        bad = VALID_SLAB.replace("backend: cpu", "backend: tpu")
        path = tmp_config(bad)
        with pytest.raises(ValueError, match="Invalid backend"):
            load_config(path)

    def test_invalid_dtype(self, tmp_config):
        bad = VALID_SLAB.replace("dtype: complex128", "dtype: float32")
        path = tmp_config(bad)
        with pytest.raises(ValueError, match="Invalid dtype"):
            load_config(path)

    def test_sphere_missing_sphere_section(self, tmp_config):
        bad = VALID_SPHERE.replace("  sphere:\n    radius: 0.5\n    n_sub: 3\n", "")
        path = tmp_config(bad)
        with pytest.raises(ValueError, match="requires a 'sphere' section"):
            load_config(path)

    def test_invalid_initial_guess(self, tmp_config):
        bad = VALID_SLAB.replace("initial_guess: born", "initial_guess: random")
        path = tmp_config(bad)
        with pytest.raises(ValueError, match="Invalid initial_guess"):
            load_config(path)


class TestProblemConfig:
    def test_to_reference_medium(self, tmp_config):
        path = tmp_config(VALID_SLAB)
        config = load_config(path)
        ref = config.problem.to_reference_medium()
        assert ref.alpha == 5000.0
        assert ref.beta == 3000.0
        assert ref.rho == 2500.0

    def test_to_material_contrast(self, tmp_config):
        path = tmp_config(VALID_SLAB)
        config = load_config(path)
        c = config.problem.to_material_contrast()
        assert c.Dlambda == 2e9
        assert c.Dmu == 1e9
        assert c.Drho == 100.0

    def test_to_slab_geometry(self, tmp_config):
        path = tmp_config(VALID_SLAB)
        config = load_config(path)
        geom = config.problem.to_slab_geometry()
        assert geom.M == 4
        assert geom.N_z == 2
        assert geom.a == 1.0


class TestDeviceConfig:
    def test_resolve_cpu(self):
        dc = DeviceConfig(backend="cpu")
        assert dc.resolve_device().type == "cpu"

    def test_resolve_auto(self):
        dc = DeviceConfig(backend="auto")
        device = dc.resolve_device()
        assert device.type in ("mps", "cuda", "cpu")

    def test_resolve_invalid_backend(self):
        dc = DeviceConfig(backend="tpu")
        with pytest.raises(ValueError, match="Invalid backend"):
            dc.resolve_device()
