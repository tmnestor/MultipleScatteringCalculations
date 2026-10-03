"""The numerics configuration fails fast, with a full diagnostic, on a missing file, key or value."""

import numpy as np
import pytest

from cubic_scattering.effective_contrasts import ReferenceMedium
from cubic_scattering.graded_voxel import kernel as K


def assert_diagnostic_error(message: str, where: str) -> None:
    """The four elements: what is wrong, where to fix it, a valid example, how to recover."""
    assert "point_kernel.backend" in message or "missing" in message  # what
    assert where in message  # where: the absolute path
    assert "backend: fortran" in message and "python, fortran" in message  # a valid example and the values
    assert "Fix:" in message  # how to recover


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    path = tmp_path / "numerics.yml"
    monkeypatch.setattr(K, "NUMERICS_YAML", path)
    K.point_kernel_backend.cache_clear()
    yield path
    K.point_kernel_backend.cache_clear()


def test_missing_file(config_file) -> None:
    with pytest.raises(ValueError) as err:
        K.point_kernel_backend()
    assert_diagnostic_error(str(err.value), str(config_file))


@pytest.mark.parametrize("text", ["{}\n", "point_kernel: {}\n", "point_kernel:\n  backend: cuda\n"])
def test_missing_or_invalid_key(config_file, text: str) -> None:
    config_file.write_text(text)
    with pytest.raises(ValueError) as err:
        K.point_kernel_backend()
    assert_diagnostic_error(str(err.value), str(config_file))


@pytest.mark.parametrize("backend", ["python", "fortran"])
def test_dispatch_follows_the_configuration(config_file, backend: str) -> None:
    config_file.write_text(f"point_kernel:\n  backend: {backend}\n")
    ref = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
    X = np.array([[1.0, 2.0, 3.0], [30.0, -20.0, 10.0]])
    from cubic_scattering.graded_voxel.kernel_fortran import kernel_9x9_fortran

    expected = (
        K.kernel_9x9_python(X, 60.0, ref) if backend == "python" else kernel_9x9_fortran(X, 60.0, ref)
    )
    np.testing.assert_array_equal(K.kernel_9x9(X, 60.0, ref), expected)


def test_the_shipped_configuration_is_valid() -> None:
    K.point_kernel_backend.cache_clear()
    assert K.point_kernel_backend() in ("python", "fortran")
