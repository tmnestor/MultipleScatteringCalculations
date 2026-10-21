"""The compiled cell-averaged pair block against the Python reference in ``cell_averaged_pair``.

``averaged_pair_block_9x9`` (Python) is the golden reference: the compiled routine is a term-for-term
transcription of its receiver-cell quadrature, so the two must agree to round-off for face, edge,
corner, near and far offsets, on both branches of the difference ladder and across the switch between
them, for real and complex omega, and with the Gauss order chosen automatically or given explicitly.
The batched entry point must also respect the memo and the receiver-clear check, follow
``cell_average.backend`` in ``numerics.yml``, and give the sphere FFT kernel unchanged.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

from cubic_scattering import cell_averaged_pair as CAP
from cubic_scattering.cell_average_fortran import averaged_pair_blocks_9x9_fortran, averaged_tensors_fortran
from cubic_scattering.effective_contrasts import ReferenceMedium

REF = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
OFFSETS = np.array(
    [
        [1, 0, 0],  # face: Gauss order 16
        [0, -1, 0],
        [1, 1, 0],  # edge: 12
        [-1, 0, 1],
        [1, 1, 1],  # corner: 8
        [2, 1, 0],  # near: 8
        [3, -1, 2],  # 6
        [5, 3, -2],
        [9, 0, 4],
    ],
    dtype=float,
)
# |k_S| d: tiny (series everywhere, where the mode difference cancels hardest), straddling the series
# limit of 0.5 across the near shells, and large (closed forms almost everywhere)
KD = [1e-3, 0.3, 1.0, 4.0]
ETAS = [0.0, 0.05]
D = 0.25


def _omega(kd: float, eta: float) -> complex:
    return (kd / D) * REF.beta * (1 + 1j * eta) if eta else (kd / D) * REF.beta


def _rel(got: np.ndarray, want: np.ndarray) -> float:
    return float(np.max(np.abs(got - want)) / np.max(np.abs(want)))


@pytest.mark.slow
@pytest.mark.parametrize("eta", ETAS)
@pytest.mark.parametrize("kd", KD)
def test_tensors_equal_python(kd: float, eta: float) -> None:
    omega = _omega(kd, eta)
    kp, ks = omega / REF.alpha, omega / REF.beta
    S = OFFSETS * D
    ng = np.array([CAP.auto_n_gauss(s, D) for s in S])
    dd2, dd3, dd4, ds0, ds1, ds2 = averaged_tensors_fortran(S, kp, ks, D, ng)
    for i, s in enumerate(S):
        want_d = CAP.averaged_pair_difference_tensors(s, kp, ks, D)
        want_s = CAP.averaged_pair_scalar_tensors(s, ks, D)
        assert _rel(dd2[i], want_d[2]) < 1e-12
        assert _rel(dd3[i], want_d[3]) < 1e-12
        assert _rel(dd4[i], want_d[4]) < 1e-12
        assert _rel(ds0[i], want_s[0]) < 1e-12
        assert _rel(ds1[i], want_s[1]) < 1e-12
        assert _rel(ds2[i], want_s[2]) < 1e-12


@pytest.mark.slow
@pytest.mark.parametrize("eta", ETAS)
@pytest.mark.parametrize("kd", KD)
def test_blocks_equal_python(kd: float, eta: float) -> None:
    omega = _omega(kd, eta)
    S = OFFSETS * D
    CAP.clear_block_cache()
    got = averaged_pair_blocks_9x9_fortran(S, omega, REF, D)
    CAP.clear_block_cache()
    for i, s in enumerate(S):
        assert _rel(got[i], CAP.averaged_pair_block_9x9(s, omega, REF, D)) < 1e-12


@pytest.mark.parametrize("n_gauss", [4, 10])
def test_explicit_gauss_order_is_honoured(n_gauss: int) -> None:
    omega = _omega(0.3, 0.0)
    S = OFFSETS[:4] * D
    CAP.clear_block_cache()
    got = averaged_pair_blocks_9x9_fortran(S, omega, REF, D, n_gauss=n_gauss)
    CAP.clear_block_cache()
    for i, s in enumerate(S):
        assert _rel(got[i], CAP.averaged_pair_block_9x9(s, omega, REF, D, n_gauss=n_gauss)) < 1e-12


def test_off_lattice_separation_is_computed_not_cached() -> None:
    omega = _omega(0.3, 0.0)
    s = (OFFSETS[2] + 0.37) * D
    CAP.clear_block_cache()
    got = averaged_pair_blocks_9x9_fortran(s[None], omega, REF, D)[0]
    assert len(CAP._BLOCK_CACHE) == 0
    assert _rel(got, CAP.averaged_pair_block_9x9(s, omega, REF, D)) < 1e-12


def test_blocks_fill_and_reuse_the_memo() -> None:
    omega = _omega(0.3, 0.0)
    S = OFFSETS * D
    CAP.clear_block_cache()
    first = averaged_pair_blocks_9x9_fortran(S, omega, REF, D)
    assert len(CAP._BLOCK_CACHE) == len(S)
    first[0, 0, 0] = 99.0  # the caller owns its copy: writing to it must not reach the memo
    again = averaged_pair_blocks_9x9_fortran(S, omega, REF, D)
    CAP.clear_block_cache()
    assert again[0, 0, 0] != 99.0
    assert _rel(again[0], CAP.averaged_pair_block_9x9(S[0], omega, REF, D)) < 1e-12


def test_receiver_reaching_the_source_raises() -> None:
    with pytest.raises(ValueError, match="receiver cell reaches the source"):
        averaged_pair_blocks_9x9_fortran(np.array([[0.4 * D, 0.0, 0.0]]), _omega(0.3, 0.0), REF, D)


def test_result_is_independent_of_thread_count() -> None:
    """Bitwise: each offset is averaged by one thread, with no reduction across offsets."""
    code = (
        "import numpy as np, sys; sys.path.insert(0, '.');"
        "from cubic_scattering.tests.test_cell_average_fortran import OFFSETS, D, REF, _omega;"
        "from cubic_scattering.cell_average_fortran import averaged_pair_blocks_9x9_fortran;"
        "sys.stdout.buffer.write(averaged_pair_blocks_9x9_fortran(OFFSETS * D, _omega(1.0, 0.05), REF, D)"
        ".tobytes())"
    )
    outs = []
    for threads in ("1", "4"):
        env = dict(os.environ, OMP_NUM_THREADS=threads)
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, env=env, check=True)
        outs.append(done.stdout)
    assert outs[0] == outs[1]


# ---- the backend switch in numerics.yml -------------------------------------------------------------


def assert_diagnostic_error(message: str, where: str) -> None:
    """The four elements: what is wrong, where to fix it, a valid example, how to recover."""
    assert "cell_average.backend" in message or "missing" in message  # what
    assert where in message  # where: the absolute path
    assert "backend: fortran" in message and "python, fortran" in message  # a valid example and the values
    assert "Fix:" in message  # how to recover


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    path = tmp_path / "numerics.yml"
    monkeypatch.setattr(CAP, "NUMERICS_YAML", path)
    CAP.cell_average_backend.cache_clear()
    yield path
    CAP.cell_average_backend.cache_clear()


def test_missing_file(config_file) -> None:
    with pytest.raises(ValueError) as err:
        CAP.cell_average_backend()
    assert_diagnostic_error(str(err.value), str(config_file))


@pytest.mark.parametrize("text", ["{}\n", "cell_average: {}\n", "cell_average:\n  backend: cuda\n"])
def test_missing_or_invalid_key(config_file, text: str) -> None:
    config_file.write_text(text)
    with pytest.raises(ValueError) as err:
        CAP.cell_average_backend()
    assert_diagnostic_error(str(err.value), str(config_file))


@pytest.mark.slow
@pytest.mark.parametrize("backend", ["python", "fortran"])
def test_dispatch_follows_the_configuration(config_file, backend: str) -> None:
    config_file.write_text(f"cell_average:\n  backend: {backend}\n")
    omega = _omega(0.3, 0.0)
    S = OFFSETS[:3] * D
    CAP.clear_block_cache()
    expected = (
        np.stack([CAP.averaged_pair_block_9x9(s, omega, REF, D) for s in S])
        if backend == "python"
        else averaged_pair_blocks_9x9_fortran(S, omega, REF, D)
    )
    CAP.clear_block_cache()
    np.testing.assert_array_equal(CAP.averaged_pair_blocks_9x9(S, omega, REF, D), expected)
    CAP.clear_block_cache()


def test_the_shipped_configuration_is_valid() -> None:
    CAP.cell_average_backend.cache_clear()
    assert CAP.cell_average_backend() in ("python", "fortran")


# ---- the consumer: the sphere FFT kernel ---------------------------------------------------------


@pytest.mark.slow
def test_sphere_fft_kernel_is_unchanged(monkeypatch) -> None:
    """The batched build gives the kernel the per-offset Python loop gave, to round-off."""
    from cubic_scattering.effective_contrasts import MaterialContrast, compute_cube_tmatrix
    from cubic_scattering.resonance_tmatrix import _sub_cell_tmatrix_9x9
    from cubic_scattering.sphere_scattering_fft import _build_fft_kernel, _build_grid_index_map

    contrast = MaterialContrast(Dlambda=2e9, Dmu=1e9, Drho=100.0)
    n_sub, omega = 4, 3000.0
    _, _, a_sub = _build_grid_index_map(0.5, n_sub)
    T_loc = _sub_cell_tmatrix_9x9(compute_cube_tmatrix(omega, a_sub, REF, contrast), omega, a_sub)

    CAP.clear_block_cache()
    monkeypatch.setattr(CAP, "cell_average_backend", lambda: "python")
    want = _build_fft_kernel(n_sub, a_sub, T_loc, omega, REF)
    CAP.clear_block_cache()
    monkeypatch.setattr(CAP, "cell_average_backend", lambda: "fortran")
    got = _build_fft_kernel(n_sub, a_sub, T_loc, omega, REF)
    CAP.clear_block_cache()
    assert _rel(got, want) < 1e-12
