"""The compiled point kernel against the Python ``kernel_9x9`` and the 50-digit reference.

The Python kernel is the golden reference: the compiled one is a term-for-term transcription, so they
must agree to round-off at every k r, on both branches and across the switch between them, for real
and complex omega, with the static and dynamic parts separately as well as together. The 50-digit
evaluation of ``test_greens_small_kr`` is checked too, so that agreement between the two
implementations is not agreement in a shared error. And the result must not depend on the number of
OpenMP threads.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

from cubic_scattering.graded_voxel.kernel import SERIES_LIMIT
from cubic_scattering.graded_voxel.kernel import kernel_9x9_python as kernel_9x9
from cubic_scattering.graded_voxel.kernel_fortran import kernel_9x9_fortran
from cubic_scattering.resonance_tmatrix import _voigt_contract
from cubic_scattering.tests.test_greens_small_kr import DIRECTION, ETAS, OMEGA, REF, _reference_tensors

KR = [1e-4, 1e-3, 1e-2, 0.1, 0.3, SERIES_LIMIT * (1 - 1e-9), SERIES_LIMIT * (1 + 1e-9), 0.7, 1.0, 3.0, 10.0]


def _points(omega: complex, n: int = 400) -> np.ndarray:
    """Separations with |k_S| r spread log-uniformly over 1e-4 to 20, in random directions."""
    rng = np.random.default_rng(3)
    kr = 10 ** rng.uniform(-4, np.log10(20.0), n)
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return d * (kr / abs(omega / REF.beta))[:, None]


def _rel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-point relative difference, against each point's largest entry."""
    return np.max(np.abs(a - b), axis=(1, 2)) / np.max(np.abs(b), axis=(1, 2))


@pytest.mark.parametrize("eta", ETAS)
@pytest.mark.parametrize("parts", [(True, True), (True, False), (False, True)])
def test_compiled_equals_python(eta: float, parts: tuple[bool, bool]) -> None:
    omega = OMEGA * (1 + 1j * eta) if eta else OMEGA
    X = _points(omega)
    static, dynamic = parts
    got = kernel_9x9_fortran(X, omega, REF, static=static, dynamic=dynamic)
    want = kernel_9x9(X, omega, REF, static=static, dynamic=dynamic)
    assert float(np.max(_rel(got, want))) < 1e-13


@pytest.mark.parametrize("eta", ETAS)
@pytest.mark.parametrize("kr", KR)
def test_compiled_holds_round_off_against_50_digits(kr: float, eta: float) -> None:
    omega = OMEGA * (1 + 1j * eta) if eta else OMEGA
    x = DIRECTION * (kr / abs(omega / REF.beta))
    G, Gd, Gdd = _reference_tensors(x, omega)
    C, H, S = _voigt_contract(Gd, Gdd)
    ref_block = np.block([[G, C], [H, S]])
    got = kernel_9x9_fortran(x[None], omega, REF)[0]
    assert float(np.max(np.abs(got - ref_block)) / np.max(np.abs(ref_block))) < 5e-14


def test_zero_separation_raises() -> None:
    with pytest.raises(ValueError, match="r = 0"):
        kernel_9x9_fortran(np.zeros((1, 3)), OMEGA, REF)


def test_result_is_independent_of_thread_count() -> None:
    """Bitwise: each point is computed once, by one thread, with no reduction across points."""
    code = (
        "import numpy as np, sys; sys.path.insert(0, '.');"
        "from cubic_scattering.tests.test_kernel_fortran import _points, OMEGA, REF;"
        "from cubic_scattering.graded_voxel.kernel_fortran import kernel_9x9_fortran;"
        "sys.stdout.buffer.write(kernel_9x9_fortran(_points(OMEGA, 2000), OMEGA, REF).tobytes())"
    )
    outs = []
    for threads in ("1", "4"):
        env = dict(os.environ, OMP_NUM_THREADS=threads)
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, env=env, check=True)
        outs.append(done.stdout)
    assert outs[0] == outs[1]
