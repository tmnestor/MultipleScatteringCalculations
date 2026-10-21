"""The compiled product of the transformed kernel with the transformed sources (the graded FFT matvec).

yh[row, f] = sum_c kh[row, c, f] sh[c, f] over the existing layout, in OpenMP over the Fourier points: it
must equal the NumPy product it replaces (``np.einsum("cf,cf->f", ...)`` per row) to round-off for a
complex128 and a complex64 kernel, read the kernel in place without copying it, and not depend on the
number of threads.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

from cubic_scattering.graded_voxel.kernel_fortran import apply_hat_fortran


def _case(dtype, rows=12, cols=30, nf=4001, seed=0):
    rng = np.random.default_rng(seed)
    kh = (rng.normal(size=(rows, cols, nf)) + 1j * rng.normal(size=(rows, cols, nf))).astype(dtype)
    sh = rng.normal(size=(cols, nf)) + 1j * rng.normal(size=(cols, nf))
    return kh, sh


def _numpy(kh, sh):
    return np.stack([np.einsum("cf,cf->f", kh[r], sh) for r in range(kh.shape[0])])


@pytest.mark.parametrize("dtype", [np.complex128, np.complex64])
def test_compiled_product_equals_the_numpy_product(dtype) -> None:
    kh, sh = _case(dtype)
    got = apply_hat_fortran(kh, sh)
    want = _numpy(kh, sh)
    assert got.dtype == np.complex128 and got.shape == want.shape
    assert np.abs(got - want).max() / np.abs(want).max() < 1e-14


def test_a_five_dimensional_kernel_as_the_solve_stores_it() -> None:
    kh, sh = _case(np.complex64, nf=5**3)
    kh5 = kh.reshape(kh.shape[0], kh.shape[1], 5, 5, 5)
    np.testing.assert_array_equal(apply_hat_fortran(kh5, sh), apply_hat_fortran(kh, sh))


def test_the_kernel_is_read_in_place() -> None:
    """A memory-mapped 6.5 GB kernel must not be copied: the call takes the array as it lies."""
    from cubic_scattering.graded_voxel import kernel_fortran as KF

    kh, sh = _case(np.complex64)
    seen = {}
    real = KF._apply_hat

    def spy(k, s, *args):
        seen["shares_memory"] = np.shares_memory(k, kh)
        return real(k, s, *args)

    KF._apply_hat = spy
    try:
        apply_hat_fortran(kh, sh)
    finally:
        KF._apply_hat = real
    assert seen["shares_memory"]


def test_result_is_independent_of_thread_count() -> None:
    code = (
        "import numpy as np, sys; sys.path.insert(0, '.');"
        "from cubic_scattering.tests.test_kernel_apply_fortran import _case;"
        "from cubic_scattering.graded_voxel.kernel_fortran import apply_hat_fortran;"
        "kh, sh = _case(np.complex64, nf=20011);"
        "sys.stdout.buffer.write(apply_hat_fortran(kh, sh).tobytes())"
    )
    outs = []
    for threads in ("1", "4"):
        env = dict(os.environ, OMP_NUM_THREADS=threads)
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, env=env, check=True)
        outs.append(done.stdout)
    assert outs[0] == outs[1]
