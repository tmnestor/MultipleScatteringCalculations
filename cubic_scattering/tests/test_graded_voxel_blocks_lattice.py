"""The canonical far blocks on a shared lattice: ``coupling_blocks_far_lattice`` against the batched ones.

Neighbouring offsets evaluate the kernel at the same points (the lower Gauss piece of offset o is the upper
piece of o - 1, on h (2m + 2t_j)), and the piece weights are separable by axis, so every canonical
non-touching block of a grid follows from one evaluation of the kernel on a lattice and axis-by-axis
contractions. The same Gauss rule, so the same blocks to round-off: at every distance band of
``gauss_order`` (n_sub = 7 reaches distances 2 to 6), at a frequency that raises the order, at a complex
frequency, and for the bases of p = 1, r = 2 and p = r = 2 (20 and 35 source monomials).
"""

import itertools

import numpy as np
import pytest

from cubic_scattering.effective_contrasts import ReferenceMedium
from cubic_scattering.graded_voxel import blocks as B

REF = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
N_SUB, H = 7, 10.0 / 7


def _far(n_sub: int) -> list[tuple[int, int, int]]:
    return [c for c in itertools.product(range(n_sub), repeat=3) if c[0] >= c[1] >= c[2] and c[0] > 1]


@pytest.mark.parametrize(
    ("omega", "n_source", "n_test"),
    [
        (0.5 * REF.beta / 10.0, 10, 4),  # k_S h = 0.07: the orders by distance alone
        (1.6 * REF.beta / H, 10, 4),  # k_S h = 1.6: the frequency raises the far order to 10
        (3.0 * REF.beta / 10.0 * (1 + 0.05j), 10, 4),  # complex frequency
        (0.5 * REF.beta / 10.0, 20, 4),  # p = 1, r = 2
        (0.5 * REF.beta / 10.0, 35, 10),  # p = r = 2: the test functions converted from monomials
    ],
)
def test_lattice_blocks_equal_the_batched_blocks(omega, n_source, n_test) -> None:
    far = _far(N_SUB)
    got = B.coupling_blocks_far_lattice(N_SUB, H, omega, REF, n_source, n_test)
    assert sorted(got) == far
    want = B.coupling_blocks_far(np.array(far), H, omega, REF, n_source, n_test)
    scale = np.abs(want).max()
    for k, off in enumerate(far):
        assert np.abs(got[off] - want[k]).max() <= 1e-13 * scale, off


@pytest.mark.parametrize("backend", ["fortran", "python"])
def test_the_transposed_kernel_is_the_kernel_transposed(monkeypatch, backend) -> None:
    """kernel_9x9(..., transposed=True) is each block transposed, a contiguous (N, 9, 9) array; on the
    compiled route it is the routine's own output, taken without the copy that transposes it."""
    from cubic_scattering.graded_voxel import kernel as KN

    monkeypatch.setattr(KN, "point_kernel_backend", lambda: backend)
    X = np.random.default_rng(4).normal(size=(50, 3)) * 7.0
    omega = 900.0 * (1 + 0.03j)
    want = KN.kernel_9x9(X, omega, REF).transpose(0, 2, 1)
    got = KN.kernel_9x9(X, omega, REF, transposed=True)
    np.testing.assert_array_equal(got, want)
    assert got.shape == (50, 9, 9) and (backend == "python" or got.flags["C_CONTIGUOUS"])


def _lattice_case():
    rng = np.random.default_rng(9)
    x0 = 3.0 + rng.uniform(0, 1, 6)
    x1 = rng.uniform(-1, 1, 4)
    x2 = rng.uniform(-1, 1, 5)
    w = rng.normal(size=(3, 6))
    return x0, x1, x2, w


def test_the_fused_axis0_sum_equals_the_kernel_contracted_afterwards() -> None:
    from cubic_scattering.graded_voxel.kernel import kernel_9x9
    from cubic_scattering.graded_voxel.kernel_fortran import lattice_axis0_fortran

    x0, x1, x2, w = _lattice_case()
    omega = 900.0 * (1 + 0.03j)
    X = np.stack(np.meshgrid(x0, x1, x2, indexing="ij"), -1).reshape(-1, 3)
    P = kernel_9x9(X, omega, REF, transposed=True).reshape(len(x0), -1)
    want = (w @ P).reshape(3, len(x1), len(x2), 81)
    got = lattice_axis0_fortran(x0, x1, x2, w, omega, REF)
    assert got.shape == want.shape and got.flags["C_CONTIGUOUS"]
    assert np.abs(got - want).max() <= 1e-14 * np.abs(want).max()


def test_the_fused_axis0_sum_is_independent_of_thread_count() -> None:
    import os
    import subprocess
    import sys

    code = (
        'import numpy as np, sys; sys.path.insert(0, ".");'
        "from cubic_scattering.tests.test_graded_voxel_blocks_lattice import _lattice_case, REF;"
        "from cubic_scattering.graded_voxel.kernel_fortran import lattice_axis0_fortran;"
        "x0, x1, x2, w = _lattice_case();"
        "sys.stdout.buffer.write(lattice_axis0_fortran(x0, x1, x2, w, 900.0, REF).tobytes())"
    )
    outs = [
        subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            check=True,
            env=dict(os.environ, OMP_NUM_THREADS=n),
        ).stdout
        for n in ("1", "4")
    ]
    assert outs[0] == outs[1]


def test_the_lattice_blocks_through_the_reference_kernel(monkeypatch) -> None:
    """The reference backend contracts the kernel's values afterwards; the compiled one fuses the sum."""
    from cubic_scattering.graded_voxel import kernel as KN

    omega = 0.5 * REF.beta / 10.0
    fused = B.coupling_blocks_far_lattice(5, H, omega, REF)
    monkeypatch.setattr(KN, "point_kernel_backend", lambda: "python")
    monkeypatch.setattr(B, "point_kernel_backend", lambda: "python")
    plain = B.coupling_blocks_far_lattice(5, H, omega, REF)
    scale = max(np.abs(v).max() for v in plain.values())
    assert max(np.abs(fused[k] - plain[k]).max() for k in plain) <= 1e-13 * scale
