"""The cube group in Fourier space: the transformed kernel stored at the canonical k only.

K^(Q k) = R(Q) K^(k) R(Q)^T for the 48 signed permutations (R the signed permutation of the rows and of
the columns), so the kernel at one k per orbit, the canonical offsets a >= b >= d >= 0, determines it
everywhere. The canonical kernel must equal the full kernel at those k, the product through it must equal
the full product to round-off in both precisions and not depend on the thread count, and the solve with
it must equal the solve with the full kernel.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

from cubic_scattering.graded_voxel.fft import (
    canonical_blocks,
    canonical_transformed_kernel,
    fourier_symmetry_tables,
    transformed_kernel,
)
from cubic_scattering.graded_voxel.kernel_fortran import apply_canonical_fortran, apply_hat_fortran
from cubic_scattering.tests.test_graded_voxel_fft import CON, KHAT, OMEGA, REF, H, _profile

N_SUB, NA, NC = 4, 4, 10


@pytest.fixture(scope="module")  # once per module: a second alone, 20 s under parallel load
def kernels():
    canonical = canonical_blocks(N_SUB, H, OMEGA, REF)
    full = transformed_kernel(canonical, N_SUB, NA, NC)
    return canonical, full


def _flat(k, npad):
    return (k[0] * npad + k[1]) * npad + k[2]


def _check_canonical(canonical, full, n_sub, na, nc, **kw) -> None:
    """The canonical kernel equals the full kernel at the canonical k, to round-off: it is built from one
    array per orbit of index pairs, on the non-negative octant, by cosine and sine sums, not by the FFT."""
    npad = 2 * n_sub - 1
    rows, cols = na * 9, nc * 9
    khc = canonical_transformed_kernel(canonical, n_sub, na, nc, **kw)
    keys = sorted(canonical)
    assert khc.shape == (len(keys), cols, rows) and khc.dtype == np.complex128
    flat = full.reshape(rows, cols, -1)
    want = np.stack([flat[:, :, _flat(k, npad)].T for k in keys])
    assert np.abs(khc - want).max() <= 1e-13 * np.abs(want).max()


def test_the_canonical_kernel_is_the_full_kernel_at_the_canonical_k(kernels) -> None:
    canonical, full = kernels
    _check_canonical(canonical, full, N_SUB, NA, NC)


def test_the_canonical_kernel_in_any_grouping_of_the_arrays(kernels) -> None:
    canonical, full = kernels
    _check_canonical(canonical, full, N_SUB, NA, NC, budget_bytes=1.0)  # one array orbit at a time


@pytest.mark.parametrize(("n_sub", "na", "nc"), [(5, 4, 10), (5, 1, 1), (3, 4, 4)])
def test_the_canonical_kernel_for_odd_grids_and_other_degrees(n_sub, na, nc) -> None:
    canonical = canonical_blocks(n_sub, H, OMEGA, REF)
    _check_canonical(canonical, transformed_kernel(canonical, n_sub, na, nc), n_sub, na, nc)


def test_the_tables_cover_every_point_with_its_orbit() -> None:
    q_f, k_f, order, pr, sr, pc, sc = fourier_symmetry_tables(N_SUB, NA, NC, 10, 4)
    npad = 2 * N_SUB - 1
    assert q_f.shape == k_f.shape == order.shape == (npad**3,)
    assert np.all(np.diff(k_f[order]) >= 0)  # the points grouped by their canonical k
    assert sorted(np.unique(k_f)) == list(range(20))
    for perm in (*pr, *pc):  # every row and column map is a permutation
        np.testing.assert_array_equal(np.sort(perm), np.arange(len(perm)))
    assert set(np.unique(np.concatenate([sr.ravel(), sc.ravel()]))) <= {-1.0, 1.0}


@pytest.mark.parametrize("dtype", [np.complex128, np.complex64])
def test_the_canonical_product_equals_the_full_product(kernels, dtype) -> None:
    canonical, full = kernels
    rng = np.random.default_rng(3)
    cols, f = NC * 9, (2 * N_SUB - 1) ** 3
    sh = rng.normal(size=(cols, f)) + 1j * rng.normal(size=(cols, f))
    want = apply_hat_fortran(full.astype(dtype), sh)  # (rows, F)
    khc = canonical_transformed_kernel(canonical, N_SUB, NA, NC).astype(dtype)
    got = apply_canonical_fortran(
        khc, np.ascontiguousarray(sh.T), fourier_symmetry_tables(N_SUB, NA, NC, 10, 4)
    )
    assert got.shape == (f, NA * 9) and got.dtype == np.complex128
    assert np.abs(got.T - want).max() <= 1e-13 * np.abs(want).max()


def test_the_canonical_product_is_independent_of_thread_count() -> None:
    code = (
        "import numpy as np, sys; sys.path.insert(0, '.');"
        "from cubic_scattering.tests.test_graded_voxel_fourier_symmetry import N_SUB, NA, NC;"
        "from cubic_scattering.graded_voxel.fft import fourier_symmetry_tables;"
        "from cubic_scattering.graded_voxel.kernel_fortran import apply_canonical_fortran;"
        "rng = np.random.default_rng(5); c = lambda *s: rng.normal(size=s) + 1j * rng.normal(size=s);"
        "khc = c(20, 90, 36); sh = c(343, 90);"  # the tables are what is tested: any kernel will do
        "out = apply_canonical_fortran(khc, sh, fourier_symmetry_tables(N_SUB, NA, NC, 10, 4));"
        "sys.stdout.buffer.write(out.tobytes())"
    )
    outs = []
    for threads in ("1", "4"):
        env = dict(os.environ, OMP_NUM_THREADS=threads)
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, env=env, check=True)
        outs.append(done.stdout)
    assert outs[0] == outs[1]


@pytest.mark.slow  # two solves: 39 s under parallel load (21 October 2026)
def test_the_solve_through_the_canonical_kernel_equals_the_full_kernel_solve(monkeypatch) -> None:
    """The compiled backend solves through the canonical kernel, the reference backend the full one."""
    from cubic_scattering.graded_voxel import fft as F

    args = (OMEGA, 10.0, REF, CON, N_SUB, _profile, KHAT, KHAT, "P")
    built = []
    real = F.canonical_transformed_kernel
    monkeypatch.setattr(F, "canonical_transformed_kernel", lambda *a, **k: built.append(1) or real(*a, **k))
    canon = F.solve_graded_sphere_fft(*args, p=1, r=1, gmres_tol=1e-13)
    assert built == [1]  # the compiled backend took the canonical path
    monkeypatch.setattr(F, "point_kernel_backend", lambda: "python")
    full = F.solve_graded_sphere_fft(*args, p=1, r=1, gmres_tol=1e-13)
    assert np.linalg.norm(canon.psi - full.psi) / np.linalg.norm(full.psi) < 1e-12


def test_the_solve_reports_its_stage_timings() -> None:
    from cubic_scattering.graded_voxel import fft as F

    timings: dict[str, float] = {}
    F.solve_graded_sphere_fft(
        OMEGA, 10.0, REF, CON, 3, _profile, KHAT, KHAT, "P", p=1, r=1, gmres_tol=1e-9, timings=timings
    )
    assert set(timings) == {"kernel_s", "solve_s", "products"}
    assert timings["products"] >= 1 and timings["kernel_s"] > 0 and timings["solve_s"] > 0


def test_the_block_jacobi_inverses_equal_the_cell_by_cell_ones() -> None:
    from cubic_scattering.graded_voxel import fft as F

    rng = np.random.default_rng(2)
    na, nc, n = 4, 10, 37
    m9 = np.diag(rng.uniform(1.0, 2.0, na))
    k0 = rng.normal(size=(na, nc, 9, 9)) + 1j * rng.normal(size=(na, nc, 9, 9))
    e_cells = 0.1 * (rng.normal(size=(n, nc, na, 9, 9)) + 1j * rng.normal(size=(n, nc, na, 9, 9)))
    rows = na * 9
    want = np.array(
        [
            np.linalg.inv(np.kron(m9, np.eye(9)) - np.einsum("acij,cbjk->aibk", k0, en).reshape(rows, rows))
            for en in e_cells
        ]
    )
    got = F._block_jacobi(m9, k0, e_cells)
    assert np.abs(got - want).max() <= 1e-13 * np.abs(want).max()
