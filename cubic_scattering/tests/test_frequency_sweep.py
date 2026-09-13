"""Kernel reuse across right-hand sides, and parallel frequency sweeps.

Two independent wins measured on 2026-09-13:

* ``_build_slab_kernels`` was called unconditionally inside every solve, even
  though it depends only on (geometry, omega, ref) and not on the incident
  field.  An angle sweep therefore rebuilt it once per angle.  The build costs
  87-215 matvecs, so reusing it is worth ~100x on any multi-angle sweep.
* The frequency loop is embarrassingly parallel and was running on one core of
  eight.  The kernel build is thousands of small NumPy calls and so is
  GIL-bound; processes, not threads.

Kernel reuse is bit-exact: it is the same arithmetic, merely not repeated.
The frequency sweep agrees only to round-off (~2e-16 relative), because BLAS is
threaded differently in parent and worker -- see the module docstring.  Gates
and validation should therefore run serially; the sweep is for production.
"""

import numpy as np
import pytest

from cubic_scattering import (
    MaterialContrast,
    ReferenceMedium,
    SlabGeometry,
    compute_slab_scattering,
    uniform_slab_material,
)
from cubic_scattering.frequency_sweep import sweep_frequencies
from cubic_scattering.slab_scattering import build_slab_kernels

REF = ReferenceMedium(alpha=4.0, beta=2.22, rho=2.6)
CONTRAST = MaterialContrast(Dlambda=2.0, Dmu=1.0, Drho=0.1)
K_HAT = np.array([1.0, 0.0, 0.0])


@pytest.fixture
def small():
    """A lattice small enough to solve several times in a test."""
    geom = SlabGeometry(M=8, N_z=2, a=0.5)
    return geom, uniform_slab_material(geom, REF, CONTRAST)


# ------------------------------------------------------------ kernel reuse


def test_public_builder_matches_private(small):
    """build_slab_kernels is the supported entry point to the same object."""
    from cubic_scattering.slab_scattering import _build_slab_kernels

    geom, _ = small
    w = 2 * np.pi * 10.0
    got = build_slab_kernels(geom, w, REF, periodic=True)
    want = _build_slab_kernels(geom, w, REF, periodic=True)
    assert got.shape == want.shape
    assert np.array_equal(got, want)


def test_reused_kernel_gives_identical_result(small):
    """Passing a prebuilt kernel must not change a single bit."""
    geom, mat = small
    w = 2 * np.pi * 10.0
    fresh = compute_slab_scattering(geom, mat, w, K_HAT, periodic=True)
    kern = build_slab_kernels(geom, w, REF, periodic=True)
    reused = compute_slab_scattering(geom, mat, w, K_HAT, periodic=True, kernel_hat=kern)
    assert np.array_equal(fresh.psi, reused.psi)
    assert fresh.n_gmres_iter == reused.n_gmres_iter


def test_reused_kernel_skips_the_build(small, monkeypatch):
    """The point of the parameter: the build must not run at all."""
    import cubic_scattering.slab_scattering as ss

    geom, mat = small
    w = 2 * np.pi * 10.0
    kern = build_slab_kernels(geom, w, REF, periodic=True)

    calls: list[int] = []

    def _spy(*_args, **_kwargs):
        calls.append(1)
        return kern

    monkeypatch.setattr(ss, "_build_slab_kernels", _spy)
    compute_slab_scattering(geom, mat, w, K_HAT, periodic=True, kernel_hat=kern)
    assert calls == [], "prebuilt kernel was ignored and the build ran anyway"


def test_one_kernel_serves_many_incidence_angles(small):
    """The sweep this exists for: many right-hand sides, one build."""
    geom, mat = small
    w = 2 * np.pi * 10.0
    kern = build_slab_kernels(geom, w, REF, periodic=True)
    for tilt in (0.0, 0.2, 0.4):
        k_hat = np.array([np.cos(tilt), np.sin(tilt), 0.0])
        shared = compute_slab_scattering(geom, mat, w, k_hat, periodic=True, kernel_hat=kern)
        fresh = compute_slab_scattering(geom, mat, w, k_hat, periodic=True)
        assert np.array_equal(shared.psi, fresh.psi)


# ------------------------------------------------------- frequency parallel


def test_sweep_matches_serial_to_round_off(small):
    """Parallel frequencies reproduce the serial loop to floating-point round-off.

    NOT bit-exact, and an earlier version of this test asserted that it was --
    passing only because the lattice was small enough to converge in 2 GMRES
    iterations, leaving the difference nothing to accumulate through.  At M=32
    with a 5-iteration frequency the real discrepancy shows: 1.78e-15 absolute,
    1.87e-16 relative, from BLAS being threaded differently in parent and
    worker.  A tolerance test that only passes because the problem is too easy
    is worse than no test.
    """
    geom, mat = small
    omegas = [2 * np.pi * f for f in (6.0, 10.0, 14.0)]

    serial = [compute_slab_scattering(geom, mat, w, K_HAT, periodic=True) for w in omegas]
    parallel = sweep_frequencies(geom, mat, omegas, K_HAT, periodic=True, n_workers=2)

    assert len(parallel) == len(serial)
    for got, want in zip(parallel, serial, strict=True):
        scale = np.abs(want.psi).max()
        rel = np.abs(got.psi - want.psi).max() / scale
        assert rel < 1e-13, f"parallel differs by {rel:.3e} relative, not round-off"
        assert got.omega == want.omega
        assert got.n_gmres_iter == want.n_gmres_iter


def test_serial_path_is_deterministic(small):
    """Within one process the solver is bit-identical run to run.

    This is what makes the round-off in the parallel comparison attributable to
    the process boundary rather than to the solver itself.
    """
    geom, mat = small
    omegas = [2 * np.pi * 6.0]
    a = sweep_frequencies(geom, mat, omegas, K_HAT, periodic=True, n_workers=1)
    b = sweep_frequencies(geom, mat, omegas, K_HAT, periodic=True, n_workers=1)
    assert np.array_equal(a[0].psi, b[0].psi)


def test_sweep_preserves_input_order(small):
    """Results come back in the order the frequencies were given."""
    geom, mat = small
    omegas = [2 * np.pi * f for f in (14.0, 6.0, 10.0)]
    out = sweep_frequencies(geom, mat, omegas, K_HAT, periodic=True, n_workers=2)
    assert [r.omega for r in out] == omegas


def test_sweep_serial_fallback(small):
    """n_workers=1 runs in-process, which keeps debugging tractable."""
    geom, mat = small
    omegas = [2 * np.pi * 10.0]
    out = sweep_frequencies(geom, mat, omegas, K_HAT, periodic=True, n_workers=1)
    want = compute_slab_scattering(geom, mat, omegas[0], K_HAT, periodic=True)
    assert np.array_equal(out[0].psi, want.psi)


def test_sweep_does_not_touch_the_callers_environment(small):
    """A numerical call must not mutate global interpreter state.

    Regression: pinning BLAS to one thread was done inside the solve, so the
    in-process path (n_workers=1) pinned the CALLER permanently. The project's
    own suite went from 11 to 21 minutes once any serial sweep had run. The
    pinning now happens only as a worker-process initializer.
    """
    import os

    geom, mat = small
    watched = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    before = {k: os.environ.get(k) for k in watched}
    sweep_frequencies(geom, mat, [2 * np.pi * 10.0], K_HAT, periodic=True, n_workers=1)
    sweep_frequencies(geom, mat, [2 * np.pi * 10.0], K_HAT, periodic=True, n_workers=2)
    after = {k: os.environ.get(k) for k in watched}
    assert after == before, f"sweep_frequencies mutated the environment: {before} -> {after}"


def test_sweep_rejects_empty_frequency_list(small):
    """Fail fast with a diagnostic rather than returning an empty list."""
    geom, mat = small
    with pytest.raises(ValueError, match="at least one frequency"):
        sweep_frequencies(geom, mat, [], K_HAT, periodic=True)
