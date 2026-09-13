#!/usr/bin/env python3
"""Run independent frequency solves across all available cores.

WHY PROCESSES, NOT THREADS.  Profiling a solve on 2026-09-13 put 98% of the
time in ``_build_slab_kernels`` and 1% in the GMRES matvec, and the kernel build
is thousands of small NumPy calls on 3x3x3 tensors -- Python-level work, so it
holds the GIL and does not thread.  It is also insensitive to OpenBLAS threads:
524 ms at ``OMP_NUM_THREADS=1`` and 524 ms at 8.  Frequencies are fully
independent, so distributing them over processes is the one change that turns 8
idle cores into ~7x, with no restructuring of the physics and no risk to the
numerics.

BLAS OVERSUBSCRIPTION.  Each worker pins its own BLAS to a single thread.
Without that, N workers each spawning 8 BLAS threads thrash a 10-core machine.
The solve does not benefit from BLAS threading anyway (measured above), so this
costs nothing.

AGREEMENT WITH THE SERIAL PATH IS TO ROUND-OFF, NOT BIT-EXACT.  Measured:
1.78e-15 absolute, 1.87e-16 relative -- one ulp -- and only on the frequency
that needed 5 GMRES iterations rather than 2, where the difference has something
to accumulate through.  Serial-vs-serial in one process IS bit-identical, so the
solver is deterministic; the discrepancy is floating-point non-associativity
between differently-threaded BLAS in parent and worker.

That is below the 1e-15 the project's gates assert at, but not by a wide margin.
**Run gates and validation serially.**  Use this for production sweeps, where a
1e-16 relative perturbation is irrelevant, not for the measurements that decide
whether a formulation is right.

WHEN THIS IS ACTUALLY FASTER.  Process start-up on macOS costs ~1.8 s (each
worker re-imports NumPy, SciPy and this package), so short tasks lose outright.
Measured, 8 frequencies over 8 workers:

    M    N_z   per solve   serial    8 proc   speedup
    16   4       0.14 s     1.09 s    1.90 s    0.58x   <- SLOWER
    32   4       0.53 s     4.28 s    2.78 s    1.54x
    48   6       1.92 s    15.34 s    4.29 s    3.57x

The model is ``t_parallel ~= t_serial / n_workers + 1.8 s``.  Approaching the
~7x this machine can give needs a serial time well past a minute, i.e. genuine
production sizes.  Below roughly 0.5 s per solve, pass ``n_workers=1``.  No
automatic fallback is applied: silently choosing a different execution path
would contradict this project's fail-fast stance, so the choice stays explicit.

CALLING THIS FROM A SCRIPT -- READ THIS.  macOS starts processes by *spawn*, so
each worker re-imports the calling module.  A script that calls
``sweep_frequencies`` at module level will therefore re-run its own top-level
code once per worker.  The symptom is your output appearing N times and the run
taking longer than serial.  Guard the entry point::

    if __name__ == "__main__":
        results = sweep_frequencies(geom, mat, omegas, k_hat)

Inside pytest, a notebook, or any importable module this does not arise; it bites
only bare scripts.  ``n_workers=1`` runs in-process and sidesteps it entirely,
which is also the setting to use when a traceback or a debugger is needed.
"""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from numpy.typing import NDArray

from cubic_scattering.slab_scattering import (
    SlabGeometry,
    SlabMaterial,
    SlabResult,
    compute_slab_scattering,
)

__all__ = ["sweep_frequencies"]


def _pin_blas_to_one_thread() -> None:
    """Stop each worker from spawning a full BLAS thread pool.

    Used ONLY as a ``ProcessPoolExecutor`` initializer, so it runs in worker
    processes and never in the caller's.  An earlier version called this from
    the solve itself, which meant ``n_workers=1`` -- the in-process path --
    silently pinned the CALLER's BLAS to one thread for the rest of its life.
    The symptom was the project's test suite slowing from 11 to 21 minutes once
    a single serial sweep had run.  Mutating global interpreter state as a side
    effect of a numerical call is not acceptable; keep this in the workers.
    """
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = "1"


def _solve_one(args: tuple) -> SlabResult:
    """Worker entry point: one frequency, serial, in its own process.

    Args:
        args: ``(geometry, material, omega, k_hat, wave_type, gmres_tol,
            max_iter, volume_averaged, n_orders, periodic)``.

    Returns:
        The solve result for that frequency.
    """
    (
        geometry,
        material,
        omega,
        k_hat,
        wave_type,
        gmres_tol,
        max_iter,
        volume_averaged,
        n_orders,
        periodic,
    ) = args
    return compute_slab_scattering(
        geometry,
        material,
        omega,
        k_hat,
        wave_type=wave_type,
        gmres_tol=gmres_tol,
        max_iter=max_iter,
        volume_averaged=volume_averaged,
        n_orders=n_orders,
        periodic=periodic,
    )


def sweep_frequencies(
    geometry: SlabGeometry,
    material: SlabMaterial,
    omegas: list[float],
    k_hat: NDArray[np.floating],
    wave_type: str = "P",
    gmres_tol: float = 1e-6,
    max_iter: int = 500,
    *,
    volume_averaged: bool = False,
    n_orders: int = 2,
    periodic: bool = False,
    n_workers: int | None = None,
) -> list[SlabResult]:
    """Solve the slab problem at many frequencies, in parallel.

    Args:
        geometry: Slab lattice geometry.
        material: Per-cube material contrasts.
        omegas: Angular frequencies (rad/s); at least one.
        k_hat: Unit incident propagation direction (z, x, y).
        wave_type: 'P', 'S' (SV), or 'SH'.
        gmres_tol: GMRES relative tolerance.
        max_iter: Maximum GMRES iterations.
        volume_averaged: Use the volume-averaged inter-voxel propagator.
        n_orders: Dynamic correction orders when volume_averaged is True.
        periodic: Circular convolution for an infinite periodic slab.
        n_workers: Processes to use. ``None`` uses one per performance core;
            ``1`` runs serially in this process, which keeps tracebacks and
            debuggers usable.

    Returns:
        One result per frequency, in the order the frequencies were given.

    Raises:
        ValueError: If ``omegas`` is empty.
    """
    if not omegas:
        msg = (
            "sweep_frequencies needs at least one frequency.\n"
            "  What: `omegas` was empty, so there is nothing to solve.\n"
            "  Where: the `omegas` argument of sweep_frequencies.\n"
            "  Expected: a non-empty sequence of angular frequencies in rad/s,\n"
            "            e.g. [2*np.pi*f for f in (6.0, 12.0, 25.0)].\n"
            "  Fix: build the frequency list before calling, and skip the call\n"
            "       entirely if the list is empty."
        )
        raise ValueError(msg) from None

    payload = [
        (
            geometry,
            material,
            float(w),
            np.asarray(k_hat, dtype=float),
            wave_type,
            gmres_tol,
            max_iter,
            volume_averaged,
            n_orders,
            periodic,
        )
        for w in omegas
    ]

    if n_workers == 1:
        return [_solve_one(a) for a in payload]

    workers = n_workers if n_workers is not None else _default_workers()
    workers = max(1, min(workers, len(payload)))
    with ProcessPoolExecutor(max_workers=workers, initializer=_pin_blas_to_one_thread) as pool:
        return list(pool.map(_solve_one, payload))


def _default_workers() -> int:
    """One worker per performance core, falling back to the logical count.

    Returns:
        Number of worker processes to use by default.
    """
    try:
        import subprocess

        out = subprocess.run(
            ["sysctl", "-n", "hw.perflevel0.logicalcpu"],
            capture_output=True,
            text=True,
            check=True,
        )
        return max(1, int(out.stdout.strip()))
    except (OSError, ValueError, subprocess.SubprocessError):
        return max(1, os.cpu_count() or 1)
