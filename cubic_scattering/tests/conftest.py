"""The per-test time budget, and skipping the PyTorch solvers' tests where torch is not installed.

The budget (``time_budget``) fails any test over ``slow_budget_seconds`` that is not marked slow, and
limits each parallel worker to one thread. torch is imported lazily by the package, so its tests are
skipped rather than failing at collection.
"""

import importlib.util

from cubic_scattering.tests.time_budget import *  # noqa: F403

collect_ignore = (
    []
    if importlib.util.find_spec("torch")
    else [
        "test_slab_scattering_gpu.py",
        "test_solver_config.py",
        "test_sphere_scattering_fft_gpu.py",
        "test_torch_gmres.py",
    ]
)
