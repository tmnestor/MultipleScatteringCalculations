"""The per-test time budget, and skipping the PyTorch solvers' tests where torch is not installed.

The budget (``time_budget``) fails any test over ``slow_budget_seconds`` that is not marked slow, and
limits each parallel worker to one thread. torch is imported lazily by the package, so its tests are
skipped rather than failing at collection.
"""

import importlib.util

import pytest

from cubic_scattering.tests.time_budget import *  # noqa: F403


@pytest.fixture(autouse=True, scope="session")
def _private_kseries_store(tmp_path_factory):
    """Keep the tests out of the user's coefficient store (``cell_average.kseries_store``).

    The default route builds and grows that store on first use, so any test reaching a sphere kernel would
    otherwise write into ~/.cache, and parallel workers would write the same file. Each session (each xdist
    worker) gets its own store in a temporary directory instead.
    """
    from cubic_scattering import cell_average_kseries as ck

    path = tmp_path_factory.mktemp("kseries") / "receiver_kseries.npy"
    original = ck.store_path
    ck.store_path = lambda: path
    ck.clear_loaded_store()
    yield path
    ck.store_path = original
    ck.clear_loaded_store()


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
