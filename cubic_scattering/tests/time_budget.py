"""A per-test time budget, so that slow tests cannot drift into the commit gate unnoticed.

The suite runs in two tiers:

    pytest -n auto --dist worksteal -m "not slow"    the gate, before every commit
    pytest -n auto --dist worksteal                  the full suite, by hand before a release

A test (its setup or its body) that takes longer than ``slow_budget_seconds`` (pytest.ini) and is not
marked ``@pytest.mark.slow`` fails, with the instruction to mark it or make it faster. A slow-marked test
is never checked: the budget polices the gate, not the full run.

Each xdist worker is limited to one OpenMP / BLAS thread, so that N workers use N cores rather than
starting N threads each. A test that measures threading sets its own thread count in a subprocess.

Load it from a conftest with ``from cubic_scattering.tests.time_budget import *``.
"""

import os

import pytest

__all__ = ["pytest_addoption", "pytest_configure", "pytest_runtest_makereport"]

_KEY = "slow_budget_seconds"
_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
_budget = pytest.StashKey[float]()


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addini(_KEY, "seconds a test not marked slow may take (setup or body); required", default=None)


def _ini_path(config: pytest.Config) -> str:
    return str(config.inipath) if config.inipath else "pytest.ini at the repository root (none found)"


def pytest_configure(config: pytest.Config) -> None:
    raw = config.getini(_KEY)
    try:
        budget = float(raw)
        if budget <= 0:
            raise ValueError
    except (TypeError, ValueError):
        found = "missing" if raw is None else repr(raw)
        raise pytest.UsageError(
            f"{_KEY} is {found}; it must be a positive number of seconds.\n"
            f"  Where: {_ini_path(config)}, key {_KEY} under [pytest]\n"
            f"  Valid: {_KEY} = 5\n"
            f"  Fix:   add the line above to the [pytest] section of that file."
        ) from None
    config.stash[_budget] = budget
    if os.environ.get("PYTEST_XDIST_WORKER"):
        for var in _THREAD_VARS:
            os.environ[var] = "1"


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]):
    report = yield
    budget = item.config.stash[_budget]
    over = report.when in ("setup", "call") and report.passed and call.duration > budget
    if over and item.get_closest_marker("slow") is None:
        report.outcome = "failed"
        report.longrepr = (
            f"{item.nodeid} took {call.duration:.1f} s in {report.when}, over the per-test budget of "
            f"{budget:g} s, and is not marked slow.\n"
            f"  Where: the test itself; the budget is {_KEY} in {_ini_path(item.config)}\n"
            f"  Valid: decorate it with @pytest.mark.slow (it then runs only in the full suite),\n"
            f"         or make it faster (a smaller problem, a cached fixture, a module-scoped setup).\n"
            f"  Fix:   add @pytest.mark.slow above the test, or shrink it under {budget:g} s."
        )
    return report
