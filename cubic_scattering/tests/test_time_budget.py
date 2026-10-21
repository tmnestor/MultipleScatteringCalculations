"""The per-test time budget: an unmarked test over budget fails; a slow-marked one, or a fast one, passes.

The budget keeps the commit gate (``pytest -n auto --dist worksteal -m "not slow"``) fast as the suite
grows: a test that takes longer than ``slow_budget_seconds`` must carry ``@pytest.mark.slow`` or be made
faster, so it can no longer drift into the gate unnoticed.
"""

import pytest

pytest_plugins = ["pytester"]

CONFTEST = "from cubic_scattering.tests.time_budget import *  # noqa: F403\n"
INI = "[pytest]\nmarkers =\n    slow: long-running\nslow_budget_seconds = {budget}\n"
TESTS = """
import time
import pytest

def test_fast():
    pass

def test_over_budget():
    time.sleep(0.3)

@pytest.mark.slow
def test_over_budget_but_marked():
    time.sleep(0.3)

@pytest.fixture
def slow_setup():
    time.sleep(0.3)

def test_over_budget_in_setup(slow_setup):
    pass
"""


@pytest.fixture
def project(pytester):
    pytester.makeconftest(CONFTEST)
    pytester.makepyfile(test_sample=TESTS)
    return pytester


def test_unmarked_test_over_budget_fails(project) -> None:
    project.makefile(".ini", pytest=INI.format(budget=0.1))
    result = project.runpytest("-p", "no:cacheprovider")
    result.assert_outcomes(passed=2, failed=1, errors=1)
    result.stdout.fnmatch_lines(["*test_over_budget*over the per-test budget*"])


def test_slow_marker_and_fast_tests_pass(project) -> None:
    project.makefile(".ini", pytest=INI.format(budget=0.1))
    result = project.runpytest("-p", "no:cacheprovider", "-k", "fast or marked")
    result.assert_outcomes(passed=2)


def test_generous_budget_passes_everything(project) -> None:
    project.makefile(".ini", pytest=INI.format(budget=10))
    project.runpytest("-p", "no:cacheprovider").assert_outcomes(passed=4)


def test_failure_message_says_what_where_and_how_to_fix(project) -> None:
    project.makefile(".ini", pytest=INI.format(budget=0.1))
    out = project.runpytest("-p", "no:cacheprovider", "-k", "test_over_budget and not marked and not setup")
    text = out.stdout.str()
    assert "over the per-test budget of 0.1 s" in text  # what
    assert "slow_budget_seconds" in text and "pytest.ini" in text  # where
    assert "@pytest.mark.slow" in text  # what valid looks like
    assert "Fix:" in text  # how to recover


@pytest.mark.parametrize("ini", ["[pytest]\n", "[pytest]\nslow_budget_seconds = soon\n"])
def test_missing_or_invalid_budget_stops_the_run(pytester, ini: str) -> None:
    pytester.makeconftest(CONFTEST)
    pytester.makepyfile(test_sample="def test_x():\n    pass\n")
    pytester.makefile(".ini", pytest=ini)
    result = pytester.runpytest("-p", "no:cacheprovider")
    assert result.ret != 0
    text = result.stderr.str() + result.stdout.str()
    assert "slow_budget_seconds" in text and "pytest.ini" in text  # what, where
    assert "slow_budget_seconds = 5" in text  # a valid example
    assert "Fix:" in text  # how to recover


def test_the_shipped_budget_is_configured(pytestconfig) -> None:
    assert float(pytestconfig.getini("slow_budget_seconds")) > 0
