"""pytest-plugin integration tests: option registration + terminal summary.

Uses the ``pytester`` fixture to run flakehound as an installed pytest plugin
against a throwaway pytest suite, in-process.
"""

from __future__ import annotations

pytest_plugins = ["pytester"]

_FLAKY_FILE = """
import random

def test_a():
    random.seed(1)
"""

_CLEAN_FILE = """
def test_a():
    assert 1 + 1 == 2
"""


def test_plugin_off_by_default(pytester):
    pytester.makepyfile(test_bad=_FLAKY_FILE)
    result = pytester.runpytest("-q")
    result.assert_outcomes(passed=1)
    assert "flakehound" not in result.stdout.str()


def test_plugin_reports_findings_for_directory_run(pytester):
    pytester.makepyfile(test_bad=_FLAKY_FILE)
    result = pytester.runpytest("--flakehound", "-q")
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(
        [
            "*flakehound*",
            "*test_bad.py*G1/high*mutates global RNG state*",
        ]
    )


def test_plugin_reports_no_patterns_on_clean_suite(pytester):
    pytester.makepyfile(test_good=_CLEAN_FILE)
    result = pytester.runpytest("--flakehound", "-q")
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(["*flakehound*", "*no flaky-prone patterns in 1 test file(s)*"])


def test_plugin_scans_node_id_invocation(pytester):
    pytester.makepyfile(test_bad=_FLAKY_FILE)
    result = pytester.runpytest("--flakehound", "-q", "test_bad.py::test_a")
    result.assert_outcomes(passed=1)
    # regression: node-id args used to be treated as literal file paths and
    # silently produced "0 test file(s)" instead of scanning test_bad.py
    result.stdout.fnmatch_lines(["*flakehound*", "*G1/high*"])


def test_plugin_falls_back_to_rootpath_when_no_paths_given(pytester):
    pytester.makepyfile(test_bad=_FLAKY_FILE)
    result = pytester.runpytest("--flakehound", "-q")
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(["*flakehound*", "*G1/high*"])


def test_plugin_ignores_option_like_and_missing_args(pytester):
    pytester.makepyfile(test_bad=_FLAKY_FILE)
    # "-v" is consumed by pytest itself; this exercises that leftover
    # option-looking strings in config.args don't crash root resolution.
    result = pytester.runpytest("--flakehound", "-v", "test_bad.py")
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(["*flakehound*", "*G1/high*"])
