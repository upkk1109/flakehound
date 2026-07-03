"""pytest plugin (v0 surface).

v0.1 ships the static scanner + CLI. The run-history scorer (SQLite outcome
tracking, flip-rate scoring, xdist merge) lands in M2 behind these same hooks —
kept minimal and honest until then: today the plugin only adds a summary line
pointing at `flakehound scan` when findings exist in changed test files.
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("flakehound")
    group.addoption(
        "--flakehound",
        action="store_true",
        default=False,
        help="enable flakehound static summary after the run",
    )


@pytest.hookimpl(trylast=True)
def pytest_terminal_summary(terminalreporter, exitstatus: int, config: pytest.Config) -> None:
    if not config.getoption("--flakehound"):
        return
    from pathlib import Path

    from flakehound.config import load_config
    from flakehound.scanner import scan_paths

    roots = [Path(str(p)) for p in config.args] or [Path.cwd()]
    result = scan_paths(roots, load_config())
    tr = terminalreporter
    tr.section("flakehound")
    if not result.findings:
        tr.write_line(f"no flaky-prone patterns in {result.files_scanned} test file(s)")
        return
    for f in result.findings[:20]:
        tr.write_line(f.format_text())
    if len(result.findings) > 20:
        tr.write_line(f"... and {len(result.findings) - 20} more (run `flakehound scan`)")
