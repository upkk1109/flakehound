"""pytest plugin (v0 surface).

v0.1 ships the static scanner + CLI. The run-history scorer (SQLite outcome
tracking, flip-rate scoring, xdist merge) lands in M2 behind these same hooks —
kept minimal and honest until then: today the plugin only adds a summary line
pointing at `flakehound scan` when findings exist in changed test files.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("flakehound")
    group.addoption(
        "--flakehound",
        action="store_true",
        default=False,
        help="enable flakehound static summary after the run",
    )


def _resolve_roots(config: pytest.Config) -> list[Path]:
    """Turn ``config.args`` into filesystem paths ``scan_paths`` can walk.

    ``config.args`` mixes file paths, ``file::nodeid`` selectors, and (in some
    invocation shapes) option-looking leftovers. Node ids are reduced to their
    file part; anything that isn't an existing path on disk is dropped. If
    nothing survives, fall back to the pytest rootpath rather than an empty
    (silently no-op) scan.
    """
    invocation_dir = Path(config.invocation_params.dir)
    roots: list[Path] = []
    for raw in config.args:
        text = str(raw)
        if not text or text.startswith("-"):
            continue
        file_part = text.split("::", 1)[0]
        if not file_part:
            continue
        candidate = Path(file_part)
        if not candidate.is_absolute():
            candidate = invocation_dir / candidate
        if candidate.exists():
            roots.append(candidate)
    return roots or [config.rootpath]


@pytest.hookimpl(trylast=True)
def pytest_terminal_summary(terminalreporter, exitstatus: int, config: pytest.Config) -> None:
    if not config.getoption("--flakehound"):
        return
    from flakehound.config import load_config
    from flakehound.scanner import scan_paths

    roots = _resolve_roots(config)
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
