"""G4 rule tests — the TP/FP-guard pattern every rule must follow."""

from __future__ import annotations

import ast

from flakehound.rules.base import Confidence, FileContext
from flakehound.rules.g4_naive_now import NaiveNow


def _ctx(source: str, name: str = "test_x.py") -> FileContext:
    return FileContext(
        path=name,
        source=source,
        tree=ast.parse(source),
        is_test_file=True,
        is_conftest=False,
    )


def _run(source: str):
    return list(NaiveNow().check(_ctx(source)))


def test_detects_now_assigned_then_used_in_asserted_arithmetic():
    src = (
        "import time\n"
        "\n"
        "def test_completes_quickly():\n"
        "    start = time.time()\n"
        "    do_work()\n"
        "    elapsed = time.time() - start\n"
        "    assert elapsed < 1.0\n"
    )
    findings = _run(src)
    assert len(findings) == 2
    by_line = {f.line: f for f in findings}
    assert by_line[6].rule_id == "G4"
    assert by_line[6].confidence == Confidence.MEDIUM
    assert by_line[4].confidence == Confidence.ADVISORY


def test_detects_now_read_directly_inside_assert():
    src = (
        "from datetime import datetime\n"
        "\n"
        "def test_not_expired():\n"
        "    deadline = datetime(2026, 1, 1)\n"
        "    assert datetime.now() < deadline\n"
    )
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].rule_id == "G4"
    assert findings[0].line == 5
    assert findings[0].confidence == Confidence.MEDIUM


def test_fp_guard_freezegun_import_suppresses_file():
    src = (
        "import freezegun\n"
        "from datetime import datetime\n"
        "\n"
        "def test_frozen():\n"
        "    with freezegun.freeze_time('2026-01-01'):\n"
        "        assert datetime.now().year == 2026\n"
        "        deadline = datetime.now()\n"
        "        assert deadline.year == 2026\n"
    )
    assert _run(src) == []


def test_fp_guard_perf_timing_without_assert_is_clean():
    src = (
        "import time\n"
        "\n"
        "def test_records_duration(caplog):\n"
        "    start = time.time()\n"
        "    do_work()\n"
        "    elapsed = time.time() - start\n"
        "    caplog.info(f'took {elapsed}s')\n"
    )
    assert _run(src) == []


def test_fp_guard_monkeypatched_clock_is_clean():
    src = (
        "import time\n"
        "\n"
        "def test_uses_patched_clock(monkeypatch):\n"
        "    monkeypatch.setattr('time.time', lambda: 1000.0)\n"
        "    start = time.time()\n"
        "    elapsed = time.time() - start\n"
        "    assert elapsed == 0\n"
    )
    assert _run(src) == []
