"""G3 rule tests — the TP/FP-guard pattern every rule must follow."""

from __future__ import annotations

import ast

from flakehound.rules.base import Confidence, FileContext
from flakehound.rules.g3_sleep_in_test import SleepInTest


def _ctx(source: str, name: str = "test_x.py") -> FileContext:
    return FileContext(
        path=name,
        source=source,
        tree=ast.parse(source),
        is_test_file=True,
        is_conftest=False,
    )


def _run(source: str):
    return list(SleepInTest().check(_ctx(source)))


def test_detects_bare_time_sleep_before_assert():
    src = (
        "import time\n"
        "def test_worker_processes_item(worker):\n"
        "    worker.start()\n"
        "    time.sleep(0.05)\n"
        "    assert worker.processed == 1\n"
    )
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].rule_id == "G3"
    assert findings[0].confidence == Confidence.HIGH
    assert findings[0].line == 4


def test_detects_asyncio_sleep_in_async_test():
    src = (
        "import asyncio\n"
        "async def test_task_completes(task):\n"
        "    await asyncio.sleep(0.2)\n"
        "    assert task.done()\n"
    )
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].rule_id == "G3"
    assert findings[0].line == 3


def test_detects_sleep_in_fixture():
    src = (
        "import time, pytest\n@pytest.fixture\ndef slow_setup():\n    time.sleep(0.1)\n    yield\n"
    )
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].line == 4


def test_polling_loop_sleep_is_downgraded_to_medium():
    src = (
        "import time\n"
        "def test_eventually_ready(flag):\n"
        "    while not flag.is_set():\n"
        "        time.sleep(0.01)\n"
        "    assert flag.is_set()\n"
    )
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].confidence == Confidence.MEDIUM


def test_non_literal_delay_is_downgraded_to_advisory():
    src = "import time\ndef test_a(interval):\n    time.sleep(interval)\n"
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].confidence == Confidence.ADVISORY


def test_fp_guard_sleep_zero_is_clean():
    src = (
        "import time, asyncio\n"
        "def test_a():\n"
        "    time.sleep(0)\n"
        "async def test_b():\n"
        "    await asyncio.sleep(0)\n"
    )
    assert _run(src) == []


def test_fp_guard_monkeypatched_sleep_is_clean():
    src = (
        "import time\n"
        "def test_a(monkeypatch):\n"
        "    monkeypatch.setattr(time, 'sleep', lambda *_: None)\n"
        "    time.sleep(5)\n"
        "    assert True\n"
    )
    assert _run(src) == []


def test_fp_guard_sleep_in_non_test_helper_is_out_of_scope():
    src = (
        "import time\n"
        "def _drain_queue():\n"
        "    time.sleep(0.5)\n"
        "\n"
        "def test_a():\n"
        "    _drain_queue()\n"
        "    assert True\n"
    )
    assert _run(src) == []


def test_fp_guard_unrelated_sleep_method_on_object():
    src = "def test_a(scheduler):\n    scheduler.sleep(3)\n"
    assert _run(src) == []
