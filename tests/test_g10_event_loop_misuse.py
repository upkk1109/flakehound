"""G10 rule tests — the TP/FP-guard pattern every rule must follow."""

from __future__ import annotations

import ast

from flakehound.rules.base import Confidence, FileContext
from flakehound.rules.g10_event_loop_misuse import EventLoopMisuse


def _ctx(source: str, name: str = "test_x.py") -> FileContext:
    return FileContext(
        path=name,
        source=source,
        tree=ast.parse(source),
        is_test_file=True,
        is_conftest=False,
    )


def _run(source: str):
    return list(EventLoopMisuse().check(_ctx(source)))


def test_detects_get_event_loop_and_run_until_complete():
    src = (
        "import asyncio\n"
        "def test_a():\n"
        "    loop = asyncio.get_event_loop()\n"
        "    loop.run_until_complete(work())\n"
    )
    findings = _run(src)
    assert len(findings) == 2
    assert all(f.rule_id == "G10" for f in findings)
    assert findings[0].line == 3
    assert findings[1].line == 4
    assert findings[1].confidence == Confidence.MEDIUM


def test_detects_loop_stored_at_module_scope():
    src = (
        "import asyncio\n"
        "loop = asyncio.new_event_loop()\n"
        "asyncio.set_event_loop(loop)\n"
        "\n"
        "def test_a():\n"
        "    loop.run_until_complete(work())\n"
    )
    findings = _run(src)
    assert [f.line for f in findings] == [2, 3, 6]
    assert all(f.rule_id == "G10" for f in findings)


def test_detects_asyncio_run_inside_async_test():
    src = "import asyncio\nasync def test_a():\n    asyncio.run(work())\n"
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].line == 3


def test_fp_guard_event_loop_fixture_override():
    src = (
        "import asyncio\n"
        "import pytest\n"
        "\n"
        "@pytest.fixture\n"
        "def event_loop():\n"
        "    loop = asyncio.new_event_loop()\n"
        "    yield loop\n"
        "    loop.close()\n"
    )
    assert _run(src) == []


def test_fp_guard_sync_helper_module():
    src = (
        "import asyncio\n"
        "\n"
        "def run_sync(coro):\n"
        "    loop = asyncio.new_event_loop()\n"
        "    try:\n"
        "        return loop.run_until_complete(coro)\n"
        "    finally:\n"
        "        loop.close()\n"
        "\n"
        "def test_a():\n"
        "    assert run_sync(work()) == 1\n"
    )
    assert _run(src) == []


def test_run_until_complete_on_ambiguous_receiver_is_downgraded():
    src = "def test_a(driver):\n    driver.run_until_complete(work())\n"
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].confidence == Confidence.ADVISORY
