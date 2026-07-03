"""G5 rule tests — the TP/FP-guard pattern every rule must follow."""

from __future__ import annotations

import ast

from flakehound.rules.base import Confidence, FileContext
from flakehound.rules.g5_shared_state_fixture import SharedStateFixture


def _ctx(source: str, name: str = "test_x.py") -> FileContext:
    return FileContext(
        path=name,
        source=source,
        tree=ast.parse(source),
        is_test_file=True,
        is_conftest=False,
    )


def _run(source: str):
    return list(SharedStateFixture().check(_ctx(source)))


# --- true positives ---------------------------------------------------------


def test_detects_module_scoped_fixture_returning_mutable_dict():
    src = 'import pytest\n@pytest.fixture(scope="module")\ndef config():\n    return {"a": 1}\n'
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].rule_id == "G5"
    assert findings[0].confidence == Confidence.MEDIUM
    assert findings[0].line == 4


def test_detects_session_scoped_fixture_mutating_declared_global():
    src = (
        "import pytest\n"
        "_SEEN = []\n"
        '@pytest.fixture(scope="session", autouse=True)\n'
        "def track():\n"
        "    global _SEEN\n"
        "    _SEEN.append(1)\n"
    )
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].rule_id == "G5"
    assert findings[0].line == 6


def test_detects_class_scoped_fixture_returning_class_instance_as_advisory():
    src = (
        "import pytest\n"
        "class Tracker:\n"
        "    pass\n"
        '@pytest.fixture(scope="class")\n'
        "def tracker():\n"
        "    return Tracker()\n"
    )
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].confidence == Confidence.ADVISORY
    assert findings[0].line == 6


def test_detects_module_scoped_fixture_yielding_mutable_list_no_teardown():
    src = 'import pytest\n@pytest.fixture(scope="module")\ndef registry():\n    yield []\n'
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].line == 4


# --- false-positive guards ---------------------------------------------------


def test_fp_guard_function_scope_is_clean():
    src = 'import pytest\n@pytest.fixture\ndef config():\n    return {"a": 1}\n'
    assert _run(src) == []


def test_fp_guard_explicit_function_scope_is_clean():
    src = 'import pytest\n@pytest.fixture(scope="function")\ndef registry():\n    return []\n'
    assert _run(src) == []


def test_fp_guard_module_scope_returning_immutable_is_clean():
    for body in ("return (1, 2, 3)", 'return "fixed"', "return 7", "return frozenset({1, 2})"):
        src = f'import pytest\n@pytest.fixture(scope="module")\ndef limits():\n    {body}\n'
        assert _run(src) == [], body


def test_fp_guard_module_scope_yield_then_restore_in_teardown_is_clean():
    src = (
        "import pytest\n"
        '@pytest.fixture(scope="module")\n'
        "def session_data():\n"
        '    data = {"a": 1}\n'
        "    yield data\n"
        "    data.clear()\n"
    )
    assert _run(src) == []


def test_fp_guard_module_scope_yield_try_finally_restore_is_clean():
    src = (
        "import pytest\n"
        '@pytest.fixture(scope="module")\n'
        "def session_data():\n"
        '    data = {"a": 1}\n'
        "    try:\n"
        "        yield data\n"
        "    finally:\n"
        "        data.clear()\n"
    )
    assert _run(src) == []


def test_fp_guard_locally_built_object_mutation_is_clean():
    src = (
        "import pytest\n"
        '@pytest.fixture(scope="module")\n'
        "def settings():\n"
        "    Local = build()\n"
        "    Local.debug = True\n"
    )
    assert _run(src) == []


def test_fp_guard_deepcopy_of_module_constant_is_clean():
    src = (
        "import copy\n"
        "import pytest\n"
        '_TEMPLATE = {"a": 1}\n'
        '@pytest.fixture(scope="module")\n'
        "def config():\n"
        "    return copy.deepcopy(_TEMPLATE)\n"
    )
    assert _run(src) == []
