"""G6 rule tests — the TP/FP-guard pattern every rule must follow."""

from __future__ import annotations

import ast

from flakehound.rules.base import Confidence, FileContext
from flakehound.rules.g6_import_time_side_effects import ImportTimeSideEffects


def _ctx(source: str, name: str = "test_x.py", is_conftest: bool = False) -> FileContext:
    return FileContext(
        path=name,
        source=source,
        tree=ast.parse(source),
        is_test_file=not is_conftest,
        is_conftest=is_conftest,
    )


def _run(source: str, **kwargs):
    return list(ImportTimeSideEffects().check(_ctx(source, **kwargs)))


def test_detects_module_level_requests_session():
    src = "import requests\n\n_SESSION = requests.Session()\n\n\ndef test_a():\n    pass\n"
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].rule_id == "G6"
    assert findings[0].line == 3
    assert findings[0].confidence == Confidence.MEDIUM


def test_detects_module_level_os_environ_assignment():
    src = 'import os\n\nos.environ["JAX_DISABLE_JIT"] = "1"\n\n\ndef test_a():\n    pass\n'
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].line == 3


def test_detects_module_level_path_write():
    src = (
        "from pathlib import Path\n\n"
        'Path("state.json").write_text("{}")\n\n\n'
        "def test_a():\n    pass\n"
    )
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].line == 3


def test_detects_module_level_db_connect():
    src = 'import sqlite3\n\n_DB = sqlite3.connect("test.db")\n\n\ndef test_a():\n    pass\n'
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].line == 3
    assert findings[0].confidence == Confidence.MEDIUM


def test_downgrades_confidence_for_client_naming_heuristic():
    src = (
        "from mylib import HeavyInferenceClient\n\n"
        "AGENT = HeavyInferenceClient(config={})\n\n\n"
        "def test_a():\n    AGENT.run()\n"
    )
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].confidence == Confidence.ADVISORY


def test_fp_guard_pytest_importorskip():
    src = 'import pytest\n\npytest.importorskip("torch")\n\n\ndef test_a():\n    pass\n'
    assert _run(src) == []


def test_fp_guard_module_level_constants():
    src = (
        "TIMEOUT_SECONDS = 30\n"
        'BASE_URL = "http://example.com"\n\n\n'
        "def test_a():\n    assert TIMEOUT_SECONDS == 30\n"
    )
    assert _run(src) == []


def test_fp_guard_path_construction_without_io():
    src = (
        "from pathlib import Path\n\n"
        'DATA_DIR = Path(__file__).parent / "fixtures"\n'
        'FIXTURE = Path("fixtures/x.json")\n\n\n'
        "def test_a():\n    assert FIXTURE.name\n"
    )
    assert _run(src) == []


def test_fp_guard_call_inside_fixture_not_module_level():
    src = (
        "import pytest, sqlite3\n\n\n"
        "@pytest.fixture\n"
        "def db():\n"
        '    conn = sqlite3.connect(":memory:")\n'
        "    yield conn\n"
        "    conn.close()\n\n\n"
        '@pytest.mark.parametrize("x", [1, 2])\n'
        "def test_a(db, x):\n"
        "    assert db\n"
        "    assert x\n"
    )
    assert _run(src) == []


def test_fp_guard_under_dunder_main_guard():
    src = (
        "import requests\n\n"
        'if __name__ == "__main__":\n'
        '    requests.get("http://example.com")\n\n\n'
        "def test_a():\n    pass\n"
    )
    assert _run(src) == []


def test_fp_guard_conftest_is_excluded():
    src = 'import os\n\nos.environ["FLAKEHOUND_TEST_ENV"] = "1"\n'
    assert _run(src, name="conftest.py", is_conftest=True) == []
