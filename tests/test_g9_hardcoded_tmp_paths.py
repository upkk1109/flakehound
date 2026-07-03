"""G9 rule tests -- the TP/FP-guard pattern every rule must follow."""

from __future__ import annotations

import ast

from flakehound.rules.base import Confidence, FileContext
from flakehound.rules.g9_hardcoded_tmp_paths import HardcodedTmpPath


def _ctx(source: str, name: str = "test_x.py") -> FileContext:
    return FileContext(
        path=name,
        source=source,
        tree=ast.parse(source),
        is_test_file=True,
        is_conftest=False,
    )


def _run(source: str):
    return list(HardcodedTmpPath().check(_ctx(source)))


# -- true positives ----------------------------------------------------------


def test_detects_hardcoded_tmp_subpath():
    src = "import os\ndef test_a():\n    os.makedirs('/tmp/strategy_tests', exist_ok=True)\n"
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].rule_id == "G9"
    assert findings[0].line == 3
    assert findings[0].confidence == Confidence.HIGH


def test_detects_tempfile_mktemp():
    src = "import tempfile\ndef test_a():\n    path = tempfile.mktemp()\n"
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].rule_id == "G9"
    assert findings[0].line == 3


def test_detects_bare_mktemp_import():
    src = "from tempfile import mktemp\ndef test_a():\n    p = mktemp()\n"
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].line == 3


def test_detects_relative_output_path_opened_for_write():
    src = "def test_a():\n    f = open('output.txt', 'w')\n    f.write('x')\n"
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].rule_id == "G9"
    assert findings[0].line == 2


def test_bare_tmp_root_is_downgraded_to_medium():
    src = "import os\ndef test_a():\n    d = os.environ.get('TMPDIR', '/tmp')\n"
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].confidence == Confidence.MEDIUM


# -- false-positive guards ----------------------------------------------------


def test_fp_guard_tmp_path_fixture_used_for_open():
    src = "def test_a(tmp_path):\n    f = open(tmp_path / 'output.txt', 'w')\n    f.write('x')\n"
    assert _run(src) == []


def test_fp_guard_tmpdir_fixture_join_for_open():
    src = (
        "import os\n"
        "def test_a(tmpdir):\n"
        "    f = open(os.path.join(str(tmpdir), 'output.txt'), 'w')\n"
        "    f.write('x')\n"
    )
    assert _run(src) == []


def test_fp_guard_safe_tempfile_apis():
    src = (
        "import tempfile\n"
        "def test_a():\n"
        "    fd, path = tempfile.mkstemp()\n"
        "    with tempfile.TemporaryDirectory() as d:\n"
        "        pass\n"
    )
    assert _run(src) == []


def test_fp_guard_read_only_open_of_checked_in_fixture():
    src = (
        "def test_a():\n    with open('tests/fixtures/data.json') as f:\n        data = f.read()\n"
    )
    assert _run(src) == []


def test_fp_guard_negative_assertion_on_tmp_literal():
    src = (
        "def test_a(cache_dir_env):\n"
        "    assert cache_dir_env != '/tmp/jax_cache'\n"
        "    assert not cache_dir_env.startswith('/tmp/')\n"
    )
    assert _run(src) == []
