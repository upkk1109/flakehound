from __future__ import annotations

import importlib
import sys
from pathlib import Path

from flakehound import __version__
from flakehound.cli import main
from flakehound.rules.base import all_rules
from flakehound.scanner import ScanConfig, scan_paths

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

# The corpus this release ships: G1-G12 + M1-M5 (see AGENTS.md / docs/rules.md).
# A rule module that fails to auto-import, or whose @register silently no-ops,
# must fail this test rather than shrink the registry unnoticed.
_EXPECTED_RULE_IDS = {f"G{i}" for i in range(1, 13)} | {f"M{i}" for i in range(1, 6)}


def test_version_matches_pyproject():
    """`__version__` must not drift from whatever pyproject.toml resolves to.

    Handles both a static ``[project] version`` and (the single-sourced form)
    ``dynamic = ["version"]`` resolved via ``[tool.setuptools.dynamic]``.
    """
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    project = data["project"]
    if "version" in project:
        assert __version__ == project["version"]
    else:
        assert "version" in project.get("dynamic", [])
        attr_path = data["tool"]["setuptools"]["dynamic"]["version"]["attr"]
        module_name, _, attr_name = attr_path.rpartition(".")
        resolved = getattr(importlib.import_module(module_name), attr_name)
        assert __version__ == resolved


def test_registry_discovers_exact_rule_set():
    rules = all_rules()
    assert set(rules) == _EXPECTED_RULE_IDS
    for rule_id, cls in rules.items():
        assert cls.id == rule_id  # dict key must match the class's own id


def test_scan_on_fixture(tmp_path: Path):
    f = tmp_path / "test_bad.py"
    f.write_text("import numpy as np\ndef test_a():\n    np.random.seed(1)\n")
    result = scan_paths([tmp_path], ScanConfig())
    assert result.files_scanned == 1
    assert any(x.rule_id == "G1" for x in result.findings)


def test_cli_json_exit_codes(tmp_path: Path, capsys):
    bad = tmp_path / "test_bad.py"
    bad.write_text("import random\ndef test_a():\n    random.seed(7)\n")
    rc = main(["scan", str(tmp_path), "--format", "json"])
    out = capsys.readouterr().out
    assert rc == 1 and '"G1"' in out
    rc_never = main(["scan", str(tmp_path), "--fail-on", "never"])
    assert rc_never == 0
