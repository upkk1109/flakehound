from __future__ import annotations

from pathlib import Path

from flakehound import __version__
from flakehound.cli import main
from flakehound.rules.base import all_rules
from flakehound.scanner import ScanConfig, scan_paths


def test_version():
    assert __version__


def test_registry_discovers_rules():
    rules = all_rules()
    assert "G1" in rules
    # ids unique + well-formed
    assert all(r[0] in "GM" and r[1:].isdigit() for r in rules)


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
