"""Scanner integration tests: file-discovery edge cases and containment."""

from __future__ import annotations

from pathlib import Path

import pytest

from flakehound.rules.base import Finding
from flakehound.scanner import ScanConfig, is_test_path, iter_test_files, scan_paths


def test_empty_file_scans_clean(tmp_path: Path):
    f = tmp_path / "test_empty.py"
    f.write_text("")
    result = scan_paths([tmp_path], ScanConfig())
    assert result.files_scanned == 1
    assert result.findings == []
    assert result.parse_errors == []


def test_syntax_error_file_recorded_without_crash(tmp_path: Path):
    f = tmp_path / "test_broken.py"
    f.write_text("def test_a(:\n    pass\n")
    result = scan_paths([tmp_path], ScanConfig())
    assert result.files_scanned == 0
    assert len(result.parse_errors) == 1
    assert "test_broken.py" in result.parse_errors[0]


def test_syntax_error_file_does_not_block_other_files(tmp_path: Path):
    (tmp_path / "test_broken.py").write_text("def test_a(:\n    pass\n")
    good = tmp_path / "test_good.py"
    good.write_text("import random\ndef test_a():\n    random.seed(1)\n")
    result = scan_paths([tmp_path], ScanConfig())
    assert result.files_scanned == 1
    assert len(result.parse_errors) == 1
    assert any(f.rule_id == "G1" for f in result.findings)


def test_unicode_filename_and_content_scans_clean(tmp_path: Path):
    f = tmp_path / "test_ünïcödé_日本語.py"
    f.write_text(
        "# ünïcödé comment\ndef test_a():\n    assert '日本語' == '日本語'\n",
        encoding="utf-8",
    )
    result = scan_paths([tmp_path], ScanConfig())
    assert result.files_scanned == 1
    assert result.parse_errors == []


def test_symlink_escaping_root_is_not_scanned(tmp_path: Path):
    secret_dir = tmp_path / "secret"
    secret_dir.mkdir()
    secret_file = secret_dir / "test_secret.py"
    secret_file.write_text("import random\ndef test_a():\n    random.seed(1)\n")

    root = tmp_path / "root"
    root.mkdir()
    link = root / "test_link.py"
    try:
        link.symlink_to(secret_file)
    except OSError:
        pytest.skip("symlinks not supported in this environment")

    files = iter_test_files([root], exclude=())
    assert files == []

    result = scan_paths([root], ScanConfig())
    assert result.files_scanned == 0
    assert result.findings == []


def test_symlink_inside_root_pointing_inside_root_is_scanned(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    real = root / "test_real.py"
    real.write_text("import random\ndef test_a():\n    random.seed(1)\n")
    link = root / "test_link.py"
    try:
        link.symlink_to(real)
    except OSError:
        pytest.skip("symlinks not supported in this environment")

    files = iter_test_files([root], exclude=())
    # both names resolve to the same real file, so it is only scanned once
    assert len(files) == 1


def test_duplicate_roots_are_deduped(tmp_path: Path):
    f = tmp_path / "test_a.py"
    f.write_text("def test_a():\n    pass\n")

    files = iter_test_files([tmp_path, tmp_path], exclude=())
    assert files == [f]


def test_duplicate_root_as_dir_and_file_is_deduped(tmp_path: Path):
    f = tmp_path / "test_a.py"
    f.write_text("def test_a():\n    pass\n")

    files = iter_test_files([tmp_path, f], exclude=())
    assert files == [f]


def test_non_test_files_are_ignored(tmp_path: Path):
    (tmp_path / "helpers.py").write_text("def helper():\n    pass\n")
    (tmp_path / "__init__.py").write_text("")
    (tmp_path / "utils.py").write_text("X = 1\n")

    files = iter_test_files([tmp_path], exclude=())
    assert files == []


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("test_foo.py", True),
        ("foo_test.py", True),
        ("conftest.py", True),
        ("foo.py", False),
        ("test_foo.txt", False),
        ("helpers.py", False),
    ],
)
def test_is_test_path(name: str, expected: bool):
    assert is_test_path(Path(name)) is expected


def test_is_test_file_and_is_conftest_derivation(tmp_path: Path, monkeypatch):
    captured: list = []

    class _CaptureRule:
        id = "X0"

        def check(self, ctx):
            captured.append(ctx)
            return []

    monkeypatch.setattr(
        "flakehound.scanner.make_rules",
        lambda disabled=frozenset(), predicate=None: [_CaptureRule()],
    )

    (tmp_path / "test_a.py").write_text("def test_a():\n    pass\n")
    (tmp_path / "conftest.py").write_text("")

    scan_paths([tmp_path], ScanConfig())

    by_name = {Path(ctx.path).name: ctx for ctx in captured}
    assert by_name["test_a.py"].is_test_file is True
    assert by_name["test_a.py"].is_conftest is False
    assert by_name["conftest.py"].is_test_file is False
    assert by_name["conftest.py"].is_conftest is True


def test_exclude_matches_posix_style_pattern(tmp_path: Path):
    sub = tmp_path / "fixtures"
    sub.mkdir()
    excluded = sub / "test_excluded.py"
    excluded.write_text("def test_a():\n    pass\n")
    kept = tmp_path / "test_kept.py"
    kept.write_text("def test_a():\n    pass\n")

    files = iter_test_files([tmp_path], exclude=("*/fixtures/*",))
    assert files == [kept]


def test_scan_result_findings_are_sorted(tmp_path: Path):
    (tmp_path / "test_b.py").write_text("import random\ndef test_a():\n    random.seed(2)\n")
    (tmp_path / "test_a.py").write_text("import random\ndef test_a():\n    random.seed(1)\n")

    result = scan_paths([tmp_path], ScanConfig())
    paths = [f.path for f in result.findings]
    assert paths == sorted(paths)
    assert all(isinstance(f, Finding) for f in result.findings)
