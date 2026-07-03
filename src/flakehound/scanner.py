"""Static scanner: walk test files, parse, run rules, collect findings."""

from __future__ import annotations

import ast
import fnmatch
from dataclasses import dataclass, field
from pathlib import Path

from flakehound.rules.base import FileContext, Finding, make_rules

TEST_FILE_PATTERNS = ("test_*.py", "*_test.py", "conftest.py")


@dataclass
class ScanConfig:
    disabled: frozenset[str] = frozenset()
    exclude: tuple[str, ...] = ()
    include_ml_rules: bool = True
    fail_on: str = "high"  # high | medium | advisory | never


@dataclass
class ScanResult:
    findings: list[Finding] = field(default_factory=list)
    files_scanned: int = 0
    parse_errors: list[str] = field(default_factory=list)


def is_test_path(path: Path) -> bool:
    return any(fnmatch.fnmatch(path.name, pat) for pat in TEST_FILE_PATTERNS)


def iter_test_files(roots: list[Path], exclude: tuple[str, ...]) -> list[Path]:
    resolved_roots = [root.resolve() for root in roots]
    seen: set[Path] = set()
    out: list[Path] = []
    for root in roots:
        candidates = [root] if root.is_file() else sorted(root.rglob("*.py"))
        for p in candidates:
            rp = p.resolve()
            if rp in seen or not is_test_path(p):
                continue
            if not any(rp == rr or rr in rp.parents for rr in resolved_roots):
                continue  # resolved path escapes every requested root (symlink traversal)
            if any(fnmatch.fnmatch(p.as_posix(), pat) for pat in exclude):
                continue
            seen.add(rp)
            out.append(p)
    return out


def scan_paths(roots: list[Path], config: ScanConfig | None = None) -> ScanResult:
    config = config or ScanConfig()
    predicate = None if config.include_ml_rules else (lambda cls: not cls.id.startswith("M"))
    rules = make_rules(disabled=config.disabled, predicate=predicate)
    result = ScanResult()
    for path in iter_test_files(roots, config.exclude):
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            result.parse_errors.append(f"{path}: {exc}")
            continue
        ctx = FileContext(
            path=str(path),
            source=source,
            tree=tree,
            is_test_file=path.name != "conftest.py",
            is_conftest=path.name == "conftest.py",
        )
        for rule in rules:
            result.findings.extend(rule.check(ctx))
        result.files_scanned += 1
    result.findings.sort(key=lambda f: (f.path, f.line, f.rule_id))
    return result


_TIER_ORDER = {"high": 0, "medium": 1, "advisory": 2}


def exit_code(result: ScanResult, fail_on: str) -> int:
    if fail_on == "never":
        return 0
    threshold = _TIER_ORDER.get(fail_on, 0)
    for f in result.findings:
        if _TIER_ORDER[f.confidence.value] <= threshold:
            return 1
    return 0
