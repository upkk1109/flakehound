"""Load [tool.flakehound] config from pyproject.toml (nearest ancestor)."""

from __future__ import annotations

import sys
from pathlib import Path

from flakehound.scanner import ScanConfig

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:  # graceful: config is optional
        tomllib = None  # type: ignore[assignment]


def find_pyproject(start: Path) -> Path | None:
    cur = start.resolve()
    for parent in [cur, *cur.parents]:
        candidate = parent / "pyproject.toml"
        if candidate.is_file():
            return candidate
    return None


def load_config(start: Path | None = None) -> ScanConfig:
    start = start or Path.cwd()
    pyproject = find_pyproject(start)
    if pyproject is None or tomllib is None:
        return ScanConfig()
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):  # type: ignore[union-attr]
        return ScanConfig()
    section = data.get("tool", {}).get("flakehound", {})
    return ScanConfig(
        disabled=frozenset(section.get("disable", [])),
        exclude=tuple(section.get("exclude", [])),
        include_ml_rules=bool(section.get("ml_rules", True)),
        fail_on=str(section.get("fail_on", "high")),
    )
