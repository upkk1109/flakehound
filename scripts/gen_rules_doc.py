#!/usr/bin/env python
"""Regenerate docs/rules.md from the live rule registry.

The catalog is generated, not hand-maintained: every entry's id/name/tier/cause
comes straight from `flakehound.rules.all_rules()`, and the bad/good code samples
are lifted verbatim from that rule's own true-positive (`test_detects_*`) and
false-positive-guard (`test_fp_guard_*`) tests — so the catalog can never drift
from what the rule actually does.

Usage:
    .venv/bin/python scripts/gen_rules_doc.py            # print to stdout
    .venv/bin/python scripts/gen_rules_doc.py --write     # overwrite docs/rules.md
"""

from __future__ import annotations

import argparse
import ast
import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from flakehound.rules import all_rules  # noqa: E402


def _test_path_for(rule_cls: type) -> Path:
    modname = rule_cls.__module__.rsplit(".", 1)[-1]
    return REPO_ROOT / "tests" / f"test_{modname}.py"


def _extract_src(test_path: Path, prefix: str) -> str:
    """Return the literal `src = "..."` string from the first top-level test
    function whose name starts with `prefix`."""
    tree = ast.parse(test_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not (isinstance(node, ast.FunctionDef) and node.name.startswith(prefix)):
            continue
        for stmt in node.body:
            if not (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and stmt.targets[0].id == "src"
            ):
                continue
            try:
                value = ast.literal_eval(stmt.value)
            except ValueError:
                continue
            if isinstance(value, str):
                return value
    raise ValueError(f"no `{prefix}*` test with a literal `src = ...` found in {test_path}")


def _split_docstring(doc: str) -> tuple[str, str]:
    """Module docstrings are `"<ID>: <title>.\\n\\n<body>"` — split them."""
    doc = (doc or "").strip()
    title, _, body = doc.partition("\n\n")
    title = title.split(": ", 1)[-1].rstrip(".")
    return title, body.strip()


def _fence(code: str) -> str:
    return f"```python\n{code.rstrip(chr(10))}\n```"


def render(rule_classes: list[type]) -> str:
    lines = [
        "# Rule catalog",
        "",
        "Generated from the live rule registry by `scripts/gen_rules_doc.py` — do not edit",
        "by hand; regenerate instead:",
        "",
        "```bash",
        ".venv/bin/python scripts/gen_rules_doc.py --write",
        "```",
        "",
        f"{len(rule_classes)} rules today: `G1`–`G12` general Python flakiness causes "
        "(ranked by measured frequency in a 22k-project study), `M1`–`M5` the ML pack "
        "(JAX/PyTorch/NumPy-aware). Every finding prints its `[ID/tier]`, cause, and a `fix:` "
        "suggestion inline. Check here (and open PRs) before claiming a new rule ID — see "
        "[CONTRIBUTING.md](../CONTRIBUTING.md).",
        "",
        "| ID | Rule | Tier | Cause |",
        "|---|---|---|---|",
    ]
    for cls in rule_classes:
        anchor = f"{cls.id.lower()}-{cls.name}"
        lines.append(
            f"| [{cls.id}](#{anchor}) | `{cls.name}` | {cls.confidence.value} | {cls.cause} |"
        )
    lines.append("")

    for cls in rule_classes:
        test_path = _test_path_for(cls)
        bad = _extract_src(test_path, "test_detects")
        good = _extract_src(test_path, "test_fp_guard")
        module = importlib.import_module(cls.__module__)
        title, body = _split_docstring(module.__doc__ or "")
        lines += [
            f"## {cls.id}: {cls.name}",
            "",
            f"*{title}.*",
            "",
            f"**Tier:** `{cls.confidence.value}`  **Cause:** `{cls.cause}`",
            "",
            body,
            "",
            "**Bad:**",
            "",
            _fence(bad),
            "",
            "**Good:**",
            "",
            _fence(good),
            "",
            f"**Fix:** {cls.fix_suggestion}",
            "",
            "---",
            "",
        ]
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write", action="store_true", help="write docs/rules.md instead of stdout"
    )
    args = parser.parse_args()

    def sort_key(item: tuple[str, type]) -> tuple[str, int]:
        rule_id, _ = item
        return (rule_id[0], int(rule_id[1:]))

    rule_classes = [cls for _, cls in sorted(all_rules().items(), key=sort_key)]
    out = render(rule_classes)
    if args.write:
        out_path = REPO_ROOT / "docs" / "rules.md"
        out_path.write_text(out, encoding="utf-8")
        sys.stderr.write(f"wrote {out_path} ({len(rule_classes)} rules)\n")
    else:
        sys.stdout.write(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
