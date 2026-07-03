"""G9: hardcoded `/tmp` (or other shared, non-isolated) paths in tests.

A bare string literal under `/tmp`, `tempfile.mktemp()` (which returns a path
without creating it -- a classic time-of-check/time-of-use race), or a plain
relative filename opened for write are all *not* isolated per test process.
Two tests reusing the same literal path collide under `pytest-xdist` parallel
workers, or when `pytest-randomly` interleaves files that happen to share a
subdirectory/file name; a killed previous run can also leave stale state that
pollutes a later "file does not exist yet" assertion. `tmp_path` (or
`tmp_path_factory`) gives every test its own directory, cleaned up
automatically, and removes the whole failure class.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable

from flakehound.rules.base import Confidence, FileContext, Finding, Rule, register

_COMPARISON_LIKE_ATTRS = {"startswith", "endswith"}
_WRITE_MODE_CHARS = set("wax+")


def _dotted(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _comparison_operand_ids(tree: ast.AST) -> set[int]:
    """Literals used only to *check* a path (`== "/tmp/x"`, `.startswith("/tmp/")`)
    are assertions about correctness, not a filesystem write -- not a collision
    risk. Skip them so we don't flag the assertion that guards against the bug.
    """
    skip: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            skip.add(id(node.left))
            skip.update(id(c) for c in node.comparators)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _COMPARISON_LIKE_ATTRS
        ):
            skip.update(id(a) for a in node.args)
    return skip


def _open_mode(node: ast.Call) -> str | None:
    if len(node.args) > 1:
        arg = node.args[1]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value
        return None
    for kw in node.keywords:
        if (
            kw.arg == "mode"
            and isinstance(kw.value, ast.Constant)
            and isinstance(kw.value.value, str)
        ):
            return kw.value.value
    return None  # default mode is "r" -- a read, not a write


def _check_literal(
    rule: Rule, ctx: FileContext, node: ast.Constant, skip_ids: set[int]
) -> Iterable[Finding]:
    if id(node) in skip_ids:
        return
    value = node.value
    if not isinstance(value, str):
        return
    if value == "/tmp":
        yield rule.finding(
            ctx,
            node,
            'bare `"/tmp"` literal points at a shared, machine-wide directory, '
            "not a test-isolated one",
            confidence=Confidence.MEDIUM,
        )
    elif value.startswith("/tmp/") and len(value) > len("/tmp/"):
        yield rule.finding(
            ctx,
            node,
            f"hardcoded path `{value!r}` under the shared system tmp dir is not "
            "isolated per test process/worker",
        )


def _check_mktemp(rule: Rule, ctx: FileContext, node: ast.Call) -> Iterable[Finding]:
    dotted = _dotted(node.func)
    is_bare_mktemp = isinstance(node.func, ast.Name) and node.func.id == "mktemp"
    if dotted == "tempfile.mktemp" or is_bare_mktemp:
        yield rule.finding(
            ctx,
            node,
            "`tempfile.mktemp()` returns a path without creating it (a "
            "time-of-check/time-of-use race) and is not test-isolated",
        )


def _check_relative_write(rule: Rule, ctx: FileContext, node: ast.Call) -> Iterable[Finding]:
    if not (isinstance(node.func, ast.Name) and node.func.id == "open"):
        return
    if not node.args:
        return
    target = node.args[0]
    if not (isinstance(target, ast.Constant) and isinstance(target.value, str)):
        return
    path = target.value
    if not path or path.startswith("/"):
        return  # empty, or absolute (the /tmp case is covered by _check_literal)
    mode = _open_mode(node)
    if mode is None or not (_WRITE_MODE_CHARS & set(mode)):
        return  # unknown or read-only mode: no static evidence of a write
    yield rule.finding(
        ctx,
        node,
        f"`open({path!r}, {mode!r})` writes to a shared relative path; concurrent "
        "runs/workers or a leftover file from a previous run can collide",
    )


@register
class HardcodedTmpPath(Rule):
    id = "G9"
    name = "hardcoded-tmp-paths"
    cause = "filesystem/shared-state"
    confidence = Confidence.HIGH
    fix_suggestion = (
        "use the `tmp_path` fixture (or `tmp_path_factory` for session-scoped "
        "needs) instead of a hardcoded path; pytest gives every test its own "
        "isolated, auto-cleaned directory"
    )

    def check(self, ctx: FileContext) -> Iterable[Finding]:
        skip_ids = _comparison_operand_ids(ctx.tree)
        for node in ast.walk(ctx.tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                yield from _check_literal(self, ctx, node, skip_ids)
            elif isinstance(node, ast.Call):
                yield from _check_mktemp(self, ctx, node)
                yield from _check_relative_write(self, ctx, node)
