"""G4: naive `now()` reads driving test assertions.

`datetime.now()`/`datetime.utcnow()`/`date.today()`/`time.time()` read the
real wall clock. When a test's assertion (directly, or via a variable that
was derived from one of these calls) depends on the value read, the test's
outcome now depends on *when* it happened to run — slow CI hosts, DST
transitions, and midnight/month/year boundaries all become sources of
flakiness. The fix is a frozen or injected clock, not a faster runner.

Overlaps ruff's flake8-datetimez (DTZ003 `utcnow`, DTZ005 `now`, DTZ011
`date.today`), but the framing differs: DTZ flags any naive construction,
unconditionally, because it is a timezone-correctness bug regardless of how
the value is used. G4 only fires when the clock read flows into an assertion
-- it is a flakiness rule, not a silent DTZ duplicate.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from typing import TypeGuard

from flakehound.rules.base import Confidence, FileContext, Finding, Rule, register

_NOW_CALLS = {
    ("datetime", "now"),
    ("datetime", "utcnow"),
    ("datetime.datetime", "now"),
    ("datetime.datetime", "utcnow"),
    ("dt", "now"),
    ("dt", "utcnow"),
    ("date", "today"),
    ("datetime.date", "today"),
    ("time", "time"),
}


def _dotted(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _is_clock_call(node: ast.AST) -> TypeGuard[ast.Call]:
    if not isinstance(node, ast.Call):
        return False
    dotted = _dotted(node.func)
    if dotted is None or "." not in dotted:
        return False
    prefix, _, attr = dotted.rpartition(".")
    return (prefix, attr) in _NOW_CALLS


def _first_clock_call(node: ast.AST) -> ast.Call | None:
    for sub in ast.walk(node):
        if _is_clock_call(sub):
            return sub
    return None


def _names_in(node: ast.AST) -> set[str]:
    return {sub.id for sub in ast.walk(node) if isinstance(sub, ast.Name)}


def _imports_freezegun(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(
            alias.name == "freezegun" or alias.name.startswith("freezegun.") for alias in node.names
        ):
            return True
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and (node.module == "freezegun" or node.module.startswith("freezegun."))
        ):
            return True
    return False


def _iter_functions(tree: ast.Module) -> Iterable[ast.FunctionDef | ast.AsyncFunctionDef]:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def _own_scope_nodes(func: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.AST]:
    """Descendants of ``func``, not descending into nested defs/lambdas/classes.

    Nested functions get analyzed on their own pass (``_iter_functions`` walks
    the whole tree), so this keeps each function's taint-tracking scoped to
    "this test body", matching how a reader would judge the risk.
    """
    out: list[ast.AST] = []

    def _walk(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            out.append(child)
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                continue
            _walk(child)

    _walk(func)
    return out


def _monkeypatches_clock(nodes: list[ast.AST]) -> bool:
    for node in nodes:
        if isinstance(node, ast.Call) and _dotted(node.func) == "monkeypatch.setattr":
            return True
    return False


@register
class NaiveNow(Rule):
    id = "G4"
    name = "naive-now"
    cause = "time/wall-clock-nondeterminism"
    confidence = Confidence.MEDIUM
    fix_suggestion = (
        "freeze the clock with `freezegun.freeze_time(...)` (or a "
        "`monkeypatch.setattr` clock stub), or inject the timestamp/clock as "
        "a parameter instead of reading it inside the function under test"
    )

    def check(self, ctx: FileContext) -> Iterable[Finding]:
        if _imports_freezegun(ctx.tree):
            return
        for func in _iter_functions(ctx.tree):
            yield from self._check_function(ctx, func)

    def _check_function(
        self, ctx: FileContext, func: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> Iterable[Finding]:
        nodes = _own_scope_nodes(func)
        if _monkeypatches_clock(nodes):
            return

        assert_nodes = [n for n in nodes if isinstance(n, ast.Assert)]
        names_in_asserts: set[str] = set()
        for a in assert_nodes:
            names_in_asserts |= _names_in(a.test)
        function_has_assert = bool(assert_nodes)

        reported: set[int] = set()

        # Direct signature: the clock call itself is written inside `assert ...`.
        for a in assert_nodes:
            for sub in ast.walk(a.test):
                if _is_clock_call(sub) and id(sub) not in reported:
                    reported.add(id(sub))
                    dotted = _dotted(sub.func)
                    yield self.finding(
                        ctx,
                        sub,
                        f"`{dotted}()` is read directly inside an assertion; the "
                        "test's pass/fail now depends on wall-clock timing",
                    )

        # Indirect signature: `var = now()` (optionally combined via arithmetic
        # with another tainted var), then `var` feeds a later assertion.
        propagated_from: dict[str, ast.Call] = {}
        changed = True
        while changed:
            changed = False
            for n in nodes:
                if not (
                    isinstance(n, ast.Assign)
                    and len(n.targets) == 1
                    and isinstance(n.targets[0], ast.Name)
                ):
                    continue
                tgt = n.targets[0].id
                if tgt in propagated_from:
                    continue
                call = _first_clock_call(n.value)
                if call is not None:
                    propagated_from[tgt] = call
                    changed = True
                    continue
                for name_id in _names_in(n.value):
                    if name_id in propagated_from:
                        propagated_from[tgt] = propagated_from[name_id]
                        changed = True
                        break

        arithmetic_roots: set[int] = set()
        for n in nodes:
            if not isinstance(n, ast.BinOp):
                continue
            call = _first_clock_call(n)
            if call is not None:
                arithmetic_roots.add(id(call))
            for name_id in _names_in(n):
                if name_id in propagated_from:
                    arithmetic_roots.add(id(propagated_from[name_id]))

        for var_name, root_call in propagated_from.items():
            if id(root_call) in reported:
                continue
            dotted = _dotted(root_call.func)
            if var_name in names_in_asserts:
                reported.add(id(root_call))
                yield self.finding(
                    ctx,
                    root_call,
                    f"`{dotted}()` assigned to `{var_name}`, which a later "
                    "assertion in this test depends on; freeze the clock or "
                    "inject it explicitly",
                )
            elif id(root_call) in arithmetic_roots and function_has_assert:
                reported.add(id(root_call))
                yield self.finding(
                    ctx,
                    root_call,
                    f"`{dotted}()` assigned to `{var_name}` and used in "
                    "arithmetic in a test that also asserts; confirm the "
                    "assertion isn't wall-clock-dependent, or freeze the clock",
                    confidence=Confidence.ADVISORY,
                )
            # else: no assert reference and (no arithmetic, or no assert at all
            # in this test) — perf-timing/filler usage, not flagged.
