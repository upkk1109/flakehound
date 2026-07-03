"""G3: `time.sleep`/`asyncio.sleep` used as ad hoc test synchronization.

A fixed sleep before asserting on background-thread or async-task state assumes
the awaited work finishes inside that window — true on a quiet machine, false
the moment a CI runner is under contention and the real event takes longer
than the guessed delay. `thread.start()` / `publisher.start()` followed by a
bare `time.sleep(N)` and then a direct assertion, with no actual wait
primitive in between, is one of the most common raw flakiness idioms in
real-world suites.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable

from flakehound.rules.base import Confidence, FileContext, Finding, Rule, register

_SLEEP_CALLS = {
    ("time", "sleep"),
    ("asyncio", "sleep"),
}

_PATCH_MARKERS = ("setattr", "patch")


def _dotted(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _iter_test_or_fixture_funcs(
    node: ast.AST,
) -> Iterable[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Top-level and class-method defs that are pytest test functions or fixtures.

    Deliberately does not descend into a function's own body looking for more
    defs to qualify (nested helpers are swept for sleeps via ``_iter_calls``
    below, attributed to their enclosing test/fixture, but are not themselves
    separate qualifying scopes).
    """
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
            if _is_test_or_fixture(child):
                yield child
        elif isinstance(child, ast.ClassDef):
            yield from _iter_test_or_fixture_funcs(child)


def _is_test_or_fixture(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    if func.name.startswith("test_"):
        return True
    for dec in func.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        attr = (
            target.attr
            if isinstance(target, ast.Attribute)
            else (target.id if isinstance(target, ast.Name) else None)
        )
        if attr == "fixture":
            return True
    return False


def _iter_calls(node: ast.AST, in_loop: bool = False) -> Iterable[tuple[ast.Call, bool]]:
    """Yield every Call in ``node``'s subtree with whether it sits under a loop."""
    if isinstance(node, ast.Call):
        yield node, in_loop
    child_in_loop = in_loop or isinstance(node, ast.While | ast.For | ast.AsyncFor)
    for child in ast.iter_child_nodes(node):
        yield from _iter_calls(child, child_in_loop)


def _monkeypatches_sleep(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True if the function stubs out something named "sleep" (monkeypatch/mock)."""
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        dotted = _dotted(node.func) or ""
        tail = dotted.rsplit(".", 1)[-1]
        if not (tail in _PATCH_MARKERS or "patch" in dotted):
            continue
        args = [*node.args, *node.keywords]
        for arg in args:
            value = arg.value if isinstance(arg, ast.keyword) else arg
            if (
                isinstance(value, ast.Constant)
                and isinstance(value.value, str)
                and "sleep" in value.value
            ):
                return True
    return False


def _positive_constant(call: ast.Call) -> tuple[float | int | None, bool]:
    """Return (value, is_literal) for the delay argument, unwrapping unary minus."""
    if not call.args:
        return None, False
    arg = call.args[0]
    if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
        inner = arg.operand
        if (
            isinstance(inner, ast.Constant)
            and isinstance(inner.value, int | float)
            and not isinstance(inner.value, bool)
        ):
            return -inner.value, True
        return None, False
    if (
        isinstance(arg, ast.Constant)
        and isinstance(arg.value, int | float)
        and not isinstance(arg.value, bool)
    ):
        return arg.value, True
    return None, False


@register
class SleepInTest(Rule):
    id = "G3"
    name = "sleep-in-test"
    cause = "timing/synchronization"
    confidence = Confidence.HIGH
    fix_suggestion = (
        "wait on the real condition instead of guessing a delay: an "
        "`Event`/`Condition`.wait(timeout=...), a small polling helper with a "
        "deadline, `.join(timeout=...)` on the thread, or `pytest-timeout` to "
        "bound the worst case"
    )

    def check(self, ctx: FileContext) -> Iterable[Finding]:
        for func in _iter_test_or_fixture_funcs(ctx.tree):
            if _monkeypatches_sleep(func):
                continue
            for call, in_loop in _iter_calls(func):
                dotted = _dotted(call.func)
                if dotted is None:
                    continue
                prefix, _, attr = dotted.rpartition(".")
                if (prefix, attr) not in _SLEEP_CALLS:
                    continue
                value, is_literal = _positive_constant(call)
                if is_literal and value is not None and value <= 0:
                    continue  # sleep(0) (or non-positive) is an explicit no-op wait
                if not is_literal:
                    yield self.finding(
                        ctx,
                        call,
                        f"`{dotted}(...)` delay is not a literal constant; cannot "
                        "statically confirm it is non-zero, but a sleep here is a "
                        "common synchronization smell worth a manual look",
                        confidence=Confidence.ADVISORY,
                    )
                elif in_loop:
                    yield self.finding(
                        ctx,
                        call,
                        f"`{dotted}({value!r})` inside a loop reads as a polling "
                        "wait rather than a single blind delay; still wall-clock "
                        "coupled but lower risk than a bare sleep-then-assert",
                        confidence=Confidence.MEDIUM,
                    )
                else:
                    yield self.finding(
                        ctx,
                        call,
                        f"`{dotted}({value!r})` is a fixed delay standing in for real "
                        "synchronization; flaky under CI load whenever the awaited "
                        "work takes longer than the guessed duration",
                    )
