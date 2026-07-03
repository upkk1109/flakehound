"""G10: manual asyncio event-loop management in tests.

Calling `asyncio.get_event_loop()` / `new_event_loop()` / `set_event_loop()` /
`loop.run_until_complete(...)` by hand — instead of letting pytest-asyncio or
anyio own the event loop's lifecycle — ties a test's outcome to loop state a
previous test created, closed, or never closed. The event loop (and the
process's event-loop policy) is process-global state, so this is the async
analogue of G1's global-RNG problem: order-dependence via shared mutable
state, this time a loop object instead of an RNG. `asyncio.run(...)` called
from inside an already-running `async def` test starts a second, nested loop
instead of reusing the one pytest-asyncio already provides.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable

from flakehound.rules.base import Confidence, FileContext, Finding, Rule, register

_LOOP_CALLS = {
    ("asyncio", "get_event_loop"),
    ("asyncio", "new_event_loop"),
    ("asyncio", "set_event_loop"),
}

_LOOP_LIKE_RECEIVERS = {"loop", "event_loop", "el", "_loop", "new_loop", "test_loop"}

_FIXTURE_DECORATOR_NAMES = {"fixture"}  # pytest.fixture / pytest_asyncio.fixture
_RECOGNIZED_LOOP_FIXTURE = "event_loop"


def _dotted(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _is_fixture(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for dec in func.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        dotted = _dotted(target)
        if dotted is not None and dotted.rsplit(".", 1)[-1] in _FIXTURE_DECORATOR_NAMES:
            return True
    return False


def _is_test_function(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return func.name.startswith("test")


def _receiver_is_loop_like(node: ast.expr) -> bool:
    if isinstance(node, ast.Name):
        return node.id in _LOOP_LIKE_RECEIVERS or node.id.endswith("loop")
    if isinstance(node, ast.Attribute):
        return node.attr in _LOOP_LIKE_RECEIVERS or node.attr.endswith("loop")
    if isinstance(node, ast.Call):
        dotted = _dotted(node.func)
        if dotted is not None:
            prefix, _, attr = dotted.rpartition(".")
            return (prefix, attr) in _LOOP_CALLS
    return False


@register
class EventLoopMisuse(Rule):
    id = "G10"
    name = "event-loop-misuse"
    cause = "concurrency/event-loop-lifecycle"
    confidence = Confidence.MEDIUM
    fix_suggestion = (
        "let pytest-asyncio (or anyio) own the loop: use `@pytest.mark.asyncio` "
        "with an explicit loop scope (e.g. `@pytest.mark.asyncio(loop_scope="
        '"function")`, or anyio\'s equivalent) instead of calling '
        "`asyncio.get_event_loop()`/`new_event_loop()`/`set_event_loop()`/"
        "`run_until_complete()` by hand"
    )

    def check(self, ctx: FileContext) -> Iterable[Finding]:
        yield from _walk(ctx.tree, ctx, self, enclosing_func=None, suppressed=False)


def _walk(
    node: ast.AST,
    ctx: FileContext,
    rule: EventLoopMisuse,
    enclosing_func: ast.FunctionDef | ast.AsyncFunctionDef | None,
    suppressed: bool,
) -> Iterable[Finding]:
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if _is_fixture(child) and child.name == _RECOGNIZED_LOOP_FIXTURE:
                # pytest-asyncio's documented custom-loop override: recognized,
                # not flagged.
                continue
            child_in_scope = _is_test_function(child) or _is_fixture(child)
            yield from _walk(child, ctx, rule, child, not child_in_scope)
            continue
        if isinstance(child, ast.Call) and not suppressed:
            yield from _check_call(child, ctx, rule, enclosing_func)
        yield from _walk(child, ctx, rule, enclosing_func, suppressed)


def _check_call(
    node: ast.Call,
    ctx: FileContext,
    rule: EventLoopMisuse,
    enclosing_func: ast.FunctionDef | ast.AsyncFunctionDef | None,
) -> Iterable[Finding]:
    func = node.func
    dotted = _dotted(func)

    if dotted is not None:
        prefix, _, attr = dotted.rpartition(".")
        if (prefix, attr) in _LOOP_CALLS:
            yield _loop_call_finding(ctx, rule, node, dotted, enclosing_func)
            return
        if prefix == "asyncio" and attr == "run":
            run_finding = _asyncio_run_finding(ctx, rule, node, enclosing_func)
            if run_finding is not None:
                yield run_finding
            return

    if isinstance(func, ast.Attribute) and func.attr == "run_until_complete":
        yield _run_until_complete_finding(ctx, rule, node, func.value)


def _loop_call_finding(
    ctx: FileContext,
    rule: EventLoopMisuse,
    node: ast.Call,
    dotted: str,
    enclosing_func: ast.FunctionDef | ast.AsyncFunctionDef | None,
) -> Finding:
    if enclosing_func is None:
        return rule.finding(
            ctx,
            node,
            f"`{dotted}(...)` at module scope binds an event loop shared by "
            "every test in this module for the life of the process",
        )
    return rule.finding(
        ctx,
        node,
        f"`{dotted}(...)` manages the event loop by hand instead of letting "
        "pytest-asyncio/anyio own its lifecycle; loop state can leak across "
        "test order",
    )


def _asyncio_run_finding(
    ctx: FileContext,
    rule: EventLoopMisuse,
    node: ast.Call,
    enclosing_func: ast.FunctionDef | ast.AsyncFunctionDef | None,
) -> Finding | None:
    if (
        enclosing_func is not None
        and isinstance(enclosing_func, ast.AsyncFunctionDef)
        and _is_test_function(enclosing_func)
    ):
        return rule.finding(
            ctx,
            node,
            "`asyncio.run(...)` called from inside an `async def` test starts "
            "a second, nested event loop instead of using the one "
            "pytest-asyncio already provides for this test",
        )
    return None


def _run_until_complete_finding(
    ctx: FileContext, rule: EventLoopMisuse, node: ast.Call, receiver: ast.expr
) -> Finding:
    dotted = _dotted(node.func) or "<expr>.run_until_complete"
    confidence = Confidence.MEDIUM if _receiver_is_loop_like(receiver) else Confidence.ADVISORY
    return rule.finding(
        ctx,
        node,
        f"`{dotted}(...)` drives a coroutine via a manually managed loop "
        "instead of pytest-asyncio/anyio; the loop's create/close lifecycle "
        "is now this test's responsibility",
        confidence=confidence,
    )
