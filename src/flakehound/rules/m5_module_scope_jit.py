"""M5: JAX `jit()` compiled at module scope, or cached in a module/session fixture.

`jax.jit(...)` applied at module level in a test file — or a module-/session-scoped
pytest fixture that hands out a `jax.jit(...)`-wrapped function — compiles once at
collection/first-use time and shares the compiled callable across every test in the
module for the rest of the session. A JIT'd function is pure, so sharing it is
usually harmless when every caller feeds it the same input shape; the real risk is
compilation-cache order effects (a call site with a new input shape forces a
recompile whose *timing* becomes test-order-dependent) and, on GPU, OOM depending on
which test happened to run — and leave device memory fragmented — first. Static
analysis cannot see call-site input shapes or device placement, so this rule is
advisory only: worth a look, never a hard fail. Fix: compile inside the test or a
function-scoped fixture, or call `jax.clear_caches()` in teardown for heavy suites.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable

from flakehound.rules.base import Confidence, FileContext, Finding, Rule, register

_CACHING_SCOPES = {"module", "session"}


def _dotted(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _is_jax_jit(node: ast.expr) -> bool:
    """`jax.jit` used bare (`@jax.jit`) or called (`jax.jit(...)`)."""
    target = node.func if isinstance(node, ast.Call) else node
    return _dotted(target) == "jax.jit"


def _decorator_name(dec: ast.expr) -> str | None:
    target = dec.func if isinstance(dec, ast.Call) else dec
    if isinstance(target, ast.Attribute):
        return target.attr
    if isinstance(target, ast.Name):
        return target.id
    return None


def _fixture_scope(dec: ast.expr) -> str | None:
    """This decorator's pytest fixture scope, or None if it isn't `@pytest.fixture`."""
    if _decorator_name(dec) != "fixture":
        return None
    if not isinstance(dec, ast.Call):
        return "function"
    for kw in dec.keywords:
        if (
            kw.arg == "scope"
            and isinstance(kw.value, ast.Constant)
            and isinstance(kw.value.value, str)
        ):
            return kw.value.value
    return "function"


def _fixture_scope_of(func: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    for dec in func.decorator_list:
        scope = _fixture_scope(dec)
        if scope is not None:
            return scope
    return None


def _walk_body(node: ast.AST) -> Iterable[ast.AST]:
    """Descendants of ``node``, not crossing into a nested def/class/lambda scope."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        yield child
        yield from _walk_body(child)


@register
class ModuleScopeJit(Rule):
    id = "M5"
    name = "module-scope-jit"
    cause = "ml-jax/compilation-cache-order"
    confidence = Confidence.ADVISORY
    fix_suggestion = (
        "compile inside the test or a function-scoped fixture instead; if "
        "recompiling every test is too slow, keep the wider-scoped fixture but call "
        "`jax.clear_caches()` in an autouse teardown so compilation-cache and "
        "device-memory state don't carry across tests"
    )

    def check(self, ctx: FileContext) -> Iterable[Finding]:
        if ctx.is_test_file:
            yield from self._check_module_level(ctx)
        yield from self._check_fixtures(ctx)

    def _check_module_level(self, ctx: FileContext) -> Iterable[Finding]:
        """`@jax.jit` / `NAME = jax.jit(...)` as a top-level module statement.

        Only the module's own top-level body is scanned — anything nested inside a
        function (test body, helper, function-scoped fixture) is a different,
        deliberately-safe shape (see the fix suggestion) and must not be flagged.
        """
        for stmt in ctx.tree.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield from self._check_module_level_def(ctx, stmt)
            elif isinstance(stmt, ast.Assign):
                yield from self._check_module_level_assign(ctx, stmt)

    def _check_module_level_def(
        self, ctx: FileContext, stmt: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> Iterable[Finding]:
        for dec in stmt.decorator_list:
            if _is_jax_jit(dec):
                yield self.finding(
                    ctx,
                    dec,
                    f"`{stmt.name}` is compiled with `@jax.jit` at module scope; "
                    "the compiled function is shared by every test in this file "
                    "for the life of the test session",
                )
                return

    def _check_module_level_assign(self, ctx: FileContext, stmt: ast.Assign) -> Iterable[Finding]:
        if not isinstance(stmt.value, ast.Call) or not _is_jax_jit(stmt.value):
            return
        target = stmt.targets[0]
        name = target.id if isinstance(target, ast.Name) else "<value>"
        yield self.finding(
            ctx,
            stmt,
            f"`{name} = jax.jit(...)` compiles at module import time; the "
            "compiled function is shared by every test in this file for the "
            "life of the test session",
        )

    def _check_fixtures(self, ctx: FileContext) -> Iterable[Finding]:
        """A `scope="module"`/`"session"` fixture that manufactures a jitted function.

        Not gated on ``is_test_file``: these fixtures commonly live in conftest.py,
        which is exactly where the shared-across-files caching risk matters most.
        """
        for func in ast.walk(ctx.tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            scope = _fixture_scope_of(func)
            if scope not in _CACHING_SCOPES:
                continue
            for node in _walk_body(func):
                if isinstance(node, ast.Call) and _is_jax_jit(node):
                    yield self.finding(
                        ctx,
                        node,
                        f"fixture `{func.name}` (scope={scope!r}) compiles with "
                        "`jax.jit(...)` and hands the compiled function to every "
                        "test that requests it; recompilation timing and, on GPU, "
                        "the device-memory cache can become order-dependent",
                    )
