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

from flakehound.rules._imports import build_alias_map, resolve_call, resolve_expr
from flakehound.rules.base import Confidence, FileContext, Finding, Rule, register

_CACHING_SCOPES = {"module", "session"}


def _is_jax_jit(node: ast.expr, aliases: dict[str, str]) -> bool:
    """`jax.jit` used bare (`@jax.jit`) or called (`jax.jit(...)`), alias-resolved
    so `from jax import jit; @jit` and `import jax.numpy as ...`-style aliasing of
    `jax` itself also match."""
    target = node.func if isinstance(node, ast.Call) else node
    return resolve_expr(target, aliases) == "jax.jit"


def _wraps_jax_jit_via_partial(node: ast.expr, aliases: dict[str, str]) -> bool:
    """True for `functools.partial(jax.jit, ...)` (any alias of either name), used
    as a decorator: `@functools.partial(jax.jit, static_argnums=0)`."""
    if not isinstance(node, ast.Call):
        return False
    if resolve_call(node, aliases) != "functools.partial":
        return False
    if not node.args:
        return False
    return resolve_expr(node.args[0], aliases) == "jax.jit"


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
        aliases = build_alias_map(ctx.tree)
        if ctx.is_test_file:
            yield from self._check_module_level(ctx, aliases)
        yield from self._check_fixtures(ctx, aliases)

    def _check_module_level(self, ctx: FileContext, aliases: dict[str, str]) -> Iterable[Finding]:
        """`@jax.jit` / `NAME = jax.jit(...)` as a top-level module statement.

        Only the module's own top-level body is scanned — anything nested inside a
        function (test body, helper, function-scoped fixture) is a different,
        deliberately-safe shape (see the fix suggestion) and must not be flagged.
        """
        for stmt in ctx.tree.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield from self._check_module_level_def(ctx, stmt, aliases)
            elif isinstance(stmt, ast.Assign):
                yield from self._check_module_level_assign(ctx, stmt, aliases)

    def _check_module_level_def(
        self,
        ctx: FileContext,
        stmt: ast.FunctionDef | ast.AsyncFunctionDef,
        aliases: dict[str, str],
    ) -> Iterable[Finding]:
        for dec in stmt.decorator_list:
            if _is_jax_jit(dec, aliases):
                label = "@jax.jit"
            elif _wraps_jax_jit_via_partial(dec, aliases):
                # `functools.partial(jax.jit, ...)` used *as a decorator* is
                # immediately applied to `stmt` by Python's decorator syntax, so
                # (unlike a bare assignment/fixture return of the same partial)
                # this always compiles at module import time.
                label = "@functools.partial(jax.jit, ...)"
            else:
                continue
            yield self.finding(
                ctx,
                dec,
                f"`{stmt.name}` is compiled with `{label}` at module scope; "
                "the compiled function is shared by every test in this file "
                "for the life of the test session",
            )
            return

    def _check_module_level_assign(
        self, ctx: FileContext, stmt: ast.Assign, aliases: dict[str, str]
    ) -> Iterable[Finding]:
        if not isinstance(stmt.value, ast.Call) or not _is_jax_jit(stmt.value, aliases):
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

    def _check_fixtures(self, ctx: FileContext, aliases: dict[str, str]) -> Iterable[Finding]:
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
                if isinstance(node, ast.Call) and _is_jax_jit(node, aliases):
                    yield self.finding(
                        ctx,
                        node,
                        f"fixture `{func.name}` (scope={scope!r}) compiles with "
                        "`jax.jit(...)` and hands the compiled function to every "
                        "test that requests it; recompilation timing and, on GPU, "
                        "the device-memory cache can become order-dependent",
                    )
