"""G5: module/session/class-scoped fixture holds shared mutable state.

A fixture with `scope="module"` (or `"session"`/`"class"`) is built once and
handed to every test that requests it. If it returns or yields a mutable
object — a list/dict/set literal, or a freshly constructed class instance —
with no defensive copy, every test sharing the fixture reads and writes the
*same* object: whichever test runs first (or wherever xdist/pytest-randomly
lands it) silently seeds the state the next test sees. The same risk shows up
when the fixture body itself reaches into a module global or a shared class
attribute and mutates it directly — shared-state-via-fixture is one of the
classic order-dependence contaminators alongside global RNG mutation (G1) and
import-time construction.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable

from flakehound.rules.base import Confidence, FileContext, Finding, Rule, register

_SHARED_SCOPES = {"module", "session", "class"}

_LITERAL_KIND: dict[type, str] = {
    ast.List: "list",
    ast.ListComp: "list",
    ast.Dict: "dict",
    ast.DictComp: "dict",
    ast.Set: "set",
    ast.SetComp: "set",
}

_COPY_CALLEES = {"deepcopy", "copy"}
_IMMUTABLE_WRAPPERS = {"tuple", "frozenset", "str", "int", "float", "bool", "bytes"}
_MUTATING_METHODS = {
    "append",
    "extend",
    "insert",
    "remove",
    "pop",
    "sort",
    "reverse",
    "clear",
    "update",
    "add",
    "discard",
    "setdefault",
    "popitem",
}


def _decorator_target(dec: ast.expr) -> ast.expr:
    return dec.func if isinstance(dec, ast.Call) else dec


def _decorator_name(dec: ast.expr) -> str | None:
    target = _decorator_target(dec)
    if isinstance(target, ast.Attribute):
        return target.attr
    if isinstance(target, ast.Name):
        return target.id
    return None


def _fixture_scope(dec: ast.expr) -> str | None:
    """Return this decorator's pytest fixture scope, or None if not a fixture."""
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
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda | ast.ClassDef):
            continue
        yield child
        yield from _walk_body(child)


def _contains(root: ast.AST, target: ast.AST) -> bool:
    return root is target or any(n is target for n in _walk_body(root))


def _callee_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _value_risk(value: ast.expr) -> str | None:
    """Classify a returned/yielded expression: "literal", "instance", or None (safe)."""
    if isinstance(value, tuple(_LITERAL_KIND)):
        return "literal"
    if isinstance(value, ast.Call):
        name = _callee_name(value.func)
        if name in _COPY_CALLEES or (
            isinstance(value.func, ast.Attribute) and value.func.attr == "copy"
        ):
            return None  # explicit defensive copy
        if name in _IMMUTABLE_WRAPPERS:
            return None
        if name and name[:1].isupper():
            return "instance"  # looks like `SomeClass(...)` — weaker, static-only evidence
        return None  # plain function/factory call — can't tell, don't guess
    return None  # Name/Tuple/Constant/BoolOp/... — not a literal we can confidently flag


def _base_name(node: ast.expr) -> str | None:
    while isinstance(node, ast.Attribute | ast.Subscript):
        node = node.value
    if isinstance(node, ast.Name):
        return node.id
    return None


def _mutation_target(node: ast.AST) -> str | None:
    """Name of the object mutated by ``node``, if it looks like a mutation."""
    if isinstance(node, ast.Assign):
        for t in node.targets:
            name = _base_name(t)
            if name:
                return name
        return None
    if isinstance(node, ast.AugAssign):
        return _base_name(node.target)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in _MUTATING_METHODS
    ):
        return _base_name(node.func.value)
    return None


def _mutates_declared_global(func: ast.FunctionDef | ast.AsyncFunctionDef) -> ast.AST | None:
    """A `global NAME` decl paired with a mutation of NAME anywhere in the body."""
    names = {n for stmt in _walk_body(func) if isinstance(stmt, ast.Global) for n in stmt.names}
    if not names:
        return None
    for node in _walk_body(func):
        if isinstance(node, ast.Global):
            continue
        if _mutation_target(node) in names:
            return node
    return None


def _mutates_class_like_attr(func: ast.FunctionDef | ast.AsyncFunctionDef) -> ast.AST | None:
    """Heuristic: mutation of `Capitalized.attr` where `Capitalized` isn't local.

    Weak on purpose — capitalization is a naming *convention*, not proof the
    target is a shared class/module singleton, so callers downgrade this to
    ADVISORY.
    """
    params = {a.arg for a in (*func.args.args, *func.args.kwonlyargs, *func.args.posonlyargs)}
    local = {
        t.id
        for stmt in _walk_body(func)
        if isinstance(stmt, ast.Assign)
        for t in stmt.targets
        if isinstance(t, ast.Name)
    }
    for node in _walk_body(func):
        name = _mutation_target(node)
        if not name or name in params or name in local:
            continue
        if name[:1].isupper():
            return node
    return None


def _yield_has_teardown(
    func: ast.FunctionDef | ast.AsyncFunctionDef, yield_node: ast.Yield
) -> bool:
    """True if body code after this yield looks like it restores state."""
    for idx, stmt in enumerate(func.body):
        if isinstance(stmt, ast.Try) and any(_contains(s, yield_node) for s in stmt.body):
            return bool(stmt.finalbody)
        if _contains(stmt, yield_node):
            return idx < len(func.body) - 1
    return False


@register
class SharedStateFixture(Rule):
    id = "G5"
    name = "shared-state-fixture"
    cause = "shared-state"
    confidence = Confidence.MEDIUM
    fix_suggestion = (
        "scope the fixture to `function` (the default) so each test gets its own "
        "object, or keep the wider scope but hand out a factory function / "
        "`copy.deepcopy(...)` / frozen data (tuple, `MappingProxyType`, a frozen "
        "dataclass) instead of the shared mutable object itself"
    )

    def check(self, ctx: FileContext) -> Iterable[Finding]:
        for func in ast.walk(ctx.tree):
            if not isinstance(func, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            scope = _fixture_scope_of(func)
            if scope not in _SHARED_SCOPES:
                continue
            yield from self._check_fixture(ctx, func, scope)

    def _check_fixture(
        self, ctx: FileContext, func: ast.FunctionDef | ast.AsyncFunctionDef, scope: str
    ) -> Iterable[Finding]:
        for ret in _walk_body(func):
            if isinstance(ret, ast.Return) and ret.value is not None:
                yield from self._value_finding(ctx, func, scope, ret, ret.value, "returns")

        for y in _walk_body(func):
            if not isinstance(y, ast.Yield) or y.value is None:
                continue
            if _yield_has_teardown(func, y):
                continue
            yield from self._value_finding(ctx, func, scope, y, y.value, "yields")

        gnode = _mutates_declared_global(func)
        if gnode is not None:
            yield self.finding(
                ctx,
                gnode,
                f"fixture `{func.name}` (scope={scope!r}) mutates a module `global` in "
                "its body; that state persists across every test that shares the "
                "fixture for as long as the scope is alive",
            )

        cnode = _mutates_class_like_attr(func)
        if cnode is not None:
            yield self.finding(
                ctx,
                cnode,
                f"fixture `{func.name}` (scope={scope!r}) appears to mutate a shared "
                "attribute on an object it did not construct locally; if that "
                "target is a class or module-level singleton, the mutation leaks "
                "between tests (static analysis can't fully confirm the target's "
                "identity — verify manually)",
                confidence=Confidence.ADVISORY,
            )

    def _value_finding(
        self,
        ctx: FileContext,
        func: ast.FunctionDef | ast.AsyncFunctionDef,
        scope: str,
        node: ast.AST,
        value: ast.expr,
        verb: str,
    ) -> Iterable[Finding]:
        risk = _value_risk(value)
        if risk == "literal":
            kind = _LITERAL_KIND[type(value)]
            yield self.finding(
                ctx,
                node,
                f"fixture `{func.name}` (scope={scope!r}) {verb} a mutable {kind} "
                "with no defensive copy; every test sharing this fixture reads and "
                "writes the same object",
            )
        elif risk == "instance":
            name = _callee_name(value.func) if isinstance(value, ast.Call) else "?"
            yield self.finding(
                ctx,
                node,
                f"fixture `{func.name}` (scope={scope!r}) {verb} a fresh `{name}(...)` "
                "instance with no defensive copy; if that class is mutable, tests "
                "sharing the fixture can leak state into each other (static analysis "
                "can't confirm the class is mutable — verify manually)",
                confidence=Confidence.ADVISORY,
            )
