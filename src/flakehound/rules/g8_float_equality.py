"""G8: float equality without tolerance.

`assert a == b` (or `assertEqual`) where an operand is a float literal with a
fractional part, a true-division result, or a `math`/`np` call that produces a
float is comparing IEEE-754 values for exact equality -- the classic source of
"passes on my machine, fails on CI" flakiness once either side has gone
through any arithmetic (accumulated rounding, differing libm/BLAS builds,
platform FMA differences). Fix: `math.isclose`/`pytest.approx`/`np.isclose`.

Local dataflow (function-local type inference lite): the dominant real-world
shape is `x = compute(); y = other(); assert x == y`, where neither operand is
itself a literal/division/call at the compare site. To catch this without a
general-purpose dataflow pass, each function/method scope is scanned once,
top-to-bottom, tracking which local names currently hold a float-producing
value (a fractional literal, a division result, a float-producing math/np
call, or another already-tracked float name). Any other assignment to a name
-- including reassignment to a non-float expression, tuple unpacking, or a
for/with/except target -- clears it. This is deliberately shallow: single-
target `name = expr` only, no branch-sensitivity, no cross-function tracking
(mirrors `_simple_assigns` in `m1_unseeded_stochastic_assert.py`). A finding
sourced *only* from this inference (neither side has direct literal/division/
call evidence) is reported at MEDIUM, not HIGH -- static tracking can't be as
certain as seeing the literal/call at the compare site itself (tier honesty).

Constant-fold guard: `assert 0.1 + 0.2 == 0.3` is fully static -- both sides
are constant expressions (literals and +/-/*//%/** on them), so the result
never varies between runs. It is still flagged (the representation bug is
real: `0.1 + 0.2 != 0.3` in IEEE-754), but at MEDIUM with a message noting the
expression is constant, since the "flaky across runs/platforms" framing that
justifies HIGH doesn't apply to a value that is the same every time.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable

from flakehound.rules.base import Confidence, FileContext, Finding, Rule, register

_SENTINELS = {0.0, 1.0, -1.0, 0.5}

_MATH_NON_FLOAT = {
    "floor",
    "ceil",
    "trunc",
    "gcd",
    "lcm",
    "isnan",
    "isclose",
    "isfinite",
    "isinf",
    "isqrt",
    "comb",
    "factorial",
    "perm",
    "frexp",
    "modf",
}

_NP_FLOAT_CALLS = {
    "mean",
    "std",
    "var",
    "median",
    "average",
    "sum",
    "dot",
    "sqrt",
    "log",
    "log2",
    "log10",
    "exp",
    "norm",
    "ptp",
    "trace",
    "corrcoef",
}

_STATIC_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)

# Nested scopes: a name assigned here does not leak into (or read from) the
# enclosing function's tracked set, so the scan does not descend past them.
_SCOPE_BOUNDARY = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.Lambda,
    ast.ClassDef,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)


def _dotted(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _literal_float(node: ast.AST) -> float | None:
    """Return the value of a float literal, unwrapping a leading unary +/-."""
    if isinstance(node, ast.Constant) and isinstance(node.value, float):
        return node.value
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub | ast.UAdd)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, float)
    ):
        value = node.operand.value
        return -value if isinstance(node.op, ast.USub) else value
    return None


def _is_float_producing_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    dotted = _dotted(node.func)
    if dotted is None or "." not in dotted:
        return False
    root, _, _ = dotted.partition(".")
    _, _, attr = dotted.rpartition(".")
    if root == "math":
        return attr not in _MATH_NON_FLOAT
    if root in {"np", "numpy"}:
        return attr in _NP_FLOAT_CALLS
    return False


def _float_evidence(node: ast.AST) -> str | None:
    """Why this operand is suspected of being an imprecise float, if at all.

    Direct evidence only -- a literal/division/call visible at this node.
    Does not consult local variable tracking (see `_local_float_names`).
    """
    literal = _literal_float(node)
    if literal is not None and not literal.is_integer():
        return "a float literal with a fractional part"
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return "a division result"
    if _is_float_producing_call(node):
        return "a float-producing math/np call"
    return None


def _is_approx_wrapped(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    dotted = _dotted(node.func)
    return dotted is not None and (dotted == "approx" or dotted.endswith(".approx"))


def _is_sentinel_identical_literal(left: ast.AST, right: ast.AST) -> bool:
    lv, rv = _literal_float(left), _literal_float(right)
    return lv is not None and rv is not None and lv == rv and lv in _SENTINELS


def _is_assert_equal_call(node: ast.Call) -> bool:
    return isinstance(node.func, ast.Attribute) and node.func.attr == "assertEqual"


def _pair_from(node: ast.AST) -> tuple[ast.AST, ast.AST] | None:
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq):
        return node.left, node.comparators[0]
    if isinstance(node, ast.Call) and _is_assert_equal_call(node) and len(node.args) >= 2:
        return node.args[0], node.args[1]
    return None


def _is_fully_static(node: ast.AST) -> bool:
    """True if `node` is built entirely from numeric constants and +/-/*//%/**
    on them (e.g. `0.1 + 0.2`, `-3.0`) -- a value that never varies between
    runs or platforms. `bool` is excluded even though it subclasses `int`."""
    if isinstance(node, ast.Constant):
        return isinstance(node.value, int | float) and not isinstance(node.value, bool)
    if isinstance(node, ast.UnaryOp):
        return isinstance(node.op, ast.USub | ast.UAdd) and _is_fully_static(node.operand)
    if isinstance(node, ast.BinOp):
        return (
            isinstance(node.op, _STATIC_BINOPS)
            and _is_fully_static(node.left)
            and _is_fully_static(node.right)
        )
    return False


def _bound_names(target: ast.expr) -> Iterable[str]:
    """All local names an assignment target binds, including tuple/list
    unpacking and starred targets. `Attribute`/`Subscript` targets (`obj.x =
    ...`, `d[k] = ...`) don't bind a local name and are ignored."""
    if isinstance(target, ast.Name):
        yield target.id
    elif isinstance(target, ast.Starred):
        yield from _bound_names(target.value)
    elif isinstance(target, ast.Tuple | ast.List):
        for elt in target.elts:
            yield from _bound_names(elt)


def _rhs_float_evidence(node: ast.expr, tracked: set[str]) -> bool:
    """Whether an assignment's RHS should mark its target as a tracked float:
    direct literal/division/call evidence, or propagation from another name
    already tracked as float (`y = x`)."""
    if _float_evidence(node) is not None:
        return True
    return isinstance(node, ast.Name) and node.id in tracked


def _iter_funcs(node: ast.AST) -> Iterable[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Top-level and class-method function defs (mirrors the pattern used by
    M1's `_iter_funcs`): does not descend into a def's own body looking for
    further nested defs -- those get their own, independently-tracked scope
    when `_local_float_names` walks them (or no tracking at all, which is the
    safe fallback, if they're never visited as a top-level/method def)."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
            yield child
        elif isinstance(child, ast.ClassDef):
            yield from _iter_funcs(child)


def _own_scope_nodes(func: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.expr | ast.stmt]:
    """All descendant nodes in `func`'s own local scope, unordered. Stops at
    nested function/class defs and comprehensions -- those introduce their own
    scope and must not leak bindings into (or read bindings from) this one."""
    out: list[ast.expr | ast.stmt] = []
    stack = list(ast.iter_child_nodes(func))
    while stack:
        node = stack.pop()
        if isinstance(node, ast.expr | ast.stmt):
            out.append(node)
        if isinstance(node, _SCOPE_BOUNDARY):
            continue
        stack.extend(ast.iter_child_nodes(node))
    return out


def _local_float_names(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[ast.AST, frozenset[str]]:
    """For each `==`/`assertEqual` site in `func`'s own scope, the set of local
    names known -- as of that point in a simple top-to-bottom source-order
    scan -- to hold a float-producing expression. See the module docstring
    for what counts as evidence and what clears it."""
    nodes = sorted(_own_scope_nodes(func), key=lambda n: (n.lineno, n.col_offset))
    tracked: set[str] = set()
    result: dict[ast.AST, frozenset[str]] = {}
    for node in nodes:
        if isinstance(node, ast.Assign):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if _rhs_float_evidence(node.value, tracked):
                    tracked.add(name)
                else:
                    tracked.discard(name)
            else:
                for t in node.targets:
                    for name in _bound_names(t):
                        tracked.discard(name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
            if node.value is not None and _rhs_float_evidence(node.value, tracked):
                tracked.add(name)
            else:
                tracked.discard(name)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
            if isinstance(node.op, ast.Div):
                tracked.add(name)
            else:
                tracked.discard(name)
        elif isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
            name = node.target.id
            if _rhs_float_evidence(node.value, tracked):
                tracked.add(name)
            else:
                tracked.discard(name)
        elif isinstance(node, ast.For | ast.AsyncFor):
            for name in _bound_names(node.target):
                tracked.discard(name)
        elif isinstance(node, ast.With | ast.AsyncWith):
            for item in node.items:
                if item.optional_vars is not None:
                    for name in _bound_names(item.optional_vars):
                        tracked.discard(name)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            tracked.discard(node.name)
        elif _pair_from(node) is not None:
            result[node] = frozenset(tracked)
    return result


@register
class FloatEqualityWithoutTolerance(Rule):
    id = "G8"
    name = "float-equality-without-tolerance"
    cause = "floating-point/precision"
    confidence = Confidence.HIGH
    fix_suggestion = (
        "compare with tolerance: `math.isclose(a, b, rel_tol=...)`, "
        "`assert a == pytest.approx(b)`, or `np.isclose(a, b)`"
    )

    def check(self, ctx: FileContext) -> Iterable[Finding]:
        local_tracked: dict[ast.AST, frozenset[str]] = {}
        for func in _iter_funcs(ctx.tree):
            local_tracked.update(_local_float_names(func))
        for node in ast.walk(ctx.tree):
            pair = _pair_from(node)
            if pair is None:
                continue
            left, right = pair
            tracked = local_tracked.get(node, frozenset())
            yield from self._check_pair(ctx, node, left, right, tracked)

    def _check_pair(
        self,
        ctx: FileContext,
        node: ast.AST,
        left: ast.AST,
        right: ast.AST,
        tracked: frozenset[str],
    ) -> Iterable[Finding]:
        if _is_approx_wrapped(left) or _is_approx_wrapped(right):
            return
        evidence = _float_evidence(left) or _float_evidence(right)
        inferred = False
        if evidence is None:
            tracked_name: str | None = None
            if isinstance(left, ast.Name) and left.id in tracked:
                tracked_name = left.id
            elif isinstance(right, ast.Name) and right.id in tracked:
                tracked_name = right.id
            if tracked_name is not None:
                evidence = (
                    f"a local variable (`{tracked_name}`) last assigned a "
                    "float-producing expression earlier in this function"
                )
                inferred = True
        if evidence is None:
            return
        if _is_sentinel_identical_literal(left, right):
            return
        confidence = self.confidence
        message = (
            f"`==` compares floats where one side is {evidence}; "
            "IEEE-754 rounding makes this brittle without a tolerance"
        )
        if _is_fully_static(left) and _is_fully_static(right):
            confidence = Confidence.MEDIUM
            message += (
                "; both sides are constant expressions (deterministic across runs, but "
                "still a genuine float-representation bug worth fixing)"
            )
        elif inferred:
            confidence = Confidence.MEDIUM
        yield self.finding(ctx, node, message, confidence=confidence)
