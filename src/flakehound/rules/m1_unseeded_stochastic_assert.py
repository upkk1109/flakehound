"""M1: tight-tolerance assertion on stochastic output with no seed in scope.

`np.testing.assert_allclose`/`assert_array_almost_equal`/`pytest.approx`
compare two values *up to a tolerance*, not exactly -- appropriate when the
values come from floating-point arithmetic, but not a substitute for
reproducibility when one of the compared values is the output of a
stochastic op (`np.random.normal`, `torch.dropout`, `df.sample`, a freshly
initialized model's `.fit`/`.predict`, ...). Without a deterministic
generator seeded in the test (or one of the fixtures it depends on), the
compared value is drawn fresh every run and the assertion's tolerance is
gambling against how far two independent draws can land apart -- it will
pass on most runs and fail on an unlucky one, exactly the "green locally,
red on CI" signature this tool exists to catch. Fix: derive a per-test seed
(`np.random.default_rng(seed)`, `torch.Generator().manual_seed(seed)`,
`jax.random.PRNGKey(seed)`) local to the test or its fixtures.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from typing import TypeGuard

from flakehound.rules.base import Confidence, FileContext, Finding, Rule, register

_TOLERANT_CALL_NAMES = {"assert_allclose", "assert_array_almost_equal"}
_SEED_CALL_NAMES = {"default_rng", "Generator", "PRNGKey", "manual_seed", "seed_everything"}
_STOCHASTIC_SUBSTRINGS = ("random", "normal", "uniform", "sample", "dropout")
# `.fit`/`.predict` are only *weak* stochastic evidence: the model may or may
# not have been randomly initialized, and static analysis can't tell -- see
# `_stochastic_evidence`, which downgrades findings sourced from these.
_MODEL_CALL_ATTRS = {"fit", "predict"}


def _dotted(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _iter_funcs(node: ast.AST) -> Iterable[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Top-level and class-method function defs (does not descend into a def's own body)."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
            yield child
        elif isinstance(child, ast.ClassDef):
            yield from _iter_funcs(child)


def _is_fixture(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for dec in func.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        name = (
            target.attr
            if isinstance(target, ast.Attribute)
            else (target.id if isinstance(target, ast.Name) else None)
        )
        if name == "fixture":
            return True
    return False


def _param_names(func: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    a = func.args
    return [p.arg for p in (*a.posonlyargs, *a.args, *a.kwonlyargs)]


def _fixture_deps(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    fixtures: dict[str, ast.FunctionDef | ast.AsyncFunctionDef],
    seen: set[str] | None = None,
) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Fixtures reachable from `func`'s parameters, transitively through fixture params."""
    seen = seen if seen is not None else set()
    out: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for name in _param_names(func):
        if name in seen:
            continue
        fx = fixtures.get(name)
        if fx is None:
            continue
        seen.add(name)
        out.append(fx)
        out.extend(_fixture_deps(fx, fixtures, seen))
    return out


def _has_seed_call(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        dotted = _dotted(node.func)
        if dotted is not None and dotted.rsplit(".", 1)[-1] in _SEED_CALL_NAMES:
            return True
    return False


def _is_approx_call(node: ast.AST) -> TypeGuard[ast.Call]:
    if not isinstance(node, ast.Call):
        return False
    dotted = _dotted(node.func)
    return dotted is not None and dotted.rsplit(".", 1)[-1] == "approx"


def _iter_tolerant_comparisons(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
) -> Iterable[tuple[ast.AST, list[ast.expr]]]:
    """Yield (site, compared-exprs) for each assert_allclose/-array_almost_equal call
    or `== pytest.approx(...)` comparison found in `func`'s body."""
    for node in ast.walk(func):
        if isinstance(node, ast.Call):
            dotted = _dotted(node.func)
            if dotted is None or dotted.rsplit(".", 1)[-1] not in _TOLERANT_CALL_NAMES:
                continue
            exprs = list(node.args[:2])
            exprs.extend(kw.value for kw in node.keywords if kw.arg in ("actual", "desired"))
            yield node, exprs
        elif (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and isinstance(node.ops[0], ast.Eq | ast.NotEq)
        ):
            sides = [node.left, node.comparators[0]]
            if not any(_is_approx_call(s) for s in sides):
                continue
            exprs: list[ast.expr] = []
            for s in sides:
                if _is_approx_call(s):
                    exprs.extend(s.args)
                    exprs.extend(kw.value for kw in s.keywords if kw.arg == "expected")
                else:
                    exprs.append(s)
            yield node, exprs


def _simple_assigns(func: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, ast.expr]:
    """Map `name -> last-assigned RHS` for single-target `name = <expr>` assignments
    in `func`. A cheap, intentionally shallow stand-in for real dataflow: it lets the
    rule follow the extremely common `x = np.random.normal(...); assert_allclose(x, ...)`
    shape without a general-purpose dataflow pass."""
    out: dict[str, ast.expr] = {}
    for node in ast.walk(func):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            out[node.targets[0].id] = node.value
    return out


def _stochastic_evidence(
    exprs: list[ast.expr], assigns: dict[str, ast.expr], _depth: int = 0
) -> tuple[ast.Call, bool] | None:
    """Search the compared expressions -- and, up to a few hops, the local
    variables they reference via `assigns` -- for a stochastic-looking call.

    Returns (call, is_direct): `is_direct` is True for a call whose dotted name
    names a random op (strong evidence), False for a `.fit`/`.predict` call
    (weak evidence -- can't statically confirm a random init preceded it).
    """
    if _depth > 3:
        return None
    weak: ast.Call | None = None
    for expr in exprs:
        for node in ast.walk(expr):
            if isinstance(node, ast.Call):
                dotted = _dotted(node.func)
                if dotted is None:
                    continue
                if any(s in dotted.lower() for s in _STOCHASTIC_SUBSTRINGS):
                    return node, True
                if weak is None and dotted.rsplit(".", 1)[-1] in _MODEL_CALL_ATTRS:
                    weak = node
            elif isinstance(node, ast.Name) and node.id in assigns:
                nested = _stochastic_evidence([assigns[node.id]], assigns, _depth + 1)
                if nested is None:
                    continue
                call, is_direct = nested
                if is_direct:
                    return call, True
                weak = weak or call
    if weak is not None:
        return weak, False
    return None


@register
class UnseededStochasticAssert(Rule):
    id = "M1"
    name = "unseeded-stochastic-assert"
    cause = "randomness/reproducibility"
    confidence = Confidence.MEDIUM
    fix_suggestion = (
        "seed a local generator before drawing the compared value -- "
        "`rng = np.random.default_rng(seed)`, `torch.Generator().manual_seed(seed)`, "
        "or `jax.random.PRNGKey(seed)` -- in the test or the fixture that produces it"
    )

    def check(self, ctx: FileContext) -> Iterable[Finding]:
        funcs = list(_iter_funcs(ctx.tree))
        fixtures = {f.name: f for f in funcs if _is_fixture(f)}
        for func in funcs:
            if not func.name.startswith("test_"):
                continue
            yield from self._check_test(ctx, func, fixtures)

    def _check_test(
        self,
        ctx: FileContext,
        func: ast.FunctionDef | ast.AsyncFunctionDef,
        fixtures: dict[str, ast.FunctionDef | ast.AsyncFunctionDef],
    ) -> Iterable[Finding]:
        scope = [func, *_fixture_deps(func, fixtures)]
        if any(_has_seed_call(f) for f in scope):
            return
        assigns = _simple_assigns(func)
        for site, exprs in _iter_tolerant_comparisons(func):
            evidence = _stochastic_evidence(exprs, assigns)
            if evidence is None:
                continue
            call, is_direct = evidence
            dotted = _dotted(call.func) or "?"
            if is_direct:
                yield self.finding(
                    ctx,
                    site,
                    f"tolerant comparison against `{dotted}(...)` (a stochastic op) with "
                    "no deterministic seed established in this test or its fixtures; the "
                    "compared value is drawn fresh every run",
                )
            else:
                yield self.finding(
                    ctx,
                    site,
                    f"tolerant comparison against `{dotted}(...)` with no deterministic "
                    "seed established in this test or its fixtures; if the model was "
                    "randomly initialized this result is run-dependent (static analysis "
                    "can't confirm random init preceded this call -- verify manually)",
                    confidence=Confidence.ADVISORY,
                )
