"""M3: suspicious tolerance on model-output comparisons.

`assert_allclose`/`np.isclose`/`pytest.approx` calls whose `atol`/`rtol` (or
`abs`/`rel` for `approx`) is <= 1e-7 assume near machine-precision agreement.
That bound is legitimate for deterministic, float64 linear algebra, but
stochastic outputs (dropout, sampling, non-deterministic GPU reduction order)
and float32 model forward passes routinely disagree by more than that
run-to-run — a tolerance this tight on a model/predict/forward/apply/loss/
grad-derived value fails intermittently for reasons unrelated to correctness.
The fix is not "loosen it blindly": derive the bound from an observed
distribution of repeated runs (FLEX FSE-21) rather than copy-pasting a tight
default.

Static analysis cannot see whether the compared values are actually
stochastic — only that the calls that produced them *look* model-related —
so every finding here is advisory, never confirmed. (The rule's confidence
is already the floor of the `Confidence` enum; there is no lower tier to
downgrade individual findings to.)
"""

from __future__ import annotations

import ast
from collections.abc import Iterable

from flakehound.rules.base import Confidence, FileContext, Finding, Rule, register

_TIGHT_THRESHOLD = 1e-7

# Snake_case/dotted tokens that heuristically mark a call as producing a
# model output. Matched as whole tokens (split on "." and "_"), not raw
# substrings, so e.g. `np.gradient` ("gradient") and `"lossless"` don't
# collide with `grad`/`loss`. `apply` still collides with pandas'/numpy's
# generic `.apply()` — accepted, per spec, because this is advisory-only and
# gated on an explicit tight atol/rtol/abs/rel besides.
_MODEL_MARKERS = {"model", "predict", "forward", "apply", "loss", "grad"}

_FLOAT64_TOKENS = {"float64", "double"}
_EXACT_INT_TOKENS = {"int", "int8", "int16", "int32", "int64", "intc", "intp", "integer"}


def _dotted(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _tokens(dotted: str) -> set[str]:
    return {t for t in dotted.lower().replace(".", "_").split("_") if t}


def _is_model_ish(dotted: str) -> bool:
    return not _tokens(dotted).isdisjoint(_MODEL_MARKERS)


def _model_evidence(node: ast.expr | None) -> ast.Call | None:
    """First call inside `node` whose dotted name looks model-ish, if any."""
    if node is None:
        return None
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        dotted = _dotted(sub.func)
        if dotted and _is_model_ish(dotted):
            return sub
    return None


def _mentions_dtype_token(node: ast.expr | None, tokens: set[str]) -> bool:
    if node is None:
        return False
    for sub in ast.walk(node):
        if isinstance(sub, ast.Attribute) and sub.attr.lower() in tokens:
            return True
        if isinstance(sub, ast.Name) and sub.id.lower() in tokens:
            return True
        if (
            isinstance(sub, ast.Constant)
            and isinstance(sub.value, str)
            and sub.value.lower() in tokens
        ):
            return True
    return False


def _fp_guard_reason(operands: tuple[ast.expr | None, ast.expr | None]) -> str | None:
    """Why a model-ish, tight-tolerance match is still a known-safe shape."""
    for operand in operands:
        if _mentions_dtype_token(operand, _FLOAT64_TOKENS):
            return "an explicit float64 cast/dtype"
        if _mentions_dtype_token(operand, _EXACT_INT_TOKENS):
            return "an explicit exact-integer cast/dtype"
    return None


def _numeric_const(node: ast.AST) -> float | None:
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ):
        return float(node.value)
    return None


def _arg(call: ast.Call, index: int, name: str) -> ast.expr | None:
    """Positional arg at `index`, falling back to keyword `name`."""
    if index < len(call.args):
        return call.args[index]
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _tight_tolerance(
    call: ast.Call, names: tuple[str, str]
) -> tuple[str, float, ast.keyword] | None:
    """First keyword in `names` set to a strictly-positive value <= 1e-7.

    `0` is excluded on purpose: `atol=0, rtol=1e-5` (or the reverse) is the
    idiomatic way to say "use only the other bound," not "make this one
    suspiciously tight."
    """
    for kw in call.keywords:
        if kw.arg not in names:
            continue
        value = _numeric_const(kw.value)
        if value is not None and 0 < value <= _TIGHT_THRESHOLD:
            return kw.arg, value, kw
    return None


@register
class SuspiciousTolerance(Rule):
    id = "M3"
    name = "suspicious-tolerance"
    cause = "ml-numerics/tolerance"
    confidence = Confidence.ADVISORY
    fix_suggestion = (
        "justify this bound or derive it from an observed distribution of repeated "
        "runs (FLEX FSE-21) instead of copy-pasting a tight default; float32 "
        "model/GPU outputs commonly need rtol=1e-5/atol=1e-6 floors"
    )

    def check(self, ctx: FileContext) -> Iterable[Finding]:
        for node in ast.walk(ctx.tree):
            if isinstance(node, ast.Call):
                yield from self._check_direct_call(ctx, node)
            elif isinstance(node, ast.Compare):
                yield from self._check_approx_compare(ctx, node)

    def _check_direct_call(self, ctx: FileContext, call: ast.Call) -> Iterable[Finding]:
        dotted = _dotted(call.func)
        if dotted is None:
            return
        last = dotted.rpartition(".")[-1]
        root = dotted.partition(".")[0]
        if last == "assert_allclose":
            operands = (_arg(call, 0, "actual"), _arg(call, 1, "desired"))
        elif last == "isclose" and root in {"np", "numpy"}:
            operands = (_arg(call, 0, "a"), _arg(call, 1, "b"))
        else:
            return
        yield from self._maybe_finding(ctx, call, operands, ("atol", "rtol"))

    def _check_approx_compare(self, ctx: FileContext, cmp: ast.Compare) -> Iterable[Finding]:
        if len(cmp.ops) != 1 or not isinstance(cmp.ops[0], ast.Eq):
            return
        approx_call: ast.Call | None = None
        other_side: ast.expr | None = None
        for side in (cmp.left, cmp.comparators[0]):
            if isinstance(side, ast.Call):
                dotted = _dotted(side.func)
                if dotted and dotted.rpartition(".")[-1] == "approx":
                    approx_call = side
                    continue
            other_side = side
        if approx_call is None:
            return
        expected_arg = approx_call.args[0] if approx_call.args else None
        operands = (other_side, expected_arg)
        yield from self._maybe_finding(ctx, approx_call, operands, ("abs", "rel"))

    def _maybe_finding(
        self,
        ctx: FileContext,
        call: ast.Call,
        operands: tuple[ast.expr | None, ast.expr | None],
        tol_names: tuple[str, str],
    ) -> Iterable[Finding]:
        tol = _tight_tolerance(call, tol_names)
        if tol is None:
            return
        kwname, value, _kwnode = tol
        model_call = _model_evidence(operands[0]) or _model_evidence(operands[1])
        if model_call is None:
            return
        if _fp_guard_reason(operands) is not None:
            return
        model_dotted = _dotted(model_call.func) or "?"
        yield self.finding(
            ctx,
            call,
            f"`{kwname}={value!r}` on a value derived from `{model_dotted}(...)` is "
            "machine-precision tight; stochastic/float32 model outputs routinely "
            "disagree by more than this run-to-run",
        )
