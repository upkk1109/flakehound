"""M2: JAX PRNGKey reuse / missing `split()`.

Unlike NumPy's stateful global RNG, `jax.random.PRNGKey`/`key` is a pure value:
calling `jax.random.normal(key, ...)` does not advance any hidden state, so
passing the *same* key to a second `jax.random.*` call reproduces the exact
same draw instead of an independent one. That silently violates the
"each random call is independent" assumption most tests are written against
(two supposedly-uncorrelated samples turn out identical/correlated, masking
real bugs a genuinely independent draw would have caught) -- and it gets
worse when the shared key lives at module scope and several test functions
consume it directly, because whichever test happens to run first quietly
becomes load-bearing for the others' "randomness".

A nastier, historically-real variant of the same family is seeding via
`PRNGKey(hash(some_string))`: Python's builtin `hash()` is salted by
`PYTHONHASHSEED`, which is randomized by default, so a hash-seeded key is
reproducible *within one process* (same-process reruns pass) but silently
differs *across* processes -- exactly the "green locally, red on a different
CI run" signature this tool exists to catch, and it needs no reuse/missing
`split()` at all to be broken. That is the highest-confidence half of this
rule; flagging same-key reuse without an intervening `split()`/`fold_in()`
is the second, more heuristic half (single-function, syntactic dataflow only
-- no cross-call-site aliasing, no loop-unrolling), so specific findings from
that half are downgraded via `finding(..., confidence=...)` where the
evidence is weaker than the hash-seeded case.

Fix: split before reusing (`k1, k2 = jax.random.split(key)`, one consumption
per half) and derive a fresh key per test instead of sharing one across
functions (`jax.random.fold_in(key, <per-test discriminator>)`); never seed
from `hash(...)`.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable

from flakehound.rules._imports import build_alias_map, resolve_call
from flakehound.rules.base import Confidence, FileContext, Finding, Rule, register

_DERIVING_ATTRS = {"split", "fold_in"}
_CONSTRUCTOR_ATTRS = {"PRNGKey"}


def _is_prngkey_call(dotted: str) -> bool:
    return dotted == "PRNGKey" or dotted.endswith(".PRNGKey")


def _split_kind(dotted: str) -> str | None:
    """ "split" / "fold_in" if `dotted` looks like a JAX key-deriving call, else None."""
    prefix, sep, attr = dotted.rpartition(".")
    if attr not in _DERIVING_ATTRS:
        return None
    if not sep or prefix.endswith("random"):
        return attr
    return None


def _is_consuming_random_call(dotted: str) -> bool:
    """True for a `*.random.<op>` call that *consumes* a key (not construct/derive it)."""
    prefix, sep, attr = dotted.rpartition(".")
    if not sep or attr in _CONSTRUCTOR_ATTRS or attr in _DERIVING_ATTRS:
        return False
    return prefix.endswith("random")


def _key_arg_name(call: ast.Call) -> str | None:
    """The Name of the (by convention, first-positional) key/rng argument, if any."""
    if call.args and isinstance(call.args[0], ast.Name):
        return call.args[0].id
    for kw in call.keywords:
        if kw.arg in ("key", "rng") and isinstance(kw.value, ast.Name):
            return kw.value.id
    return None


def _contains_hash_call(expr: ast.AST) -> bool:
    return any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "hash"
        for n in ast.walk(expr)
    )


def _names_in_target(target: ast.expr) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        out: list[str] = []
        for elt in target.elts:
            out.extend(_names_in_target(elt))
        return out
    return []


def _assign_target_names(targets: Iterable[ast.expr]) -> list[str]:
    names: list[str] = []
    for t in targets:
        names.extend(_names_in_target(t))
    return names


def _iter_stmts(body: list[ast.stmt]) -> Iterable[ast.stmt]:
    """Statements in `body`, in source order, descending into control-flow blocks
    but never into a nested function/class's own body -- that gets its own scope."""
    for stmt in body:
        yield stmt
        if isinstance(stmt, (ast.If, ast.For, ast.AsyncFor, ast.While)):
            yield from _iter_stmts(stmt.body)
            yield from _iter_stmts(stmt.orelse)
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            yield from _iter_stmts(stmt.body)
        elif isinstance(stmt, ast.Try):
            yield from _iter_stmts(stmt.body)
            for handler in stmt.handlers:
                yield from _iter_stmts(handler.body)
            yield from _iter_stmts(stmt.orelse)
            yield from _iter_stmts(stmt.finalbody)


def _derives_new_key(stmt: ast.stmt, aliases: dict[str, str]) -> list[str] | None:
    """If `stmt` is `<target(s)> = PRNGKey(...)` or `<target(s)> = split/fold_in(...)`,
    return the freshly-bound target names (new, unconsumed key material)."""
    if not isinstance(stmt, ast.Assign) or not isinstance(stmt.value, ast.Call):
        return None
    dotted = resolve_call(stmt.value, aliases)
    if dotted and (_is_prngkey_call(dotted) or _split_kind(dotted) is not None):
        return _assign_target_names(stmt.targets)
    return None


def _directly_consumes(
    func: ast.FunctionDef | ast.AsyncFunctionDef, name: str, aliases: dict[str, str]
) -> bool:
    """True if `func` passes the raw variable `name` into a consuming `jax.random.*`
    call without ever deriving a fresh key from it via `split`/`fold_in` first."""
    derived = False
    used_directly = False
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        dotted = resolve_call(node, aliases)
        if not dotted:
            continue
        if _split_kind(dotted) is not None:
            if _key_arg_name(node) == name:
                derived = True
        elif _is_consuming_random_call(dotted) and _key_arg_name(node) == name:
            used_directly = True
    return used_directly and not derived


@register
class PRNGKeyReuse(Rule):
    id = "M2"
    name = "jax-prngkey-reuse"
    cause = "randomness/jax-prng-reuse"
    confidence = Confidence.HIGH
    fix_suggestion = (
        "split before each additional use -- `k1, k2 = jax.random.split(key)` -- and "
        "consume each half once; derive a fresh key per test instead of sharing one "
        "module-level key (e.g. `jax.random.fold_in(key, <per-test discriminator>)`); "
        "seed from an explicit int, never `PRNGKey(hash(...))` (salted by "
        "PYTHONHASHSEED, differs across processes)"
    )

    def check(self, ctx: FileContext) -> Iterable[Finding]:
        aliases = build_alias_map(ctx.tree)
        yield from self._check_hash_seeded(ctx, aliases)
        for func in ast.walk(ctx.tree):
            if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield from self._check_function_reuse(ctx, func, aliases)
        yield from self._check_module_scope_reuse(ctx, aliases)

    def _check_hash_seeded(self, ctx: FileContext, aliases: dict[str, str]) -> Iterable[Finding]:
        for node in ast.walk(ctx.tree):
            if not isinstance(node, ast.Call):
                continue
            dotted = resolve_call(node, aliases)
            if dotted is None or not _is_prngkey_call(dotted):
                continue
            exprs = [*node.args, *(kw.value for kw in node.keywords)]
            if any(_contains_hash_call(e) for e in exprs):
                yield self.finding(
                    ctx,
                    node,
                    f"`{dotted}(hash(...))` seeds a JAX PRNGKey from Python's salted "
                    "`hash()`; reproducible within one process but not across "
                    "processes/runs (`PYTHONHASHSEED` is randomized by default), so "
                    "reruns can silently diverge",
                )

    def _check_function_reuse(
        self,
        ctx: FileContext,
        func: ast.FunctionDef | ast.AsyncFunctionDef,
        aliases: dict[str, str],
    ) -> Iterable[Finding]:
        consumed: set[str] = set()
        for stmt in _iter_stmts(func.body):
            fresh = _derives_new_key(stmt, aliases)
            if fresh is not None:
                for name in fresh:
                    consumed.discard(name)
                continue
            for node in ast.walk(stmt):
                if not isinstance(node, ast.Call):
                    continue
                dotted = resolve_call(node, aliases)
                if not dotted or not _is_consuming_random_call(dotted):
                    continue
                name = _key_arg_name(node)
                if name is None:
                    continue
                if name in consumed:
                    yield self.finding(
                        ctx,
                        node,
                        f"`{name}` is passed to `{dotted}(...)` again with no "
                        "`jax.random.split`/`fold_in` since its last use; the same "
                        "key produces identical randomness both times",
                    )
                else:
                    consumed.add(name)

    def _check_module_scope_reuse(
        self, ctx: FileContext, aliases: dict[str, str]
    ) -> Iterable[Finding]:
        module_keys: dict[str, ast.stmt] = {}
        for stmt in ctx.tree.body:
            fresh = _derives_new_key(stmt, aliases)
            if fresh and isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
                dotted = resolve_call(stmt.value, aliases)
                if dotted and _is_prngkey_call(dotted):
                    for name in fresh:
                        module_keys[name] = stmt
        if not module_keys:
            return

        test_funcs = [
            n
            for n in ctx.tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test_")
        ]

        for name, origin in module_keys.items():
            direct_users = [f.name for f in test_funcs if _directly_consumes(f, name, aliases)]
            if len(direct_users) < 2:
                continue
            yield self.finding(
                ctx,
                origin,
                f"module-scope PRNGKey `{name}` is passed directly into jax.random "
                f"calls in {len(direct_users)} test functions "
                f"({', '.join(direct_users)}) with no per-test `split`/`fold_in`; "
                "whichever test runs first is silently load-bearing for the others",
                confidence=Confidence.MEDIUM,
            )
