"""Shared helper: resolve local import aliases to canonical dotted names.

Most rules match calls/decorators by a literal dotted prefix (`np.random.seed`,
`jax.random.normal`, `jax.jit`). That kind of matching silently misses anything
imported under a different local name: `import numpy.random as nr; nr.seed(0)`,
`from torch import manual_seed as ms; ms(0)`, `from jax import jit; @jit`. This
module builds a `{local_name: canonical_dotted_path}` map from a module's
`Import`/`ImportFrom` statements and resolves a Name/Attribute expression (a
call's `func`, a decorator, an argument) through that map, so rules can match
against the canonical dotted name regardless of how it was imported/aliased in
this particular file.

Scope/limitations (deliberate, keep this module small and pure):
- Only `Import`/`ImportFrom` bindings are tracked; a later module-scope
  reassignment or same-name `def`/`class` that shadows an import is not
  detected (rare for the RNG/JIT-call names rules care about, and shadow
  tracking is real dataflow analysis this v1 helper does not attempt).
- Relative imports (`from . import x`) have no resolvable absolute module, so
  they are skipped -- the caller falls back to the unresolved dotted name.
- `import *` cannot introduce a known alias (the bound names are unknown
  statically), so `ImportFrom` with `alias.name == "*"` is skipped.
"""

from __future__ import annotations

import ast


def dotted_name(expr: ast.AST) -> str | None:
    """Dotted string for a plain `Name`/`Attribute` chain, e.g. `a.b.c` -> "a.b.c".

    Returns None if `expr` is not such a chain (e.g. it's a call result, a
    subscript, a literal) -- those have nothing for alias resolution to expand.
    """
    parts: list[str] = []
    while isinstance(expr, ast.Attribute):
        parts.append(expr.attr)
        expr = expr.value
    if isinstance(expr, ast.Name):
        parts.append(expr.id)
        return ".".join(reversed(parts))
    return None


def build_alias_map(tree: ast.Module) -> dict[str, str]:
    """Map every name this module's imports bind to its canonical dotted path.

    Examples::

        import numpy.random as nr          -> {"nr": "numpy.random"}
        import numpy.random                -> {"numpy": "numpy"}  (identity; the
                                               binding is the top package, already
                                               what a literal `numpy.random.seed`
                                               dotted-chain would use unresolved)
        from torch import manual_seed as ms -> {"ms": "torch.manual_seed"}
        from jax import random as jr        -> {"jr": "jax.random"}
        from jax import jit                 -> {"jit": "jax.jit"}

    Imports anywhere in the module (including nested inside a function/fixture)
    are included -- rules that walk the whole tree for calls should see aliases
    bound anywhere the call could plausibly reach.
    """
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    aliases[alias.asname] = alias.name
                else:
                    top = alias.name.split(".", 1)[0]
                    aliases[top] = top
        elif isinstance(node, ast.ImportFrom):
            if node.level or node.module is None:
                continue  # relative import: no resolvable absolute module
            for alias in node.names:
                if alias.name == "*":
                    continue
                bound = alias.asname or alias.name
                aliases[bound] = f"{node.module}.{alias.name}"
    return aliases


def resolve_dotted(dotted: str, aliases: dict[str, str]) -> str:
    """Expand a leading alias in a dotted string to its canonical form.

    `"nr.seed"` with `aliases={"nr": "numpy.random"}` -> `"numpy.random.seed"`.
    If the leading component is not a known alias, `dotted` is returned as-is.
    """
    head, sep, rest = dotted.partition(".")
    canonical_head = aliases.get(head)
    if canonical_head is None:
        return dotted
    return f"{canonical_head}{sep}{rest}" if rest else canonical_head


def resolve_expr(expr: ast.AST, aliases: dict[str, str]) -> str | None:
    """Canonical dotted string for any Name/Attribute expression, alias-resolved.

    None if `expr` is not a plain dotted-name chain (see `dotted_name`).
    """
    dotted = dotted_name(expr)
    if dotted is None:
        return None
    return resolve_dotted(dotted, aliases)


def resolve_call(node: ast.Call, aliases: dict[str, str]) -> str | None:
    """Canonical dotted string for a Call's `func`, alias-resolved.

    None if `node.func` is not a plain dotted-name chain (e.g. a call result
    like `get_rng().seed()`), same as the pre-alias-resolution `_dotted` helper
    rules used to define locally.
    """
    return resolve_expr(node.func, aliases)
