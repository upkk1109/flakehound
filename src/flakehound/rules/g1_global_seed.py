"""G1: global RNG seed mutation in tests.

`random.seed(...)` / `np.random.seed(...)` mutate process-global RNG state, so a
test's outcome starts depending on which tests ran before it — the #1 measured
cause of order-dependent flakiness in Python suites.
"""

from __future__ import annotations

import ast
from typing import Iterable

from flakehound.rules.base import Confidence, FileContext, Finding, Rule, register

_SEED_CALLS = {
    ("random", "seed"),
    ("np.random", "seed"),
    ("numpy.random", "seed"),
    ("torch", "manual_seed"),
    ("torch.cuda", "manual_seed_all"),
    ("tf.random", "set_seed"),
}


def _dotted(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


@register
class GlobalSeedMutation(Rule):
    id = "G1"
    name = "global-seed-mutation"
    cause = "randomness/order-dependence"
    confidence = Confidence.HIGH
    fix_suggestion = (
        "use a local generator: `rng = np.random.default_rng(seed)` (or "
        "`torch.Generator().manual_seed(seed)`), or isolate with an autouse "
        "save/restore fixture; pytest-randomly resets stdlib/numpy seeds per test"
    )

    def check(self, ctx: FileContext) -> Iterable[Finding]:
        for node in ast.walk(ctx.tree):
            if not isinstance(node, ast.Call):
                continue
            dotted = _dotted(node.func)
            if dotted is None or "." not in dotted:
                continue
            prefix, _, attr = dotted.rpartition(".")
            if (prefix, attr) in _SEED_CALLS or (
                attr == "seed" and prefix.endswith("random")
            ):
                yield self.finding(
                    ctx,
                    node,
                    f"`{dotted}(...)` mutates global RNG state; test outcomes now "
                    "depend on execution order",
                )
