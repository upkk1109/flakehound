"""G1: global RNG seed mutation in tests.

`random.seed(...)` / `np.random.seed(...)` mutate process-global RNG state, so a
test's outcome starts depending on which tests ran before it — the #1 measured
cause of order-dependent flakiness in Python suites.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable

from flakehound.rules._imports import build_alias_map, resolve_call
from flakehound.rules.base import Confidence, FileContext, Finding, Rule, register

_SEED_CALLS = {
    ("random", "seed"),
    ("np.random", "seed"),
    ("numpy.random", "seed"),
    ("torch", "manual_seed"),
    ("torch.cuda", "manual_seed_all"),
    ("tf.random", "set_seed"),
}

# Known RNG module prefixes for the generic "<module>.seed(...)" fallback below.
# Deliberately an exact-match allowlist, not a `str.endswith("random")` check:
# the substring check false-positived on unrelated locals like `my_random.seed(1)`
# (review low #42).
_RNG_MODULE_PREFIXES = {"random", "np.random", "numpy.random", "jax.random", "cupy.random"}


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
        aliases = build_alias_map(ctx.tree)
        for node in ast.walk(ctx.tree):
            if not isinstance(node, ast.Call):
                continue
            dotted = resolve_call(node, aliases)
            if dotted is None or "." not in dotted:
                continue
            prefix, _, attr = dotted.rpartition(".")
            if (prefix, attr) in _SEED_CALLS or (attr == "seed" and prefix in _RNG_MODULE_PREFIXES):
                yield self.finding(
                    ctx,
                    node,
                    f"`{dotted}(...)` mutates global RNG state; test outcomes now "
                    "depend on execution order",
                )
