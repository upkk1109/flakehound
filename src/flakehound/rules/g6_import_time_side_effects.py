"""G6: import-time side effects in test modules.

Module-level statements (top-level, not under `if __name__ == "__main__":`)
that perform I/O, network, subprocess, or process-global mutation run once at
*collection* time, in whichever order pytest/xdist happens to import the file
— coupling side-effect timing (and, for held object references, shared
mutable state) to test-collection order rather than test-execution order.
Constructing heavy stateful objects at module scope instead of inside a
fixture is a commonly-fixed order-dependence contaminator in real-world
suites; the mechanical fix is always the same: wrap the statement in a
fixture, or gate it behind `if __name__ == "__main__":`.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable

from flakehound.rules.base import Confidence, FileContext, Finding, Rule, register

_IO_MODULE_PREFIXES = ("requests.", "socket.", "subprocess.")
_ENV_MUTATING_METHODS = {"setdefault", "update", "pop", "clear"}
_PATH_CALLEES = {"Path", "pathlib.Path"}


def _dotted(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _is_path_call(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Call) and _dotted(node.func) in _PATH_CALLEES


def _is_os_environ_subscript(node: ast.AST) -> bool:
    return isinstance(node, ast.Subscript) and _dotted(node.value) == "os.environ"


def _classify_call(call: ast.Call) -> str | None:
    """Return a human-readable reason if `call` is an I/O/mutation call, else None."""
    func = call.func

    if isinstance(func, ast.Name) and func.id == "open":
        return "`open(...)` performs file I/O at import time"

    if isinstance(func, ast.Attribute):
        if func.attr.startswith("write_") and _is_path_call(func.value):
            return f"`Path(...).{func.attr}(...)` writes to disk at import time"
        if func.attr == "connect":
            base = _dotted(func.value) or "<expr>"
            return f"`{base}.connect(...)` opens a network/database connection at import time"

    dotted = _dotted(func)
    if dotted is None or dotted == "pytest.importorskip":
        return None

    if any(dotted.startswith(p) for p in _IO_MODULE_PREFIXES):
        mod = dotted.split(".", 1)[0]
        return f"`{dotted}(...)` performs {mod} I/O at import time"

    prefix, _, attr = dotted.rpartition(".")
    if attr == "seed" and prefix.endswith("random"):
        return f"`{dotted}(...)` mutates global RNG state at import time"

    if dotted.startswith("os.environ.") and attr in _ENV_MUTATING_METHODS:
        return f"`{dotted}(...)` mutates process-global environment state at import time"

    return None


@register
class ImportTimeSideEffects(Rule):
    id = "G6"
    name = "import-time-side-effects"
    cause = "import-time-side-effects/order-dependence"
    confidence = Confidence.MEDIUM
    fix_suggestion = (
        "move I/O, network, subprocess, or global-state mutation into a fixture "
        "(function- or module-scoped as appropriate) or behind "
        '`if __name__ == "__main__":`; module-level statements run once at '
        "collection time, in whatever order pytest/xdist imports the file"
    )

    def check(self, ctx: FileContext) -> Iterable[Finding]:
        if ctx.is_conftest or not ctx.is_test_file:
            return
        for stmt in ctx.tree.body:
            yield from self._check_stmt(ctx, stmt)

    def _check_stmt(self, ctx: FileContext, stmt: ast.stmt) -> Iterable[Finding]:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            reason = _classify_call(stmt.value)
            if reason:
                yield self.finding(ctx, stmt, reason)
            return

        if not isinstance(stmt, ast.Assign):
            return

        for target in stmt.targets:
            if _is_os_environ_subscript(target):
                yield self.finding(
                    ctx,
                    stmt,
                    "`os.environ[...] = ...` mutates process-global environment "
                    "state at import time",
                )
                return

        if not isinstance(stmt.value, ast.Call):
            return

        reason = _classify_call(stmt.value)
        if reason:
            yield self.finding(ctx, stmt, reason)
            return

        # Weaker heuristic: `NAME = SomeClient(...)` held as a module-level
        # reference. Naming-convention evidence only (no dataflow), so this
        # tier is downgraded to advisory rather than the rule's default.
        if len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            callee = _dotted(stmt.value.func)
            looks_like_client = bool(callee) and callee.rsplit(".", 1)[-1].endswith("Client")
            if callee and callee not in _PATH_CALLEES and looks_like_client:
                name = stmt.targets[0].id
                yield self.finding(
                    ctx,
                    stmt,
                    f"`{name} = {callee}(...)` constructs a stateful client at import "
                    "time and holds a module-level reference; tests sharing this name "
                    "can observe state left behind by whichever test ran first",
                    confidence=Confidence.ADVISORY,
                )
