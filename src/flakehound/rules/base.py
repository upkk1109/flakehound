"""Rule framework: every detection rule is a small AST visitor registered here.

Design contract (binding for contributors and agents):
- One rule = one module in ``flakehound/rules/`` decorated with ``@register``.
- Rules are pure: ``check()`` receives a parsed tree + source and returns Findings.
  No I/O, no network, no global state.
- Confidence tiers are honest: HIGH means "safe to fail a pre-commit on",
  ADVISORY means "worth a look, never blocks". A rule that cannot statically
  prove its case must not claim HIGH.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    ADVISORY = "advisory"


@dataclass(frozen=True)
class Finding:
    rule_id: str
    rule_name: str
    cause: str
    confidence: Confidence
    message: str
    fix_suggestion: str
    path: str
    line: int
    col: int = 0

    def format_text(self) -> str:
        return (
            f"{self.path}:{self.line}:{self.col}: "
            f"[{self.rule_id}/{self.confidence.value}] {self.message}\n"
            f"    fix: {self.fix_suggestion}"
        )


@dataclass
class FileContext:
    """Everything a rule may look at for one file."""

    path: str
    source: str
    tree: ast.Module
    is_test_file: bool
    is_conftest: bool
    lines: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.lines:
            self.lines = self.source.splitlines()


class Rule:
    """Base class. Subclasses set the class attributes and implement check()."""

    id: ClassVar[str]  # e.g. "G1", "M2"
    name: ClassVar[str]  # kebab-case, e.g. "global-seed-mutation"
    cause: ClassVar[str]  # taxonomy cause, e.g. "randomness"
    confidence: ClassVar[Confidence]
    fix_suggestion: ClassVar[str]

    def check(self, ctx: FileContext) -> Iterable[Finding]:  # pragma: no cover
        raise NotImplementedError

    def finding(
        self,
        ctx: FileContext,
        node: ast.AST,
        message: str,
        fix: str | None = None,
        confidence: Confidence | None = None,
    ) -> Finding:
        return Finding(
            rule_id=self.id,
            rule_name=self.name,
            cause=self.cause,
            confidence=confidence or self.confidence,
            message=message,
            fix_suggestion=fix or self.fix_suggestion,
            path=ctx.path,
            line=getattr(node, "lineno", 1),
            col=getattr(node, "col_offset", 0),
        )


_REGISTRY: dict[str, type[Rule]] = {}


def register(cls: type[Rule]) -> type[Rule]:
    if not getattr(cls, "id", None):
        raise ValueError(f"Rule {cls.__name__} missing id")
    if cls.id in _REGISTRY:
        raise ValueError(
            f"Duplicate rule id {cls.id}: {cls.__name__} vs {_REGISTRY[cls.id].__name__}"
        )
    _REGISTRY[cls.id] = cls
    return cls


def all_rules() -> dict[str, type[Rule]]:
    # Import side effect: rules/__init__ walks submodules so decorators run.
    from flakehound import rules  # noqa: F401

    return dict(_REGISTRY)


def make_rules(
    disabled: frozenset[str] = frozenset(), predicate: Callable[[type[Rule]], bool] | None = None
) -> list[Rule]:
    out: list[Rule] = []
    for rid, cls in sorted(all_rules().items()):
        if rid in disabled:
            continue
        if predicate and not predicate(cls):
            continue
        out.append(cls())
    return out
