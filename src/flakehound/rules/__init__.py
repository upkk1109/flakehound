"""Rule package: auto-import every rule module so @register decorators run."""

from __future__ import annotations

import importlib
import pkgutil

from flakehound.rules.base import (  # noqa: F401
    Confidence,
    FileContext,
    Finding,
    Rule,
    all_rules,
    make_rules,
    register,
)

for _mod in pkgutil.iter_modules(__path__):
    if _mod.name not in {"base"}:
        importlib.import_module(f"{__name__}.{_mod.name}")
