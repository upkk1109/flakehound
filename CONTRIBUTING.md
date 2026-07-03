# Contributing to flakehound

Thanks for looking. flakehound's rule corpus is deliberately tiny and modular —
**one rule = one small file** — so once your environment is set up, your first PR
can realistically land in about 30 minutes (dev setup itself: about 10 minutes).
This doc gets you from clone to green CI.

## Dev setup

```bash
git clone https://github.com/pctablet505/flakehound.git
cd flakehound
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
pre-commit install
```

## Quality gates

Everything below must pass before you open a PR (CI runs the same commands):

```bash
ruff check src tests && ruff format --check src tests
pyright src
pytest -p randomly -q
flakehound scan tests/ --fail-on high     # we dogfood our own tool (matches CI)
```

## Adding a rule (the 30-minute recipe)

A rule is a pure function of a parsed test file: no I/O, no network, no global
state, stdlib `ast` only. Every rule ships in its own module plus **two tests**:
a true-positive (it fires on the pattern) and a false-positive guard (it stays
quiet on the obvious lookalike). Use
[`src/flakehound/rules/g1_global_seed.py`](src/flakehound/rules/g1_global_seed.py)
and [`tests/test_g1_global_seed.py`](tests/test_g1_global_seed.py) as the
template — copy them, then edit.

Walkthrough — adding a hypothetical `G99` rule that flags bare `time.sleep(...)`
in tests (`G99` is a placeholder; pick the next free ID — check
[docs/rules.md](docs/rules.md) and open PRs before you claim one):

```python
# src/flakehound/rules/g99_sleep_call.py
"""G99: unconditional time.sleep(...) in tests — timing-based flakiness."""

from __future__ import annotations

import ast
from typing import Iterable

from flakehound.rules.base import Confidence, FileContext, Finding, Rule, register


@register
class SleepCall(Rule):
    id = "G99"
    name = "sleep-call"
    cause = "timing/race-condition"
    confidence = Confidence.MEDIUM  # heuristic: sleep isn't always wrong
    fix_suggestion = (
        "poll for the condition instead (e.g. tenacity/wait_for), or inject a "
        "fake clock; a fixed sleep is a race with the CI runner's load"
    )

    def check(self, ctx: FileContext) -> Iterable[Finding]:
        for node in ast.walk(ctx.tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "sleep"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "time"
            ):
                yield self.finding(ctx, node, "`time.sleep(...)` in a test races the runner's load")
```

```python
# tests/test_g99_sleep_call.py
from __future__ import annotations

import ast

from flakehound.rules.base import FileContext
from flakehound.rules.g99_sleep_call import SleepCall


def _ctx(source: str) -> FileContext:
    return FileContext(path="test_x.py", source=source, tree=ast.parse(source),
                        is_test_file=True, is_conftest=False)


def _run(source: str):
    return list(SleepCall().check(_ctx(source)))


def test_detects_time_sleep():
    findings = _run("import time\ndef test_a():\n    time.sleep(1)\n")
    assert len(findings) == 1
    assert findings[0].rule_id == "G99"


def test_fp_guard_other_sleep_method():
    src = "def test_a(clock):\n    clock.sleep(1)\n"
    assert _run(src) == []
```

That's it — no registry file to edit, `@register` + import-time discovery in
`rules/__init__.py` handles it. Run the gates above, and you're done.

### Confidence-tier honesty (read this before picking a tier)

- **HIGH** — the static match is near-certain flaky-prone. Fine to fail a
  pre-commit hook on.
- **MEDIUM** — a real heuristic with plausible false positives.
- **ADVISORY** — needs runtime evidence to confirm; never blocks a commit.

Overclaiming a tier is treated as a correctness bug, not a style nit — it's the
fastest way a real project uninstalls flakehound. If you're unsure, start at
the tier below your instinct and let review argue you up, not down.

## PR checklist

- [ ] One rule per PR (or a focused, single-purpose fix)
- [ ] True-positive test **and** a false-positive guard test
- [ ] Confidence tier matches what the rule can actually prove statically
- [ ] `fix_suggestion` is concrete and actionable, not generic advice
- [ ] `ruff check`, `ruff format --check`, `pyright src`, `pytest -p randomly -q` all pass
- [ ] `flakehound scan tests/` stays clean
- [ ] Docstring cites *why* the pattern flakes, not just *what* it matches

If your team fought a flaky pattern flakehound doesn't catch yet, that's the
best kind of contribution — open an issue with the
[rule proposal template](.github/ISSUE_TEMPLATE/rule_proposal.md) or just send
the PR.
