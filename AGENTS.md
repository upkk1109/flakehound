# flakehound — agent instructions

Open-source pytest plugin/CLI: static flaky-test detection + local run-history scoring,
ML-testing aware. Public repo; everything here ships to strangers — quality bar is "would a
pytest-core reviewer merge this?"

## Binding design contract

- Plan of record: hub `kimi-colab-context/data/context/projects/flakehound/PLAN-flakehound-v1.md`.
  Implementers implement; design changes need a `proposal:` to the planner — do not invent
  architecture mid-task.
- One rule = one module in `src/flakehound/rules/` + `@register` + class attrs
  (`id`, `name`, `cause`, `confidence`, `fix_suggestion`). Copy the `g1_global_seed.py` +
  `tests/test_g1_global_seed.py` pattern exactly. Rules are pure functions of `FileContext`:
  no I/O, no globals, stdlib `ast` only (no tree-sitter/libcst in v1).
- **Confidence honesty**: HIGH only if a static match is near-certain flaky-prone; anything
  heuristic = MEDIUM; anything needing runtime evidence = ADVISORY. Overclaiming tiers is the
  fastest way to lose users — treat it as a correctness bug.
- Every rule ships a true-positive test AND a false-positive guard test (the FP guard is what
  keeps this tool installable; a noisy linter gets uninstalled in a day).
- v1 scope locks: no LLM calls, no auto-applied fixes, no Jest, no cloud. Suggestions are text.

## Quality gates (all must pass before commit)

```bash
ruff check src tests && ruff format --check src tests
pyright src
pytest -p randomly -q
flakehound scan tests/            # dogfood: our own tests stay clean (fixtures excluded)
```

Fail-before/pass-after: a bugfix commit must contain the test that fails without it.
One commit per task, message `M<milestone>-<task>: summary`. No AI co-author trailers.

## Style

Match existing code: `from __future__ import annotations`, dataclasses, type hints, ~100 col.
Docstrings explain *why* (cite the research: rule docstrings reference the cause taxonomy).
User-facing text (CLI, findings, README) is product copy — crisp, no jargon, no emoji in CLI
output. README/docs edits keep the marketing voice (confident, concrete, zero fluff).

## Anti-slop (highest priority)

VERIFY-FIRST against existing rules/linters before adding anything (no duplicating ruff/
flake8-pytest-style checks). No new deps without planner sign-off. No new abstractions the
plan doesn't name. Delete dead code, don't comment it out. Two failed attempts → stop, write
down what you tried, escalate.
