<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/logo.svg">
    <img src="docs/assets/logo.svg" alt="flakehound" width="140">
  </picture>
</p>

<h1 align="center">flakehound</h1>

<p align="center"><b>Hunt flaky tests before they bite.</b><br>
Static flaky-pattern detection + local run-history scoring for <code>pytest</code> —
with first-class awareness of <b>ML test flakiness</b> (JAX · PyTorch · NumPy).</p>

<p align="center">
  <a href="https://github.com/pctablet505/flakehound/actions"><img src="https://github.com/pctablet505/flakehound/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/flakehound/"><img src="https://img.shields.io/pypi/v/flakehound.svg" alt="PyPI"></a>
  <img src="https://img.shields.io/pypi/pyversions/flakehound.svg" alt="Python versions">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License"></a>
</p>

---

Flaky tests — tests that pass and fail on the same code — cause **13% of failing CI builds**
(Travis-scale study; **26%** at Microsoft). Every existing answer is either *retry harder*
(`pytest-rerunfailures`, `flaky`) or a *paid cloud dashboard* that notices flakiness only after
it has already burned your CI for weeks. Nothing predicts, nothing diagnoses, and nothing
understands why **ML test suites** flake (seeds, tolerances, GPU nondeterminism).

**flakehound is different:**

- 🔎 **Predict** — an AST rule corpus flags flaky-*prone* patterns at pre-commit time, before
  the first flake: global seed mutation, unordered-collection asserts, unfrozen `datetime.now()`,
  un-mocked network calls, leaked threads, shared-state fixtures, and more.
- 📈 **Score** — a pytest plugin logs every outcome to a **local SQLite** history (no cloud, no
  account) and computes per-test flakiness scores from flip-rate, recency, duration variance,
  and failure entropy. Plays clean with `xdist`, `randomly`, `rerunfailures`.
- 🧪 **The ML wedge** — the only tool that knows `assert_allclose(atol=1e-8)` on a stochastic
  model output is a time bomb, that a reused JAX `PRNGKey` breaks test independence, and that
  your GPU test needs `torch.use_deterministic_algorithms(True)`.
- 🩺 **Diagnose, don't just retry** — every finding carries a root cause and a concrete fix
  suggestion; the report ranks tests with evidence (e.g. "fails only under 3 of 40
  pytest-randomly seeds → order-dependent").

## Quickstart

```bash
pip install flakehound

# static scan — no tests executed, instant
flakehound scan tests/

# as part of a pytest run
pytest --flakehound
```

Example output:

```
tests/test_model.py:41:4: [G1/high] `np.random.seed(...)` mutates global RNG state; test outcomes now depend on execution order
    fix: use a local generator: `rng = np.random.default_rng(seed)` ...
tests/test_infer.py:88:8: [M3/medium] atol=1e-8 on stochastic model output is tighter than the output's observed variance
    fix: derive the bound from the output distribution (see FLEX, FSE'21) ...
```

### Pre-commit

```yaml
repos:
  - repo: https://github.com/pctablet505/flakehound
    rev: v0.1.0
    hooks:
      - id: flakehound
```

## Why not just …?

| | retries (`rerunfailures`, `flaky`) | SaaS (Trunk / BuildPulse / Datadog) | **flakehound** |
|---|---|---|---|
| Predicts before first flake | ❌ | ❌ | ✅ static rules |
| Root-cause diagnosis | ❌ | ❌ (history only) | ✅ per-rule cause + fix |
| ML-testing aware | ❌ | ❌ | ✅ JAX/PyTorch/NumPy rules |
| Local-first / no cloud | ✅ | ❌ | ✅ SQLite in your repo |
| Price | free | $0–499/mo | free, Apache-2.0 |

## Rules

Run `flakehound scan --help` for the toggles. Current corpus: `G1–G12` (general Python
flakiness causes, ranked by measured frequency in a 22k-project study) + `M1–M5` (the ML pack).
Full catalog with examples: [docs/rules.md](docs/rules.md).

Configure in `pyproject.toml`:

```toml
[tool.flakehound]
fail_on = "high"          # high | medium | advisory | never
disable = ["G3"]
exclude = ["tests/legacy/*"]
ml_rules = true
```

## Roadmap

- **v0.1** — static rule corpus (G+M), `scan` CLI, pre-commit hook *(you are here)*
- **v0.2** — run-history scoring: SQLite outcome log, flip-rate model, `flakehound report`,
  xdist-safe merging, seed-covariate order-dependence evidence
- **v0.3** — GitHub Action with cached history + PR annotations; quarantine recommendations
- **later** — auto-fix codemods, FLEX-style tolerance estimation

Benchmarked against the [IDoFT](https://github.com/TestingResearchIllinois/idoft) Python
dataset and Gruber et al.'s FlaPy corpus — numbers published honestly in
[docs/benchmarks.md](docs/benchmarks.md) as they land.

## Contributing

Rules are tiny, self-contained, and *very* contributable — one module + a true-positive
fixture + a false-positive guard. See [CONTRIBUTING.md](CONTRIBUTING.md) and the
[`good first issue`](https://github.com/pctablet505/flakehound/labels/good%20first%20issue)
label. If your team fought a flaky pattern we don't catch, we want the rule.

## License

Apache-2.0.
