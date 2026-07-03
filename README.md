<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/pctablet505/flakehound/main/docs/assets/logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/pctablet505/flakehound/main/docs/assets/logo.svg">
    <img src="https://raw.githubusercontent.com/pctablet505/flakehound/main/docs/assets/logo.svg" alt="flakehound" width="140">
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

Flaky tests — tests that pass and fail on the same code — are expensive and common: about
**13% of failing builds** in a 61-project study of Travis CI Java projects were caused by
flaky tests ([Labuschagne et al., 2017](#references)), and an internal Microsoft study found
**26% of tests** in a large industrial suite exhibited flaky behavior
([Lam et al., ISSTA 2019](#references)). Every existing answer is either *retry harder*
(`pytest-rerunfailures`, `flaky`) or a *paid cloud dashboard* that notices flakiness only after
it has already burned your CI for weeks. Nothing predicts, nothing diagnoses, and nothing
understands why **ML test suites** flake (seeds, tolerances, GPU nondeterminism).

**flakehound is different:**

- 🔎 **Predict** — an AST rule corpus flags flaky-*prone* patterns at pre-commit time, before
  the first flake: global seed mutation, unordered-collection asserts, unfrozen `datetime.now()`,
  un-mocked network calls, leaked threads, shared-state fixtures, and more.
- 🧪 **The ML wedge** — a static rule pack tuned for ML numeric and RNG patterns
  (JAX/PyTorch/NumPy): tight `assert_allclose(atol=1e-8)` tolerances on stochastic model
  output, reused JAX `PRNGKey`s that break test independence, and missing
  `torch.use_deterministic_algorithms(True)` on GPU tests.
- 🩺 **Diagnose, don't just retry** — every finding carries a root cause and a concrete fix
  suggestion (e.g. "mutates global RNG state; use a local generator instead").
- 📈 **Score (coming in v0.2)** — a pytest plugin will log every outcome to a **local SQLite**
  history (no cloud, no account) and compute per-test flakiness scores from flip-rate, recency,
  duration variance, and failure entropy, ranking tests with evidence (e.g. "fails only under
  3 of 40 pytest-randomly seeds → order-dependent"). Today the plugin adds a terminal summary;
  see [Roadmap](#roadmap).

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
tests/test_infer.py:88:8: [M3/advisory] `atol=1e-08` on a value derived from `model.predict(...)` is machine-precision tight; stochastic/float32 model outputs routinely disagree by more than this run-to-run
    fix: justify this bound or derive it from an observed distribution of repeated runs (FLEX FSE-21) ...
```

### Pre-commit

`rev` must match a released git tag (tags are cut together with the corresponding
GitHub Release) — it will not resolve before that tag exists.

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
| Local-first / no cloud | ✅ | ❌ | ✅ static scan, no server (SQLite history in v0.2) |
| Price | free | $0–499/mo | free, Apache-2.0 |

<sup>Competitor names, features, and pricing as of July 2026 — verify against each
vendor's own site before relying on them; see [References](#references).</sup>

## Rules

Run `flakehound scan --help` for the toggles. Current corpus: `G1–G12` (general Python
flakiness causes) + `M1–M5` (the ML pack).

| ID | Rule | ID | Rule |
|---|---|---|---|
| G1 | global-seed-mutation | G9 | hardcoded-tmp-paths |
| G2 | unordered-collection-compare | G10 | event-loop-misuse |
| G3 | sleep-in-test | G11 | leaked-threads-timers |
| G4 | naive-now | G12 | env-mutation |
| G5 | shared-state-fixture | M1 | unseeded-stochastic-assert |
| G6 | import-time-side-effects | M2 | jax-prngkey-reuse |
| G7 | unmocked-network | M3 | suspicious-tolerance |
| G8 | float-equality-without-tolerance | M4 | missing-determinism-flags |
| | | M5 | module-scope-jit |

Every finding prints its cause and fix suggestion inline — see the example output above.
Full catalog with a bad/good example and fix for every rule: [docs/rules.md](docs/rules.md).

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
dataset and Gruber et al.'s FlaPy corpus — numbers will be published honestly in
`docs/benchmarks.md` here as they land.

## Contributing

Rules are tiny, self-contained, and *very* contributable — one module + a true-positive
fixture + a false-positive guard. See [CONTRIBUTING.md](CONTRIBUTING.md) and the
[`good first issue`](https://github.com/pctablet505/flakehound/labels/good%20first%20issue)
label. If your team fought a flaky pattern we don't catch, we want the rule.

## References

- Labuschagne, A., Inozemtseva, L., & Holmes, R. (2017). *Measuring the cost of regression
  testing in practice: a study of Java projects using continuous integration.* ESEC/FSE 2017.
  [doi:10.1145/3106237.3106288](https://doi.org/10.1145/3106237.3106288) — source of the ~13%
  failing-builds-due-to-flakiness figure (935 builds, 61 Travis CI Java projects).
- Lam, W., Godefroid, P., Nath, S., Santhiar, A., & Thummalapenta, S. (2019). *Root causing
  flaky tests in a large-scale industrial setting.* ISSTA 2019.
  [doi:10.1145/3293882.3330570](https://doi.org/10.1145/3293882.3330570) — source of the ~26%
  flaky-test figure at Microsoft.
- Parry, O., Kapfhammer, G. M., Hilton, M., & McMinn, P. (2021). *A survey of flaky tests.*
  ACM Transactions on Software Engineering and Methodology, 31(1).
  [doi:10.1145/3476105](https://doi.org/10.1145/3476105) — background survey covering causes,
  detection, and mitigation across the flaky-test literature.

Competitor names/pricing in the comparison table above are not independently cited here;
verify current features and pricing directly with each vendor.

## License

Apache-2.0.
