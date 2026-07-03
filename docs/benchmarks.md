# Benchmarks

**Status: not yet published.** flakehound v0.1.0 ships the static rule corpus
(`G1`–`G12`, `M1`–`M5`) without a measured benchmark run. This file is the honest
placeholder referenced from [README.md](../README.md) and
[docs/GROWTH.md](GROWTH.md) — it exists so the reference isn't a dead link, not to
imply numbers exist yet.

## What's coming with v0.2

The benchmark harness lands alongside v0.2's run-history scoring. It will measure
the static rule corpus against two public datasets:

- **[IDoFT](https://github.com/TestingResearchIllinois/idoft)** (`py-data.csv`,
  Python subset) — precision and recall of the static rules against known
  flaky/non-flaky tests.
- **Gruber et al.'s FlaPy corpus** — detection latency (how many runs until a known
  flaky test is flagged) for the history-scoring path.

## Pre-registered gates

These thresholds are committed *before* the numbers exist, so they can't be
picked after the fact to flatter the result:

| Metric | Gate |
|---|---|
| Precision (IDoFT `py-data.csv`, static rules) | ≥ 70% |
| Recall (IDoFT `py-data.csv`, static rules) | ≥ 50% |
| False-positive rate (known-stable subset) | < 10% |
| Detection latency (FlaPy traces, history scoring) | ≤ 10 runs |

## What will be published here

When the harness runs, this page will report the actual numbers against every
gate above — including any gate that is **missed**. A miss disclosed here reads
as rigor; a miss discovered by a stranger reads as marketing. No cherry-picked
subsets, no silent rule-tuning after the numbers are in.
