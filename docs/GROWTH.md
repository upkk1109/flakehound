# flakehound growth playbook

Gates, not vibes — this doc is the popularity plan for a tool with zero users today.
Every phase below has a trigger condition tied to `PLAN-flakehound-v1.md`'s milestones.
Don't skip ahead: a Show HN post against an empty rule corpus burns the launch slot you
only get once per domain.

**Status right now:** v0.1.0 corpus shipped (`G1`–`G12`, `M1`–`M5`). Launch sequence is
unblocked pending the FIX-FIRST wave (see the pre-release review) and the M4 gate
(IDoFT/FlaPy benchmarks in `docs/benchmarks.md`). "Credibility assets" and "Contribution
flywheel" are actionable now — do those first, they make the eventual launch land harder.

## Positioning

**One-liner:** flakehound predicts and diagnoses flaky pytest tests before they ever
flake — static rules for the cause, local run-history for the evidence, and the only
rule pack that understands ML test nondeterminism (JAX/PyTorch/NumPy).

**Elevator pitch (2 sentences, use verbatim in bios/directory listings):**
> flakehound is a free, local-first pytest plugin that flags flaky-prone code at
> commit time and scores real flakiness from your own CI history — no cloud, no
> account, and the first tool that knows a reused JAX `PRNGKey` or an `atol=1e-8`
> assertion on stochastic model output is a time bomb.

**Who it's for, in order:** (1) ML/scientific-Python teams (JAX, PyTorch, NumPy test
suites) — the wedge, the only unclaimed niche; (2) any pytest user who's hand-diagnosed
a flaky test and wants the "why" instead of a retry; (3) teams already burned by a paid
flaky-test SaaS who want the static/predictive half those tools don't do.

**Anti-positioning (say this explicitly, don't let people assume):** not a retry
wrapper (`pytest-rerunfailures`/`flaky` compete on "hide it", we compete on "prevent
it"), not a hosted dashboard, not a code-rewriting bot, not an LLM wrapper. v1 has zero
network calls at runtime — say that in every launch post, it's the most credible thing
about the project.

## Launch sequence

Order matters more than any individual post's wording. Each step gates the next.

1. **Ship the benchmark gate first.** Do not launch before `docs/benchmarks.md` exists
   with real IDoFT/FlaPy numbers and the binding gates from the plan are met (precision
   ≥ 70%, recall ≥ 50% on IDoFT `py-data.csv`; known-flaky flagged within ≤ 10 runs on
   FlaPy traces; false-positive rate on known-stable < 10%). If a gate misses, publish
   the miss — see "Credibility assets" below. A launch post with no evidence table is a
   launch post nobody reshares.
2. **Add the credibility assets to README before any public post.** HN/Reddit/pytest
   readers click through to the repo in the first 10 seconds; if the README doesn't
   already have the benchmark table and the "why not just retries/SaaS" comparison, the
   post itself cannot save the conversion.
3. **Post order (spread across ~1 week, never same-day cross-post):**
   - **Day 0 — pytest-dev/pytest Discussions**, "Show and tell" category (verify it's
     still enabled before posting — check `github.com/pytest-dev/pytest/discussions`
     directly, community-feature availability drifts). Lowest-stakes, most technically
     literal audience, catches plugin-architecture mistakes (hook misuse, xdist bugs)
     before the bigger crowds see them. Fix anything they flag before Day 3.
   - **Day 3–4 — Show HN.** Post Tuesday–Thursday, ~7–9am Pacific — the commonly-cited
     HN window for catching the US morning read and getting enough early upvotes before
     the front-page algorithm's time-decay kicks in. This is community folklore, not a
     measured stat — don't repeat it as fact in the post itself, just use it for timing.
   - **Day 5–6 — r/Python.** Weekday morning US Eastern, never Friday evening/weekend
     (mod queue + engagement both drop). Different post copy than HN — see drafts below.
   - **Optional, only after the above land well:** r/MachineLearning or r/mlops, ML-wedge
     framing only, own copy, own day. Do not run all four in the same week.
4. **Be present.** Answer every comment within a few hours on HN/Reddit launch days
   especially — an absentee author reads as a drive-by post and kills momentum even if
   the tool is good.
5. **Convert feedback into `good first issue`s within 48h.** Every "you missed pattern
   X" comment becomes a rule-proposal issue (template below) with the commenter
   `@`-credited in the issue body. This is the actual growth mechanism — launch day gets
   eyeballs, the issue tracker converts eyeballs into contributors.

### Post drafts (verbatim — fill placeholders from `docs/benchmarks.md`, never post with
a placeholder still in it, never invent a number to fill one)

#### Show HN

Title:
```
Show HN: flakehound – predict and diagnose flaky pytest tests before they flake
```

Body:
```
Hi HN — I built flakehound because every flaky-test tool I could find either retries
harder after the fact (pytest-rerunfailures, flaky) or is a paid SaaS dashboard that
tells you a test is flaky only after it's already burned a few weeks of your CI.
Nothing predicts flakiness before the first flake, and nothing understands why ML test
suites flake specifically.

flakehound is a pytest plugin + CLI, local-first, free, Apache-2.0:

1. Static prediction (`flakehound scan`) — an AST rule corpus flags flaky-prone
   patterns at commit time: global RNG seed mutation, unordered-collection asserts,
   un-mocked network calls, tolerance-too-tight assert_allclose on stochastic model
   output, JAX PRNGKey reuse, missing torch.use_deterministic_algorithms, and more.
   Every finding carries a confidence tier (HIGH/MEDIUM/ADVISORY, never overclaimed)
   and a concrete fix — not "flaky? just retry it."
2. Local history scoring — a pytest plugin logs every outcome to SQLite in your own
   repo (no account, no cloud) and scores tests on flip-rate, recency-weighted
   transitions, duration variance, and failure entropy.

The niche I actually built this for: ML/scientific Python test suites.
assert_allclose(atol=1e-8) on a stochastic model output, a reused JAX PRNGKey, a GPU
test with no determinism flags — these are common, unowned by any existing tool, and
they're the failures I've personally lost hours to.

Benchmarked against the public IDoFT and FlaPy flaky-test datasets: <PRECISION>%
precision / <RECALL>% recall on IDoFT's Python corpus, known-flaky tests flagged
within <N> runs on FlaPy traces. Misses documented honestly, no cherry-picking:
docs/benchmarks.md.

v0.1 is static rules + scan CLI + pre-commit hook. v0.2 adds the history scoring.
Zero LLM calls anywhere in the codebase, zero telemetry.

github.com/pctablet505/flakehound — feedback and "you missed rule X" reports very
welcome, especially from anyone who's fought JAX/PyTorch/TF test flakiness.
```

#### r/Python

Title:
```
flakehound – a static analyzer + local history scorer for flaky pytest tests (ML-aware: JAX/PyTorch/NumPy)
```

Body:
```
Sharing a tool I've been building: flakehound, a pytest plugin + CLI that predicts
flaky tests statically (AST rules, like a linter but for flakiness causes — seed
mutation, unordered asserts, unmocked network, timing races, ML-specific stuff like
JAX PRNGKey reuse) and scores real flakiness from your own local run history via a
SQLite log — no cloud, no account, Apache-2.0.

Why not pytest-rerunfailures / flaky? Those hide flakiness by retrying. flakehound
tries to catch the pattern before the test ever flakes, and tells you the root cause
+ a concrete fix instead of "this test is unreliable, retrying."

The part I'd love r/Python's eyes on specifically: the ML rule pack (M1-M5). If you've
ever had a JAX/PyTorch/TF test suite that flaked and spent an afternoon figuring out it
was a reused PRNGKey or a tolerance that was never going to hold — that's exactly the
class of bug this is built to catch before it ships.

Benchmarked against the IDoFT and FlaPy public datasets: <PRECISION>% precision /
<RECALL>% recall on IDoFT (numbers + misses in docs/benchmarks.md, nothing hidden).

Repo: github.com/pctablet505/flakehound. Rules are intentionally tiny and
contributable — one rule = one small file + a true-positive test + a false-positive
guard test, ~30 min for a first PR (CONTRIBUTING.md walks through it). If your team has
fought a flaky pattern we don't catch yet, that's the best kind of issue to open.
```

#### pytest-dev/pytest Discussions (Show and tell)

Title:
```
flakehound: static flaky-pattern detection + local history scoring plugin (feedback wanted on hook usage)
```

Body:
```
Posting here first because I'd rather have plugin-authors catch my hook mistakes
before a wider crowd sees them.

flakehound is a pytest plugin that (a) statically flags flaky-prone AST patterns via a
`scan` CLI / pre-commit hook, and (b) as a pytest plugin, logs every test outcome to a
local SQLite DB and scores flakiness from flip-rate/recency/duration-variance/failure-
entropy.

Plugin-architecture specifics I'd appreciate a second pair of eyes on:
- hooks used: pytest_addoption, pytest_configure (captures pytest-randomly's seed and
  the xdist worker id), pytest_collection_modifyitems, pytest_runtest_logreport
  (outcomes — filtering `when=="rerun"` so pytest-rerunfailures reruns don't
  double-count), pytest_terminal_summary.
- xdist: per-worker DB (`.flakehound/<workerid>.db`), merged at sessionfinish, modeled
  on pytest-cov's merge pattern.
- plays alongside pytest-randomly and pytest-rerunfailures rather than replacing them —
  wanted to confirm that's the right community expectation rather than trying to own
  retry/ordering behavior myself.

Repo: github.com/pctablet505/flakehound. Also curious whether a pytest plugin-list
entry makes sense once v0.2 (the history-scoring half) lands — happy to follow whatever
the current process is.
```

## Credibility assets

### IDoFT benchmark table (goes in README, directly under the existing "Why not just
…?" table, once the M4 gate passes)

```markdown
## Benchmarked, not asserted

Measured against the [IDoFT](https://github.com/TestingResearchIllinois/idoft) Python
corpus and Gruber et al.'s FlaPy traces. Pre-registered gates (`PLAN-flakehound-v1.md`),
published whether they pass or not — full breakdown: [docs/benchmarks.md](docs/benchmarks.md).

| Metric | Gate | Measured |
|---|---|---|
| Precision (IDoFT `py-data.csv`, static rules) | ≥ 70% | `<fill from benchmarks.md>` |
| Recall (IDoFT `py-data.csv`, static rules) | ≥ 50% | `<fill from benchmarks.md>` |
| False-positive rate (known-stable subset) | < 10% | `<fill from benchmarks.md>` |
| Detection latency (FlaPy traces, history scoring) | ≤ 10 runs | `<fill from benchmarks.md>` |
```

Rule: if any row misses its gate, ship the table anyway with the miss visible and a
one-line note on what's being adjusted (per the plan: "adjust rules, don't ship" applies
to the *rule corpus*, not to hiding a published number). A miss you disclosed reads as
rigor; a miss discovered by a stranger reads as marketing.

### Comparison-vs-Trunk page (`docs/vs-trunk.md` — outline only, write it once you can
cite specifics, not from memory)

Don't write this page speculatively. Before drafting, re-verify Trunk's (and any other
named competitor's) current features/pricing directly from their own site and date the
citation — SaaS feature sets and pricing change, and a stale claim in a comparison page
is the single fastest way to lose credibility with the exact audience (senior ICs) who
will fact-check it. Outline:

1. **What they do well** — be genuinely fair (cross-language CI history, org-wide
   dashboards, auto-quarantine, chat/PR integrations). A comparison page that only
   attacks reads as insecure, not confident.
2. **What flakehound does that they structurally can't** — predicts before the first
   flake (static rules vs. history-only), local-first (nothing leaves your repo), free,
   ML-specific rule pack.
3. **What flakehound doesn't do (yet)** — no cross-language support, no hosted
   dashboard, no org-wide aggregation across repos. State this plainly.
4. **"Use both"** — position as complementary, not a replacement: flakehound prevents,
   a CI-scale SaaS manages the aftermath once you're past what static analysis can see.
5. Footer: "features/pricing verified as of `<date>` — check their site for current
   details" on every claim about a competitor.

## Contribution flywheel

The corpus is deliberately tiny (`G1`–`G12` general, `M1`–`M5` ML — see
`PLAN-flakehound-v1.md` §2) so it undersells what's catchable. Seed the gap as `good
first issue`s *now*, before launch — an empty issue tracker on launch day converts zero
visitors into contributors. Each proposal below is scoped to the CONTRIBUTING.md
30-minute recipe (one module, TP fixture, FP-guard fixture) and picks up numbering after
the locked v1 corpus (`G13`+, `M6`+ — confirm against `docs/rules.md` and open PRs before
filing, in case one landed since this was written).

Create them with the repo's own template:

```bash
for spec in \
  "G13|retry-loops-without-backoff|A test manually retries an assert/call in a for/while loop (try/except + continue) with no backoff and no bound — masks a race instead of fixing it. cause: masked-flakiness. confidence: MEDIUM." \
  "G14|uuid-hash-order-assumption|Asserts on the Nth element of list(some_set) or dict built from uuid4()/hash-derived keys without sorted() — order isn't guaranteed across PYTHONHASHSEED. cause: randomness/order-dependence (sibling of G1/G2). confidence: MEDIUM." \
  "G15|silent-except-in-test|A bare except: / except Exception: in a test body that only pass/continues — swallows the real failure and produces a false green. cause: masked-flakiness. confidence: HIGH." \
  "G16|unlinked-flaky-marker|@pytest.mark.flaky or xfail(strict=False) with no comment/issue-URL nearby — a band-aid with no path back to a real fix. cause: masked-flakiness (process rule). confidence: ADVISORY." \
  "G17|cross-test-class-mutation|A test method assigns to self.__class__.<attr> or appends to a class/module-level list — later tests in the same class depend on run order. cause: shared-state (sibling of G5). confidence: MEDIUM." \
  "G18|unseeded-input-generation|random.choice/sample/randint (stdlib random, not a local Random()) used inside a test body to build an expected-value assertion. cause: randomness (sibling of G1, but about consuming randomness for inputs, not seeding). confidence: MEDIUM." \
  "G19|wall-clock-perf-assertion|assert <duration> < <literal seconds> where duration comes from time.time()/perf_counter() deltas — flakes under CI load. cause: timing/environment. confidence: MEDIUM." \
  "M6|dataloader-worker-nondeterminism|torch.utils.data.DataLoader(..., num_workers=N>0) with no worker_init_fn= or generator= — each worker process gets an independently-seeded RNG, so shuffling/augmentation order varies run to run. cause: ML/concurrency. confidence: MEDIUM." \
  "M7|unseeded-tf-shuffle|tf.data .shuffle(buffer_size) or model.fit(..., shuffle=True) with no tf.random.set_seed(...) reachable in scope. cause: ML/randomness. confidence: MEDIUM." \
  "M8|random-split-no-generator|torch.utils.data.random_split(...) call missing a generator= keyword — dataset partition changes between runs. cause: ML/randomness. confidence: HIGH." \
  ; do
  IFS='|' read -r id name desc <<< "$spec"
  gh issue create \
    --title "[rule] $id $name" \
    --label "good first issue,rule-proposal" \
    --body "$desc

Use the CONTRIBUTING.md 30-minute recipe: one module in src/flakehound/rules/, \`@register\`, a true-positive fixture test, a false-positive guard test. Confidence tier above is a starting suggestion, not a mandate — argue it up or down in review per the confidence-tier-honesty policy."
done
```

Run this once, after confirming the repo's issue labels (`good first issue`,
`rule-proposal`) exist (`gh label create` if not). Don't run it twice — check open/closed
issues first so you don't duplicate.

## Badge / social-preview checklist

- [ ] GitHub "About" panel: description matches `pyproject.toml`'s, topics set to
      `pytest`, `flaky-tests`, `testing`, `static-analysis`, `ci`, `machine-learning`,
      `jax`, `pytorch` (mirrors the package keywords — this is how GitHub topic search
      surfaces the repo).
- [ ] Social preview image (repo Settings → General → Social preview): 1280×640px PNG
      — GitHub does not accept SVG here, so render `docs/assets/logo-dark.svg` to PNG at
      that canvas with the wordmark and tagline, dark background (most link-preview
      surfaces — Slack, X, Discord — default to showing it on a dark or neutral card).
- [ ] README badges already present (CI, PyPI, pyversions, license) — leave as is. Add
      a `good first issue` count badge (`shields.io` GitHub issues-by-label query) once
      the flywheel issues above are filed. Do **not** add a downloads badge while the
      count is near-zero — a visible "12 downloads" undercuts credibility harder than no
      badge at all; add it once the number is a story, not an apology.
- [ ] Tag a real GitHub **Release** for `v0.1.0` (not just a git tag) with a short
      changelog — HN/Reddit readers check the Releases tab as a maturity signal before
      they check the code.
- [ ] `docs/assets/icon.svg` is ready for a docs-site favicon (mkdocs or similar) if/when
      one exists; not needed for the GitHub repo page itself.
- [ ] `.pre-commit-hooks.yaml` already ships in the repo root — once `v0.1.0` is tagged,
      the README's pre-commit snippet (`rev: v0.1.0`) is immediately correct; no separate
      registry submission needed for self-hosted pre-commit usage.

## What NOT to do

- **No fake stars.** No star-seeding services, no asking friends to mass-star in a short
  window. GitHub's abuse detection flags velocity spikes, and it's simply dishonest —
  the number is supposed to mean something.
- **No vote manipulation on HN/Reddit.** No asking for upvotes, no vote rings, no
  multiple accounts. This is an explicit HN guideline violation and can get a post
  killed or the domain penalized long-term — one launch is not worth that.
- **No same-day cross-posting.** Identical copy posted to HN/Reddit/Discussions within
  hours of each other reads as spam to anyone who sees more than one, and strips out the
  chance to tailor the pitch per audience (see the drafts above — they're deliberately
  different).
- **No trash-talking competitors.** The vs-Trunk page states facts, dated and sourced.
  Don't disparage, don't cherry-pick their worst feature, don't undersell your own gaps.
- **No fabricated or rounded-up benchmark numbers.** If a gate misses, publish the miss
  (see Credibility assets). An invented precision number is one GitHub issue away from a
  public correction that costs far more trust than the honest number would have.
- **No padding the contributor count.** Every PR — including the seeded good-first-issues
  above — clears the same bar: true-positive test, false-positive guard, honest
  confidence tier. Merging a weak PR to look active is the fastest way to become the
  noisy linter people uninstall.
- **No good-first-issue bait.** Every seeded issue must be genuinely completable by a
  first-timer inside the ~30-minute promise in CONTRIBUTING.md working from the template
  alone. If you can't picture a stranger finishing it unaided, don't file it as "good
  first issue" — file it without the label instead.
- **No hype-speak creep.** Findings text, CLI output, and README copy stay in the
  existing product voice — confident, concrete, zero fluff, no emoji in CLI output (see
  `AGENTS.md`). Growth pressure is exactly what erodes this first; don't let it.
