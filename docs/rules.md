# Rule catalog

Generated from the live rule registry by `scripts/gen_rules_doc.py` — do not edit
by hand; regenerate instead:

```bash
.venv/bin/python scripts/gen_rules_doc.py --write
```

17 rules today: `G1`–`G12` general Python flakiness causes (ranked by measured frequency in a 22k-project study), `M1`–`M5` the ML pack (JAX/PyTorch/NumPy-aware). Every finding prints its `[ID/tier]`, cause, and a `fix:` suggestion inline. Check here (and open PRs) before claiming a new rule ID — see [CONTRIBUTING.md](../CONTRIBUTING.md).

| ID | Rule | Tier | Cause |
|---|---|---|---|
| [G1](#g1-global-seed-mutation) | `global-seed-mutation` | high | randomness/order-dependence |
| [G2](#g2-unordered-collection-compare) | `unordered-collection-compare` | high | iteration-order/nondeterminism |
| [G3](#g3-sleep-in-test) | `sleep-in-test` | high | timing/synchronization |
| [G4](#g4-naive-now) | `naive-now` | medium | time/wall-clock-nondeterminism |
| [G5](#g5-shared-state-fixture) | `shared-state-fixture` | medium | shared-state |
| [G6](#g6-import-time-side-effects) | `import-time-side-effects` | medium | import-time-side-effects/order-dependence |
| [G7](#g7-unmocked-network) | `unmocked-network` | high | network/infrastructure |
| [G8](#g8-float-equality-without-tolerance) | `float-equality-without-tolerance` | high | floating-point/precision |
| [G9](#g9-hardcoded-tmp-paths) | `hardcoded-tmp-paths` | high | filesystem/shared-state |
| [G10](#g10-event-loop-misuse) | `event-loop-misuse` | medium | concurrency/event-loop-lifecycle |
| [G11](#g11-leaked-threads-timers) | `leaked-threads-timers` | medium | concurrency/resource-leak |
| [G12](#g12-env-mutation) | `env-mutation` | high | environment/order-dependence |
| [M1](#m1-unseeded-stochastic-assert) | `unseeded-stochastic-assert` | medium | randomness/reproducibility |
| [M2](#m2-jax-prngkey-reuse) | `jax-prngkey-reuse` | high | randomness/jax-prng-reuse |
| [M3](#m3-suspicious-tolerance) | `suspicious-tolerance` | advisory | ml-numerics/tolerance |
| [M4](#m4-missing-determinism-flags) | `missing-determinism-flags` | advisory | ml-gpu/nondeterminism |
| [M5](#m5-module-scope-jit) | `module-scope-jit` | advisory | ml-jax/compilation-cache-order |

## G1: global-seed-mutation

*global RNG seed mutation in tests.*

**Tier:** `high`  **Cause:** `randomness/order-dependence`

`random.seed(...)` / `np.random.seed(...)` mutate process-global RNG state, so a
test's outcome starts depending on which tests ran before it — the #1 measured
cause of order-dependent flakiness in Python suites.

**Bad:**

```python
import random, torch
def test_a():
    random.seed(0)
    torch.manual_seed(0)
```

**Good:**

```python
import numpy as np
def test_a():
    rng = np.random.default_rng(42)
    x = rng.normal()
```

**Fix:** use a local generator: `rng = np.random.default_rng(seed)` (or `torch.Generator().manual_seed(seed)`), or isolate with an autouse save/restore fixture; pytest-randomly resets stdlib/numpy seeds per test

---

## G2: unordered-collection-compare

*equality assertions on unordered-collection materializations.*

**Tier:** `high`  **Cause:** `iteration-order/nondeterminism`

Converting a `set`, a dict view, or a filesystem listing into a `list` and then
comparing it to a literal with `==` bakes in an iteration order Python does not
promise: `set` iteration order depends on hashes (and `PYTHONHASHSEED` for
str/bytes), and `os.listdir`/`glob.glob` order is filesystem/platform
dependent. The test then passes or fails on incidental order, not on outcome.

**Bad:**

```python
def test_a():
    x = {3, 1, 2}
    assert list(set(x)) == [1, 2, 3]
```

**Good:**

```python
import os
def test_a():
    assert sorted(os.listdir('.')) == ['a.txt', 'b.txt']
```

**Fix:** sort both sides before comparing (`sorted(x) == sorted(y)`), or compare as sets/dicts directly (`set(x) == set(y)`) if order was never the point

---

## G3: sleep-in-test

*`time.sleep`/`asyncio.sleep` used as ad hoc test synchronization.*

**Tier:** `high`  **Cause:** `timing/synchronization`

A fixed sleep before asserting on background-thread or async-task state assumes
the awaited work finishes inside that window — true on a quiet machine, false
the moment a CI runner is under contention and the real event takes longer
than the guessed delay. `thread.start()` / `publisher.start()` followed by a
bare `time.sleep(N)` and then a direct assertion, with no actual wait
primitive in between, is one of the most common raw flakiness idioms in
real-world suites.

**Bad:**

```python
import time
def test_worker_processes_item(worker):
    worker.start()
    time.sleep(0.05)
    assert worker.processed == 1
```

**Good:**

```python
import time, asyncio
def test_a():
    time.sleep(0)
async def test_b():
    await asyncio.sleep(0)
```

**Fix:** wait on the real condition instead of guessing a delay: an `Event`/`Condition`.wait(timeout=...), a small polling helper with a deadline, `.join(timeout=...)` on the thread, or `pytest-timeout` to bound the worst case

---

## G4: naive-now

*naive `now()` reads driving test assertions.*

**Tier:** `medium`  **Cause:** `time/wall-clock-nondeterminism`

`datetime.now()`/`datetime.utcnow()`/`date.today()`/`time.time()` read the
real wall clock. When a test's assertion (directly, or via a variable that
was derived from one of these calls) depends on the value read, the test's
outcome now depends on *when* it happened to run — slow CI hosts, DST
transitions, and midnight/month/year boundaries all become sources of
flakiness. The fix is a frozen or injected clock, not a faster runner.

Overlaps ruff's flake8-datetimez (DTZ003 `utcnow`, DTZ005 `now`, DTZ011
`date.today`), but the framing differs: DTZ flags any naive construction,
unconditionally, because it is a timezone-correctness bug regardless of how
the value is used. G4 only fires when the clock read flows into an assertion
-- it is a flakiness rule, not a silent DTZ duplicate.

**Bad:**

```python
import time

def test_completes_quickly():
    start = time.time()
    do_work()
    elapsed = time.time() - start
    assert elapsed < 1.0
```

**Good:**

```python
import freezegun
from datetime import datetime

def test_frozen():
    with freezegun.freeze_time('2026-01-01'):
        assert datetime.now().year == 2026
        deadline = datetime.now()
        assert deadline.year == 2026
```

**Fix:** freeze the clock with `freezegun.freeze_time(...)` (or a `monkeypatch.setattr` clock stub), or inject the timestamp/clock as a parameter instead of reading it inside the function under test

---

## G5: shared-state-fixture

*module/session/class-scoped fixture holds shared mutable state.*

**Tier:** `medium`  **Cause:** `shared-state`

A fixture with `scope="module"` (or `"session"`/`"class"`) is built once and
handed to every test that requests it. If it returns or yields a mutable
object — a list/dict/set literal, or a freshly constructed class instance —
with no defensive copy, every test sharing the fixture reads and writes the
*same* object: whichever test runs first (or wherever xdist/pytest-randomly
lands it) silently seeds the state the next test sees. The same risk shows up
when the fixture body itself reaches into a module global or a shared class
attribute and mutates it directly — shared-state-via-fixture is one of the
classic order-dependence contaminators alongside global RNG mutation (G1) and
import-time construction.

**Bad:**

```python
import pytest
@pytest.fixture(scope="module")
def config():
    return {"a": 1}
```

**Good:**

```python
import pytest
@pytest.fixture
def config():
    return {"a": 1}
```

**Fix:** scope the fixture to `function` (the default) so each test gets its own object, or keep the wider scope but hand out a factory function / `copy.deepcopy(...)` / frozen data (tuple, `MappingProxyType`, a frozen dataclass) instead of the shared mutable object itself

---

## G6: import-time-side-effects

*import-time side effects in test modules.*

**Tier:** `medium`  **Cause:** `import-time-side-effects/order-dependence`

Module-level statements (top-level, not under `if __name__ == "__main__":`)
that perform I/O, network, subprocess, or process-global mutation run once at
*collection* time, in whichever order pytest/xdist happens to import the file
— coupling side-effect timing (and, for held object references, shared
mutable state) to test-collection order rather than test-execution order.
Constructing heavy stateful objects at module scope instead of inside a
fixture is a commonly-fixed order-dependence contaminator in real-world
suites; the mechanical fix is always the same: wrap the statement in a
fixture, or gate it behind `if __name__ == "__main__":`.

**Bad:**

```python
import requests

_SESSION = requests.Session()


def test_a():
    pass
```

**Good:**

```python
import pytest

pytest.importorskip("torch")


def test_a():
    pass
```

**Fix:** move I/O, network, subprocess, or global-state mutation into a fixture (function- or module-scoped as appropriate) or behind `if __name__ == "__main__":`; module-level statements run once at collection time, in whatever order pytest/xdist imports the file

---

## G7: unmocked-network

*un-mocked network calls in tests.*

**Tier:** `high`  **Cause:** `network/infrastructure`

A direct `requests`/`httpx`/`urllib`/`socket`/`aiohttp` call in a test body hits a
live network in CI — DNS hiccups, rate limits, and third-party downtime become test
flakiness that has nothing to do with the code under test. Gruber et al.'s 22k-project
study buckets this under the measured 13% "network/randomness" cause.

Real suites mock at this exact transport boundary: `responses`, `respx`,
`requests_mock`, `aioresponses`, a `pytest-httpserver`/localserver fixture, `vcr`, or a
plain `monkeypatch`/`unittest.mock.patch` of the callee. Presence of any of those
signals anywhere in the file suppresses the whole file — one shared mock/session
fixture commonly covers every test in it, and this rule is pure `ast` with no
cross-function dataflow, so it cannot prove a *specific* call site is still unmocked
once any of those signals exist.

**Bad:**

```python
import requests
def test_a():
    resp = requests.get('https://example.com/api')
    assert resp.status_code == 200
```

**Good:**

```python
import requests
import responses
@responses.activate
def test_a():
    requests.get('https://example.com')
```

**Fix:** mock the transport boundary — `responses`/`respx`/`requests_mock`/`aioresponses` for the client library in use, or a `pytest-httpserver`/localserver fixture for a real socket target — never hit the live network from a test

---

## G8: float-equality-without-tolerance

*float equality without tolerance.*

**Tier:** `high`  **Cause:** `floating-point/precision`

`assert a == b` (or `assertEqual`) where an operand is a float literal with a
fractional part, a true-division result, or a `math`/`np` call that produces a
float is comparing IEEE-754 values for exact equality — the classic source of
"passes on my machine, fails on CI" flakiness once either side has gone
through any arithmetic (accumulated rounding, differing libm/BLAS builds,
platform FMA differences). Fix: `math.isclose`/`pytest.approx`/`np.isclose`.

**Bad:**

```python
def test_a():
    result = compute()
    assert result == 3.14
```

**Good:**

```python
import pytest
def test_a():
    assert compute() == pytest.approx(3.14)
```

**Fix:** compare with tolerance: `math.isclose(a, b, rel_tol=...)`, `assert a == pytest.approx(b)`, or `np.isclose(a, b)`

---

## G9: hardcoded-tmp-paths

*hardcoded `/tmp` (or other shared, non-isolated) paths in tests.*

**Tier:** `high`  **Cause:** `filesystem/shared-state`

A bare string literal under `/tmp`, `tempfile.mktemp()` (which returns a path
without creating it -- a classic time-of-check/time-of-use race), or a plain
relative filename opened for write are all *not* isolated per test process.
Two tests reusing the same literal path collide under `pytest-xdist` parallel
workers, or when `pytest-randomly` interleaves files that happen to share a
subdirectory/file name; a killed previous run can also leave stale state that
pollutes a later "file does not exist yet" assertion. `tmp_path` (or
`tmp_path_factory`) gives every test its own directory, cleaned up
automatically, and removes the whole failure class.

Overlaps bandit's S108 (hardcoded_tmp_directory), but the framing differs:
S108 is a security check (predictable shared paths as a symlink-attack/
tamper vector) and isn't in ruff's default select. G9 is about per-test
isolation -- collisions under xdist/randomly -- and also covers
`tempfile.mktemp()` and relative write paths that S108 doesn't.

**Bad:**

```python
import os
def test_a():
    os.makedirs('/tmp/strategy_tests', exist_ok=True)
```

**Good:**

```python
def test_a(tmp_path):
    f = open(tmp_path / 'output.txt', 'w')
    f.write('x')
```

**Fix:** use the `tmp_path` fixture (or `tmp_path_factory` for session-scoped needs) instead of a hardcoded path; pytest gives every test its own isolated, auto-cleaned directory

---

## G10: event-loop-misuse

*manual asyncio event-loop management in tests.*

**Tier:** `medium`  **Cause:** `concurrency/event-loop-lifecycle`

Calling `asyncio.get_event_loop()` / `new_event_loop()` / `set_event_loop()` /
`loop.run_until_complete(...)` by hand — instead of letting pytest-asyncio or
anyio own the event loop's lifecycle — ties a test's outcome to loop state a
previous test created, closed, or never closed. The event loop (and the
process's event-loop policy) is process-global state, so this is the async
analogue of G1's global-RNG problem: order-dependence via shared mutable
state, this time a loop object instead of an RNG. `asyncio.run(...)` called
from inside an already-running `async def` test starts a second, nested loop
instead of reusing the one pytest-asyncio already provides.

**Bad:**

```python
import asyncio
def test_a():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(work())
```

**Good:**

```python
import asyncio
import pytest

@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
```

**Fix:** let pytest-asyncio (or anyio) own the loop: use `@pytest.mark.asyncio` with an explicit loop scope (e.g. `@pytest.mark.asyncio(loop_scope="function")`, or anyio's equivalent) instead of calling `asyncio.get_event_loop()`/`new_event_loop()`/`set_event_loop()`/`run_until_complete()` by hand

---

## G11: leaked-threads-timers

*threads/timers/processes/executors started but never cleaned up.*

**Tier:** `medium`  **Cause:** `concurrency/resource-leak`

A `threading.Thread`/`threading.Timer`/`multiprocessing.Process`/
`concurrent.futures` executor that is started but never joined, cancelled,
terminated, or shut down can outlive the test that created it: it leaks into
later tests sharing the same worker process (state mutation, contention for
shared resources), or leaves teardown hanging on a `pytest-timeout` kill. The
reference corpus mined for this rule shows this is usually paired with a
`time.sleep()` used as an (unreliable) synchronization proxy in place of a
real join — see G3 — and that the dominant safe idiom holds threads in a
list/collection (`threads = [Thread(...) for _ in range(n)]`) rather than a
single name, so the check must look at the whole function body, not just the
line immediately after `.start()`.

**Bad:**

```python
import threading

def test_leak():
    t = threading.Thread(target=lambda: None)
    t.start()
```

**Good:**

```python
from concurrent.futures import ThreadPoolExecutor

def test_executor_cm():
    with ThreadPoolExecutor(max_workers=2) as ex:
        ex.submit(lambda: None)
```

**Fix:** join/cancel/terminate with a timeout in a teardown or finalizer (`t.join(timeout=5.0)`, `timer.cancel()`, `p.join(timeout=5.0)`), or use executors as a context manager (`with ThreadPoolExecutor() as ex:`) / call `.shutdown(wait=True)`

---

## G12: env-mutation

*`os.environ` mutation inside a test/fixture without `monkeypatch`.*

**Tier:** `high`  **Cause:** `environment/order-dependence`

Writing directly to `os.environ` mutates process-global environment state
that outlives the function that set it -- the next test to read that key (or
a wholly unrelated test that assumes its *absence*) now depends on whichever
test ran before it in the same worker process. `monkeypatch.setenv`/
`monkeypatch.delenv` give the identical effect scoped to the fixture/test and
restore the prior value through pytest's own teardown, which still fires on a
raised exception or a fixture-level error -- a hand-rolled `try/finally`
inside the test body does not fully replicate that (nothing runs the restore
if the process is interrupted between the mutation and the `try:`).

Mined from a real ML/trading suite: raw `os.environ[...] = ...` and a
hand-rolled save/restore both coexisted densely alongside 27 files' worth of
`monkeypatch.setenv`/`delenv` and `unittest.mock.patch.dict(os.environ, ...)`
-- the true-positive and false-positive shapes below are drawn from that
corpus. Module-level mutation (import-time, before any function runs) is a
different risk shape already covered by G6; this rule only looks inside
function bodies (test functions and fixtures).

**Bad:**

```python
import os

def test_a():
    os.environ["JAX_DISABLE_JIT"] = "1"
```

**Good:**

```python
def test_a(monkeypatch):
    monkeypatch.setenv("FLAG", "1")
```

**Fix:** use `monkeypatch.setenv(...)`/`monkeypatch.delenv(...)` -- restores the prior value through pytest's own teardown, which still runs on a raised exception or fixture error; for a `with` block, `unittest.mock.patch.dict(os.environ, {...})` is the equivalent safe form

---

## M1: unseeded-stochastic-assert

*tight-tolerance assertion on stochastic output with no seed in scope.*

**Tier:** `medium`  **Cause:** `randomness/reproducibility`

`np.testing.assert_allclose`/`assert_array_almost_equal`/`pytest.approx`
compare two values *up to a tolerance*, not exactly -- appropriate when the
values come from floating-point arithmetic, but not a substitute for
reproducibility when one of the compared values is the output of a
stochastic op (`np.random.normal`, `torch.dropout`, `df.sample`, a freshly
initialized model's `.fit`/`.predict`, ...). Without a deterministic
generator seeded in the test (or one of the fixtures it depends on), the
compared value is drawn fresh every run and the assertion's tolerance is
gambling against how far two independent draws can land apart -- it will
pass on most runs and fail on an unlucky one, exactly the "green locally,
red on CI" signature this tool exists to catch. Fix: derive a per-test seed
(`np.random.default_rng(seed)`, `torch.Generator().manual_seed(seed)`,
`jax.random.PRNGKey(seed)`) local to the test or its fixtures.

**Bad:**

```python
import numpy as np

def test_a():
    x = np.random.normal(size=5)
    np.testing.assert_allclose(x, [0.1, 0.2, 0.3, 0.4, 0.5])
```

**Good:**

```python
import numpy as np

def test_d():
    rng = np.random.default_rng(42)
    x = rng.normal(size=5)
    np.testing.assert_allclose(x, [0.1, 0.2, 0.3, 0.4, 0.5])
```

**Fix:** seed a local generator before drawing the compared value -- `rng = np.random.default_rng(seed)`, `torch.Generator().manual_seed(seed)`, or `jax.random.PRNGKey(seed)` -- in the test or the fixture that produces it

---

## M2: jax-prngkey-reuse

*JAX PRNGKey reuse / missing `split()`.*

**Tier:** `high`  **Cause:** `randomness/jax-prng-reuse`

Unlike NumPy's stateful global RNG, `jax.random.PRNGKey`/`key` is a pure value:
calling `jax.random.normal(key, ...)` does not advance any hidden state, so
passing the *same* key to a second `jax.random.*` call reproduces the exact
same draw instead of an independent one. That silently violates the
"each random call is independent" assumption most tests are written against
(two supposedly-uncorrelated samples turn out identical/correlated, masking
real bugs a genuinely independent draw would have caught) -- and it gets
worse when the shared key lives at module scope and several test functions
consume it directly, because whichever test happens to run first quietly
becomes load-bearing for the others' "randomness".

A nastier, historically-real variant of the same family is seeding via
`PRNGKey(hash(some_string))`: Python's builtin `hash()` is salted by
`PYTHONHASHSEED`, which is randomized by default, so a hash-seeded key is
reproducible *within one process* (same-process reruns pass) but silently
differs *across* processes -- exactly the "green locally, red on a different
CI run" signature this tool exists to catch, and it needs no reuse/missing
`split()` at all to be broken. That is the highest-confidence half of this
rule; flagging same-key reuse without an intervening `split()`/`fold_in()`
is the second, more heuristic half (single-function, syntactic dataflow only
-- no cross-call-site aliasing, no loop-unrolling), so specific findings from
that half are downgraded via `finding(..., confidence=...)` where the
evidence is weaker than the hash-seeded case.

Fix: split before reusing (`k1, k2 = jax.random.split(key)`, one consumption
per half) and derive a fresh key per test instead of sharing one across
functions (`jax.random.fold_in(key, <per-test discriminator>)`); never seed
from `hash(...)`.

**Bad:**

```python
import jax

def test_a():
    key = jax.random.PRNGKey(hash("some-run-id"))
    return jax.random.normal(key, (3,))
```

**Good:**

```python
import jax

def test_split_then_use():
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    a = jax.random.normal(k1, (3,))
    b = jax.random.uniform(k2, (3,))
```

**Fix:** split before each additional use -- `k1, k2 = jax.random.split(key)` -- and consume each half once; derive a fresh key per test instead of sharing one module-level key (e.g. `jax.random.fold_in(key, <per-test discriminator>)`); seed from an explicit int, never `PRNGKey(hash(...))` (salted by PYTHONHASHSEED, differs across processes)

---

## M3: suspicious-tolerance

*suspicious tolerance on model-output comparisons.*

**Tier:** `advisory`  **Cause:** `ml-numerics/tolerance`

`assert_allclose`/`np.isclose`/`pytest.approx` calls whose `atol`/`rtol` (or
`abs`/`rel` for `approx`) is <= 1e-7 assume near machine-precision agreement.
That bound is legitimate for deterministic, float64 linear algebra, but
stochastic outputs (dropout, sampling, non-deterministic GPU reduction order)
and float32 model forward passes routinely disagree by more than that
run-to-run — a tolerance this tight on a model/predict/forward/apply/loss/
grad-derived value fails intermittently for reasons unrelated to correctness.
The fix is not "loosen it blindly": derive the bound from an observed
distribution of repeated runs (FLEX FSE-21) rather than copy-pasting a tight
default.

Static analysis cannot see whether the compared values are actually
stochastic — only that the calls that produced them *look* model-related —
so every finding here is advisory, never confirmed. (The rule's confidence
is already the floor of the `Confidence` enum; there is no lower tier to
downgrade individual findings to.)

**Bad:**

```python
import numpy as np
def test_a():
    np.testing.assert_allclose(model.predict(x), expected, atol=1e-8)
```

**Good:**

```python
import numpy as np
def test_a():
    np.testing.assert_allclose(np.dot(a, b), expected, atol=1e-9)
```

**Fix:** justify this bound or derive it from an observed distribution of repeated runs (FLEX FSE-21) instead of copy-pasting a tight default; float32 model/GPU outputs commonly need rtol=1e-5/atol=1e-6 floors

---

## M4: missing-determinism-flags

*missing determinism flags in GPU/accelerator test envs (advisory tier).*

**Tier:** `advisory`  **Cause:** `ml-gpu/nondeterminism`

CUDA kernels for convolution, reduction, and RNN ops are chosen by cuDNN/cuBLAS
autotuning by default -- the fastest kernel for the current shape/hardware is
picked at runtime and can differ between runs, and several reduction kernels
use non-associative floating-point accumulation order that is itself
run-to-run nondeterministic on top of that. PyTorch's opt-in fix is
`torch.use_deterministic_algorithms(True)`, which for some CUDA ops (certain
RNN/CTC paths, `>=10.2` toolkits) additionally requires the
`CUBLAS_WORKSPACE_CONFIG` environment variable to be set before those ops run.
JAX/XLA has the same class of nondeterminism on GPU (kernel autotuning,
non-deterministic reduction order); the dominant workaround observed in
practice is not an XLA determinism flag at all but simply forcing the test
process onto CPU (`JAX_PLATFORMS=cpu`) or noting the relevant `XLA_FLAGS`
explicitly, so this rule accepts either as evidence determinism was
considered.

This is a *test-file-scoped* static signal, not a runtime one: the real
detection surface for whether determinism flags are actually wired into a GPU
test run is CI config / launch scripts, which this rule (an `ast`-only,
single-`FileContext` check, per the plan) cannot see -- and it cannot see a
`conftest.py` in the same directory either, so a project that sets
`torch.use_deterministic_algorithms(True)` centrally in a fixture will still
get flagged per test file. That is exactly why this rule is advisory only:
it is genuine "worth a look" signal, not a confirmed defect. It fires at most
once per file per accelerator (torch/CUDA, JAX), not once per CUDA call site.

**Bad:**

```python
import torch

def test_a():
    model = torch.nn.Linear(2, 2)
    model.cuda()
```

**Good:**

```python
import torch

def test_a():
    torch.use_deterministic_algorithms(True)
    model = torch.nn.Linear(2, 2)
    model.cuda()
```

**Fix:** enable deterministic algorithms in a session-scoped autouse fixture -- `torch.use_deterministic_algorithms(True)` plus `CUBLAS_WORKSPACE_CONFIG` for CUDA, or `JAX_PLATFORMS=cpu`/deterministic `XLA_FLAGS` for JAX -- and document any op left non-deterministic on purpose

---

## M5: module-scope-jit

*JAX `jit()` compiled at module scope, or cached in a module/session fixture.*

**Tier:** `advisory`  **Cause:** `ml-jax/compilation-cache-order`

`jax.jit(...)` applied at module level in a test file — or a module-/session-scoped
pytest fixture that hands out a `jax.jit(...)`-wrapped function — compiles once at
collection/first-use time and shares the compiled callable across every test in the
module for the rest of the session. A JIT'd function is pure, so sharing it is
usually harmless when every caller feeds it the same input shape; the real risk is
compilation-cache order effects (a call site with a new input shape forces a
recompile whose *timing* becomes test-order-dependent) and, on GPU, OOM depending on
which test happened to run — and leave device memory fragmented — first. Static
analysis cannot see call-site input shapes or device placement, so this rule is
advisory only: worth a look, never a hard fail. Fix: compile inside the test or a
function-scoped fixture, or call `jax.clear_caches()` in teardown for heavy suites.

**Bad:**

```python
import jax

@jax.jit
def _step(x):
    return x + 1

def test_step():
    assert _step(1) == 2
```

**Good:**

```python
import jax

def test_local_jit():
    stepped = jax.jit(lambda x: x + 1)
    assert stepped(1) == 2
```

**Fix:** compile inside the test or a function-scoped fixture instead; if recompiling every test is too slow, keep the wider-scoped fixture but call `jax.clear_caches()` in an autouse teardown so compilation-cache and device-memory state don't carry across tests

---
