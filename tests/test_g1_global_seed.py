"""G1 rule tests — the TP/FP-guard pattern every rule must follow."""

from __future__ import annotations

import ast

from flakehound.rules.base import FileContext
from flakehound.rules.g1_global_seed import GlobalSeedMutation


def _ctx(source: str, name: str = "test_x.py") -> FileContext:
    return FileContext(
        path=name,
        source=source,
        tree=ast.parse(source),
        is_test_file=True,
        is_conftest=False,
    )


def _run(source: str):
    return list(GlobalSeedMutation().check(_ctx(source)))


def test_detects_np_random_seed():
    findings = _run("import numpy as np\n\ndef test_a():\n    np.random.seed(42)\n")
    assert len(findings) == 1
    assert findings[0].rule_id == "G1"
    assert findings[0].line == 4


def test_detects_stdlib_random_seed_and_torch():
    src = "import random, torch\ndef test_a():\n    random.seed(0)\n    torch.manual_seed(0)\n"
    assert len(_run(src)) == 2


def test_fp_guard_local_generator_is_clean():
    src = (
        "import numpy as np\n"
        "def test_a():\n"
        "    rng = np.random.default_rng(42)\n"
        "    x = rng.normal()\n"
    )
    assert _run(src) == []


def test_fp_guard_unrelated_seed_method_on_object():
    src = "def test_a(planter):\n    planter.seed(3)\n"
    assert _run(src) == []


# --- import-alias resolution (pack D1) ----------------------------------------


def test_detects_aliased_submodule_import():
    src = "import numpy.random as nr\n\ndef test_a():\n    nr.seed(0)\n"
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].rule_id == "G1"
    assert findings[0].line == 4


def test_detects_aliased_from_import_call():
    src = "from torch import manual_seed as ms\n\ndef test_a():\n    ms(0)\n"
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].rule_id == "G1"
    assert findings[0].line == 4


def test_detects_bare_from_import_call():
    src = "from torch import manual_seed\n\ndef test_a():\n    manual_seed(0)\n"
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].line == 4


def test_fp_guard_aliased_import_unchanged_planter():
    """Alias resolution must not turn unrelated names into false matches."""
    src = "import numpy.random as nr\n\ndef test_a(planter):\n    planter.seed(3)\n"
    assert _run(src) == []


def test_fp_guard_local_function_named_manual_seed_does_not_fire():
    """A locally-defined `manual_seed` (no import at all) must not be treated as
    torch's global-seed mutator -- it's an unrelated function that happens to
    share a name."""
    src = "def manual_seed(x):\n    return x + 1\n\ndef test_a():\n    assert manual_seed(1) == 2\n"
    assert _run(src) == []


def test_fp_guard_endswith_random_no_longer_matches_arbitrary_local():
    """`my_random` merely ends with 'random' but is not a known RNG module --
    the old `str.endswith('random')` fallback used to false-positive here
    (review low #42); the restricted allowlist must not."""
    src = "def test_a(my_random):\n    my_random.seed(1)\n"
    assert _run(src) == []
