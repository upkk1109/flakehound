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
