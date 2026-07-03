"""M4 rule tests -- the TP/FP-guard pattern every rule must follow."""

from __future__ import annotations

import ast

from flakehound.rules.base import FileContext
from flakehound.rules.m4_missing_determinism_flags import MissingDeterminismFlags


def _ctx(source: str, name: str = "test_x.py") -> FileContext:
    return FileContext(
        path=name,
        source=source,
        tree=ast.parse(source),
        is_test_file=True,
        is_conftest=False,
    )


def _conftest_ctx(source: str, name: str = "conftest.py") -> FileContext:
    return FileContext(
        path=name,
        source=source,
        tree=ast.parse(source),
        is_test_file=False,
        is_conftest=True,
    )


def _run(source: str):
    return list(MissingDeterminismFlags().check(_ctx(source)))


def test_detects_torch_cuda_call_without_determinism_flag():
    src = "import torch\n\ndef test_a():\n    model = torch.nn.Linear(2, 2)\n    model.cuda()\n"
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].rule_id == "M4"
    assert findings[0].line == 5


def test_detects_torch_to_cuda_string_without_determinism_flag():
    src = 'import torch\n\ndef test_b():\n    x = torch.randn(4)\n    y = x.to("cuda")\n'
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].rule_id == "M4"
    assert findings[0].line == 5


def test_detects_jax_jit_plus_random_without_determinism_note():
    src = (
        "import jax\n"
        "\n"
        "@jax.jit\n"
        "def step(x):\n"
        "    return x\n"
        "\n"
        "def test_c():\n"
        "    key = jax.random.PRNGKey(0)\n"
        "    step(key)\n"
    )
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].rule_id == "M4"
    assert findings[0].line == 3


def test_flags_once_per_file_not_per_cuda_call_site():
    src = (
        "import torch\n"
        "\n"
        "def test_multi():\n"
        "    a = torch.randn(2).cuda()\n"
        "    b = torch.randn(2).cuda()\n"
        "    c = torch.randn(2).cuda()\n"
    )
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].line == 4


def test_fp_guard_use_deterministic_algorithms_present():
    src = (
        "import torch\n"
        "\n"
        "def test_a():\n"
        "    torch.use_deterministic_algorithms(True)\n"
        "    model = torch.nn.Linear(2, 2)\n"
        "    model.cuda()\n"
    )
    assert _run(src) == []


def test_fp_guard_cublas_workspace_config_present():
    src = (
        "import os\n"
        "import torch\n"
        "\n"
        'os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")\n'
        "\n"
        "def test_a():\n"
        "    model = torch.nn.Linear(2, 2)\n"
        "    model.cuda()\n"
    )
    assert _run(src) == []


def test_fp_guard_torch_cpu_only_no_cuda_usage():
    src = 'import torch\n\ndef test_a():\n    x = torch.randn(4)\n    y = x.to("cpu")\n'
    assert _run(src) == []


def test_fp_guard_jax_platforms_marker_present():
    src = (
        "import os\n"
        "import jax\n"
        "\n"
        'os.environ.setdefault("JAX_PLATFORMS", "cpu")\n'
        "\n"
        "@jax.jit\n"
        "def step(x):\n"
        "    return x\n"
        "\n"
        "def test_c():\n"
        "    key = jax.random.PRNGKey(0)\n"
        "    step(key)\n"
    )
    assert _run(src) == []


def test_fp_guard_jax_jit_without_random_is_clean():
    src = "import jax\n\n@jax.jit\ndef step(x):\n    return x + 1\n\ndef test_c():\n    step(1)\n"
    assert _run(src) == []


def test_fp_guard_conftest_is_never_flagged():
    src = "import torch\n\ndef cuda_fixture():\n    return torch.randn(2).cuda()\n"
    assert list(MissingDeterminismFlags().check(_conftest_ctx(src))) == []
