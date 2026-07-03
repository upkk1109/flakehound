"""M3 rule tests — the TP/FP-guard pattern every rule must follow."""

from __future__ import annotations

import ast

from flakehound.rules.base import FileContext
from flakehound.rules.m3_suspicious_tolerance import SuspiciousTolerance


def _ctx(source: str, name: str = "test_x.py") -> FileContext:
    return FileContext(
        path=name,
        source=source,
        tree=ast.parse(source),
        is_test_file=True,
        is_conftest=False,
    )


def _run(source: str):
    return list(SuspiciousTolerance().check(_ctx(source)))


# --- true positives ---------------------------------------------------------


def test_detects_tight_atol_on_model_predict():
    src = (
        "import numpy as np\n"
        "def test_a():\n"
        "    np.testing.assert_allclose(model.predict(x), expected, atol=1e-8)\n"
    )
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].rule_id == "M3"
    assert findings[0].confidence.value == "advisory"
    assert findings[0].line == 3


def test_detects_tight_rtol_on_np_isclose_forward():
    src = (
        "import numpy as np\n"
        "def test_a():\n"
        "    assert np.isclose(model.forward(x), target, rtol=1e-8)\n"
    )
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].line == 3


def test_detects_tight_abs_on_pytest_approx():
    src = (
        "import pytest\n"
        "def test_a():\n"
        "    assert model.predict(x) == pytest.approx(target, abs=1e-9)\n"
    )
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].line == 3


def test_detects_loss_and_grad_derived_operands():
    src = (
        "import numpy as np\n"
        "def test_a():\n"
        "    np.testing.assert_allclose(compute_loss(pred, y), 0.0, atol=1e-9)\n"
        "    np.testing.assert_allclose(jax.grad(f)(params), expected_grad, atol=1e-9)\n"
    )
    assert len(_run(src)) == 2


# --- false-positive guards ---------------------------------------------------


def test_fp_guard_pure_numpy_deterministic_math_is_clean():
    src = (
        "import numpy as np\n"
        "def test_a():\n"
        "    np.testing.assert_allclose(np.dot(a, b), expected, atol=1e-9)\n"
    )
    assert _run(src) == []


def test_fp_guard_explicit_float64_cast_is_clean():
    src = (
        "import numpy as np\n"
        "def test_a():\n"
        "    np.testing.assert_allclose(\n"
        "        model.predict(x).astype(np.float64), expected, atol=1e-9\n"
        "    )\n"
    )
    assert _run(src) == []


def test_fp_guard_exact_integer_array_is_clean():
    src = (
        "import numpy as np\n"
        "def test_a():\n"
        "    np.testing.assert_allclose(\n"
        "        model.predict(x).astype(np.int64), expected, atol=1e-9\n"
        "    )\n"
    )
    assert _run(src) == []


def test_fp_guard_rtol_zero_means_disabled_not_tight():
    # atol=1e-5 is not suspicious on its own; rtol=0 explicitly disables the
    # relative bound rather than making it "suspiciously tight."
    src = (
        "import numpy as np\n"
        "def test_a():\n"
        "    np.testing.assert_allclose(model.predict(x), expected, atol=1e-5, rtol=0)\n"
    )
    assert _run(src) == []


def test_fp_guard_loose_tolerance_on_model_output_is_clean():
    src = (
        "import numpy as np\n"
        "def test_a():\n"
        "    np.testing.assert_allclose(model.predict(x), expected, rtol=1e-5, atol=1e-6)\n"
    )
    assert _run(src) == []
