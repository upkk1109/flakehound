"""G8 rule tests — the TP/FP-guard pattern every rule must follow."""

from __future__ import annotations

import ast

from flakehound.rules.base import Confidence, FileContext
from flakehound.rules.g8_float_equality import FloatEqualityWithoutTolerance


def _ctx(source: str, name: str = "test_x.py") -> FileContext:
    return FileContext(
        path=name,
        source=source,
        tree=ast.parse(source),
        is_test_file=True,
        is_conftest=False,
    )


def _run(source: str):
    return list(FloatEqualityWithoutTolerance().check(_ctx(source)))


def test_detects_float_literal_equality():
    src = "def test_a():\n    result = compute()\n    assert result == 3.14\n"
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].rule_id == "G8"
    assert findings[0].line == 3
    assert findings[0].confidence == Confidence.HIGH


def test_detects_division_result_equality():
    src = "def test_a():\n    assert a / b == expected\n"
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].line == 2


def test_detects_math_call_equality():
    src = "import math\ndef test_a():\n    assert math.sqrt(2) == 1.4142135623730951\n"
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].line == 3


def test_detects_np_call_equality():
    src = "import numpy as np\ndef test_a():\n    assert np.mean(values) == expected\n"
    findings = _run(src)
    assert len(findings) == 1


def test_detects_assert_equal_unittest_style():
    src = "class T:\n    def test_a(self):\n        self.assertEqual(compute_ratio(), 0.3333333)\n"
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].line == 3


def test_fully_static_literal_comparison_is_downgraded_to_medium():
    src = "def test_a():\n    assert 3.14 == 3.14\n"
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].confidence == Confidence.MEDIUM


def test_fp_guard_pytest_approx_present():
    src = "import pytest\ndef test_a():\n    assert compute() == pytest.approx(3.14)\n"
    assert _run(src) == []


def test_fp_guard_bare_approx_import():
    src = "from pytest import approx\ndef test_a():\n    assert compute() == approx(0.1 + 0.2)\n"
    assert _run(src) == []


def test_fp_guard_integers_are_clean():
    src = "def test_a():\n    assert compute_count() == 5\n"
    assert _run(src) == []


def test_fp_guard_sentinel_zero_compared_to_identical_literal():
    src = "def test_a():\n    assert 0.0 == 0.0\n"
    assert _run(src) == []


def test_fp_guard_sentinel_half_compared_to_identical_literal():
    src = "def test_a():\n    assert 0.5 == 0.5\n"
    assert _run(src) == []


def test_fp_guard_math_isclose_call_itself_not_flagged():
    src = "import math\ndef test_a():\n    assert math.isclose(a, b)\n"
    assert _run(src) == []


def test_fp_guard_integer_valued_float_literal():
    src = "def test_a():\n    assert count == 3.0\n"
    assert _run(src) == []


def test_constant_folded_binop_both_sides_downgraded_to_medium():
    src = "def test_a():\n    assert 0.1 + 0.2 == 0.3\n"
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].confidence == Confidence.MEDIUM
    assert "constant" in findings[0].message.lower()


def test_detects_var_var_float_equality_review_repro():
    src = "def test_a():\n    x = compute()\n    y = 0.3 / 3\n    assert x == y\n"
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].line == 4
    assert findings[0].confidence == Confidence.MEDIUM


def test_detects_var_var_float_equality_from_fractional_literals():
    src = "def test_a():\n    x = 1.234\n    y = 5.678\n    assert x == y\n"
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].confidence == Confidence.MEDIUM


def test_detects_division_assigned_names_both_sides():
    src = "def test_a():\n    x = a / b\n    y = c / d\n    assert x == y\n"
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].confidence == Confidence.MEDIUM


def test_detects_propagated_tracked_name():
    src = "def test_a():\n    x = 1.234\n    y = x\n    z = compute()\n    assert y == z\n"
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].confidence == Confidence.MEDIUM


def test_var_var_direct_evidence_on_one_side_stays_high():
    src = "def test_a():\n    x = 1.234\n    assert x == 3.14\n"
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].confidence == Confidence.HIGH


def test_fp_guard_int_assigned_names():
    src = "def test_a():\n    x = 5\n    y = 5\n    assert x == y\n"
    assert _run(src) == []


def test_fp_guard_str_assigned_names():
    src = 'def test_a():\n    x = "abc"\n    y = "abc"\n    assert x == y\n'
    assert _run(src) == []


def test_fp_guard_approx_present_with_tracked_var():
    src = "import pytest\ndef test_a():\n    x = 1.234\n    assert x == pytest.approx(expected)\n"
    assert _run(src) == []


def test_fp_guard_reassignment_to_non_float_clears_tracking():
    src = "def test_a():\n    x = 1.234\n    x = 5\n    y = compute()\n    assert x == y\n"
    assert _run(src) == []
