"""M5 rule tests — the TP/FP-guard pattern every rule must follow."""

from __future__ import annotations

import ast

from flakehound.rules.base import Confidence, FileContext
from flakehound.rules.m5_module_scope_jit import ModuleScopeJit


def _ctx(source: str, name: str = "test_x.py") -> FileContext:
    return FileContext(
        path=name,
        source=source,
        tree=ast.parse(source),
        is_test_file=name != "conftest.py",
        is_conftest=name == "conftest.py",
    )


def _run(source: str, name: str = "test_x.py"):
    return list(ModuleScopeJit().check(_ctx(source, name)))


# --- true positives ---------------------------------------------------------


def test_detects_module_level_jit_decorator():
    src = (
        "import jax\n"
        "\n"
        "@jax.jit\n"
        "def _step(x):\n"
        "    return x + 1\n"
        "\n"
        "def test_step():\n"
        "    assert _step(1) == 2\n"
    )
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].rule_id == "M5"
    assert findings[0].confidence == Confidence.ADVISORY
    assert findings[0].line == 3


def test_detects_module_level_jit_assignment():
    src = (
        "import jax\n"
        "\n"
        "def _step_fn(x):\n"
        "    return x + 1\n"
        "\n"
        "stepped = jax.jit(_step_fn)\n"
        "\n"
        "def test_stepped():\n"
        "    assert stepped(1) == 2\n"
    )
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].rule_id == "M5"
    assert findings[0].line == 6


def test_detects_module_scoped_fixture_caching_jit():
    src = (
        "import jax\n"
        "import pytest\n"
        "\n"
        '@pytest.fixture(scope="module")\n'
        "def jitted_step():\n"
        "    return jax.jit(lambda x: x + 1)\n"
        "\n"
        "def test_jitted_step(jitted_step):\n"
        "    assert jitted_step(1) == 2\n"
    )
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].rule_id == "M5"
    assert findings[0].line == 6


def test_detects_session_scoped_fixture_in_conftest():
    src = (
        "import jax\n"
        "import pytest\n"
        "\n"
        '@pytest.fixture(scope="session")\n'
        "def compiled_step():\n"
        "    fn = jax.jit(lambda x: x + 1)\n"
        "    return fn\n"
    )
    findings = _run(src, name="conftest.py")
    assert len(findings) == 1
    assert findings[0].line == 6


# --- import-alias resolution (pack D1) ---------------------------------------


def test_detects_from_import_jit_bare_decorator():
    src = (
        "from jax import jit\n"
        "\n"
        "@jit\n"
        "def _step(x):\n"
        "    return x + 1\n"
        "\n"
        "def test_step():\n"
        "    assert _step(1) == 2\n"
    )
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].rule_id == "M5"
    assert findings[0].line == 3


def test_detects_from_import_jit_alias_assignment():
    src = (
        "from jax import jit as jj\n"
        "\n"
        "def _step_fn(x):\n"
        "    return x + 1\n"
        "\n"
        "stepped = jj(_step_fn)\n"
        "\n"
        "def test_stepped():\n"
        "    assert stepped(1) == 2\n"
    )
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].line == 6


def test_detects_functools_partial_jit_decorator():
    src = (
        "import functools\n"
        "import jax\n"
        "\n"
        "@functools.partial(jax.jit, static_argnums=0)\n"
        "def _step(n, x):\n"
        "    return x + n\n"
        "\n"
        "def test_step():\n"
        "    assert _step(1, 1) == 2\n"
    )
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].rule_id == "M5"
    assert findings[0].line == 4


def test_detects_functools_partial_jit_decorator_via_from_import():
    src = (
        "from functools import partial\n"
        "from jax import jit\n"
        "\n"
        "@partial(jit, static_argnums=0)\n"
        "def _step(n, x):\n"
        "    return x + n\n"
        "\n"
        "def test_step():\n"
        "    assert _step(1, 1) == 2\n"
    )
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].line == 4


# --- false-positive guards ---------------------------------------------------


def test_fp_guard_jit_inside_test_function_body_is_clean():
    src = (
        "import jax\n"
        "\n"
        "def test_local_jit():\n"
        "    stepped = jax.jit(lambda x: x + 1)\n"
        "    assert stepped(1) == 2\n"
    )
    assert _run(src) == []


def test_fp_guard_jit_inside_function_scoped_fixture_is_clean():
    src = (
        "import jax\n"
        "import pytest\n"
        "\n"
        "@pytest.fixture\n"
        "def jitted_step():\n"
        "    return jax.jit(lambda x: x + 1)\n"
        "\n"
        "def test_jitted_step(jitted_step):\n"
        "    assert jitted_step(1) == 2\n"
    )
    assert _run(src) == []


def test_fp_guard_explicit_function_scope_fixture_is_clean():
    src = (
        "import jax\n"
        "import pytest\n"
        "\n"
        '@pytest.fixture(scope="function")\n'
        "def jitted_step():\n"
        "    return jax.jit(lambda x: x + 1)\n"
    )
    assert _run(src) == []


def test_fp_guard_nested_jit_decorator_inside_test_body_is_clean():
    src = (
        "import jax\n"
        "\n"
        "def test_nested():\n"
        "    @jax.jit\n"
        "    def _step(x):\n"
        "        return x + 1\n"
        "\n"
        "    assert _step(1) == 2\n"
    )
    assert _run(src) == []


def test_fp_guard_class_scoped_fixture_is_clean():
    src = (
        "import jax\n"
        "import pytest\n"
        "\n"
        '@pytest.fixture(scope="class")\n'
        "def jitted_step():\n"
        "    return jax.jit(lambda x: x + 1)\n"
    )
    assert _run(src) == []


def test_fp_guard_unrelated_partial_is_clean():
    """`functools.partial` wrapping something other than `jax.jit` must not fire."""
    src = (
        "import functools\n"
        "\n"
        "def _add(a, b):\n"
        "    return a + b\n"
        "\n"
        "@functools.partial(_add, 1)\n"
        "def test_a():\n"
        "    pass\n"
    )
    assert _run(src) == []


def test_fp_guard_bare_partial_assignment_not_applied_is_clean():
    """`functools.partial(jax.jit, ...)` stored but never applied to a function
    has not compiled anything yet -- only the decorator form (immediately
    applied by Python) should fire."""
    src = (
        "import functools\n"
        "import jax\n"
        "\n"
        "make_jitted = functools.partial(jax.jit, static_argnums=0)\n"
        "\n"
        "def test_a():\n"
        "    stepped = make_jitted(lambda n, x: x + n)\n"
        "    assert stepped(1, 1) == 2\n"
    )
    assert _run(src) == []


def test_fp_guard_unrelated_jit_named_function_via_alias_is_clean():
    """A locally-defined `jit` name (no `from jax import jit`) must not be
    mistaken for jax's jit."""
    src = (
        "def jit(fn):\n"
        "    return fn\n"
        "\n"
        "@jit\n"
        "def _step(x):\n"
        "    return x + 1\n"
        "\n"
        "def test_step():\n"
        "    assert _step(1) == 2\n"
    )
    assert _run(src) == []
