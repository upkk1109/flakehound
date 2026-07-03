"""M2 rule tests -- the TP/FP-guard pattern every rule must follow."""

from __future__ import annotations

import ast

from flakehound.rules.base import Confidence, FileContext
from flakehound.rules.m2_jax_prngkey_reuse import PRNGKeyReuse


def _ctx(source: str, name: str = "test_x.py") -> FileContext:
    return FileContext(
        path=name,
        source=source,
        tree=ast.parse(source),
        is_test_file=True,
        is_conftest=False,
    )


def _run(source: str, name: str = "test_x.py"):
    return list(PRNGKeyReuse().check(_ctx(source, name)))


# --- true positives ----------------------------------------------------------


def test_detects_hash_seeded_key():
    src = (
        "import jax\n"
        "\n"
        "def test_a():\n"
        '    key = jax.random.PRNGKey(hash("some-run-id"))\n'
        "    return jax.random.normal(key, (3,))\n"
    )
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].rule_id == "M2"
    assert findings[0].confidence == Confidence.HIGH
    assert findings[0].line == 4


def test_detects_key_reused_across_two_calls_without_split():
    src = (
        "import jax\n"
        "\n"
        "def test_two_draws():\n"
        "    key = jax.random.PRNGKey(0)\n"
        "    a = jax.random.normal(key, (3,))\n"
        "    b = jax.random.uniform(key, (3,))\n"
    )
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].rule_id == "M2"
    assert findings[0].confidence == Confidence.HIGH
    assert findings[0].line == 6


def test_detects_module_scope_key_reused_across_test_functions():
    src = (
        "import jax\n"
        "\n"
        "key = jax.random.PRNGKey(0)\n"
        "\n"
        "def test_a():\n"
        "    return jax.random.normal(key, (3,))\n"
        "\n"
        "def test_b():\n"
        "    return jax.random.uniform(key, (3,))\n"
    )
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].rule_id == "M2"
    assert findings[0].confidence == Confidence.MEDIUM
    assert findings[0].line == 3


def test_detects_missing_split_after_use_in_third_call():
    src = (
        "import jax\n"
        "\n"
        "def test_three_draws():\n"
        "    key = jax.random.PRNGKey(1)\n"
        "    a = jax.random.normal(key, (3,))\n"
        "    b = jax.random.normal(key, (3,))\n"
        "    c = jax.random.normal(key, (3,))\n"
    )
    findings = _run(src)
    # second and third consumption each fire against the still-consumed key.
    assert len(findings) == 2
    assert [f.line for f in findings] == [6, 7]


# --- false-positive guards -----------------------------------------------------


def test_fp_guard_split_before_reuse_is_clean():
    src = (
        "import jax\n"
        "\n"
        "def test_split_then_use():\n"
        "    key = jax.random.PRNGKey(0)\n"
        "    k1, k2 = jax.random.split(key)\n"
        "    a = jax.random.normal(k1, (3,))\n"
        "    b = jax.random.uniform(k2, (3,))\n"
    )
    assert _run(src) == []


def test_fp_guard_fold_in_derivation_is_clean():
    src = (
        "import jax\n"
        "\n"
        "def test_fold_in():\n"
        "    key = jax.random.PRNGKey(0)\n"
        "    k1 = jax.random.fold_in(key, 1)\n"
        "    k2 = jax.random.fold_in(key, 2)\n"
        "    a = jax.random.normal(k1, (3,))\n"
        "    b = jax.random.normal(k2, (3,))\n"
    )
    assert _run(src) == []


def test_fp_guard_distinct_keys_is_clean():
    src = (
        "import jax\n"
        "\n"
        "def test_distinct():\n"
        "    key1 = jax.random.PRNGKey(0)\n"
        "    key2 = jax.random.PRNGKey(1)\n"
        "    a = jax.random.normal(key1, (3,))\n"
        "    b = jax.random.normal(key2, (3,))\n"
    )
    assert _run(src) == []


def test_fp_guard_single_use_is_clean():
    src = (
        "import jax\n"
        "\n"
        "def test_single():\n"
        "    key = jax.random.PRNGKey(0)\n"
        "    a = jax.random.normal(key, (3,))\n"
        "    return a\n"
    )
    assert _run(src) == []


def test_fp_guard_module_key_derived_per_test_is_clean():
    src = (
        "import jax\n"
        "\n"
        "key = jax.random.PRNGKey(0)\n"
        "\n"
        "def test_a():\n"
        "    k = jax.random.fold_in(key, 1)\n"
        "    return jax.random.normal(k, (3,))\n"
        "\n"
        "def test_b():\n"
        "    k = jax.random.fold_in(key, 2)\n"
        "    return jax.random.normal(k, (3,))\n"
    )
    assert _run(src) == []


def test_fp_guard_module_key_used_in_only_one_test_is_clean():
    src = (
        "import jax\n"
        "\n"
        "key = jax.random.PRNGKey(0)\n"
        "\n"
        "def test_a():\n"
        "    return jax.random.normal(key, (3,))\n"
    )
    assert _run(src) == []


def test_fp_guard_plain_int_seed_is_clean():
    src = (
        "import jax\n"
        "\n"
        "def test_a():\n"
        "    key = jax.random.PRNGKey(42)\n"
        "    return jax.random.normal(key, (3,))\n"
    )
    assert _run(src) == []


# --- import-alias resolution (pack D1) ------------------------------------------


def test_detects_key_reused_via_jax_random_alias():
    src = (
        "import jax.random as jr\n"
        "\n"
        "def test_two_draws():\n"
        "    key = jr.PRNGKey(0)\n"
        "    a = jr.normal(key, (3,))\n"
        "    b = jr.uniform(key, (3,))\n"
    )
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].rule_id == "M2"
    assert findings[0].confidence == Confidence.HIGH
    assert findings[0].line == 6


def test_detects_hash_seeded_key_via_from_import_alias():
    src = (
        "from jax import random as jr\n"
        "\n"
        "def test_a():\n"
        '    key = jr.PRNGKey(hash("some-run-id"))\n'
        "    return jr.normal(key, (3,))\n"
    )
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].confidence == Confidence.HIGH
    assert findings[0].line == 4


def test_fp_guard_split_before_reuse_is_clean_via_alias():
    src = (
        "import jax.random as jr\n"
        "\n"
        "def test_split_then_use():\n"
        "    key = jr.PRNGKey(0)\n"
        "    k1, k2 = jr.split(key)\n"
        "    a = jr.normal(k1, (3,))\n"
        "    b = jr.uniform(k2, (3,))\n"
    )
    assert _run(src) == []
