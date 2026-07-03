"""Unit tests for the `_imports` alias-resolution helper (pack D1)."""

from __future__ import annotations

import ast

from flakehound.rules._imports import (
    build_alias_map,
    dotted_name,
    resolve_call,
    resolve_dotted,
    resolve_expr,
)


def _tree(source: str) -> ast.Module:
    return ast.parse(source)


def _first_call(source: str) -> ast.Call:
    tree = _tree(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            return node
    raise AssertionError(f"no Call found in: {source!r}")


# --- dotted_name --------------------------------------------------------------


def test_dotted_name_simple_attribute_chain():
    expr = ast.parse("a.b.c", mode="eval").body
    assert dotted_name(expr) == "a.b.c"


def test_dotted_name_bare_name():
    expr = ast.parse("jit", mode="eval").body
    assert dotted_name(expr) == "jit"


def test_dotted_name_none_for_call_result():
    expr = ast.parse("get_rng().seed", mode="eval").body
    assert dotted_name(expr) is None


# --- build_alias_map -----------------------------------------------------------


def test_import_submodule_as_alias():
    aliases = build_alias_map(_tree("import numpy.random as nr\n"))
    assert aliases["nr"] == "numpy.random"


def test_import_submodule_no_alias_binds_top_package_identity():
    aliases = build_alias_map(_tree("import numpy.random\n"))
    assert aliases["numpy"] == "numpy"


def test_import_dotted_with_alias():
    aliases = build_alias_map(_tree("import a.b.c as x\n"))
    assert aliases["x"] == "a.b.c"


def test_from_import_with_alias():
    aliases = build_alias_map(_tree("from torch import manual_seed as ms\n"))
    assert aliases["ms"] == "torch.manual_seed"


def test_from_import_without_alias():
    aliases = build_alias_map(_tree("from torch import manual_seed\n"))
    assert aliases["manual_seed"] == "torch.manual_seed"


def test_from_import_random_as_jr():
    aliases = build_alias_map(_tree("from jax import random as jr\n"))
    assert aliases["jr"] == "jax.random"


def test_from_import_jit_no_alias():
    aliases = build_alias_map(_tree("from jax import jit\n"))
    assert aliases["jit"] == "jax.jit"


def test_relative_import_is_skipped():
    aliases = build_alias_map(_tree("from . import helpers\n"))
    assert aliases == {}


def test_star_import_is_skipped():
    aliases = build_alias_map(_tree("from os.path import *\n"))
    assert aliases == {}


def test_import_inside_function_is_seen():
    src = "def test_a():\n    import numpy.random as nr\n    nr.seed(0)\n"
    aliases = build_alias_map(_tree(src))
    assert aliases["nr"] == "numpy.random"


# --- resolve_dotted --------------------------------------------------------------


def test_resolve_dotted_expands_leading_alias():
    assert resolve_dotted("nr.seed", {"nr": "numpy.random"}) == "numpy.random.seed"


def test_resolve_dotted_bare_alias_no_rest():
    assert resolve_dotted("ms", {"ms": "torch.manual_seed"}) == "torch.manual_seed"


def test_resolve_dotted_unknown_head_unchanged():
    assert resolve_dotted("planter.seed", {"nr": "numpy.random"}) == "planter.seed"


# --- resolve_expr / resolve_call -------------------------------------------------


def test_resolve_call_aliased_attribute_chain():
    call = _first_call("nr.seed(0)")
    assert resolve_call(call, {"nr": "numpy.random"}) == "numpy.random.seed"


def test_resolve_call_aliased_bare_name():
    call = _first_call("ms(0)")
    assert resolve_call(call, {"ms": "torch.manual_seed"}) == "torch.manual_seed"


def test_resolve_call_unaliased_passthrough():
    call = _first_call("planter.seed(3)")
    assert resolve_call(call, {}) == "planter.seed"


def test_resolve_call_none_for_non_dotted_func():
    call = _first_call("get_rng().seed(0)")
    assert resolve_call(call, {}) is None


def test_resolve_expr_bare_decorator_name():
    expr = ast.parse("jit", mode="eval").body
    assert resolve_expr(expr, {"jit": "jax.jit"}) == "jax.jit"


def test_resolve_expr_unknown_name_unchanged():
    expr = ast.parse("something_else", mode="eval").body
    assert resolve_expr(expr, {"jit": "jax.jit"}) == "something_else"
