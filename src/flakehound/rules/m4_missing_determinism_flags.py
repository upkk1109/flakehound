"""M4: missing determinism flags in GPU/accelerator test envs (advisory tier).

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
"""

from __future__ import annotations

import ast
from collections.abc import Iterable

from flakehound.rules.base import Confidence, FileContext, Finding, Rule, register

_JAX_DETERMINISM_ENV_MARKERS = {"JAX_PLATFORMS", "XLA_FLAGS"}


def _dotted(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _imports_torch(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(
            a.name == "torch" or a.name.startswith("torch.") for a in node.names
        ):
            return True
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and (node.module == "torch" or node.module.startswith("torch."))
        ):
            return True
    return False


def _is_cuda_str(value: object) -> bool:
    return isinstance(value, str) and (value == "cuda" or value.startswith("cuda:"))


def _is_cuda_call(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr == "cuda":
        return True  # e.g. `model.cuda()`, `tensor.cuda()`
    if isinstance(func, ast.Attribute) and func.attr == "to" and node.args:
        first = node.args[0]
        if isinstance(first, ast.Constant) and _is_cuda_str(first.value):
            return True  # e.g. `tensor.to("cuda")`
    if _dotted(func) == "torch.device" and node.args:
        first = node.args[0]
        if isinstance(first, ast.Constant) and _is_cuda_str(first.value):
            return True  # e.g. `torch.device("cuda")`
    for kw in node.keywords:
        if (
            kw.arg == "device"
            and isinstance(kw.value, ast.Constant)
            and _is_cuda_str(kw.value.value)
        ):
            return True  # e.g. `tensor.to(device="cuda")`, `Module(..., device="cuda")`
    return False


def _find_cuda_usage(tree: ast.Module) -> ast.Call | None:
    matches = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and _is_cuda_call(n)]
    if not matches:
        return None
    return min(matches, key=lambda n: (n.lineno, n.col_offset))


def _bool_arg_is_false(call: ast.Call) -> bool:
    """True only if the mode argument is the literal `False` (an explicit opt-out)."""
    if call.args:
        first = call.args[0]
        return isinstance(first, ast.Constant) and first.value is False
    for kw in call.keywords:
        if kw.arg in ("mode", None):
            return isinstance(kw.value, ast.Constant) and kw.value.value is False
    return False


def _has_torch_determinism_marker(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == "CUBLAS_WORKSPACE_CONFIG":
            return True
        # A literal `False` is a deliberate opt-out, not evidence determinism was considered
        # as *enabled*; anything else (True, or a runtime-only value we can't evaluate
        # statically) gets the benefit of the doubt.
        if (
            isinstance(node, ast.Call)
            and _dotted(node.func) == "torch.use_deterministic_algorithms"
            and not _bool_arg_is_false(node)
        ):
            return True
    return False


def _find_jax_jit(tree: ast.Module) -> ast.expr | None:
    matches: list[ast.expr] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _dotted(node.func) == "jax.jit":
            matches.append(node)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            for dec in node.decorator_list:
                target = dec.func if isinstance(dec, ast.Call) else dec
                if _dotted(target) == "jax.jit":
                    matches.append(dec)
    if not matches:
        return None
    return min(matches, key=lambda n: (n.lineno, n.col_offset))


def _uses_jax_random(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            dotted = _dotted(node.func)
            if dotted and dotted.startswith("jax.random."):
                return True
    return False


def _has_jax_determinism_marker(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value in _JAX_DETERMINISM_ENV_MARKERS:
            return True
    return False


@register
class MissingDeterminismFlags(Rule):
    id = "M4"
    name = "missing-determinism-flags"
    cause = "ml-gpu/nondeterminism"
    confidence = Confidence.ADVISORY
    fix_suggestion = (
        "enable deterministic algorithms in a session-scoped autouse fixture -- "
        "`torch.use_deterministic_algorithms(True)` plus `CUBLAS_WORKSPACE_CONFIG` "
        "for CUDA, or `JAX_PLATFORMS=cpu`/deterministic `XLA_FLAGS` for JAX -- and "
        "document any op left non-deterministic on purpose"
    )

    def check(self, ctx: FileContext) -> Iterable[Finding]:
        if ctx.is_conftest or not ctx.is_test_file:
            return
        tree = ctx.tree

        if _imports_torch(tree):
            cuda_site = _find_cuda_usage(tree)
            if cuda_site is not None and not _has_torch_determinism_marker(tree):
                yield self.finding(
                    ctx,
                    cuda_site,
                    'CUDA is used (`.cuda()`/`.to("cuda")`/`device="cuda"`) but neither '
                    "`torch.use_deterministic_algorithms(True)` nor `CUBLAS_WORKSPACE_CONFIG` "
                    "appears anywhere in this file; cuDNN/cuBLAS kernel autotuning and "
                    "reduction order can vary run-to-run without them",
                )

        jit_site = _find_jax_jit(tree)
        if (
            jit_site is not None
            and _uses_jax_random(tree)
            and not _has_jax_determinism_marker(tree)
        ):
            yield self.finding(
                ctx,
                jit_site,
                "`jax.jit` is combined with `jax.random` in this file with no "
                "`JAX_PLATFORMS`/`XLA_FLAGS` determinism note anywhere in it; jitted, "
                "randomized computation can vary across runs/devices without one",
            )
