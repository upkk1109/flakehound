"""G7: un-mocked network calls in tests.

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
"""

from __future__ import annotations

import ast
from collections.abc import Iterable

from flakehound.rules.base import Confidence, FileContext, Finding, Rule, register

# (dotted-prefix, attribute) pairs that perform network I/O in a single call.
_HIGH_NETWORK_CALLS = {
    ("requests", "get"),
    ("requests", "post"),
    ("requests", "put"),
    ("requests", "delete"),
    ("requests", "patch"),
    ("requests", "head"),
    ("requests", "options"),
    ("requests", "request"),
    ("httpx", "get"),
    ("httpx", "post"),
    ("httpx", "put"),
    ("httpx", "delete"),
    ("httpx", "patch"),
    ("httpx", "head"),
    ("httpx", "options"),
    ("httpx", "request"),
    ("httpx", "stream"),
    ("urllib.request", "urlopen"),
    ("request", "urlopen"),  # `from urllib import request; request.urlopen(...)`
    ("socket", "create_connection"),
}

# Constructing a client/session isn't itself a network call — it's weaker static
# evidence (the object might never be used, or might be handed to a mock transport),
# so these are reported at MEDIUM rather than the rule's default HIGH.
_MEDIUM_NETWORK_CALLS = {
    ("httpx", "Client"),
    ("httpx", "AsyncClient"),
    ("aiohttp", "ClientSession"),
}

_MOCK_LIBS = {
    "responses",
    "requests_mock",
    "respx",
    "aioresponses",
    "vcr",
    "pytest_httpserver",
    "pytest_localserver",
}

_MOCK_FIXTURE_PARAMS = {
    "httpserver",
    "httpserver_ssl",
    "make_httpserver",
    "webserver",
    "local_server",
    "localserver",
    "respx_mock",
    "requests_mock",
    "vcr_cassette",
    "vcr_cassette_dir",
}

_NETWORK_KEYWORDS = (
    "requests",
    "httpx",
    "urllib",
    "urlopen",
    "socket",
    "aiohttp",
    "clientsession",
    "create_connection",
)

_LOCALHOST_MARKERS = ("localhost", "127.0.0.1")


def _dotted(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _is_localhost_call(node: ast.Call) -> bool:
    values: list[ast.expr] = list(node.args) + [kw.value for kw in node.keywords]
    for value in values:
        if (
            isinstance(value, ast.Constant)
            and isinstance(value.value, str)
            and any(marker in value.value for marker in _LOCALHOST_MARKERS)
        ):
            return True
        if isinstance(value, (ast.Tuple, ast.List)):
            for elt in value.elts:
                if (
                    isinstance(elt, ast.Constant)
                    and isinstance(elt.value, str)
                    and any(marker in elt.value for marker in _LOCALHOST_MARKERS)
                ):
                    return True
    return False


def _imported_top_level_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def _imports_urlopen(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "urllib.request"
            and any(alias.name == "urlopen" for alias in node.names)
        ):
            return True
    return False


def _has_mock_fixture_params(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            all_args = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
            if any(a.arg in _MOCK_FIXTURE_PARAMS for a in all_args):
                return True
    return False


def _mentions_network(node: ast.AST | None) -> bool:
    if node is None:
        return False
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        low = node.value.lower()
        return any(kw in low for kw in _NETWORK_KEYWORDS)
    dotted = _dotted(node)
    if dotted:
        low = dotted.lower()
        return any(kw in low for kw in _NETWORK_KEYWORDS)
    return False


def _is_patch_like_call(dotted: str | None) -> bool:
    if dotted is None:
        return False
    tail = dotted.rsplit(".", 1)[-1]
    if tail in {"patch", "object"} and "patch" in dotted:
        return True
    return tail in {"setattr", "setitem"} and "monkeypatch" in dotted


def _has_network_patch(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not _is_patch_like_call(_dotted(node.func)):
            continue
        candidates = list(node.args[:2]) + [kw.value for kw in node.keywords]
        if any(_mentions_network(c) for c in candidates):
            return True
    return False


def _has_mock_signals(ctx: FileContext) -> bool:
    tree = ctx.tree
    if _imported_top_level_names(tree) & _MOCK_LIBS:
        return True
    if _has_mock_fixture_params(tree):
        return True
    return bool(_has_network_patch(tree))


@register
class UnmockedNetworkCall(Rule):
    id = "G7"
    name = "unmocked-network"
    cause = "network/infrastructure"
    confidence = Confidence.HIGH
    fix_suggestion = (
        "mock the transport boundary — `responses`/`respx`/`requests_mock`/"
        "`aioresponses` for the client library in use, or a `pytest-httpserver`/"
        "localserver fixture for a real socket target — never hit the live network "
        "from a test"
    )

    def check(self, ctx: FileContext) -> Iterable[Finding]:
        if _has_mock_signals(ctx):
            return
        for node in ast.walk(ctx.tree):
            if not isinstance(node, ast.Call):
                continue
            if _is_localhost_call(node):
                continue
            if isinstance(node.func, ast.Name) and node.func.id == "urlopen":
                confirmed = _imports_urlopen(ctx.tree)
                yield self.finding(
                    ctx,
                    node,
                    "`urlopen(...)` hits the network directly; no mocking signal "
                    "found in this file",
                    confidence=None if confirmed else Confidence.MEDIUM,
                )
                continue
            dotted = _dotted(node.func)
            if dotted is None:
                continue
            prefix, _, attr = dotted.rpartition(".")
            if (prefix, attr) in _HIGH_NETWORK_CALLS:
                yield self.finding(
                    ctx,
                    node,
                    f"`{dotted}(...)` hits the network directly; no mocking "
                    "signal (responses/respx/requests_mock/aioresponses/vcr/"
                    "monkeypatch/patch) found in this file",
                )
            elif (prefix, attr) in _MEDIUM_NETWORK_CALLS:
                yield self.finding(
                    ctx,
                    node,
                    f"`{dotted}(...)` constructs a live network client with no "
                    "mocking signal found in this file; flagged as construction "
                    "only, not a confirmed request",
                    confidence=Confidence.MEDIUM,
                )
