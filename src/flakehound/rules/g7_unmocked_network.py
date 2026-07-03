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
once any of those signals exist. That file-level suppression heuristic is also why
this rule's default tier is `MEDIUM`, not `HIGH`: it can show a call site with no
mocking signal anywhere in the file, but it can never *prove* that specific site is
unmocked (AGENTS.md tier-honesty contract — HIGH is reserved for near-certain static
matches).

Chained/session-object calls are matched too: `requests.Session().get(...)`,
`httpx.Client()`/`AsyncClient()` methods, `aiohttp.ClientSession().get(...)`, and
`socket.socket(...).connect(...)`, including the two-step form
(`s = requests.Session(); s.get(...)`) via simple local-name tracking within a file
— not full dataflow, just "last assignment of this name wins."
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

# Chained/tracked session-object calls: `requests.Session().get(...)`,
# `s = httpx.Client(); s.get(...)`, `socket.socket(...).connect(...)`. Maps a
# constructor's dotted name to the "kind" key used to look up its network-facing
# methods below.
_SESSION_CONSTRUCTOR_KIND = {
    "requests.Session": "requests",
    "httpx.Client": "httpx",
    "httpx.AsyncClient": "httpx",
    "aiohttp.ClientSession": "aiohttp",
    "socket.socket": "socket",
}

_SESSION_METHODS: dict[str, frozenset[str]] = {
    "requests": frozenset({"get", "post", "put", "delete", "patch", "head", "options", "request"}),
    "httpx": frozenset(
        {"get", "post", "put", "delete", "patch", "head", "options", "request", "stream"}
    ),
    "aiohttp": frozenset({"get", "post", "put", "delete", "patch", "head", "options", "request"}),
    "socket": frozenset({"connect"}),
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
    "responses_mock",
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


def _local_session_vars(tree: ast.Module) -> dict[str, tuple[str, ast.Call]]:
    """Track `name = <session/socket constructor>()` assignments.

    Simple local-name tracking, not dataflow: one name -> (kind, constructor call)
    per file, last assignment wins on reassignment. Good enough to catch the common
    `s = requests.Session(); s.get(...)` two-step form.
    """
    mapping: dict[str, tuple[str, ast.Call]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        if not isinstance(node.value, ast.Call):
            continue
        kind = _SESSION_CONSTRUCTOR_KIND.get(_dotted(node.value.func) or "")
        if kind is not None:
            mapping[node.targets[0].id] = (kind, node.value)
    return mapping


def _chained_session_call(func: ast.expr) -> tuple[str, str, ast.Call] | None:
    """Match `<ctor>(...).method(...)`, e.g. `requests.Session().get(url)`."""
    if not isinstance(func, ast.Attribute) or not isinstance(func.value, ast.Call):
        return None
    ctor_call = func.value
    kind = _SESSION_CONSTRUCTOR_KIND.get(_dotted(ctor_call.func) or "")
    if kind is None or func.attr not in _SESSION_METHODS[kind]:
        return None
    return kind, func.attr, ctor_call


def _tracked_session_call(
    func: ast.expr, local_vars: dict[str, tuple[str, ast.Call]]
) -> tuple[str, str, ast.Call] | None:
    """Match `s.method(...)` where `s = <ctor>()` was tracked by `_local_session_vars`."""
    if not isinstance(func, ast.Attribute) or not isinstance(func.value, ast.Name):
        return None
    tracked = local_vars.get(func.value.id)
    if tracked is None:
        return None
    kind, ctor_call = tracked
    if func.attr not in _SESSION_METHODS[kind]:
        return None
    return kind, func.attr, ctor_call


def _session_call_matches(
    tree: ast.Module, local_vars: dict[str, tuple[str, ast.Call]]
) -> list[tuple[ast.Call, str, str, ast.Call]]:
    """Every chained or tracked session/socket method call in the file.

    Returns `(call_node, kind, method, ctor_call_node)` tuples. `ctor_call_node` is
    the constructor call the method was invoked on/through — used by `check()` to
    suppress the separate "construction only" MEDIUM finding for the same object
    once there's direct evidence it was actually called.
    """
    matches: list[tuple[ast.Call, str, str, ast.Call]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _is_localhost_call(node):
            continue
        match = _chained_session_call(node.func) or _tracked_session_call(node.func, local_vars)
        if match is not None:
            kind, method, ctor_call = match
            matches.append((node, kind, method, ctor_call))
    return matches


@register
class UnmockedNetworkCall(Rule):
    id = "G7"
    name = "unmocked-network"
    cause = "network/infrastructure"
    confidence = Confidence.MEDIUM
    fix_suggestion = (
        "mock the transport boundary — `responses`/`respx`/`requests_mock`/"
        "`aioresponses` for the client library in use, or a `pytest-httpserver`/"
        "localserver fixture for a real socket target — never hit the live network "
        "from a test"
    )

    def check(self, ctx: FileContext) -> Iterable[Finding]:
        if _has_mock_signals(ctx):
            return

        local_session_vars = _local_session_vars(ctx.tree)
        session_matches = _session_call_matches(ctx.tree, local_session_vars)
        consumed_ctor_ids = {id(ctor_call) for _c, _k, _m, ctor_call in session_matches}

        for call_node, kind, method, _ctor_call in session_matches:
            yield self.finding(
                ctx,
                call_node,
                f"`.{method}(...)` on a `{kind}` session/client object hits the "
                "network directly; no mocking signal found in this file",
            )

        for node in ast.walk(ctx.tree):
            if not isinstance(node, ast.Call):
                continue
            if _is_localhost_call(node):
                continue
            if isinstance(node.func, ast.Name) and node.func.id == "urlopen":
                yield self.finding(
                    ctx,
                    node,
                    "`urlopen(...)` hits the network directly; no mocking signal "
                    "found in this file",
                )
                continue
            if id(node) in consumed_ctor_ids:
                # superseded by a stronger chained/tracked call finding above
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
                )
