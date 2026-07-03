"""G7 rule tests — the TP/FP-guard pattern every rule must follow."""

from __future__ import annotations

import ast

from flakehound.rules.base import Confidence, FileContext
from flakehound.rules.g7_unmocked_network import UnmockedNetworkCall


def _ctx(source: str, name: str = "test_x.py") -> FileContext:
    return FileContext(
        path=name,
        source=source,
        tree=ast.parse(source),
        is_test_file=True,
        is_conftest=False,
    )


def _run(source: str):
    return list(UnmockedNetworkCall().check(_ctx(source)))


def test_detects_unmocked_requests_get():
    src = (
        "import requests\n"
        "def test_a():\n"
        "    resp = requests.get('https://example.com/api')\n"
        "    assert resp.status_code == 200\n"
    )
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].rule_id == "G7"
    assert findings[0].line == 3
    assert findings[0].confidence == Confidence.HIGH


def test_detects_unmocked_urlopen_and_socket():
    src = (
        "from urllib.request import urlopen\n"
        "import socket\n"
        "def test_a():\n"
        "    urlopen('https://example.com')\n"
        "    socket.create_connection(('example.com', 80))\n"
    )
    findings = _run(src)
    assert {f.line for f in findings} == {4, 5}
    assert all(f.confidence == Confidence.HIGH for f in findings)


def test_httpx_client_construction_is_downgraded_to_medium():
    src = "import httpx\ndef test_a():\n    client = httpx.AsyncClient()\n"
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].line == 3
    assert findings[0].confidence == Confidence.MEDIUM


def test_fp_guard_responses_activate_suppresses_file():
    src = (
        "import requests\n"
        "import responses\n"
        "@responses.activate\n"
        "def test_a():\n"
        "    requests.get('https://example.com')\n"
    )
    assert _run(src) == []


def test_fp_guard_monkeypatch_setattr_on_callee_suppresses_file():
    src = (
        "import requests\n"
        "def fake_get(*a, **kw):\n"
        "    return None\n"
        "def test_a(monkeypatch):\n"
        "    monkeypatch.setattr(requests, 'get', fake_get)\n"
        "    requests.get('https://example.com')\n"
    )
    assert _run(src) == []


def test_fp_guard_requests_mock_fixture_suppresses_file():
    src = (
        "import requests\n"
        "def test_a(requests_mock):\n"
        "    requests_mock.get('https://example.com', json={})\n"
        "    requests.get('https://example.com')\n"
    )
    assert _run(src) == []


def test_fp_guard_localhost_and_loopback_literals_not_flagged():
    src = (
        "import requests\n"
        "import socket\n"
        "def test_a():\n"
        "    requests.get('http://127.0.0.1:8000/health')\n"
        "    requests.get('http://localhost:8000/health')\n"
        "    socket.create_connection(('localhost', 8000))\n"
    )
    assert _run(src) == []
