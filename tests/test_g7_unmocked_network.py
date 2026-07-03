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
    assert findings[0].confidence == Confidence.MEDIUM


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
    assert all(f.confidence == Confidence.MEDIUM for f in findings)


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


def test_fp_guard_responses_mock_fixture_suppresses_file():
    src = (
        "import requests\n"
        "def test_a(responses_mock):\n"
        "    responses_mock.get('https://example.com', json={})\n"
        "    requests.get('https://example.com')\n"
    )
    assert _run(src) == []


def test_fp_guard_respx_mock_fixture_suppresses_file():
    src = (
        "import httpx\n"
        "def test_a(respx_mock):\n"
        "    respx_mock.get('https://example.com').respond(json={})\n"
        "    httpx.get('https://example.com')\n"
    )
    assert _run(src) == []


def test_detects_chained_requests_session_get():
    src = (
        "import requests\n"
        "def test_a():\n"
        "    resp = requests.Session().get('https://example.com/api')\n"
        "    assert resp.status_code == 200\n"
    )
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].line == 3
    assert findings[0].confidence == Confidence.MEDIUM


def test_detects_two_step_requests_session_get():
    src = (
        "import requests\n"
        "def test_a():\n"
        "    s = requests.Session()\n"
        "    resp = s.get('https://example.com/api')\n"
        "    assert resp.status_code == 200\n"
    )
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].line == 4


def test_detects_chained_httpx_client_get():
    src = "import httpx\ndef test_a():\n    httpx.Client().get('https://example.com')\n"
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].line == 3
    # the constructor call is superseded by the stronger chained-call finding,
    # not double-reported as a separate "construction only" finding
    assert "construction only" not in findings[0].message


def test_detects_two_step_httpx_asyncclient_post():
    src = (
        "import httpx\n"
        "def test_a():\n"
        "    client = httpx.AsyncClient()\n"
        "    client.post('https://example.com', json={})\n"
    )
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].line == 4
    assert "construction only" not in findings[0].message


def test_detects_chained_aiohttp_clientsession_get():
    src = (
        "import aiohttp\n"
        "async def test_a():\n"
        "    async with aiohttp.ClientSession().get('https://example.com') as r:\n"
        "        assert r.status == 200\n"
    )
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].line == 3
    assert "construction only" not in findings[0].message


def test_detects_chained_socket_connect():
    src = (
        "import socket\n"
        "def test_a():\n"
        "    socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(('example.com', 80))\n"
    )
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].line == 3


def test_detects_two_step_socket_connect():
    src = (
        "import socket\n"
        "def test_a():\n"
        "    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
        "    sock.connect(('example.com', 80))\n"
    )
    findings = _run(src)
    assert len(findings) == 1
    assert findings[0].line == 4


def test_fp_guard_chained_session_suppressed_by_mock_signal():
    src = (
        "import requests\n"
        "import responses\n"
        "@responses.activate\n"
        "def test_a():\n"
        "    requests.Session().get('https://example.com')\n"
    )
    assert _run(src) == []


def test_fp_guard_two_step_session_localhost_not_flagged():
    src = (
        "import requests\n"
        "def test_a():\n"
        "    s = requests.Session()\n"
        "    s.get('http://127.0.0.1:8000/health')\n"
    )
    assert _run(src) == []


def test_fp_guard_chained_socket_localhost_not_flagged():
    src = (
        "import socket\n"
        "def test_a():\n"
        "    socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(('localhost', 8000))\n"
    )
    assert _run(src) == []
