"""MCP server (proxy mode) tests.

The MCP server is a thin frontier over the dashboard's unified tool
registry — it owns no models. These tests guard:
  1. sys.path import safety (inserting src/ipa shadowed the `mcp` SDK).
  2. Tool generation from registry specs (name/docstring/args).
  3. The HTTP roundtrip: URL, payload and response parsing.
  4. Dashboard-down → clear JSON error, never a raw exception.
"""
import json
import sys
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ipa.mcp import mcp_server  # noqa: E402


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# ---------------------------------------------------------------------------

def test_mcp_server_imports_and_exposes_generic_tools():
    """Guards the sys.path fix: inserting src/ipa (not src/) made `ipa/mcp/`
    shadow the `mcp` SDK and the module failed to import entirely."""
    assert callable(mcp_server.ipa_tool)
    assert callable(mcp_server.list_ipa_tools)
    assert callable(mcp_server.tutor_focus)


def test_make_tool_uses_spec_name_and_docstring():
    spec = {"name": "search_corpus",
            "description": "busca en el corpus. Cita hits como [n].",
            "args_doc": '{"query": "texto a buscar", "limit": 5}'}
    tool = mcp_server._make_tool(spec)
    assert tool.__name__ == "search_corpus"
    assert "busca en el corpus" in (tool.__doc__ or "")
    assert '"query"' in (tool.__doc__ or "")


def test_tool_roundtrip_posts_to_dashboard(monkeypatch):
    """La tool generada POSTea {name, args} al endpoint unificado y parsea
    la respuesta JSON del dashboard."""
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["method"] = req.method
        captured["body"] = json.loads(req.data.decode())
        return _FakeResponse({"ok": True, "tool": "search_corpus",
                              "summary": "3 hits", "data": {"hits": []}})

    monkeypatch.setattr(mcp_server.urllib.request, "urlopen", fake_urlopen)
    tool = mcp_server._make_tool({"name": "search_corpus",
                                  "description": "busca", "args_doc": "{}"})
    out = json.loads(tool({"query": "fotonica"}))

    assert captured["url"] == mcp_server.PROXY_URL + "/api/tools/execute"
    assert captured["method"] == "POST"
    assert captured["body"] == {"name": "search_corpus", "args": {"query": "fotonica"}}
    assert out["ok"] is True and out["summary"] == "3 hits"


def test_tool_dashboard_down_returns_clear_error(monkeypatch):
    """Dashboard caído → JSON con instrucción, no excepción cruda."""

    def boom(req, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(mcp_server.urllib.request, "urlopen", boom)
    tool = mcp_server._make_tool({"name": "search_corpus",
                                  "description": "busca", "args_doc": "{}"})
    out = json.loads(tool({}))
    assert out["ok"] is False
    assert "dashboard" in out["error"].lower()
    assert "start" in out["error"].lower() or "start_ipa_dashboard" in out["error"]


def test_register_registry_tools_generates_one_per_spec():
    catalog = {"tools": [
        {"name": "search_corpus", "description": "busca", "args_doc": "{}"},
        {"name": "recall_memory", "description": "memoria", "args_doc": "{}"},
    ]}
    registered = mcp_server.register_registry_tools(catalog)
    assert registered == ["search_corpus", "recall_memory"]


def test_startup_registration_is_best_effort(monkeypatch):
    """Dashboard caído al arrancar → el server igual levanta (sin tools
    del registry; ipa_tool sigue disponible)."""

    def boom(req, timeout=None):
        raise urllib.error.URLError("refused")

    monkeypatch.setattr(mcp_server.urllib.request, "urlopen", boom)
    assert mcp_server._register_registry_tools() == []


def test_tutor_tools_use_read_endpoints(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["path"] = urllib.parse.urlsplit(req.full_url).path
        return _FakeResponse({"ok": True, "focus": None})

    import urllib.parse
    monkeypatch.setattr(mcp_server.urllib.request, "urlopen", fake_urlopen)
    mcp_server.tutor_focus()
    assert captured["path"] == "/api/tutor/focus"
    mcp_server.tutor_projects()
    assert captured["path"] == "/api/tutor/projects"
    mcp_server.tutor_roadmap_context("roadmap:abc")
    assert captured["path"] == "/api/tutor/roadmap/context"
