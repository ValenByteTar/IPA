"""IPA MCP Server — thin proxy to the dashboard's unified tool registry.

The dashboard (watchdog-managed, always-on) owns the models, the corpus
and the tool registry. This server owns NOTHING heavy: every tool is an
HTTP call to the dashboard, so external agents (Devin, Claude, Ollama)
use the exact same retrieval/ingestion/memory implementation as the
built-in chat — one frontier, zero drift, zero extra VRAM.

Tools:
  - One MCP tool per registry spec (search_corpus, recall_memory,
    research_topic, plan_task, run_ingestion, get_user_profile, ...),
    generated from GET /api/tools/catalog at startup — the MCP surface
    IS the registry, so it cannot drift from what the 9B chat sees.
  - ipa_tool(name, args): generic dispatcher (works even if the catalog
    fetch failed at startup).
  - list_ipa_tools(): the current registry with arg docs.
  - Tutor read-only: tutor_focus / tutor_projects / tutor_roadmap_context
    — lets an external session resume the tutoring context.

Requires the dashboard running (default http://127.0.0.1:8765, override
with IPA_PROXY_URL). research_topic is async (same as the dashboard
chat): it returns "investigación en curso" and the material lands later
— check with list_promotions / get_report. Local files: drop them in
Landing/ and call run_ingestion.

Usage (MCP client config):
  {
    "mcpServers": {
      "ipa": {
        "command": "C:\\path\\to\\.venv\\Scripts\\python.exe",
        "args": ["-m", "ipa.mcp.mcp_server"],
        "env": {"PYTHONPATH": "C:\\path\\to\\IPA\\src"}
      }
    }
  }
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

# Ensure src is on the path when run as module. NOTE: this must be the src/
# root (the ipa package's parent), NOT src/ipa — inserting the package dir
# itself would make `ipa/mcp/` shadow the `mcp` SDK on sys.path and the
# import below would fail with "No module named 'mcp.server'".
_src = Path(__file__).resolve().parents[2]
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from mcp.server.fastmcp import FastMCP

PROXY_URL = os.environ.get("IPA_PROXY_URL", "http://127.0.0.1:8765").rstrip("/")
HTTP_TIMEOUT = int(os.environ.get("IPA_MCP_TIMEOUT", "300"))

mcp = FastMCP(
    "ipa",
    instructions=(
        "IPA personal agent — shared corpus, memory, research pipeline and "
        "Tutor state, served by the always-on local dashboard. Prefer the "
        "specific tools; use list_ipa_tools to discover the full registry."
    ),
)


# ---------------------------------------------------------------------------
# HTTP plumbing
# ---------------------------------------------------------------------------

class DashboardUnavailable(RuntimeError):
    pass


def _request(method: str, path: str, payload: dict | None = None) -> dict[str, Any]:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        PROXY_URL + path, data=data, method=method,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise DashboardUnavailable(
            f"IPA dashboard not reachable at {PROXY_URL} — start it with "
            f"start_ipa_dashboard.ps1 ({exc})") from exc


def _json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _call(method: str, path: str, payload: dict | None = None) -> str:
    """HTTP call with the dashboard-down case rendered as a clear JSON
    error — the MCP client sees the reason, not a stack trace."""
    try:
        return _json(_request(method, path, payload))
    except DashboardUnavailable as exc:
        return _json({"ok": False, "error": str(exc)})


def _fetch_catalog() -> dict[str, Any]:
    try:
        return _request("GET", "/api/tools/catalog")
    except DashboardUnavailable as exc:
        return {"tools": [], "error": str(exc)}


# ---------------------------------------------------------------------------
# Registry-backed tools (generated from the dashboard's registry)
# ---------------------------------------------------------------------------

def _make_tool(spec: dict[str, str]):
    """Build one MCP tool closure from a registry spec."""
    name = spec["name"]

    def tool(args: dict[str, Any] | None = None) -> str:
        return _call("POST", "/api/tools/execute", {"name": name, "args": args or {}})

    tool.__name__ = name
    tool.__doc__ = (
        f"{spec['description']}\n\n"
        f"Args (JSON object): {spec['args_doc']}\n"
        f"Runs in the IPA dashboard process (shared corpus, memory, executors)."
    )
    return tool


def register_registry_tools(catalog: dict[str, Any]) -> list[str]:
    """Register one MCP tool per registry spec. Returns registered names."""
    registered: list[str] = []
    for spec in catalog.get("tools", []):
        mcp.tool()(_make_tool(spec))
        registered.append(spec["name"])
    return registered


def _register_registry_tools() -> list[str]:
    """Best-effort startup registration: if the dashboard is down, the
    generic ipa_tool still works once it comes back."""
    try:
        return register_registry_tools(_request("GET", "/api/tools/catalog"))
    except Exception:
        return []


# Generic dispatcher — always available.
@mcp.tool()
def ipa_tool(name: str, args: dict[str, Any] | None = None) -> str:
    """Execute any IPA registry tool by name. Use list_ipa_tools to
    discover valid names and their args schemas."""
    return _call("POST", "/api/tools/execute", {"name": name, "args": args or {}})


@mcp.tool()
def list_ipa_tools() -> str:
    """List every IPA registry tool with its description and args docs."""
    return _json(_fetch_catalog())


# ---------------------------------------------------------------------------
# Tutor state (read-only) — resume tutoring context from any surface
# ---------------------------------------------------------------------------

@mcp.tool()
def tutor_focus() -> str:
    """What the Tutor is working on right now: focused roadmap, current
    unit and progress."""
    return _call("GET", "/api/tutor/focus?session_id=")


@mcp.tool()
def tutor_projects() -> str:
    """All learning projects (goals) with their roadmaps and statuses."""
    return _call("GET", "/api/tutor/projects")


@mcp.tool()
def tutor_roadmap_context(roadmap_id: str) -> str:
    """Full roadmap context: objective, success criteria, rationale
    (assumptions/uncertainties/change reason), per-unit progress and
    source concepts."""
    return _call(
        "GET", f"/api/tutor/roadmap/context?roadmap_id={urllib.parse.quote(roadmap_id)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_registered = _register_registry_tools()


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
