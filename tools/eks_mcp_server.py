"""Read-only dev-time MCP server for IPA's Engineering Knowledge System."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

try:
    from .eks_repository import EKSRepository
except ImportError:  # Supports direct stdio execution by Devin.
    from eks_repository import EKSRepository

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _project_root_path(value: str, default: str) -> Path:
    candidate = Path(value or default)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (PROJECT_ROOT / candidate).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise RuntimeError("EKS MCP path must remain inside the IPA project") from exc
    return resolved


EKS_ROOT = _project_root_path(os.environ.get("IPA_EKS_ROOT", "knowledge"), "knowledge")
REFERENCE_ROOTS = tuple(
    _project_root_path(value.strip(), "docs/adr")
    for value in os.environ.get("IPA_EKS_REFERENCE_ROOTS", "docs/adr").split(",")
    if value.strip()
)
REPOSITORY = EKSRepository(EKS_ROOT, REFERENCE_ROOTS)

mcp = FastMCP("ipa-eks")


def _limit(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 10
    return max(1, min(value, 50))


@mcp.tool()
def eks_list(category: str | None = None, status: str | None = None, component: str | None = None, tag: str | None = None) -> dict[str, Any]:
    """List local EKS records by metadata filters. This tool is read-only."""
    records = []
    for record in REPOSITORY.records():
        if category and record.category != category:
            continue
        if status and record.metadata.get("status") != status:
            continue
        components = record.metadata.get("components", [])
        tags = record.metadata.get("tags", [])
        if component and component not in components:
            continue
        if tag and tag not in tags:
            continue
        records.append(REPOSITORY._summary(record))
    return {"records": records, "count": len(records)}


@mcp.tool()
def eks_get(identifier: str) -> dict[str, Any]:
    """Read one EKS or explicitly allowed ADR Markdown document."""
    try:
        return {"document": REPOSITORY.serialize(REPOSITORY.get(identifier))}
    except (KeyError, ValueError) as exc:
        return {"error": str(exc), "document": None}


@mcp.tool()
def eks_search(query: str, category: str | None = None, status: str | None = None, component: str | None = None, tag: str | None = None, limit: int = 10) -> dict[str, Any]:
    """Search EKS metadata and Markdown content deterministically."""
    try:
        results = REPOSITORY.search(query, category=category, status=status, component=component, tag=tag, limit=_limit(limit))
        return {"query": query, "results": results, "count": len(results)}
    except ValueError as exc:
        return {"query": query, "results": [], "count": 0, "error": str(exc)}


@mcp.tool()
def eks_context(task: str, components: list[str] | None = None, tags: list[str] | None = None, limit: int = 8) -> dict[str, Any]:
    """Build a compact engineering context package without changing repository state."""
    try:
        return REPOSITORY.context(task, components=components or [], tags=tags or [], limit=_limit(limit))
    except ValueError as exc:
        return {"task": task, "results": [], "knowledge_available": False, "error": str(exc)}


if __name__ == "__main__":
    mcp.run()
