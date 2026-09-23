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
# No default reference root: `docs/adr/` does not exist in this repo (DEC-008
# keeps DEC-* as the ADR format), so an unset env var must not register a dead
# root. Point IPA_EKS_REFERENCE_ROOTS at external/legacy ADRs explicitly.
REFERENCE_ROOTS = tuple(
    _project_root_path(value.strip(), "docs/adr")
    for value in os.environ.get("IPA_EKS_REFERENCE_ROOTS", "").split(",")
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
        if component and not REPOSITORY.component_matches(
                components if isinstance(components, list) else [], component):
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
def eks_search(query: str, category: str | None = None, status: str | None = None, component: str | None = None, tag: str | None = None, limit: int = 10, include_historical: bool = True) -> dict[str, Any]:
    """Search EKS metadata and Markdown content deterministically."""
    try:
        results = REPOSITORY.search(query, category=category, status=status, component=component, tag=tag, limit=_limit(limit), include_historical=include_historical)
        return {"query": query, "results": results, "count": len(results)}
    except ValueError as exc:
        return {"query": query, "results": [], "count": 0, "error": str(exc)}


@mcp.tool()
def eks_governing(paths: list[str]) -> dict[str, Any]:
    """EKS records whose `affects` globs govern the given repo-relative paths.

    Includes rejected/superseded records — the graveyard is what stops an
    agent from re-trying an already-discarded approach. Read-only.
    """
    if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
        return {"error": "paths must be a list of strings", "records": []}
    records = REPOSITORY.governing(paths)
    return {"paths": paths, "records": records, "count": len(records)}


@mcp.tool()
def eks_context(task: str, components: list[str] | None = None, tags: list[str] | None = None, paths: list[str] | None = None, limit: int = 8, include_historical: bool = False) -> dict[str, Any]:
    """Build a compact engineering context package without changing repository state.

    By default superseded/rejected records are excluded and every hit is
    expanded one hop through its `related` links so the package carries the
    decisions and benchmarks behind each result. `paths` additionally
    activates records whose `affects` globs cover files being edited.
    """
    try:
        return REPOSITORY.context(task, components=components or [], tags=tags or [], paths=paths or [], limit=_limit(limit), include_historical=include_historical)
    except ValueError as exc:
        return {"task": task, "results": [], "knowledge_available": False, "error": str(exc)}


@mcp.tool()
def eks_report() -> dict[str, Any]:
    """Hygiene report over the EKS catalog: open items, coverage, link health."""
    return REPOSITORY.report()


if __name__ == "__main__":
    mcp.run()
