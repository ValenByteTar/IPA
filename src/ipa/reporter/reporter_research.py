"""Bounded ResearchRequest bridge for Reporter deep dives."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from ipa.reporter.reporter_contracts import generation_provenance


def create_research_request(
    goal_id: str,
    concept_id: str,
    question: str,
    gap_evidence: list[dict[str, Any]],
    allowed_domains: list[str],
    budget: dict[str, int],
    require_approval: bool = True,
) -> dict[str, Any]:
    if not question.strip() or not gap_evidence:
        raise ValueError("research requires a concrete question and gap evidence")
    if not allowed_domains:
        raise ValueError("research requires at least one allowed domain")
    required = {"max_urls", "max_seconds", "max_bytes", "max_depth"}
    if not required <= budget.keys() or any(int(budget[key]) < 0 for key in required):
        raise ValueError("research budget is incomplete or invalid")
    request_hash = "sha256:" + hashlib.sha256((question + "|" + "|".join(allowed_domains)).encode()).hexdigest()
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "request_id": "research:" + request_hash[7:39],
        "goal_id": goal_id,
        "concept_id": concept_id,
        "question": question,
        "trigger": "user_deepening_request",
        "gap_evidence": gap_evidence,
        "allowed_domains": allowed_domains,
        "budget": budget,
        "status": "pending_approval" if require_approval else "draft",
        "job_id": None,
        "result_source_refs": [],
        "created_at": now,
        "updated_at": now,
        "approval": None,
        "generation": generation_provenance(request_hash),
        "field_origins": {
            "question": "user", "trigger": "system", "gap_evidence": "source",
            "allowed_domains": "user", "budget": "system",
        },
    }


def can_execute(request: dict[str, Any], require_approval: bool = True) -> bool:
    if request.get("status") not in {"approved", "running"}:
        return False
    if require_approval:
        return (request.get("approval") or {}).get("decision") == "approved"
    return True

