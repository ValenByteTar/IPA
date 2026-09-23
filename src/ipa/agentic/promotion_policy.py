"""Promotion policy — decides which documents get promoted to the main corpus.

Policy:
  - configured_scrape: auto-promote (trusted source, no score threshold)
  - user_provided: auto-promote (URL pasted by the user — a known source;
    still passes the curation gates for duplicates/insufficient evidence)
  - agent_research: promote only if promotion_score >= 0.70

This is independent of the Reporter. The Reporter can still produce reports,
but promotion is decided by provenance + scoring, not by report approval.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ipa.reporter.reporter_curation import promotion_score
from ipa.reporter.reporter_contracts import ReporterDecision, ReporterDocumentDecision, ScoreBundle


# Threshold for agent-research documents.
AGENT_RESEARCH_THRESHOLD = 0.70


@dataclass(frozen=True)
class PromotionDecision:
    """Result of evaluating a document for promotion."""
    document_id: str
    should_promote: bool
    reason: str
    provenance: str
    score: float | None  # promotion_score, or None if not scored


def evaluate_promotion(
    document_id: str,
    provenance: str,
    decision: ReporterDocumentDecision | None = None,
    scores: ScoreBundle | None = None,
) -> PromotionDecision:
    """Evaluate whether a document should be promoted to the main corpus.

    Args:
        document_id: The document ID.
        provenance: "configured_scrape", "user_provided" or "agent_research".
        decision: The curation decision (optional, for decision-based checks).
        scores: The score bundle (optional, for score-based checks).

    Returns:
        A PromotionDecision indicating whether to promote and why.
    """
    # If we have a curation decision, use its scores
    if decision is not None and scores is None:
        scores = decision.scores

    # --- Policy: configured_scrape / user_provided → auto-promote ---
    # user_provided = URL pegada por el usuario en el chat (seed explícita de
    # una corrida research): fuente conocida por autorización directa, mismo
    # tratamiento que un sitio configurado. Ambas igual respetan los gates de
    # curación (duplicate / insufficient_evidence).
    if provenance in ("configured_scrape", "user_provided"):
        if decision is not None:
            if decision.decision == ReporterDecision.DUPLICATE:
                return PromotionDecision(
                    document_id, False, "duplicate document", provenance, None
                )
            if decision.decision == ReporterDecision.INSUFFICIENT_EVIDENCE:
                return PromotionDecision(
                    document_id, False, "insufficient evidence", provenance, None
                )
        return PromotionDecision(
            document_id, True, f"{provenance}: auto-promote", provenance, None
        )

    # --- Policy: agent_research → score >= 0.70 ---
    if provenance == "agent_research":
        if scores is None:
            return PromotionDecision(
                document_id, False, "agent research: no scores available", provenance, None
            )
        # Skip duplicates and insufficient evidence
        if decision is not None:
            if decision.decision == ReporterDecision.DUPLICATE:
                return PromotionDecision(
                    document_id, False, "duplicate document", provenance, None
                )
            if decision.decision == ReporterDecision.INSUFFICIENT_EVIDENCE:
                return PromotionDecision(
                    document_id, False, "insufficient evidence", provenance, None
                )
        score = promotion_score(scores)
        if score >= AGENT_RESEARCH_THRESHOLD:
            return PromotionDecision(
                document_id, True,
                f"agent research: score {score:.2f} >= {AGENT_RESEARCH_THRESHOLD}",
                provenance, score,
            )
        return PromotionDecision(
            document_id, False,
            f"agent research: score {score:.2f} < {AGENT_RESEARCH_THRESHOLD}",
            provenance, score,
        )

    # --- Unknown provenance → don't promote ---
    return PromotionDecision(
        document_id, False, f"unknown provenance: {provenance}", provenance, None
    )


def evaluate_batch(
    documents: list[dict[str, Any]],
    provenance_map: dict[str, dict],
    decisions: list[ReporterDocumentDecision] | None = None,
) -> list[PromotionDecision]:
    """Evaluate a batch of documents for promotion.

    Args:
        documents: List of document dicts (with document_id).
        provenance_map: {document_id: {provenance, source_url, ...}} from DocumentStore.
        decisions: Curation decisions (optional, for score-based checks).

    Returns:
        List of PromotionDecision, one per document.
    """
    decision_map: dict[str, ReporterDocumentDecision] = {}
    if decisions:
        for d in decisions:
            decision_map[d.document_id] = d

    results: list[PromotionDecision] = []
    for doc in documents:
        doc_id = doc["document_id"]
        prov_info = provenance_map.get(doc_id, {})
        provenance = prov_info.get("provenance", "unknown")
        decision = decision_map.get(doc_id)
        scores = decision.scores if decision else None
        results.append(evaluate_promotion(doc_id, provenance, decision, scores))
    return results
