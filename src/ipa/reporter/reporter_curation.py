"""Deterministic curation and deduplication for Reporter documents."""
from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from datetime import datetime
from typing import Any

from ipa.reporter.reporter_contracts import ReporterDecision, ReporterDocumentDecision, ReviewStatus, ScoreBundle, generation_provenance


def promotion_score(scores: ScoreBundle) -> float:
    return round(0.30 * scores.relevance + 0.20 * scores.novelty + 0.20 * scores.source_quality + 0.15 * scores.impact + 0.10 * scores.depth + 0.05 * scores.actionability, 4)


def auto_promotion_eligible(scores: ScoreBundle, decision: ReporterDecision, *, duplicate_of: str | None, in_period: bool, has_text: bool) -> bool:
    return (
        decision == ReporterDecision.PROMOTE and duplicate_of is None and in_period and has_text
        and scores.relevance >= 0.80 and scores.novelty >= 0.60
        and scores.source_quality >= 0.65 and scores.impact >= 0.60
        and promotion_score(scores) >= 0.78
    )


def normalized_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    return "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _lexical_similarity(left: str, right: str) -> float:
    a = set(re.findall(r"[\w-]{4,}", left.lower()))
    b = set(re.findall(r"[\w-]{4,}", right.lower()))
    return len(a & b) / len(a | b) if a and b else 0.0


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


# Domains known for high-quality technical content (for source_quality heuristic).
_TRUSTED_DOMAINS = {
    "arxiv.org", "developer.nvidia.com", "blog.google", "research.meta.ai",
    "openai.com", "anthropic.com", "huggingface.co", "thehackernews.com",
    "www.cisa.gov", "www.microsoft.com", "vllm.ai", "claude.com",
    "www.emergentmind.com", "blog.isecauditors.com",
}

# Verb patterns suggesting actionable content.
_ACTION_VERBS = re.compile(
    r"\b(should|must|need to|recommend|implement|deploy|configure|install|"
    r"update|patch|upgrade|migrate|enable|disable|set up|follow|ensure|"
    r"verify|check|monitor|alert|block|allow|restrict|encrypt|backup)\b",
    re.IGNORECASE,
)


def _in_period(value: str | None, start: str, end: str) -> bool:
    if not value or not start or not end:
        return True
    try:
        current = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return datetime.fromisoformat(start.replace("Z", "+00:00")) <= current < datetime.fromisoformat(end.replace("Z", "+00:00"))
    except ValueError:
        return False


def curate_documents(
    documents: list[dict[str, Any]],
    report_id: str,
    period_start: str,
    period_end: str,
    interests: tuple[str, ...] = (),
    quality_threshold: float = 0.25,
    historical_documents: list[dict[str, Any]] | None = None,
    classifier=None,
    classifier_batch=None,
    progress_callback=None,
    document_embeddings: dict[str, list[float]] | None = None,
    interest_embeddings: list[list[float]] | None = None,
    historical_embeddings: list[list[float]] | None = None,
) -> list[ReporterDocumentDecision]:
    historical_documents = historical_documents or []
    document_embeddings = document_embeddings or {}
    interest_embeddings = interest_embeddings or []
    historical_embeddings = historical_embeddings or []
    by_url: dict[str, str] = {}
    by_hash: dict[str, str] = {}
    decisions: list[ReporterDocumentDecision] = []
    llm_progress_callback = None
    if progress_callback is not None:
        def llm_progress_callback(current: int, total: int, title: str) -> None:
            progress_callback(current + 1, total, title)
    batch_results = classifier_batch(documents, llm_progress_callback) if classifier_batch else []
    for index, document in enumerate(documents):
        if progress_callback is not None:
            progress_callback(index + 1, len(documents), str(document.get("title", "")))
        document_id = str(document["document_id"])
        text = str(document.get("text", ""))
        url = str(document.get("canonical_url") or document.get("source_url") or "")
        content_hash = str(document.get("content_hash") or normalized_hash(text))
        score = float(document.get("quality_score") or 0.0)
        domain = str(document.get("source_domain") or "")
        doc_emb = document_embeddings.get(document_id)

        # --- Relevance: cosine similarity with interest embeddings (semantic) ---
        if doc_emb and interest_embeddings:
            relevance = max(_cosine_similarity(doc_emb, ie) for ie in interest_embeddings)
        elif not interests:
            relevance = 0.5
        else:
            relevance = min(1.0, 0.3 + 0.2 * sum(1 for term in interests if term.lower() in (document.get("title", "") + " " + text).lower()))

        # --- Novelty: cosine distance to historical embeddings (semantic) ---
        if doc_emb and historical_embeddings:
            novelty = 1.0 - max(_cosine_similarity(doc_emb, he) for he in historical_embeddings)
        else:
            novelty = 1.0 - max((_lexical_similarity(text, old.get("text", "")) for old in historical_documents), default=0.0)

        # --- Source quality: domain trust + scraper score ---
        domain_bonus = 0.15 if domain in _TRUSTED_DOMAINS else 0.0
        source_quality = max(0.0, min(1.0, (score if score else 0.5) + domain_bonus))

        # --- Impact: text density + entity count ---
        text_len = len(text)
        numbers = len(re.findall(r"\b\d+(?:\.\d+)?\b", text))
        impact = min(1.0, 0.3 + min(text_len / 5000, 0.3) + min(numbers / 50, 0.2) + domain_bonus)

        # --- Depth: structure indicators ---
        has_sections = bool(re.search(r"^#{1,3}\s|\n[A-Z][a-z]+:\s", text, re.MULTILINE))
        has_references = bool(re.search(r"\b(?:ref|source|see|doi|arxiv)\b", text, re.IGNORECASE))
        has_lists = bool(re.search(r"^\s*[-*]\s|^\s*\d+\.\s", text, re.MULTILINE))
        depth = min(1.0, min(text_len / 5000, 0.4) + (0.15 if has_sections else 0.0) + (0.15 if has_references else 0.0) + (0.1 if has_lists else 0.0))

        # --- Actionability: action verbs + recommendations ---
        action_matches = len(_ACTION_VERBS.findall(text))
        actionability = min(1.0, 0.2 + min(action_matches / 10, 0.4) + (0.2 if has_lists else 0.0))

        decision = ReporterDecision.REPORTER_ONLY
        reason = "Contenido procesable conservado en el corpus Reporter."
        duplicate_of = None
        if not _in_period(document.get("published_at"), period_start, period_end):
            decision, reason = ReporterDecision.DEFER, "Fecha fuera del perÃ­odo solicitado o no confiable."
        elif not text.strip():
            decision, reason = ReporterDecision.INSUFFICIENT_EVIDENCE, "El documento no contiene texto utilizable."
        elif score and score < quality_threshold:
            decision, reason = ReporterDecision.IRRELEVANT, "El quality gate del scraper estÃ¡ por debajo del umbral."
        elif url and url in by_url:
            decision, duplicate_of, reason = ReporterDecision.DUPLICATE, by_url[url], "Canonical URL duplicada."
        elif content_hash in by_hash:
            decision, duplicate_of, reason = ReporterDecision.DUPLICATE, by_hash[content_hash], "Content hash duplicado."
        elif relevance < 0.3:
            decision, reason = ReporterDecision.REPORTER_ONLY, "Relevancia baja para los intereses configurados; se conserva para anÃ¡lisis neutral."
        elif novelty < 0.2:
            decision, reason = ReporterDecision.DUPLICATE, None, "Contenido muy similar al corpus histÃ³rico."
        else:
            decision = ReporterDecision.PROMOTE
            reason = "Contenido relevante y suficientemente novedoso; requiere revisiÃ³n humana para promociÃ³n."
        by_url[url] = document_id
        by_hash[content_hash] = document_id
        semantic = batch_results[index] if index < len(batch_results) else (classifier(document) if classifier else {})
        if semantic:
            values = {key: max(0.0, min(1.0, float(semantic.get(key, value)))) for key, value in {
                "relevance": relevance, "novelty": novelty, "source_quality": source_quality, "impact": impact,
                "depth": depth, "actionability": actionability,
            }.items()}
            scores = ScoreBundle(**values)
            if decision == ReporterDecision.PROMOTE and semantic.get("decision") in {"reporter_only", "defer", "irrelevant", "insufficient_evidence"}:
                decision = ReporterDecision(str(semantic["decision"]))
            reason = str(semantic.get("reason") or reason)
            content_type = semantic.get("content_type")
        else:
            scores = ScoreBundle(relevance, novelty, source_quality, impact, depth, actionability)
            content_type = document.get("content_type")
        auto_approved = auto_promotion_eligible(
            scores, decision, duplicate_of=duplicate_of,
            in_period=_in_period(document.get("published_at"), period_start, period_end),
            has_text=bool(text.strip()),
        )
        decisions.append(ReporterDocumentDecision(
            decision_id="decision:" + hashlib.sha256((report_id + document_id).encode()).hexdigest()[:32],
            report_id=report_id, document_id=document_id, artifact_id=str(document.get("artifact_id", content_hash)),
            decision=decision, scores=scores,
            reason=(reason + f" Auto-promociÃ³n habilitada (score {promotion_score(scores):.2f}).") if auto_approved else reason,
            evidence=[{"source_id": document_id, "source_type": "document", "content_hash": content_hash}],
            generation=generation_provenance(content_hash), duplicate_of=duplicate_of,
            content_type=content_type,
            review_status=ReviewStatus.APPROVED if auto_approved else ReviewStatus.PENDING,
            approval={"decision": "approved", "decided_by": "auto-promotion-v1", "note": f"promotion_score={promotion_score(scores):.4f}"} if auto_approved else None,
        ))
    return decisions

