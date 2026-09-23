"""Deterministic curation and deduplication for Reporter documents."""
from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ipa.reporter.reporter_contracts import ReporterDecision, ReporterDocumentDecision, ReviewStatus, ScoreBundle, generation_provenance


# ---------------------------------------------------------------------------
# Tiered curation (ranking-based, not absolute-threshold-based).
#
# The legacy promotion_score() + auto_promotion_eligible() path uses fixed
# absolute thresholds (relevance>=0.80, score>=0.78, etc.). Those thresholds
# were never calibrated against ground truth, so using them as hard gates is
# fragile: a doc with score 0.77 is treated very differently from 0.78 even
# though the score is not calibrated.
#
# NOTE: auto_promotion_eligible() is still used by curate_documents() to mark
# curation decisions as review_status=approved/pending (metadata only).
# Physical promotion to the main corpus is now decided by
# ipa.agentic.promotion_policy, which uses provenance (configured_scrape vs
# agent_research) + score thresholds, independent of the report.
#
# The tiered path ranks documents WITHIN the current batch and assigns tiers
# by percentile. This is robust to score-scale drift and does not require
# calibration. The LLM judge (curation_judge.py) only sees the gray tier.
# ---------------------------------------------------------------------------

# Default tier boundaries (percentiles within the batch of PROMOTE-eligible
# candidates after dedup/period/quality gates). Tunable via calibration.
DEFAULT_AUTO_PROMOTE_PERCENTILE = 0.80  # top 20% of ranked candidates
DEFAULT_AUTO_REJECT_PERCENTILE = 0.40   # bottom 60% → auto-reject
# Between 0.40 and 0.80 = gray tier → LLM judge


@dataclass(frozen=True)
class CurationTier:
    """Resultado de la clasificación por ranking de un documento.

    El tier indica qué camino sigue el documento en la cascada:
      - auto_promote: pasa directo (top del ranking, sin LLM)
      - auto_reject: rechazado (bottom del ranking, sin LLM)
      - gray: zona gris → LLM judge con think_mode
      - hard_reject: rechazado por gate duro (duplicado, fuera de período,
        sin texto, quality gate del scraper) — no depende del ranking
    """
    document_id: str
    tier: str  # "auto_promote" | "auto_reject" | "gray" | "hard_reject"
    rank: int  # 1-based rank within the batch (1 = best)
    percentile: float  # 0.0 (worst) .. 1.0 (best) within the batch
    score: float  # promotion_score value (for transparency/debugging)
    reason: str
    scores: ScoreBundle
    decision: ReporterDecision  # the underlying ReporterDecision
    duplicate_of: str | None


def _percentile_rank(values: list[float]) -> list[float]:
    """Convierte scores absolutos a percentiles dentro del lote.

    Devuelve, para cada valor, su posición relativa en [0.0, 1.0].
    1.0 = el mejor del lote, 0.0 = el peor. Empates se resuelven por orden
    de aparición (estable).
    """
    if not values:
        return []
    sorted_vals = sorted(values, reverse=True)
    n = len(values)
    out: list[float] = []
    for v in values:
        # Posición del valor en el ranking descendente, normalizada.
        rank_pos = sorted_vals.index(v)
        out.append(1.0 - (rank_pos / max(n - 1, 1)))
    return out


def classify_tiers(
    decisions: list[ReporterDocumentDecision],
    *,
    auto_promote_percentile: float = DEFAULT_AUTO_PROMOTE_PERCENTILE,
    auto_reject_percentile: float = DEFAULT_AUTO_REJECT_PERCENTILE,
) -> list[CurationTier]:
    """Clasifica decisiones de curación en tiers por ranking del lote.

    Solo los documentos con decision=PROMOTE y sin duplicate_of entran al
    ranking. Los demás (DUPLICATE, DEFER, IRRELEVANT, INSUFFICIENT_EVIDENCE,
    REPORTER_ONLY) son hard_reject — no dependen del ranking.

    Args:
        decisions: lista de ReporterDocumentDecision (salida de curate_documents).
        auto_promote_percentile: percentil mínimo para auto_promote (0.80 = top 20%).
        auto_reject_percentile: percentil máximo para auto_reject (0.40 = bottom 60%).

    Returns:
        Lista de CurationTier en el mismo orden que `decisions`.
    """
    # 1. Identificar candidatos al ranking: PROMOTE, no duplicado, en período.
    candidate_indices = [
        i for i, d in enumerate(decisions)
        if d.decision == ReporterDecision.PROMOTE and d.duplicate_of is None
    ]
    # 2. Calcular scores y percentiles solo sobre candidatos.
    candidate_scores = [promotion_score(decisions[i].scores) for i in candidate_indices]
    candidate_percentiles = _percentile_rank(candidate_scores)

    # 3. Asignar tier por percentil.
    tiers: list[CurationTier] = []
    cand_pos = 0
    for i, d in enumerate(decisions):
        if i in candidate_indices:
            pct = candidate_percentiles[cand_pos]
            # rank es 1-based dentro de los candidatos (1 = mejor score)
            rank = sorted(range(len(candidate_scores)),
                          key=lambda k: -candidate_scores[k]).index(cand_pos) + 1
            score = candidate_scores[cand_pos]
            if pct >= auto_promote_percentile:
                tier = "auto_promote"
                reason = f"Top del ranking (percentil {pct:.2f}, score {score:.2f})."
            elif pct < auto_reject_percentile:
                tier = "auto_reject"
                reason = f"Bottom del ranking (percentil {pct:.2f}, score {score:.2f})."
            else:
                tier = "gray"
                reason = f"Zona gris (percentil {pct:.2f}, score {score:.2f}) → LLM judge."
            tiers.append(CurationTier(
                document_id=d.document_id, tier=tier, rank=rank,
                percentile=round(pct, 4), score=round(score, 4),
                reason=reason, scores=d.scores,
                decision=d.decision, duplicate_of=d.duplicate_of,
            ))
            cand_pos += 1
        else:
            # hard_reject: gate duro, no depende del ranking
            reason = f"Gate duro: {d.decision.value}"
            if d.duplicate_of:
                reason += f" (duplicado de {d.duplicate_of})"
            tiers.append(CurationTier(
                document_id=d.document_id, tier="hard_reject", rank=0,
                percentile=0.0, score=promotion_score(d.scores),
                reason=reason, scores=d.scores,
                decision=d.decision, duplicate_of=d.duplicate_of,
            ))
    return tiers


def _bounded_float(raw: Any, fallback: float) -> float:
    """float(raw) clamped to [0,1]; fallback on non-numeric input.

    quality_score/max_cosine/LLM judge values can arrive as strings
    ('significant', 'high') from upstream classifiers — one unparseable
    value used to abort the whole curation batch."""
    try:
        return max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        return fallback


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

    def _parse(raw: str):
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        # Fechas naive ("2026-09-10", típicas de trafilatura) se interpretan
        # UTC — comparar naive vs aware lanza TypeError fuera del try.
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

    try:
        return _parse(start) <= _parse(value) < _parse(end)
    except (ValueError, TypeError):
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
    known_url_hashes: dict[str, str] | None = None,
    novelty_hints: dict[str, dict] | None = None,
) -> list[ReporterDocumentDecision]:
    historical_documents = historical_documents or []
    novelty_hints = novelty_hints or {}
    document_embeddings = document_embeddings or {}
    interest_embeddings = interest_embeddings or []
    historical_embeddings = historical_embeddings or []
    # Main-corpus URL → normalized content hash. Used to detect identical
    # re-downloads deterministically. A same-URL document with DIFFERENT
    # content is NOT a duplicate: it may be a new study or an updated
    # version — both carry value, so it flows to the normal decision path.
    known_url_hashes = known_url_hashes or {}
    by_url: dict[str, str] = {}
    by_hash: dict[str, str] = {}

    # Pre-normalize embedding matrices once — the per-doc loop used to
    # recompute pure-Python cosines for every (doc, historical) pair
    # (~1.5G interpreted ops per corpus). Vectorized this is milliseconds.
    import numpy as _np

    def _norm_rows(embs: list[list[float]]) -> "_np.ndarray | None":
        if not embs:
            return None
        m = _np.asarray(embs, dtype=_np.float32)
        norms = _np.linalg.norm(m, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        return m / norms

    _hist_mat = _norm_rows(historical_embeddings)
    _interest_mat = _norm_rows(interest_embeddings)

    def _max_cosine(emb: list[float], mat: "_np.ndarray | None") -> tuple[float, int | None]:
        """Return (max cosine, index of the matching row) — index needed so
        the novelty gate can confirm lexical overlap against the matched
        historical document's text."""
        if mat is None:
            return 0.0, None
        d = _np.asarray(emb, dtype=_np.float32)
        n = _np.linalg.norm(d)
        if n == 0:
            return 0.0, None
        sims = mat @ (d / n)
        idx = int(sims.argmax())
        return float(sims[idx]), idx
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
        # Hash normalizado consistente con known_url_hashes (Tier 0 lo
        # persiste en document_metadata; antes se comparaba contra
        # sha256(text)[:32] — formato distinto, nunca matcheaba).
        norm_hash = str(document.get("normalized_hash") or normalized_hash(text))
        score = _bounded_float(document.get("quality_score"), 0.0)
        domain = str(document.get("source_domain") or "")
        doc_emb = document_embeddings.get(document_id)
        hint = novelty_hints.get(document_id)

        # --- Relevance: cosine similarity with interest embeddings (semantic) ---
        if doc_emb and _interest_mat is not None:
            # Clamp: floating-point cosine can exceed [0,1] by epsilon and
            # one out-of-range score aborts the whole curation batch.
            relevance = max(0.0, min(1.0, _max_cosine(doc_emb, _interest_mat)[0]))
        elif not interests:
            relevance = 0.5
        else:
            relevance = min(1.0, 0.3 + 0.2 * sum(1 for term in interests if term.lower() in (document.get("title", "") + " " + text).lower()))

        # --- Novelty: cosine distance to historical embeddings (semantic) ---
        # Un novelty_hint precomputado (Tier 0, post-drain) reemplaza la
        # búsqueda contra toda la matriz histórica: mismo valor, sin cargarla.
        hint_text = ""
        hint_doc_id: Any = None
        if hint is not None:
            novelty_via_embedding = True
            hist_sim = _bounded_float(hint.get("max_cosine"), 0.0)
            hist_idx = None
            hint_text = str(hint.get("nearest_text") or "")
            hint_doc_id = hint.get("nearest_doc_id")
            novelty = max(0.0, min(1.0, 1.0 - hist_sim))
        elif doc_emb and _hist_mat is not None:
            novelty_via_embedding = True
            hist_sim, hist_idx = _max_cosine(doc_emb, _hist_mat)
            novelty = max(0.0, min(1.0, 1.0 - hist_sim))
        else:
            novelty_via_embedding = False
            hist_idx = None
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
        elif url and url in known_url_hashes and norm_hash == known_url_hashes[url]:
            # Identical re-download: same canonical URL AND same normalized
            # content as an already-approved main document. Only this exact
            # case is a duplicate — a same-URL document with different
            # content may be a new study or an updated version and must
            # survive curation.
            decision, duplicate_of, reason = (
                ReporterDecision.DUPLICATE, "__main__",
                "Re-descarga idÃ©ntica de un documento ya aprobado en main.")
        elif url and url in by_url:
            decision, duplicate_of, reason = (
                ReporterDecision.DUPLICATE, by_url[url],
                "Canonical URL duplicada.")
        elif norm_hash in by_hash:
            decision, duplicate_of, reason = ReporterDecision.DUPLICATE, by_hash[norm_hash], "Content hash duplicado."
        elif relevance < 0.3:
            decision, reason = ReporterDecision.REPORTER_ONLY, "Relevancia baja para los intereses configurados; se conserva para anÃ¡lisis neutral."
        elif novelty < 0.05:
            # Near-identical embedding (>0.95 cosine to a historical doc) is
            # NOT sufficient evidence of duplication on its own: a single
            # doc-level vector is dominated by site boilerplate, so distinct
            # articles in a recurring series (weekly CVE alerts, interview
            # spotlights, product announcements sharing a template) measure
            # ~identical semantically while carrying different content.
            # Confirm with lexical overlap >= 0.85 against the matched
            # historical document's text before discarding; otherwise keep
            # the document for the reporter corpus. When novelty came from
            # the lexical fallback it already IS a token overlap >0.95, so
            # no extra confirmation is needed. When no aligned historical
            # text is available to verify against, keep the document — an
            # unverifiable fuzzy match must not cause deletion.
            if novelty_via_embedding:
                matched_text = ""
                matched_doc_id: Any = None
                if hint is not None:
                    matched_text = hint_text
                    matched_doc_id = hint_doc_id
                elif hist_idx is not None and hist_idx < len(historical_documents):
                    matched = historical_documents[hist_idx]
                    matched_text = str(matched.get("text", "") or "")
                    matched_doc_id = matched.get("document_id")
                if matched_text and _lexical_similarity(text, matched_text) >= 0.85:
                    decision, duplicate_of, reason = (
                        ReporterDecision.DUPLICATE,
                        str(matched_doc_id) if matched_doc_id else "__main__",
                        "Contenido casi idéntico al corpus histórico (embedding + léxico).")
                else:
                    decision, reason = ReporterDecision.REPORTER_ONLY, (
                        "Similitud semántica alta con el corpus histórico (template/sitio "
                        "compartido) pero contenido léxico distinto; se conserva.")
            else:
                decision, reason = ReporterDecision.DUPLICATE, "Contenido casi idÃ©ntico al corpus histÃ³rico."
        else:
            decision = ReporterDecision.PROMOTE
            reason = "Contenido relevante y suficientemente novedoso; requiere revisiÃ³n humana para promociÃ³n."
        by_url[url] = document_id
        by_hash[norm_hash] = document_id
        semantic = batch_results[index] if index < len(batch_results) else (classifier(document) if classifier else {})
        if semantic:
            values = {key: _bounded_float(semantic.get(key, value), value) for key, value in {
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

