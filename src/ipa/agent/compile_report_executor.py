"""compile_report executor — agent-invoked report compilation (Opción A).

The agent gathers documents (via search_corpus + research_topic) and then
calls this tool to compile a fine-grained report from those specific
documents. The tool runs deterministic curation, topic discovery, and report
generation over the selected documents — it does NOT ingest, promote, or
mutate the main corpus.

Flow:
  1. Load specified documents from the corpus DocumentStore
  2. Build document representations (title, abstract, keywords)
  3. Load embeddings from LanceDB for semantic curation/clustering
  4. Run curate_documents() — deterministic scoring
  5. Run discover_topics() — deterministic clustering
  6. Run group_topics_into_categories() — deterministic grouping
  7. Build and write report to an isolated output directory
  8. Persist decisions/topics to a ReporterStore
  9. Return structured ToolCall/ToolResult with report metadata + citations

Architectural constraints (DEC-003, PAT-001, PAT-003):
  - DocumentStore remains canonical; this tool only reads.
  - No ingestion, no promotion, no main-corpus mutation.
  - Output is an isolated report artifact under outputs/reporter/agent/.
  - Provenance is preserved from document_sources.
  - The tool is deterministic — no LLM calls (future: optional LLM labels).
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agent_memory import _compact_stamp
from .agent_tools import ToolCall, ToolResult, ToolContext, _result_hash, _now


@dataclass(frozen=True)
class CompileReportResult:
    """Structured output of a compile_report execution."""
    report_id: str
    report_path: str
    document_count: int
    category_count: int
    curation_summary: dict[str, int]
    categories: list[dict[str, Any]]
    source_refs: list[dict[str, Any]]
    output_dir: str
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None and self.document_count > 0


def _period_from_args(args: dict[str, Any]) -> tuple[str, str, str]:
    """Build a (start, end, label) period tuple from tool arguments."""
    now = datetime.now(timezone.utc)
    start = str(args.get("period_start", "") or "").strip()
    end = str(args.get("period_end", "") or "").strip()
    label = str(args.get("period_label", "") or "").strip()
    if not start:
        start = now.replace(day=1).strftime("%Y-%m-%dT00:00:00Z")
    if not end:
        end = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    if not label:
        label = now.strftime("%Y-%m-%d-%H%M%S")
    return start, end, label


def _load_document_for_report(
    store: Any,
    doc_id: str,
    source: dict | None,
) -> dict[str, Any] | None:
    """Load a document from DocumentStore and build a report-ready dict."""
    from ipa.reporter.reporter_representation import build_representation

    doc = store.get_document(doc_id)
    if doc is None:
        return None
    chunks = list(store.get_chunks(doc_id))
    # Use centroid chunks if available, else first 5000 chars
    centroid = store.get_centroid(doc_id)
    representative_text = ""
    if centroid:
        parts = []
        total_len = 0
        for cid in centroid:
            chunk = store.get_chunk(cid)
            if chunk and chunk.text:
                parts.append(chunk.text)
                total_len += len(chunk.text)
                if total_len >= 5000:
                    break
        if parts:
            representative_text = "\n\n".join(parts)[:5000]
    if not representative_text:
        representative_text = doc.text[:5000] if doc.text else ""

    # Get artifact_id from the documents table
    row = store._conn.execute(
        "SELECT artifact_id FROM documents WHERE document_id = ?",
        (doc_id,),
    ).fetchone()
    artifact_id = row[0] if row else doc_id

    title = (source or {}).get("title", "") or doc_id
    # Try to extract a better title from the text
    representation = build_representation(doc.text or "", title)
    content_hash = chunks[0].content_hash if chunks else ""

    return {
        "document_id": doc_id,
        "artifact_id": artifact_id,
        "title": representation.title or title,
        "title_source": representation.title_source,
        "title_confidence": representation.title_confidence,
        "abstract": representation.abstract,
        "keywords": list(representation.keywords),
        "representation_text": representation.embedding_text,
        "text": doc.text or "",
        "representative_text": representative_text,
        "content_hash": content_hash,
        "source_url": (source or {}).get("source_url", ""),
        "canonical_url": (source or {}).get("source_url", ""),
        "source_domain": (source or {}).get("source_domain", ""),
        "published_at": None,  # DocumentStore doesn't track published_at
        "quality_score": (source or {}).get("quality_score", 0.0),
        "mime_type": doc.mime_type,
    }


def execute_compile_report(
    arguments: dict[str, Any],
    ctx: ToolContext,
    *,
    session_id: str,
    episode_id: str,
) -> tuple[ToolCall, ToolResult, CompileReportResult]:
    """Compile a fine-grained report from a set of corpus documents.

    The agent selects documents (via search_corpus + research_topic) and
    calls this tool to produce a structured report. The tool is deterministic:
    it runs curation, topic discovery, and report generation without an LLM.

    Arguments:
        document_ids: list[str] (required) — document IDs from the corpus
        topic: str (optional) — label/description for the report
        period_start: str (optional, ISO) — report period start
        period_end: str (optional, ISO) — report period end
        period_label: str (optional) — report period label
        interests: list[str] (optional) — interest terms for curation scoring
        similarity_threshold: float (optional, default 0.52) — topic clustering threshold
        min_documents: int (optional, default 2) — min docs per topic cluster
        allow_singletons: bool (optional, default True) — allow single-doc topics
        output_dir: str (optional) — override output directory

    Returns:
        (ToolCall, ToolResult, CompileReportResult) — contract records + structured output.
    """
    from ipa.reporter.reporter_curation import curate_documents
    from ipa.reporter.reporter_topics import discover_topics, group_topics_into_categories
    from ipa.reporter.reporter_report import build_report, write_report
    from ipa.reporter.reporter_store import ReporterStore
    from ipa.reporter.reporter_contracts import sha256_hash

    call_id = f"tool_call:{_compact_stamp()}"
    result_id = f"tool_result:{_compact_stamp()}"
    started = _now()
    t0 = time.monotonic()

    # --- Parse and validate arguments ---
    document_ids = arguments.get("document_ids", [])
    if not isinstance(document_ids, list) or not document_ids:
        raise ValueError("compile_report requires a non-empty 'document_ids' list")
    document_ids = [str(did).strip() for did in document_ids if str(did).strip()]
    if not document_ids:
        raise ValueError("compile_report requires at least one valid document_id")
    if len(document_ids) > 200:
        document_ids = document_ids[:200]  # bounded

    topic_label = str(arguments.get("topic", "") or "").strip()
    interests = tuple(str(i) for i in (arguments.get("interests") or []))
    similarity_threshold = float(arguments.get("similarity_threshold", 0.52))
    min_documents = int(arguments.get("min_documents", 2))
    allow_singletons = bool(arguments.get("allow_singletons", True))
    period_start, period_end, period_label = _period_from_args(arguments)

    store = ctx.document_store()
    if store is None:
        raise ValueError("corpus document store is not available; set corpus_dir in ToolContext")

    # --- Load all document sources for provenance ---
    all_sources = store.all_sources()

    # --- Build document dicts for curation/clustering ---
    documents: list[dict[str, Any]] = []
    skipped: list[str] = []
    for doc_id in document_ids:
        source = all_sources.get(doc_id)
        doc_dict = _load_document_for_report(store, doc_id, source)
        if doc_dict is None:
            skipped.append(doc_id)
        else:
            documents.append(doc_dict)

    if not documents:
        raise ValueError(
            f"no documents could be loaded from corpus; skipped {len(skipped)} IDs"
        )

    # --- Load embeddings from LanceDB for semantic curation/clustering ---
    doc_embeddings: dict[str, list[float]] = {}
    interest_embeddings: list[list[float]] = []
    lance = ctx.lance_index()
    if lance is not None and lance.is_queryable():
        try:
            doc_embeddings = lance.document_embeddings()
        except Exception:
            pass  # non-fatal — fall back to lexical similarity

    # Embed interests if available
    if interests and doc_embeddings:
        try:
            embed = ctx.embedding_adapter()
            interest_embeddings = embed.embed_texts(list(interests))
        except Exception:
            pass  # non-fatal

    # --- Curation (deterministic, no LLM) ---
    report_id = "report:agent:" + hashlib.sha256(
        ("|".join(sorted(document_ids)) + period_label).encode()
    ).hexdigest()[:16]

    decisions = curate_documents(
        documents, report_id, period_start, period_end,
        interests, quality_threshold=0.25,
        document_embeddings=doc_embeddings,
        interest_embeddings=interest_embeddings,
    )

    selected = [
        doc for doc, decision in zip(documents, decisions)
        if decision.decision.value not in {"duplicate", "irrelevant", "insufficient_evidence"}
    ]

    # --- Topic clustering (deterministic, no LLM) ---
    selected_embeddings = [doc_embeddings.get(doc["document_id"]) for doc in selected]
    if any(e is None for e in selected_embeddings):
        selected_embeddings = None

    categories = discover_topics(
        selected,
        similarity_threshold=similarity_threshold,
        min_documents=min_documents,
        allow_singletons=allow_singletons,
        embeddings=selected_embeddings,
        report_id=report_id,
    )

    parent_categories = group_topics_into_categories(categories, llm_grouper=None)

    # --- Build report ---
    source_refs = [
        {"source_id": doc["document_id"], "source_type": "document"}
        for doc in selected
    ]

    uncertainties = []
    if skipped:
        uncertainties.append(f"{len(skipped)} document(s) could not be loaded from the corpus.")
    if not selected:
        uncertainties.append("No documents passed curation; report is empty.")

    report = build_report(
        report_id, "agent",
        {"start": period_start, "end": period_end, "label": period_label},
        categories, [d.to_dict() for d in decisions], source_refs,
        uncertainties=uncertainties or None,
        parent_categories=parent_categories,
    )

    # --- Write report to isolated output directory ---
    default_output = Path("outputs/reporter/agent") / period_label
    output_dir = Path(arguments.get("output_dir") or default_output)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path, md_path = write_report(report, output_dir)

    # --- Persist to ReporterStore ---
    reporter_db_path = output_dir / "reporter.db"
    reporter_store = ReporterStore(reporter_db_path)
    try:
        for decision in decisions:
            reporter_store.put_decision(decision, period_start)
        for category in categories:
            reporter_store.put_topic(category["category_id"], report_id, category)
            for did in category["document_ids"]:
                reporter_store.put_topic_document(category["category_id"], did, 1.0, "Membership by connected similarity component")
        corpus_fingerprint = sha256_hash("|".join(sorted(d.get("content_hash", "") for d in documents)))
        reporter_store.put_run(
            report_id, "agent", period_start, period_end,
            report["status"], "agent-compile-report-v1", corpus_fingerprint,
            str(json_path), report["generation"]["generated_at"],
        )
        reporter_store.commit()
    finally:
        reporter_store.close()

    # --- Build result ---
    curation_summary = report.get("curation_summary", {})

    result_dict = {
        "report_id": report_id,
        "report_path": str(json_path),
        "markdown_path": str(md_path),
        "output_dir": str(output_dir),
        "document_count": len(documents),
        "selected_count": len(selected),
        "skipped_count": len(skipped),
        "category_count": len(categories),
        "parent_category_count": len(parent_categories),
        "curation_summary": curation_summary,
        "categories": [
            {
                "category_id": c["category_id"],
                "label": c["label"],
                "document_count": c["document_count"],
                "document_ids": c["document_ids"],
            }
            for c in categories
        ],
        "uncertainties": report.get("uncertainties", []),
    }

    tool_source_refs = [
        {"source_id": doc["document_id"], "source_type": "document",
         "content_hash": doc.get("content_hash")}
        for doc in selected
    ]

    elapsed = int((time.monotonic() - t0) * 1000)
    call = ToolCall(
        tool_call_id=call_id, session_id=session_id, episode_id=episode_id,
        tool_name="compile_report",
        arguments={
            "document_ids": document_ids,
            "topic": topic_label or None,
            "period_label": period_label,
            "interests": list(interests) or None,
        },
        called_at=started, status="completed",
    )
    result = ToolResult(
        tool_result_id=result_id, tool_call_id=call_id, session_id=session_id,
        tool_name="compile_report", result=result_dict,
        result_hash=_result_hash(result_dict),
        source_refs=tool_source_refs,
        started_at=started, completed_at=_now(),
        elapsed_ms=elapsed, status="completed",
    )

    compile_result = CompileReportResult(
        report_id=report_id,
        report_path=str(json_path),
        document_count=len(documents),
        category_count=len(categories),
        curation_summary=curation_summary,
        categories=categories,
        source_refs=tool_source_refs,
        output_dir=str(output_dir),
    )

    return call, result, compile_result


__all__ = ["CompileReportResult", "execute_compile_report"]
