"""Deterministic agent tools (Fase 1).

Each tool is a pure function that receives bounded arguments, queries existing
stores, and returns a structured result. Tools never call an LLM, never reason,
and never decide — they are deterministic adapters over the corpus and memory
(DEC-002, roadmap Fase 1).

Every tool execution emits a ``ToolCall`` and ``ToolResult`` contract record
validatable via ``scripts/validation/validate_agent_contract.py``.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .agent_memory import AgentMemory, _compact_stamp, content_hash


# ---------------------------------------------------------------------------
# Tool registry and execution framework
# ---------------------------------------------------------------------------

TOOL_NAMES = frozenset({
    "search_corpus",
    "list_topics",
    "get_topic_info",
    "recall_conversation",
    "research_topic",
    "compile_report",
})


@dataclass(frozen=True)
class ToolCall:
    """Contract record: a tool was invoked with bounded arguments."""
    tool_call_id: str
    session_id: str
    episode_id: str
    tool_name: str
    arguments: dict[str, Any]
    called_at: str
    status: str
    error: str | None = None

    def to_contract(self) -> dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "session_id": self.session_id,
            "episode_id": self.episode_id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "called_at": self.called_at,
            "status": self.status,
            "error": self.error,
        }


@dataclass(frozen=True)
class ToolResult:
    """Contract record: the structured output of a tool execution."""
    tool_result_id: str
    tool_call_id: str
    session_id: str
    tool_name: str
    result: dict[str, Any]
    result_hash: str
    source_refs: list[dict[str, Any]]
    started_at: str
    completed_at: str
    elapsed_ms: int
    status: str
    error: str | None = None

    def to_contract(self) -> dict[str, Any]:
        return {
            "tool_result_id": self.tool_result_id,
            "tool_call_id": self.tool_call_id,
            "session_id": self.session_id,
            "tool_name": self.tool_name,
            "result": self.result,
            "result_hash": self.result_hash,
            "source_refs": list(self.source_refs),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "elapsed_ms": self.elapsed_ms,
            "status": self.status,
            "error": self.error,
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _result_hash(result: dict[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(result, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


@dataclass
class ToolContext:
    """Dependencies injected into every tool execution.

    Tools are deterministic functions over these stores. The context makes the
    dependencies explicit and testable — no hidden globals, no service locators.
    """
    memory: AgentMemory
    corpus_dir: str | Path | None = None
    # Lazy-loaded adapters (cached for reuse across calls in a session)
    _document_store: Any = field(default=None, repr=False)
    _lance_index: Any = field(default=None, repr=False)
    _embedding_adapter: Any = field(default=None, repr=False)
    _bm25_index: Any = field(default=None, repr=False)

    def document_store(self):
        if self._document_store is None and self.corpus_dir:
            from ipa.storage.document_store import DocumentStore
            self._document_store = DocumentStore(Path(self.corpus_dir) / "document_store.db")
        return self._document_store

    def lance_index(self):
        if self._lance_index is None and self.corpus_dir:
            from ipa.indexes.lancedb_index import LanceDBIndex
            self._lance_index = LanceDBIndex(Path(self.corpus_dir) / "vector" / "lancedb")
        return self._lance_index

    def embedding_adapter(self):
        if self._embedding_adapter is None:
            from ipa.indexes.embedding_adapter import EmbeddingAdapter
            self._embedding_adapter = EmbeddingAdapter(show_progress=False)
        return self._embedding_adapter

    def bm25_index(self):
        if self._bm25_index is None and self.corpus_dir:
            from ipa.indexes.bm25_index import BM25Index
            self._bm25_index = BM25Index(Path(self.corpus_dir) / "bm25_index.db")
        return self._bm25_index

    def close(self):
        for attr in ("_document_store", "_lance_index", "_embedding_adapter", "_bm25_index"):
            obj = getattr(self, attr, None)
            if obj is not None and hasattr(obj, "close"):
                try:
                    obj.close()
                except Exception:
                    pass
            setattr(self, attr, None)


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _execute_tool(
    tool_name: str,
    arguments: dict[str, Any],
    ctx: ToolContext,
    session_id: str,
    episode_id: str,
) -> tuple[ToolCall, ToolResult]:
    """Execute a tool and emit contract-shaped call + result records."""
    call_id = f"tool_call:{_compact_stamp()}"
    result_id = f"tool_result:{_compact_stamp()}"
    started = _now()
    t0 = time.monotonic()

    # Executor-style tools have richer signatures (session/episode context,
    # structured domain results). Dispatch them directly — the contract
    # records are produced by the executors themselves. Argument validation
    # errors raise inside the executor; wrap them into failed contracts so
    # callers always get (ToolCall, ToolResult) back.
    if tool_name in ("compile_report", "research_topic"):
        try:
            if tool_name == "compile_report":
                from ipa.agent.compile_report_executor import execute_compile_report
                call, result, _structured = execute_compile_report(
                    arguments, ctx, session_id=session_id, episode_id=episode_id,
                )
            else:
                from ipa.agent.research_executor import execute_research
                _subs = arguments.get("sub_queries")
                call, result, _structured = execute_research(
                    str(arguments.get("query", "") or ""), ctx,
                    session_id=session_id, episode_id=episode_id,
                    max_urls=int(arguments.get("max_urls", 5)),
                    max_seconds=int(arguments.get("max_seconds", 120)),
                    freshness=str(arguments.get("freshness", "lenient")),
                    sub_queries=[str(s) for s in _subs] if isinstance(_subs, list) else None,
                )
            return call, result
        except Exception as exc:
            completed = _now()
            call = ToolCall(
                tool_call_id=call_id, session_id=session_id, episode_id=episode_id,
                tool_name=tool_name, arguments=arguments, called_at=started,
                status="failed", error=str(exc),
            )
            result = ToolResult(
                tool_result_id=result_id, tool_call_id=call_id, session_id=session_id,
                tool_name=tool_name, result={}, result_hash=_result_hash({}),
                source_refs=[], started_at=started, completed_at=completed,
                elapsed_ms=int((time.monotonic() - t0) * 1000),
                status="failed", error=str(exc),
            )
            return call, result

    tool_fn = _TOOL_IMPLEMENTATIONS.get(tool_name)
    if tool_fn is None:
        call = ToolCall(
            tool_call_id=call_id, session_id=session_id, episode_id=episode_id,
            tool_name=tool_name, arguments=arguments, called_at=started,
            status="failed", error=f"unknown tool: {tool_name}",
        )
        result = ToolResult(
            tool_result_id=result_id, tool_call_id=call_id, session_id=session_id,
            tool_name=tool_name, result={}, result_hash=_result_hash({}),
            source_refs=[], started_at=started, completed_at=_now(),
            elapsed_ms=int((time.monotonic() - t0) * 1000),
            status="failed", error=f"unknown tool: {tool_name}",
        )
        return call, result

    # Mark call as running
    call = ToolCall(
        tool_call_id=call_id, session_id=session_id, episode_id=episode_id,
        tool_name=tool_name, arguments=arguments, called_at=started,
        status="running",
    )

    try:
        result_dict, source_refs = tool_fn(arguments, ctx)
        elapsed = int((time.monotonic() - t0) * 1000)
        completed = _now()
        call = ToolCall(
            tool_call_id=call_id, session_id=session_id, episode_id=episode_id,
            tool_name=tool_name, arguments=arguments, called_at=started,
            status="completed",
        )
        result = ToolResult(
            tool_result_id=result_id, tool_call_id=call_id, session_id=session_id,
            tool_name=tool_name, result=result_dict, result_hash=_result_hash(result_dict),
            source_refs=source_refs, started_at=started, completed_at=completed,
            elapsed_ms=elapsed, status="completed",
        )
    except Exception as exc:
        elapsed = int((time.monotonic() - t0) * 1000)
        completed = _now()
        call = ToolCall(
            tool_call_id=call_id, session_id=session_id, episode_id=episode_id,
            tool_name=tool_name, arguments=arguments, called_at=started,
            status="failed", error=str(exc),
        )
        result = ToolResult(
            tool_result_id=result_id, tool_call_id=call_id, session_id=session_id,
            tool_name=tool_name, result={}, result_hash=_result_hash({}),
            source_refs=[], started_at=started, completed_at=completed,
            elapsed_ms=elapsed, status="failed", error=str(exc),
        )

    return call, result


# --- search_corpus ---------------------------------------------------------

def _date_where(date_from: str, date_to: str) -> str | None:
    """SQL pre-filter on the LanceDB published_at metadata column.

    Documents without a date are kept (same permissive semantics as before);
    values are sanitized to ISO-date characters only.
    """
    parts = []
    for op, raw in ((">=", date_from), ("<=", date_to)):
        v = re.sub(r"[^0-9TtZz:\-.+ ]", "", raw.strip())[:25]
        if v:
            parts.append(f"published_at {op} '{v}'")
    if not parts:
        return None
    return "(published_at = '' OR published_at IS NULL OR (" + " AND ".join(parts) + "))"


def _search_corpus(args: dict[str, Any], ctx: ToolContext) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Search the corpus via LanceDB hybrid search (dense + FTS + sparse,
    3-way RRF) — same pipeline as the dashboard auto-retrieval. BM25Index
    stays the first_queryable fallback for corpora without a vector index.

    Arguments:
        query: str (required) — natural language query
        limit: int (default 10) — max results
        date_from: str (optional, ISO) — solo documentos publicados desde
        date_to: str (optional, ISO) — solo documentos publicados hasta
    """
    query = args.get("query", "").strip()
    if not query:
        raise ValueError("search_corpus requires a non-empty 'query' argument")
    limit = int(args.get("limit", 10))
    limit = max(1, min(limit, 50))
    date_from = str(args.get("date_from", "")).strip()
    date_to = str(args.get("date_to", "")).strip()

    store = ctx.document_store()
    if store is None:
        raise ValueError("corpus document store is not available; set corpus_dir in ToolContext")

    # Over-fetch for RRF fusion, doc-level dedup and optional reranking.
    fetch_limit = limit * 3
    hits = []
    retrieval_backend = "none"
    lance = ctx.lance_index()
    if lance is not None and lance.is_queryable():
        embed = ctx.embedding_adapter()
        dense_vec, sparse_weights = embed.embed_query_hybrid(query)
        hits = lance.search_hybrid(
            query, dense_vec, limit=fetch_limit,
            query_sparse=sparse_weights,
            where=_date_where(date_from, date_to),
        )
        retrieval_backend = "lancedb_hybrid"
    else:
        bm25 = ctx.bm25_index()
        if bm25 is not None and bm25.is_queryable():
            hits = bm25.search(query, limit=fetch_limit)
            retrieval_backend = "bm25"
        else:
            raise ValueError(
                "no queryable index available; need either LanceDB or BM25 in corpus_dir"
            )

    results = []
    seen_docs: set[str] = set()
    for hit in hits:
        chunk = store.get_chunk(hit.chunk_id)
        doc_id = chunk.document_id if chunk else "unknown"
        # Dedup por documento: cobertura de fuentes distintas sobre varios
        # chunks del mismo doc (mismo criterio que el retrieval del dashboard).
        if doc_id in seen_docs:
            continue
        seen_docs.add(doc_id)
        src = store.get_source(doc_id) if doc_id != "unknown" else None
        text_preview = (chunk.text[:200] + "...") if chunk and len(chunk.text) > 200 else (chunk.text if chunk else "")
        results.append({
            "chunk_id": hit.chunk_id,
            "document_id": doc_id,
            "score": round(hit.score, 4),
            "retrieval_backend": hit.retrieval_backend,
            "published_at": store.document_stored_at(doc_id) if doc_id != "unknown" else None,
            "source_domain": (src or {}).get("source_domain"),
            "provenance": (src or {}).get("provenance"),
            "text_preview": text_preview,
            "_text": chunk.text if chunk else "",
            "_content_hash": chunk.content_hash if chunk else None,
        })

    # Stage-2 rerank (default activo, opt-out IPA_RERANK=0): cross-encoder
    # sobre el texto completo del chunk — el preview de 200 chars no alcanza
    # para rerankear.
    from ipa.indexes.reranker_adapter import maybe_rerank
    results = maybe_rerank(query, results, limit, text_key="_text")[:limit]
    # source_refs se derivan del resultado final (post-rerank/dedup).
    source_refs = [
        {
            "source_id": r["chunk_id"],
            "source_type": "chunk",
            "content_hash": r["_content_hash"],
        }
        for r in results
    ]
    for r in results:
        r.pop("_text", None)
        r.pop("_content_hash", None)

    return {"query": query, "hits": results, "total": len(results),
            "date_filter": {"from": date_from or None, "to": date_to or None} if (date_from or date_to) else None}, source_refs


# --- list_topics -----------------------------------------------------------

def _list_topics(args: dict[str, Any], ctx: ToolContext) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """List emergent topics from the corpus.

    Uses document centroids when available (computed by LanceDB indexing).
    Falls back to listing documents directly when centroids are not stored.
    Full topic clustering arrives in Fase 3.
    """
    limit = int(args.get("limit", 20))
    limit = max(1, min(limit, 100))

    store = ctx.document_store()
    if store is None:
        raise ValueError("corpus document store is not available; set corpus_dir in ToolContext")

    centroids = store.all_centroids()
    topics = []
    if centroids:
        for doc_id, chunk_ids in centroids.items():
            doc = store.get_document(doc_id)
            if doc is None:
                # Centroide huérfano: el doc fue tombstoneado/eliminado tras
                # indexar (repairs, dedupe). El índice derivado queda stale
                # hasta el próximo topify — filtrar al leer.
                continue
            topics.append({
                "document_id": doc_id,
                "representative_chunk_count": len(chunk_ids),
                "mime_type": doc.mime_type,
                "pages": doc.pages,
            })
            if len(topics) >= limit:
                break
    else:
        # Fallback: list documents by iterating chunks (no centroid index).
        seen_docs: set[str] = set()
        for chunk in store.all_chunks():
            if chunk.document_id not in seen_docs:
                seen_docs.add(chunk.document_id)
                doc = store.get_document(chunk.document_id)
                if doc is None:
                    continue  # chunk huérfano de doc tombstoneado
                topics.append({
                    "document_id": chunk.document_id,
                    "representative_chunk_count": 0,
                    "mime_type": doc.mime_type,
                    "pages": doc.pages,
                })
                if len(topics) >= limit:
                    break

    return {"topics": topics, "total": len(topics)}, []


# --- get_topic_info --------------------------------------------------------

def _get_topic_info(args: dict[str, Any], ctx: ToolContext) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Get details about a specific topic (document) and its chunks.

    Arguments:
        document_id: str (required) — the document to inspect
        max_chunks: int (default 5) — max chunk previews to return
    """
    doc_id = args.get("document_id", "").strip()
    if not doc_id:
        raise ValueError("get_topic_info requires a 'document_id' argument")
    max_chunks = int(args.get("max_chunks", 5))
    max_chunks = max(1, min(max_chunks, 20))

    store = ctx.document_store()
    if store is None:
        raise ValueError("corpus document store is not available; set corpus_dir in ToolContext")

    doc = store.get_document(doc_id)
    if doc is None:
        raise ValueError(f"document not found: {doc_id}")

    chunks = list(store.get_chunks(doc_id))[:max_chunks]
    chunk_previews = []
    source_refs = []
    for chunk in chunks:
        preview = chunk.text[:300] + "..." if len(chunk.text) > 300 else chunk.text
        chunk_previews.append({
            "chunk_id": chunk.chunk_id,
            "content_hash": chunk.content_hash,
            "text_preview": preview,
            "metadata": chunk.metadata,
        })
        source_refs.append({
            "source_id": chunk.chunk_id,
            "source_type": "chunk",
            "content_hash": chunk.content_hash,
        })

    centroid = store.get_centroid(doc_id)
    return {
        "document_id": doc_id,
        "mime_type": doc.mime_type,
        "pages": doc.pages,
        "parser_id": doc.parser_id,
        "total_chunks_in_store": len(list(store.get_chunks(doc_id))),
        "centroid_chunk_ids": centroid or [],
        "chunk_previews": chunk_previews,
    }, source_refs


# --- recall_conversation ---------------------------------------------------

def _recall_conversation(args: dict[str, Any], ctx: ToolContext) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Recall past episodes from agent memory.

    Arguments:
        session_id: str (optional) — restrict to a specific session
        query: str (optional) — text filter (simple substring match)
        limit: int (default 10) — max episodes to return
    """
    limit = int(args.get("limit", 10))
    limit = max(1, min(limit, 100))
    query_filter = args.get("query", "").strip().lower()
    session_filter = args.get("session_id", "").strip()

    if session_filter:
        episodes = ctx.memory.get_episodes(session_filter, limit=limit * 3)
    else:
        episodes = ctx.memory.recent_episodes(limit=limit * 3)

    results = []
    for ep in episodes:
        if query_filter and query_filter not in ep.content.lower():
            continue
        results.append({
            "episode_id": ep.episode_id,
            "session_id": ep.session_id,
            "turn_role": ep.turn_role,
            "content_preview": ep.content[:200] + "..." if len(ep.content) > 200 else ep.content,
            "content_hash": ep.content_hash,
            "created_at": ep.created_at,
        })
        if len(results) >= limit:
            break

    return {"episodes": results, "total": len(results)}, []


# --- Tool registry ---------------------------------------------------------

_TOOL_IMPLEMENTATIONS: dict[str, Callable[[dict[str, Any], ToolContext], tuple[dict[str, Any], list[dict[str, Any]]]]] = {
    "search_corpus": _search_corpus,
    "list_topics": _list_topics,
    "get_topic_info": _get_topic_info,
    "recall_conversation": _recall_conversation,
}


def execute_tool(
    tool_name: str,
    arguments: dict[str, Any],
    ctx: ToolContext,
    *,
    session_id: str,
    episode_id: str,
) -> tuple[ToolCall, ToolResult]:
    """Execute a deterministic tool and return contract-shaped records."""
    if tool_name not in TOOL_NAMES:
        raise ValueError(f"unknown tool: {tool_name}; valid tools: {sorted(TOOL_NAMES)}")
    return _execute_tool(tool_name, arguments, ctx, session_id, episode_id)


# ---------------------------------------------------------------------------
# Knowledge gap detection (deterministic scaffold for the agentic research flow)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CoverageAssessment:
    """Deterministic assessment of whether the local corpus can answer a query.

    This is the andamiaje (scaffold): a cheap deterministic check that decides
    whether the agent should trigger web research. The LLM never decides this —
    per the roadmap, scaffolding is deterministic and the LLM only classifies.
    """
    query: str
    sufficient: bool
    hit_count: int
    max_score: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "sufficient": self.sufficient,
            "hit_count": self.hit_count,
            "max_score": round(self.max_score, 4),
            "reason": self.reason,
        }


def assess_corpus_coverage(
    query: str,
    ctx: ToolContext,
    *,
    min_hits: int = 3,
    limit: int = 5,
) -> CoverageAssessment:
    """Assess whether the local corpus has enough material for a query.

    Deterministic gap detection: if the corpus returns fewer than ``min_hits``
    relevant chunks, the agent should consider web research (roadmap Fase 2:
    "si el corpus no alcanza → ResearchRequest web").
    """
    query = (query or "").strip()
    if not query:
        return CoverageAssessment(query, False, 0, 0.0, "empty query")

    try:
        result_dict, _ = _search_corpus({"query": query, "limit": limit}, ctx)
    except ValueError as exc:
        return CoverageAssessment(query, False, 0, 0.0, f"corpus unavailable: {exc}")

    hits = result_dict.get("hits", [])
    hit_count = len(hits)
    max_score = max((float(h.get("score", 0.0)) for h in hits), default=0.0)

    if hit_count == 0:
        return CoverageAssessment(query, False, 0, max_score, "no hits in local corpus")
    if hit_count < min_hits:
        return CoverageAssessment(
            query, False, hit_count, max_score,
            f"only {hit_count} hits (need {min_hits}) — corpus coverage insufficient",
        )
    return CoverageAssessment(
        query, True, hit_count, max_score,
        f"{hit_count} hits available — corpus coverage sufficient",
    )


__all__ = [
    "TOOL_NAMES",
    "ToolCall",
    "ToolResult",
    "ToolContext",
    "execute_tool",
    "CoverageAssessment",
    "assess_corpus_coverage",
]
