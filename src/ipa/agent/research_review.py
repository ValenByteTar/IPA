"""Review queue for research-rejected scraped docs + short-idle LLM re-read.

The research executor rejects scraped documents at the content stage
(quality/date/judge). Instead of discarding them outright, the text is
queued here. After a short idle window (IPA_RESEARCH_REVIEW_IDLE_SECONDS,
default 60s of no user activity) the dashboard review worker re-reads each
pending doc with the LLM and decides: promote to the main corpus or
discard permanently.

Derived + resumable: the queue is a plain SQLite table — pending rows
survive restarts and the worker is interruptible between items. Canonical
stores are untouched until a doc is promoted (FastPath + provenance +
embedding, same path as accepted research material).
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB = _ROOT / "outputs" / "agent" / "research_review.db"

_MAX_TEXT_CHARS = 40_000


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _review_id(url: str, query: str) -> str:
    digest = hashlib.sha256(f"{url}|{query}".encode("utf-8")).hexdigest()[:20]
    return f"review:{digest}"


class ResearchReviewStore:
    """SQLite queue of scraped-but-rejected docs pending an LLM re-read.

    status: pending | promoted | discarded | error
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            import os
            db_path = os.environ.get("IPA_RESEARCH_REVIEW_DB") or DEFAULT_DB
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), timeout=30)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS review_queue (
                review_id TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                title TEXT,
                query TEXT,
                reason TEXT,
                text TEXT NOT NULL,
                research_request_id TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                queued_at TEXT NOT NULL,
                decided_at TEXT,
                decision_reason TEXT,
                document_id TEXT
            )
            """
        )
        self._conn.commit()

    def enqueue(
        self,
        *,
        url: str,
        title: str | None,
        text: str,
        reason: str,
        query: str,
        research_request_id: str | None = None,
    ) -> str:
        """Queue a rejected doc for review. Idempotent per (url, query)."""
        rid = _review_id(url, query)
        self._conn.execute(
            "INSERT OR REPLACE INTO review_queue "
            "(review_id, url, title, query, reason, text, research_request_id,"
            " status, queued_at) "
            "VALUES (?,?,?,?,?,?,?, 'pending', ?)",
            (
                rid, url, title, query, reason[:400],
                text[:_MAX_TEXT_CHARS], research_request_id, _now(),
            ),
        )
        self._conn.commit()
        return rid

    def pending(self, limit: int = 3) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT review_id, url, title, query, reason, text,"
            " research_request_id, queued_at FROM review_queue"
            " WHERE status = 'pending' ORDER BY queued_at LIMIT ?",
            (limit,),
        ).fetchall()
        cols = ("review_id", "url", "title", "query", "reason", "text",
                "research_request_id", "queued_at")
        return [dict(zip(cols, row)) for row in rows]

    def mark(
        self,
        review_id: str,
        status: str,
        reason: str = "",
        document_id: str | None = None,
    ) -> None:
        self._conn.execute(
            "UPDATE review_queue SET status = ?, decided_at = ?,"
            " decision_reason = ?, document_id = COALESCE(?, document_id)"
            " WHERE review_id = ?",
            (status, _now(), reason[:400], document_id, review_id),
        )
        self._conn.commit()

    def counts(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT status, COUNT(*) FROM review_queue GROUP BY status"
        ).fetchall()
        return {status: n for status, n in rows}

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# LLM re-read: does the rejected doc deserve promotion after all?
# ---------------------------------------------------------------------------

_REVIEW_SYSTEM = (
    "Sos el curador de un corpus de conocimiento personal. Vas a leer un "
    "documento scrapeado de la web que un filtro heurístico rechazó, junto "
    "con la consulta de investigación que lo trajo. Decidí si vale la pena "
    "promoverlo al corpus principal.\n\n"
    "Promové SOLO si el documento aporta información sustancial y relevante "
    "para la consulta o para el dominio del usuario (IA, tecnología, "
    "programación). Descartá si es thin content, boilerplate, una lista de "
    "enlaces, contenido duplicado, off-topic o de baja calidad.\n\n"
    'Respondé SOLO con JSON: {"promote": true|false, "reason": "<=30 palabras"}'
)


def _review_messages(item: dict[str, Any]) -> list[dict[str, str]]:
    """Mensajes del veredicto de un doc rechazado (128 tok de salida)."""
    excerpt = (item.get("text") or "")[:6000]
    return [
        {"role": "system", "content": _REVIEW_SYSTEM},
        {"role": "user", "content": (
            f"Consulta original: {item.get('query') or '?'}\n"
            f"URL: {item.get('url') or '?'}\n"
            f"Título: {item.get('title') or '?'}\n"
            f"Motivo del rechazo heurístico: {item.get('reason') or '?'}\n\n"
            f"--- DOCUMENTO ---\n{excerpt}"
        )},
    ]


def _parse_verdict(raw: str) -> dict[str, Any]:
    from .judge import _extract_json
    parsed = _extract_json(raw)
    return {
        "promote": bool(parsed.get("promote")),
        "reason": str(parsed.get("reason", ""))[:300],
        "error": None,
    }


def review_doc_with_llm(provider: Any, item: dict[str, Any]) -> dict[str, Any]:
    """Ask the LLM whether a rejected doc deserves promotion.

    Returns {"promote": bool, "reason": str, "error": str|None}.
    Bounded call: JSON verdict only, deterministic temperature.
    """
    try:
        result = provider.generate_chat(
            _review_messages(item), max_new_tokens=128, temperature=0.0,
        )
        raw = result.text if hasattr(result, "text") else str(result or "")
        if getattr(result, "error", None):
            return {"promote": False, "reason": "", "error": result.error}
        return _parse_verdict(raw)
    except Exception as exc:
        return {"promote": False, "reason": "", "error": str(exc)[:200]}


def review_docs_with_llm(
    provider: Any, items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Veredictos para N docs en un pase batched (ExL3) o serial (Ollama).

    Mismo contrato por ítem que `review_doc_with_llm`. Con ExL3 y batch 3-4 el
    throughput agregado es ~2.4x el serial (EXP-008 §8); la cola de review es
    el caso de uso: veredictos de 128 tok, muchos ítems.
    """
    from ipa.agentic.batch_llm import generate_many

    if not items:
        return []
    pairs = generate_many(
        provider, [_review_messages(it) for it in items],
        max_new_tokens=128, temperature=0.0,
    )
    verdicts: list[dict[str, Any]] = []
    for raw, error in pairs:
        if error:
            verdicts.append({"promote": False, "reason": "", "error": error})
            continue
        try:
            verdicts.append(_parse_verdict(raw))
        except Exception as exc:
            verdicts.append({"promote": False, "reason": "", "error": str(exc)[:200]})
    return verdicts


# ---------------------------------------------------------------------------
# Promotion: same ingest path as accepted research material.
# ---------------------------------------------------------------------------

class _EmbedCtx:
    """Minimal ctx shim for research_executor._embed_new_chunks."""

    def __init__(self, corpus: Path, adapter: Any) -> None:
        self._corpus = corpus
        self._adapter = adapter
        self._store = None
        self._lance = None

    def document_store(self):
        if self._store is None:
            from ipa.storage.document_store import DocumentStore
            self._store = DocumentStore(self._corpus / "document_store.db")
        return self._store

    def lance_index(self):
        if self._lance is None:
            from ipa.indexes.lancedb_index import LanceDBIndex
            self._lance = LanceDBIndex(self._corpus / "vector" / "lancedb")
        return self._lance

    def embedding_adapter(self):
        return self._adapter


def ingest_reviewed_doc(
    corpus_dir: str | Path,
    landing_dir: str | Path,
    item: dict[str, Any],
    *,
    embedding_adapter: Any | None = None,
) -> str | None:
    """Promote a reviewed doc into the given corpus.

    DEC-003b: the caller passes the research staging corpus by default —
    the main corpus only when IPA_RESEARCH_STAGING=0. Writes the text as
    a Landing file, runs FastPath ingestion, records
    agent_research provenance, and embeds the new chunks into LanceDB when
    an embedding adapter is provided. Returns the document_id or None.
    """
    from ipa.ingestion.fast_path import FastPathRunner
    from ipa.ingestion.provenance import record_agent_research
    from ipa.storage.document_store import DocumentStore

    corpus = Path(corpus_dir)
    landing = Path(landing_dir)
    landing.mkdir(parents=True, exist_ok=True)

    fname = f"review-{(item.get('review_id') or 'x').split(':')[-1]}.md"
    target = landing / fname
    header = (
        f"# {item.get('title') or item.get('url') or 'documento revisado'}\n"
        f"Fuente: {item.get('url') or '?'}\n\n"
    )
    target.write_text(header + (item.get("text") or ""), encoding="utf-8")

    runner = FastPathRunner(
        landing_db=str(corpus / "landing.db"),
        store_db=str(corpus / "document_store.db"),
        index_db=str(corpus / "bm25_index.db"),
        landing_root=str(landing),
    )
    document_id = None
    try:
        results = runner.ingest_directory(
            str(landing), progress=False, skip_indexed=True)
        # artifact_id → source_uri via landing.db, matched on the filename.
        art_conn = sqlite3.connect(str(corpus / "landing.db"))
        try:
            row = art_conn.execute(
                "SELECT artifact_id FROM artifacts WHERE source_uri LIKE ?",
                (f"%{fname}",),
            ).fetchone()
        finally:
            art_conn.close()
        if row:
            for r in results:
                if r.artifact_id == row[0] and r.document_id:
                    document_id = r.document_id
                    break
    finally:
        runner.close()

    if document_id and item.get("url"):
        store = DocumentStore(corpus / "document_store.db")
        try:
            record_agent_research(store, document_id, item["url"])
            # Señales Tier 0: title/hash/dates + flag "corpus changed" para el
            # gate de topify (este path bypasea la promotion queue).
            try:
                from ipa.ingestion.ingest_metadata import record_ingest_metadata
                record_ingest_metadata(store, [document_id])
            except Exception:
                pass
        finally:
            store.close()
        try:
            from ipa.agentic.topic_clusters import TopicClusterStore
            _cs = TopicClusterStore()
            try:
                _cs.set_meta(f"dirty:{corpus.resolve()}", "1")
            finally:
                _cs.close()
        except Exception:
            pass

    if document_id and embedding_adapter is not None:
        try:
            from ipa.agent.research_executor import _embed_new_chunks
            ctx = _EmbedCtx(corpus, embedding_adapter)
            _embed_new_chunks(corpus, ctx)
        except Exception:
            pass  # non-fatal; BM25 retrieval still works

    return document_id


# ---------------------------------------------------------------------------
# Auto-research dedup: one research run per normalized query per window.
# ---------------------------------------------------------------------------

_RECENT_PATH = _ROOT / "outputs" / "web_dashboard" / "research_recent.json"
DEDUP_MINUTES_DEFAULT = 10
DEDUP_SIMILARITY_DEFAULT = 0.6

# Stopwords español + términos genéricos de pedidos de investigación: se
# excluyen del matching para que "dame toda la informacion" no matchee
# cualquier query previa.
_QUERY_STOPWORDS = frozenset({
    "de", "la", "el", "en", "y", "a", "que", "los", "las", "un", "una",
    "por", "para", "con", "del", "sobre", "su", "sus", "al", "lo", "como",
    "mas", "o", "e", "u", "se", "es", "son", "me", "mi", "te", "tu", "nos",
    "les", "le", "fue", "hay",
    "investigar", "investigacion", "buscar", "busqueda", "tema", "info",
    "informacion", "dame", "toda", "todo", "quiero", "saber", "porque",
})


def _norm_query(text: str) -> str:
    import re as _re
    import unicodedata as _ud
    t = _ud.normalize("NFD", text.lower().strip())
    t = "".join(c for c in t if _ud.category(c) != "Mn")
    return _re.sub(r"\s+", " ", t)[:200]


def _query_tokens(norm: str) -> set[str]:
    """Content tokens de una query normalizada (sin stopwords ni 1-char)."""
    return {t for t in norm.split() if t not in _QUERY_STOPWORDS and len(t) > 1}


def recently_researched(
    query: str,
    *,
    minutes: int = DEDUP_MINUTES_DEFAULT,
    path: Path = _RECENT_PATH,
) -> bool:
    """True if a research run for this query started within the window."""
    return find_recent_research(query, minutes=minutes, path=path) is not None


def find_recent_research(
    query: str,
    *,
    minutes: int = DEDUP_MINUTES_DEFAULT,
    threshold: float = DEDUP_SIMILARITY_DEFAULT,
    path: Path = _RECENT_PATH,
) -> dict | None:
    """Most recent stored research that matches `query`, or None.

    Matching: exact normalized equality OR token containment — the model
    reformulates queries between turns ("IA big techs" → "IA tres grandes
    tecnologicas"), so exact match alone misses re-launches. Containment =
    |intersection| / min(|a|, |b|) over content tokens: a shorter query
    that is a subset of a stored one counts as the same research. Queries
    with no content tokens only match exactly.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        return None
    now = datetime.now(timezone.utc)
    qn = _norm_query(query)
    qt = _query_tokens(qn)
    best: dict | None = None
    for stored_q, ts in data.items():
        try:
            t0 = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            age = (now - t0).total_seconds()
        except Exception:
            continue
        if age >= minutes * 60:
            continue
        exact = stored_q == qn
        similar = False
        if not exact and qt:
            st = _query_tokens(stored_q)
            if st:
                similar = len(qt & st) / min(len(qt), len(st)) >= threshold
        if (exact or similar) and (best is None or ts > best["ts"]):
            best = {"query": stored_q, "ts": ts,
                    "age_minutes": max(0, int(age // 60)), "exact": exact}
    return best


def mark_researched(query: str, *, path: Path = _RECENT_PATH) -> None:
    """Record that a research run for this query just launched."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        data[_norm_query(query)] = _now()
        # Bound the map: keep the 50 most recent entries.
        if len(data) > 50:
            data = dict(sorted(data.items(), key=lambda kv: kv[1])[-50:])
        path.write_text(json.dumps(data, indent=1), encoding="utf-8")
    except Exception:
        pass


__all__ = [
    "ResearchReviewStore",
    "review_doc_with_llm",
    "review_docs_with_llm",
    "ingest_reviewed_doc",
    "recently_researched",
    "find_recent_research",
    "mark_researched",
]
