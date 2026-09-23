"""DocumentStore â€” durable source of truth for documents and chunks.

Stores CanonicalDocuments and DocumentChunks in SQLite.  Idempotent: re-ingesting
the same document (by document_id) does not duplicate.  Deletion uses
tombstones rather than history loss.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Iterator

from ipa.contracts import CanonicalDocument, DocumentChunk, SourceSpan

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    document_id   TEXT PRIMARY KEY,
    artifact_id   TEXT NOT NULL,
    parser_id     TEXT NOT NULL,
    mime_type     TEXT NOT NULL,
    pages         INTEGER NOT NULL,
    text          TEXT NOT NULL,
    elements_json TEXT NOT NULL,
    spans_json    TEXT NOT NULL,
    stored_at     TEXT NOT NULL,
    tombstoned    INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id      TEXT PRIMARY KEY,
    document_id   TEXT NOT NULL,
    content_hash  TEXT NOT NULL,
    text          TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    span_json     TEXT,
    stored_at     TEXT NOT NULL,
    tombstoned    INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (document_id) REFERENCES documents(document_id)
);

CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);

CREATE TABLE IF NOT EXISTS embedding_jobs (
    chunk_id TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    claimed_at REAL,
    completed_at REAL,
    embedding_fingerprint TEXT,
    error TEXT,
    FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id)
);
CREATE INDEX IF NOT EXISTS idx_embedding_jobs_status ON embedding_jobs(status, chunk_id);

CREATE TABLE IF NOT EXISTS document_centroids (
    document_id      TEXT PRIMARY KEY,
    representative_chunk_ids TEXT NOT NULL,
    chunk_count      INTEGER NOT NULL DEFAULT 0,
    computed_at      TEXT NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents(document_id)
);

CREATE TABLE IF NOT EXISTS document_sources (
    document_id   TEXT PRIMARY KEY,
    source_url    TEXT,
    source_domain TEXT,
    provenance    TEXT NOT NULL,
    quality_score REAL,
    published_at  TEXT,
    recorded_at   TEXT NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents(document_id)
);
CREATE INDEX IF NOT EXISTS idx_document_sources_provenance ON document_sources(provenance);

-- Derived per-document signals computed once at ingest (Tier 0) instead of
-- being re-derived by every idle topify cycle: normalized content hash
-- (reporter_curation.normalized_hash format), extracted title, publish date
-- and free-form extras (novelty hints, duplicate flags).
CREATE TABLE IF NOT EXISTS document_metadata (
    document_id     TEXT PRIMARY KEY,
    normalized_hash TEXT,
    title           TEXT,
    published_at    TEXT,
    char_count      INTEGER,
    extra_json      TEXT NOT NULL DEFAULT '{}',
    computed_at     TEXT NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents(document_id)
);
"""


def _span_to_dict(span: SourceSpan | None) -> dict | None:
    if span is None:
        return None
    return {
        "artifact_id": span.artifact_id,
        "page": span.page,
        "offset_start": span.offset_start,
        "offset_end": span.offset_end,
    }


def _dict_to_span(d: dict | None) -> SourceSpan | None:
    if d is None:
        return None
    return SourceSpan(
        artifact_id=d["artifact_id"],
        page=d["page"],
        offset_start=d["offset_start"],
        offset_end=d["offset_end"],
    )


class DocumentStore:
    """Durable store for canonical documents and their chunks."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = NORMAL")
        self._conn.execute("PRAGMA busy_timeout = 30000")
        self._conn.execute("PRAGMA temp_store = MEMORY")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(documents)")}
        if "spans_json" not in columns:
            self._conn.execute(
                "ALTER TABLE documents ADD COLUMN spans_json TEXT NOT NULL DEFAULT '[]'"
            )
        source_columns = {row[1] for row in self._conn.execute("PRAGMA table_info(document_sources)")}
        if "published_at" not in source_columns:
            self._conn.execute(
                "ALTER TABLE document_sources ADD COLUMN published_at TEXT"
            )
        self._conn.execute(
            "INSERT OR IGNORE INTO embedding_jobs (chunk_id) "
            "SELECT chunk_id FROM chunks WHERE tombstoned = 0"
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "DocumentStore":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def put_document(self, doc: CanonicalDocument, artifact_id: str) -> None:
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        spans_json = json.dumps([_span_to_dict(s) for s in doc.source_spans])
        self._conn.execute(
            "INSERT OR REPLACE INTO documents "
            "(document_id, artifact_id, parser_id, mime_type, pages, text, "
            "elements_json, spans_json, stored_at, tombstoned) "
            "VALUES (?,?,?,?,?,?,?,?, ?, 0)",
            (doc.document_id, artifact_id, doc.parser_id, doc.mime_type,
             doc.pages, doc.text, json.dumps(doc.elements), spans_json, now),
        )

    def put_chunks(self, chunks: list[DocumentChunk]) -> None:
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._conn.executemany(
            "INSERT OR REPLACE INTO chunks "
            "(chunk_id, document_id, content_hash, text, metadata_json, "
            "span_json, stored_at, tombstoned) "
            "VALUES (?,?,?,?,?,?, ?, 0)",
            [
                (chunk.chunk_id, chunk.document_id, chunk.content_hash,
                 chunk.text, json.dumps(chunk.metadata),
                 json.dumps(_span_to_dict(chunk.source_span)), now)
                for chunk in chunks
            ],
        )
        self._conn.executemany(
            "INSERT INTO embedding_jobs (chunk_id, status) VALUES (?, 'pending') "
            "ON CONFLICT(chunk_id) DO UPDATE SET status = CASE "
            "WHEN embedding_jobs.status = 'complete' THEN embedding_jobs.status "
            "ELSE 'pending' END, error = NULL",
            [(chunk.chunk_id,) for chunk in chunks],
        )

    def commit(self) -> None:
        """Flush pending writes to disk."""
        self._conn.commit()

    def get_document(self, document_id: str) -> CanonicalDocument | None:
        row = self._conn.execute(
            "SELECT document_id, pages, elements_json, spans_json, text, mime_type, parser_id, tombstoned "
            "FROM documents WHERE document_id = ?",
            (document_id,),
        ).fetchone()
        if not row or row[7]:
            return None
        return CanonicalDocument(
            document_id=row[0], pages=row[1],
            elements=json.loads(row[2]),
            source_spans=[_dict_to_span(span) for span in json.loads(row[3])],
            text=row[4], mime_type=row[5], parser_id=row[6],
        )

    def get_chunks(self, document_id: str) -> Iterator[DocumentChunk]:
        rows = self._conn.execute(
            "SELECT chunk_id, document_id, content_hash, text, metadata_json, span_json "
            "FROM chunks WHERE document_id = ? AND tombstoned = 0 "
            "ORDER BY json_extract(metadata_json, '$.chunk_index'), chunk_id",
            (document_id,),
        ).fetchall()
        for row in rows:
            yield DocumentChunk(
                chunk_id=row[0], document_id=row[1], content_hash=row[2],
                text=row[3], metadata=json.loads(row[4]),
                source_span=_dict_to_span(json.loads(row[5]) if row[5] else None),
            )

    def get_chunk(self, chunk_id: str) -> DocumentChunk | None:
        row = self._conn.execute(
            "SELECT chunk_id, document_id, content_hash, text, metadata_json, span_json "
            "FROM chunks WHERE chunk_id = ? AND tombstoned = 0",
            (chunk_id,),
        ).fetchone()
        if not row:
            return None
        return DocumentChunk(
            chunk_id=row[0], document_id=row[1], content_hash=row[2],
            text=row[3], metadata=json.loads(row[4]),
            source_span=_dict_to_span(json.loads(row[5]) if row[5] else None),
        )

    def tombstone_document(self, document_id: str) -> None:
        self._conn.execute(
            "UPDATE documents SET tombstoned = 1 WHERE document_id = ?",
            (document_id,),
        )
        self._conn.execute(
            "UPDATE chunks SET tombstoned = 1 WHERE document_id = ?",
            (document_id,),
        )
        self._conn.commit()

    def count_documents(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM documents WHERE tombstoned = 0"
        ).fetchone()
        return row[0]

    def count_chunks(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE tombstoned = 0"
        ).fetchone()
        return row[0]

    def all_chunks(self) -> Iterator[DocumentChunk]:
        rows = self._conn.execute(
            "SELECT chunk_id, document_id, content_hash, text, metadata_json, span_json "
            "FROM chunks WHERE tombstoned = 0 "
            "ORDER BY json_extract(metadata_json, '$.chunk_index'), chunk_id"
        ).fetchall()
        for row in rows:
            yield DocumentChunk(
                chunk_id=row[0], document_id=row[1], content_hash=row[2],
                text=row[3], metadata=json.loads(row[4]),
                source_span=_dict_to_span(json.loads(row[5]) if row[5] else None),
            )

    def put_centroid(self, document_id: str, chunk_ids: list[str], chunk_count: int) -> None:
        """Store the representative chunk IDs for a document (computed from embedding centroid)."""
        import json as _json
        from datetime import datetime as _dt, timezone as _tz
        self._conn.execute(
            "INSERT OR REPLACE INTO document_centroids (document_id, representative_chunk_ids, chunk_count, computed_at) "
            "VALUES (?, ?, ?, ?)",
            (document_id, _json.dumps(chunk_ids), chunk_count, _dt.now(_tz.utc).isoformat()),
        )

    def get_centroid(self, document_id: str) -> list[str] | None:
        """Return the representative chunk IDs for a document, or None if not computed."""
        row = self._conn.execute(
            "SELECT representative_chunk_ids FROM document_centroids WHERE document_id = ?",
            (document_id,),
        ).fetchone()
        if not row:
            return None
        import json as _json
        return _json.loads(row[0])

    def all_centroids(self) -> dict[str, list[str]]:
        """Return all centroids as {document_id: [chunk_id, ...]}."""
        import json as _json
        rows = self._conn.execute(
            "SELECT document_id, representative_chunk_ids FROM document_centroids"
        ).fetchall()
        return {row[0]: _json.loads(row[1]) for row in rows}

    # --- Document provenance ---

    def put_source(self, document_id: str, source_url: str, source_domain: str,
                   provenance: str, quality_score: float = 0.0,
                   published_at: str | None = None) -> None:
        """Record the provenance of a document (where it came from).

        provenance: "configured_scrape" (known source) or "agent_research" (agent search).
        """
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._conn.execute(
            "INSERT OR REPLACE INTO document_sources "
            "(document_id, source_url, source_domain, provenance, quality_score, "
            "published_at, recorded_at) VALUES (?,?,?,?,?,?,?)",
            (document_id, source_url, source_domain, provenance, quality_score,
             published_at, now),
        )

    def get_source(self, document_id: str) -> dict | None:
        """Return provenance info for a document, or None if not recorded."""
        row = self._conn.execute(
            "SELECT source_url, source_domain, provenance, quality_score, published_at "
            "FROM document_sources WHERE document_id = ?",
            (document_id,),
        ).fetchone()
        if not row:
            return None
        return {"source_url": row[0], "source_domain": row[1], "provenance": row[2],
                "quality_score": row[3], "published_at": row[4]}

    def document_stored_at(self, document_id: str) -> str | None:
        """Return stored_at for a live document, or None."""
        row = self._conn.execute(
            "SELECT stored_at FROM documents WHERE document_id = ? AND tombstoned = 0",
            (document_id,),
        ).fetchone()
        return row[0] if row else None

    def all_document_stored_at(self) -> dict[str, str]:
        """Return {document_id: stored_at} for live documents — used to sync
        the derived LanceDB metadata columns (published_at proxy)."""
        rows = self._conn.execute(
            "SELECT document_id, stored_at FROM documents WHERE tombstoned = 0"
        ).fetchall()
        return {row[0]: row[1] for row in rows}

    def all_sources(self) -> dict[str, dict]:
        """Return all document provenance records as {document_id: {source_url, source_domain, provenance, quality_score}}."""
        rows = self._conn.execute(
            "SELECT document_id, source_url, source_domain, provenance, quality_score, "
            "published_at FROM document_sources"
        ).fetchall()
        return {row[0]: {"source_url": row[1], "source_domain": row[2], "provenance": row[3],
                         "quality_score": row[4], "published_at": row[5]} for row in rows}

    def all_document_texts(self) -> dict[str, str]:
        """Return {document_id: text} for every live (non-tombstoned) document."""
        rows = self._conn.execute(
            "SELECT document_id, text FROM documents WHERE tombstoned = 0"
        ).fetchall()
        return {row[0]: (row[1] or "") for row in rows}

    def sources_by_provenance(self, provenance: str) -> dict[str, dict]:
        """Return all documents with a given provenance type."""
        rows = self._conn.execute(
            "SELECT document_id, source_url, source_domain, quality_score FROM document_sources WHERE provenance = ?",
            (provenance,),
        ).fetchall()
        return {row[0]: {"source_url": row[1], "source_domain": row[2], "quality_score": row[3]} for row in rows}

    # --- Document metadata (derived signals computed at ingest) ---

    def put_doc_meta(self, document_id: str, *, normalized_hash: str | None = None,
                     title: str | None = None, published_at: str | None = None,
                     char_count: int | None = None, extra: dict | None = None) -> None:
        """Store derived per-document signals (computed once, e.g. at ingest).

        ``extra`` is merged into ``extra_json`` — callers add keys like
        ``novelty_hint`` or ``duplicate_of_main`` without clobbering others.
        """
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        row = self._conn.execute(
            "SELECT extra_json FROM document_metadata WHERE document_id = ?",
            (document_id,),
        ).fetchone()
        merged_extra: dict = {}
        if row:
            try:
                merged_extra = json.loads(row[0] or "{}") or {}
            except (TypeError, ValueError):
                merged_extra = {}
        if extra:
            merged_extra.update(extra)
        self._conn.execute(
            "INSERT INTO document_metadata "
            "(document_id, normalized_hash, title, published_at, char_count, "
            "extra_json, computed_at) VALUES (?,?,?,?,?,?,?) "
            "ON CONFLICT(document_id) DO UPDATE SET "
            "normalized_hash = COALESCE(excluded.normalized_hash, normalized_hash), "
            "title = COALESCE(excluded.title, title), "
            "published_at = COALESCE(excluded.published_at, published_at), "
            "char_count = COALESCE(excluded.char_count, char_count), "
            "extra_json = excluded.extra_json, computed_at = excluded.computed_at",
            (document_id, normalized_hash, title, published_at, char_count,
             json.dumps(merged_extra, ensure_ascii=False), now),
        )

    def get_doc_meta(self, document_id: str) -> dict | None:
        """Return derived metadata for a document, or None."""
        row = self._conn.execute(
            "SELECT normalized_hash, title, published_at, char_count, extra_json "
            "FROM document_metadata WHERE document_id = ?",
            (document_id,),
        ).fetchone()
        if not row:
            return None
        try:
            extra = json.loads(row[4] or "{}")
        except (TypeError, ValueError):
            extra = {}
        return {"normalized_hash": row[0], "title": row[1], "published_at": row[2],
                "char_count": row[3], "extra": extra}

    def all_doc_meta(self) -> dict[str, dict]:
        """Return {document_id: metadata dict} for every document with a row."""
        rows = self._conn.execute(
            "SELECT document_id, normalized_hash, title, published_at, char_count, "
            "extra_json FROM document_metadata"
        ).fetchall()
        out: dict[str, dict] = {}
        for row in rows:
            try:
                extra = json.loads(row[5] or "{}")
            except (TypeError, ValueError):
                extra = {}
            out[row[0]] = {"normalized_hash": row[1], "title": row[2],
                           "published_at": row[3], "char_count": row[4],
                           "extra": extra}
        return out

    def url_normalized_hashes(self) -> dict[str, str]:
        """{source_url: normalized_hash} — cheap replacement for re-hashing
        every document text on each curation cycle."""
        rows = self._conn.execute(
            "SELECT s.source_url, m.normalized_hash FROM document_sources s "
            "JOIN document_metadata m ON m.document_id = s.document_id "
            "WHERE s.source_url IS NOT NULL AND s.source_url != '' "
            "AND m.normalized_hash IS NOT NULL"
        ).fetchall()
        return {row[0]: row[1] for row in rows}

