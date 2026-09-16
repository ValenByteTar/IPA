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
    recorded_at   TEXT NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents(document_id)
);
CREATE INDEX IF NOT EXISTS idx_document_sources_provenance ON document_sources(provenance);
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
                   provenance: str, quality_score: float = 0.0) -> None:
        """Record the provenance of a document (where it came from).

        provenance: "configured_scrape" (known source) or "agent_research" (agent search).
        """
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._conn.execute(
            "INSERT OR REPLACE INTO document_sources "
            "(document_id, source_url, source_domain, provenance, quality_score, recorded_at) "
            "VALUES (?,?,?,?,?,?)",
            (document_id, source_url, source_domain, provenance, quality_score, now),
        )

    def get_source(self, document_id: str) -> dict | None:
        """Return provenance info for a document, or None if not recorded."""
        row = self._conn.execute(
            "SELECT source_url, source_domain, provenance, quality_score FROM document_sources WHERE document_id = ?",
            (document_id,),
        ).fetchone()
        if not row:
            return None
        return {"source_url": row[0], "source_domain": row[1], "provenance": row[2], "quality_score": row[3]}

    def all_sources(self) -> dict[str, dict]:
        """Return all document provenance records as {document_id: {source_url, source_domain, provenance, quality_score}}."""
        rows = self._conn.execute(
            "SELECT document_id, source_url, source_domain, provenance, quality_score FROM document_sources"
        ).fetchall()
        return {row[0]: {"source_url": row[1], "source_domain": row[2], "provenance": row[3], "quality_score": row[4]} for row in rows}

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

