"""BM25Index â€” incremental lexical index backed by SQLite FTS5.

Provides immediate lexical availability after the fast path.  Supports
incremental append, search with BM25 ranking, and tombstone-based deletion.
This is the ``first_queryable`` index: it must be available before embeddings
or enrichment.
"""
from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path

from ipa.contracts import DocumentChunk, SearchHit, SourceSpan

_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    document_id UNINDEXED,
    content_hash UNINDEXED,
    text,
    span_json UNINDEXED,
    tokenize = 'porter unicode61'
);

CREATE TABLE IF NOT EXISTS chunks_meta (
    chunk_id      TEXT PRIMARY KEY,
    document_id   TEXT NOT NULL,
    content_hash  TEXT NOT NULL,
    span_json     TEXT,
    tombstoned    INTEGER NOT NULL DEFAULT 0,
    indexed_at    TEXT NOT NULL
);
"""


def _span_to_json(span: SourceSpan | None) -> str | None:
    if span is None:
        return None
    import json
    return json.dumps({
        "artifact_id": span.artifact_id,
        "page": span.page,
        "offset_start": span.offset_start,
        "offset_end": span.offset_end,
    })


def _json_to_span(s: str | None) -> SourceSpan | None:
    if not s:
        return None
    import json
    d = json.loads(s)
    return SourceSpan(
        artifact_id=d["artifact_id"], page=d["page"],
        offset_start=d["offset_start"], offset_end=d["offset_end"],
    )


class BM25Index:
    """Incremental FTS5 lexical index with BM25 ranking."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = NORMAL")
        self._conn.execute("PRAGMA temp_store = MEMORY")
        self._conn.executescript(_SCHEMA)
        # Disable FTS5 auto-merge to avoid O(n^2) degradation during bulk insert.
        # We merge manually after all inserts are done.
        try:
            self._conn.execute("INSERT INTO chunks_fts(chunks_fts, rank) VALUES('automerge', 0)")
        except sqlite3.OperationalError:
            pass  # table may not exist yet on first run
        self._conn.commit()

    def commit(self) -> None:
        """Flush pending writes to disk."""
        self._conn.commit()

    def optimize(self) -> None:
        """Merge all FTS5 segments into one for optimal query performance.

        Call this after bulk indexing is complete.  This is expensive but
        only needs to run once after all inserts are done.
        """
        self._conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('merge')")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "BM25Index":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def add_chunk(self, chunk: DocumentChunk) -> None:
        """Insert or replace a chunk in the FTS index.  Idempotent by chunk_id."""
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        span_json = _span_to_json(chunk.source_span)
        # Remove existing entry if present (for re-indexing).
        self._conn.execute(
            "DELETE FROM chunks_fts WHERE chunk_id = ?", (chunk.chunk_id,)
        )
        self._conn.execute(
            "INSERT INTO chunks_fts (chunk_id, document_id, content_hash, text, span_json) "
            "VALUES (?,?,?,?,?)",
            (chunk.chunk_id, chunk.document_id, chunk.content_hash,
             chunk.text, span_json),
        )
        self._conn.execute(
            "INSERT OR REPLACE INTO chunks_meta "
            "(chunk_id, document_id, content_hash, span_json, tombstoned, indexed_at) "
            "VALUES (?,?,?, ?, 0, ?)",
            (chunk.chunk_id, chunk.document_id, chunk.content_hash,
             span_json, now),
        )
        self._conn.commit()

    def add_chunks(self, chunks: list[DocumentChunk], commit: bool = True) -> None:
        """Batch-insert chunks.  Commits by default; set commit=False to defer."""
        if not chunks:
            return
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        # Delete existing entries (for re-indexing idempotency).
        self._conn.executemany(
            "DELETE FROM chunks_fts WHERE chunk_id = ?",
            [(chunk.chunk_id,) for chunk in chunks],
        )
        # Insert into FTS.
        self._conn.executemany(
            "INSERT INTO chunks_fts (chunk_id, document_id, content_hash, text, span_json) "
            "VALUES (?,?,?,?,?)",
            [
                (chunk.chunk_id, chunk.document_id, chunk.content_hash,
                 chunk.text, _span_to_json(chunk.source_span))
                for chunk in chunks
            ],
        )
        # Insert into meta.
        self._conn.executemany(
            "INSERT OR REPLACE INTO chunks_meta "
            "(chunk_id, document_id, content_hash, span_json, tombstoned, indexed_at) "
            "VALUES (?,?,?, ?, 0, ?)",
            [
                (chunk.chunk_id, chunk.document_id, chunk.content_hash,
                 _span_to_json(chunk.source_span), now)
                for chunk in chunks
            ],
        )
        if commit:
            self._conn.commit()

    def remove_chunk(self, chunk_id: str) -> None:
        """Tombstone a chunk: remove from FTS, mark as tombstoned in meta."""
        self._conn.execute(
            "DELETE FROM chunks_fts WHERE chunk_id = ?", (chunk_id,)
        )
        self._conn.execute(
            "UPDATE chunks_meta SET tombstoned = 1 WHERE chunk_id = ?",
            (chunk_id,),
        )
        self._conn.commit()

    def search(self, query: str, limit: int = 10) -> list[SearchHit]:
        """Search using FTS5 BM25 ranking.  Returns SearchHit records."""
        if limit <= 0:
            raise ValueError("limit must be positive")
        terms = re.findall(r"[\w]+", query, flags=re.UNICODE)
        if not terms:
            return []
        # Quote individual terms so user punctuation cannot become FTS5 syntax.
        safe_query = " ".join(f'"{term}"' for term in terms)
        # FTS5 MATCH with BM25 ranking (lower score = better match in FTS5,
        # so we negate to get higher = better).
        rows = self._conn.execute(
            "SELECT chunk_id, content_hash, span_json, bm25(chunks_fts) AS score "
            "FROM chunks_fts WHERE chunks_fts MATCH ? "
            "ORDER BY score ASC LIMIT ?",
            (safe_query, limit),
        ).fetchall()
        hits: list[SearchHit] = []
        for chunk_id, content_hash, span_json, score in rows:
            hits.append(SearchHit(
                chunk_id=chunk_id,
                score=-score,  # FTS5 returns negative BM25; negate for higher=better
                source_span=_json_to_span(span_json),
                retrieval_backend="sqlite_fts5",
            ))
        return hits

    def count(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM chunks_meta WHERE tombstoned = 0"
        ).fetchone()
        return row[0]

    def is_queryable(self) -> bool:
        """True if at least one non-tombstoned chunk is indexed."""
        return self.count() > 0

