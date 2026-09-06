"""SQLiteVecIndex â€” vector index backed by sqlite-vec.

Competitor in E7.  Provides the same interface as BM25Index:
add_chunks, search, count, is_queryable, close.

sqlite-vec is a SQLite extension that adds vector search capabilities.
No server required â€” works within the same SQLite ecosystem as FTS5.
"""
from __future__ import annotations

import json
import sqlite3
import struct
from pathlib import Path

import sqlite_vec

from ipa.contracts import DocumentChunk, SearchHit, SourceSpan


def _span_to_dict(span: SourceSpan | None) -> dict:
    if span is None:
        return {}
    return {
        "artifact_id": span.artifact_id,
        "page": span.page,
        "offset_start": span.offset_start,
        "offset_end": span.offset_end,
    }


def _dict_to_span(d: dict) -> SourceSpan | None:
    if not d or not d.get("artifact_id"):
        return None
    return SourceSpan(
        artifact_id=d["artifact_id"], page=d["page"],
        offset_start=d["offset_start"], offset_end=d["offset_end"],
    )


def _vector_to_blob(vec: list[float]) -> bytes:
    """Pack a float32 vector into a binary blob for sqlite-vec."""
    return struct.pack(f"{len(vec)}f", *vec)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS vec_chunks_meta (
    chunk_id      TEXT PRIMARY KEY,
    document_id   TEXT NOT NULL,
    content_hash  TEXT NOT NULL,
    text          TEXT NOT NULL,
    span_json     TEXT,
    tombstoned    INTEGER NOT NULL DEFAULT 0,
    stored_at     TEXT NOT NULL
);
"""


class SQLiteVecIndex:
    """Vector index backed by sqlite-vec with cosine similarity search."""

    def __init__(
        self,
        db_path: str | Path,
        vector_dim: int = 1024,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.vector_dim = vector_dim
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = NORMAL")
        self._conn.executescript(_SCHEMA)
        # Create the virtual vec0 table.
        self._conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0("
            f"chunk_id TEXT PRIMARY KEY, "
            f"embedding float[{vector_dim}] distance_metric=cosine"
            f")"
        )
        self._conn.commit()

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        vectors: list[list[float]],
    ) -> None:
        """Batch-insert chunks with their pre-computed embedding vectors."""
        if not chunks:
            return
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks ({len(chunks)}) and vectors ({len(vectors)}) must have same length"
            )
        import time as _time
        now = _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime())
        for chunk, vec in zip(chunks, vectors):
            blob = _vector_to_blob(vec)
            span_json = json.dumps(_span_to_dict(chunk.source_span))
            self._conn.execute(
                "INSERT OR REPLACE INTO vec_chunks (chunk_id, embedding) VALUES (?, ?)",
                (chunk.chunk_id, blob),
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO vec_chunks_meta "
                "(chunk_id, document_id, content_hash, text, span_json, tombstoned, stored_at) "
                "VALUES (?,?,?,?,?, 0, ?)",
                (chunk.chunk_id, chunk.document_id, chunk.content_hash,
                 chunk.text, span_json, now),
            )
        self._conn.commit()

    def search(
        self,
        query_vector: list[float],
        limit: int = 10,
    ) -> list[SearchHit]:
        """Search by vector similarity.  Returns SearchHit records."""
        if limit <= 0:
            raise ValueError("limit must be positive")
        blob = _vector_to_blob(query_vector)
        rows = self._conn.execute(
            "SELECT v.chunk_id, v.distance, m.text, m.span_json "
            "FROM vec_chunks v "
            "JOIN vec_chunks_meta m ON v.chunk_id = m.chunk_id "
            "WHERE m.tombstoned = 0 "
            "AND v.embedding MATCH ? "
            "AND k = ? "
            "ORDER BY v.distance ASC",
            (blob, limit),
        ).fetchall()
        hits: list[SearchHit] = []
        for chunk_id, distance, text, span_json in rows:
            # sqlite-vec cosine distance: 0 = identical, 2 = opposite.
            # Negate so higher = better, consistent with other backends.
            hits.append(SearchHit(
                chunk_id=chunk_id,
                score=-distance,
                source_span=_dict_to_span(json.loads(span_json) if span_json else {}),
                retrieval_backend="sqlite_vec",
            ))
        return hits

    def count(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM vec_chunks_meta WHERE tombstoned = 0"
        ).fetchone()
        return row[0]

    def is_queryable(self) -> bool:
        return self.count() > 0

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "SQLiteVecIndex":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

