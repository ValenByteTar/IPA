"""LanceDBIndex â€” vector index backed by LanceDB.

Competitor in E7.  Provides the same interface as BM25Index:
add_chunks, search, count, is_queryable, close.

LanceDB is a local vector database built on Apache Arrow.  No server required.
Supports cosine similarity, L2 distance, and hybrid search (dense + FTS + sparse).

Hybrid search combines:
  - Dense vector search (semantic similarity via BGE-M3)
  - Full-text search (keyword matching via BM25 on raw text)
  - Sparse vector search (learned token weights via BGE-M3 sparse head)
  - 3-way RRF fusion (Reciprocal Rank Fusion) to merge all result sets

BGE-M3 sparse embeddings are produced in the same forward pass as dense (free).
They are stored as JSON in a `sparse_json` column. LanceDB does not natively
index sparse vectors, so sparse search is a manual dot product over a
dense-pruned candidate set (top-N dense results re-ranked with sparse).

This gives the best of both worlds: semantic understanding catches
paraphrased concepts, BM25 catches exact keyword matches, and learned sparse
weights capture term importance better than heuristic BM25.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa

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


class LanceDBIndex:
    """Vector index backed by LanceDB with cosine similarity and hybrid search."""

    TABLE_NAME = "chunks"

    def __init__(
        self,
        db_path: str | Path,
        vector_dim: int = 1024,
        metric: str = "cosine",
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.vector_dim = vector_dim
        self.metric = metric
        self._db = lancedb.connect(str(self.db_path))
        self._table = None
        self._fts_indexed = False
        # Open existing table if present, so search/count work without add_chunks.
        self._open_existing_table()

    def _open_existing_table(self) -> None:
        """Open an existing table if it exists in the database."""
        try:
            # list_tables() returns a ListTablesResponse object in newer
            # LanceDB versions; table_names() returns a plain list (deprecated).
            if hasattr(self._db, "table_names"):
                existing = self._db.table_names()
            else:
                resp = self._db.list_tables()
                existing = resp.tables if hasattr(resp, "tables") else list(resp)
            if self.TABLE_NAME in existing:
                self._table = self._db.open_table(self.TABLE_NAME)
        except Exception:
            pass  # Table doesn't exist yet; will be created on add_chunks.

    def _ensure_table(self, sample_vector: list[float] | None = None) -> None:
        """Create the table if it doesn't exist."""
        if self._table is not None:
            return
        if hasattr(self._db, "table_names"):
            existing_tables = self._db.table_names()
        else:
            resp = self._db.list_tables()
            existing_tables = resp.tables if hasattr(resp, "tables") else list(resp)
        if self.TABLE_NAME in existing_tables:
            self._table = self._db.open_table(self.TABLE_NAME)
        else:
            if sample_vector is None:
                sample_vector = [0.0] * self.vector_dim
            schema = pa.schema([
                pa.field("chunk_id", pa.string()),
                pa.field("document_id", pa.string()),
                pa.field("content_hash", pa.string()),
                pa.field("text", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), len(sample_vector))),
                pa.field("span_json", pa.string()),
                # BGE-M3 sparse weights: JSON string of {token_id: weight}.
                # LanceDB has no native sparse index, so these are used for
                # manual dot-product re-ranking in search_sparse().
                pa.field("sparse_json", pa.string()),
            ])
            self._table = self._db.create_table(
                self.TABLE_NAME,
                schema=schema,
                mode="overwrite",
            )

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        vectors: list[list[float]],
        sparse_weights: list[dict] | None = None,
    ) -> None:
        """Batch-insert chunks with their pre-computed embedding vectors.

        Args:
            chunks: DocumentChunk records to insert.
            vectors: Dense embedding vectors (1024-dim float lists).
            sparse_weights: Optional BGE-M3 sparse weights (list of dicts
                {token_id: weight}). When provided, stored in sparse_json
                column for use in search_sparse() and 3-way hybrid search.

        Idempotent: deletes existing records with matching chunk_id before
        inserting to prevent duplicates on reprocessing."""
        if not chunks:
            return
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks ({len(chunks)}) and vectors ({len(vectors)}) must have same length"
            )
        if sparse_weights is not None and len(sparse_weights) != len(chunks):
            raise ValueError(
                f"sparse_weights ({len(sparse_weights)}) must match chunks ({len(chunks)})"
            )
        self._ensure_table(vectors[0])
        # Delete existing records with same chunk_id (idempotency on reprocess)
        chunk_ids = [c.chunk_id for c in chunks]
        if chunk_ids:
            id_list = ", ".join(f"'{cid}'" for cid in chunk_ids)
            try:
                self._table.delete(f"chunk_id IN ({id_list})")
            except Exception:
                pass  # table may be empty or column may not exist yet
        records = []
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            sparse_json = ""
            if sparse_weights is not None and sparse_weights[i]:
                # Convert keys to str for JSON serialization (BGE-M3 uses str keys)
                sparse = sparse_weights[i]
                sparse_json = json.dumps(
                    {str(k): float(v) for k, v in sparse.items()}
                )
            records.append({
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "content_hash": chunk.content_hash,
                "text": chunk.text,
                "vector": vec,
                "span_json": json.dumps(_span_to_dict(chunk.source_span)),
                "sparse_json": sparse_json,
            })
        self._table.add(records)
        # FTS index needs to be rebuilt after data changes
        self._fts_indexed = False

    def create_fts_index(self) -> None:
        """Create a full-text search index on the text column.

        Required for hybrid search.  Must be called after chunks are inserted.
        Uses BM25 for keyword-based retrieval.
        """
        if self._table is None or self._fts_indexed:
            return
        try:
            self._table.create_fts_index("text")
            self._fts_indexed = True
        except Exception as e:
            if "already exists" in str(e).lower():
                # Index already exists from a previous run — that is the
                # desired state; mark as indexed instead of warning on every
                # subsequent call.
                self._fts_indexed = True
            else:
                # tantivy not available or other failure — hybrid search
                # will fall back to dense-only
                print(f"  Warning: FTS index creation failed: {e}")

    def get_vectors(self, chunk_ids: list[str]) -> dict[str, list[float]]:
        """Fetch dense vectors for the given chunk ids: {chunk_id: vector}.

        Used by the topic backfill (document centroids) — no search, direct
        row access by id.
        """
        if not chunk_ids or self._table is None:
            return {}
        id_list = ", ".join(f"'{cid}'" for cid in chunk_ids)
        try:
            rows = self._table.search().where(f"chunk_id IN ({id_list})").limit(len(chunk_ids)).to_list()
        except Exception:
            return {}
        return {r["chunk_id"]: r["vector"] for r in rows if r.get("vector")}

    def search(
        self,
        query_vector: list[float],
        limit: int = 10,
    ) -> list[SearchHit]:
        """Search by vector similarity (dense only).  Returns SearchHit records."""
        if limit <= 0:
            raise ValueError("limit must be positive")
        if self._table is None:
            return []
        results = self._table.search(query_vector).limit(limit).to_list()
        hits: list[SearchHit] = []
        for r in results:
            # LanceDB returns _distance (lower = better for L2, higher = better for cosine).
            # For cosine, LanceDB returns 1 - cosine_similarity as distance.
            # We negate to get higher = better, consistent with FTS5 BM25.
            distance = r.get("_distance", 0.0)
            score = -distance if self.metric == "cosine" else -distance
            hits.append(SearchHit(
                chunk_id=r["chunk_id"],
                score=score,
                source_span=_dict_to_span(json.loads(r["span_json"]) if r["span_json"] else {}),
                retrieval_backend="lancedb",
            ))
        return hits

    def search_fts(
        self,
        query_text: str,
        limit: int = 10,
    ) -> list[SearchHit]:
        """Search by full-text search (BM25 keyword matching).

        Requires create_fts_index() to have been called.
        """
        if limit <= 0:
            raise ValueError("limit must be positive")
        if self._table is None:
            return []
        if not self._fts_indexed:
            self.create_fts_index()
        try:
            results = self._table.search(query_text, query_type="fts").limit(limit).to_list()
        except Exception:
            return []
        hits: list[SearchHit] = []
        for r in results:
            score = r.get("_score", 0.0)
            hits.append(SearchHit(
                chunk_id=r["chunk_id"],
                score=score,
                source_span=_dict_to_span(json.loads(r["span_json"]) if r["span_json"] else {}),
                retrieval_backend="lancedb_fts",
            ))
        return hits

    def search_sparse(
        self,
        query_sparse: dict,
        limit: int = 10,
        candidate_ids: list[str] | None = None,
    ) -> list[SearchHit]:
        """Search by sparse vector dot product (BGE-M3 learned lexical weights).

        LanceDB has no native sparse index, so this is a manual dot product
        over candidate chunks. For efficiency, pass candidate_ids from a
        dense search to prune the search space. If no candidates are given,
        all rows are scanned (slow for large corpora).

        Args:
            query_sparse: dict of {token_id: weight} from embed_query_hybrid.
            limit: max results to return.
            candidate_ids: optional list of chunk_ids to score (from dense
                search). If None, scans all rows.
        """
        if limit <= 0 or self._table is None:
            return []

        # Load candidate rows (to_list() returns list[dict] in current LanceDB)
        if candidate_ids:
            id_set = set(candidate_ids)
            try:
                rows = self._table.search().where(f"chunk_id IN ({id_list})").to_list()
            except Exception:
                rows = [
                    row for row in self._table.to_arrow().to_pylist()
                    if row.get("chunk_id") in id_set
                ]
        else:
            rows = self._table.to_arrow().to_pylist()

        if not rows:
            return []

        # Convert query sparse keys to str (BGE-M3 uses str keys)
        query_weights = {str(k): float(v) for k, v in query_sparse.items()}

        # Score each candidate by dot product
        scored: list[tuple[str, float, dict]] = []
        for row in rows:
            sparse_json = row.get("sparse_json", "")
            if not sparse_json:
                continue
            try:
                doc_weights = json.loads(sparse_json)
            except (json.JSONDecodeError, TypeError):
                continue
            # Dot product: sum of query_weight * doc_weight for shared tokens
            score = 0.0
            for token, q_weight in query_weights.items():
                d_weight = doc_weights.get(token, 0.0)
                if d_weight > 0:
                    score += q_weight * d_weight
            if score > 0:
                scored.append((row["chunk_id"], score, row))

        scored.sort(key=lambda x: x[1], reverse=True)

        hits: list[SearchHit] = []
        for cid, score, row in scored[:limit]:
            span_json = row.get("span_json", "")
            hits.append(SearchHit(
                chunk_id=cid,
                score=score,
                source_span=_dict_to_span(json.loads(span_json) if span_json else {}),
                retrieval_backend="lancedb_sparse",
            ))
        return hits

    def search_hybrid(
        self,
        query_text: str,
        query_vector: list[float],
        limit: int = 10,
        query_sparse: dict | None = None,
    ) -> list[SearchHit]:
        """Hybrid search combining dense + FTS + sparse with 3-way RRF fusion.

        This is the recommended retrieval method for production RAG:
          1. Dense vector search catches semantic similarity (paraphrased concepts)
          2. FTS/BM25 catches exact keyword matches (CVE IDs, model names, terms)
          3. BGE-M3 sparse catches learned term importance (better than BM25)
          4. 3-way RRF merges all result sets by rank

        When query_sparse is None, falls back to 2-way RRF (dense + FTS only).

        RRF formula: score = sum(1 / (k + rank_i)) for each result list i
        Default k=60 (standard from the original RRF paper).

        Returns SearchHit records with fused scores.
        """
        if limit <= 0:
            raise ValueError("limit must be positive")
        if self._table is None:
            return []
        if not self._fts_indexed:
            self.create_fts_index()

        # Get dense results (more than limit for better fusion)
        dense_hits = self.search(query_vector, limit=limit * 3)
        # Get FTS results
        fts_hits = self.search_fts(query_text, limit=limit * 3)
        # Get sparse results (if query_sparse provided)
        sparse_hits: list[SearchHit] = []
        if query_sparse:
            # Use dense candidates to prune sparse search
            dense_ids = [h.chunk_id for h in dense_hits]
            sparse_hits = self.search_sparse(
                query_sparse, limit=limit * 3, candidate_ids=dense_ids
            )

        # Build rank maps
        dense_ranks = {h.chunk_id: i + 1 for i, h in enumerate(dense_hits)}
        fts_ranks = {h.chunk_id: i + 1 for i, h in enumerate(fts_hits)}
        sparse_ranks = {h.chunk_id: i + 1 for i, h in enumerate(sparse_hits)}

        # Collect all chunk_ids
        all_ids = set(dense_ranks.keys()) | set(fts_ranks.keys()) | set(sparse_ranks.keys())

        # Build hit lookup (prefer dense hits for span info)
        hit_map: dict[str, SearchHit] = {}
        for h in dense_hits:
            hit_map[h.chunk_id] = h
        for h in fts_hits:
            if h.chunk_id not in hit_map:
                hit_map[h.chunk_id] = h
        for h in sparse_hits:
            if h.chunk_id not in hit_map:
                hit_map[h.chunk_id] = h

        # Compute 3-way RRF scores
        k = 60
        scored: list[tuple[str, float]] = []
        for cid in all_ids:
            rrf_score = 0.0
            if cid in dense_ranks:
                rrf_score += 1.0 / (k + dense_ranks[cid])
            if cid in fts_ranks:
                rrf_score += 1.0 / (k + fts_ranks[cid])
            if cid in sparse_ranks:
                rrf_score += 1.0 / (k + sparse_ranks[cid])
            scored.append((cid, rrf_score))

        # Sort by RRF score (descending)
        scored.sort(key=lambda x: x[1], reverse=True)

        # Build result hits
        hits: list[SearchHit] = []
        backend = "lancedb_hybrid_3way" if query_sparse else "lancedb_hybrid_rrf"
        for cid, score in scored[:limit]:
            original = hit_map[cid]
            hits.append(SearchHit(
                chunk_id=cid,
                score=score,
                source_span=original.source_span,
                retrieval_backend=backend,
            ))
        return hits

    def count(self) -> int:
        """Return the number of indexed vectors."""
        if self._table is None:
            return 0
        return self._table.count_rows()

    def is_queryable(self) -> bool:
        """True if at least one vector is indexed."""
        return self.count() > 0

    def close(self) -> None:
        """LanceDB doesn't require explicit close."""
        pass

    def document_embeddings(self) -> dict[str, list[float]]:
        """Return mean embedding vector per document_id.

        Reads all chunk vectors from LanceDB, groups by document_id,
        and computes the centroid (mean) for each document.
        Used by Reporter curation to avoid re-embedding documents.
        """
        import numpy as np
        from collections import defaultdict

        if self._table is None:
            return {}
        try:
            tbl = self._table.to_arrow()
        except Exception:
            return {}
        if "document_id" not in tbl.column_names or "vector" not in tbl.column_names:
            return {}

        doc_ids = tbl.column("document_id").to_pylist()
        vectors = tbl.column("vector").to_pylist()

        docs: dict[str, list] = defaultdict(list)
        for did, vec in zip(doc_ids, vectors):
            docs[did].append(np.array(vec, dtype=np.float32))

        result: dict[str, list[float]] = {}
        for doc_id, vecs in docs.items():
            centroid = np.stack(vecs).mean(axis=0)
            result[doc_id] = centroid.tolist()
        return result

    def __enter__(self) -> "LanceDBIndex":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def _compute_centroids(store, lance: "LanceDBIndex") -> None:
    """Compute embedding centroid per document and store representative chunk IDs.

    For each document:
    1. Get all its chunks' dense vectors from LanceDB
    2. Compute the centroid (mean vector)
    3. Select the 3-5 chunks closest to the centroid (cosine similarity)
    4. Store the representative chunk_ids in document_centroids table
    """
    import numpy as np
    from collections import defaultdict

    if lance._table is None:
        return
    try:
        tbl = lance._table.to_arrow()
    except Exception:
        return
    if "chunk_id" not in tbl.column_names or "document_id" not in tbl.column_names or "vector" not in tbl.column_names:
        return

    chunk_ids = tbl.column("chunk_id").to_pylist()
    doc_ids = tbl.column("document_id").to_pylist()
    vectors = tbl.column("vector").to_pylist()

    # Group chunks by document
    docs: dict[str, list] = defaultdict(list)
    for cid, did, vec in zip(chunk_ids, doc_ids, vectors):
        docs[did].append((cid, np.array(vec, dtype=np.float32)))

    for doc_id, items in docs.items():
        if len(items) <= 3:
            # Small documents: use all chunks
            rep_ids = [cid for cid, _ in items]
        else:
            # Compute centroid
            vecs = np.stack([v for _, v in items])
            centroid = vecs.mean(axis=0)
            # Cosine similarity of each chunk to centroid
            norms = np.linalg.norm(vecs, axis=1) * np.linalg.norm(centroid)
            norms[norms == 0] = 1.0
            sims = (vecs @ centroid) / norms
            # Select top 5 (or fewer if doc is small)
            n_select = min(5, len(items))
            top_idx = np.argsort(-sims)[:n_select]
            rep_ids = [items[i][0] for i in sorted(top_idx)]
        store.put_centroid(doc_id, rep_ids, len(items))
    store.commit()

