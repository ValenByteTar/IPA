"""Promotion executor — physically copies documents to the main corpus.

This is independent of the Reporter. The Reporter can still produce reports,
 but promotion is decided by provenance + scoring (promotion_policy.py),
 not by report approval.

The executor copies:
  1. Documents + chunks from source corpus → main corpus (DocumentStore)
  2. Vector embeddings from source LanceDB → main LanceDB
  3. Provenance metadata from source document_sources → main document_sources

After a successful copy the promoted documents are PURGED from the source
(staging) corpus — the main corpus is canonical and the staging copy would
otherwise accumulate forever and double-count metrics. Raw source files in
Landing/web are NOT moved; the landing zone remains the artifact registry.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


def promote_documents_to_main(
    document_ids: list[str],
    source_corpus_path: Path,
    main_corpus_path: Path,
) -> dict[str, Any]:
    """Physically promote specific documents from a source corpus to the main corpus.

    Args:
        document_ids: List of document IDs to promote.
        source_corpus_path: Path to the source corpus (e.g. reporter corpus).
        main_corpus_path: Path to the main corpus (e.g. outputs/experiments/E12-corpus).

    Returns:
        Dict with promoted_docs, promoted_chunks, promoted_vectors counts.
    """
    if not document_ids:
        return {"promoted_docs": 0, "promoted_chunks": 0, "promoted_vectors": 0}

    source_store_db = source_corpus_path / "document_store.db"
    source_lancedb = source_corpus_path / "vector" / "lancedb"

    main_store_db = main_corpus_path / "document_store.db"
    main_bm25_db = main_corpus_path / "bm25_index.db"
    main_lancedb = main_corpus_path / "vector" / "lancedb"

    promoted_docs = 0
    promoted_chunks = 0
    promoted_vectors = 0

    import time as _time
    _t_copy = _time.time()

    # 1. Copy documents and chunks from source → main
    if source_store_db.exists():
        from ipa import DocumentStore, BM25Index
        from ipa.contracts import DocumentChunk

        main_store = DocumentStore(main_store_db)
        main_bm25 = BM25Index(main_bm25_db)
        source_conn = sqlite3.connect(str(source_store_db))

        try:
            # Build placeholder string for IN clause
            placeholders = ",".join("?" * len(document_ids))

            # Copy documents
            docs = source_conn.execute(
                f"SELECT document_id, artifact_id, parser_id, mime_type, pages, text, "
                f"elements_json, spans_json, stored_at FROM documents "
                f"WHERE document_id IN ({placeholders}) AND tombstoned=0",
                document_ids,
            ).fetchall()

            # Copy documents and chunks
            all_chunk_objs: list[Any] = []
            for doc_row in docs:
                doc_id = doc_row[0]
                # Check if already exists in main
                existing = main_store._conn.execute(
                    "SELECT 1 FROM documents WHERE document_id=?", (doc_id,)
                ).fetchone()
                if existing:
                    continue

                # Insert document
                main_store._conn.execute(
                    "INSERT OR REPLACE INTO documents "
                    "(document_id, artifact_id, parser_id, mime_type, pages, text, "
                    "elements_json, spans_json, stored_at, tombstoned) "
                    "VALUES (?,?,?,?,?,?,?,?,?,0)",
                    doc_row,
                )
                promoted_docs += 1

                # Copy chunks for this document
                chunks = source_conn.execute(
                    "SELECT chunk_id, document_id, content_hash, text, metadata_json, span_json "
                    "FROM chunks WHERE document_id=? AND tombstoned=0",
                    (doc_id,),
                ).fetchall()

                for ch_row in chunks:
                    all_chunk_objs.append(DocumentChunk(
                        chunk_id=ch_row[0], document_id=ch_row[1], content_hash=ch_row[2],
                        text=ch_row[3], metadata=json.loads(ch_row[4]),
                        source_span=None,
                    ))

            # Batch all chunk writes: one put_chunks + one BM25 add_chunks
            # (commit deferred) for the whole batch. Per-document calls made
            # every batch pay thousands of separate FTS/commit round-trips.
            if all_chunk_objs:
                main_store.put_chunks(all_chunk_objs)
                main_bm25.add_chunks(all_chunk_objs, commit=False)
                promoted_chunks = len(all_chunk_objs)

            # Copy provenance metadata from document_sources
            try:
                source_sources = source_conn.execute(
                    f"SELECT document_id, source_url, source_domain, provenance, quality_score "
                    f"FROM document_sources WHERE document_id IN ({placeholders})",
                    document_ids,
                ).fetchall()
                for row in source_sources:
                    main_store.put_source(row[0], row[1] or "", row[2] or "", row[3], float(row[4] or 0.0))
            except Exception:
                pass  # document_sources may not exist on older corpora

            main_store.commit()
            main_bm25._conn.commit()
        finally:
            source_conn.close()
            main_store.close()
            main_bm25.close()
    print(f"  [promote] phase copy: {_time.time()-_t_copy:.1f}s "
          f"({promoted_docs} docs, {promoted_chunks} chunks)", flush=True)

    _t_lance = _time.time()
    # 2. Copy LanceDB vectors from source → main
    if source_lancedb.exists() and main_lancedb.parent.exists():
        try:
            from ipa.indexes.lancedb_index import LanceDBIndex
            import pyarrow as pa

            main_lance = LanceDBIndex(main_lancedb, vector_dim=1024)
            source_lance = LanceDBIndex(source_lancedb, vector_dim=1024)

            if source_lance._table is not None:
                existing_ids: set[str] = set()
                if main_lance._table is not None:
                    try:
                        tbl = main_lance._table.to_arrow()
                        existing_ids = set(tbl.column("chunk_id").to_pylist())
                    except Exception as exc:
                        # If we can't read existing IDs, skip the merge
                        # rather than risk duplicates.
                        print(f"  [promote] cannot read main LanceDB IDs: {exc}", flush=True)
                        main_lance.close()
                        source_lance.close()
                        return {
                            "promoted_docs": promoted_docs,
                            "promoted_chunks": promoted_chunks,
                            "promoted_vectors": 0,
                        }

                source_tbl = source_lance._table.to_arrow()

                # Vectorized filter (pyarrow compute, C-speed): rows belonging
                # to this batch's documents whose chunk_id is not already in
                # main. Replaces the former per-row Python loop, which was the
                # promotion bottleneck (7 .as_py() calls × every source row ×
                # every batch).
                doc_mask = pa.compute.is_in(
                    source_tbl.column("document_id"),
                    value_set=pa.array(document_ids, type=pa.string()),
                )
                if existing_ids:
                    fresh_mask = pa.compute.invert(pa.compute.is_in(
                        source_tbl.column("chunk_id"),
                        value_set=pa.array(list(existing_ids), type=pa.string()),
                    ))
                    filtered = source_tbl.filter(pa.compute.and_(doc_mask, fresh_mask))
                else:
                    filtered = source_tbl.filter(doc_mask)

                if filtered.num_rows:
                    main_lance._ensure_table(filtered.column("vector")[0].as_py())
                    main_lance._table.add(filtered)
                    promoted_vectors = filtered.num_rows

            main_lance.close()
            source_lance.close()
        except Exception as exc:
            print(f"  [promote] LanceDB merge failed: {exc}", flush=True)
    print(f"  [promote] phase lance: {_time.time()-_t_lance:.1f}s "
          f"({promoted_vectors} vectors)", flush=True)

    _t_purge = _time.time()
    # 3. Purge the source copies — the main corpus is canonical. Without
    # this the staging corpus retains every promoted document forever and
    # its metrics diverge from the main corpus.
    purged = purge_promoted_from_source(document_ids, source_corpus_path, main_corpus_path)
    print(f"  [promote] phase purge: {_time.time()-_t_purge:.1f}s "
          f"({purged.get('purged_docs', 0)} docs)", flush=True)

    return {
        "promoted_docs": promoted_docs,
        "promoted_chunks": promoted_chunks,
        "promoted_vectors": promoted_vectors,
        "purged_docs": purged.get("purged_docs", 0),
        "purged_chunks": purged.get("purged_chunks", 0),
    }


def purge_promoted_from_source(
    document_ids: list[str],
    source_corpus_path: Path,
    main_corpus_path: Path,
) -> dict[str, int]:
    """Remove promoted documents from the source (staging) corpus.

    Promotion copies; without a purge the staging corpus keeps every promoted
    document forever and its metrics diverge from the main corpus. Only
    documents confirmed live (non-tombstoned) in the MAIN corpus are purged —
    the main corpus is canonical, so the staging copy is redundant.

    Purges: documents+chunks tombstoned in the source DocumentStore, chunks
    dropped from the source BM25 FTS index, vectors deleted from the source
    LanceDB table. All indexes are derived and rebuildable, so this is safe.

    Returns:
        Dict with purged_docs, purged_chunks counts.
    """
    if not document_ids:
        return {"purged_docs": 0, "purged_chunks": 0}

    source_store_db = source_corpus_path / "document_store.db"
    source_bm25_db = source_corpus_path / "bm25_index.db"
    source_lancedb = source_corpus_path / "vector" / "lancedb"
    main_store_db = main_corpus_path / "document_store.db"
    if not source_store_db.exists() or not main_store_db.exists():
        return {"purged_docs": 0, "purged_chunks": 0}

    from ipa import DocumentStore

    main_store = DocumentStore(main_store_db)
    try:
        placeholders = ",".join("?" * len(document_ids))
        # Only purge documents the main corpus actually holds live.
        rows = main_store._conn.execute(
            f"SELECT document_id FROM documents "
            f"WHERE document_id IN ({placeholders}) AND tombstoned=0",
            document_ids,
        ).fetchall()
        confirmed = [r[0] for r in rows]
    finally:
        main_store.close()
    if not confirmed:
        return {"purged_docs": 0, "purged_chunks": 0}

    purged_docs = 0
    purged_chunks = 0

    # 1. Source DocumentStore: count live chunks, then tombstone docs + chunks
    #    and drop their stale embedding-job rows.
    try:
        source_store = DocumentStore(source_store_db)
        try:
            ph = ",".join("?" * len(confirmed))
            row = source_store._conn.execute(
                f"SELECT COUNT(*) FROM chunks "
                f"WHERE document_id IN ({ph}) AND tombstoned=0",
                confirmed,
            ).fetchone()
            purged_chunks = int(row[0]) if row else 0
            # Batched tombstone: one UPDATE per table for the whole batch.
            # tombstone_document() commits per call — 1000 fsyncs per batch.
            source_store._conn.execute(
                f"UPDATE documents SET tombstoned = 1 WHERE document_id IN ({ph})",
                confirmed,
            )
            source_store._conn.execute(
                f"UPDATE chunks SET tombstoned = 1 WHERE document_id IN ({ph})",
                confirmed,
            )
            # embedding_jobs is a derived job queue — stale rows for purged
            # chunks would keep reporting as pending forever.
            source_store._conn.execute(
                f"DELETE FROM embedding_jobs WHERE chunk_id IN "
                f"(SELECT chunk_id FROM chunks WHERE document_id IN ({ph}) AND tombstoned=1)",
                confirmed,
            )
            source_store.commit()
            purged_docs = len(confirmed)
        finally:
            source_store.close()
    except Exception as exc:
        print(f"  [promote] source DocumentStore purge failed: {exc}", flush=True)

    # 2. Source BM25 index: drop chunk text from FTS.
    bm25_db = source_corpus_path / "bm25_index.db"
    if bm25_db.exists():
        try:
            from ipa import BM25Index
            bm25 = BM25Index(bm25_db)
            try:
                ph = ",".join("?" * len(confirmed))
                bm25._conn.execute(
                    f"DELETE FROM chunks_fts WHERE chunk_id IN "
                    f"(SELECT chunk_id FROM chunks_meta WHERE document_id IN ({ph}))",
                    confirmed,
                )
                bm25._conn.execute(
                    f"UPDATE chunks_meta SET tombstoned = 1 WHERE document_id IN ({ph})",
                    confirmed,
                )
                bm25._conn.commit()
            finally:
                bm25.close()
        except Exception as exc:
            print(f"  [promote] source BM25 purge failed: {exc}", flush=True)

    # 3. Source LanceDB: delete vectors of promoted documents.
    if source_lancedb.exists():
        try:
            from ipa.indexes.lancedb_index import LanceDBIndex
            source_lance = LanceDBIndex(source_lancedb, vector_dim=1024)
            if source_lance._table is not None:
                id_list = ", ".join(f"'{d}'" for d in confirmed)
                source_lance._table.delete(f"document_id IN ({id_list})")
            source_lance.close()
        except Exception as exc:
            print(f"  [promote] source LanceDB purge failed: {exc}", flush=True)

    return {"purged_docs": purged_docs, "purged_chunks": purged_chunks}


def process_promotion_queue(
    cluster_store: Any,
    source_corpus_path: Path,
    main_corpus_path: Path,
    *,
    batch_size: int = 1000,
) -> dict[str, Any]:
    """Process pending promotions from the cluster store's promotion queue.

    Documents are grouped by their source_corpus field. If source_corpus is
    empty or matches source_corpus_path, they are promoted from
    source_corpus_path. Otherwise, the source_corpus field is used as the
    source path.

    Large batches matter: every batch re-reads both LanceDB tables
    (to_arrow), so many small batches dominated promotion time. 1000 docs
    per batch turns ~28 full-table passes into 1-2.

    Args:
        cluster_store: TopicClusterStore with a promotion_queue table.
        source_corpus_path: Default source corpus path (fallback).
        main_corpus_path: Path to the main corpus.
        batch_size: Max documents to promote per batch.

    Returns:
        Dict with promoted_docs, promoted_chunks, promoted_vectors, processed count.
    """
    pending = cluster_store.pending_promotions()
    if not pending:
        return {"promoted_docs": 0, "promoted_chunks": 0, "promoted_vectors": 0, "processed": 0}

    # Group documents by source_corpus
    by_source: dict[str, list[str]] = {}
    for p in pending:
        src = p.get("source_corpus") or ""
        if not src:
            src = str(source_corpus_path)
        by_source.setdefault(src, []).append(p["document_id"])

    total_docs = 0
    total_chunks = 0
    total_vectors = 0
    processed = 0

    for src_path_str, doc_ids in by_source.items():
        src_path = Path(src_path_str)
        if not src_path.exists():
            # Source corpus doesn't exist — skip these documents
            for doc_id in doc_ids:
                cluster_store.mark_promotion_done(doc_id)
                processed += 1
            continue

        # Skip if source is the same as main (no-op promotion)
        if src_path.resolve() == main_corpus_path.resolve():
            for doc_id in doc_ids:
                cluster_store.mark_promotion_done(doc_id)
                processed += 1
            continue

        # Process in batches
        for i in range(0, len(doc_ids), batch_size):
            batch = doc_ids[i:i + batch_size]
            result = promote_documents_to_main(batch, src_path, main_corpus_path)
            total_docs += result["promoted_docs"]
            total_chunks += result["promoted_chunks"]
            total_vectors += result["promoted_vectors"]

            for doc_id in batch:
                cluster_store.mark_promotion_done(doc_id)
                processed += 1

    return {
        "promoted_docs": total_docs,
        "promoted_chunks": total_chunks,
        "promoted_vectors": total_vectors,
        "processed": processed,
    }
