"""Fast path pipeline CLI over a directory of artifacts.

Canonical implementation; entrypoint is a thin wrapper
(``scripts/cli/run_fast_path.py``).

Usage:
    python scripts/cli/run_fast_path.py --input data/sample/input --output outputs/experiments/E1
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

from ipa import FastPathRunner, TraceLog


def index_lancedb(store_db: Path, lance_path: Path, batch_size: int = 64) -> dict:
    """Index chunks from document_store.db into LanceDB using BGE-M3 embeddings.

    Only embeds NEW chunks (not already in LanceDB) â€” idempotent and incremental.
    Runs after BM25 indexing so the corpus is queryable immediately.
    Returns a summary dict with counts and timing.
    """
    from ipa import DocumentStore
    from ipa.indexes.embedding_adapter import EmbeddingAdapter
    from ipa.indexes.lancedb_index import LanceDBIndex

    store = DocumentStore(store_db)
    embedding = EmbeddingAdapter(batch_size=batch_size, show_progress=True)
    lance = LanceDBIndex(lance_path, vector_dim=1024)
    result = _index_lancedb_incremental(store, lance, embedding, batch_size)
    embedding.close()
    lance.close()
    store.close()
    print(f"  LanceDB: {result['new_chunks']} new chunks embedded + indexed ({result['skipped']} skipped) in {result['elapsed_seconds']}s", flush=True)
    return result


def _index_lancedb_incremental(store, lance, embedding, batch_size: int = 64) -> dict:
    """Index only new chunks into LanceDB. Reuses already-loaded embedding model."""
    import time as _time

    # Get existing chunk_ids in LanceDB to skip already-indexed chunks
    existing_ids: set[str] = set()
    if lance._table is not None:
        try:
            tbl = lance._table.to_arrow()
            if "chunk_id" in tbl.column_names:
                existing_ids = set(tbl.column("chunk_id").to_pylist())
        except Exception:
            pass

    total_chunks = 0
    skipped = len(existing_ids)
    start = _time.monotonic()

    batch: list = []
    BATCH = batch_size

    def flush_batch():
        nonlocal total_chunks
        if not batch:
            return
        texts = [c.text for c in batch]
        dense, sparse = embedding.embed_texts_hybrid(texts)
        lance.add_chunks(batch, dense, sparse)
        total_chunks += len(batch)
        batch.clear()

    for chunk in store.all_chunks():
        if chunk.chunk_id in existing_ids:
            continue
        batch.append(chunk)
        if len(batch) >= BATCH:
            flush_batch()
    flush_batch()

    elapsed = _time.monotonic() - start
    result = {"chunks_indexed": total_chunks + skipped, "new_chunks": total_chunks, "skipped": skipped, "elapsed_seconds": round(elapsed, 2)}
    # Compute document centroids (representative chunks per document)
    if total_chunks > 0 or skipped > 0:
        try:
            from ipa.indexes.lancedb_index import _compute_centroids
            _compute_centroids(store, lance)
        except Exception as exc:
            print(f"  [centroid] computation skipped: {exc}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RES-023 fast path pipeline.")
    parser.add_argument("--input", default="Landing", help="Landing directory with artifacts (default: Landing).")
    parser.add_argument(
        "--output",
        default="outputs/experiments/E1",
        help="Output directory for databases and results.",
    )
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    parser.add_argument("--query", default=None, help="Optional query to run after ingestion.")
    parser.add_argument("--trace-db", default=None, help="Enable E11 traceability, writing events to this SQLite DB.")
    parser.add_argument("--no-lancedb", action="store_true", help="Skip LanceDB vector indexing (BM25 only).")
    parser.add_argument("--watch", type=int, default=0, metavar="SECONDS", help="Keep running: re-ingest + re-index every N seconds (keeps BGE-M3 in memory).")
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    trace = None
    if args.trace_db:
        trace = TraceLog(args.trace_db)
        print(f"Tracing enabled: {args.trace_db}")

    runner = FastPathRunner(
        landing_db=output / "landing.db",
        store_db=output / "document_store.db",
        index_db=output / "bm25_index.db",
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        landing_root=input_dir,
        trace_log=trace,
    )

    start = time.monotonic()
    results = runner.ingest_directory(input_dir)
    total_elapsed = time.monotonic() - start

    report = {
        "input": str(input_dir),
        "output": str(output),
        "artifacts_ingested": len(results),
        "total_chunks": sum(r.chunks_created for r in results),
        "first_queryable": all(r.first_queryable for r in results) if results else False,
        "total_elapsed_seconds": round(total_elapsed, 4),
        "results": [
            {
                "artifact_id": r.artifact_id,
                "mime_type": r.mime_type,
                "parser_id": r.parser_id,
                "document_id": r.document_id,
                "pages": r.pages,
                "chunks_created": r.chunks_created,
                "first_queryable": r.first_queryable,
                "elapsed_seconds": round(r.elapsed_seconds, 4),
                "errors": r.errors,
            }
            for r in results
        ],
    }

    if args.query:
        hits = runner.search(args.query, limit=10)
        report["query"] = args.query
        report["search_hits"] = [
            {
                "chunk_id": h.chunk_id,
                "score": h.score,
                "retrieval_backend": h.retrieval_backend,
                "page": h.source_span.page if h.source_span else None,
            }
            for h in hits
        ]

    runner.close()

    # Stage 2: LanceDB vector indexing (BGE-M3) â€” runs after BM25 so corpus is queryable immediately
    lance_report = None
    if not args.no_lancedb:
        print("Indexing LanceDB (BGE-M3 embeddings)...", flush=True)
        lance_path = output / "vector" / "lancedb"
        lance_report = index_lancedb(output / "document_store.db", lance_path)
        report["lancedb"] = lance_report

    if trace:
        s = trace.summary()
        report["trace_summary"] = s
        trace.close()
        print(f"Trace: {s['total_events']} events, {s['artifacts']} artifacts, {s['failed_events']} failed")

    report_path = output / "fast_path_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Ingested {len(results)} artifacts, {report['total_chunks']} chunks in {total_elapsed:.3f}s")
    print(f"first_queryable={report['first_queryable']}")
    print(f"Report: {report_path}")
    if args.query:
        hits = report.get("search_hits", [])
        print(f"Query '{args.query}': {len(hits)} hits")
        for h in hits[:5]:
            print(f"  {h['chunk_id']} score={h['score']:.4f} page={h['page']}")

    # --- Watch mode: keep re-ingesting + re-indexing with BGE-M3 in memory ---
    if args.watch > 0 and not args.no_lancedb:
        import time as _time
        from ipa import DocumentStore
        from ipa.indexes.embedding_adapter import EmbeddingAdapter
        from ipa.indexes.lancedb_index import LanceDBIndex

        lance_path = output / "vector" / "lancedb"
        print(f"\nWatch mode: re-ingesting every {args.watch}s (BGE-M3 stays in GPU memory)", flush=True)

        # Keep model and LanceDB open for the entire watch session
        embedding = EmbeddingAdapter(batch_size=64, show_progress=False)
        lance = LanceDBIndex(lance_path, vector_dim=1024)

        iteration = 0
        while True:
            _time.sleep(args.watch)
            iteration += 1
            print(f"\n[watch] Iteration {iteration} â€” re-ingesting...", flush=True)

            # Re-run fast path (BM25 â€” idempotent, skips already-indexed)
            runner2 = FastPathRunner(
                landing_db=output / "landing.db",
                store_db=output / "document_store.db",
                index_db=output / "bm25_index.db",
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                landing_root=input_dir,
            )
            try:
                results2 = runner2.ingest_directory(input_dir)
                new_artifacts = sum(1 for r in results2 if r.chunks_created > 0)
                new_chunks = sum(r.chunks_created for r in results2)
                print(f"  BM25: {new_artifacts} new artifacts, {new_chunks} new chunks", flush=True)
            except (PermissionError, OSError) as exc:
                # File might be being written by scraper â€” skip this iteration
                print(f"  BM25: skipped iteration (file busy: {exc})", flush=True)
            finally:
                runner2.close()

            # Incremental LanceDB indexing â€” model already loaded, no reload
            store = DocumentStore(output / "document_store.db")
            result = _index_lancedb_incremental(store, lance, embedding, batch_size=64)
            store.close()
            if result["new_chunks"] > 0:
                print(f"  LanceDB: {result['new_chunks']} new chunks embedded ({result['skipped']} skipped) in {result['elapsed_seconds']}s", flush=True)
            else:
                print(f"  LanceDB: no new chunks ({result['skipped']} already indexed)", flush=True)


if __name__ == "__main__":
    main()
