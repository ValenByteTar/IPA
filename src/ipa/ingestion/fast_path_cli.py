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
import threading
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


def _index_lancedb_incremental(store, lance, embedding, batch_size: int = 64,
                               known_ids: set | None = None,
                               centroids: bool = True) -> dict:
    """Index only new chunks into LanceDB. Reuses already-loaded embedding model.

    known_ids: optional caller-owned set of already-indexed chunk_ids. When
    provided it is used as the skip set and updated in place — avoids
    re-reading the whole LanceDB table on every pass (used by the
    background drain loop).
    centroids: recompute document centroids after indexing. Pass False for
    intermediate passes; run once with True at the end.
    """
    import time as _time

    # Get existing chunk_ids in LanceDB to skip already-indexed chunks
    existing_ids: set[str] = known_ids if known_ids is not None else set()
    if known_ids is None and lance._table is not None:
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
        if known_ids is not None:
            known_ids.update(c.chunk_id for c in batch)
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
    if centroids and (total_chunks > 0 or skipped > 0):
        try:
            from ipa.indexes.lancedb_index import _compute_centroids
            _compute_centroids(store, lance)
        except Exception as exc:
            print(f"  [centroid] computation skipped: {exc}", flush=True)
    return result


def _embed_drain_loop(store_db: Path, lance_path: Path, done: threading.Event,
                      stats: dict, batch_size: int = 64) -> None:
    """Background thread: continuously drain new chunks into LanceDB.

    Runs concurrently with ingestion — parsing/chunking (CPU) overlaps
    embedding (GPU). The DocumentStore uses WAL so a second connection reads
    committed chunks while the ingest writer keeps appending. Idempotent per
    chunk_id: intermediate passes only embed what is new.
    """
    from ipa import DocumentStore
    from ipa.indexes.embedding_adapter import EmbeddingAdapter
    from ipa.indexes.lancedb_index import LanceDBIndex

    store = DocumentStore(store_db)
    embedding = EmbeddingAdapter(batch_size=batch_size, show_progress=False)
    lance = LanceDBIndex(lance_path, vector_dim=1024)
    known: set = set()
    stats["indexed"] = 0
    stats["idle"] = False
    try:
        while not done.is_set():
            result = _index_lancedb_incremental(
                store, lance, embedding, batch_size,
                known_ids=known, centroids=False)
            stats["indexed"] += result["new_chunks"]
            stats["idle"] = result["new_chunks"] == 0
            if result["new_chunks"] > 0:
                print(f"  [embed] +{result['new_chunks']} chunks "
                      f"(total {stats['indexed']})", flush=True)
            else:
                done.wait(10)
        # Final drain: ingest is over, every chunk is committed. Loop until a
        # full scan finds nothing new (a scan that started before the last
        # commit could miss it otherwise).
        while True:
            result = _index_lancedb_incremental(
                store, lance, embedding, batch_size,
                known_ids=known, centroids=False)
            stats["indexed"] += result["new_chunks"]
            if result["new_chunks"] > 0:
                print(f"  [embed] final drain +{result['new_chunks']} chunks "
                      f"(total {stats['indexed']})", flush=True)
            else:
                break
        stats["idle"] = True
        try:
            from ipa.indexes.lancedb_index import _compute_centroids
            _compute_centroids(store, lance)
        except Exception as exc:
            print(f"  [centroid] computation skipped: {exc}", flush=True)
    finally:
        embedding.close()
        lance.close()
        store.close()


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
    parser.add_argument("--idle-exit", type=int, default=0, metavar="N", help="In watch mode: exit after N consecutive iterations with no new artifacts/chunks. 0 = run forever (default).")
    parser.add_argument("--idle-gate", default=None, metavar="PATH", help="In watch mode: only start counting idle iterations once this file exists (e.g. scraper-done sentinel).")
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

    # Stage 2: LanceDB vector indexing runs CONCURRENTLY in a background
    # thread — CPU parsing/chunking overlaps GPU embedding (BGE-M3). The
    # thread drains committed chunks as the ingest produces them.
    embed_done: threading.Event | None = None
    embed_thread: threading.Thread | None = None
    embed_stats: dict | None = None
    if not args.no_lancedb:
        embed_done = threading.Event()
        embed_stats = {"indexed": 0, "idle": False}
        embed_thread = threading.Thread(
            target=_embed_drain_loop,
            args=(output / "document_store.db", output / "vector" / "lancedb",
                  embed_done, embed_stats),
            kwargs={"batch_size": 64},
            daemon=True,
            name="lancedb-embed-drain",
        )
        embed_thread.start()
        print("LanceDB embedder: background drain started (BGE-M3)", flush=True)

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

    # LanceDB indexing is already running in the background drain thread;
    # record a snapshot (final counts land when the thread is joined below).
    if embed_stats is not None:
        report["lancedb"] = {
            "mode": "background-drain",
            "chunks_indexed_so_far": embed_stats.get("indexed", 0),
        }

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

    # --- Watch mode: keep re-ingesting BM25; embeddings drain in background ---
    if args.watch > 0 and not args.no_lancedb:
        import time as _time

        print(f"\nWatch mode: re-ingesting every {args.watch}s "
              f"(embeddings drain in background)", flush=True)

        iteration = 0
        idle_rounds = 0
        idle_gate = Path(args.idle_gate) if args.idle_gate else None
        while True:
            _time.sleep(args.watch)
            iteration += 1
            print(f"\n[watch] Iteration {iteration} — re-ingesting...", flush=True)

            # Re-run fast path (BM25 — idempotent, skips already-indexed)
            new_artifacts = 0
            new_chunks = 0
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
                # File might be being written by scraper — skip this iteration
                print(f"  BM25: skipped iteration (file busy: {exc})", flush=True)
                new_artifacts = -1  # busy file is not idle
            finally:
                runner2.close()

            embed_indexed = embed_stats.get("indexed", 0) if embed_stats else 0
            embed_idle = embed_stats.get("idle", True) if embed_stats else True
            print(f"  LanceDB: {embed_indexed} chunks embedded "
                  f"({'idle' if embed_idle else 'draining'})", flush=True)

            # Idle-exit: once the gate file exists (scraper done), stop the
            # watch after N consecutive iterations with no new work AND the
            # embedding drain fully caught up.
            if args.idle_exit > 0 and (idle_gate is None or idle_gate.exists()):
                if new_artifacts == 0 and embed_idle:
                    idle_rounds += 1
                    print(f"  [watch] idle {idle_rounds}/{args.idle_exit}", flush=True)
                    if idle_rounds >= args.idle_exit:
                        print(f"  [watch] {args.idle_exit} consecutive idle iterations — exiting watch mode", flush=True)
                        break
                else:
                    idle_rounds = 0

    # Shutdown: signal the embedder that ingestion is over and wait for the
    # final drain so no committed chunk is left unembedded.
    if embed_thread is not None:
        print("Waiting for LanceDB embedder final drain...", flush=True)
        embed_done.set()
        embed_thread.join()
        print(f"LanceDB embedder done: {embed_stats.get('indexed', 0)} chunks embedded", flush=True)


if __name__ == "__main__":
    main()
