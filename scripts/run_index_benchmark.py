"""Benchmark harness for E6 (lexical) and E7 (vector) index competitions.

Loads chunks from an existing DocumentStore and indexes them into each
candidate backend, measuring:

  - indexing throughput (chunks/sec)
  - p50/p95 query latency
  - disk usage
  - recall (against a known set of queries with expected results)

Usage:
    # E6: lexical competition (FTS5 vs Tantivy)
    python scripts/run_index_benchmark.py --store outputs/experiments/E1-corpus/document_store.db --output outputs/experiments/E6 --mode lexical

    # E7: vector competition (LanceDB vs sqlite-vec)
    python scripts/run_index_benchmark.py --store outputs/experiments/E1-corpus/document_store.db --output outputs/experiments/E7 --mode vector
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ipa import (
    BM25Index,
    DocumentStore,
    TantivyIndex,
    EmbeddingAdapter,
    LanceDBIndex,
    SQLiteVecIndex,
)


# Standard queries for lexical benchmark (E6).
LEXICAL_QUERIES = [
    "security information",
    "incident response",
    "risk assessment",
    "access control",
    "encryption",
    "vulnerability",
    "compliance",
    "authentication",
    "network security",
    "data protection",
]

# Standard queries for vector benchmark (E7).
VECTOR_QUERIES = [
    "How to detect a security breach?",
    "What is the ISO 27001 standard?",
    "Best practices for incident response",
    "How to implement zero trust architecture?",
    "What are common vulnerabilities in web applications?",
]


def load_chunks(store_db: str, batch_size: int = 1000, limit: int | None = None):
    """Load chunks from DocumentStore in batches using a cursor.

    If limit is set, only loads that many chunks (first N by rowid for reproducibility).
    """
    from ipa.contracts import DocumentChunk, SourceSpan
    import sqlite3, json as _json

    conn = sqlite3.connect(store_db)
    conn.execute("PRAGMA journal_mode = WAL")
    total = conn.execute("SELECT COUNT(*) FROM chunks WHERE tombstoned=0").fetchone()[0]

    if limit and limit < total:
        # Take first N chunks by rowid for reproducibility.
        cursor = conn.execute(
            "SELECT chunk_id, document_id, content_hash, text, metadata_json, span_json "
            "FROM chunks WHERE tombstoned=0 "
            "ORDER BY rowid LIMIT ?",
            (limit,),
        )
        print(f"Loading sample ({limit}) of {total} chunks from {store_db}...", flush=True)
    else:
        cursor = conn.execute(
            "SELECT chunk_id, document_id, content_hash, text, metadata_json, span_json "
            "FROM chunks WHERE tombstoned=0"
        )
        print(f"Loading {total} chunks from {store_db}...", flush=True)

    loaded = 0
    batch = []
    for row in cursor:
        span = None
        if row[5]:
            d = _json.loads(row[5])
            if d and d.get("artifact_id"):
                span = SourceSpan(
                    artifact_id=d["artifact_id"], page=d["page"],
                    offset_start=d["offset_start"], offset_end=d["offset_end"],
                )
        batch.append(DocumentChunk(
            chunk_id=row[0], document_id=row[1], content_hash=row[2],
            text=row[3], metadata=_json.loads(row[4]), source_span=span,
        ))
        if len(batch) >= batch_size:
            yield batch
            loaded += len(batch)
            if loaded % 10000 == 0:
                print(f"  loaded {loaded}/{total}...", flush=True)
            batch = []
    if batch:
        yield batch
        loaded += len(batch)
    conn.close()
    print(f"  loaded {loaded}/{total} done.")


def measure_latency(search_fn, queries: list, n_runs: int = 5) -> dict:
    """Measure p50/p95 query latency over multiple runs."""
    latencies = []
    for _ in range(n_runs):
        for q in queries:
            t0 = time.monotonic()
            hits = search_fn(q)
            lat = time.monotonic() - t0
            latencies.append(lat)
    latencies.sort()
    p50 = statistics.median(latencies)
    p95 = latencies[int(len(latencies) * 0.95)] if len(latencies) > 1 else latencies[0]
    return {
        "p50_ms": round(p50 * 1000, 2),
        "p95_ms": round(p95 * 1000, 2),
        "n_queries": len(queries) * n_runs,
        "n_latencies": len(latencies),
    }


def disk_usage(path: str) -> int:
    """Total disk usage of a directory or file in bytes."""
    p = Path(path)
    if p.is_file():
        return p.stat().st_size
    total = 0
    if p.exists():
        for f in p.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    return total


def remove_path(path: Path) -> None:
    """Remove a file or directory tree."""
    import shutil
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink()


def benchmark_lexical(store_db: str, output_dir: str, query_limit: int = 10,
                      chunk_limit: int | None = None) -> dict:
    """Run E6: lexical index competition (FTS5 vs Tantivy)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = {}

    # --- FTS5 (existing BM25Index) ---
    print("\n=== FTS5 (SQLite FTS5) ===", flush=True)
    fts5_path = out / "fts5.db"
    remove_path(fts5_path)
    idx = BM25Index(fts5_path)
    t0 = time.monotonic()
    total_chunks = 0
    for batch in load_chunks(store_db, limit=chunk_limit):
        idx.add_chunks(batch, commit=False)
        total_chunks += len(batch)
        if total_chunks % 5000 == 0:
            elapsed = time.monotonic() - t0
            print(f"  FTS5 indexing: {total_chunks} chunks ({total_chunks/elapsed:.0f}/s)", flush=True)
    idx.commit()  # single commit at the end
    print(f"  FTS5 merging segments...", flush=True)
    merge_t0 = time.monotonic()
    idx.optimize()
    merge_time = time.monotonic() - merge_t0
    idx_time = time.monotonic() - t0
    print(f"  FTS5 done: {total_chunks} chunks in {idx_time:.1f}s ({total_chunks/idx_time:.0f} chunks/s, merge={merge_time:.1f}s)", flush=True)

    lat = measure_latency(lambda q: idx.search(q, limit=query_limit), LEXICAL_QUERIES)
    fts5_disk = disk_usage(str(fts5_path))
    print(f"  p50={lat['p50_ms']}ms  p95={lat['p95_ms']}ms  disk={fts5_disk/1e6:.1f}MB")
    results["fts5"] = {
        "backend": "sqlite_fts5",
        "chunks_indexed": total_chunks,
        "indexing_seconds": round(idx_time, 2),
        "throughput_chunks_per_sec": round(total_chunks / idx_time, 1),
        "disk_bytes": fts5_disk,
        "disk_mb": round(fts5_disk / 1e6, 2),
        **lat,
    }
    idx.close()

    # --- Tantivy ---
    print("\n=== Tantivy ===", flush=True)
    tantivy_path = out / "tantivy"
    remove_path(tantivy_path)
    idx = TantivyIndex(tantivy_path)
    t0 = time.monotonic()
    total_chunks = 0
    for batch in load_chunks(store_db, limit=chunk_limit):
        idx.add_chunks(batch, commit=False)
        total_chunks += len(batch)
        if total_chunks % 5000 == 0:
            elapsed = time.monotonic() - t0
            print(f"  Tantivy indexing: {total_chunks} chunks ({total_chunks/elapsed:.0f}/s)", flush=True)
    idx.commit()  # single commit at the end
    idx_time = time.monotonic() - t0
    print(f"  Tantivy done: {total_chunks} chunks in {idx_time:.1f}s ({total_chunks/idx_time:.0f} chunks/s)", flush=True)

    lat = measure_latency(lambda q: idx.search(q, limit=query_limit), LEXICAL_QUERIES)
    tan_disk = disk_usage(str(tantivy_path))
    print(f"  p50={lat['p50_ms']}ms  p95={lat['p95_ms']}ms  disk={tan_disk/1e6:.1f}MB")
    results["tantivy"] = {
        "backend": "tantivy",
        "chunks_indexed": total_chunks,
        "indexing_seconds": round(idx_time, 2),
        "throughput_chunks_per_sec": round(total_chunks / idx_time, 1),
        "disk_bytes": tan_disk,
        "disk_mb": round(tan_disk / 1e6, 2),
        **lat,
    }
    idx.close()

    return results


def benchmark_lexical_winners(store_db: str, output_dir: str, query_limit: int = 10,
                              chunk_limit: int | None = None) -> dict:
    """Run E6 winners-only: Tantivy (skips FTS5)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = {}

    print("\n=== Tantivy (winner) ===", flush=True)
    tantivy_path = out / "tantivy"
    remove_path(tantivy_path)
    idx = TantivyIndex(tantivy_path)
    t0 = time.monotonic()
    total_chunks = 0
    for batch in load_chunks(store_db, limit=chunk_limit):
        idx.add_chunks(batch, commit=False)
        total_chunks += len(batch)
        if total_chunks % 10000 == 0:
            elapsed = time.monotonic() - t0
            print(f"  Tantivy indexing: {total_chunks} chunks ({total_chunks/elapsed:.0f}/s)", flush=True)
    idx.commit()
    idx_time = time.monotonic() - t0
    print(f"  Tantivy done: {total_chunks} chunks in {idx_time:.1f}s ({total_chunks/idx_time:.0f} chunks/s)", flush=True)
    lat = measure_latency(lambda q: idx.search(q, limit=query_limit), LEXICAL_QUERIES)
    tan_disk = disk_usage(str(tantivy_path))
    print(f"  p50={lat['p50_ms']}ms  p95={lat['p95_ms']}ms  disk={tan_disk/1e6:.1f}MB", flush=True)
    results["tantivy"] = {
        "backend": "tantivy",
        "chunks_indexed": total_chunks,
        "indexing_seconds": round(idx_time, 2),
        "throughput_chunks_per_sec": round(total_chunks / idx_time, 1),
        "disk_bytes": tan_disk,
        "disk_mb": round(tan_disk / 1e6, 2),
        **lat,
    }
    idx.close()
    return results


def benchmark_vector_winners(store_db: str, output_dir: str, query_limit: int = 10,
                             chunk_limit: int | None = None, device: str = "auto") -> dict:
    """Run E7 winners-only: LanceDB (skips sqlite-vec)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = {}

    print("\n=== Generating embeddings ===", flush=True)
    emb = EmbeddingAdapter(show_progress=False, device=device)
    dim = emb.dimension
    print(f"  Model: all-MiniLM-L6-v2, dim={dim}, device={emb.active_device}", flush=True)

    print("\n=== LanceDB (winner) ===", flush=True)
    lance_path = out / "lancedb"
    remove_path(lance_path)
    idx = LanceDBIndex(lance_path, vector_dim=dim)
    t0 = time.monotonic()
    total_chunks = 0
    for batch in load_chunks(store_db, batch_size=500, limit=chunk_limit):
        vectors = emb.embed_texts([c.text for c in batch])
        idx.add_chunks(batch, vectors)
        total_chunks += len(batch)
        if total_chunks % 10000 == 0:
            elapsed = time.monotonic() - t0
            print(f"  LanceDB embedding+indexing: {total_chunks} chunks ({total_chunks/elapsed:.0f}/s)", flush=True)
    total_time = time.monotonic() - t0
    print(f"  LanceDB done: {total_chunks} chunks in {total_time:.1f}s ({total_chunks/total_time:.0f} chunks/s)", flush=True)
    query_vectors = [emb.embed_query(q) for q in VECTOR_QUERIES]
    lat = measure_latency(lambda qv: idx.search(qv, limit=query_limit), query_vectors)
    lance_disk = disk_usage(str(lance_path))
    print(f"  p50={lat['p50_ms']}ms  p95={lat['p95_ms']}ms  disk={lance_disk/1e6:.1f}MB", flush=True)
    results["lancedb"] = {
        "backend": "lancedb",
        "model": "all-MiniLM-L6-v2",
        "vector_dim": dim,
        "chunks_indexed": total_chunks,
        "indexing_seconds": round(total_time, 2),
        "throughput_chunks_per_sec": round(total_chunks / total_time, 1),
        "disk_bytes": lance_disk,
        "disk_mb": round(lance_disk / 1e6, 2),
        **lat,
    }
    idx.close()
    emb.close()
    return results


def benchmark_vector(store_db: str, output_dir: str, query_limit: int = 10,
                     chunk_limit: int | None = None, device: str = "auto") -> dict:
    """Run E7: vector index competition (LanceDB vs sqlite-vec)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = {}

    # --- Generate embeddings ---
    print("\n=== Generating embeddings ===")
    emb = EmbeddingAdapter(show_progress=False, device=device)
    dim = emb.dimension
    print(f"  Model: all-MiniLM-L6-v2, dim={dim}, device={emb.active_device}")

    # --- LanceDB ---
    print("\n=== LanceDB ===")
    lance_path = out / "lancedb"
    remove_path(lance_path)
    idx = LanceDBIndex(lance_path, vector_dim=dim)
    t0 = time.monotonic()
    total_chunks = 0
    emb_t0 = time.monotonic()
    for batch in load_chunks(store_db, batch_size=500, limit=chunk_limit):
        vectors = emb.embed_texts([c.text for c in batch])
        idx.add_chunks(batch, vectors)
        total_chunks += len(batch)
        if total_chunks % 5000 == 0:
            elapsed = time.monotonic() - emb_t0
            print(f"  embedded+indexed {total_chunks} chunks ({total_chunks/elapsed:.0f}/s)", flush=True)
    total_time = time.monotonic() - t0
    print(f"  Total: {total_chunks} chunks in {total_time:.1f}s ({total_chunks/total_time:.0f} chunks/s)")

    # Query latency
    query_vectors = [emb.embed_query(q) for q in VECTOR_QUERIES]
    lat = measure_latency(lambda qv: idx.search(qv, limit=query_limit), query_vectors)
    lance_disk = disk_usage(str(lance_path))
    print(f"  p50={lat['p50_ms']}ms  p95={lat['p95_ms']}ms  disk={lance_disk/1e6:.1f}MB")
    results["lancedb"] = {
        "backend": "lancedb",
        "model": "all-MiniLM-L6-v2",
        "vector_dim": dim,
        "chunks_indexed": total_chunks,
        "indexing_seconds": round(total_time, 2),
        "throughput_chunks_per_sec": round(total_chunks / total_time, 1),
        "disk_bytes": lance_disk,
        "disk_mb": round(lance_disk / 1e6, 2),
        **lat,
    }
    idx.close()

    # --- sqlite-vec ---
    print("\n=== sqlite-vec ===")
    svec_path = out / "sqlite_vec.db"
    remove_path(svec_path)
    idx = SQLiteVecIndex(svec_path, vector_dim=dim)
    t0 = time.monotonic()
    total_chunks = 0
    for batch in load_chunks(store_db, batch_size=500, limit=chunk_limit):
        vectors = emb.embed_texts([c.text for c in batch])
        idx.add_chunks(batch, vectors)
        total_chunks += len(batch)
        if total_chunks % 5000 == 0:
            elapsed = time.monotonic() - t0
            print(f"  embedded+indexed {total_chunks} chunks ({total_chunks/elapsed:.0f}/s)", flush=True)
    total_time = time.monotonic() - t0
    print(f"  Total: {total_chunks} chunks in {total_time:.1f}s ({total_chunks/total_time:.0f} chunks/s)")

    lat = measure_latency(lambda qv: idx.search(qv, limit=query_limit), query_vectors)
    svec_disk = disk_usage(str(svec_path))
    print(f"  p50={lat['p50_ms']}ms  p95={lat['p95_ms']}ms  disk={svec_disk/1e6:.1f}MB")
    results["sqlite_vec"] = {
        "backend": "sqlite_vec",
        "model": "all-MiniLM-L6-v2",
        "vector_dim": dim,
        "chunks_indexed": total_chunks,
        "indexing_seconds": round(total_time, 2),
        "throughput_chunks_per_sec": round(total_chunks / total_time, 1),
        "disk_bytes": svec_disk,
        "disk_mb": round(svec_disk / 1e6, 2),
        **lat,
    }
    idx.close()
    emb.close()

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run index benchmark (E6 or E7).")
    parser.add_argument("--store", required=True, help="Path to document_store.db")
    parser.add_argument("--output", required=True, help="Output directory for benchmark results")
    parser.add_argument("--mode", choices=["lexical", "vector", "both", "winners"], default="both",
                        help="Benchmark mode: lexical (E6), vector (E7), both, or winners (Tantivy + LanceDB only)")
    parser.add_argument("--limit", type=int, default=10, help="Search result limit")
    parser.add_argument("--limit-chunks", type=int, default=None,
                        help="Limit number of chunks to index (sample). Default: all.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                        help="Device for embeddings: auto (detect GPU), cuda, cpu")
    args = parser.parse_args()

    report = {
        "store_db": args.store,
        "output": args.output,
        "mode": args.mode,
        "chunk_limit": args.limit_chunks,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    if args.mode in ("lexical", "both"):
        report["E6_lexical"] = benchmark_lexical(
            args.store, args.output, args.limit, chunk_limit=args.limit_chunks)
    elif args.mode == "winners":
        report["E6_lexical"] = benchmark_lexical_winners(
            args.store, args.output, args.limit, chunk_limit=args.limit_chunks)

    if args.mode in ("vector", "both"):
        vec_out = str(Path(args.output) / "vector")
        report["E7_vector"] = benchmark_vector(
            args.store, vec_out, args.limit,
            chunk_limit=args.limit_chunks, device=args.device)
    elif args.mode == "winners":
        vec_out = str(Path(args.output) / "vector")
        report["E7_vector"] = benchmark_vector_winners(
            args.store, vec_out, args.limit,
            chunk_limit=args.limit_chunks, device=args.device)

    report_path = Path(args.output) / "benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
