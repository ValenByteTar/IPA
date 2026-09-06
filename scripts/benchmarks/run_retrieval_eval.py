"""Run E10 â€” End-to-end retrieval evaluation.

Compares Tantivy (lexical), LanceDB (vector), and hybrid retrieval on the
full 166k chunk corpus using a synthetic query set with ground truth.

Usage:
    python scripts/benchmarks/run_retrieval_eval.py
        --store outputs/experiments/E1-corpus/document_store.db
        --tantivy outputs/experiments/E6-full/tantivy
        --lancedb outputs/experiments/E6-full/vector
        --output outputs/experiments/E10
        --n-queries 200
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E10 retrieval evaluation.")
    parser.add_argument("--store", required=True, help="Path to document_store.db")
    parser.add_argument("--tantivy", required=True, help="Path to Tantivy index directory")
    parser.add_argument("--lancedb", required=True, help="Path to LanceDB directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--n-queries", type=int, default=200, help="Number of eval queries")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for query generation")
    parser.add_argument("--k", type=int, default=20, help="Top-k results to retrieve per query")
    parser.add_argument(
        "--backends", nargs="+",
        default=["tantivy", "lancedb", "hybrid"],
        help="Backends to evaluate",
    )
    args = parser.parse_args()

    from ipa import TantivyIndex, LanceDBIndex, EmbeddingAdapter
    from ipa.agentic.retrieval_eval import (
        generate_query_set,
        compute_metrics_from_hits,
        hybrid_fuse,
    )

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Generate query set
    # ------------------------------------------------------------------
    print(f"\nE10: End-to-end retrieval evaluation", flush=True)
    print(f"  Generating {args.n_queries} queries from corpus (seed={args.seed})...", flush=True)
    queries = generate_query_set(args.store, n_queries=args.n_queries, seed=args.seed)
    print(f"  Generated {len(queries)} queries", flush=True)

    # Save query set for reproducibility.
    query_set_path = out / "query_set.json"
    query_set_path.write_text(json.dumps([
        {
            "query_id": q.query_id,
            "query_text": q.query_text,
            "relevant_chunk_id": q.relevant_chunk_id,
            "relevant_document_id": q.relevant_document_id,
            "source_text_preview": q.source_text_preview,
        }
        for q in queries
    ], indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Query set saved: {query_set_path}", flush=True)

    # ------------------------------------------------------------------
    # 2. Load chunk_id -> document_id mapping from store
    # ------------------------------------------------------------------
    import sqlite3
    print(f"  Loading chunk->doc mapping from store...", flush=True)
    conn = sqlite3.connect(args.store)
    chunk_to_doc = {}
    for row in conn.execute("SELECT chunk_id, document_id FROM chunks"):
        chunk_to_doc[row[0]] = row[1]
    conn.close()
    print(f"  Loaded {len(chunk_to_doc):,} chunk mappings", flush=True)

    # ------------------------------------------------------------------
    # 3. Open indexes
    # ------------------------------------------------------------------
    tantivy_index = None
    lancedb_index = None
    embedding = None

    if "tantivy" in args.backends or "hybrid" in args.backends:
        print(f"  Opening Tantivy index: {args.tantivy}", flush=True)
        tantivy_index = TantivyIndex(args.tantivy)
        print(f"  Tantivy: {tantivy_index.count():,} chunks indexed", flush=True)

    if "lancedb" in args.backends or "hybrid" in args.backends:
        print(f"  Opening LanceDB index: {args.lancedb}", flush=True)
        lancedb_index = LanceDBIndex(args.lancedb)
        print(f"  LanceDB: {lancedb_index.count():,} vectors indexed", flush=True)
        print(f"  Loading embedding model...", flush=True)
        embedding = EmbeddingAdapter(show_progress=False)

    # ------------------------------------------------------------------
    # 4. Run evaluation per backend
    # ------------------------------------------------------------------
    report = {
        "output": args.output,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_queries": len(queries),
        "seed": args.seed,
        "k": args.k,
        "corpus_chunks": len(chunk_to_doc),
        "backends": {},
    }

    for backend_name in args.backends:
        print(f"\n=== {backend_name} ===", flush=True)
        per_query = []

        for i, q in enumerate(queries):
            t0 = time.monotonic()

            if backend_name == "tantivy":
                hits = tantivy_index.search(q.query_text, limit=args.k)
            elif backend_name == "lancedb":
                vec = embedding.embed_query(q.query_text)
                hits = lancedb_index.search(vec, limit=args.k)
            elif backend_name == "hybrid":
                lex_hits = tantivy_index.search(q.query_text, limit=args.k)
                vec = embedding.embed_query(q.query_text)
                vec_hits = lancedb_index.search(vec, limit=args.k)
                hits = hybrid_fuse(lex_hits, vec_hits, limit=args.k)
            else:
                raise ValueError(f"Unknown backend: {backend_name}")

            latency_ms = (time.monotonic() - t0) * 1000

            ranked_chunk_ids = [h.chunk_id for h in hits]
            ranked_doc_ids = [chunk_to_doc.get(h.chunk_id, "") for h in hits]

            per_query.append({
                "ranked_chunk_ids": ranked_chunk_ids,
                "ranked_doc_ids": ranked_doc_ids,
                "relevant_chunk_id": q.relevant_chunk_id,
                "relevant_doc_id": q.relevant_document_id,
                "latency_ms": latency_ms,
            })

            if (i + 1) % 50 == 0:
                avg_lat = sum(pq["latency_ms"] for pq in per_query[-50:]) / min(50, len(per_query))
                print(f"  {backend_name}: {i+1}/{len(queries)} done, avg_latency={avg_lat:.1f}ms", flush=True)

        metrics = compute_metrics_from_hits(backend_name, per_query)
        print(f"  {backend_name} results:", flush=True)
        print(f"    recall@1={metrics.recall_at_1:.4f}  recall@5={metrics.recall_at_5:.4f}  "
              f"recall@10={metrics.recall_at_10:.4f}  recall@20={metrics.recall_at_20:.4f}", flush=True)
        print(f"    doc_recall@10={metrics.doc_recall_at_10:.4f}  doc_recall@20={metrics.doc_recall_at_20:.4f}", flush=True)
        print(f"    MRR={metrics.mrr:.4f}  nDCG@10={metrics.ndcg_at_10:.4f}", flush=True)
        print(f"    p50={metrics.p50_latency_ms:.1f}ms  p95={metrics.p95_latency_ms:.1f}ms  "
              f"avg={metrics.avg_latency_ms:.1f}ms", flush=True)

        report["backends"][backend_name] = metrics.to_dict()

    # ------------------------------------------------------------------
    # 5. Save report
    # ------------------------------------------------------------------
    report_path = out / "benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport: {report_path}", flush=True)

    # Cleanup
    if tantivy_index:
        tantivy_index.close()
    if lancedb_index:
        lancedb_index.close()
    if embedding:
        embedding.close()


if __name__ == "__main__":
    main()
