"""Run E9 â€” Selective enrichment experiment on 2000 chunks.

Full pipeline:
  Phase 1: Sample 2000 chunks (excluding pure noise)
  Phase 2: Generate 2000 natural language queries with LLM (3 workers)
  Phase 3: Enrich 2000 chunks Ã— 3 strategies = 6000 LLM calls (3 workers each)
  Phase 4: Build 4 Tantivy indexes (baseline + 3 enriched)
  Phase 5: Build 4 LanceDB indexes (baseline + 3 enriched)
  Phase 6: Evaluate recall@k of NL queries against all 4 indexes
  Phase 7: Comparative report

Total LLM inferences: 8000 (2000 NL queries + 6000 enrichment)
Estimated time: ~49 min with 3 workers on GPU

Usage:
    python scripts/benchmarks/run_enrichment_experiment.py
        --store outputs/experiments/E1-corpus/document_store.db
        --output outputs/experiments/E9-experiment
        --n-chunks 2000
        --model qwen3.5:4b-q4_K_M
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
})


def lexical_density(text: str) -> float:
    """Compute lexical density (unique content words / total content words)."""
    tokens = re.findall(r'[a-zA-Z]{2,}', text.lower())
    content = [t for t in tokens if t not in STOPWORDS]
    if not content:
        return 0.0
    return len(set(content)) / len(content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E9 enrichment experiment.")
    parser.add_argument("--store", required=True, help="Path to document_store.db")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--n-chunks", type=int, default=2000, help="Number of chunks to sample")
    parser.add_argument("--model", default="qwen3.5:4b-q4_K_M", help="Ollama model name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--workers", type=int, default=3, help="Concurrent LLM workers")
    args = parser.parse_args()

    from ipa.enrichment.ollama_adapter import OllamaAdapter
    from ipa.enrichment.enrichment import STRATEGIES, generate_nl_query
    from ipa.contracts import DocumentChunk, SourceSpan
    from ipa import TantivyIndex, LanceDBIndex, EmbeddingAdapter
    from ipa.agentic.retrieval_eval import (
        recall_at_k, document_recall_at_k, reciprocal_rank, ndcg_at_k,
    )

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    # ==================================================================
    # PHASE 1: Sample chunks
    # ==================================================================
    print(f"\n{'='*70}", flush=True)
    print(f"E9: Selective enrichment experiment", flush=True)
    print(f"{'='*70}", flush=True)

    print(f"\n--- Phase 1: Sample {args.n_chunks} chunks ---", flush=True)
    t0 = time.monotonic()
    conn = sqlite3.connect(args.store)
    all_rows = conn.execute(
        "SELECT chunk_id, document_id, text FROM chunks WHERE length(text) >= 50"
    ).fetchall()
    conn.close()
    print(f"  Total chunks with >=50 chars: {len(all_rows):,}", flush=True)

    # Filter out pure noise (lexical_density == 0 or < 3 content words)
    filtered = []
    for chunk_id, doc_id, text in all_rows:
        ld = lexical_density(text)
        if ld > 0.1:  # exclude pure noise
            filtered.append((chunk_id, doc_id, text, ld))
    print(f"  After filtering pure noise (ld>0.1): {len(filtered):,}", flush=True)

    # Sample uniformly across density tiers
    # Tiers: low (<0.6), medium (0.6-0.8), high (>0.8)
    tiers = {"low": [], "medium": [], "high": []}
    for row in filtered:
        ld = row[3]
        if ld < 0.6:
            tiers["low"].append(row)
        elif ld < 0.8:
            tiers["medium"].append(row)
        else:
            tiers["high"].append(row)

    print(f"  Tier distribution: low={len(tiers['low']):,}, "
          f"medium={len(tiers['medium']):,}, high={len(tiers['high']):,}", flush=True)

    # Sample proportionally, but ensure at least 100 from each tier
    n = args.n_chunks
    total_filtered = len(filtered)
    samples = []
    for tier_name, tier_chunks in tiers.items():
        proportion = len(tier_chunks) / total_filtered
        n_tier = max(100, int(n * proportion))
        n_tier = min(n_tier, len(tier_chunks))
        rng.shuffle(tier_chunks)
        samples.extend(tier_chunks[:n_tier])

    # Trim to exact n
    rng.shuffle(samples)
    samples = samples[:n]
    print(f"  Sampled {len(samples)} chunks", flush=True)

    tier_counts = {"low": 0, "medium": 0, "high": 0}
    for row in samples:
        ld = row[3]
        if ld < 0.6:
            tier_counts["low"] += 1
        elif ld < 0.8:
            tier_counts["medium"] += 1
        else:
            tier_counts["high"] += 1
    print(f"  Sample tiers: low={tier_counts['low']}, "
          f"medium={tier_counts['medium']}, high={tier_counts['high']}", flush=True)
    print(f"  Phase 1 done in {time.monotonic()-t0:.1f}s", flush=True)

    # ==================================================================
    # PHASE 2: Generate natural language queries
    # ==================================================================
    print(f"\n--- Phase 2: Generate {len(samples)} NL queries (3 workers) ---", flush=True)
    t0 = time.monotonic()
    nl_queries = {}  # chunk_id -> query_text
    completed = 0
    errors = 0

    def gen_query_worker(chunk_id, text):
        llm = OllamaAdapter(model=args.model, think=False, temperature=0.7)
        return chunk_id, generate_nl_query(text, llm)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(gen_query_worker, r[0], r[2]): r[0] for r in samples}
        for future in as_completed(futures):
            chunk_id, (query, latency) = future.result()
            nl_queries[chunk_id] = query
            completed += 1
            if not query:
                errors += 1
            if completed % 100 == 0:
                elapsed = time.monotonic() - t0
                avg = elapsed / completed
                remaining = avg * (len(samples) - completed)
                print(f"  NL queries: {completed}/{len(samples)} done, "
                      f"errors={errors}, avg={avg:.2f}s, ETA={remaining:.0f}s", flush=True)

    print(f"  Phase 2 done: {len(nl_queries)} queries in {time.monotonic()-t0:.1f}s, "
          f"errors={errors}", flush=True)

    # Show sample queries
    print(f"\n  Sample NL queries:", flush=True)
    for i, (chunk_id, doc_id, text, ld) in enumerate(samples[:3]):
        q = nl_queries.get(chunk_id, "?")
        # Sanitize for console output (strip non-ASCII)
        q_safe = q[:80].encode("ascii", "replace").decode()
        text_safe = text[:80].encode("ascii", "replace").decode()
        print(f"    [{i}] ld={ld:.2f} | query: {q_safe}", flush=True)
        print(f"         chunk: {text_safe}...", flush=True)

    # ==================================================================
    # PHASE 3: Enrich chunks with 3 strategies
    # ==================================================================
    enriched_data = {}  # strategy -> {chunk_id -> enriched_text}

    for strategy_name in ["synthetic_queries", "claim_extraction", "summary"]:
        strategy_fn = STRATEGIES[strategy_name]
        print(f"\n--- Phase 3.{['synthetic_queries','claim_extraction','summary'].index(strategy_name)+1}: "
              f"Enrich with {strategy_name} (3 workers) ---", flush=True)
        t0 = time.monotonic()
        results = {}
        completed = 0
        strat_errors = 0

        def enrich_worker(chunk_id, text):
            llm = OllamaAdapter(model=args.model, think=False, temperature=0.3)
            return strategy_fn(chunk_id, text, llm)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(enrich_worker, r[0], r[2]): r[0] for r in samples}
            for future in as_completed(futures):
                result = future.result()
                results[result.chunk_id] = result.enriched_text
                completed += 1
                if result.error:
                    strat_errors += 1
                if completed % 100 == 0:
                    elapsed = time.monotonic() - t0
                    avg = elapsed / completed
                    remaining = avg * (len(samples) - completed)
                    print(f"  {strategy_name}: {completed}/{len(samples)} done, "
                          f"errors={strat_errors}, avg={avg:.2f}s, ETA={remaining:.0f}s", flush=True)

        enriched_data[strategy_name] = results
        print(f"  {strategy_name} done: {len(results)} chunks in {time.monotonic()-t0:.1f}s, "
              f"errors={strat_errors}", flush=True)

    # ==================================================================
    # PHASE 4: Build Tantivy indexes (baseline + 3 enriched)
    # ==================================================================
    print(f"\n--- Phase 4: Build 4 Tantivy indexes ---", flush=True)
    t0 = time.monotonic()

    def make_chunk(chunk_id, doc_id, text, idx):
        return DocumentChunk(
            chunk_id=chunk_id,
            document_id=doc_id,
            content_hash=f"sha256:exp_{idx}",
            text=text,
            metadata={"chunk_index": idx},
            source_span=SourceSpan(
                artifact_id="sha256:experiment",
                page=1,
                offset_start=0,
                offset_end=len(text),
            ),
        )

    indexes_to_build = {
        "baseline": {r[0]: r[2] for r in samples},  # original text
    }
    for strat, enriched in enriched_data.items():
        indexes_to_build[f"enriched_{strat}"] = enriched

    tantivy_indexes = {}
    for name, text_map in indexes_to_build.items():
        idx_path = out / f"tantivy_{name}"
        if idx_path.exists():
            import shutil
            shutil.rmtree(idx_path)
        idx = TantivyIndex(idx_path)
        chunks = []
        for i, (chunk_id, doc_id, text, ld) in enumerate(samples):
            enriched_text = text_map.get(chunk_id, text)
            chunks.append(make_chunk(chunk_id, doc_id, enriched_text, i))
        idx.add_chunks(chunks)
        tantivy_indexes[name] = idx
        print(f"  Tantivy {name}: {idx.count()} chunks indexed", flush=True)

    print(f"  Phase 4 done in {time.monotonic()-t0:.1f}s", flush=True)

    # ==================================================================
    # PHASE 5: Build LanceDB indexes (baseline + 3 enriched)
    # ==================================================================
    print(f"\n--- Phase 5: Build 4 LanceDB indexes ---", flush=True)
    t0 = time.monotonic()
    print(f"  Loading embedding model...", flush=True)
    emb = EmbeddingAdapter(show_progress=False)

    lancedb_indexes = {}
    for name, text_map in indexes_to_build.items():
        idx_path = out / f"lancedb_{name}"
        if idx_path.exists():
            import shutil
            shutil.rmtree(idx_path)
        idx = LanceDBIndex(idx_path)
        chunks = []
        texts = []
        for i, (chunk_id, doc_id, text, ld) in enumerate(samples):
            enriched_text = text_map.get(chunk_id, text)
            chunks.append(make_chunk(chunk_id, doc_id, enriched_text, i))
            texts.append(enriched_text)
        vectors = emb.embed_texts(texts)
        idx.add_chunks(chunks, vectors)
        lancedb_indexes[name] = idx
        print(f"  LanceDB {name}: {idx.count()} vectors indexed", flush=True)

    print(f"  Phase 5 done in {time.monotonic()-t0:.1f}s", flush=True)

    # ==================================================================
    # PHASE 6: Evaluate recall@k of NL queries
    # ==================================================================
    print(f"\n--- Phase 6: Evaluate NL queries against all indexes ---", flush=True)
    t0 = time.monotonic()

    # Build ground truth: chunk_id -> relevant_chunk_id (self)
    ground_truth = {r[0]: r[0] for r in samples}
    chunk_to_doc = {r[0]: r[1] for r in samples}

    report = {
        "output": args.output,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": args.model,
        "n_chunks": len(samples),
        "n_queries": len(nl_queries),
        "seed": args.seed,
        "tier_counts": tier_counts,
        "results": {},
    }

    k_values = [1, 5, 10, 20]
    all_index_names = list(tantivy_indexes.keys())

    for idx_name in all_index_names:
        print(f"\n  Evaluating: {idx_name} (Tantivy)", flush=True)
        t_idx = time.monotonic()
        per_query = []

        for i, (chunk_id, query_text) in enumerate(nl_queries.items()):
            if not query_text:
                continue
            t_q0 = time.monotonic()

            # Tantivy search
            lex_hits = tantivy_indexes[idx_name].search(query_text, limit=20)
            ranked_chunks = [h.chunk_id for h in lex_hits]
            ranked_docs = [chunk_to_doc.get(h.chunk_id, "") for h in lex_hits]
            latency_ms = (time.monotonic() - t_q0) * 1000

            per_query.append({
                "ranked_chunk_ids": ranked_chunks,
                "ranked_doc_ids": ranked_docs,
                "relevant_chunk_id": chunk_id,
                "relevant_doc_id": chunk_to_doc[chunk_id],
                "latency_ms": latency_ms,
            })

            if (i + 1) % 500 == 0:
                print(f"    {idx_name}: {i+1}/{len(nl_queries)} queries", flush=True)

        # Compute metrics
        n_q = len(per_query)
        metrics = {}
        for k in k_values:
            r = sum(recall_at_k(pq["ranked_chunk_ids"], pq["relevant_chunk_id"], k) for pq in per_query) / n_q
            dr = sum(document_recall_at_k(pq["ranked_doc_ids"], pq["relevant_doc_id"], k) for pq in per_query) / n_q
            metrics[f"recall@{k}"] = round(r, 4)
            metrics[f"doc_recall@{k}"] = round(dr, 4)
        mrr = sum(reciprocal_rank(pq["ranked_chunk_ids"], pq["relevant_chunk_id"]) for pq in per_query) / n_q
        ndcg = sum(ndcg_at_k(pq["ranked_chunk_ids"], pq["relevant_chunk_id"], 10) for pq in per_query) / n_q
        latencies = sorted(pq["latency_ms"] for pq in per_query)
        metrics["mrr"] = round(mrr, 4)
        metrics["ndcg@10"] = round(ndcg, 4)
        metrics["p50_ms"] = round(latencies[n_q // 2], 2)
        metrics["p95_ms"] = round(latencies[int(n_q * 0.95)], 2)
        metrics["n_queries"] = n_q

        print(f"    {idx_name} results:", flush=True)
        print(f"      recall@1={metrics['recall@1']}  recall@5={metrics['recall@5']}  "
              f"recall@10={metrics['recall@10']}  recall@20={metrics['recall@20']}", flush=True)
        print(f"      doc_recall@10={metrics['doc_recall@10']}  MRR={metrics['mrr']}  "
              f"nDCG@10={metrics['ndcg@10']}", flush=True)
        print(f"      p50={metrics['p50_ms']}ms  p95={metrics['p95_ms']}ms", flush=True)

        report["results"][f"tantivy_{idx_name}"] = metrics
        print(f"    Done in {time.monotonic()-t_idx:.1f}s", flush=True)

    # Also evaluate LanceDB for each index
    for idx_name in all_index_names:
        print(f"\n  Evaluating: {idx_name} (LanceDB)", flush=True)
        t_idx = time.monotonic()
        per_query = []

        for i, (chunk_id, query_text) in enumerate(nl_queries.items()):
            if not query_text:
                continue
            t_q0 = time.monotonic()

            vec = emb.embed_query(query_text)
            vec_hits = lancedb_indexes[idx_name].search(vec, limit=20)
            ranked_chunks = [h.chunk_id for h in vec_hits]
            ranked_docs = [chunk_to_doc.get(h.chunk_id, "") for h in vec_hits]
            latency_ms = (time.monotonic() - t_q0) * 1000

            per_query.append({
                "ranked_chunk_ids": ranked_chunks,
                "ranked_doc_ids": ranked_docs,
                "relevant_chunk_id": chunk_id,
                "relevant_doc_id": chunk_to_doc[chunk_id],
                "latency_ms": latency_ms,
            })

            if (i + 1) % 500 == 0:
                print(f"    {idx_name}: {i+1}/{len(nl_queries)} queries", flush=True)

        n_q = len(per_query)
        metrics = {}
        for k in k_values:
            r = sum(recall_at_k(pq["ranked_chunk_ids"], pq["relevant_chunk_id"], k) for pq in per_query) / n_q
            dr = sum(document_recall_at_k(pq["ranked_doc_ids"], pq["relevant_doc_id"], k) for pq in per_query) / n_q
            metrics[f"recall@{k}"] = round(r, 4)
            metrics[f"doc_recall@{k}"] = round(dr, 4)
        mrr = sum(reciprocal_rank(pq["ranked_chunk_ids"], pq["relevant_chunk_id"]) for pq in per_query) / n_q
        ndcg = sum(ndcg_at_k(pq["ranked_chunk_ids"], pq["relevant_chunk_id"], 10) for pq in per_query) / n_q
        latencies = sorted(pq["latency_ms"] for pq in per_query)
        metrics["mrr"] = round(mrr, 4)
        metrics["ndcg@10"] = round(ndcg, 4)
        metrics["p50_ms"] = round(latencies[n_q // 2], 2)
        metrics["p95_ms"] = round(latencies[int(n_q * 0.95)], 2)
        metrics["n_queries"] = n_q

        print(f"    {idx_name} results:", flush=True)
        print(f"      recall@1={metrics['recall@1']}  recall@5={metrics['recall@5']}  "
              f"recall@10={metrics['recall@10']}  recall@20={metrics['recall@20']}", flush=True)
        print(f"      doc_recall@10={metrics['doc_recall@10']}  MRR={metrics['mrr']}  "
              f"nDCG@10={metrics['ndcg@10']}", flush=True)
        print(f"      p50={metrics['p50_ms']}ms  p95={metrics['p95_ms']}ms", flush=True)

        report["results"][f"lancedb_{idx_name}"] = metrics
        print(f"    Done in {time.monotonic()-t_idx:.1f}s", flush=True)

    print(f"\n  Phase 6 done in {time.monotonic()-t0:.1f}s", flush=True)

    # ==================================================================
    # PHASE 7: Save report
    # ==================================================================
    report_path = out / "benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{'='*70}", flush=True)
    print(f"Report: {report_path}", flush=True)
    print(f"{'='*70}", flush=True)

    # Cleanup
    for idx in tantivy_indexes.values():
        idx.close()
    for idx in lancedb_indexes.values():
        idx.close()
    emb.close()


if __name__ == "__main__":
    main()
