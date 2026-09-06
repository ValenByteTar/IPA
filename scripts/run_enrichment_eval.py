"""Run E9 — Semantic enrichment with 3 strategies in parallel.

Processes the first N chunks from the corpus with 3 enrichment strategies
(synthetic queries, claim extraction, summary) using 3 parallel workers.
Saves enriched chunks for later retrieval evaluation.

Usage:
    python scripts/run_enrichment_eval.py
        --store outputs/experiments/E1-corpus/document_store.db
        --output outputs/experiments/E9
        --n-chunks 100
        --model qwen3.5:4b-q4_K_M
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E9 enrichment evaluation.")
    parser.add_argument("--store", required=True, help="Path to document_store.db")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--n-chunks", type=int, default=100, help="Number of chunks to enrich")
    parser.add_argument("--model", default="qwen3.5:4b-q4_K_M", help="Ollama model name")
    parser.add_argument("--strategies", nargs="+",
                        default=["synthetic_queries", "claim_extraction", "summary"],
                        help="Enrichment strategies to run")
    parser.add_argument("--min-chunk-len", type=int, default=100,
                        help="Skip chunks shorter than this")
    args = parser.parse_args()

    from ipa.ollama_adapter import OllamaAdapter
    from ipa.enrichment import STRATEGIES

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load chunks from store
    # ------------------------------------------------------------------
    print(f"\nE9: Semantic enrichment evaluation", flush=True)
    print(f"  Loading {args.n_chunks} chunks from {args.store}...", flush=True)
    conn = sqlite3.connect(args.store)
    rows = conn.execute(
        "SELECT chunk_id, document_id, text FROM chunks "
        "WHERE length(text) >= ? "
        "ORDER BY rowid LIMIT ?",
        (args.min_chunk_len, args.n_chunks),
    ).fetchall()
    conn.close()
    print(f"  Loaded {len(rows)} chunks", flush=True)

    # ------------------------------------------------------------------
    # 2. Verify Ollama is available
    # ------------------------------------------------------------------
    print(f"  Checking Ollama availability (model={args.model})...", flush=True)
    llm = OllamaAdapter(model=args.model, think=False, temperature=0.3)
    if not llm.is_available():
        print(f"  ERROR: Ollama not running or model {args.model} not found", flush=True)
        sys.exit(1)
    print(f"  Ollama OK", flush=True)

    # Warm up the model with a trivial call.
    print(f"  Warming up model...", flush=True)
    warm = llm.generate("Say OK.", system="Respond with a single word.")
    print(f"  Warmup: {warm.text[:20]} in {warm.latency_seconds:.2f}s", flush=True)

    # ------------------------------------------------------------------
    # 3. Run enrichment with parallel workers
    # ------------------------------------------------------------------
    report = {
        "output": args.output,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": args.model,
        "n_chunks": len(rows),
        "strategies": {},
    }

    for strategy_name in args.strategies:
        strategy_fn = STRATEGIES[strategy_name]
        print(f"\n=== {strategy_name} ===", flush=True)

        results = []
        t0 = time.monotonic()
        completed = 0

        # Each worker gets its own OllamaAdapter (they're stateless HTTP clients).
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {}
            for chunk_id, doc_id, text in rows:
                worker_llm = OllamaAdapter(model=args.model, think=False, temperature=0.3)
                future = pool.submit(strategy_fn, chunk_id, text, worker_llm)
                futures[future] = (chunk_id, doc_id)

            for future in as_completed(futures):
                chunk_id, doc_id = futures[future]
                result = future.result()
                results.append({
                    "chunk_id": result.chunk_id,
                    "document_id": doc_id,
                    "strategy": result.strategy,
                    "enriched_text": result.enriched_text,
                    "llm_output": result.llm_output,
                    "latency_seconds": round(result.latency_seconds, 3),
                    "error": result.error,
                })
                completed += 1
                if completed % 10 == 0:
                    elapsed = time.monotonic() - t0
                    avg = elapsed / completed
                    remaining = avg * (len(rows) - completed)
                    print(f"  {strategy_name}: {completed}/{len(rows)} done, "
                          f"avg={avg:.2f}s/chunk, ETA={remaining:.0f}s", flush=True)

        elapsed = time.monotonic() - t0
        errors = sum(1 for r in results if r["error"])
        latencies = [r["latency_seconds"] for r in results if r["error"] is None]
        avg_lat = sum(latencies) / len(latencies) if latencies else 0
        enriched_sizes = [len(r["enriched_text"]) for r in results]
        avg_enriched = sum(enriched_sizes) / len(enriched_sizes) if enriched_sizes else 0

        print(f"  {strategy_name} done: {len(results)} chunks in {elapsed:.1f}s, "
              f"avg_latency={avg_lat:.2f}s, errors={errors}, "
              f"avg_enriched_size={avg_enriched:.0f} chars", flush=True)

        # Save enriched chunks to JSON.
        out_file = out / f"enriched_{strategy_name}.json"
        out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  Saved: {out_file}", flush=True)

        report["strategies"][strategy_name] = {
            "n_chunks": len(results),
            "errors": errors,
            "elapsed_seconds": round(elapsed, 2),
            "avg_latency_seconds": round(avg_lat, 3),
            "avg_enriched_size": round(avg_enriched, 1),
        }

    # ------------------------------------------------------------------
    # 4. Save report
    # ------------------------------------------------------------------
    report_path = out / "benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport: {report_path}", flush=True)

    llm.close()


if __name__ == "__main__":
    main()
