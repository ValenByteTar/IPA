"""Optimized batched enrichment for future pipeline runs.

Two optimizations over the sequential/parallel approach:
  1. Prompt batching: send N chunks in 1 Ollama request (1 GPU forward pass
     for N summaries instead of N separate passes)
  2. Pipeline with queue: producer reads from DB, batcher groups prompts,
     workers send batched requests concurrently, consumer writes to DB

Architecture:
  [DB reader] → [batch queue] → [3 batch workers] → [result queue] → [DB writer]

  - DB reader: streams chunks from SQLite (resumable, skips [Summary])
  - Batcher: groups BATCH_SIZE chunks into 1 combined prompt
  - Workers: 3 concurrent batch requests to Ollama (GPU saturated)
  - DB writer: serializes SQLite writes, commits every BATCH_SIZE chunks

Throughput expectation:
  - Sequential (1W):     0.27 chunks/s  (3.68s/chunk)
  - Parallel (3W):       0.63 chunks/s  (1.58s/chunk)  ← current
  - Batched (3W, B=5):  ~1.5-2.0 chunks/s (0.5-0.7s/chunk)  ← this module

  The batched approach does 1 forward pass for 5 chunks instead of 5
  separate passes. Even though the combined prompt is longer (5×1500 chars),
  a single forward pass of 7500 tokens is faster than 5 passes of 1500
  because GPU utilization is higher and HTTP roundtrips are 5× fewer.

Usage:
  python scripts/_run_enrichment_batched.py [--batch-size 5] [--workers 3]

  # Future: import as module
  from run_enrichment_batched import run_batched_enrichment
  run_batched_enrichment(store_db, batch_size=5, workers=3)
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Any

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

from ipa.contracts import DocumentChunk
from ipa.ollama_adapter import OllamaAdapter
from run_continuous_pipeline import should_summarize

# Thread-safe SQLite writes
_write_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Batched prompt construction and parsing
# ---------------------------------------------------------------------------

_BATCH_SYSTEM = (
    "You are a summarization system. Summarize each text concisely in 1-2 sentences. "
    "Output one summary per text, prefixed with [N] where N is the text number. "
    "No preamble, no commentary, no extra text."
)

_BATCH_PROMPT_TEMPLATE = """Summarize each of the following {count} texts. Output one summary per line, prefixed with [N].

{sections}

Output format:
[1] first summary
[2] second summary
[{count}] last summary"""

_SECTION_TEMPLATE = "--- TEXT {n} ---\n{text}"


def build_batched_prompt(texts: list[str]) -> str:
    """Build a single prompt containing multiple texts to summarize.

    Each text is truncated to 1500 chars (same as single-chunk enrichment)
    to keep the combined prompt within BGE-M3's 8K token context.
    """
    sections = []
    for i, text in enumerate(texts, 1):
        truncated = text[:1500]
        sections.append(_SECTION_TEMPLATE.format(n=i, text=truncated))
    return _BATCH_PROMPT_TEMPLATE.format(
        count=len(texts),
        sections="\n\n".join(sections),
    )


def parse_batched_response(response: str, expected_count: int) -> list[str | None]:
    """Parse batched LLM response into individual summaries.

    Expected format:
        [1] summary text
        [2] summary text
        [3] summary text

    Returns list of summaries (or None if parsing failed for that index).
    Robust against:
        - Missing numbers
        - Extra whitespace
        - Multi-line summaries (joins continuation lines)
        - Model adding preamble before [1]
    """
    summaries: list[str | None] = [None] * expected_count

    # Pattern: [N] followed by summary text (may span multiple lines
    # until the next [N] or end of text)
    pattern = re.compile(r"\[(\d+)\]\s*(.+?)(?=\[\d+\]|$)", re.DOTALL)

    for match in pattern.finditer(response):
        idx = int(match.group(1)) - 1  # 0-based
        text = match.group(2).strip()
        # Clean up: collapse whitespace, remove trailing newlines
        text = " ".join(text.split())
        if 0 <= idx < expected_count and text:
            summaries[idx] = text

    return summaries


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BatchItem:
    """A batch of chunks to process in a single LLM call."""
    chunks: list[tuple[str, str, str, str]]  # (chunk_id, doc_id, text, content_hash)
    batch_idx: int


@dataclass
class BatchResult:
    """Result of processing a batch."""
    batch_idx: int
    summaries: list[str | None]  # One per chunk, None if failed
    latency: float
    error: str | None = None


# ---------------------------------------------------------------------------
# Batched enrichment core
# ---------------------------------------------------------------------------

def enrich_batch(
    batch: BatchItem,
    model: str = "qwen3.5:4b-q4_K_M",
) -> BatchResult:
    """Send a batch of chunks to Ollama and parse the response.

    Creates a temporary OllamaAdapter (thread-safe: each thread gets its own).
    """
    texts = [text for _, _, text, _ in batch.chunks]
    prompt = build_batched_prompt(texts)

    try:
        llm = OllamaAdapter(model=model, think=False, temperature=0.3)
        resp = llm.generate(prompt, system=_BATCH_SYSTEM)
        llm.close()

        summaries = parse_batched_response(resp.text, len(texts))

        # Check if any summaries failed to parse
        failed = sum(1 for s in summaries if s is None)
        error = f"{failed}/{len(texts)} summaries failed to parse" if failed > 0 else None

        return BatchResult(
            batch_idx=batch.batch_idx,
            summaries=summaries,
            latency=resp.latency_seconds,
            error=error,
        )
    except Exception as e:
        return BatchResult(
            batch_idx=batch.batch_idx,
            summaries=[None] * len(texts),
            latency=0.0,
            error=str(e),
        )


def run_batched_enrichment(
    store_db: Path,
    batch_size: int = 5,
    workers: int = 3,
    model: str = "qwen3.5:4b-q4_K_M",
    min_chars: int = 800,
    commit_every: int = 50,
) -> dict[str, Any]:
    """Run batched enrichment on all qualifying chunks.

    Args:
        store_db: Path to document_store.db
        batch_size: Number of chunks per LLM request (higher = fewer requests
            but longer prompts; 5 is a good default for 1500-char truncation)
        workers: Concurrent batch requests to Ollama (3 is optimal for 6GB GPU)
        model: Ollama model name
        min_chars: Minimum chunk length to qualify for enrichment
        commit_every: Commit to SQLite every N enriched chunks

    Returns:
        Dict with statistics: total, enriched, errors, time, rate
    """
    conn = sqlite3.connect(str(store_db))
    conn.execute("PRAGMA journal_mode=WAL")
    c = conn.cursor()

    # Load chunks to process
    c.execute("SELECT chunk_id, document_id, text, content_hash FROM chunks")
    rows = c.fetchall()
    total = len(rows)

    to_process = []
    already_enriched = 0
    skipped = 0

    for chunk_id, doc_id, text, content_hash in rows:
        if text.startswith("[Summary]"):
            already_enriched += 1
            continue
        chunk = DocumentChunk(
            chunk_id=chunk_id, document_id=doc_id,
            text=text, content_hash=content_hash,
        )
        if should_summarize(chunk, min_chars=min_chars):
            to_process.append((chunk_id, doc_id, text, content_hash))
        else:
            skipped += 1

    print(f"Total chunks: {total}")
    print(f"Already enriched (skipping): {already_enriched}")
    print(f"Skipped (short or dense): {skipped}")
    print(f"To process: {len(to_process)}")
    print(f"Batch size: {batch_size}")
    print(f"Workers: {workers}")
    print(f"Model: {model}")
    print(f"Expected requests: {(len(to_process) + batch_size - 1) // batch_size}")
    print()

    if not to_process:
        print("No chunks need enrichment. Exiting.")
        conn.close()
        return {"total": total, "enriched": 0, "errors": 0, "time": 0}

    # Build batches
    batches: list[BatchItem] = []
    for i in range(0, len(to_process), batch_size):
        batch_chunks = to_process[i:i + batch_size]
        batches.append(BatchItem(chunks=batch_chunks, batch_idx=len(batches)))

    print(f"Batches created: {len(batches)}")
    print()

    # Test Ollama connection
    print("Testing Ollama connection...")
    test_llm = OllamaAdapter(model=model, think=False, temperature=0.3)
    try:
        test = test_llm.generate("Say OK", system="Reply with only OK.")
        print(f"Ollama test: '{test.text.strip()}' ({test.latency_seconds:.1f}s)")
    except Exception as e:
        print(f"Ollama connection failed: {e}")
        conn.close()
        return {"total": total, "enriched": 0, "errors": 0, "time": 0}
    test_llm.close()

    # Process batches with thread pool
    enriched = 0
    errors = 0
    parse_failures = 0
    fallback_enriched = 0
    t0 = time.monotonic()
    completed_batches = 0
    total_batches = len(batches)

    # Collect chunks that fail batched parsing for fallback
    fallback_chunks: list[tuple[str, str, str, str]] = []
    fallback_lock = threading.Lock()

    print(f"Starting batched enrichment with {workers} workers...")
    print()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit all batch tasks
        futures = {
            executor.submit(enrich_batch, batch, model): batch
            for batch in batches
        }

        # Process results as they complete
        for future in as_completed(futures):
            batch = futures[future]
            completed_batches += 1

            try:
                result: BatchResult = future.result()
            except Exception as e:
                errors += 1
                if errors <= 3:
                    print(f"  EXCEPTION batch {batch.batch_idx}: {e}")
                # All chunks in this batch go to fallback
                with fallback_lock:
                    fallback_chunks.extend(batch.chunks)
                continue

            # Write summaries to DB, collect failures for fallback
            batch_failures: list[tuple[str, str, str, str]] = []
            for i, (chunk_id, doc_id, text, content_hash) in enumerate(batch.chunks):
                summary = result.summaries[i] if i < len(result.summaries) else None

                if summary is None:
                    parse_failures += 1
                    batch_failures.append((chunk_id, doc_id, text, content_hash))
                    continue

                enriched_text = f"[Summary] {summary}\n\n{text}"
                with _write_lock:
                    c.execute(
                        "UPDATE chunks SET text = ? WHERE chunk_id = ?",
                        (enriched_text, chunk_id),
                    )
                    enriched += 1

                    if enriched % commit_every == 0:
                        conn.commit()

            # Add failures to fallback queue
            if batch_failures:
                with fallback_lock:
                    fallback_chunks.extend(batch_failures)

            # Progress report
            if completed_batches % 10 == 0 or completed_batches == total_batches:
                elapsed = time.monotonic() - t0
                rate = enriched / elapsed if elapsed > 0 else 0
                remaining = len(to_process) - enriched - parse_failures
                eta = remaining / rate if rate > 0 else 0
                print(
                    f"  [{completed_batches}/{total_batches}] batches | "
                    f"enriched={enriched} parse_fail={parse_failures} errors={errors} | "
                    f"{elapsed:.0f}s | {rate:.2f} chunks/s | "
                    f"ETA: {eta:.0f}s ({eta/60:.0f} min) | "
                    f"fallback_queue={len(fallback_chunks)}"
                )

    # Phase 2: Fallback for chunks that failed batched parsing
    # Process them individually (batch_size=1) — slower but reliable
    if fallback_chunks:
        print(f"\n{'='*60}")
        print(f"Phase 2: Fallback enrichment for {len(fallback_chunks)} chunks")
        print(f"(batch_size=1, workers={workers})")
        print(f"{'='*60}")

        from ipa.enrichment import enrich_summary

        # Thread-local LLM for fallback
        _fallback_local = threading.local()

        def get_fallback_llm():
            if not hasattr(_fallback_local, "llm"):
                _fallback_local.llm = OllamaAdapter(model=model, think=False, temperature=0.3)
            return _fallback_local.llm

        def enrich_single(chunk_id, doc_id, text, content_hash):
            llm = get_fallback_llm()
            result = enrich_summary(chunk_id, text, llm)
            return chunk_id, result.enriched_text, result.error

        t_fb = time.monotonic()
        fb_completed = 0
        fb_errors = 0

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(enrich_single, cid, did, text, chash): cid
                for cid, did, text, chash in fallback_chunks
            }

            for future in as_completed(futures):
                chunk_id, enriched_text, error = future.result()
                fb_completed += 1

                if error:
                    fb_errors += 1
                    if fb_errors <= 3:
                        print(f"  FALLBACK ERROR {chunk_id[:16]}...: {error}")
                    continue

                with _write_lock:
                    c.execute(
                        "UPDATE chunks SET text = ? WHERE chunk_id = ?",
                        (enriched_text, chunk_id),
                    )
                    enriched += 1
                    fallback_enriched += 1

                    if enriched % commit_every == 0:
                        conn.commit()

                if fb_completed % 20 == 0:
                    elapsed_fb = time.monotonic() - t_fb
                    rate_fb = fb_completed / elapsed_fb if elapsed_fb > 0 else 0
                    remaining_fb = len(fallback_chunks) - fb_completed
                    eta_fb = remaining_fb / rate_fb if rate_fb > 0 else 0
                    print(f"  [fallback {fb_completed}/{len(fallback_chunks)}] "
                          f"enriched={fallback_enriched} errors={fb_errors} | "
                          f"{elapsed_fb:.0f}s | ETA: {eta_fb:.0f}s")

        fb_elapsed = time.monotonic() - t_fb
        print(f"  Fallback done: {fallback_enriched} enriched, {fb_errors} errors, {fb_elapsed:.0f}s")
    else:
        print("\nNo fallback needed — all chunks parsed successfully in batched mode.")

    # Final commit
    conn.commit()
    elapsed = time.monotonic() - t0

    stats = {
        "total": total,
        "already_enriched": already_enriched,
        "skipped": skipped,
        "to_process": len(to_process),
        "enriched": enriched,
        "batched_enriched": enriched - fallback_enriched,
        "fallback_enriched": fallback_enriched,
        "parse_failures": parse_failures,
        "errors": errors,
        "batches": total_batches,
        "batch_size": batch_size,
        "workers": workers,
        "time_seconds": round(elapsed, 1),
        "rate_chunks_per_sec": round(enriched / elapsed, 2) if elapsed > 0 else 0,
        "per_chunk_seconds": round(elapsed / enriched, 2) if enriched > 0 else 0,
    }

    print(f"\n{'='*60}")
    print(f"Batched enrichment complete!")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"{'='*60}")

    conn.close()
    return stats


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Batched enrichment with Ollama")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Chunks per LLM request (default: 5)")
    parser.add_argument("--workers", type=int, default=3,
                        help="Concurrent batch requests (default: 3)")
    parser.add_argument("--model", type=str, default="qwen3.5:4b-q4_K_M",
                        help="Ollama model name")
    parser.add_argument("--min-chars", type=int, default=800,
                        help="Minimum chunk length to enrich (default: 800)")
    parser.add_argument("--store-db", type=str,
                        default="outputs/experiments/E12-corpus/document_store.db",
                        help="Path to document_store.db")
    args = parser.parse_args()

    run_batched_enrichment(
        store_db=Path(args.store_db),
        batch_size=args.batch_size,
        workers=args.workers,
        model=args.model,
        min_chars=args.min_chars,
    )


if __name__ == "__main__":
    main()
