"""Standalone chunk-enrichment runner (ExLlamaV3, carga su propio modelo).

DEPRECATED como job del pipeline del Orchestrator — el enriquecimiento ahora
corre como tarea `enrich_chunks` del Tier 2 idle (sobre el 9B ya cargado del
pase; ver ipa.agentic.chunk_enrichment). Este script queda para corridas
manuales/depuración: carga el 4B directo por la API cruda de exllamav3 y
reusa scan/parse/write/re-embed del módulo compartido.

Enrichment por chunk:
  1. Summary: 1-2 oraciones prepend como [Summary] ...
  2. Synthetic queries: 3 preguntas prepend como [Questions] ...
Re-embede en LanceDB (híbrido denso+sparse). Resumable vía checkpoints.
"""
import os
import sys
from pathlib import Path

os.environ["CUDA_PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
_ninja_dir = Path(sys.executable).parent
os.environ["PATH"] = os.environ["CUDA_PATH"] + r"\bin;" + str(_ninja_dir) + os.pathsep + os.environ["PATH"]


sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
import sqlite3
import time
from pathlib import Path

from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.generator import Generator, Job
from exllamav3.generator.sampler import ComboSampler

from ipa.agentic.chunk_enrichment import (
    ENRICH_PROMPT, ENRICH_SYSTEM, REEMBED_BATCH, TEXT_LIMIT,
    build_enriched_text, mark_reembedded, parse_enrichment,
    reembed_batch, scan_chunks, write_enrichment,
)
from ipa.indexes.embedding_adapter import EmbeddingAdapter
from ipa.indexes.lancedb_index import LanceDBIndex

MODEL_PATH = "models/Qwen3.5-4B-exl3-4bpw"
STORE_DB = Path("outputs/experiments/E12-corpus/document_store.db")
LANCEDB_PATH = Path("outputs/experiments/E12-corpus/vector/lancedb")
BATCH_SIZE = 16
MAX_NEW_TOKENS = 150  # summary + 3 questions

# Thinking tags (built via chr to avoid XML rendering issues)
THINK_START = chr(60) + "think" + chr(62)
THINK_END = chr(60) + "/think" + chr(62)
IM_END = chr(60) + "|im_end|" + chr(62)


def make_prompt(text: str) -> str:
    """ChatML prompt con thinking deshabilitado vía bloque think vacío."""
    prompt = ENRICH_PROMPT.format(text=text[:TEXT_LIMIT])
    im_start = chr(60) + "|im_start|"
    im_end = chr(60) + "|im_end|"
    return (
        f"{im_start}system\n{ENRICH_SYSTEM}{im_end}\n"
        f"{im_start}user\n{prompt}{im_end}\n"
        f"{im_start}assistant\n"
        f"{THINK_START}\n\n{THINK_END}\n\n"
    )


# ---------------------------------------------------------------------------
# Load chunks to process
# ---------------------------------------------------------------------------

conn = sqlite3.connect(str(STORE_DB))
conn.execute("PRAGMA journal_mode=WAL")

scan = scan_chunks(conn)
to_process = scan.to_process
pending_reembed = list(scan.pending_reembed)
total_to_process = len(to_process)

print("=" * 60)
print("ExLlamaV3 Enrichment (summary + synthetic queries + LanceDB re-embed)")
print("=" * 60)
print(f"  Total chunks: {scan.total}")
print(f"  Already enriched (skipping): {scan.already_enriched}")
print(f"  Skipped (short or dense): {scan.skipped}")
print(f"  To process: {total_to_process}")
print(f"  Pending re-embed (recovered): {len(pending_reembed)}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Re-embed batch: {REEMBED_BATCH}")
print()

if not scan.pending:
    print("No chunks need enrichment or re-embedding. Exiting.")
    conn.close()
    exit()

# ---------------------------------------------------------------------------
# Load ExLlamaV3 model
# ---------------------------------------------------------------------------

print("Loading ExLlamaV3 model...")
t_load = time.monotonic()

config = Config.from_directory(MODEL_PATH)
model = Model.from_config(config, component="text")
tokenizer = Tokenizer(config)
cache = Cache(model, max_num_tokens=8192, max_batch_size=BATCH_SIZE)
model.load()

generator = Generator(model, cache, tokenizer)

print(f"  Model loaded in {time.monotonic() - t_load:.1f}s")
print()

# ---------------------------------------------------------------------------
# Load BGE-M3 for re-embedding
# ---------------------------------------------------------------------------

print("Loading BGE-M3 for re-embedding...")
t_embed_load = time.monotonic()
embedding = EmbeddingAdapter(batch_size=REEMBED_BATCH, max_length=512)
lancedb = LanceDBIndex(LANCEDB_PATH, vector_dim=embedding.dimension)
print(f"  BGE-M3 loaded in {time.monotonic() - t_embed_load:.1f}s")
print(f"  LanceDB: {lancedb._table.count_rows()} existing rows")
print()

# ---------------------------------------------------------------------------
# Process in batches
# ---------------------------------------------------------------------------

sampler = ComboSampler()
sampler.temperature = 0.3
sampler.top_p = 0.9

enriched = 0
errors = 0
reembedded = 0
t0 = time.monotonic()

print(f"Starting enrichment (batch={BATCH_SIZE})...")
print()


def _flush_pending() -> None:
    global reembedded
    while pending_reembed:
        batch, pending_reembed[:] = (
            pending_reembed[:REEMBED_BATCH], pending_reembed[REEMBED_BATCH:])
        conn.commit()  # ensure DB is consistent
        reembedded += reembed_batch(lancedb, embedding, batch)
        mark_reembedded(conn, batch)


try:
    if pending_reembed:
        print(f"  Recovering {len(pending_reembed)} pending re-embeddings...")
        _flush_pending()

    for batch_start in range(0, total_to_process, BATCH_SIZE):
        batch = to_process[batch_start:batch_start + BATCH_SIZE]

        prompts = [make_prompt(text) for _, _, text, _ in batch]
        input_ids_list = [tokenizer.encode(p, add_bos=False) for p in prompts]

        jobs = []
        for i, ids in enumerate(input_ids_list):
            jobs.append(Job(
                input_ids=ids,
                max_new_tokens=MAX_NEW_TOKENS,
                sampler=sampler,
                stop_conditions=[IM_END],
                identifier=i,
            ))
        for job in jobs:
            generator.enqueue(job)

        results = [None] * len(jobs)
        while generator.num_remaining_jobs() > 0:
            for result in generator.iterate():
                idx = result["identifier"]
                if result["stage"] == "streaming":
                    text = result.get("text", "")
                    if results[idx] is None:
                        results[idx] = ""
                    results[idx] += text

        for i, (chunk_id, doc_id, text, content_hash) in enumerate(batch):
            raw = results[i] if i < len(results) else None
            if raw is None or len(raw.strip()) < 10:
                errors += 1
                if errors <= 5:
                    print(f"  ERROR chunk {chunk_id[:16]}...: empty output")
                continue

            summary, questions = parse_enrichment(raw)
            if not summary and not questions:
                errors += 1
                if errors <= 5:
                    print(f"  ERROR chunk {chunk_id[:16]}...: parse failed, raw[:80]={raw[:80]!r}")
                continue

            enriched_text = build_enriched_text(summary, questions, text)
            write_enrichment(conn, chunk_id, enriched_text, text)
            enriched += 1
            pending_reembed.append((chunk_id, doc_id, enriched_text, content_hash))

        if enriched > 0:
            conn.commit()

        if len(pending_reembed) >= REEMBED_BATCH:
            _flush_pending()

        completed = batch_start + len(batch)
        if (completed // BATCH_SIZE) % 5 == 0 or completed >= total_to_process:
            elapsed = time.monotonic() - t0
            rate = enriched / elapsed if elapsed > 0 else 0
            remaining = total_to_process - completed
            eta = remaining / rate if rate > 0 else 0
            print(
                f"  [{completed}/{total_to_process}] enriched={enriched} "
                f"reembedded={reembedded} errors={errors} | "
                f"{elapsed:.0f}s | {rate:.2f} chunks/s | "
                f"ETA: {eta:.0f}s ({eta/60:.0f} min)"
            )

    conn.commit()
    _flush_pending()

except KeyboardInterrupt:
    print("\nInterrupted — shutting down gracefully...")
    conn.commit()
    if pending_reembed:
        print(f"  Re-embedding {len(pending_reembed)} pending chunks...")
        _flush_pending()
    print("  Graceful shutdown complete.")

# Final stats
elapsed = time.monotonic() - t0
print(f"\n{'='*60}")
print(f"ExLlamaV3 enrichment complete!")
print(f"  Total chunks: {scan.total}")
print(f"  Already enriched: {scan.already_enriched}")
print(f"  To process: {total_to_process}")
print(f"  Enriched: {enriched}")
print(f"  Re-embedded: {reembedded}")
print(f"  Errors: {errors}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
if enriched > 0 and elapsed > 0:
    print(f"  Rate: {enriched/elapsed:.2f} chunks/s")
    print(f"  Per-chunk: {elapsed/enriched:.2f}s")
print(f"  LanceDB rows: {lancedb._table.count_rows()}")
print(f"{'='*60}")

conn.close()
lancedb.close()
embedding.close()
