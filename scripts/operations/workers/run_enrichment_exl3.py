"""Run enrichment with ExLlamaV3 (native GPU inference, no HTTP overhead).

Uses ExLlamaV3 with Qwen3.5-4B EXL3 4.0bpw for batched summarization.
Thinking mode disabled for concise summaries.

Enrichment per chunk:
  1. Summary: 1-2 sentence summary prepended as [Summary] ...
  2. Synthetic queries: 3 questions prepended as [Questions] ...

After enriching, re-embeds enriched chunks in LanceDB (hybrid dense+sparse)
so vector retrieval benefits from the enriched text.

Resumable: skips chunks that already have [Summary] prefix.
"""
import os
import sys
from pathlib import Path

os.environ["CUDA_PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
_ninja_dir = Path(sys.executable).parent
os.environ["PATH"] = os.environ["CUDA_PATH"] + r"\bin;" + str(_ninja_dir) + os.pathsep + os.environ["PATH"]


sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
import hashlib
import json
import sqlite3
import time
from pathlib import Path

from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.generator import Generator, Job
from exllamav3.generator.sampler import ComboSampler

from ipa.contracts import DocumentChunk
from ipa.indexes.embedding_adapter import EmbeddingAdapter
from ipa.indexes.lancedb_index import LanceDBIndex
from ipa.ingestion.continuous_pipeline import should_summarize

MODEL_PATH = "models/Qwen3.5-4B-exl3-4bpw"
STORE_DB = Path("outputs/experiments/E12-corpus/document_store.db")
LANCEDB_PATH = Path("outputs/experiments/E12-corpus/vector/lancedb")
BATCH_SIZE = 16
MAX_NEW_TOKENS = 150  # summary + 3 questions
COMMIT_EVERY = 50
REEMBED_BATCH = 192  # BGE-M3 batch size for re-embedding

# Thinking tags (built via chr to avoid XML rendering issues)
THINK_START = chr(60) + "think" + chr(62)
THINK_END = chr(60) + "/think" + chr(62)
IM_END = chr(60) + "|im_end|" + chr(62)

ENRICH_SYSTEM = (
    "You are a document enrichment system. "
    "First, summarize the text in 1-2 sentences. "
    "Then, generate 3 questions that this text would answer. "
    "Format your response as:\n"
    "SUMMARY: <your summary>\n"
    "Q1: <question 1>\n"
    "Q2: <question 2>\n"
    "Q3: <question 3>"
)

ENRICH_PROMPT = (
    'Enrich this text with a summary and 3 synthetic questions.\n\n'
    'Text:\n'
    '"""\n'
    '{text}\n'
    '"""\n'
)

def make_prompt(text: str) -> str:
    """ChatML prompt with thinking disabled via empty think block."""
    prompt = ENRICH_PROMPT.format(text=text[:2000])
    im_start = chr(60) + "|im_start|"
    im_end = chr(60) + "|im_end|"
    return (
        f"{im_start}system\n{ENRICH_SYSTEM}{im_end}\n"
        f"{im_start}user\n{prompt}{im_end}\n"
        f"{im_start}assistant\n"
        f"{THINK_START}\n\n{THINK_END}\n\n"
    )

def parse_enrichment(raw_output: str) -> tuple[str, list[str]]:
    """Parse LLM output into (summary, questions).

    Expected format:
      SUMMARY: <summary text>
      Q1: <question 1>
      Q2: <question 2>
      Q3: <question 3>
    """
    # Strip think block if present
    if THINK_END in raw_output:
        parts = raw_output.split(THINK_END, 1)
        raw_output = parts[1].strip()

    summary = ""
    questions = []

    for line in raw_output.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.upper().startswith("SUMMARY:"):
            summary = line[8:].strip()
        elif line.upper().startswith("Q") and ":" in line:
            # Q1: ..., Q2: ..., etc.
            q = line.split(":", 1)[1].strip()
            if len(q) > 5:
                questions.append(q)

    # Fallback: if no structured output, use raw as summary
    if not summary and not questions:
        summary = raw_output.strip()[:200]

    return summary, questions

def build_enriched_text(summary: str, questions: list[str], original: str) -> str:
    """Build enriched text with summary and synthetic questions prepended."""
    parts = []
    if summary:
        parts.append(f"[Summary] {summary}")
    if questions:
        q_block = "\n".join(f"Q: {q}" for q in questions)
        parts.append(f"[Questions]\n{q_block}")
    parts.append(original)
    return "\n\n".join(parts)

ENRICHMENT_VERSION = "exl3-v1"


def _metadata(text: str) -> dict:
    try:
        value = json.loads(text or "{}")
        return value if isinstance(value, dict) else {}
    except (TypeError, ValueError):
        return {}


def _enriched_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _reembed_batch(
    lancedb: LanceDBIndex,
    embedding: EmbeddingAdapter,
    chunks: list[tuple[str, str, str, str]],
) -> int:
    """Re-embed enriched chunks in LanceDB.

    add_chunks is idempotent (deletes existing chunk_ids before insert),
    so we just call it with the enriched text â€” old embeddings are replaced.
    """
    if not chunks:
        return 0
    chunk_objs = [
        DocumentChunk(chunk_id=c[0], document_id=c[1], text=c[2], content_hash=c[3])
        for c in chunks
    ]
    texts = [c.text for c in chunk_objs]
    dense, sparse = embedding.embed_texts_hybrid(texts)
    lancedb.add_chunks(chunk_objs, dense, sparse_weights=sparse)
    return len(chunks)


def _mark_reembedded(conn: sqlite3.Connection, chunks: list[tuple[str, str, str, str]]) -> None:
    """Advance checkpoints only after LanceDB accepts the whole batch."""
    for chunk_id, _, text, _ in chunks:
        c = conn.execute("SELECT metadata_json FROM chunks WHERE chunk_id = ?", (chunk_id,))
        metadata = _metadata(c.fetchone()[0])
        enrichment = metadata.setdefault("enrichment", {})
        enrichment.update({"embedding_status": "complete", "embedded_at": time.time(), "text_hash": _enriched_hash(text)})
        conn.execute("UPDATE chunks SET metadata_json = ? WHERE chunk_id = ?", (json.dumps(metadata, ensure_ascii=False), chunk_id))
    conn.commit()

# ---------------------------------------------------------------------------
# Load chunks to process
# ---------------------------------------------------------------------------

conn = sqlite3.connect(str(STORE_DB))
conn.execute("PRAGMA journal_mode=WAL")
c = conn.cursor()

c.execute("SELECT chunk_id, document_id, text, content_hash, metadata_json FROM chunks")
rows = c.fetchall()
total = len(rows)

to_process = []
pending_reembed: list[tuple[str, str, str, str]] = []
already_enriched = 0
skipped = 0

for chunk_id, doc_id, text, content_hash, metadata_json in rows:
    metadata = _metadata(metadata_json)
    if text.startswith("[Summary]"):
        already_enriched += 1
        # Legacy enriched rows have no checkpoint and are deliberately retried.
        if metadata.get("enrichment", {}).get("embedding_status") != "complete" or metadata.get("enrichment", {}).get("text_hash") != _enriched_hash(text):
            pending_reembed.append((chunk_id, doc_id, text, content_hash))
        continue
    chunk = DocumentChunk(
        chunk_id=chunk_id, document_id=doc_id,
        text=text, content_hash=content_hash,
    )
    if should_summarize(chunk, min_chars=800):
        to_process.append((chunk_id, doc_id, text, content_hash))
    else:
        skipped += 1

print("=" * 60)
print("ExLlamaV3 Enrichment (summary + synthetic queries + LanceDB re-embed)")
print("=" * 60)
print(f"  Total chunks: {total}")
print(f"  Already enriched (skipping): {already_enriched}")
print(f"  Skipped (short or dense): {skipped}")
print(f"  To process: {len(to_process)}")
print(f"  Pending re-embed (recovered): {len(pending_reembed)}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Re-embed batch: {REEMBED_BATCH}")
print()

if not to_process and not pending_reembed:
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

load_time = time.monotonic() - t_load
print(f"  Model loaded in {load_time:.1f}s")
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
total_to_process = len(to_process)

# pending_reembed also contains enriched rows recovered from an interrupted run.
print(f"Starting enrichment (batch={BATCH_SIZE})...")
print()

try:
    # Recover SQLite-enriched chunks whose LanceDB checkpoint was not committed.
    if pending_reembed:
        print(f"  Recovering {len(pending_reembed)} pending re-embeddings...")
        for _i in range(0, len(pending_reembed), REEMBED_BATCH):
            _batch = pending_reembed[_i:_i + REEMBED_BATCH]
            reembedded += _reembed_batch(lancedb, embedding, _batch)
            _mark_reembedded(conn, _batch)
        pending_reembed = []

    for batch_start in range(0, total_to_process, BATCH_SIZE):
        batch = to_process[batch_start:batch_start + BATCH_SIZE]

        # Tokenize prompts
        prompts = [make_prompt(text) for _, _, text, _ in batch]
        input_ids_list = [tokenizer.encode(p, add_bos=False) for p in prompts]

        # Create jobs
        jobs = []
        for i, ids in enumerate(input_ids_list):
            job = Job(
                input_ids=ids,
                max_new_tokens=MAX_NEW_TOKENS,
                sampler=sampler,
                stop_conditions=[IM_END],
                identifier=i,
            )
            jobs.append(job)

        # Enqueue all
        for job in jobs:
            generator.enqueue(job)

        # Collect results
        results = [None] * len(jobs)
        while generator.num_remaining_jobs() > 0:
            for result in generator.iterate():
                idx = result["identifier"]
                if result["stage"] == "streaming":
                    text = result.get("text", "")
                    if results[idx] is None:
                        results[idx] = ""
                    results[idx] += text
                elif result["stage"] == "end":
                    pass

        # Write to DB
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
            # Keep the pre-enrichment text in metadata: chunks.text remains compatible
            # with existing consumers, while the canonical chunk content is recoverable.
            c.execute("SELECT metadata_json FROM chunks WHERE chunk_id = ?", (chunk_id,))
            old_metadata = _metadata(c.fetchone()[0])
            enrichment_metadata = {
                "version": ENRICHMENT_VERSION,
                "status": "enriched",
                "embedding_status": "pending",
                "text_hash": _enriched_hash(enriched_text),
                "enriched_at": time.time(),
            }
            old_metadata.setdefault("enrichment", {}).update(enrichment_metadata)
            old_metadata["enrichment"].setdefault("canonical_text", text)
            c.execute(
                "UPDATE chunks SET text = ?, metadata_json = ? WHERE chunk_id = ?",
                (enriched_text, json.dumps(old_metadata, ensure_ascii=False), chunk_id),
            )
            enriched += 1
            pending_reembed.append((chunk_id, doc_id, enriched_text, content_hash))

        # Commit every inference batch so an interrupted process leaves a
        # durable enriched/pending checkpoint, not an uncommitted mutation.
        if enriched > 0:
            conn.commit()

        # Re-embed in LanceDB when we have enough pending chunks
        if len(pending_reembed) >= REEMBED_BATCH:
            conn.commit()  # ensure DB is consistent
            _re = _reembed_batch(lancedb, embedding, pending_reembed)
            _mark_reembedded(conn, pending_reembed)
            reembedded += _re
            pending_reembed = []

        # Progress report
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

    # Final: re-embed any remaining pending chunks
    conn.commit()
    if pending_reembed:
        _re = _reembed_batch(lancedb, embedding, pending_reembed)
        _mark_reembedded(conn, pending_reembed)
        reembedded += _re

except KeyboardInterrupt:
    print("\nInterrupted â€” shutting down gracefully...")
    # Commit what we have and re-embed pending
    conn.commit()
    if pending_reembed:
        print(f"  Re-embedding {len(pending_reembed)} pending chunks...")
        _re = _reembed_batch(lancedb, embedding, pending_reembed)
        _mark_reembedded(conn, pending_reembed)
        reembedded += _re
    print("  Graceful shutdown complete.")

# Final stats
elapsed = time.monotonic() - t0
print(f"\n{'='*60}")
print(f"ExLlamaV3 enrichment complete!")
print(f"  Total chunks: {total}")
print(f"  Already enriched: {already_enriched}")
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
