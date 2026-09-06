"""Re-chunk all documents in document_store using semantic chunker.

Loads BGE-M3 ONCE and reuses it for all documents (avoids reloading 1.2 GB
model 967 times).  Commits after each document so progress is not lost.

Reads documents from document_store.db, re-chunks each with
chunk_document_semantic (BGE-M3 embeddings), and replaces the old
512-char fixed-window chunks.
"""
import sys
import json
import sqlite3
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from ipa.contracts import CanonicalDocument, DocumentChunk
from ipa.ingestion.alt_chunkers import chunk_document_semantic
from ipa.indexes.embedding_adapter import EmbeddingAdapter

corpus = Path("outputs/experiments/E12-corpus")
store_db = corpus / "document_store.db"

conn = sqlite3.connect(str(store_db))
conn.execute("PRAGMA journal_mode=WAL")
c = conn.cursor()

# Get all documents
c.execute("SELECT document_id, parser_id, mime_type, pages, text FROM documents")
docs = c.fetchall()
print(f"Documents to re-chunk: {len(docs)}")

# Count current chunks
c.execute("SELECT COUNT(*) FROM chunks")
old_count = c.fetchone()[0]
print(f"Current chunks (512-char fixed): {old_count}")

# Find documents that already have semantic chunks (resumable mode).
# Semantic chunks have avg text length > 600 chars (min_chunk_size=500).
# Old fixed-window chunks were exactly 512 chars.
# We skip documents that already have chunks with avg length > 600.
c.execute("""
    SELECT d.document_id, AVG(LENGTH(c.text)) as avg_chunk_len, COUNT(c.chunk_id) as n_chunks
    FROM documents d
    JOIN chunks c ON d.document_id = c.document_id
    GROUP BY d.document_id
    HAVING avg_chunk_len > 600
""")
already_done = {row[0] for row in c.fetchall()}
print(f"Documents already semantic-chunked (skipping): {len(already_done)}")
print(f"Documents remaining: {len(docs) - len(already_done)}")
print()

# Load BGE-M3 ONCE â€” reuse for all documents
print("Loading BGE-M3 (once)...")
emb = EmbeddingAdapter(batch_size=16, show_progress=False)
print(f"Device: {emb.active_device}, Dims: {emb.dimension}")
print()

# Process each document
total_new_chunks = 0
t0 = time.monotonic()

for i, (doc_id, parser_id, mime_type, pages, text) in enumerate(docs):
    if not text or len(text) < 100:
        continue

    # Skip documents already semantic-chunked (resumable mode)
    if doc_id in already_done:
        continue

    doc = CanonicalDocument(
        document_id=doc_id,
        pages=pages,
        elements=[],
        source_spans=[],
        text=text,
        mime_type=mime_type,
        parser_id=parser_id,
    )

    # Semantic chunk â€” reuse the loaded BGE-M3
    try:
        new_chunks = chunk_document_semantic(
            doc, min_chunk_size=500, max_chunk_size=3000,
            embedding_adapter=emb,
        )
    except Exception as e:
        print(f"  [{i+1}/{len(docs)}] ERROR doc {doc_id[:16]}...: {e}")
        continue

    if not new_chunks:
        continue

    # Delete old chunks for this document
    c.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))

    # Insert new chunks
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for chunk in new_chunks:
        span_json = json.dumps({
            "page": chunk.source_span.page if chunk.source_span else 0,
            "char_start": chunk.source_span.char_start if chunk.source_span else 0,
            "char_end": chunk.source_span.char_end if chunk.source_span else 0,
        }) if chunk.source_span else "{}"
        c.execute(
            "INSERT INTO chunks (chunk_id, document_id, content_hash, text, metadata_json, span_json, stored_at, tombstoned) VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
            (chunk.chunk_id, chunk.document_id, chunk.content_hash, chunk.text,
             json.dumps(chunk.metadata), span_json, now),
        )

    # Commit after each document â€” don't lose progress
    conn.commit()
    total_new_chunks += len(new_chunks)
    elapsed = time.monotonic() - t0
    rate = (i + 1) / elapsed if elapsed > 0 else 0
    eta = (len(docs) - i - 1) / rate if rate > 0 else 0

    if (i + 1) % 10 == 0 or i == 0:
        print(f"  [{i+1}/{len(docs)}] doc {doc_id[:16]}... â†’ {len(new_chunks)} chunks "
              f"(total: {total_new_chunks}, {elapsed:.0f}s, ETA: {eta:.0f}s)")

emb.close()

# Final stats
c.execute("SELECT COUNT(*) FROM chunks")
new_total = c.fetchone()[0]
c.execute("SELECT MIN(LENGTH(text)), MAX(LENGTH(text)), AVG(LENGTH(text)) FROM chunks")
stats = c.fetchone()

print(f"\n{'='*60}")
print(f"Re-chunking complete!")
print(f"  Documents processed: {len(docs)}")
print(f"  Old chunks: {old_count}")
print(f"  New chunks: {new_total}")
print(f"  Chunk size: min={stats[0]}, max={stats[1]}, avg={stats[2]:.0f}")
print(f"  Time: {time.monotonic()-t0:.0f}s")
print(f"{'='*60}")

conn.close()
