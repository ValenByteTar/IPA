"""Incremental LanceDB builder â€” runs as a separate process.

Polls document_store.db for new chunks and appends them to LanceDB.
Uses BGE-M3 hybrid embeddings (dense + sparse) in a single forward pass.
Only embeds NEW chunks (not already in LanceDB). Small batches, frequent polls.

Usage:
    python scripts/_lancedb_incremental.py
"""
import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from ipa.contracts import DocumentChunk
from ipa.indexes.embedding_adapter import EmbeddingAdapter
from ipa.indexes.lancedb_index import LanceDBIndex

CORPUS = Path("outputs/experiments/E12-corpus")
STORE_DB = CORPUS / "document_store.db"
LANDEDB_PATH = CORPUS / "vector" / "lancedb"
POLL_INTERVAL = 3   # seconds between polls
BATCH_SIZE = 192    # chunks per embedding batch (3x faster)

print("=" * 60)
print("Incremental LanceDB Builder (hybrid: dense + sparse)")
print("=" * 60)
print(f"  Store:    {STORE_DB}")
print(f"  LanceDB:  {LANDEDB_PATH}")
print(f"  Poll:     {POLL_INTERVAL}s")
print(f"  Batch:    {BATCH_SIZE} chunks")
print()

# Initialize embedding + LanceDB
print("Loading BGE-M3 embedding model (hybrid mode)...")
embedding = EmbeddingAdapter(batch_size=BATCH_SIZE, max_length=512)
print(f"  Embedding: {embedding.model_name}, dim={embedding.dimension}")

lancedb = LanceDBIndex(LANDEDB_PATH, vector_dim=embedding.dimension)

# Track embedded chunk_ids in memory (fast, no LanceDB queries needed)
embedded_ids: set[str] = set()
try:
    df = lancedb._table.to_pandas()
    embedded_ids = set(df["chunk_id"].tolist())
    print(f"  LanceDB: {len(embedded_ids)} existing rows loaded")
    # Reconcile durable queue with the index after restart.
    _reconcile = sqlite3.connect(str(STORE_DB))
    _reconcile.executemany(
        "UPDATE embedding_jobs SET status='complete', completed_at=? "
        "WHERE chunk_id=? AND status != 'complete'",
        [(time.time(), cid) for cid in embedded_ids],
    )
    _reconcile.commit()
    _reconcile.close()
except Exception as e:
    print(f"  LanceDB: could not load existing IDs ({e}), starting fresh")
    # Fallback: count rows
    try:
        count = lancedb._table.count_rows()
        print(f"  LanceDB: {count} rows (IDs not loaded â€” will re-embed all)")
    except Exception:
        print(f"  LanceDB: empty, starting fresh")

print()

# Track store size to detect when pipeline is done
prev_store_count = 0
stable_count = 0
total_added = 0
round_num = 0

try:
    while True:
        round_num += 1

        # Get ALL chunk_ids from store
        conn = sqlite3.connect(str(STORE_DB))
        conn.execute("PRAGMA journal_mode=WAL")
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM chunks")
        store_count = c.fetchone()[0]

        # Consume the durable embedding queue instead of scanning all chunks.
        new_chunks = []
        c.execute(
            "SELECT c.chunk_id, c.document_id, c.text, c.content_hash "
            "FROM embedding_jobs j JOIN chunks c ON c.chunk_id=j.chunk_id "
            "WHERE j.status IN ('pending', 'retry') AND c.tombstoned=0 "
            "ORDER BY c.stored_at LIMIT ?",
            (BATCH_SIZE,),
        )
        for row in c.fetchall():
            new_chunks.append(DocumentChunk(
                chunk_id=row[0], document_id=row[1],
                text=row[2], content_hash=row[3]
            ))
        conn.close()

        if new_chunks:
            # Embed with hybrid mode (dense + sparse in single forward pass)
            texts = [c.text for c in new_chunks]
            dense_vectors, sparse_weights = embedding.embed_texts_hybrid(texts)
            lancedb.add_chunks(new_chunks, dense_vectors, sparse_weights=sparse_weights)

            _mark_conn = sqlite3.connect(str(STORE_DB))
            _completed_at = time.time()
            for chunk in new_chunks:
                embedded_ids.add(chunk.chunk_id)
                _mark_conn.execute(
                    "UPDATE embedding_jobs SET status='complete', completed_at=?, "
                    "attempts=attempts+1, error=NULL WHERE chunk_id=?",
                    (_completed_at, chunk.chunk_id),
                )
            _mark_conn.commit()
            _mark_conn.close()
            total_added += len(new_chunks)

            total_rows = lancedb._table.count_rows()
            print(f"[Round {round_num}] +{len(new_chunks)} chunks â†’ "
                  f"{total_rows} total rows | "
                  f"Store: {store_count}, Embedded: {len(embedded_ids)}, "
                  f"Missing: {store_count - len(embedded_ids)}")

            # Don't sleep if there are more to process
            if store_count - len(embedded_ids) > BATCH_SIZE:
                continue
            time.sleep(POLL_INTERVAL)
        else:
            # Nothing new â€” check if pipeline is done
            if store_count == prev_store_count:
                stable_count += 1
            else:
                stable_count = 0
            prev_store_count = store_count

            # After a stable store, rely on the pipeline's explicit state,
            # not on counting unrelated Python processes on Windows.
            if stable_count >= 6:
                pipeline_state = STORE_DB.parent / "process_state" / "pipeline.json"
                pipeline_done = False
                try:
                    state = json.loads(pipeline_state.read_text(encoding="utf-8"))
                    pipeline_done = state.get("status") == "done"
                except (OSError, ValueError, json.JSONDecodeError):
                    pass
                _pending_conn = sqlite3.connect(str(STORE_DB))
                pending = _pending_conn.execute(
                    "SELECT COUNT(*) FROM embedding_jobs "
                    "WHERE status IN ('pending', 'retry')"
                ).fetchone()[0]
                _pending_conn.close()
                if pipeline_done and store_count == len(embedded_ids) and pending == 0:
                    print(f"\n[Round {round_num}] Pipeline finished. Creating FTS index...")
                    lancedb.create_fts_index()
                    print(f"  FTS index created. Total rows: {lancedb._table.count_rows()}")
                    break
                stable_count = 0

            print(f"[Round {round_num}] No new chunks | "
                  f"Store: {store_count}, Embedded: {len(embedded_ids)}, "
                  f"Waiting {POLL_INTERVAL}s...")
            time.sleep(POLL_INTERVAL)

except KeyboardInterrupt:
    print("\nInterrupted â€” shutting down gracefully...")

# Final
print(f"\n{'='*60}")
print(f"Incremental LanceDB builder complete!")
print(f"  Total added: {total_added}")
print(f"  Total rows:  {lancedb._table.count_rows()}")
print(f"  FTS index:   created")
print(f"{'='*60}")

lancedb.close()
embedding.close()
print("  LanceDB and embedding closed.")
