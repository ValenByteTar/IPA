"""Continuous ingestion pipeline â€” watches Landing/, processes files as they
appear, runs the full pipeline (parse â†’ chunk â†’ store â†’ BM25 â†’ LanceDB â†’
enrich), then moves processed files to Archive/.

Flow:
    Landing/  (scraper writes here)
        â†“ (watcher detects new files)
    mime â†’ parse (PyMuPDF with smart OCR) â†’ chunk â†’ store â†’ BM25
        â†“
    embedding â†’ LanceDB
        â†“
    enrichment (selective summarize with Qwen3.5 by density/size)
        â†“
    Archive/  (processed files moved here)

Usage:
    python scripts/operations/run_continuous_pipeline.py \\
        --landing Landing \\
        --archive Archive \\
        --corpus outputs/experiments/E12-corpus \\
        --ollama-model qwen3.5:4b-q4_K_M

The pipeline runs in a loop, polling Landing/ every few seconds for new
files.  It exits when no new files appear for --idle-timeout seconds.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

from ipa import FastPathRunner, TraceLog
from ipa.indexes.lancedb_index import LanceDBIndex
from ipa.indexes.embedding_adapter import EmbeddingAdapter
from ipa.enrichment.enrichment import enrich_summary
from ipa.enrichment.ollama_adapter import OllamaAdapter
from ipa.contracts import DocumentChunk


# ---------------------------------------------------------------------------
# Selective enrichment: only summarize chunks by density/size
# ---------------------------------------------------------------------------

def should_summarize(chunk: DocumentChunk, min_chars: int = 800) -> bool:
    """Decide if a chunk should be summarized.

    Criteria (selective by density/size):
      1. Chunk is long enough to benefit from summarization (>min_chars)
      2. Chunk has low information density (lots of whitespace/repetition)

    Short chunks are already concise â€” summarizing them wastes LLM calls
    and can lose information.
    """
    if len(chunk.text) < min_chars:
        return False

    # Information density: ratio of non-whitespace unique words to total chars
    words = chunk.text.split()
    if not words:
        return False

    unique_words = set(w.lower() for w in words)
    density = len(unique_words) / max(len(words), 1)

    # Low density = repetitive text benefits from summarization
    # High density = dense factual text, keep as-is
    return density < 0.6


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def _is_lock_error(error: object) -> bool:
    """Check if an error is a lock/IO/permission error (retryable).

    Tantivy lock conflicts and Windows file permission errors can be
    transient â€” the lock may be released after a brief wait.
    """
    err_str = str(error).lower()
    return any(s in err_str for s in [
        "permissiondenied",
        "acceso denegado",
        "os error 5",
        "io error",
        "lock",
        "would block",
    ])


def process_file(runner: FastPathRunner, filepath: Path) -> dict | None:
    """Run fast_path on a single file. Returns result dict or None on error."""
    try:
        result = runner.ingest(filepath)
        return {
            "artifact_id": result.artifact_id,
            "mime_type": result.mime_type,
            "parser_id": result.parser_id,
            "document_id": result.document_id,
            "pages": result.pages,
            "chunks_created": result.chunks_created,
            "elapsed_seconds": round(result.elapsed_seconds, 4),
            "errors": result.errors,
        }
    except Exception as e:
        return {
            "artifact_id": "",
            "mime_type": "",
            "parser_id": "",
            "document_id": None,
            "pages": 0,
            "chunks_created": 0,
            "elapsed_seconds": 0,
            "errors": [str(e)],
        }


def build_lancedb(store_db: Path, lancedb_path: Path,
                  embedding: EmbeddingAdapter) -> tuple[int, int]:
    """Build LanceDB index from all chunks in document_store.

    Returns (total_chunks, embedded_chunks).
    """
    import sqlite3

    index = LanceDBIndex(lancedb_path, vector_dim=embedding.dimension)
    conn = sqlite3.connect(str(store_db))
    c = conn.cursor()

    # Get all chunks
    c.execute("SELECT chunk_id, document_id, text, content_hash FROM chunks")
    rows = c.fetchall()
    conn.close()

    if not rows:
        return 0, 0

    chunks = []
    for row in rows:
        chunk = DocumentChunk(
            chunk_id=row[0],
            document_id=row[1],
            text=row[2],
            content_hash=row[3],
        )
        chunks.append(chunk)

    # Generate embeddings
    texts = [c.text for c in chunks]
    vectors = embedding.embed_texts(texts)

    # Add to LanceDB
    index.add_chunks(chunks, vectors)

    # Create FTS index for hybrid search (dense + BM25 keyword matching)
    print("  Creating FTS index for hybrid search...")
    index.create_fts_index()

    index.close()

    return len(chunks), len(vectors)


def enrich_selective(store_db: Path, llm: OllamaAdapter,
                     min_chars: int = 800) -> tuple[int, int]:
    """Run selective summary enrichment on chunks in the store.

    Only chunks that pass should_summarize() are sent to the LLM.
    The enriched text replaces the chunk text in the store.

    Returns (total_chunks, enriched_chunks).
    """
    import sqlite3

    conn = sqlite3.connect(str(store_db))
    c = conn.cursor()

    # Get all chunks
    c.execute("SELECT chunk_id, document_id, text, content_hash FROM chunks")
    rows = c.fetchall()

    total = len(rows)
    enriched = 0

    for chunk_id, doc_id, text, content_hash in rows:
        chunk = DocumentChunk(
            chunk_id=chunk_id,
            document_id=doc_id,
            text=text,
            content_hash=content_hash,
        )

        if not should_summarize(chunk, min_chars=min_chars):
            continue

        # Summarize with Qwen3.5
        result = enrich_summary(chunk_id, text, llm)
        if result.error:
            continue

        # Update chunk text in store with enriched version
        c.execute(
            "UPDATE chunks SET text = ? WHERE chunk_id = ?",
            (result.enriched_text, chunk_id),
        )
        enriched += 1
        print(f"    Enriched {chunk_id[:16]}... ({len(text)} â†’ {len(result.enriched_text)} chars)")

    conn.commit()
    conn.close()
    return total, enriched


def move_to_archive(filepath: Path, archive_dir: Path) -> None:
    """Move a processed file from Landing to Archive, preserving structure.

    On Windows, PyMuPDF and other parsers may hold file handles briefly even
    after close. We use a robust 3-phase approach:

    1. gc.collect() to release lingering Python-level handles
    2. shutil.copy2 to copy to Archive (always succeeds)
    3. filepath.unlink() with retries â€” if it still fails, rename to
       .pending_delete so the file is not re-processed, and it will be
       cleaned up on the next pipeline startup via cleanup_pending_delete().

    This guarantees no duplicates: either the file is deleted from Landing,
    or it's renamed to .pending_delete (which the pipeline ignores).
    """
    rel = filepath  # relative path within Landing
    dest = archive_dir / rel.name
    # Handle name collisions
    if dest.exists():
        stem = dest.stem
        suffix = dest.suffix
        dest = archive_dir / f"{stem}_{int(time.time()) % 10000}{suffix}"

    # Phase 1: Force garbage collection to release any lingering file handles
    # held by PyMuPDF, docling, or other parser objects.
    gc.collect()

    # Phase 2: Copy to Archive (this should always succeed since we're
    # only reading the source file, not writing to it).
    shutil.copy2(str(filepath), str(dest))

    # Phase 3: Delete from Landing with retries.
    # If unlink fails after all retries, rename to .pending_delete so
    # the file is not re-processed in the next cycle.
    for attempt in range(5):
        try:
            filepath.unlink(missing_ok=True)
            # Verify deletion â€” Windows may silently fail
            if not filepath.exists():
                return
            # File still exists after unlink â€” retry
        except (PermissionError, OSError) as e:
            if attempt < 4:
                # Exponential backoff: 0.5s, 1s, 2s, 4s, 8s
                time.sleep(0.5 * (2 ** attempt))
                gc.collect()  # retry GC between attempts
            else:
                # Last resort: rename to .pending_delete so the pipeline
                # ignores it in future cycles. The file will be cleaned up
                # by cleanup_pending_delete() on next startup.
                try:
                    pending = filepath.with_suffix(filepath.suffix + ".pending_delete")
                    filepath.rename(pending)
                    print(f"    âš  Could not delete (handle locked) â€” renamed to {pending.name}")
                except (PermissionError, OSError):
                    # Even rename failed â€” the file is locked hard.
                    # Log warning but don't raise; the file is already in Archive.
                    print(f"    âš  WARNING: Could not delete or rename {filepath.name} â€” "
                          f"file is locked. Already copied to Archive. "
                          f"Manual cleanup may be required.")
                return


def cleanup_pending_delete(landing_dir: Path) -> int:
    """Remove any .pending_delete files left from previous runs.

    These are files that were successfully processed and copied to Archive,
    but could not be deleted from Landing due to Windows file handle locks.
    On the next startup, the locks should be released and we can clean them.

    Returns the number of files cleaned up.
    """
    cleaned = 0
    for f in landing_dir.rglob("*.pending_delete"):
        try:
            f.unlink(missing_ok=True)
            cleaned += 1
        except (PermissionError, OSError):
            pass  # still locked â€” try again next run
    if cleaned:
        print(f"  Cleaned up {cleaned} .pending_delete file(s) from previous run(s)")
    return cleaned


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Continuous ingestion pipeline.")
    parser.add_argument("--landing", default="Landing", help="Landing directory to watch.")
    parser.add_argument("--archive", default="Archive", help="Archive directory for processed files.")
    parser.add_argument("--corpus", default="outputs/experiments/E12-corpus", help="Output for DBs.")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    parser.add_argument("--chunker", default="fixed", choices=["fixed", "semantic"],
                        help="Chunker: 'fixed' (512-char window) or 'semantic' (BGE-M3 similarity).")
    parser.add_argument("--chunker-threshold", type=float, default=0.3,
                        help="Semantic chunker: cosine similarity below which a boundary is created.")
    parser.add_argument("--chunker-min-size", type=int, default=500,
                        help="Semantic chunker: minimum chunk size in chars.")
    parser.add_argument("--chunker-max-size", type=int, default=3000,
                        help="Semantic chunker: maximum chunk size in chars (force split).")
    parser.add_argument("--ollama-model", default="qwen3.5:4b-q4_K_M", help="Ollama model for enrichment.")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--no-enrichment", action="store_true", help="Skip LLM enrichment.")
    parser.add_argument("--no-lancedb", action="store_true", help="Skip LanceDB index build.")
    parser.add_argument("--min-chars-summarize", type=int, default=800, help="Min chunk chars to summarize.")
    parser.add_argument("--poll-interval", type=float, default=2.0, help="Seconds between Landing polls.")
    parser.add_argument("--idle-timeout", type=float, default=120.0, help="Exit after this many seconds with no new files.")
    parser.add_argument("--trace-db", default=None, help="Enable trace logging.")
    parser.add_argument("--lancedb-incremental", action="store_true",
                        help="Build LanceDB incrementally as chunks are added (not just at end).")
    parser.add_argument("--lancedb-batch-size", type=int, default=256,
                        help="How many chunks to embed per incremental LanceDB batch.")
    args = parser.parse_args()

    landing = Path(args.landing)
    archive = Path(args.archive)
    corpus = Path(args.corpus)

    landing.mkdir(parents=True, exist_ok=True)
    archive.mkdir(parents=True, exist_ok=True)
    corpus.mkdir(parents=True, exist_ok=True)

    print(f"Continuous ingestion pipeline")
    print(f"  Landing:  {landing}")
    print(f"  Archive:  {archive}")
    print(f"  Corpus:   {corpus}")
    print(f"  Enrichment: {'disabled' if args.no_enrichment else args.ollama_model}")
    print(f"  LanceDB:    {'disabled' if args.no_lancedb else 'enabled'}")
    if args.lancedb_incremental:
        print(f"  LanceDB incremental: enabled (batch={args.lancedb_batch_size})")
    print()

    # Clean up .pending_delete files from previous runs
    cleanup_pending_delete(landing)

    # Initialize pipeline components
    trace = TraceLog(args.trace_db) if args.trace_db else None
    runner = FastPathRunner(
        landing_db=corpus / "landing.db",
        store_db=corpus / "document_store.db",
        index_db=corpus / "tantivy_index",
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        landing_root=landing,
        trace_log=trace,
        index_backend="tantivy",
        chunker=args.chunker,
        chunker_threshold=args.chunker_threshold,
        chunker_min_size=args.chunker_min_size,
        chunker_max_size=args.chunker_max_size,
    )

    # Initialize LanceDB incremental mode if requested
    lancedb_index = None
    lancedb_embedding = None
    lancedb_path = corpus / "vector" / "lancedb"
    if args.lancedb_incremental and not args.no_lancedb:
        lancedb_path.parent.mkdir(parents=True, exist_ok=True)
        # Don't delete existing â€” we're doing incremental
        lancedb_embedding = EmbeddingAdapter(
            batch_size=args.lancedb_batch_size, max_length=512,
        )
        from ipa.indexes.lancedb_index import LanceDBIndex
        lancedb_index = LanceDBIndex(lancedb_path, vector_dim=lancedb_embedding.dimension)
        print(f"  LanceDB incremental index initialized at {lancedb_path}")
        print(f"  Embedding: {lancedb_embedding.model_name}, hybrid mode, "
              f"batch={args.lancedb_batch_size}")

    # Track processed files (by content hash to avoid reprocessing)
    processed_files: set[str] = set()

    # Load already-processed from landing.db if it exists
    import sqlite3
    landing_db = corpus / "landing.db"
    if landing_db.exists():
        conn = sqlite3.connect(str(landing_db))
        c = conn.cursor()
        try:
            c.execute("SELECT source_uri FROM artifacts")
            for row in c.fetchall():
                # A landing.db record is only considered processed when the
                # source is no longer physically in Landing. A previous run
                # may have registered a file before failing to archive it;
                # such a file must remain eligible for retry.
                source_path = Path(row[0])
                if not source_path.exists():
                    processed_files.add(row[0])
        except sqlite3.OperationalError:
            pass
        conn.close()

    total_processed = 0
    total_chunks = 0
    last_activity = time.monotonic()

    # --- Backlog embedding: embed chunks already in store but not in LanceDB ---
    if lancedb_index and lancedb_embedding:
        import sqlite3 as _sqlite3
        _conn = _sqlite3.connect(str(corpus / "document_store.db"))
        _conn.execute("PRAGMA journal_mode=WAL")
        _c = _conn.cursor()
        try:
            _existing = set()
            try:
                _existing = set(r[0] for r in lancedb_index._table.to_table().to_pydict()["chunk_id"])
            except Exception:
                pass
            _c.execute("SELECT chunk_id, document_id, text, content_hash FROM chunks")
            _all = _c.fetchall()
            _backlog = [r for r in _all if r[0] not in _existing]
            _conn.close()
            if _backlog:
                print(f"  Backlog: {len(_backlog)} chunks to embed (from previous runs)...")
                _bs = args.lancedb_batch_size
                for _i in range(0, len(_backlog), _bs):
                    _batch = _backlog[_i:_i + _bs]
                    _chunks = [DocumentChunk(chunk_id=r[0], document_id=r[1], text=r[2], content_hash=r[3]) for r in _batch]
                    _texts = [c.text for c in _chunks]
                    _vectors, _sparse = lancedb_embedding.embed_texts_hybrid(_texts)
                    lancedb_index.add_chunks(_chunks, _vectors, sparse_weights=_sparse)
                    _done = _i + len(_batch)
                    print(f"    [backlog] {_done}/{len(_backlog)} â†’ {lancedb_index._table.count_rows()} rows", flush=True)
                print(f"  Backlog complete: {lancedb_index._table.count_rows()} total rows")
        except Exception as e:
            print(f"  Backlog ERROR: {e}")

    print("Watching Landing/ for new files...")
    print(f"  (idle timeout: {args.idle_timeout}s)")
    print()

    try:
        while True:
            # Scan Landing for new files
            new_files = []
            for f in landing.rglob("*"):
                if f.is_file() and f.suffix in {".txt", ".pdf", ".html", ".json",
                                                 ".docx", ".doc", ".pptx", ".xlsx",
                                                 ".csv", ".md", ".rst", ".rtf",
                                                 ".odt", ".epub"}:
                    abs_path = str(f.resolve())
                    if abs_path not in processed_files:
                        new_files.append(f)

            if new_files:
                last_activity = time.monotonic()
                # Sort: fast files (.txt, .html, .json, .csv) first,
                # slow files (.pdf) last â€” prevents fast files being
                # blocked behind slow PDF parsing.
                FAST_EXT = {".txt", ".html", ".htm", ".json", ".csv", ".md", ".rst"}
                new_files.sort(key=lambda f: (0 if f.suffix.lower() in FAST_EXT else 1, f.name))

                print(f"[{time.strftime('%H:%M:%S')}] Found {len(new_files)} new file(s) "
                      f"({sum(1 for f in new_files if f.suffix.lower() in FAST_EXT)} fast, "
                      f"{sum(1 for f in new_files if f.suffix.lower() not in FAST_EXT)} slow)")

                for filepath in new_files:
                    print(f"  Processing: {filepath.name}")
                    result = process_file(runner, filepath)

                    if result and not result["errors"]:
                        processed_files.add(str(filepath.resolve()))
                        total_processed += 1
                        total_chunks += result["chunks_created"]
                        print(f"    OK: {result['mime_type']} â†’ {result['chunks_created']} chunks, "
                              f"{result['pages']} pages, {result['elapsed_seconds']}s")

                        # Move to Archive
                        move_to_archive(filepath, archive)
                        print(f"    â†’ Archived to {archive.name}/")
                    elif result and result["errors"] and _is_lock_error(result["errors"][0]):
                        # IO/permission error (Tantivy lock, Windows file lock) â€”
                        # retry with exponential backoff: 2s, 4s, 8s
                        err_str = str(result["errors"][0])
                        for attempt, wait in enumerate([2, 4, 8], 1):
                            print(f"    Lock/IO error (attempt {attempt}/3), retrying in {wait}s...")
                            time.sleep(wait)
                            result = process_file(runner, filepath)
                            if result and not result["errors"]:
                                processed_files.add(str(filepath.resolve()))
                                total_processed += 1
                                total_chunks += result["chunks_created"]
                                print(f"    OK (retry {attempt}): {result['mime_type']} â†’ {result['chunks_created']} chunks")
                                move_to_archive(filepath, archive)
                                result = None  # mark as handled
                                break
                        if result is not None:
                            # All retries failed â€” leave in Landing/ for next cycle
                            err = result["errors"][0] if result and result["errors"] else "unknown"
                            print(f"    ERROR (all retries failed): {err}")
                            print(f"    âš  Left in Landing/ for retry next cycle")
                            # Do NOT add to processed_files â€” will be retried
                    else:
                        err = result["errors"][0] if result and result["errors"] else "unknown"
                        print(f"    ERROR: {err}")
                        print(f"    âš  Left in Landing/ for retry next cycle")
                        # Do NOT add to processed_files, do NOT move to Archive
                        # The file stays in Landing/ and will be retried next cycle

                    time.sleep(0.01)  # minimal delay between files

                # --- Incremental LanceDB update (only new chunks from this file) ---
                if lancedb_index and lancedb_embedding and result and not result["errors"]:
                    import sqlite3 as _sqlite3
                    _conn = _sqlite3.connect(str(corpus / "document_store.db"))
                    _conn.execute("PRAGMA journal_mode=WAL")
                    _c = _conn.cursor()
                    try:
                        # Get chunk_ids for the document just processed
                        _c.execute("SELECT chunk_id, document_id, text, content_hash FROM chunks WHERE document_id = ?", (result["document_id"],))
                        _doc_chunks = _c.fetchall()
                        _conn.close()
                        if _doc_chunks:
                            # Check which ones are already in LanceDB
                            _existing = set()
                            try:
                                _existing = set(r[0] for r in lancedb_index._table.to_table().to_pydict()["chunk_id"])
                            except Exception:
                                pass
                            _new = [r for r in _doc_chunks if r[0] not in _existing]
                            if _new:
                                _batch_size = args.lancedb_batch_size
                                for _i in range(0, len(_new), _batch_size):
                                    _batch = _new[_i:_i + _batch_size]
                                    _chunks = [DocumentChunk(chunk_id=r[0], document_id=r[1], text=r[2], content_hash=r[3]) for r in _batch]
                                    _texts = [c.text for c in _chunks]
                                    _vectors, _sparse = lancedb_embedding.embed_texts_hybrid(_texts)
                                    lancedb_index.add_chunks(_chunks, _vectors, sparse_weights=_sparse)
                                print(f"    [LanceDB incremental] +{len(_new)} chunks â†’ {lancedb_index._table.count_rows()} total rows")
                    except Exception as e:
                        print(f"    [LanceDB incremental] ERROR: {e}")
            else:
                # No new files â€” check idle timeout
                idle = time.monotonic() - last_activity
                if idle > args.idle_timeout:
                    print(f"\n[{time.strftime('%H:%M:%S')}] Idle for {idle:.0f}s â€” no new files.")
                    break
                time.sleep(args.poll_interval)

    except KeyboardInterrupt:
        print("\nInterrupted by user â€” shutting down gracefully...")

        # Close runner (releases Tantivy writer lock)
        try:
            runner.close()
            print("  Runner closed (Tantivy writer released).")
        except Exception as e:
            print(f"  Runner close error: {e}")

        # Close incremental LanceDB if active
        if lancedb_index:
            try:
                lancedb_index.create_fts_index()
                lancedb_index.close()
                print("  LanceDB incremental: FTS index created, closed.")
            except Exception as e:
                print(f"  LanceDB incremental close error: {e}")
        if lancedb_embedding:
            try:
                lancedb_embedding.close()
                print("  Embedding adapter closed.")
            except Exception as e:
                print(f"  Embedding close error: {e}")

        print("Graceful shutdown complete. No post-processing will run.")
        sys.exit(0)

    # Close runner (normal exit â€” all files processed or idle timeout)
    runner.close()

    # Close incremental LanceDB if active
    if lancedb_index:
        try:
            lancedb_index.create_fts_index()
            lancedb_index.close()
            print(f"\n  LanceDB incremental: FTS index created, closed.")
        except Exception as e:
            print(f"\n  LanceDB incremental close error: {e}")
    if lancedb_embedding:
        lancedb_embedding.close()

    # --- Post-processing: LanceDB + Enrichment ---
    print(f"\n{'='*60}")
    print(f"Fast path complete: {total_processed} files, {total_chunks} chunks")
    print(f"{'='*60}")

    # Build LanceDB (skip if incremental mode already built it)
    if not args.no_lancedb and total_chunks > 0 and not args.lancedb_incremental:
        print("\n--- Building LanceDB index ---")
        lancedb_path = corpus / "vector" / "lancedb"
        lancedb_path.parent.mkdir(parents=True, exist_ok=True)

        # Remove old LanceDB if exists (rebuild from scratch)
        if lancedb_path.exists():
            import shutil as sh
            sh.rmtree(lancedb_path)

        embedding = EmbeddingAdapter()
        total, embedded = build_lancedb(
            corpus / "document_store.db",
            lancedb_path,
            embedding,
        )
        print(f"  LanceDB: {embedded}/{total} chunks embedded")
        embedding.close()

    # Selective enrichment
    if not args.no_enrichment and total_chunks > 0:
        print(f"\n--- Selective enrichment (Qwen3.5, min {args.min_chars_summarize} chars) ---")
        llm = OllamaAdapter(
            model=args.ollama_model,
            base_url=args.ollama_url,
            think=False,
        )
        total, enriched = enrich_selective(
            corpus / "document_store.db",
            llm,
            min_chars=args.min_chars_summarize,
        )
        print(f"  Enriched: {enriched}/{total} chunks (selective by density/size)")

        # Rebuild LanceDB with enriched text
        if not args.no_lancedb:
            print("\n--- Rebuilding LanceDB with enriched text ---")
            lancedb_path = corpus / "vector" / "lancedb"
            if lancedb_path.exists():
                import shutil as sh
                sh.rmtree(lancedb_path)
            embedding = EmbeddingAdapter()
            total, embedded = build_lancedb(
                corpus / "document_store.db",
                lancedb_path,
                embedding,
            )
            print(f"  LanceDB rebuilt: {embedded}/{total} chunks")
            embedding.close()

    # Final summary
    print(f"\n{'='*60}")
    print(f"Pipeline complete!")
    print(f"  Files processed: {total_processed}")
    print(f"  Total chunks:    {total_chunks}")
    print(f"  Landing:         {landing} ({sum(1 for _ in landing.rglob('*') if _.is_file())} files remaining)")
    print(f"  Archive:         {archive} ({sum(1 for _ in archive.rglob('*') if _.is_file())} files archived)")
    print(f"  Corpus DBs:      {corpus}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
