"""Hammer the E12 corpus with queries while it's still being built.
Polls every 10s, runs queries, reports hits + corpus size + latency.

Architecture: snapshot-based reading to avoid Windows lock conflicts.
  - At the start of each round, copies the Tantivy index directory to a
    temp location. The pipeline writes to the original; the hammer reads
    from the copy. Zero lock conflicts.
  - SQLite document_store is opened read-only with WAL mode (readers
    don't block writers in SQLite WAL).
  - Tracks per-query latency (p50, p99).
"""
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from ipa.indexes.tantivy_index import TantivyIndex

corpus = Path("outputs/experiments/E12-corpus")
tantivy_path = corpus / "tantivy_index"
lancedb_path = corpus / "vector" / "lancedb"

QUERIES = [
    "Claude AI model capabilities",
    "cybersecurity threat intelligence report",
    "Qwen inference performance GPU",
    "DeepSeek training mixture of experts",
    "OpenAI safety research",
    "AI agent architecture",
    "language model evaluation benchmark",
    "neural network optimization",
    "reinforcement learning",
    "transformer architecture attention",
    "protein design biology",
    "robotics navigation policy",
    "cryptographic weaknesses",
    "multi-agent systems",
    "global workspace theory",
    "Riemann zeta mathematics",
    "responsible scaling policy",
    "watermarking text detection",
    "GPU memory optimization",
    "speculative decoding",
]


class CorpusQueryEngine:
    """Query engine that reads from snapshots â€” no lock conflicts with pipeline."""

    def __init__(self):
        self.tantivy = None
        self.tantivy_snapshot = None
        self.store_conn = None
        self.lancedb = None
        self.embedding = None
        self._open_store()

    def _open_store(self):
        """Open document_store in read-only mode. SQLite WAL allows
        concurrent readers + writers without blocking."""
        store = corpus / "document_store.db"
        if store.exists():
            try:
                # URI mode read-only (NOT immutable â€” we want to see new data)
                uri = f"file:{store.resolve()}?mode=ro"
                self.store_conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
            except Exception:
                try:
                    self.store_conn = sqlite3.connect(str(store), check_same_thread=False)
                except Exception:
                    self.store_conn = None

    def refresh_store(self):
        """Reopen store connection to see new data committed by pipeline."""
        if self.store_conn:
            try:
                self.store_conn.close()
            except Exception:
                pass
        self._open_store()

    def snapshot_tantivy(self):
        """Copy the Tantivy index to a temp dir and open it read-only.
        This avoids all Windows lock conflicts with the pipeline writer."""
        # Close previous snapshot if any
        if self.tantivy:
            try:
                self.tantivy.close()
            except Exception:
                pass
            self.tantivy = None
        if self.tantivy_snapshot and self.tantivy_snapshot.exists():
            try:
                shutil.rmtree(self.tantivy_snapshot)
            except Exception:
                pass

        if not tantivy_path.exists():
            return False

        # Copy to temp dir
        tmpdir = Path(tempfile.mkdtemp(prefix="tantivy_snap_"))
        try:
            shutil.copytree(tantivy_path, tmpdir / "index", dirs_exist_ok=True)
        except Exception:
            # Pipeline might be mid-commit â€” try once more
            time.sleep(0.5)
            try:
                shutil.copytree(tantivy_path, tmpdir / "index", dirs_exist_ok=True)
            except Exception:
                shutil.rmtree(tmpdir, ignore_errors=True)
                return False

        self.tantivy_snapshot = tmpdir
        try:
            self.tantivy = TantivyIndex(tmpdir / "index", read_only=True)
            return True
        except Exception:
            return False

    def _open_lancedb(self):
        if lancedb_path.exists() and self.lancedb is None:
            try:
                from ipa.indexes.lancedb_index import LanceDBIndex
                self.lancedb = LanceDBIndex(lancedb_path)
            except Exception:
                self.lancedb = None

    def get_chunk_text(self, chunk_id: str) -> str:
        if not self.store_conn:
            return ""
        try:
            c = self.store_conn.cursor()
            c.execute("SELECT text FROM chunks WHERE chunk_id = ?", (chunk_id,))
            row = c.fetchone()
            return row[0] if row else ""
        except Exception:
            return ""

    def query_lexical(self, query: str, limit: int = 3):
        """Query Tantivy snapshot (lexical/BM25). Returns (hits, latency_ms)."""
        if not self.tantivy:
            return [], 0.0
        t0 = time.perf_counter()
        try:
            hits = self.tantivy.search(query, limit=limit)
        except Exception:
            return [], 0.0
        return hits, (time.perf_counter() - t0) * 1000

    def query_semantic(self, query: str, limit: int = 3):
        """Query LanceDB (semantic/embedding). Returns (hits, latency_ms)."""
        self._open_lancedb()
        if not self.lancedb or not self.embedding:
            try:
                from ipa.indexes.embedding_adapter import EmbeddingAdapter
                self.embedding = EmbeddingAdapter()
            except Exception:
                return [], 0.0
        if not self.lancedb:
            return [], 0.0
        t0 = time.perf_counter()
        try:
            vec = self.embedding.embed_query(query)
            hits = self.lancedb.search(vec, limit=limit)
        except Exception:
            return [], 0.0
        return hits, (time.perf_counter() - t0) * 1000

    def get_stats(self):
        if not self.store_conn:
            return 0, 0
        try:
            c = self.store_conn.cursor()
            c.execute("SELECT COUNT(*) FROM documents")
            docs = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM chunks")
            chunks = c.fetchone()[0]
            return docs, chunks
        except Exception:
            return 0, 0

    def close(self):
        if self.tantivy:
            try:
                self.tantivy.close()
            except Exception:
                pass
        if self.tantivy_snapshot and self.tantivy_snapshot.exists():
            shutil.rmtree(self.tantivy_snapshot, ignore_errors=True)
        if self.store_conn:
            self.store_conn.close()
        if self.lancedb:
            try:
                self.lancedb.close()
            except Exception:
                pass
        if self.embedding:
            try:
                self.embedding.close()
            except Exception:
                pass


def main():
    print("HAMMER MODE v3 â€” snapshot reading + zero lock conflicts")
    print("=" * 70)

    engine = CorpusQueryEngine()
    round_num = 0

    while True:
        round_num += 1
        engine.refresh_store()  # reopen to see new data from pipeline
        docs, chunks = engine.get_stats()
        print(f"\n[Round {round_num}] Docs: {docs}, Chunks: {chunks}")
        print("-" * 70)

        if chunks == 0:
            print("  No chunks yet â€” waiting...")
            time.sleep(5)
            continue

        # Snapshot Tantivy index (copy to temp, open read-only)
        snap_t0 = time.perf_counter()
        snap_ok = engine.snapshot_tantivy()
        snap_ms = (time.perf_counter() - snap_t0) * 1000
        if snap_ok:
            print(f"  Tantivy snapshot: {snap_ms:.0f}ms")
        else:
            print(f"  Tantivy snapshot failed ({snap_ms:.0f}ms) â€” using stale/no index")

        latencies = []
        total_hits = 0
        has_semantic = lancedb_path.exists()

        for q in QUERIES:
            # Lexical search
            hits, lat = engine.query_lexical(q, limit=2)
            latencies.append(lat)
            tag = f"lex {lat:5.1f}ms"

            # Semantic search (if LanceDB available)
            if has_semantic:
                sem_hits, sem_lat = engine.query_semantic(q, limit=2)
                tag += f" | sem {sem_lat:5.1f}ms"
                if sem_hits:
                    hits = sem_hits  # prefer semantic when available

            if hits:
                total_hits += len(hits)
                for hit in hits:
                    text = engine.get_chunk_text(hit.chunk_id)
                    snippet = text[:100].replace("\n", " ") if text else "(no text)"
                    score = hit.score if hasattr(hit, 'score') else 0.0
                    print(f"  [{q[:28]:28}] {tag} s={score:.3f} | {snippet}...")
            else:
                print(f"  [{q[:28]:28}] {tag} | no hits")

        # Latency stats
        if latencies:
            p50 = median(latencies)
            p99 = max(latencies)
            print(f"\n  Query latency: p50={p50:.1f}ms p99={p99:.1f}ms | Hits: {total_hits}")

        # Check if processes are still running
        import subprocess
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV"],
            capture_output=True, text=True
        )
        python_count = result.stdout.count("python.exe")
        print(f"  Python processes: {python_count}")

        if python_count <= 1 and docs > 0:
            print("\n  *** Pipeline + scraper finished ***")
            break

        time.sleep(10)

    engine.close()
    print("\n" + "=" * 70)
    docs, chunks = engine.get_stats()
    print(f"Final: {docs} docs, {chunks} chunks")


if __name__ == "__main__":
    main()
