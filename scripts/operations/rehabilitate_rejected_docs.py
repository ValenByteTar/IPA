"""Rehabilitate novelty-gate false positives (PM-004-adjacent incident).

Documents rejected by the embedding-only novelty gate (cosine > 0.95 against a
main-corpus document) can be false positives: recurring site templates (weekly
CVE alerts, interview spotlights) make distinct articles measure ~identical at
the doc-embedding level. The hardened gate now requires lexical overlap >=0.85
to confirm a duplicate, but documents already rejected stay tombstoned in their
staging corpus with their curation decision marked rejected.

This script (idempotent, dry-run by default):

  1. Selects curation decisions with review_status='rejected' whose reason is
     the near-identical novelty verdict ("...corpus histórico").
  2. Excludes documents already covered in main (same doc_id or same source
     URL in main document_sources) — those rejections were legitimate.
  3. Locates each remaining document across reporter corpus stores.
  4. Un-tombstones the document and its chunks in the corpus DocumentStore,
     clears stale embedding_jobs rows, and restores the BM25 index
     (chunks_meta un-tombstone + chunks_fts re-insert via add_chunks).
     LanceDB vectors are NOT restored here — the embed drain rediscovers
     live chunks missing vectors and re-embeds them (run_embed_drain.py).
  5. Resets the curation payload to review_status='pending' with a
     `rehabilitated` flag and re-queues the document for promotion with the
     correct source_corpus. The vector-coverage preflight in
     promotion_executor guarantees no purge happens before vectors land.

Usage:
    .venv/Scripts/python.exe scripts/operations/rehabilitate_rejected_docs.py            # dry-run
    .venv/Scripts/python.exe scripts/operations/rehabilitate_rejected_docs.py --apply
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

CLUSTER_DB = ROOT / "outputs" / "agent" / "topic_clusters.db"
MAIN_CORPUS = ROOT / "outputs" / "experiments" / "E12-corpus"
REHAB_REASON = "rehabilitated: novelty-gate false positive (template similarity)"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=120)
    conn.execute("PRAGMA busy_timeout = 120000")
    return conn


def _backup(db_path: Path, backup_dir: Path) -> Path:
    """Consistent backup via the SQLite backup API (WAL-safe)."""
    backup_dir.mkdir(parents=True, exist_ok=True)
    dest = backup_dir / db_path.name
    src = _connect(db_path)
    dst = sqlite3.connect(str(dest))
    try:
        src.backup(dst)
    finally:
        src.close()
        dst.close()
    return dest


def _chunks_of(values: set[str], size: int = 900) -> list[list[str]]:
    ids = sorted(values)
    return [ids[i:i + size] for i in range(0, len(ids), size)]


def load_risky_rejections(cluster_db: Path) -> list[dict]:
    """Rejected decisions from the near-identical novelty verdict."""
    conn = _connect(cluster_db)
    try:
        rows = conn.execute(
            "SELECT document_id, payload_json FROM curation_decisions"
        ).fetchall()
    finally:
        conn.close()
    out = []
    for doc_id, payload_json in rows:
        try:
            payload = json.loads(payload_json)
        except (TypeError, ValueError):
            continue
        if payload.get("review_status") != "rejected":
            continue
        if payload.get("decision") != "duplicate":
            continue
        if "corpus hist" not in str(payload.get("reason", "")):
            continue
        out.append({"document_id": doc_id, "payload": payload})
    return out


def main_coverage(main_db: Path) -> tuple[set[str], set[str]]:
    """(live doc_ids, source_urls) present in the main corpus."""
    conn = _connect(main_db)
    try:
        doc_ids = {
            r[0] for r in conn.execute(
                "SELECT document_id FROM documents WHERE tombstoned = 0"
            )
        }
        urls = {
            r[0] for r in conn.execute(
                "SELECT source_url FROM document_sources WHERE source_url IS NOT NULL"
            )
        }
        return doc_ids, urls
    finally:
        conn.close()


def scan_corpus(corpus: Path, wanted: set[str]) -> dict:
    """One pass over a corpus store: which wanted doc_ids it holds, plus the
    URL/provenance map for those docs."""
    info: dict = {"doc_ids": set(), "urls": {}, "provenance": {}}
    store_db = corpus / "document_store.db"
    conn = _connect(store_db)
    try:
        for group in _chunks_of(wanted):
            ph = ",".join("?" * len(group))
            info["doc_ids"].update(
                r[0] for r in conn.execute(
                    f"SELECT document_id FROM documents WHERE document_id IN ({ph})",
                    group)
            )
        if not info["doc_ids"]:
            return info
        try:
            for group in _chunks_of(info["doc_ids"]):
                ph = ",".join("?" * len(group))
                for d, u, p in conn.execute(
                        f"SELECT document_id, source_url, provenance FROM document_sources "
                        f"WHERE document_id IN ({ph})", group):
                    if u:
                        info["urls"][d] = u
                    info["provenance"][d] = p or "configured_scrape"
        except sqlite3.OperationalError:
            pass
        return info
    finally:
        conn.close()


def fetch_chunks(store_db: Path, doc_ids: set[str]) -> list:
    """Live chunks of the given docs, as DocumentChunk (for BM25 add_chunks)."""
    from ipa.contracts import DocumentChunk
    from ipa.storage.document_store import _dict_to_span

    conn = _connect(store_db)
    try:
        chunks = []
        for group in _chunks_of(doc_ids):
            ph = ",".join("?" * len(group))
            for cid, did, chash, text, meta, span in conn.execute(
                    "SELECT chunk_id, document_id, content_hash, text, metadata_json, span_json "
                    f"FROM chunks WHERE document_id IN ({ph}) AND tombstoned = 0", group):
                chunks.append(DocumentChunk(
                    chunk_id=cid, document_id=did, content_hash=chash,
                    text=text, metadata=json.loads(meta) if meta else {},
                    source_span=_dict_to_span(json.loads(span) if span else None),
                ))
        return chunks
    finally:
        conn.close()


def untombstone_corpus(corpus: Path, doc_ids: set[str], *, apply: bool) -> dict[str, int]:
    """Un-tombstone docs+chunks and clear stale embedding_jobs rows."""
    stats = {"docs_restored": 0, "chunks_restored": 0}
    store_db = corpus / "document_store.db"

    conn = _connect(store_db)
    try:
        for group in _chunks_of(doc_ids):
            ph = ",".join("?" * len(group))
            stats["docs_restored"] += conn.execute(
                f"SELECT COUNT(*) FROM documents WHERE document_id IN ({ph}) AND tombstoned = 1",
                group).fetchone()[0]
            stats["chunks_restored"] += conn.execute(
                f"SELECT COUNT(*) FROM chunks WHERE document_id IN ({ph}) AND tombstoned = 1",
                group).fetchone()[0]
        if not apply:
            return stats
        conn.execute("BEGIN")
        for group in _chunks_of(doc_ids):
            ph = ",".join("?" * len(group))
            conn.execute(
                f"UPDATE documents SET tombstoned = 0 WHERE document_id IN ({ph})", group)
            conn.execute(
                f"UPDATE chunks SET tombstoned = 0 WHERE document_id IN ({ph})", group)
            conn.execute(
                f"DELETE FROM embedding_jobs WHERE chunk_id IN "
                f"(SELECT chunk_id FROM chunks WHERE document_id IN ({ph}))", group)
        conn.execute("COMMIT")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return stats


def restore_bm25(corpus: Path, doc_ids: set[str], *, wait_seconds: float = 900) -> int:
    """Re-insert FTS rows + clear meta tombstones for the given docs.

    The live fast-path ingestion holds the bm25 writer lock for stretches;
    poll BEGIN IMMEDIATE until it yields (bounded by wait_seconds) instead of
    failing the whole rehab. Returns rows restored; -1 if the lock never freed.
    """
    import time as _time

    from ipa import BM25Index

    bm25_db = corpus / "bm25_index.db"
    if not bm25_db.exists():
        return 0
    chunks = fetch_chunks(corpus / "document_store.db", doc_ids)
    if not chunks:
        return 0

    deadline = _time.monotonic() + wait_seconds
    while True:
        bm25 = BM25Index(bm25_db)
        try:
            bm25._conn.execute("PRAGMA busy_timeout = 30000")
            try:
                bm25._conn.execute("BEGIN IMMEDIATE")
            except sqlite3.OperationalError:
                if _time.monotonic() >= deadline:
                    return -1
                bm25.close()
                _time.sleep(15)
                continue
            try:
                bm25.add_chunks(chunks, commit=False)
            except Exception:
                bm25._conn.execute("ROLLBACK")
                raise
            bm25._conn.execute("COMMIT")
            return len(chunks)
        finally:
            try:
                bm25.close()
            except Exception:
                pass


def rehabilitate(apply: bool) -> None:
    print("phase 1: reading curation decisions...", flush=True)
    risky = load_risky_rejections(CLUSTER_DB)
    print(f"  rejected near-identical decisions: {len(risky)}", flush=True)

    print("phase 2: main coverage...", flush=True)
    main_doc_ids, main_urls = main_coverage(MAIN_CORPUS / "document_store.db")

    wanted = {i["document_id"] for i in risky} - main_doc_ids
    print(f"  without main doc coverage: {len(wanted)}", flush=True)

    print("phase 3: locating docs across corpus stores...", flush=True)
    corpus_roots = sorted(
        p.parent for p in (ROOT / "outputs" / "reporter").rglob("corpus/document_store.db")
    )
    corpus_roots += sorted(
        p.parent for p in (ROOT / "outputs" / "experiments").glob("*/document_store.db")
        if p.parent != MAIN_CORPUS
    )

    final: dict[Path, set[str]] = {}
    provenance: dict[str, str] = {}
    skipped_covered = skipped_missing = 0
    located_total = 0
    for corpus in corpus_roots:
        info = scan_corpus(corpus, wanted)
        if not info["doc_ids"]:
            continue
        located_total += len(info["doc_ids"])
        keep: set[str] = set()
        for doc_id in info["doc_ids"]:
            url = info["urls"].get(doc_id)
            if url and url in main_urls:
                skipped_covered += 1
                continue
            keep.add(doc_id)
            provenance[doc_id] = info["provenance"].get(doc_id, "configured_scrape")
        if keep:
            final[corpus] = keep
            print(f"  {corpus.relative_to(ROOT)}: {len(keep)} docs", flush=True)

    skipped_missing = len(wanted) - located_total
    print(f"  skipped (url already in main): {skipped_covered}", flush=True)
    print(f"  skipped (no store row found):  {skipped_missing}", flush=True)
    print(f"  to rehabilitate: {sum(len(s) for s in final.values())}", flush=True)

    if not apply:
        print("\nDRY RUN — pass --apply to execute.")
        return

    # Backups before any mutation.
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_dir = ROOT / "outputs" / "agent" / f"rehab_backup_{stamp}"
    print("phase 4: backups...", flush=True)
    _backup(CLUSTER_DB, backup_dir)
    for corpus in final:
        sub = backup_dir / corpus.relative_to(ROOT)
        _backup(corpus / "document_store.db", sub)
        bm25_db = corpus / "bm25_index.db"
        if bm25_db.exists():
            _backup(bm25_db, sub)
    print(f"  backups -> {backup_dir.relative_to(ROOT)}", flush=True)

    print("phase 5: un-tombstoning docs/chunks...", flush=True)
    for corpus, ids in final.items():
        stats = untombstone_corpus(corpus, ids, apply=True)
        print(f"  {corpus.name}: {stats['docs_restored']} docs / "
              f"{stats['chunks_restored']} chunks", flush=True)

    print("phase 6: curation payloads + promotion queue...", flush=True)
    conn = _connect(CLUSTER_DB)
    conn.isolation_level = None
    try:
        conn.execute("BEGIN")
        queued = 0
        for corpus, ids in final.items():
            for doc_id in ids:
                row = conn.execute(
                    "SELECT payload_json FROM curation_decisions WHERE document_id = ?",
                    (doc_id,)).fetchone()
                if row:
                    payload = json.loads(row[0])
                    payload["review_status"] = "pending"
                    payload["rehabilitated"] = True
                    payload["rehab_note"] = REHAB_REASON
                    payload["rehabilitated_at"] = _now()
                    conn.execute(
                        "UPDATE curation_decisions SET payload_json = ? WHERE document_id = ?",
                        (json.dumps(payload, ensure_ascii=False), doc_id))
                conn.execute(
                    "INSERT OR REPLACE INTO promotion_queue "
                    "(document_id, reason, provenance, source_corpus, status, queued_at, promoted_at) "
                    "VALUES (?, ?, ?, ?, 'pending', ?, NULL)",
                    (doc_id, REHAB_REASON,
                     provenance.get(doc_id, "configured_scrape"),
                     str(corpus), _now()))
                queued += 1
        conn.execute("COMMIT")
        print(f"  payloads reset + queued: {queued}", flush=True)
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()

    print("phase 7: BM25 restore (waits for the writer lock)...", flush=True)
    pending_bm25: list[Path] = []
    for corpus, ids in final.items():
        rows = restore_bm25(corpus, ids)
        if rows < 0:
            pending_bm25.append(corpus)
            print(f"  {corpus.name}: bm25 still locked — rerun to finish", flush=True)
        else:
            print(f"  {corpus.name}: {rows} bm25 rows", flush=True)

    if pending_bm25:
        print("\nBM25 restore pending for locked corpora — rerun --apply once "
              "ingestion finishes (idempotent).")
    print("\nNext: re-embed restored chunks per corpus:")
    for corpus in final:
        print(f"  .venv/Scripts/python.exe scripts/operations/run_embed_drain.py --corpus {corpus}")
    print("The promotion preflight defers the batch until every vector lands.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="Execute mutations (default: dry-run report)")
    args = parser.parse_args()
    rehabilitate(apply=args.apply)


if __name__ == "__main__":
    main()
