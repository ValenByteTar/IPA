"""Sync LanceDB scalar metadata columns from the canonical DocumentStore.

Backfills source_domain / published_at / provenance / quality_score onto the
derived LanceDB index so pre-filtered hybrid search (where clauses) works on
indexes created before the metadata schema existed.

Usage:
    .venv\\Scripts\\python.exe scripts\\operations\\sync_index_metadata.py \\
        --corpus outputs/experiments/E12-corpus

    # Full resync (overwrite existing values):
    .venv\\Scripts\\python.exe scripts\\operations\\sync_index_metadata.py \\
        --corpus outputs/experiments/E12-corpus --all
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync LanceDB metadata columns.")
    parser.add_argument("--corpus", required=True, help="Corpus dir (contains document_store.db and vector/lancedb)")
    parser.add_argument("--all", action="store_true", help="Resync all docs, not only rows missing metadata")
    args = parser.parse_args()

    corpus = Path(args.corpus)
    store_db = corpus / "document_store.db"
    lance_dir = corpus / "vector" / "lancedb"
    if not store_db.exists():
        print(f"ERROR: {store_db} not found", flush=True)
        sys.exit(1)
    if not lance_dir.exists():
        print(f"ERROR: {lance_dir} not found", flush=True)
        sys.exit(1)

    from ipa.indexes.lancedb_index import LanceDBIndex
    from ipa.storage.document_store import DocumentStore

    lance = LanceDBIndex(lance_dir, vector_dim=1024)
    store = DocumentStore(store_db)
    try:
        if not lance.is_queryable():
            print("LanceDB index is empty — nothing to sync", flush=True)
            return
        t0 = time.monotonic()
        n = lance.sync_doc_metadata(store, only_missing=not args.all)
        print(f"Synced metadata for {n} documents in {time.monotonic() - t0:.1f}s "
              f"({lance.count()} chunks indexed)", flush=True)
    finally:
        store.close()
        lance.close()


if __name__ == "__main__":
    main()
