"""Dedupe a corpus by canonical URL — keep the most complete doc per URL.

Default is dry-run: prints the full plan (which doc survives, which get
tombstoned) without touching anything. ``--apply`` backs up the SQLite
stores first, then tombstones the losers across DocumentStore, BM25 and
LanceDB. Reversible for the store (tombstone), rebuildable for indexes.

Usage:
    .venv/Scripts/python.exe scripts/operations/dedupe_corpus_by_url.py
        --corpus outputs/experiments/E12-corpus            # dry-run
        --corpus outputs/experiments/E12-corpus --apply    # real run
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ipa.agentic.corpus_dedupe import dedupe_by_url


def _backup(path: Path) -> Path | None:
    if not path.exists():
        return None
    dst = path.with_name(f"{path.name}.bak-dedupe-{time.strftime('%Y%m%d-%H%M%S')}")
    shutil.copy2(path, dst)
    return dst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--apply", action="store_true", help="apply (default: dry-run)")
    args = ap.parse_args()

    corpus = Path(args.corpus)
    if not (corpus / "document_store.db").exists():
        sys.exit(f"no document_store.db in {corpus}")

    if args.apply:
        for name in ("document_store.db", "bm25_index.db"):
            bak = _backup(corpus / name)
            if bak:
                print(f"backup: {bak}")

    print("scanning duplicate URL groups ...", flush=True)
    report = dedupe_by_url(corpus, dry_run=not args.apply)

    print(f"\ngroups: {report['groups']}  "
          f"docs->tombstone: {report['tombstone_docs']}  "
          f"chunks->tombstone: {report['tombstone_chunks']}  "
          f"kept: {report['kept_docs']}")
    for p in report["plan"]:
        print(f"  {p['url'][:90]}")
        print(f"    keep {p['keep']} ({p['keep_chunks']} chunks) — "
              f"drop {len(p['drop'])} docs / {p['drop_chunks']} chunks")
    if report.get("lance_error"):
        print(f"WARNING lance: {report['lance_error']}")
    print("\nDRY-RUN" if not args.apply else "\nAPPLIED")


if __name__ == "__main__":
    main()
