"""Detect and tombstone boilerplate/spam chunks in a corpus.

A live chunk is flagged when its content_hash appears in >=3 distinct live
documents (site-wide boilerplate) or across >=2 distinct source domains
(identical text on unrelated sites). Default is dry-run: writes the full
candidate list (hash, doc/domain counts, text sample) to a JSON report for
review. ``--apply`` backs up the SQLite stores, then tombstones every
matching chunk across DocumentStore, BM25 and LanceDB — the owning
documents stay live with their remaining chunks.

Usage:
    .venv/Scripts/python.exe scripts/operations/clean_spam_chunks.py
        --corpus outputs/experiments/E12-corpus            # dry-run + report
        --corpus outputs/experiments/E12-corpus --apply
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ipa.agentic.corpus_dedupe import find_spam_chunks, tombstone_chunks


def _backup(path: Path) -> Path | None:
    if not path.exists():
        return None
    dst = path.with_name(f"{path.name}.bak-spam-{time.strftime('%Y%m%d-%H%M%S')}")
    shutil.copy2(path, dst)
    return dst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--apply", action="store_true", help="apply (default: dry-run)")
    ap.add_argument("--report", default=None, help="JSON report path")
    args = ap.parse_args()

    corpus = Path(args.corpus)
    store_db = corpus / "document_store.db"
    if not store_db.exists():
        sys.exit(f"no document_store.db in {corpus}")

    print("scanning cross-document duplicate chunks ...", flush=True)
    candidates = find_spam_chunks(store_db)
    total = sum(c["n_chunks"] for c in candidates)
    print(f"candidate hashes: {len(candidates)}  live chunks: {total}")

    report_path = Path(args.report) if args.report else (
        corpus / f"spam_chunks_{time.strftime('%Y%m%d-%H%M%S')}.json")
    report_path.write_text(
        json.dumps(candidates, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"report: {report_path}")

    for c in sorted(candidates, key=lambda x: -x["n_chunks"])[:20]:
        sample = c["sample"][:80].encode("ascii", "replace").decode("ascii")
        print(f"  docs={c['n_docs']} doms={c['n_domains']} chunks={c['n_chunks']} | {sample}")

    # Cross-domain pairs are NOT spam: identical article text on two domains
    # means syndication or misattributed provenance — tombstoning both would
    # delete real content from the legit doc. They go to the provenance audit.
    cross_domain = [c for c in candidates
                    if c["n_domains"] >= 2 and c["n_docs"] < 3]
    apply_set = [c for c in candidates if c["n_docs"] >= 3]
    if cross_domain:
        xd_path = report_path.with_name(
            report_path.stem + "_crossdomain_review.json")
        xd_path.write_text(json.dumps(
            cross_domain, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"cross-domain (excluded, -> provenance review): "
              f"{len(cross_domain)} hashes -> {xd_path}")

    if not args.apply:
        print("\nDRY-RUN — review the report, then rerun with --apply")
        return

    for name in ("document_store.db", "bm25_index.db"):
        bak = _backup(corpus / name)
        if bak:
            print(f"backup: {bak}")
    result = tombstone_chunks(corpus, [c["content_hash"] for c in apply_set],
                              dry_run=False)
    print(f"\nAPPLIED: {result['tombstone_chunks']} chunks tombstoned")
    if result.get("lance_error"):
        print(f"WARNING lance: {result['lance_error']}")


if __name__ == "__main__":
    main()
