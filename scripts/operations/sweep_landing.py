"""Sweep Landing/ — enforce the transit-zone policy.

Processed+approved artifacts (in the main corpus) move to Archive/;
processed+pending artifacts (in a staging corpus, promotion queue, or awaiting
human review) move to Transit/; processed+rejected artifacts (status failed or
review_status=rejected) are deleted; unprocessed files stay in Landing/.
Transit/ is re-scanned each run: promoted → Archive, rejected → delete.
See docs/operations/landing-and-archive.md.

Usage:
    .venv/Scripts/python.exe scripts/operations/sweep_landing.py [--dry-run]
        [--landing Landing] [--archive Archive] [--transit Transit]
        [--main-corpus DIR] [--corpus DIR ...] [--cluster-db PATH]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ipa.ingestion.landing_sweep import (  # noqa: E402
    _STATUS_RANK,
    _live_docs,
    _load_artifacts,
    _review_states,
    _under,
    sweep_landing,
)

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Sweep Landing → Archive / Transit / delete rejected")
    ap.add_argument("--landing", default=str(ROOT / "Landing"))
    ap.add_argument("--archive", default=str(ROOT / "Archive"))
    ap.add_argument("--transit", default=str(ROOT / "Transit"))
    ap.add_argument("--main-corpus", default=str(ROOT / "outputs" / "experiments" / "E12-corpus"))
    ap.add_argument("--cluster-db", default=str(ROOT / "outputs" / "agent" / "topic_clusters.db"))
    ap.add_argument("--reporter-db", default=None,
                    help="reporter.db with human review decisions. "
                         "Default: latest quality-check run.")
    ap.add_argument(
        "--corpus", action="append", default=None,
        help="Corpus dir with landing.db + document_store.db (repeatable). "
             "Default: E12-corpus + latest reporter corpus.",
    )
    ap.add_argument("--dry-run", action="store_true", help="classify only, no moves/deletes")
    args = ap.parse_args()

    main_corpus = Path(args.main_corpus)
    corpora = [Path(c) for c in args.corpus] if args.corpus else [
        main_corpus,
    ]
    reporter_db = Path(args.reporter_db) if args.reporter_db else None
    if not args.corpus:
        reporter_qc = ROOT / "outputs" / "reporter" / "quality-check"
        if reporter_qc.exists():
            candidates = sorted(
                (d / "corpus" for d in reporter_qc.iterdir() if (d / "corpus" / "landing.db").exists()),
                key=lambda p: p.stat().st_mtime, reverse=True,
            )
            corpora.extend(candidates[:1])
            if reporter_db is None and candidates:
                candidate_db = candidates[0].parent / "reporter.db"
                if candidate_db.exists():
                    reporter_db = candidate_db

    if args.dry_run:
        landing = Path(args.landing)
        artifacts: dict[str, dict[str, str]] = {}
        main_ids: set[str] = set()
        staged_ids: set[str] = set()
        doc_to_artifact: dict[str, str] = {}
        for c in corpora:
            for aid, info in _load_artifacts(c / "landing.db").items():
                prev = artifacts.get(aid)
                if prev is None or _STATUS_RANK.get(info["status"], -1) > _STATUS_RANK.get(prev["status"], -1):
                    artifacts[aid] = info
            docs = _live_docs(c / "document_store.db")
            for aid, doc_id in docs.items():
                doc_to_artifact[doc_id] = aid
            if c.resolve() == main_corpus.resolve():
                main_ids |= set(docs)
            else:
                staged_ids |= set(docs)
        rejected_docs, pending_docs = _review_states(Path(args.cluster_db), reporter_db)
        rejected = {doc_to_artifact[d] for d in rejected_docs if d in doc_to_artifact}
        pending = {doc_to_artifact[d] for d in pending_docs if d in doc_to_artifact}
        pending |= staged_ids - main_ids
        arch = tran = dele = skip = gone = outside = 0
        for aid, info in artifacts.items():
            p = Path(info["source_uri"])
            if not p.exists() or not p.is_file():
                gone += 1
                continue
            if not _under(p, landing) or p.suffix == ".db" or p.name.startswith("."):
                outside += 1
                continue
            if aid in main_ids:
                arch += 1
            elif aid in rejected or info["status"] == "failed":
                dele += 1
            elif aid in pending or info["status"] == "indexed":
                tran += 1
            else:
                skip += 1
        # Physical files under Landing not matched by source_uri
        unreg = 0
        swept = {Path(i["source_uri"]).resolve() for i in artifacts.values() if i["source_uri"]}
        for f in landing.rglob("*"):
            if f.is_file() and not f.name.startswith(".") and f.suffix != ".db" \
                    and f.suffix != ".pending_delete" and f.resolve() not in swept:
                unreg += 1
        print(f"dry-run: {arch} a Archive, {tran} a Transit, {dele} a eliminar, "
              f"{skip} sin procesar, {gone} sin archivo, {outside} fuera de Landing, "
              f"{unreg} archivos sin registro")
        return 0

    stats = sweep_landing(
        args.landing, args.archive, corpora,
        main_corpus=args.main_corpus,
        transit_root=args.transit,
        cluster_db=args.cluster_db,
        reporter_db=reporter_db,
    )
    print(f"sweep: {stats['archived']} a Archive, {stats['transit']} a Transit, "
          f"{stats['deleted']} eliminados, {stats['skipped']} sin procesar, "
          f"{stats['pending_cleaned']} pending_delete limpiados")
    for err in stats["errors"]:
        print(f"  error: {err}", file=sys.stderr)
    return 1 if stats["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
