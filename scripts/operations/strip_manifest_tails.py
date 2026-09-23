"""Strip the "Linked documents downloaded to Landing zone" tail from stored docs.

Bug: web_scraper._save_text() appended a manifest of downloaded linked files
with ABSOLUTE LOCAL PATHS into the artifact's text. That text was ingested,
so ~1.1k docs carry a tail of machine paths — and BM25 ranks those chunks
first when the path happens to contain query terms (observed 2026-09-23:
"Jev LLM architecture" returned chunks of "…\20260923t170315875100-
jev-llm-architecture\arxiv_*.pdf" and the final answer reported "no data").

The writer was fixed to emit basenames only; this script cleans history:

  - documents.text is truncated at the marker (manifest is always last —
    it is appended after body text and the OCR section).
  - chunks fully inside the manifest (offset_start >= marker pos) are
    tombstoned across chunks / chunks_meta / chunks_fts / LanceDB.
  - the straddling chunk keeps the same chunk_id (it is positional,
    sha256(doc:index)) with truncated text + new content_hash; its BM25
    entry is reindexed and its vector deleted — the embed drain re-adds it.
  - document_metadata.char_count / normalized_hash are recomputed.

Idempotent: docs without the marker are skipped; vectors are the resume
checkpoint for re-embedding. Dry-run by default; --apply to write.

    .venv/Scripts/python.exe -X utf8 scripts/operations/strip_manifest_tails.py --corpus outputs/experiments/E12-corpus
    .venv/Scripts/python.exe -X utf8 scripts/operations/strip_manifest_tails.py --corpus outputs/experiments/E12-corpus --apply
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

MARKER = "--- Linked documents downloaded"


def _chunk_hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _doc_norm_hash(text: str) -> str:
    from ipa.reporter.reporter_curation import normalized_hash
    return normalized_hash(text or "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", required=True, help="Corpus dir (document_store.db, bm25_index.db, vector/lancedb)")
    ap.add_argument("--apply", action="store_true", help="Write changes (default: dry-run report)")
    args = ap.parse_args()

    corpus = Path(args.corpus).resolve()
    store_db = corpus / "document_store.db"
    bm25_db = corpus / "bm25_index.db"
    lance_dir = corpus / "vector" / "lancedb"
    if not store_db.exists():
        print(f"no hay document_store.db en {corpus}")
        return 1

    conn = sqlite3.connect(str(store_db), timeout=60)
    conn.execute("PRAGMA busy_timeout = 30000")

    docs = conn.execute(
        "SELECT document_id, text FROM documents "
        "WHERE tombstoned=0 AND text LIKE ?", (f"%{MARKER}%",)).fetchall()
    print(f"docs con manifiesto: {len(docs)}")

    plan: list[dict] = []
    for doc_id, text in docs:
        pos = text.find(MARKER)
        if pos < 0:
            continue
        new_text = text[:pos].rstrip() + "\n"
        rows = conn.execute(
            "SELECT chunk_id, text, span_json FROM chunks "
            "WHERE document_id=? AND tombstoned=0", (doc_id,)).fetchall()
        tombstone, boundary = [], []
        for chunk_id, ctext, span_json in rows:
            s0 = s1 = None
            try:
                span = json.loads(span_json or "{}") or {}
                s0, s1 = span.get("offset_start"), span.get("offset_end")
            except (ValueError, TypeError):
                pass
            if s0 is not None and s1 is not None:
                if s0 >= pos:
                    tombstone.append(chunk_id)
                elif s1 > pos:
                    boundary.append((chunk_id, ctext[:pos - s0].rstrip(), s0))
            elif MARKER in ctext or "[Document " in ctext:
                # Sin offsets: un chunk mayormente manifiesto se tombstonea;
                # si mezcla contenido, se trunca en el marker dentro del texto.
                cut = ctext.find(MARKER)
                doc_refs = ctext.find("[Document ")
                cut = min(x for x in (cut, doc_refs) if x >= 0)
                kept = ctext[:cut].rstrip()
                (boundary.append((chunk_id, kept, None)) if len(kept) >= 80
                 else tombstone.append(chunk_id))
        plan.append({
            "doc_id": doc_id, "pos": pos, "new_text": new_text,
            "empty_after": len(new_text.strip()) < 40,
            "tombstone": tombstone, "boundary": boundary,
        })

    n_tomb = sum(len(p["tombstone"]) for p in plan)
    n_bound = sum(len(p["boundary"]) for p in plan)
    n_empty = sum(1 for p in plan if p["empty_after"])
    print(f"plan: {len(plan)} docs | chunks a tombstonear: {n_tomb} | "
          f"chunks frontera (texto truncado): {n_bound} | "
          f"docs que quedan vacíos: {n_empty}")

    if not args.apply:
        print("dry-run — --apply para escribir")
        return 0

    from ipa import DocumentStore
    from ipa.indexes.bm25_index import BM25Index
    from ipa.indexes.lancedb_index import LanceDBIndex
    from ipa.contracts import DocumentChunk

    bm25 = BM25Index(str(bm25_db)) if bm25_db.exists() else None
    lance = LanceDBIndex(lance_dir, vector_dim=1024) if lance_dir.exists() else None
    store = DocumentStore(store_db)

    t0 = time.time()
    n_docs_done = n_vec_del = n_reidx = 0
    try:
        conn.execute("BEGIN IMMEDIATE")
        for p in plan:
            doc_id = p["doc_id"]
            if p["empty_after"]:
                # El doc era solo manifiesto → tombstone completo (clase
                # no_text retroactiva).
                conn.execute(
                    "UPDATE documents SET tombstoned=1 WHERE document_id=?",
                    (doc_id,))
                conn.execute(
                    "UPDATE chunks SET tombstoned=1 WHERE document_id=?",
                    (doc_id,))
                if bm25 is not None:
                    bm25.remove_document(doc_id, commit=False)
                if lance is not None and lance.is_queryable():
                    lance._table.delete(f"document_id = '{doc_id}'")
                continue

            conn.execute("UPDATE documents SET text=? WHERE document_id=?",
                         (p["new_text"], doc_id))
            conn.execute(
                "UPDATE document_metadata SET char_count=?, normalized_hash=? "
                "WHERE document_id=?",
                (len(p["new_text"]), _doc_norm_hash(p["new_text"]), doc_id))

            for chunk_id in p["tombstone"]:
                conn.execute(
                    "UPDATE chunks SET tombstoned=1 WHERE chunk_id=?",
                    (chunk_id,))
                conn.execute(
                    "DELETE FROM embedding_jobs WHERE chunk_id=?", (chunk_id,))
                if bm25 is not None:
                    bm25._conn.execute(
                        "DELETE FROM chunks_fts WHERE chunk_id=?", (chunk_id,))
                    bm25._conn.execute(
                        "UPDATE chunks_meta SET tombstoned=1 WHERE chunk_id=?",
                        (chunk_id,))
                if lance is not None and lance.is_queryable():
                    lance._table.delete(f"chunk_id = '{chunk_id}'")
                    n_vec_del += 1

            for chunk_id, new_ctext, s0 in p["boundary"]:
                conn.execute(
                    "UPDATE chunks SET text=?, content_hash=? WHERE chunk_id=?",
                    (new_ctext + "\n", _chunk_hash(new_ctext + "\n"), chunk_id))
                if s0 is not None:
                    conn.execute(
                        "UPDATE chunks SET span_json="
                        "json_set(span_json,'$.offset_end',?) WHERE chunk_id=?",
                        (s0 + len(new_ctext), chunk_id))
                if bm25 is not None:
                    bm25._conn.execute(
                        "DELETE FROM chunks_fts WHERE chunk_id=?", (chunk_id,))
                    bm25.add_chunks([DocumentChunk(
                        chunk_id=chunk_id, document_id=doc_id,
                        content_hash=_chunk_hash(new_ctext + "\n"),
                        text=new_ctext + "\n")], commit=False)
                    n_reidx += 1
                if lance is not None and lance.is_queryable():
                    # El vector viejo embebía el path-listing; borrarlo →
                    # el drain re-embebe el texto limpio con el mismo id.
                    lance._table.delete(f"chunk_id = '{chunk_id}'")
                    n_vec_del += 1
            n_docs_done += 1
            if n_docs_done % 200 == 0:
                conn.commit()
                if bm25 is not None:
                    bm25._conn.commit()
                print(f"  {n_docs_done}/{len(plan)} docs…", flush=True)
        conn.commit()
        if bm25 is not None:
            bm25._conn.commit()
    finally:
        store.close()
        if bm25 is not None:
            bm25.close()
        if lance is not None:
            lance.close()
        conn.close()

    print(f"aplicado en {time.time()-t0:.0f}s: {n_docs_done} docs limpiados, "
          f"{n_tomb} chunks tombstoned, {n_reidx} frontera reindexados, "
          f"{n_vec_del} vectores borrados (el drain los re-embebe)")
    print("siguiente paso: run_embed_drain.py --corpus <corpus> --background")
    return 0


if __name__ == "__main__":
    sys.exit(main())
