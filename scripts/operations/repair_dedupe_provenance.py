"""Repair a dedupe-by-URL pass over mislabeled provenance (agent_research).

The URL dedupe groups live docs by document_sources.source_url — but
agent_research rows were found carrying the SAME url on unrelated docs
(e.g. 26 distinct arXiv papers all stamped es.wikipedia.org/.../Jev).
Grouping by a corrupt key tombstones distinct content.

Repair, per recorded group (document_metadata.extra_json.dedupe_url):
  - drops similar to the keeper (Jaccard >= 0.6) were real re-scrapes ->
    stay tombstoned;
  - dissimilar drops are distinct documents -> un-tombstone (docs + chunks),
    restore BM25 rows, and leave vectors to the embedding drain;
  - provenance fix: every restored doc gets source_url := its embedded
    ``Source: <url>`` line (or '' when absent) — never the corrupt group URL;
  - a keeper whose own embedded Source: does not match the group URL gets
    its source_url corrected too.

Vectors: the drain re-embeds live chunks missing from LanceDB on the next
pass (scan-based discovery). Dry-run by default; --apply mutates.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ipa.contracts import DocumentChunk

_SRC_RE = re.compile(r"Source:\s*(https?://\S+)", re.I)
_WORDS = re.compile(r"[\w-]{4,}")
_SIMILAR = 0.6


def _toks(text: str) -> set[str]:
    return set(_WORDS.findall((text or "").lower()))


def _embedded_source(text: str) -> str:
    m = _SRC_RE.search(text or "")
    return m.group(1).rstrip("/") if m else ""


def _backup(path: Path) -> Path | None:
    if not path.exists():
        return None
    dst = path.with_name(f"{path.name}.bak-repair-{time.strftime('%Y%m%d-%H%M%S')}")
    shutil.copy2(path, dst)
    return dst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    corpus = Path(args.corpus)
    store_db = corpus / "document_store.db"
    conn = sqlite3.connect(str(store_db), timeout=30)
    conn.execute("PRAGMA busy_timeout = 30000")

    # Rebuild the groups recorded by the dedupe pass.
    groups: dict[str, dict] = {}
    for did, ej in conn.execute(
            "SELECT document_id, extra_json FROM document_metadata "
            "WHERE extra_json LIKE '%dedupe_url%'"):
        e = json.loads(ej)
        groups.setdefault(e["dedupe_url"], {"keep": e["deduped_by"], "drop": []})
        groups[e["dedupe_url"]]["drop"].append(did)
    print(f"grupos dedupe registrados: {len(groups)}")

    restore: list[str] = []          # doc_ids a un-tombstonear
    fix_url: dict[str, str] = {}     # doc_id -> source_url corregida
    real_dups = 0
    keeper_fixes = 0

    for url, g in groups.items():
        all_ids = [g["keep"]] + g["drop"]
        texts = {d: (conn.execute(
            "SELECT text FROM documents WHERE document_id=?", (d,)
        ).fetchone() or ("",))[0] for d in all_ids}
        keep_toks = _toks(texts[g["keep"]])

        keeper_emb = _embedded_source(texts[g["keep"]])
        keeper_ok = keeper_emb.rstrip("/") == url.rstrip("/")

        for d in g["drop"]:
            t = texts[d]
            sim = (len(keep_toks & _toks(t)) / len(keep_toks | _toks(t))
                   if keep_toks and t else 0.0)
            if sim >= _SIMILAR:
                real_dups += 1
                continue
            # Doc distinto con URL corrupta: restaurar + corregir proveniencia.
            restore.append(d)
            emb = _embedded_source(t)
            fix_url[d] = emb if emb else ""

        # Keeper mal etiquetado (su Source: embebido no es el del grupo):
        # corregir su proveniencia también.
        if keeper_emb and not keeper_ok:
            fix_url[g["keep"]] = keeper_emb
            keeper_fixes += 1

    print(f"docs a restaurar: {len(restore)} | dups reales (quedan tombstoned): "
          f"{real_dups} | keepers con URL corregida: {keeper_fixes}")

    if not args.apply:
        print("DRY-RUN")
        conn.close()
        return

    for name in ("document_store.db", "bm25_index.db"):
        bak = _backup(corpus / name)
        if bak:
            print(f"backup: {bak}")

    # 1. Un-tombstone docs + chunks del store.
    for d in restore:
        conn.execute("UPDATE documents SET tombstoned=0 WHERE document_id=?", (d,))
        conn.execute("UPDATE chunks SET tombstoned=0 WHERE document_id=?", (d,))

    # 2. Corregir document_sources.source_url (dominio derivado del URL).
    for did, url in fix_url.items():
        from urllib.parse import urlparse
        dom = urlparse(url).netloc.lower().removeprefix("www.") if url else ""
        conn.execute(
            "UPDATE document_sources SET source_url=?, source_domain=? "
            "WHERE document_id=?", (url, dom, did))
    conn.commit()

    # 3. BM25: add_chunks restaura meta (tombstoned=0) + FTS.
    bm25_db = corpus / "bm25_index.db"
    if bm25_db.exists() and restore:
        from ipa import BM25Index
        bm25 = BM25Index(bm25_db)
        try:
            ph_marks = ",".join("?" * len(restore))
            rows = conn.execute(
                f"SELECT chunk_id, document_id, content_hash, text, "
                f"metadata_json, span_json FROM chunks "
                f"WHERE document_id IN ({ph_marks}) AND tombstoned=0",
                restore).fetchall()
            objs = [DocumentChunk(
                chunk_id=r[0], document_id=r[1], content_hash=r[2],
                text=r[3], metadata=json.loads(r[4] or "{}"),
                source_span=None) for r in rows]
            for i in range(0, len(objs), 500):
                bm25.add_chunks(objs[i:i + 500], commit=False)
            bm25._conn.commit()
            print(f"bm25 restaurado: {len(objs)} chunks")
        finally:
            bm25.close()
    conn.close()

    print("APPLIED — vectores: correr el drain sobre el corpus "
          "(run_embed_drain.py --corpus ...) para re-embeber los restaurados")


if __name__ == "__main__":
    main()
