"""Corpus dedupe by canonical URL — keep one live document per source_url.

Scrapers re-download the same URL with slightly different extracted text
(JS drift, timestamps), so the content-hash gate lets each variant in as a
"new" document. This module consolidates: for every source_url with more
than one live document, the most complete one survives (max live chunks;
ties break on newest stored_at) and the rest are tombstoned across the
three indexes — DocumentStore, BM25 (meta + FTS) and LanceDB.

Everything is tombstone/delete on derived indexes — reversible for the
DocumentStore and rebuildable for BM25/LanceDB. dry_run is the default.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

_SQLITE_VARS = 400  # stay well under SQLITE_MAX_VARIABLE_NUMBER


def _batched(ids: list[str]) -> list[list[str]]:
    return [ids[i:i + _SQLITE_VARS] for i in range(0, len(ids), _SQLITE_VARS)]


def find_url_duplicate_groups(store_db: Path) -> dict[str, list[dict[str, Any]]]:
    """Live documents grouped by source_url; returns only groups >1.

    Each member: {document_id, live_chunks, chars, stored_at}.
    """
    conn = sqlite3.connect(
        f"file:{Path(store_db).resolve()}?mode=ro", uri=True, timeout=30)
    try:
        rows = conn.execute(
            "SELECT s.source_url, d.document_id, length(d.text), d.stored_at, "
            "(SELECT COUNT(*) FROM chunks c WHERE c.document_id=d.document_id "
            " AND c.tombstoned=0) "
            "FROM document_sources s "
            "JOIN documents d ON d.document_id=s.document_id "
            "WHERE d.tombstoned=0 AND s.source_url IS NOT NULL "
            "AND s.source_url != ''"
        ).fetchall()
    finally:
        conn.close()
    groups: dict[str, list[dict[str, Any]]] = {}
    for url, doc_id, chars, stored_at, n_chunks in rows:
        groups.setdefault(url, []).append({
            "document_id": doc_id, "live_chunks": int(n_chunks),
            "chars": int(chars or 0), "stored_at": stored_at or "",
        })
    return {u: m for u, m in groups.items() if len(m) > 1}


def _pick_keeper(members: list[dict[str, Any]]) -> dict[str, Any]:
    """Most complete extraction wins; newest stored_at breaks ties."""
    return max(members, key=lambda m: (m["live_chunks"], m["chars"], m["stored_at"]))


def dedupe_by_url(
    corpus_path: Path,
    *,
    urls: list[str] | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Tombstone all but the most complete live document per source_url.

    ``urls`` limits the operation to those URLs (e.g. only the URLs of a
    just-promoted batch). Returns a report with kept/tombstoned detail.
    """
    corpus_path = Path(corpus_path)
    store_db = corpus_path / "document_store.db"

    groups = find_url_duplicate_groups(store_db)
    if urls is not None:
        wanted = set(urls)
        groups = {u: m for u, m in groups.items() if u in wanted}

    plan: list[dict[str, Any]] = []
    tomb_doc_ids: list[str] = []
    for url, members in sorted(groups.items()):
        keeper = _pick_keeper(members)
        losers = [m for m in members if m["document_id"] != keeper["document_id"]]
        tomb_doc_ids.extend(m["document_id"] for m in losers)
        plan.append({
            "url": url,
            "keep": keeper["document_id"],
            "keep_chunks": keeper["live_chunks"],
            "drop": [m["document_id"] for m in losers],
            "drop_chunks": sum(m["live_chunks"] for m in losers),
        })

    report: dict[str, Any] = {
        "corpus": str(corpus_path),
        "groups": len(plan),
        "tombstone_docs": len(tomb_doc_ids),
        "tombstone_chunks": sum(p["drop_chunks"] for p in plan),
        "kept_docs": len(plan),
        "plan": plan,
        "dry_run": dry_run,
    }
    if dry_run or not tomb_doc_ids:
        return report

    applied = tombstone_documents(corpus_path, tomb_doc_ids)
    # Provenance of the decision, for audit and later review.
    store = None
    try:
        from ipa import DocumentStore
        store = DocumentStore(store_db)
        for p in plan:
            for doc_id in p["drop"]:
                store.put_doc_meta(
                    doc_id, extra={"deduped_by": p["keep"], "dedupe_url": p["url"],
                                   "deduped_at": time.strftime(
                                       "%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        store.commit()
    finally:
        if store is not None:
            store.close()
    if applied.get("lance_error"):
        report["lance_error"] = applied["lance_error"]

    report["applied"] = True
    return report


def tombstone_documents(
    corpus_path: Path,
    document_ids: list[str],
) -> dict[str, Any]:
    """Tombstone whole documents across the three indexes of a corpus.

    Batched UPDATEs on DocumentStore (docs + chunks + stale embedding_jobs),
    BM25 (meta tombstone + FTS delete) and a LanceDB delete by document_id.
    Idempotent; same mechanics as the promotion purge.
    """
    corpus_path = Path(corpus_path)
    store_db = corpus_path / "document_store.db"
    bm25_db = corpus_path / "bm25_index.db"
    lance_dir = corpus_path / "vector" / "lancedb"
    out: dict[str, Any] = {"docs": len(document_ids)}

    from ipa import DocumentStore
    store = DocumentStore(store_db)
    try:
        for batch in _batched(document_ids):
            ph = ",".join("?" * len(batch))
            store._conn.execute(
                f"UPDATE documents SET tombstoned=1 WHERE document_id IN ({ph})",
                batch)
            store._conn.execute(
                f"UPDATE chunks SET tombstoned=1 WHERE document_id IN ({ph})",
                batch)
            store._conn.execute(
                f"DELETE FROM embedding_jobs WHERE chunk_id IN "
                f"(SELECT chunk_id FROM chunks WHERE document_id IN ({ph}) "
                f"AND tombstoned=1)",
                batch)
        store.commit()
    finally:
        store.close()

    if bm25_db.exists():
        conn = sqlite3.connect(str(bm25_db), timeout=30)
        try:
            conn.execute("PRAGMA busy_timeout = 10000")
            conn.execute("BEGIN IMMEDIATE")
            try:
                for batch in _batched(document_ids):
                    ph = ",".join("?" * len(batch))
                    conn.execute(
                        f"DELETE FROM chunks_fts WHERE chunk_id IN "
                        f"(SELECT chunk_id FROM chunks_meta WHERE document_id IN ({ph}))",
                        batch)
                    conn.execute(
                        f"UPDATE chunks_meta SET tombstoned=1 WHERE document_id IN ({ph})",
                        batch)
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        finally:
            conn.close()

    if lance_dir.exists():
        try:
            from ipa.indexes.lancedb_index import LanceDBIndex
            lance = LanceDBIndex(lance_dir, vector_dim=1024)
            try:
                if lance._table is not None:
                    for batch in _batched(document_ids):
                        id_list = ", ".join(f"'{d}'" for d in batch)
                        lance._table.delete(f"document_id IN ({id_list})")
            finally:
                lance.close()
        except Exception as exc:
            # Vectores residuales del doc tombstoned: los reporta el audit
            # (orphans) y el próximo pase los limpia — no bloquea.
            out["lance_error"] = str(exc)
    return out


# ---------------------------------------------------------------------------
# Boilerplate / spam chunks — identical content_hash across unrelated docs
# ---------------------------------------------------------------------------

def find_spam_chunks(
    store_db: Path,
    *,
    min_docs: int = 3,
    min_domains: int = 2,
) -> list[dict[str, Any]]:
    """Live chunks whose content_hash is shared across unrelated documents.

    Flagged when the hash appears in >= min_docs distinct live documents
    (site-wide boilerplate — arXiv nav blocks, injected tag-clouds) OR in
    >= min_domains distinct source domains (same text across unrelated
    sites is never real overlap). Same-domain pairs stay out — two related
    articles can legitimately share a passage.
    """
    conn = sqlite3.connect(
        f"file:{Path(store_db).resolve()}?mode=ro", uri=True, timeout=30)
    try:
        rows = conn.execute(
            "SELECT c.content_hash, COUNT(DISTINCT c.document_id) AS nd, "
            "COUNT(DISTINCT COALESCE(s.source_domain,'')) AS ndom, "
            "COUNT(*) AS n, substr(MIN(c.text),1,120) "
            "FROM chunks c "
            "JOIN documents d ON d.document_id=c.document_id AND d.tombstoned=0 "
            "LEFT JOIN document_sources s ON s.document_id=c.document_id "
            "WHERE c.tombstoned=0 "
            "GROUP BY c.content_hash "
            "HAVING nd>=? OR ndom>=?",
            (min_docs, min_domains),
        ).fetchall()
    finally:
        conn.close()
    return [{"content_hash": h, "n_docs": nd, "n_domains": ndom,
             "n_chunks": n, "sample": sample}
            for h, nd, ndom, n, sample in rows]


def tombstone_chunks(
    corpus_path: Path,
    content_hashes: list[str],
    *,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Tombstone every live chunk row matching `content_hashes` (all docs).

    Chunk-level purge: store tombstone + BM25 meta/FTS + LanceDB delete.
    The owning documents stay live with their remaining chunks.
    """
    corpus_path = Path(corpus_path)
    store_db = corpus_path / "document_store.db"
    bm25_db = corpus_path / "bm25_index.db"
    lance_dir = corpus_path / "vector" / "lancedb"

    conn = sqlite3.connect(
        f"file:{Path(store_db).resolve()}?mode=ro", uri=True, timeout=30)
    try:
        chunk_ids: list[str] = []
        for batch in _batched(content_hashes):
            ph = ",".join("?" * len(batch))
            chunk_ids.extend(r[0] for r in conn.execute(
                f"SELECT chunk_id FROM chunks "
                f"WHERE content_hash IN ({ph}) AND tombstoned=0",
                batch).fetchall())
    finally:
        conn.close()

    report: dict[str, Any] = {
        "corpus": str(corpus_path), "hashes": len(content_hashes),
        "tombstone_chunks": len(chunk_ids), "dry_run": dry_run,
    }
    if dry_run or not chunk_ids:
        return report

    store = None
    try:
        from ipa import DocumentStore
        store = DocumentStore(store_db)
        for batch in _batched(chunk_ids):
            ph = ",".join("?" * len(batch))
            store._conn.execute(
                f"UPDATE chunks SET tombstoned=1 WHERE chunk_id IN ({ph})",
                batch)
            store._conn.execute(
                f"DELETE FROM embedding_jobs WHERE chunk_id IN ({ph})",
                batch)
        store.commit()
    finally:
        if store is not None:
            store.close()

    if bm25_db.exists():
        conn = sqlite3.connect(str(bm25_db), timeout=30)
        try:
            conn.execute("PRAGMA busy_timeout = 10000")
            conn.execute("BEGIN IMMEDIATE")
            try:
                for batch in _batched(chunk_ids):
                    ph = ",".join("?" * len(batch))
                    conn.execute(
                        f"DELETE FROM chunks_fts WHERE chunk_id IN ({ph})",
                        batch)
                    conn.execute(
                        f"UPDATE chunks_meta SET tombstoned=1 WHERE chunk_id IN ({ph})",
                        batch)
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        finally:
            conn.close()

    if lance_dir.exists():
        try:
            from ipa.indexes.lancedb_index import LanceDBIndex
            lance = LanceDBIndex(lance_dir, vector_dim=1024)
            try:
                if lance._table is not None:
                    for batch in _batched(chunk_ids):
                        id_list = ", ".join(f"'{c}'" for c in batch)
                        lance._table.delete(f"chunk_id IN ({id_list})")
            finally:
                lance.close()
        except Exception as exc:
            report["lance_error"] = str(exc)

    report["applied"] = True
    return report
