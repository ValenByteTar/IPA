"""Tests del audit de salud del corpus (index_audit).

El audit es read-only: detecta inconsistencias (capa lógica sobre señales
Tier 0 + capa física comparando chunk_id sets entre store/BM25/FTS/LanceDB)
y reporta en index_health.json — nunca muta el corpus.
"""
import json
import sqlite3
from pathlib import Path

from ipa.agentic.index_audit import (
    run_index_audit, run_logical_checks, run_physical_checks,
)
from ipa.contracts import CanonicalDocument, DocumentChunk
from ipa.storage.document_store import DocumentStore


def _doc(doc_id: str, text: str) -> CanonicalDocument:
    return CanonicalDocument(
        document_id=doc_id, parser_id="t", mime_type="text/plain",
        pages=1, text=text, elements=[], source_spans=[],
    )


def _chunk(doc_id: str, n: int, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"{doc_id}-c{n}", document_id=doc_id,
        content_hash=f"h-{text[:8]}-{n}", text=text, metadata={},
        source_span=None,
    )


def _corpus(tmp_path: Path, *, docs: list[tuple[str, str]] | None = None,
            chunks_per_doc: int = 2) -> tuple[Path, DocumentStore]:
    corpus = tmp_path / "corpus"
    corpus.mkdir(parents=True)
    store = DocumentStore(corpus / "document_store.db")
    for did, text in (docs or [("d1", "alpha"), ("d2", "beta")]):
        store.put_document(_doc(did, text), f"a-{did}")
        store.put_chunks([
            _chunk(did, i, f"{text} chunk {i}") for i in range(chunks_per_doc)
        ])
    store.commit()
    return corpus, store


def _bm25(corpus: Path) -> "sqlite3.Connection":
    from ipa.indexes.bm25_index import BM25Index
    return BM25Index(corpus / "bm25_index.db")


def test_logical_clean_corpus(tmp_path):
    corpus, store = _corpus(tmp_path)
    store.put_doc_meta("d1", normalized_hash="sha256:a", char_count=5)
    store.put_doc_meta("d2", normalized_hash="sha256:b", char_count=4)
    store.commit(); store.close()
    conn = sqlite3.connect(str(corpus / "document_store.db"))
    try:
        out = run_logical_checks(conn)
    finally:
        conn.close()
    assert out["empty_docs_count"] == 0
    assert out["dup_hashes_count"] == 0
    assert out["missing_meta_count"] == 0
    assert out["dup_flag_leaks_count"] == 0


def test_logical_detects_empty_and_missing_meta(tmp_path):
    corpus, store = _corpus(tmp_path, docs=[("d1", ""), ("d2", "x"),
                                          ("d3", "sin meta")])
    store.put_doc_meta("d1", normalized_hash="sha256:e", char_count=0)
    store.put_doc_meta("d2", normalized_hash="sha256:f", char_count=1)
    # d3 sin fila en document_metadata
    store.commit(); store.close()
    conn = sqlite3.connect(str(corpus / "document_store.db"))
    try:
        out = run_logical_checks(conn)
    finally:
        conn.close()
    assert out["empty_docs"] == ["d1"]
    assert out["missing_meta_count"] == 1 and out["missing_meta"] == ["d3"]


def test_logical_marker_only_row_is_not_empty(tmp_path):
    """Una fila document_metadata creada solo por un marker (dedupe_url,
    novelty_hint) tiene char_count NULL — NULL = no computado, no vacío."""
    corpus, store = _corpus(tmp_path, docs=[("d1", "texto real")])
    store.put_doc_meta("d1", extra={"dedupe_url": "http://x", "deduped_by": "d9"})
    store.commit(); store.close()
    conn = sqlite3.connect(str(corpus / "document_store.db"))
    try:
        out = run_logical_checks(conn)
    finally:
        conn.close()
    assert out["empty_docs_count"] == 0
    assert out["null_hash_count"] == 1  # sin normalized_hash — backfill lo cubre


def test_logical_detects_dup_hash_and_flag_leak(tmp_path):
    corpus, store = _corpus(tmp_path, docs=[("d1", "a"), ("d2", "b"),
                                          ("d3", "c")])
    # d1 y d2 comparten normalized_hash → dup real
    store.put_doc_meta("d1", normalized_hash="sha256:same", char_count=1)
    store.put_doc_meta("d2", normalized_hash="sha256:same", char_count=1)
    # d3 flaggeado duplicate_of_main pero sigue vivo sin decisión DUPLICATE
    store.put_doc_meta("d3", normalized_hash="sha256:z", char_count=1,
                       extra={"duplicate_of_main": True})
    store.commit(); store.close()
    conn = sqlite3.connect(str(corpus / "document_store.db"))
    try:
        out = run_logical_checks(conn)
    finally:
        conn.close()
    assert out["dup_hashes_count"] == 1
    assert set(out["dup_hashes"][0]["doc_ids"]) == {"d1", "d2"}
    assert out["dup_flag_leaks"] == ["d3"]


def test_logical_flag_leak_respects_duplicate_decision(tmp_path):
    """Un doc flaggeado pero ya decidido DUPLICATE no es fuga."""
    corpus, store = _corpus(tmp_path, docs=[("d1", "a")])
    store.put_doc_meta("d1", normalized_hash="sha256:a", char_count=1,
                       extra={"duplicate_of_main": True})
    store.commit(); store.close()
    clusters = tmp_path / "clusters.db"
    c = sqlite3.connect(str(clusters))
    c.execute("CREATE TABLE curation_decisions "
              "(document_id TEXT, payload_json TEXT)")
    c.execute("INSERT INTO curation_decisions VALUES (?, ?)",
              ("d1", json.dumps({"decision": "duplicate"})))
    c.commit()
    conn = sqlite3.connect(str(corpus / "document_store.db"))
    try:
        out = run_logical_checks(conn, cluster_conn=c)
    finally:
        conn.close(); c.close()
    assert out["dup_flag_leaks_count"] == 0


def test_logical_scrape_without_url(tmp_path):
    corpus, store = _corpus(tmp_path, docs=[("d1", "a")])
    store.put_doc_meta("d1", normalized_hash="sha256:a", char_count=1)
    store.put_source("d1", "", "", "configured_scrape", 0.5)
    store.commit(); store.close()
    conn = sqlite3.connect(str(corpus / "document_store.db"))
    try:
        out = run_logical_checks(conn)
    finally:
        conn.close()
    assert out["scrape_no_url"] == ["d1"]


def test_physical_aligned_indexes(tmp_path):
    from ipa.contracts import DocumentChunk as DC
    corpus, store = _corpus(tmp_path)
    bm25 = _bm25(corpus)
    rows = store._conn.execute(
        "SELECT chunk_id, document_id, content_hash, text, metadata_json "
        "FROM chunks").fetchall()
    objs = [DC(chunk_id=r[0], document_id=r[1], content_hash=r[2],
               text=r[3], metadata=json.loads(r[4]), source_span=None)
            for r in rows]
    bm25.add_chunks(objs)
    store.close(); bm25.close()

    class _FakeLance:
        def to_arrow(self):
            import pyarrow as pa
            return pa.table({"chunk_id": [o.chunk_id for o in objs]})

    conn = sqlite3.connect(str(corpus / "document_store.db"))
    try:
        out = run_physical_checks(conn, corpus, lance_table=_FakeLance())
    finally:
        conn.close()
    assert out["bm25"]["meta_missing_count"] == 0
    assert out["bm25"]["fts_orphans_count"] == 0
    assert out["lance"]["missing_count"] == 0
    assert out["lance"]["orphans_count"] == 0
    assert out["lance"]["duplicate_chunk_ids"] == 0


def test_physical_detects_drift(tmp_path):
    from ipa.contracts import DocumentChunk as DC
    corpus, store = _corpus(tmp_path, docs=[("d1", "a"), ("d2", "b")])
    bm25 = _bm25(corpus)
    rows = store._conn.execute(
        "SELECT chunk_id, document_id, content_hash, text, metadata_json "
        "FROM chunks WHERE document_id='d1'").fetchall()
    objs = [DC(chunk_id=r[0], document_id=r[1], content_hash=r[2],
               text=r[3], metadata=json.loads(r[4]), source_span=None)
            for r in rows]
    bm25.add_chunks(objs)  # solo d1 → d2 queda fuera de BM25/FTS
    store.close(); bm25.close()

    lance_ids = [o.chunk_id for o in objs] + ["orphan-x", "orphan-x"]

    class _FakeLance:
        def to_arrow(self):
            import pyarrow as pa
            return pa.table({"chunk_id": lance_ids})

    conn = sqlite3.connect(str(corpus / "document_store.db"))
    try:
        out = run_physical_checks(conn, corpus, lance_table=_FakeLance())
    finally:
        conn.close()
    assert out["bm25"]["meta_missing_count"] == 2  # 2 chunks de d2
    assert out["lance"]["missing_count"] == 2
    assert out["lance"]["orphans"] == ["orphan-x"]
    assert out["lance"]["duplicate_chunk_ids"] == 1


def test_physical_detects_spam_chunks(tmp_path):
    corpus, store = _corpus(tmp_path, docs=[
        ("d1", "a"), ("d2", "b"), ("d3", "c"), ("d4", "d")])
    # Mismo content_hash en 3 docs distintos → boilerplate same-site.
    for did in ("d1", "d2", "d3"):
        store._conn.execute(
            "INSERT INTO chunks (chunk_id, document_id, content_hash, text, "
            "metadata_json, stored_at, tombstoned) "
            "VALUES (?, ?, 'spam-hash', 'nav boilerplate', '{}', 'now', 0)",
            (f"{did}-spam", did))
    store.commit(); store.close()
    conn = sqlite3.connect(str(corpus / "document_store.db"))
    try:
        out = run_physical_checks(conn, corpus)
    finally:
        conn.close()
    assert "spam-hash" in out["spam_chunks"]["same_site_sample"]


def test_run_index_audit_writes_health_json(tmp_path):
    corpus, store = _corpus(tmp_path)
    store.put_doc_meta("d1", normalized_hash="sha256:a", char_count=5)
    store.put_doc_meta("d2", normalized_hash="sha256:b", char_count=4)
    store.commit(); store.close()
    out_file = tmp_path / "index_health.json"
    payload = run_index_audit(corpus, layer="logical", output=out_file,
                              cluster_db=tmp_path / "nope.db")
    assert out_file.exists()
    written = json.loads(out_file.read_text(encoding="utf-8"))
    assert written["corpus_name"] == "corpus"
    assert written["logical"]["empty_docs_count"] == 0
    assert written["physical"] is None  # solo capa lógica pedida
    assert payload["status"] == "ok"


def test_run_index_audit_merges_layers(tmp_path):
    """Una corrida de capa lógica preserva el resultado físico anterior."""
    corpus, store = _corpus(tmp_path)
    store.put_doc_meta("d1", normalized_hash="sha256:a", char_count=5)
    store.put_doc_meta("d2", normalized_hash="sha256:b", char_count=4)
    store.commit(); store.close()
    out_file = tmp_path / "index_health.json"
    run_index_audit(corpus, layer="both", output=out_file)
    payload = run_index_audit(corpus, layer="logical", output=out_file)
    assert payload["physical"] is not None  # la física previa se preserva
    assert payload["logical"]["checked_at"]
