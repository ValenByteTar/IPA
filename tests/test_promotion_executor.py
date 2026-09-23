"""Tests del preflight de cobertura vectorial en promotion_executor (PM-004).

La purga del staging es la única operación destructiva de una promoción. El
preflight garantiza que nunca corre mientras un chunk vivo del source siga sin
vector en main LanceDB: la promoción se difiere (source intacto, cola pending)
hasta que el drain completa los vectores faltantes.

Se usa un LanceDBIndex falso (pyarrow real) con registro por path: la fase
lance, la purga y el preflight comparten estado por corpus sin tocar disco
real.
"""
from __future__ import annotations

import re
import sqlite3

import pyarrow as pa
import pytest

from ipa.agentic.promotion_executor import (
    promote_documents_to_main,
    process_promotion_queue,
)


# ---------------------------------------------------------------------------
# Fake LanceDBIndex (pyarrow real; registry compartido por path resuelto)
# ---------------------------------------------------------------------------

_COLUMNS = [
    "chunk_id", "document_id", "content_hash", "text", "span_json",
    "sparse_json", "vector", "source_domain", "published_at", "provenance",
    "quality_score", "stored_at",
]
_TYPES = {
    "chunk_id": pa.string(), "document_id": pa.string(),
    "content_hash": pa.string(), "text": pa.string(),
    "span_json": pa.string(), "sparse_json": pa.string(),
    "vector": pa.list_(pa.float32()), "source_domain": pa.string(),
    "published_at": pa.string(), "provenance": pa.string(),
    "quality_score": pa.float32(), "stored_at": pa.string(),
}


class _FakeLanceTable:
    def __init__(self, rows=None):
        self.rows = list(rows or [])
        self.fail_read = False

    def to_arrow(self):
        if self.fail_read:
            raise RuntimeError("simulated read failure")
        data = {c: pa.array([r.get(c) for r in self.rows], type=_TYPES[c])
                for c in _COLUMNS}
        return pa.table(data)

    def add(self, tbl):
        for i in range(tbl.num_rows):
            self.rows.append({c: tbl.column(c)[i].as_py()
                              for c in tbl.column_names})

    def delete(self, where):
        ids = set(re.findall(r"'([^']+)'", where))
        self.rows = [r for r in self.rows if r["document_id"] not in ids]

    @property
    def schema(self):
        return self.to_arrow().schema


class _FakeLanceIndex:
    _by_path: dict[str, _FakeLanceTable] = {}

    def __init__(self, path, vector_dim=1024):
        key = str(path) if not hasattr(path, "resolve") else str(path.resolve())
        if key not in _FakeLanceIndex._by_path:
            _FakeLanceIndex._by_path[key] = _FakeLanceTable()
        self._table = _FakeLanceIndex._by_path[key]

    def _ensure_table(self, sample_vector=None):
        pass

    def _ensure_metadata_columns(self):
        return True

    def sync_doc_metadata(self, store, only_missing=True):
        pass

    def close(self):
        pass


@pytest.fixture(autouse=True)
def _fake_lance(monkeypatch):
    """LanceDBIndex → fake con estado por path, limpio entre tests."""
    _FakeLanceIndex._by_path.clear()
    monkeypatch.setattr(
        "ipa.indexes.lancedb_index.LanceDBIndex", _FakeLanceIndex)


def _vector_row(chunk_id: str, document_id: str) -> dict:
    return {
        "chunk_id": chunk_id, "document_id": document_id,
        "content_hash": f"hash:{chunk_id}", "text": f"text {chunk_id}",
        "span_json": "", "sparse_json": "", "vector": [0.1] * 8,
        "source_domain": "", "published_at": "", "provenance": "",
        "quality_score": 0.0, "stored_at": "",
    }


def _make_corpora(tmp_path, *, docs=(("doc:1", 3), ("doc:2", 3)),
                  source_vectors=False):
    """Source y main corpus reales (DocumentStore) + chunks vivos."""
    from ipa import DocumentStore
    from ipa.contracts import CanonicalDocument, DocumentChunk

    src = tmp_path / "src"
    main = tmp_path / "main"
    src.mkdir()
    main.mkdir()

    store = DocumentStore(src / "document_store.db")
    all_chunk_ids: list[tuple[str, str]] = []
    for doc_id, n_chunks in docs:
        doc = CanonicalDocument(
            document_id=doc_id, parser_id="test", mime_type="text/plain",
            pages=1, text=f"texto de {doc_id}", elements=[], source_spans=[],
        )
        store.put_document(doc, f"artifact:{doc_id}")
        chunks = [
            DocumentChunk(
                chunk_id=f"{doc_id}:c{i}", document_id=doc_id,
                content_hash=f"h{doc_id}{i}", text=f"chunk {doc_id} {i}",
                metadata={}, source_span=None,
            )
            for i in range(n_chunks)
        ]
        store.put_chunks(chunks)
        all_chunk_ids.extend((f"{doc_id}:c{i}", doc_id) for i in range(n_chunks))
    store.commit()
    store.close()

    # Main solo necesita el schema inicializado.
    main_store = DocumentStore(main / "document_store.db")
    main_store.close()

    if source_vectors:
        source_lance_dir = src / "vector" / "lancedb"
        source_lance_dir.mkdir(parents=True)
        _FakeLanceIndex._by_path[str(source_lance_dir.resolve())] = (
            _FakeLanceTable(
                [_vector_row(cid, did) for cid, did in all_chunk_ids]))
        # La fase lance exige que el padre del main lance exista.
        (main / "vector").mkdir()

    return src, main


def _live_docs(corpus_dir) -> set[str]:
    conn = sqlite3.connect(
        f"file:{(corpus_dir / 'document_store.db').resolve()}?mode=ro",
        uri=True, timeout=10)
    try:
        return {row[0] for row in conn.execute(
            "SELECT document_id FROM documents WHERE tombstoned=0")}
    finally:
        conn.close()


def _live_chunks(corpus_dir) -> set[str]:
    conn = sqlite3.connect(
        f"file:{(corpus_dir / 'document_store.db').resolve()}?mode=ro",
        uri=True, timeout=10)
    try:
        return {row[0] for row in conn.execute(
            "SELECT chunk_id FROM chunks WHERE tombstoned=0")}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# promote_documents_to_main: preflight
# ---------------------------------------------------------------------------

def test_promotion_defers_when_source_vectors_missing(tmp_path):
    """PM-004: sin vectores en main, la promoción difiere y NO purga el source."""
    src, main = _make_corpora(tmp_path)

    result = promote_documents_to_main(["doc:1", "doc:2"], src, main)

    assert result["deferred"] is True
    assert result["missing_vectors"] == 6
    # El source queda intacto (la purga es lo que se previene).
    assert _live_docs(src) == {"doc:1", "doc:2"}
    assert len(_live_chunks(src)) == 6


def test_promotion_completes_when_vectors_covered(tmp_path):
    """Con vectores completos en el source, promueve, copia y purga normal."""
    src, main = _make_corpora(tmp_path, source_vectors=True)

    result = promote_documents_to_main(["doc:1", "doc:2"], src, main)

    assert result.get("deferred") is not True
    assert result["promoted_docs"] == 2
    assert result["promoted_chunks"] == 6
    assert result["promoted_vectors"] == 6
    assert result["purged_docs"] == 2
    # Source purgado; main tiene los 6 chunks con sus 6 vectores.
    assert _live_docs(src) == set()
    main_rows = _FakeLanceIndex._by_path[
        str((main / "vector" / "lancedb").resolve())].rows
    assert len(main_rows) == 6


def test_promotion_completes_after_backfill(tmp_path):
    """El retry tras el backfill de vectores completa la promoción diferida."""
    src, main = _make_corpora(tmp_path)

    first = promote_documents_to_main(["doc:1", "doc:2"], src, main)
    assert first["deferred"] is True

    # Backfill: el drain vectoriza el source; la próxima pasada copia a main.
    source_lance_dir = src / "vector" / "lancedb"
    source_lance_dir.mkdir(parents=True)
    (main / "vector").mkdir()
    _FakeLanceIndex._by_path[str(source_lance_dir.resolve())] = (
        _FakeLanceTable([_vector_row(f"{d}:c{i}", d)
                         for d in ("doc:1", "doc:2") for i in range(3)]))

    second = promote_documents_to_main(["doc:1", "doc:2"], src, main)
    assert second.get("deferred") is not True
    assert second["purged_docs"] == 2
    assert _live_docs(src) == set()
    assert second["promoted_vectors"] == 6


def test_deferred_batch_stays_pending_in_queue(tmp_path):
    """process_promotion_queue NO marca done un batch diferido."""
    from ipa.agentic.topic_clusters import TopicClusterStore

    src, main = _make_corpora(tmp_path)
    store = TopicClusterStore(tmp_path / "clusters.db")
    try:
        for doc_id in ("doc:1", "doc:2"):
            store.mark_promotion_pending(
                doc_id, "test", "configured_scrape", str(src))
        result = process_promotion_queue(store, src, main)
        assert result["deferred_docs"] == 2
        assert result["processed"] == 0
        pending = store.pending_promotions()
        assert {p["document_id"] for p in pending} == {"doc:1", "doc:2"}
    finally:
        store.close()


def test_queue_processes_batch_after_backfill(tmp_path):
    """Retry end-to-end: primera pasada difiere, tras el backfill promueve."""
    from ipa.agentic.topic_clusters import TopicClusterStore

    src, main = _make_corpora(tmp_path)
    store = TopicClusterStore(tmp_path / "clusters.db")
    try:
        for doc_id in ("doc:1", "doc:2"):
            store.mark_promotion_pending(
                doc_id, "test", "configured_scrape", str(src))

        first = process_promotion_queue(store, src, main)
        assert first["deferred_docs"] == 2

        source_lance_dir = src / "vector" / "lancedb"
        source_lance_dir.mkdir(parents=True)
        (main / "vector").mkdir()
        _FakeLanceIndex._by_path[str(source_lance_dir.resolve())] = (
            _FakeLanceTable([_vector_row(f"{d}:c{i}", d)
                             for d in ("doc:1", "doc:2") for i in range(3)]))

        second = process_promotion_queue(store, src, main)
        assert second["processed"] == 2
        assert second["deferred_docs"] == 0
        assert store.pending_promotions() == []
        assert _live_docs(src) == set()
    finally:
        store.close()


def test_require_vectors_optout_restores_old_behavior(tmp_path, monkeypatch):
    """IPA_PROMOTION_REQUIRE_VECTORS=0 permite purgar sin cobertura (emergencia)."""
    monkeypatch.setenv("IPA_PROMOTION_REQUIRE_VECTORS", "0")
    src, main = _make_corpora(tmp_path)

    result = promote_documents_to_main(["doc:1", "doc:2"], src, main)

    assert result.get("deferred") is not True
    assert result["purged_docs"] == 2
    assert _live_docs(src) == set()


def test_unreadable_main_lance_defers_instead_of_marking_done(tmp_path):
    """Si main LanceDB no se puede leer, difiere (antes marcaba done sin purgar)."""
    src, main = _make_corpora(tmp_path, source_vectors=True)
    main_lance_dir = main / "vector" / "lancedb"
    main_lance_dir.mkdir(parents=True)
    table = _FakeLanceTable()
    table.fail_read = True
    _FakeLanceIndex._by_path[str(main_lance_dir.resolve())] = table

    result = promote_documents_to_main(["doc:1", "doc:2"], src, main)

    assert result["deferred"] is True
    assert result.get("defer_reason") == "cannot read main LanceDB ids"
    assert _live_docs(src) == {"doc:1", "doc:2"}


def test_partial_vector_coverage_still_defers(tmp_path):
    """5/6 vectores presentes: igual difiere — la purga es todo-o-nada."""
    src, main = _make_corpora(tmp_path, source_vectors=True)
    # Quito un vector del source ANTES de promover → cobertura parcial.
    source_table = _FakeLanceIndex._by_path[
        str((src / "vector" / "lancedb").resolve())]
    source_table.rows = [r for r in source_table.rows
                         if r["chunk_id"] != "doc:1:c0"]

    result = promote_documents_to_main(["doc:1", "doc:2"], src, main)

    assert result["deferred"] is True
    assert result["missing_vectors"] == 1
    assert "doc:1:c0" in result["missing_sample"]
    assert _live_docs(src) == {"doc:1", "doc:2"}


def test_missing_source_store_db_completes_without_crash(tmp_path):
    """Source dir sin document_store.db: nada que proteger → done, sin crash."""
    src = tmp_path / "src"
    main = tmp_path / "main"
    src.mkdir()
    main.mkdir()

    result = promote_documents_to_main(["doc:gone"], src, main)

    assert result.get("deferred") is not True
    assert result["promoted_docs"] == 0
    assert result["purged_docs"] == 0


# ---------------------------------------------------------------------------
# Purga incompleta: BM25 del source lockeado → defer + retry
# ---------------------------------------------------------------------------

def _add_source_bm25(src, docs=(("doc:1", 3), ("doc:2", 3))):
    """BM25 real del source con los chunks vivos (como deja la ingesta)."""
    from ipa import BM25Index
    from ipa.contracts import DocumentChunk

    bm25 = BM25Index(src / "bm25_index.db")
    bm25.add_chunks([
        DocumentChunk(
            chunk_id=f"{d}:c{i}", document_id=d,
            content_hash=f"h{d}{i}", text=f"chunk {d} {i}",
            metadata={}, source_span=None)
        for d, n in docs for i in range(n)
    ])
    bm25.close()


def _bm25_live_counts(src) -> tuple[int, int]:
    conn = sqlite3.connect(
        f"file:{(src / 'bm25_index.db').resolve()}?mode=ro",
        uri=True, timeout=10)
    try:
        meta = conn.execute(
            "SELECT COUNT(*) FROM chunks_meta WHERE tombstoned=0"
        ).fetchone()[0]
        fts = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
        return meta, fts
    finally:
        conn.close()


def test_locked_source_bm25_defers_and_queue_retries(tmp_path, monkeypatch):
    """bm25_index.db del source lockeado (fast path concurrente): la purga
    del DocumentStore ya ocurrió pero el paso BM25 no — el batch difiere y
    queda pending en vez de dejar FTS desincronizado. Al liberar el lock el
    retry completa."""
    import ipa.agentic.promotion_executor as pe
    from ipa.agentic.topic_clusters import TopicClusterStore

    monkeypatch.setattr(pe, "_BM25_PURGE_ATTEMPTS", 2)
    monkeypatch.setattr(pe, "_BM25_PURGE_SLEEP_S", 0.01)
    monkeypatch.setattr(pe, "_BM25_PURGE_BUSY_MS", 50)

    src, main = _make_corpora(tmp_path, source_vectors=True)
    _add_source_bm25(src)
    assert _bm25_live_counts(src) == (6, 6)

    # Lock de escritura sostenido, como el de una ingesta fast-path activa.
    lock = sqlite3.connect(str(src / "bm25_index.db"), timeout=0)
    lock.execute("BEGIN IMMEDIATE")

    store = TopicClusterStore(tmp_path / "clusters.db")
    try:
        for doc_id in ("doc:1", "doc:2"):
            store.mark_promotion_pending(
                doc_id, "test", "configured_scrape", str(src))

        first = process_promotion_queue(store, src, main)
        assert first["deferred_docs"] == 2
        assert first["processed"] == 0
        assert {p["document_id"] for p in store.pending_promotions()} == {
            "doc:1", "doc:2"}
        # La copia a main y el tombstone del DocumentStore ya ocurrieron —
        # solo el paso BM25 quedó pendiente.
        assert _live_docs(src) == set()
        assert _bm25_live_counts(src) == (6, 6)

        lock.execute("ROLLBACK")
        lock.close()

        second = process_promotion_queue(store, src, main)
        assert second["processed"] == 2
        assert second["deferred_docs"] == 0
        assert store.pending_promotions() == []
        # BM25 reconciliado: meta tombstoned, FTS vacío.
        assert _bm25_live_counts(src) == (0, 0)
    finally:
        if lock:
            try:
                lock.execute("ROLLBACK")
                lock.close()
            except Exception:
                pass
        store.close()


def test_completed_purge_leaves_no_incomplete_steps(tmp_path):
    """Camino feliz: purga completa no reporta pasos incompletos ni difiere."""
    src, main = _make_corpora(tmp_path, source_vectors=True)
    _add_source_bm25(src)

    result = promote_documents_to_main(["doc:1", "doc:2"], src, main)

    assert result.get("deferred") is not True
    assert result["purged_docs"] == 2
    assert _bm25_live_counts(src) == (0, 0)


def test_corrupt_source_db_defers_batch_via_queue(tmp_path):
    """DB del source corrupta: el batch difiere y queda pending (no crashea
    el ciclo ni marca done). El resto de la cola sigue procesándose."""
    from ipa.agentic.topic_clusters import TopicClusterStore

    src, main = _make_corpora(tmp_path)
    # Corrompo el document_store del source: existe pero no es SQLite.
    (src / "document_store.db").write_bytes(b"not a sqlite database")

    store = TopicClusterStore(tmp_path / "clusters.db")
    try:
        for doc_id in ("doc:1", "doc:2"):
            store.mark_promotion_pending(
                doc_id, "test", "configured_scrape", str(src))
        result = process_promotion_queue(store, src, main)
        assert result["deferred_docs"] == 2
        assert result["processed"] == 0
        assert {p["document_id"] for p in store.pending_promotions()} == {
            "doc:1", "doc:2"}
    finally:
        store.close()
