"""Tests de las señales Tier 0 (ingest_metadata) y su consumo en Tier 1.

Cubre: persistencia de document_metadata (hash/title/fecha/chars), provenance
configured_scrape directo al ingerir, flag de duplicado exacto vs main,
novelty hints post-drain, merge no destructivo de extra, backfill legacy,
el gate "corpus changed" (meta dirty) y la curación con novelty_hints.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from ipa import DocumentStore
from ipa.contracts import CanonicalDocument, DocumentChunk
from ipa.ingestion.ingest_metadata import (
    backfill_doc_metadata, compute_novelty_hints, record_ingest_metadata,
)


def _doc(doc_id: str, text: str) -> CanonicalDocument:
    return CanonicalDocument(
        document_id=doc_id, parser_id="test", mime_type="text/plain",
        pages=1, text=text, elements=[], source_spans=[],
    )


def _store_with_doc(path: Path, doc_id: str, text: str,
                    artifact_id: str = "art:1") -> DocumentStore:
    store = DocumentStore(path)
    store.put_document(_doc(doc_id, text), artifact_id)
    store.put_chunks([DocumentChunk(
        chunk_id=f"{doc_id}:c0", document_id=doc_id, content_hash="h",
        text=text[:200], metadata={}, source_span=None)])
    store.commit()
    return store


def _landing_db(path: Path, artifact_id: str, source_uri: str) -> None:
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE artifacts (artifact_id TEXT, source_uri TEXT)")
    conn.execute("INSERT INTO artifacts VALUES (?, ?)", (artifact_id, source_uri))
    conn.commit()
    conn.close()


_SCRAPED = "Title: Paper X\nSource: https://arxiv.org/abs/1\nDate: 2026-09-10\n\nbody " * 20


def test_record_ingest_metadata_full(tmp_path):
    web = tmp_path / "Landing" / "web"
    artifact = web / "arxiv" / "paper.txt"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("x", encoding="utf-8")
    _landing_db(tmp_path / "landing.db", "art:1", str(artifact))

    store = _store_with_doc(tmp_path / "store.db", "d1", _SCRAPED)
    try:
        stats = record_ingest_metadata(
            store, ["d1"], landing_db_path=tmp_path / "landing.db",
            web_root=web)
        assert stats["meta"] == 1 and stats["provenance"] == 1

        meta = store.get_doc_meta("d1")
        assert meta["normalized_hash"].startswith("sha256:")
        assert meta["title"]                       # primera línea no vacía
        assert meta["published_at"] == "2026-09-10"  # línea Date: del scrape
        assert meta["char_count"] == len(_SCRAPED)

        src = store.get_source("d1")
        assert src["provenance"] == "configured_scrape"
        assert src["source_url"] == "https://arxiv.org/abs/1"
        assert src["source_domain"] == "arxiv.org"
        assert src["published_at"] == "2026-09-10"
    finally:
        store.close()


def test_manual_landing_file_not_configured_scrape(tmp_path):
    """web_root=Landing (CLI default): un archivo manual en Landing/ raíz NO
    es configured_scrape — solo lo que vive bajo web/** lo es."""
    landing = tmp_path / "Landing"
    manual = landing / "manual.pdf"
    web_file = landing / "web" / "arxiv" / "paper.txt"
    manual.parent.mkdir(parents=True)
    web_file.parent.mkdir(parents=True)
    manual.write_text("x", encoding="utf-8")
    web_file.write_text("x", encoding="utf-8")

    ldb = tmp_path / "landing.db"
    conn = sqlite3.connect(str(ldb))
    conn.execute("CREATE TABLE artifacts (artifact_id TEXT, source_uri TEXT)")
    conn.execute("INSERT INTO artifacts VALUES ('a1', ?)", (str(manual),))
    conn.execute("INSERT INTO artifacts VALUES ('a2', ?)", (str(web_file),))
    conn.commit(); conn.close()

    store = DocumentStore(tmp_path / "store.db")
    store.put_document(_doc("d1", "manual content"), "a1")
    store.put_document(_doc("d2", "scraped content"), "a2")
    store.commit()
    try:
        stats = record_ingest_metadata(
            store, ["d1", "d2"], landing_db_path=ldb, web_root=landing)
        # Ambos reciben metadata; solo el de web/** recibe provenance.
        assert stats["meta"] == 2 and stats["provenance"] == 1
        assert store.get_source("d1") is None
        assert store.get_source("d2")["provenance"] == "configured_scrape"
    finally:
        store.close()


def test_hint_only_meta_row_gets_hash_on_reingest(tmp_path):
    """Una fila de document_metadata creada solo con novelty_hint (sin
    normalized_hash) no bloquea el relleno posterior de la metadata."""
    store = _store_with_doc(tmp_path / "store.db", "d1", _SCRAPED)
    try:
        store.put_doc_meta("d1", extra={"novelty_hint": {"max_cosine": 0.5}})
        store.commit()
        stats = record_ingest_metadata(store, ["d1"])
        assert stats["meta"] == 1
        meta = store.get_doc_meta("d1")
        assert meta["normalized_hash"].startswith("sha256:")
        assert meta["extra"]["novelty_hint"]["max_cosine"] == 0.5
    finally:
        store.close()


def test_record_ingest_metadata_idempotent(tmp_path):
    store = _store_with_doc(tmp_path / "store.db", "d1", "texto simple")
    try:
        record_ingest_metadata(store, ["d1"])
        meta1 = store.get_doc_meta("d1")
        store.put_doc_meta("d1", extra={"novelty_hint": {"max_cosine": 0.1}})
        store.commit()
        # Segunda corrida: no duplica meta ni pisa extras.
        stats = record_ingest_metadata(store, ["d1"])
        assert stats["meta"] == 0
        meta2 = store.get_doc_meta("d1")
        assert meta2["normalized_hash"] == meta1["normalized_hash"]
        assert meta2["extra"]["novelty_hint"]["max_cosine"] == 0.1
    finally:
        store.close()


def test_exact_duplicate_flag_against_main(tmp_path):
    main = _store_with_doc(tmp_path / "main.db", "m1", _SCRAPED)
    store = _store_with_doc(tmp_path / "src.db", "d1", _SCRAPED)
    try:
        record_ingest_metadata(main, ["m1"])          # main ya tiene su meta
        stats = record_ingest_metadata(store, ["d1"], main_store=main)
        assert stats["duplicates"] == 1
        assert store.get_doc_meta("d1")["extra"]["duplicate_of_main"] is True
    finally:
        main.close()
        store.close()


def test_put_doc_meta_merges_extra(tmp_path):
    store = _store_with_doc(tmp_path / "store.db", "d1", "abc")
    try:
        store.put_doc_meta("d1", normalized_hash="sha256:x",
                           extra={"duplicate_of_main": True})
        store.put_doc_meta("d1", extra={"novelty_hint": {"max_cosine": 0.9}})
        store.commit()
        extra = store.get_doc_meta("d1")["extra"]
        assert extra["duplicate_of_main"] is True
        assert extra["novelty_hint"]["max_cosine"] == 0.9
        # Actualizar sin extra no pisa lo existente.
        store.put_doc_meta("d1", title="T")
        store.commit()
        assert store.get_doc_meta("d1")["extra"]["duplicate_of_main"] is True
    finally:
        store.close()


def test_backfill_doc_metadata(tmp_path):
    store = _store_with_doc(tmp_path / "store.db", "d1", _SCRAPED)
    try:
        assert store.get_doc_meta("d1") is None
        assert backfill_doc_metadata(store) == 1
        meta = store.get_doc_meta("d1")
        assert meta["normalized_hash"].startswith("sha256:")
        assert meta["published_at"] == "2026-09-10"
        assert backfill_doc_metadata(store) == 0     # ya cubierto
    finally:
        store.close()


def test_backfill_fills_hint_only_rows(tmp_path):
    """Una fila creada solo con novelty_hint (normalized_hash NULL) también
    se backfillea — hash y hint son señales independientes."""
    store = _store_with_doc(tmp_path / "store.db", "d1", _SCRAPED)
    try:
        store.put_doc_meta("d1", extra={"novelty_hint": {"max_cosine": 0.5}})
        store.commit()
        assert store.get_doc_meta("d1")["normalized_hash"] is None
        assert backfill_doc_metadata(store) == 1
        meta = store.get_doc_meta("d1")
        assert meta["normalized_hash"].startswith("sha256:")
        assert meta["extra"]["novelty_hint"]["max_cosine"] == 0.5
    finally:
        store.close()


def test_level1_stale_hint_incremental_refresh(tmp_path, monkeypatch):
    """Integración L1: un hint stale (main creció) se refresca contra los
    embeddings de SOLO los docs nuevos — no carga el histórico completo —
    y el hint refrescado se persiste para el próximo ciclo."""
    from ipa.agentic.idle_enrichment import enrich_corpus_level1
    from ipa.agentic.topic_clusters import TopicClusterStore

    # Corpus staging real: d1 con hint stale (snapshot cuando main tenía 1 doc).
    corpus = tmp_path / "corpus"
    (corpus / "vector" / "lancedb").mkdir(parents=True)
    store = DocumentStore(corpus / "document_store.db")
    text = "alpha beta gamma delta epsilon zeta eta theta iota"
    store.put_document(_doc("d1", text), "a1")
    store.put_centroid("d1", [], 1)
    store.put_doc_meta("d1", extra={"novelty_hint": {
        "max_cosine": 0.10, "nearest_doc_id": "m1",
        "main_doc_count": 1, "main_latest_stored_at": "2026-01-01T00:00:00Z",
    }})
    store.commit()

    # Main real: m1 (viejo) + m2 (nuevo, agregado tras el snapshot — casi
    # duplicado de d1, mismo texto para que el gate léxico confirme).
    main = tmp_path / "main"
    (main / "vector" / "lancedb").mkdir(parents=True)
    mstore = DocumentStore(main / "document_store.db")
    mstore.put_document(_doc("m1", "old unrelated omega psi chi"), "am1")
    mstore.put_document(_doc("m2", text), "am2")
    mstore._conn.execute(
        "UPDATE documents SET stored_at='2026-01-01T00:00:00Z' WHERE document_id='m1'")
    mstore._conn.execute(
        "UPDATE documents SET stored_at='2026-06-01T00:00:00Z' WHERE document_id='m2'")
    mstore.commit()

    corpus_lance = _FakeLance({"d1": [1.0, 0.0]})
    main_lance = _FakeLance({"m1": [0.0, 1.0], "m2": [0.99, 0.01]})
    monkeypatch.setattr(
        "ipa.indexes.lancedb_index.LanceDBIndex",
        lambda p: main_lance if "main" in str(p) else corpus_lance)

    cs = TopicClusterStore(tmp_path / "clusters.db")
    try:
        result = enrich_corpus_level1(corpus, cs, main_corpus_path=main)
        decision = cs.get_curation_decision("d1")
        assert decision["decision"] == "duplicate"
        assert decision["duplicate_of"] == "m2"  # el nearest REFRESCADO
        # Self-heal: el hint persistido ya refleja el snapshot nuevo.
        hint = store.get_doc_meta("d1")["extra"]["novelty_hint"]
        assert hint["main_doc_count"] == 2
        assert hint["nearest_doc_id"] == "m2"
        assert hint["max_cosine"] > 0.98
        # El refresh fue incremental: main_lance solo recibió el id NUEVO
        # (m2) — nunca un full-scan (None) ni el doc viejo m1.
        assert main_lance.calls and all(
            c == {"m2"} for c in main_lance.calls if c is not None)
        assert None not in main_lance.calls
    finally:
        cs.close()
        store.close()
        mstore.close()


def test_level1_main_corpus_no_self_duplicate(tmp_path, monkeypatch):
    """topify_main (corpus == main): un doc NO puede marcarse DUPLICATE de
    sí mismo — ni por url+hash (path activado por el fix de formato) ni por
    cosine 1.0 contra su propio embedding histórico."""
    from ipa.agentic.idle_enrichment import enrich_corpus_level1
    from ipa.agentic.topic_clusters import TopicClusterStore

    main = tmp_path / "main"
    (main / "vector" / "lancedb").mkdir(parents=True)
    store = DocumentStore(main / "document_store.db")
    store.put_document(_doc("d1", "contenido propio del corpus"), "a1")
    store.put_centroid("d1", [], 1)
    # URL+hash propios: sin la exclusión, norm_hash == known_url_hashes[url].
    store.put_source("d1", "https://x.com/self", "x.com", "agent_research")
    store.commit()
    record_ingest_metadata(store, ["d1"])  # meta → url_normalized_hashes lo cubre

    main_lance = _FakeLance({"d1": [1.0, 0.0]})
    monkeypatch.setattr(
        "ipa.indexes.lancedb_index.LanceDBIndex", lambda p: main_lance)

    cs = TopicClusterStore(tmp_path / "clusters.db")
    try:
        enrich_corpus_level1(main, cs, main_corpus_path=main)
        decision = cs.get_curation_decision("d1")
        assert decision is not None
        assert decision["decision"] != "duplicate"
    finally:
        cs.close()
        store.close()


def test_refresh_novelty_hint():
    """Refresh incremental: solo recomputa contra los docs nuevos de main,
    manteniendo el máximo previo si ninguno lo supera."""
    from ipa.agentic.idle_enrichment import _refresh_novelty_hint
    hint = {"max_cosine": 0.60, "nearest_doc_id": "m1", "main_doc_count": 2}

    # Doc nuevo no relacionado → hint sin cambios.
    h2 = _refresh_novelty_hint(hint, [1.0, 0.0], {"m_new": [0.0, 1.0]})
    assert h2["max_cosine"] == 0.60 and h2["nearest_doc_id"] == "m1"

    # Doc nuevo casi idéntico → el hint se actualiza al nuevo nearest.
    h3 = _refresh_novelty_hint(hint, [1.0, 0.0], {"m_new": [0.99, 0.01]})
    assert h3["max_cosine"] > 0.98 and h3["nearest_doc_id"] == "m_new"

    # Sin embeddings nuevos → devuelve el hint intacto.
    h4 = _refresh_novelty_hint(hint, [1.0, 0.0], {})
    assert h4["max_cosine"] == 0.60


class _FakeLance:
    def __init__(self, embs):
        self._embs = embs
        self.calls: list = []  # doc_ids recibidos (None = full scan)

    def document_embeddings(self, doc_ids=None):
        self.calls.append(set(doc_ids) if doc_ids else None)
        if doc_ids:
            return {k: v for k, v in self._embs.items() if k in doc_ids}
        return dict(self._embs)

    def close(self):
        pass


def test_compute_novelty_hints(tmp_path):
    store = _store_with_doc(tmp_path / "store.db", "d1", "nuevo")
    main = _store_with_doc(tmp_path / "main.db", "m1", "viejo")
    try:
        record_ingest_metadata(store, ["d1"])
        lance = _FakeLance({"d1": [1.0, 0.0]})
        main_lance = _FakeLance({"m1": [0.9, 0.1], "m2": [0.0, 1.0]})
        n = compute_novelty_hints(store, lance, main, main_lance)
        assert n == 1
        hint = store.get_doc_meta("d1")["extra"]["novelty_hint"]
        assert hint["nearest_doc_id"] == "m1"
        assert hint["max_cosine"] > 0.99
        assert hint["main_doc_count"] == 1
        # Token de versión del snapshot: permite el refresh incremental de T1.
        assert hint["main_latest_stored_at"]
        # Idempotente: el hint ya existe.
        assert compute_novelty_hints(store, lance, main, main_lance) == 0
    finally:
        store.close()
        main.close()


def test_url_normalized_hashes(tmp_path):
    store = _store_with_doc(tmp_path / "store.db", "d1", "abc")
    try:
        record_ingest_metadata(store, ["d1"])
        store.put_source("d1", "https://x.com/a", "x.com", "agent_research")
        store.commit()
        url_hashes = store.url_normalized_hashes()
        assert url_hashes["https://x.com/a"].startswith("sha256:")
    finally:
        store.close()


def test_dirty_flag_roundtrip(tmp_path):
    from ipa.agentic.topic_clusters import TopicClusterStore
    cs = TopicClusterStore(tmp_path / "clusters.db")
    try:
        assert cs.get_meta("dirty:/x") is None
        cs.set_meta("dirty:/x", "1")
        assert cs.get_meta("dirty:/x") == "1"
        cs.set_meta("dirty:/x", "0")
        assert cs.get_meta("dirty:/x") == "0"
    finally:
        cs.close()


# ── consumo en curación (novelty_hints / url hashes) ────────────────────

def _curate_doc(doc_id: str, text: str, url: str = "") -> dict:
    return {
        "document_id": doc_id, "title": "t", "text": text,
        "source_url": url, "canonical_url": url, "source_domain": "",
        "quality_score": 0.0, "published_at": "", "content_hash": "",
    }


def _curate(docs, **kw):
    from ipa.reporter.reporter_curation import curate_documents
    return curate_documents(
        docs, report_id="t", period_start="2000-01-01T00:00:00Z",
        period_end="2100-01-01T00:00:00Z", quality_threshold=0.0, **kw)


def test_curation_novelty_hint_duplicate(tmp_path):
    text = "alpha beta gamma delta epsilon zeta eta theta"
    decisions = _curate(
        [_curate_doc("d1", text)],
        novelty_hints={"d1": {
            "max_cosine": 0.97, "nearest_doc_id": "m9",
            "nearest_text": text,          # léxico idéntico → confirma
        }})
    d = decisions[0]
    assert d.decision.value == "duplicate"
    assert d.duplicate_of == "m9"


def test_curation_novelty_hint_fuzzy_only(tmp_path):
    """Coseno alto sin confirmación léxica → se conserva (DEC-003)."""
    decisions = _curate(
        [_curate_doc("d1", "alpha beta gamma delta epsilon zeta")],
        novelty_hints={"d1": {
            "max_cosine": 0.97, "nearest_doc_id": "m9",
            "nearest_text": "completely unrelated omega psi chi phi",
        }})
    assert decisions[0].decision.value == "reporter_only"


def test_curation_url_hash_duplicate(tmp_path):
    """Re-descarga idéntica por URL+hash → DUPLICATE __main__."""
    from ipa.reporter.reporter_curation import normalized_hash
    text = "contenido del estudio"
    decisions = _curate(
        [_curate_doc("d1", text, url="https://x.com/a")],
        known_url_hashes={"https://x.com/a": normalized_hash(text)})
    assert decisions[0].decision.value == "duplicate"
    assert decisions[0].duplicate_of == "__main__"


def test_curation_published_at_period_filter(tmp_path):
    """published_at real habilita el filtro de período."""
    doc = _curate_doc("d1", "texto")
    doc["published_at"] = "1990-01-01"
    decisions = _curate([doc])
    assert decisions[0].decision.value == "defer"
