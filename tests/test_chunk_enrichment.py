"""Tests del core de enriquecimiento de chunks (chunk_enrichment).

Cubre: parseo del formato SUMMARY/Q1-Q3, texto enriquecido, escaneo de la
tabla chunks (to_process / pending_reembed / ya enriquecidos), el pase
completo con provider falso (aislamiento de errores, preempción entre lotes,
checkpoint de re-embed) y la recuperación de re-embeds pendientes.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pytest

from ipa.agentic.chunk_enrichment import (
    ChunkScan, build_enriched_text, count_pending, enrich_chunks,
    enrichment_messages, parse_enrichment, scan_chunks,
)


# ── helpers ────────────────────────────────────────────────────────────

def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _make_db(path: Path, rows: list[tuple]) -> Path:
    """Crea document_store.db mínimo con la tabla chunks."""
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE chunks (chunk_id TEXT, document_id TEXT, text TEXT, "
        "content_hash TEXT, metadata_json TEXT)"
    )
    conn.executemany("INSERT INTO chunks VALUES (?,?,?,?,?)", rows)
    conn.commit()
    conn.close()
    return path


def _low_density_text(seed: str = "x") -> str:
    """≥800 chars con densidad <0.6 → should_summarize() True."""
    return (f"lorem{seed} ipsum dolor sit amet " * 40)


def _enriched_row(chunk_id: str, *, status: str = "complete") -> tuple:
    text = "[Summary] resumen\n\n[Questions]\nQ: q?\n\noriginal"
    meta = {"enrichment": {
        "embedding_status": status,
        "text_hash": _sha(text),
    }}
    return (chunk_id, "doc1", text, "h1", json.dumps(meta))


@dataclass
class _Result:
    text: str = ""
    error: str | None = None


_ENRICH_OUT = "SUMMARY: un resumen\nQ1: pregunta uno?\nQ2: pregunta dos?\nQ3: pregunta tres?"


class _SerialProvider:
    """Provider sin batch (camino Ollama): devuelve la salida enriquecida."""

    engine = "ollama"

    def __init__(self, fail_on_call: int | None = None):
        self.calls = 0
        self.fail_on_call = fail_on_call

    def generate_chat(self, messages, *, max_new_tokens=None, temperature=None):
        self.calls += 1
        if self.fail_on_call is not None and self.calls == self.fail_on_call:
            return _Result(text="", error="boom")
        return _Result(text=_ENRICH_OUT)


class _BatchProvider(_SerialProvider):
    """Provider con batch (camino ExL3)."""

    engine = "exllamav3"
    batch_size = 2

    def generate_chat_batch(self, batch_messages, *, max_new_tokens=None,
                            temperature=None):
        self.calls += len(batch_messages)
        out = []
        for i, _ in enumerate(batch_messages):
            if self.fail_on_call is not None and self.calls - len(batch_messages) + i + 1 == self.fail_on_call:
                out.append(_Result(text="", error="boom"))
            else:
                out.append(_Result(text=_ENRICH_OUT))
        return out


class _FakeEmbedding:
    def __init__(self):
        self.texts: list[str] = []

    def embed_texts_hybrid(self, texts):
        self.texts.extend(texts)
        return [[0.1] * 4 for _ in texts], [{1: 0.5} for _ in texts]


class _FakeLanceDB:
    def __init__(self):
        self.added: list[str] = []

    def add_chunks(self, chunks, vectors, sparse_weights=None):
        self.added.extend(c.chunk_id for c in chunks)


# ── parseo ─────────────────────────────────────────────────────────────

def test_parse_enrichment_structured():
    summary, qs = parse_enrichment(_ENRICH_OUT)
    assert summary == "un resumen"
    assert qs == ["pregunta uno?", "pregunta dos?", "pregunta tres?"]


def test_parse_enrichment_strips_think_block():
    raw = "<think>rumia rumia</think>\n" + _ENRICH_OUT
    summary, qs = parse_enrichment(raw)
    assert summary == "un resumen"
    assert len(qs) == 3


def test_parse_enrichment_fallback_to_raw():
    summary, qs = parse_enrichment("salida sin formato esperado")
    assert summary == "salida sin formato esperado"
    assert qs == []


def test_build_enriched_text():
    out = build_enriched_text("resumen", ["q1", "q2"], "ORIGINAL")
    assert out.startswith("[Summary] resumen")
    assert "[Questions]" in out and "Q: q1" in out
    assert out.endswith("ORIGINAL")


def test_enrichment_messages_format():
    msgs = enrichment_messages("texto " * 600)
    assert msgs[0]["role"] == "system"
    assert '"""' in msgs[1]["content"]


# ── escaneo ────────────────────────────────────────────────────────────

def test_scan_chunks_classifies(tmp_path):
    db = _make_db(tmp_path / "store.db", [
        ("c1", "d1", _low_density_text("a"), "h", "{}"),          # to_process
        ("c2", "d1", "corto", "h", "{}"),                          # skipped
        _enriched_row("c3", status="complete"),                    # already
        _enriched_row("c4", status="pending"),                     # reembed
    ])
    conn = sqlite3.connect(str(db))
    try:
        scan = scan_chunks(conn)
    finally:
        conn.close()
    assert scan.total == 4
    assert [c[0] for c in scan.to_process] == ["c1"]
    assert [c[0] for c in scan.pending_reembed] == ["c4"]
    assert scan.already_enriched == 2
    assert scan.skipped == 1
    assert scan.pending == 2


def test_scan_chunks_limit(tmp_path):
    db = _make_db(tmp_path / "store.db", [
        (f"c{i}", "d1", _low_density_text(str(i)), "h", "{}") for i in range(5)
    ])
    conn = sqlite3.connect(str(db))
    try:
        scan = scan_chunks(conn, limit=2)
    finally:
        conn.close()
    assert len(scan.to_process) == 2


def test_count_pending(tmp_path):
    db = _make_db(tmp_path / "store.db", [
        ("c1", "d1", _low_density_text("a"), "h", "{}"),
        _enriched_row("c2", status="complete"),
    ])
    assert count_pending(db) == 1


# ── pase completo ──────────────────────────────────────────────────────

def test_enrich_chunks_end_to_end(tmp_path):
    db = _make_db(tmp_path / "store.db", [
        ("c1", "d1", _low_density_text("a"), "h1", "{}"),
        ("c2", "d1", _low_density_text("b"), "h2", "{}"),
    ])
    emb, lance = _FakeEmbedding(), _FakeLanceDB()
    stats = enrich_chunks(
        _SerialProvider(), db, lancedb=lance, embedding=emb,
        reembed_batch_size=1)

    assert stats["enriched"] == 2
    assert stats["reembedded"] == 2
    assert stats["errors"] == 0
    assert sorted(lance.added) == ["c1", "c2"]

    conn = sqlite3.connect(str(db))
    try:
        for cid in ("c1", "c2"):
            text, meta_raw = conn.execute(
                "SELECT text, metadata_json FROM chunks WHERE chunk_id=?",
                (cid,)).fetchone()
            assert text.startswith("[Summary] un resumen")
            meta = json.loads(meta_raw)["enrichment"]
            assert meta["embedding_status"] == "complete"
            assert meta["text_hash"] == _sha(text)
            assert "canonical_text" in meta  # original preservado
    finally:
        conn.close()


def test_enrich_chunks_is_resumable(tmp_path):
    """Un segundo pase no reprocesa lo ya enriquecido."""
    db = _make_db(tmp_path / "store.db", [
        ("c1", "d1", _low_density_text("a"), "h1", "{}"),
    ])
    emb, lance = _FakeEmbedding(), _FakeLanceDB()
    provider = _SerialProvider()
    enrich_chunks(provider, db, lancedb=lance, embedding=emb)
    first_calls = provider.calls

    stats = enrich_chunks(provider, db, lancedb=lance, embedding=emb)
    assert provider.calls == first_calls  # no regeneró
    assert stats["enriched"] == 0
    assert stats["pending"] == 0


def test_enrich_chunks_isolates_item_errors(tmp_path):
    db = _make_db(tmp_path / "store.db", [
        ("c1", "d1", _low_density_text("a"), "h1", "{}"),
        ("c2", "d1", _low_density_text("b"), "h2", "{}"),
    ])
    provider = _BatchProvider(fail_on_call=2)
    stats = enrich_chunks(provider, db, lancedb=_FakeLanceDB(),
                          embedding=_FakeEmbedding())
    assert stats["enriched"] == 1
    assert stats["errors"] == 1


def test_enrich_chunks_aborts_between_batches(tmp_path):
    db = _make_db(tmp_path / "store.db", [
        (f"c{i}", "d1", _low_density_text(str(i)), "h", "{}") for i in range(4)
    ])
    provider = _SerialProvider()
    calls = {"n": 0}

    def _abort():
        return provider.calls >= 2  # aborta tras el primer lote (batch_size=1)

    stats = enrich_chunks(provider, db, lancedb=_FakeLanceDB(),
                          embedding=_FakeEmbedding(), should_abort=_abort)
    assert stats["aborted"] is True
    assert stats["enriched"] == 2
    # Lo enriquecido quedó con checkpoint durable → se reanuda después.
    assert count_pending(db) == 2


def test_enrich_chunks_recovers_pending_reembed(tmp_path):
    """Re-embed pendiente de una corrida cortada se recupera sin LLM."""
    db = _make_db(tmp_path / "store.db", [
        _enriched_row("c1", status="pending"),
    ])
    emb, lance = _FakeEmbedding(), _FakeLanceDB()
    provider = _SerialProvider()
    stats = enrich_chunks(provider, db, lancedb=lance, embedding=emb)

    assert provider.calls == 0          # no generó nada
    assert stats["reembedded"] == 1
    assert lance.added == ["c1"]
    conn = sqlite3.connect(str(db))
    try:
        meta = json.loads(conn.execute(
            "SELECT metadata_json FROM chunks WHERE chunk_id='c1'").fetchone()[0])
        assert meta["enrichment"]["embedding_status"] == "complete"
    finally:
        conn.close()


def test_enrich_chunks_without_lancedb_leaves_pending(tmp_path):
    """Sin lancedb/embedding: enriquece pero el checkpoint queda pending."""
    db = _make_db(tmp_path / "store.db", [
        ("c1", "d1", _low_density_text("a"), "h1", "{}"),
    ])
    stats = enrich_chunks(_SerialProvider(), db)
    assert stats["enriched"] == 1
    assert stats["reembedded"] == 0
    conn = sqlite3.connect(str(db))
    try:
        meta = json.loads(conn.execute(
            "SELECT metadata_json FROM chunks WHERE chunk_id='c1'").fetchone()[0])
        assert meta["enrichment"]["embedding_status"] == "pending"
    finally:
        conn.close()
    assert count_pending(db) == 1  # el re-embed queda para el próximo pase
