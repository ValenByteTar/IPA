"""Tests for Stage 2 index adapters: TantivyIndex, EmbeddingAdapter,
LanceDBIndex, SQLiteVecIndex."""
from __future__ import annotations

import os
import sys
import tempfile

import pytest

from ipa import (
    DocumentChunk,
    EmbeddingAdapter,
    LanceDBIndex,
    SQLiteVecIndex,
    TantivyIndex,
)
from ipa.contracts import SourceSpan


# ---------- fixtures ----------

@pytest.fixture
def sample_chunks():
    return [
        DocumentChunk(
            chunk_id="c1", document_id="d1", content_hash="h1",
            text="The quick brown fox jumps over the lazy dog",
            source_span=SourceSpan("a1", 1, 0, 44),
        ),
        DocumentChunk(
            chunk_id="c2", document_id="d1", content_hash="h2",
            text="Machine learning models require large training datasets",
            source_span=SourceSpan("a1", 1, 44, 100),
        ),
        DocumentChunk(
            chunk_id="c3", document_id="d2", content_hash="h3",
            text="Network security involves firewalls and intrusion detection",
            source_span=None,
        ),
    ]


@pytest.fixture
def embedding_adapter():
    """Shared embedding adapter — model loads once per session."""
    return EmbeddingAdapter(show_progress=False)


@pytest.fixture
def sample_vectors(embedding_adapter, sample_chunks):
    return embedding_adapter.embed_texts([c.text for c in sample_chunks])


# ---------- TantivyIndex ----------

class TestTantivyIndex:
    def test_add_and_search(self, sample_chunks):
        with tempfile.TemporaryDirectory() as td:
            idx = TantivyIndex(os.path.join(td, "tan"))
            idx.add_chunks(sample_chunks)
            assert idx.count() == 3
            hits = idx.search("fox dog", limit=5)
            assert len(hits) >= 1
            assert hits[0].chunk_id == "c1"
            assert hits[0].retrieval_backend == "tantivy"
            idx.close()

    def test_search_returns_scores(self, sample_chunks):
        with tempfile.TemporaryDirectory() as td:
            idx = TantivyIndex(os.path.join(td, "tan"))
            idx.add_chunks(sample_chunks)
            hits = idx.search("network security", limit=3)
            assert all(h.score != 0 for h in hits)
            idx.close()

    def test_search_preserves_source_span(self, sample_chunks):
        with tempfile.TemporaryDirectory() as td:
            idx = TantivyIndex(os.path.join(td, "tan"))
            idx.add_chunks(sample_chunks)
            hits = idx.search("fox", limit=1)
            assert hits[0].source_span is not None
            assert hits[0].source_span.artifact_id == "a1"
            idx.close()

    def test_empty_index_not_queryable(self):
        with tempfile.TemporaryDirectory() as td:
            idx = TantivyIndex(os.path.join(td, "tan"))
            assert not idx.is_queryable()
            idx.close()

    def test_limit_must_be_positive(self, sample_chunks):
        with tempfile.TemporaryDirectory() as td:
            idx = TantivyIndex(os.path.join(td, "tan"))
            idx.add_chunks(sample_chunks)
            with pytest.raises(ValueError):
                idx.search("fox", limit=0)
            idx.close()


# ---------- EmbeddingAdapter ----------

class TestEmbeddingAdapter:
    def test_dimension(self, embedding_adapter):
        assert embedding_adapter.dimension == 1024

    def test_embed_query_returns_vector(self, embedding_adapter):
        vec = embedding_adapter.embed_query("hello world")
        assert len(vec) == 1024
        assert all(isinstance(v, float) for v in vec)

    def test_embed_texts_returns_list(self, embedding_adapter):
        vecs = embedding_adapter.embed_texts(["hello", "world"])
        assert len(vecs) == 2
        assert len(vecs[0]) == 1024

    def test_embed_chunks_returns_pairs(self, embedding_adapter, sample_chunks):
        pairs = embedding_adapter.embed_chunks(sample_chunks)
        assert len(pairs) == 3
        assert pairs[0][0] == "c1"
        assert len(pairs[0][1]) == 1024


# ---------- LanceDBIndex ----------

class TestLanceDBIndex:
    def test_add_and_search(self, sample_chunks, sample_vectors, embedding_adapter):
        with tempfile.TemporaryDirectory() as td:
            idx = LanceDBIndex(os.path.join(td, "lance"), vector_dim=1024)
            idx.add_chunks(sample_chunks, sample_vectors)
            assert idx.count() == 3
            qv = embedding_adapter.embed_query("fox dog")
            hits = idx.search(qv, limit=3)
            assert len(hits) >= 1
            assert hits[0].retrieval_backend == "lancedb"
            idx.close()

    def test_empty_index_not_queryable(self, embedding_adapter):
        with tempfile.TemporaryDirectory() as td:
            idx = LanceDBIndex(os.path.join(td, "lance"), vector_dim=1024)
            assert not idx.is_queryable()
            idx.close()

    def test_mismatched_lengths_raises(self, sample_chunks, sample_vectors):
        with tempfile.TemporaryDirectory() as td:
            idx = LanceDBIndex(os.path.join(td, "lance"), vector_dim=1024)
            with pytest.raises(ValueError):
                idx.add_chunks(sample_chunks, sample_vectors[:2])
            idx.close()

    def test_metadata_columns_and_where(self, sample_chunks, sample_vectors):
        """doc_meta writes scalar columns usable as search pre-filters."""
        with tempfile.TemporaryDirectory() as td:
            idx = LanceDBIndex(os.path.join(td, "lance"), vector_dim=1024)
            meta = {
                "d1": {"source_domain": "example.com", "published_at": "2026-01-01",
                       "provenance": "main", "quality_score": 0.9},
                "d2": {"source_domain": "other.net", "published_at": "2025-01-01",
                       "provenance": "agent_research", "quality_score": 0.4},
            }
            idx.add_chunks(sample_chunks, sample_vectors, doc_meta=meta)
            assert idx._has_metadata_columns()
            hits = idx.search(sample_vectors[0], limit=10,
                              where="provenance = 'agent_research'")
            assert {h.chunk_id for h in hits} == {"c3"}
            hits = idx.search(sample_vectors[0], limit=10,
                              where="quality_score >= 0.5")
            assert {h.chunk_id for h in hits} == {"c1", "c2"}
            idx.close()

    def test_sparse_candidate_filter(self, sample_chunks, sample_vectors):
        """candidate_ids must bound the sparse scan (regression: undefined
        id_list fell back to a full-table scan and returned out-of-set hits)."""
        with tempfile.TemporaryDirectory() as td:
            idx = LanceDBIndex(os.path.join(td, "lance"), vector_dim=1024)
            sparse = [{"1": 1.0}, {"2": 1.0}, {"1": 5.0}]
            idx.add_chunks(sample_chunks, sample_vectors, sparse_weights=sparse)
            hits = idx.search_sparse({"1": 1.0}, limit=10, candidate_ids=["c1"])
            assert {h.chunk_id for h in hits} == {"c1"}
            idx.close()

    def test_sync_doc_metadata(self, sample_chunks, sample_vectors):
        """sync_doc_metadata backfills scalar columns from the canonical store."""
        class _StubStore:
            def all_sources(self):
                return {"d1": {"source_domain": "example.com",
                               "provenance": "main", "quality_score": 0.7}}
            def all_document_stored_at(self):
                return {"d1": "2026-03-01", "d2": "2026-03-02"}

        with tempfile.TemporaryDirectory() as td:
            idx = LanceDBIndex(os.path.join(td, "lance"), vector_dim=1024)
            idx.add_chunks(sample_chunks, sample_vectors)
            n = idx.sync_doc_metadata(_StubStore(), only_missing=False)
            assert n == 2
            hits = idx.search(sample_vectors[0], limit=10,
                              where="source_domain = 'example.com'")
            assert {h.chunk_id for h in hits} == {"c1", "c2"}
            idx.close()


    def test_legacy_table_schema_evolution(self, sample_chunks, sample_vectors):
        """Tables that predate the metadata columns get them via add_columns
        on first add_chunks/sync — the where filter only activates once the
        columns exist and are populated."""
        import lancedb as _ldb
        import pyarrow as _pa

        class _StubStore:
            def all_sources(self):
                return {"d1": {"source_domain": "example.com",
                               "provenance": "main", "quality_score": 0.7}}
            def all_document_stored_at(self):
                return {"d1": "2026-03-01", "d2": "2026-03-02"}

        with tempfile.TemporaryDirectory() as td:
            # Create a table with the pre-metadata schema directly.
            old_schema = _pa.schema([
                _pa.field("chunk_id", _pa.string()),
                _pa.field("document_id", _pa.string()),
                _pa.field("content_hash", _pa.string()),
                _pa.field("text", _pa.string()),
                _pa.field("vector", _pa.list_(_pa.float32(), 1024)),
                _pa.field("span_json", _pa.string()),
                _pa.field("sparse_json", _pa.string()),
            ])
            rows = [{
                "chunk_id": c.chunk_id, "document_id": c.document_id,
                "content_hash": c.content_hash, "text": c.text,
                "vector": v, "span_json": "{}", "sparse_json": "",
            } for c, v in zip(sample_chunks, sample_vectors)]
            db = _ldb.connect(os.path.join(td, "lance"))
            db.create_table("chunks", rows, schema=old_schema, mode="overwrite")

            idx = LanceDBIndex(os.path.join(td, "lance"), vector_dim=1024)
            assert not idx._has_metadata_columns()
            # where is ignored on the legacy schema (no silent empty results)
            hits = idx.search(sample_vectors[0], limit=10,
                              where="provenance = 'agent_research'")
            assert len(hits) == 3
            n = idx.sync_doc_metadata(_StubStore(), only_missing=False)
            assert n == 2
            assert idx._has_metadata_columns()
            hits = idx.search(sample_vectors[0], limit=10,
                              where="source_domain = 'example.com'")
            assert {h.chunk_id for h in hits} == {"c1", "c2"}
            idx.close()


def test_maybe_rerank_passthrough_when_disabled(monkeypatch):
    """IPA_RERANK=0 (opt-out) → items pass through truncated to top_k, order kept."""
    monkeypatch.setenv("IPA_RERANK", "0")
    from ipa.indexes.reranker_adapter import maybe_rerank
    items = [{"text": "a", "score": 0.5}, {"text": "b", "score": 0.9}]
    out = maybe_rerank("q", items, 1)
    assert out == items[:1]


def test_rerank_device_gate_uses_physical_vram(monkeypatch):
    """El gate usa la VRAM física (nvidia-smi): mem_get_info sobreestima en
    Windows/WDDM (cuenta memoria compartida) y mandaría el reranker a GPU
    con el LLM cargado ocupando casi toda la VRAM."""
    import torch
    from ipa.indexes import reranker_adapter as ra

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "mem_get_info",
        lambda: (5_000 * 1024 * 1024, 6_140 * 1024 * 1024))
    # nvidia-smi: el LLM ocupa casi todo → CPU aunque mem_get_info diga que hay lugar.
    monkeypatch.setattr(ra, "physical_free_vram_mb", lambda: 1_655.0)
    assert ra.RerankerAdapter()._resolve_device() == "cpu"
    # VRAM física de sobra → CUDA.
    monkeypatch.setattr(ra, "physical_free_vram_mb", lambda: 5_000.0)
    assert ra.RerankerAdapter()._resolve_device() == "cuda"
    # Sin nvidia-smi → fallback a mem_get_info.
    monkeypatch.setattr(ra, "physical_free_vram_mb", lambda: None)
    assert ra.RerankerAdapter()._resolve_device() == "cuda"


def test_embed_device_gate_uses_physical_vram(monkeypatch):
    """Mismo gate que el reranker para BGE-M3: con el LLM del chat ocupando
    la GPU, cargar BGE-M3 en CUDA agotó la VRAM y congeló la UI (EXP-008 §10)
    — debe caer a CPU aunque mem_get_info sobreestime la libre."""
    import torch
    from ipa.indexes import embedding_adapter as ea
    from ipa.indexes import reranker_adapter as ra

    # El conftest fuerza CPU (hermeticidad); este test prueba el gate, así
    # que restaura "auto" para que el adapter llegue a la lógica del gate.
    monkeypatch.setenv("IPA_EMBED_DEVICE", "auto")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "mem_get_info",
        lambda: (5_000 * 1024 * 1024, 6_140 * 1024 * 1024))
    # nvidia-smi: el LLM ocupa casi todo → CPU aunque mem_get_info diga que hay lugar.
    monkeypatch.setattr(ra, "physical_free_vram_mb", lambda: 1_655.0)
    assert ea.EmbeddingAdapter()._resolve_device() == "cpu"
    # VRAM física de sobra → CUDA.
    monkeypatch.setattr(ra, "physical_free_vram_mb", lambda: 5_000.0)
    assert ea.EmbeddingAdapter()._resolve_device() == "cuda"
    # Sin nvidia-smi → fallback a mem_get_info.
    monkeypatch.setattr(ra, "physical_free_vram_mb", lambda: None)
    assert ea.EmbeddingAdapter()._resolve_device() == "cuda"
    # Device explícito nunca lo pisa el gate.
    assert ea.EmbeddingAdapter(device="cpu")._resolve_device() == "cpu"


def test_embedding_batch_defaults_match_measured_devices():
    import os
    from ipa.indexes.embedding_adapter import (
        DEFAULT_BATCH_CPU, DEFAULT_BATCH_GPU, EmbeddingAdapter,
    )

    cpu = EmbeddingAdapter()
    cpu._device_resolved = "cpu"
    gpu = EmbeddingAdapter()
    gpu._device_resolved = "cuda"
    explicit = EmbeddingAdapter(batch_size=64)
    explicit._device_resolved = "cpu"
    assert DEFAULT_BATCH_CPU == int(os.environ.get("IPA_EMBED_BATCH_CPU", "4") or 4)
    assert DEFAULT_BATCH_GPU == 4
    assert cpu._resolve_batch() == DEFAULT_BATCH_CPU
    assert gpu._resolve_batch() == DEFAULT_BATCH_GPU
    assert explicit._resolve_batch() == 64


def test_physical_vram_probe_hides_nvidia_smi_console(monkeypatch):
    import subprocess
    from types import SimpleNamespace

    from ipa.indexes.reranker_adapter import physical_free_vram_mb

    seen = {}

    def fake_run(*args, **kwargs):
        seen["kwargs"] = kwargs
        return SimpleNamespace(stdout="1000, 6144\n", returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert physical_free_vram_mb() == 5144
    # Los flags CREATE_NO_WINDOW/DETACHED_PROCESS solo existen en Windows;
    # en POSIX creationflags debe ser 0.
    if sys.platform == "win32":
        assert seen["kwargs"]["creationflags"] == (
            subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS)
    else:
        assert seen["kwargs"]["creationflags"] == 0


# ---------- SQLiteVecIndex ----------

class TestSQLiteVecIndex:
    def test_add_and_search(self, sample_chunks, sample_vectors, embedding_adapter):
        with tempfile.TemporaryDirectory() as td:
            idx = SQLiteVecIndex(os.path.join(td, "svec.db"), vector_dim=1024)
            idx.add_chunks(sample_chunks, sample_vectors)
            assert idx.count() == 3
            qv = embedding_adapter.embed_query("fox dog")
            hits = idx.search(qv, limit=3)
            assert len(hits) >= 1
            assert hits[0].retrieval_backend == "sqlite_vec"
            idx.close()

    def test_empty_index_not_queryable(self):
        with tempfile.TemporaryDirectory() as td:
            idx = SQLiteVecIndex(os.path.join(td, "svec.db"), vector_dim=1024)
            assert not idx.is_queryable()
            idx.close()

    def test_mismatched_lengths_raises(self, sample_chunks, sample_vectors):
        with tempfile.TemporaryDirectory() as td:
            idx = SQLiteVecIndex(os.path.join(td, "svec.db"), vector_dim=1024)
            with pytest.raises(ValueError):
                idx.add_chunks(sample_chunks, sample_vectors[:2])
            idx.close()

    def test_search_preserves_chunk_id(self, sample_chunks, sample_vectors, embedding_adapter):
        with tempfile.TemporaryDirectory() as td:
            idx = SQLiteVecIndex(os.path.join(td, "svec.db"), vector_dim=1024)
            idx.add_chunks(sample_chunks, sample_vectors)
            qv = embedding_adapter.embed_query("network security firewalls")
            hits = idx.search(qv, limit=1)
            assert hits[0].chunk_id == "c3"
            idx.close()
