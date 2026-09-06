"""Tests for Stage 2 index adapters: TantivyIndex, EmbeddingAdapter,
LanceDBIndex, SQLiteVecIndex."""
from __future__ import annotations

import os
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
