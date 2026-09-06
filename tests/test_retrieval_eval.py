"""Tests for E10 retrieval evaluation metrics and query generation.

Validates:
  - IR metrics (recall@k, MRR, nDCG) on known rankings
  - Query generation from a synthetic store
  - Hybrid score fusion deduplication and weighting
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from ipa.contracts import SearchHit, SourceSpan
from ipa.retrieval_eval import (
    EvalQuery,
    RetrievalMetrics,
    recall_at_k,
    document_recall_at_k,
    reciprocal_rank,
    ndcg_at_k,
    compute_metrics_from_hits,
    generate_query_set,
    hybrid_fuse,
)


# ---------------------------------------------------------------------------
# IR metric unit tests
# ---------------------------------------------------------------------------

class TestRecallAtK:
    def test_relevant_in_top_k(self):
        ranked = ["c1", "c2", "c3", "c4", "c5"]
        assert recall_at_k(ranked, "c3", k=5) == 1.0

    def test_relevant_not_in_top_k(self):
        ranked = ["c1", "c2", "c3", "c4", "c5"]
        assert recall_at_k(ranked, "c3", k=2) == 0.0

    def test_relevant_at_position_1(self):
        ranked = ["c1", "c2", "c3"]
        assert recall_at_k(ranked, "c1", k=1) == 1.0

    def test_relevant_not_in_list(self):
        ranked = ["c1", "c2", "c3"]
        assert recall_at_k(ranked, "cX", k=10) == 0.0

    def test_empty_ranking(self):
        assert recall_at_k([], "c1", k=5) == 0.0


class TestDocumentRecallAtK:
    def test_doc_in_top_k(self):
        ranked = ["d1", "d2", "d3"]
        assert document_recall_at_k(ranked, "d2", k=3) == 1.0

    def test_doc_not_in_top_k(self):
        ranked = ["d1", "d2", "d3"]
        assert document_recall_at_k(ranked, "d3", k=2) == 0.0


class TestReciprocalRank:
    def test_rank_1(self):
        assert reciprocal_rank(["c1", "c2", "c3"], "c1") == 1.0

    def test_rank_3(self):
        assert reciprocal_rank(["c1", "c2", "c3"], "c3") == pytest.approx(1 / 3)

    def test_not_found(self):
        assert reciprocal_rank(["c1", "c2"], "cX") == 0.0

    def test_empty(self):
        assert reciprocal_rank([], "c1") == 0.0


class TestNDCG:
    def test_relevant_at_position_1(self):
        ranked = ["c1", "c2", "c3"]
        assert ndcg_at_k(ranked, "c1", k=3) == 1.0

    def test_relevant_at_position_2(self):
        ranked = ["c1", "c2", "c3"]
        # DCG = 1/log2(3) ≈ 0.6309, IDCG = 1.0
        expected = 1.0 / (2 * math.log2(3))  # = 1/log2(3)
        # Actually: DCG = 1/log2(2+1) = 1/log2(3), IDCG = 1/log2(2) = 1.0
        result = ndcg_at_k(ranked, "c2", k=3)
        assert result == pytest.approx(1.0 / math.log2(3), rel=1e-4)

    def test_not_found(self):
        ranked = ["c1", "c2", "c3"]
        assert ndcg_at_k(ranked, "cX", k=3) == 0.0

    def test_empty(self):
        assert ndcg_at_k([], "c1", k=5) == 0.0


import math


# ---------------------------------------------------------------------------
# compute_metrics_from_hits tests
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def _make_per_query(self, n=10):
        """Generate n per-query dicts with known results."""
        per_query = []
        for i in range(n):
            rel_id = f"relevant_{i}"
            # Half the queries have the relevant chunk at position 1,
            # half at position 4.
            if i % 2 == 0:
                ranked_chunks = [rel_id, "c2", "c3", "c4", "c5"]
            else:
                ranked_chunks = ["c2", "c3", "c4", rel_id, "c5"]
            per_query.append({
                "ranked_chunk_ids": ranked_chunks,
                "ranked_doc_ids": [f"d_{i}" if j == 0 else "other" for j in range(5)],
                "relevant_chunk_id": rel_id,
                "relevant_doc_id": f"d_{i}",
                "latency_ms": 10.0 + i,
            })
        return per_query

    def test_basic_metrics(self):
        per_query = self._make_per_query(10)
        metrics = compute_metrics_from_hits("test", per_query)
        assert metrics.backend == "test"
        assert metrics.n_queries == 10
        # 5/10 have relevant at position 1 -> recall@1 = 0.5
        assert metrics.recall_at_1 == 0.5
        # All have relevant in top 5 -> recall@5 = 1.0
        assert metrics.recall_at_5 == 1.0

    def test_mrr(self):
        per_query = self._make_per_query(10)
        metrics = compute_metrics_from_hits("test", per_query)
        # 5 queries at rank 1 (RR=1), 5 at rank 4 (RR=0.25)
        # MRR = (5*1 + 5*0.25) / 10 = 0.625
        assert metrics.mrr == pytest.approx(0.625, abs=1e-4)

    def test_latency(self):
        per_query = self._make_per_query(10)
        metrics = compute_metrics_from_hits("test", per_query)
        # Latencies are 10, 11, 12, ..., 19; sorted = [10..19]
        # p50 = sorted[5] = 15
        assert metrics.avg_latency_ms == pytest.approx(14.5, abs=0.1)
        assert metrics.p50_latency_ms == 15.0  # sorted[5] = 15

    def test_empty_per_query(self):
        metrics = compute_metrics_from_hits("test", [])
        assert metrics.n_queries == 0
        assert metrics.recall_at_1 == 0.0


# ---------------------------------------------------------------------------
# Hybrid fusion tests
# ---------------------------------------------------------------------------

class TestHybridFuse:
    def _make_hit(self, cid: str, score: float, backend: str = "test") -> SearchHit:
        return SearchHit(
            chunk_id=cid,
            score=score,
            source_span=None,
            retrieval_backend=backend,
        )

    def test_deduplication(self):
        """Same chunk from both backends should appear once."""
        lex = [self._make_hit("c1", 5.0, "tantivy")]
        vec = [self._make_hit("c1", 0.8, "lancedb")]
        fused = hybrid_fuse(lex, vec, limit=10)
        assert len(fused) == 1
        assert fused[0].chunk_id == "c1"
        assert fused[0].retrieval_backend == "hybrid"

    def test_weighting_prefers_lexical(self):
        """With higher lexical weight, lexical-only chunks rank higher."""
        lex = [self._make_hit("c_lex", 10.0, "tantivy")]
        vec = [self._make_hit("c_vec", 0.9, "lancedb")]
        fused = hybrid_fuse(lex, vec, lexical_weight=0.9, vector_weight=0.1, limit=10)
        # c_lex has normalized score 1.0 * 0.9 = 0.9
        # c_vec has normalized score 1.0 * 0.1 = 0.1
        assert fused[0].chunk_id == "c_lex"

    def test_empty_inputs(self):
        assert hybrid_fuse([], [], limit=10) == []

    def test_limit_respected(self):
        lex = [self._make_hit(f"c{i}", float(i), "tantivy") for i in range(10)]
        vec = [self._make_hit(f"c{i}", float(i), "lancedb") for i in range(10)]
        fused = hybrid_fuse(lex, vec, limit=5)
        assert len(fused) == 5

    def test_score_is_highest_for_best_in_both(self):
        """A chunk that's #1 in both backends should be #1 in fusion."""
        lex = [self._make_hit("best", 10.0), self._make_hit("mid", 5.0)]
        vec = [self._make_hit("best", 0.95), self._make_hit("mid", 0.5)]
        fused = hybrid_fuse(lex, vec, limit=10)
        assert fused[0].chunk_id == "best"


# ---------------------------------------------------------------------------
# Query generation tests
# ---------------------------------------------------------------------------

class TestQueryGeneration:
    @pytest.fixture
    def store(self, tmp_path: Path) -> str:
        """Create a minimal document store with chunks for query generation."""
        db_path = tmp_path / "test_store.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE chunks (
                chunk_id TEXT PRIMARY KEY,
                document_id TEXT,
                content_hash TEXT,
                text TEXT,
                metadata_json TEXT,
                source_span_json TEXT,
                stored_at TEXT
            )
        """)
        # Insert chunks with enough text for term extraction.
        chunks = [
            ("chunk:001", "doc:1", "sha256:a", "The network router configuration protocol uses BGP for routing tables and OSPF for internal paths.", "{}", "{}"),
            ("chunk:002", "doc:1", "sha256:b", "Security policies firewall rules access control lists prevent unauthorized network traffic.", "{}", "{}"),
            ("chunk:003", "doc:2", "sha256:c", "Database indexing strategies improve query performance through B-tree and hash indexes.", "{}", "{}"),
            ("chunk:004", "doc:2", "sha256:d", "Replication and sharding distribute data across multiple database servers for availability.", "{}", "{}"),
            ("chunk:005", "doc:3", "sha256:e", "Cloud infrastructure automation uses Terraform for provisioning and Ansible for configuration.", "{}", "{}"),
        ]
        conn.executemany(
            "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(c[0], c[1], c[2], c[3], c[4], c[5], "2026-01-01") for c in chunks],
        )
        conn.commit()
        conn.close()
        return str(db_path)

    def test_generates_queries(self, store: str):
        queries = generate_query_set(store, n_queries=3, seed=42, min_chunk_len=50)
        assert len(queries) > 0
        for q in queries:
            assert q.query_id.startswith("q")
            assert len(q.query_text) > 0
            assert q.relevant_chunk_id.startswith("chunk:")
            assert q.relevant_document_id.startswith("doc:")

    def test_queries_are_deterministic(self, store: str):
        """Same seed produces same queries."""
        q1 = generate_query_set(store, n_queries=3, seed=42, min_chunk_len=50)
        q2 = generate_query_set(store, n_queries=3, seed=42, min_chunk_len=50)
        assert [q.query_id for q in q1] == [q.query_id for q in q2]
        assert [q.query_text for q in q1] == [q.query_text for q in q2]

    def test_different_seeds_produce_different_queries(self, store: str):
        q1 = generate_query_set(store, n_queries=3, seed=42, min_chunk_len=50)
        q2 = generate_query_set(store, n_queries=3, seed=99, min_chunk_len=50)
        # At least some queries should differ.
        texts1 = {q.query_text for q in q1}
        texts2 = {q.query_text for q in q2}
        assert texts1 != texts2

    def test_query_text_contains_real_terms(self, store: str):
        queries = generate_query_set(store, n_queries=3, seed=42, min_chunk_len=50)
        for q in queries:
            # Query terms should appear in the source text.
            terms = q.query_text.split()
            # At least one term should be in the preview.
            assert any(term.lower() in q.source_text_preview.lower() for term in terms)
