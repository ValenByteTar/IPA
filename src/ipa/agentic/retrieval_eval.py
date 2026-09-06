"""E10 â€” End-to-end retrieval evaluation.

Compares retrieval backends (Tantivy lexical, LanceDB vector, hybrid fusion)
on the full 166k chunk corpus using a synthetic query set with ground truth.

Query generation strategy:
  - Sample N chunks from the corpus (deterministic, seeded)
  - Extract the top 3-5 TF-IDF terms from each chunk as the query
  - The source chunk is the ground-truth relevant result (chunk-level)
  - The source document is the ground-truth relevant document (doc-level)

Metrics:
  - recall@k: fraction of queries where the relevant chunk is in top-k
  - MRR: Mean Reciprocal Rank of the relevant chunk
  - nDCG@k: normalized discounted cumulative gain
  - document recall@k: fraction where any chunk from the same document is in top-k
  - p50/p95 latency per query

Candidates:
  - Tantivy (lexical BM25) â€” E6 winner
  - LanceDB (vector cosine) â€” E7 winner
  - Hybrid (weighted score fusion of both)
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ipa.contracts import SearchHit


# ---------------------------------------------------------------------------
# Query set generation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvalQuery:
    """A single evaluation query with ground truth."""
    query_id: str
    query_text: str
    relevant_chunk_id: str
    relevant_document_id: str
    source_text_preview: str


def generate_query_set(
    store_db: str,
    n_queries: int = 200,
    seed: int = 42,
    min_chunk_len: int = 100,
    terms_per_query: int = 5,
) -> list[EvalQuery]:
    """Generate a synthetic query set from the corpus.

    For each query, we:
      1. Pick a random chunk (seeded, min length filter)
      2. Tokenize and compute term frequencies
      3. Select the top terms by TF (proxy for TF-IDF without global IDF)
      4. Use those terms as the query string
      5. Record the source chunk_id and document_id as ground truth

    The query is "easy" (terms come from the chunk itself) but tests
    whether the retrieval backend can find the exact source chunk among
    166k candidates â€” a non-trivial task for both lexical and vector search.
    """
    rng = random.Random(seed)
    conn = sqlite3.connect(store_db)

    # Get total chunk count for sampling.
    total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    # Sample chunk indices deterministically.
    # We scan in random order and pick chunks that meet the min length.
    offsets = list(range(0, total, max(1, total // (n_queries * 5))))
    rng.shuffle(offsets)

    queries: list[EvalQuery] = []
    for offset in offsets:
        if len(queries) >= n_queries:
            break
        row = conn.execute(
            "SELECT chunk_id, document_id, text FROM chunks "
            "WHERE rowid = (SELECT rowid FROM chunks LIMIT 1 OFFSET ?)",
            (offset,),
        ).fetchone()
        if row is None:
            continue
        chunk_id, doc_id, text = row
        if len(text) < min_chunk_len:
            continue

        # Extract top terms by frequency (filter stopwords and short tokens).
        terms = _extract_top_terms(text, terms_per_query)
        if len(terms) < 2:
            continue

        query_text = " ".join(terms)
        queries.append(EvalQuery(
            query_id=f"q{len(queries):04d}",
            query_text=query_text,
            relevant_chunk_id=chunk_id,
            relevant_document_id=doc_id,
            source_text_preview=text[:120],
        ))

    conn.close()
    return queries


_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "can", "this", "that",
    "these", "those", "i", "you", "he", "she", "it", "we", "they",
    "what", "which", "who", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "just", "also", "as", "if", "then", "else", "about",
})


def _extract_top_terms(text: str, n: int) -> list[str]:
    """Extract the n most frequent non-stopword tokens from text."""
    import re
    tokens = re.findall(r'[a-zA-Z]{3,}', text.lower())
    freq: dict[str, int] = {}
    for tok in tokens:
        if tok in _STOPWORDS:
            continue
        freq[tok] = freq.get(tok, 0) + 1
    # Sort by frequency descending, then alphabetically for determinism.
    sorted_terms = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return [term for term, _ in sorted_terms[:n]]


# ---------------------------------------------------------------------------
# IR Metrics
# ---------------------------------------------------------------------------

def recall_at_k(
    ranked_chunk_ids: list[str], relevant_chunk_id: str, k: int
) -> float:
    """Recall@k: 1.0 if relevant chunk is in top-k, 0.0 otherwise."""
    if relevant_chunk_id in ranked_chunk_ids[:k]:
        return 1.0
    return 0.0


def document_recall_at_k(
    ranked_doc_ids: list[str], relevant_doc_id: str, k: int
) -> float:
    """Document recall@k: 1.0 if any chunk from the relevant document is in top-k."""
    if relevant_doc_id in ranked_doc_ids[:k]:
        return 1.0
    return 0.0


def reciprocal_rank(
    ranked_chunk_ids: list[str], relevant_chunk_id: str
) -> float:
    """Reciprocal rank: 1/rank of the relevant chunk, 0 if not found."""
    for i, cid in enumerate(ranked_chunk_ids):
        if cid == relevant_chunk_id:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(
    ranked_chunk_ids: list[str], relevant_chunk_id: str, k: int
) -> float:
    """nDCG@k with binary relevance: 1 for the relevant chunk, 0 for others."""
    dcg = 0.0
    for i, cid in enumerate(ranked_chunk_ids[:k]):
        if cid == relevant_chunk_id:
            dcg = 1.0 / math.log2(i + 2)  # +2 because log2(1)=0
            break
    # Ideal DCG: relevant chunk at position 1.
    idcg = 1.0 / math.log2(2)  # = 1.0
    return dcg / idcg if idcg > 0 else 0.0


@dataclass
class RetrievalMetrics:
    """Aggregated metrics for one retrieval backend."""
    backend: str
    n_queries: int = 0
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    recall_at_20: float = 0.0
    doc_recall_at_1: float = 0.0
    doc_recall_at_5: float = 0.0
    doc_recall_at_10: float = 0.0
    doc_recall_at_20: float = 0.0
    mrr: float = 0.0
    ndcg_at_10: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


def compute_metrics(
    backend: str,
    results: list[tuple[str, list[str], list[str], float]],
) -> RetrievalMetrics:
    """Compute aggregated metrics from per-query results.

    Args:
        backend: Name of the retrieval backend.
        results: List of (query_id, ranked_chunk_ids, ranked_doc_ids, latency_ms)
                 for each query.

    Returns:
        Aggregated RetrievalMetrics.
    """
    if not results:
        return RetrievalMetrics(backend=backend)

    # We need the ground truth to compute metrics â€” but results don't have it.
    # This function is called with pre-matched results.
    # Actually, let's restructure: the caller passes per-query metrics.
    raise NotImplementedError("Use compute_metrics_from_hits instead")


def compute_metrics_from_hits(
    backend: str,
    per_query: list[dict[str, Any]],
) -> RetrievalMetrics:
    """Compute aggregated metrics from per-query hit data.

    Each entry in per_query must have:
        - ranked_chunk_ids: list[str]
        - ranked_doc_ids: list[str]
        - relevant_chunk_id: str
        - relevant_doc_id: str
        - latency_ms: float
    """
    n = len(per_query)
    if n == 0:
        return RetrievalMetrics(backend=backend)

    r1 = sum(recall_at_k(q["ranked_chunk_ids"], q["relevant_chunk_id"], 1) for q in per_query) / n
    r5 = sum(recall_at_k(q["ranked_chunk_ids"], q["relevant_chunk_id"], 5) for q in per_query) / n
    r10 = sum(recall_at_k(q["ranked_chunk_ids"], q["relevant_chunk_id"], 10) for q in per_query) / n
    r20 = sum(recall_at_k(q["ranked_chunk_ids"], q["relevant_chunk_id"], 20) for q in per_query) / n

    dr1 = sum(document_recall_at_k(q["ranked_doc_ids"], q["relevant_doc_id"], 1) for q in per_query) / n
    dr5 = sum(document_recall_at_k(q["ranked_doc_ids"], q["relevant_doc_id"], 5) for q in per_query) / n
    dr10 = sum(document_recall_at_k(q["ranked_doc_ids"], q["relevant_doc_id"], 10) for q in per_query) / n
    dr20 = sum(document_recall_at_k(q["ranked_doc_ids"], q["relevant_doc_id"], 20) for q in per_query) / n

    mrr = sum(reciprocal_rank(q["ranked_chunk_ids"], q["relevant_chunk_id"]) for q in per_query) / n
    ndcg10 = sum(ndcg_at_k(q["ranked_chunk_ids"], q["relevant_chunk_id"], 10) for q in per_query) / n

    latencies = sorted(q["latency_ms"] for q in per_query)
    p50 = latencies[n // 2]
    p95_idx = int(n * 0.95)
    p95 = latencies[min(p95_idx, n - 1)]
    avg_lat = sum(latencies) / n

    return RetrievalMetrics(
        backend=backend,
        n_queries=n,
        recall_at_1=round(r1, 4),
        recall_at_5=round(r5, 4),
        recall_at_10=round(r10, 4),
        recall_at_20=round(r20, 4),
        doc_recall_at_1=round(dr1, 4),
        doc_recall_at_5=round(dr5, 4),
        doc_recall_at_10=round(dr10, 4),
        doc_recall_at_20=round(dr20, 4),
        mrr=round(mrr, 4),
        ndcg_at_10=round(ndcg10, 4),
        p50_latency_ms=round(p50, 2),
        p95_latency_ms=round(p95, 2),
        avg_latency_ms=round(avg_lat, 2),
    )


# ---------------------------------------------------------------------------
# Hybrid retrieval (score fusion)
# ---------------------------------------------------------------------------

def hybrid_fuse(
    lexical_hits: list[SearchHit],
    vector_hits: list[SearchHit],
    lexical_weight: float = 0.5,
    vector_weight: float = 0.5,
    limit: int = 20,
) -> list[SearchHit]:
    """Fuse lexical and vector search results using weighted score fusion.

    Normalizes scores to [0, 1] within each backend's result set, then
    combines with the given weights.  Deduplicates by chunk_id, keeping
    the highest fused score.
    """
    if not lexical_hits and not vector_hits:
        return []

    # Normalize scores within each backend.
    def normalize(hits: list[SearchHit]) -> dict[str, float]:
        if not hits:
            return {}
        scores = [h.score for h in hits]
        lo, hi = min(scores), max(scores)
        if hi == lo:
            return {h.chunk_id: 1.0 for h in hits}
        denom = hi - lo
        return {h.chunk_id: (h.score - lo) / denom for h in hits}

    lex_norm = normalize(lexical_hits)
    vec_norm = normalize(vector_hits)

    # Combine scores.
    all_chunk_ids = set(lex_norm.keys()) | set(vec_norm.keys())
    fused: list[tuple[str, float, SearchHit]] = []
    for cid in all_chunk_ids:
        lex_score = lex_norm.get(cid, 0.0) * lexical_weight
        vec_score = vec_norm.get(cid, 0.0) * vector_weight
        combined = lex_score + vec_score
        # Pick the SearchHit with more info (prefer lexical for span).
        hit = next((h for h in lexical_hits if h.chunk_id == cid), None)
        if hit is None:
            hit = next((h for h in vector_hits if h.chunk_id == cid), None)
        if hit is not None:
            fused.append((cid, combined, hit))

    fused.sort(key=lambda x: -x[1])
    return [
        SearchHit(
            chunk_id=cid,
            score=score,
            source_span=hit.source_span,
            retrieval_backend="hybrid",
        )
        for cid, score, hit in fused[:limit]
    ]

