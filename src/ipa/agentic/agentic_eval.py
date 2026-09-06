"""EXP-003 evaluation: agentic_v1 deep dive versus the legacy path.

Deterministic, dependency-free comparison over an isolated Reporter corpus.
No GPU and no LLM: both paths run with ``provider=None`` so the measured
surface is exactly planning, retrieval, context construction, citation
validation and decline behavior.

Metrics (from the EXP-003 design):
- doc hit@K against a deterministic term-extracted ground truth;
- precision@K over the expected document;
- citation validity via the production ``validate_claims`` machinery plus
  chunk-existence and text-hash checks;
- correct-decline rate on nonsense queries;
- latency p50/p95 per path;
- flag-off equality: two legacy runs must be identical and must not build
  agentic runtime state.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

from ipa.reporter.reporter_claims import citation_summary, validate_claims
from ipa.reporter.reporter_deep_dive import deep_dive
from ipa.storage.document_store import DocumentStore

_WORD = re.compile(r"[a-z][a-z0-9-]{3,}", re.UNICODE)
_STOPWORDS = {
    "about", "after", "also", "because", "been", "being", "between", "both", "cannot", "could",
    "desde", "does", "each", "esta", "este", "esto", "estos", "every", "from", "have", "here",
    "their", "them", "then", "there", "these", "they", "this", "those", "through", "under",
    "were", "what", "when", "where", "which", "while", "will", "with", "within", "without",
    "your", "hacia", "sobre", "entre", "para", "como", "pero", "mais", "aquel", "aquella",
}

DECLINE_QUERIES = [
    "zzq vxq plgh wflbrm",
    "qwrtlp znbmxc fdsjk",
    "pxvkkd wwozzt qqsfhn",
]


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.casefold())


def _load_documents(store: DocumentStore) -> list[dict[str, Any]]:
    grouped: dict[str, list[str]] = {}
    for chunk in store.all_chunks():
        if chunk is None:
            continue
        grouped.setdefault(str(chunk.document_id), []).append(str(chunk.text))
    return [
        {"document_id": document_id, "texts": texts}
        for document_id, texts in grouped.items()
        if texts
    ]


def _titles(corpus_dir: Path) -> dict[str, str]:
    reporter_db = corpus_dir.parent / "reporter.db"
    if not reporter_db.exists():
        return {}
    try:
        with sqlite3.connect(str(reporter_db)) as connection:
            rows = connection.execute("SELECT document_id, title FROM document_metadata").fetchall()
        return {row[0]: str(row[1]) for row in rows if row[0] and row[1]}
    except Exception:
        return {}


def build_queries(
    store: DocumentStore,
    corpus_dir: Path,
    *,
    max_docs: int = 15,
    top_terms: int = 5,
) -> list[dict[str, Any]]:
    """Deterministic ground truth: per-document distinctive-term queries."""
    documents = _load_documents(store)[:max_docs]
    titles = _titles(corpus_dir)
    n_docs = max(1, len(documents))

    df: Counter[str] = Counter()
    doc_tf: list[Counter[str]] = []
    for document in documents:
        tf: Counter[str] = Counter()
        for text in document["texts"]:
            tf.update(_tokens(text))
        doc_tf.append(tf)
        df.update(tf.keys())

    queries: list[dict[str, Any]] = []
    for document, tf in zip(documents, doc_tf):
        scored = {
            term: count * math.log(1.0 + n_docs / df[term])
            for term, count in tf.items()
            if term not in _STOPWORDS and df[term] < n_docs
        }
        top = [term for term, _ in sorted(scored.items(), key=lambda item: (-item[1], item[0]))[:top_terms]]
        if len(top) < 3:
            continue
        expected = document["document_id"]
        queries.append(
            {
                "family": "term",
                "query": " ".join(top),
                "expected_document_id": expected,
            }
        )
        title = titles.get(expected, "")
        title_words = [word for word in _tokens(title) if word not in _STOPWORDS][:4]
        if title_words:
            queries.append(
                {
                    "family": "title",
                    "query": " ".join(title_words + top[:2]),
                    "expected_document_id": expected,
                }
            )
    return queries


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _run_path(
    corpus_dir: Path,
    query: str,
    top_k: int,
    *,
    agentic: bool,
) -> tuple[dict[str, Any], float]:
    start = time.perf_counter()
    result = deep_dive(corpus_dir, query, top_k=top_k, agentic=agentic)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return result, elapsed_ms


def _citation_validity(
    corpus_dir: Path,
    result: dict[str, Any],
    *,
    agentic: bool,
) -> dict[str, Any]:
    store = DocumentStore(corpus_dir / "document_store.db")
    try:
        evidence = result.get("evidence", [])
        chunk_ids = [entry.get("chunk_id") for entry in evidence if isinstance(entry, dict)]
        existing = 0
        for chunk_id in chunk_ids:
            if chunk_id and store.get_chunk(str(chunk_id)) is not None:
                existing += 1
        claims = result.get("claims", [])
        summary = citation_summary(claims)
        total_claims = sum(summary.values())
        hash_checks = 0
        hash_ok = 0
        if agentic:
            runtime = result.get("runtime") or {}
            citation_map = (runtime.get("context") or {}).get("citation_map", {})
            for entry in citation_map.values():
                chunk = store.get_chunk(str(entry.get("chunk_id")))
                if chunk is None:
                    continue
                hash_checks += 1
                digest = "sha256:" + hashlib.sha256(str(chunk.text).encode("utf-8")).hexdigest()
                if entry.get("text_hash") == digest:
                    hash_ok += 1
        return {
            "evidence_chunks": len(chunk_ids),
            "evidence_chunks_exist": existing,
            "claims": total_claims,
            "supported_claims": summary.get("supported", 0),
            "citation_map_entries": hash_checks,
            "citation_hash_matches": hash_ok,
        }
    finally:
        store.close()


def evaluate(
    corpus_dir: str | Path,
    *,
    top_k: int = 5,
    max_docs: int = 15,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    corpus_dir = Path(corpus_dir)
    store = DocumentStore(corpus_dir / "document_store.db")
    try:
        queries = build_queries(store, corpus_dir, max_docs=max_docs)
    finally:
        store.close()

    per_query: list[dict[str, Any]] = []
    latencies: dict[str, list[float]] = {"legacy": [], "agentic": []}
    decline_results: dict[str, list[bool]] = {"legacy": [], "agentic": []}
    equality_failures = 0

    # Warm-up: both paths open the Tantivy index per call; pay the cold-start
    # cost once so latency comparison is not confounded by execution order.
    if queries:
        warm = queries[0]["query"]
        _run_path(corpus_dir, warm, top_k, agentic=False)
        _run_path(corpus_dir, warm, top_k, agentic=True)

    for item in queries:
        row: dict[str, Any] = {
            "family": item["family"],
            "query": item["query"],
            "expected_document_id": item["expected_document_id"],
        }
        runs: dict[str, dict[str, Any]] = {}
        for path in ("legacy", "agentic"):
            result, elapsed_ms = _run_path(corpus_dir, item["query"], top_k, agentic=(path == "agentic"))
            latencies[path].append(elapsed_ms)
            evidence = result.get("evidence", [])
            top_docs = [entry.get("document_id") for entry in evidence[:top_k]]
            expected = item["expected_document_id"]
            validity = _citation_validity(corpus_dir, result, agentic=(path == "agentic"))
            runs[path] = result
            row[path] = {
                "latency_ms": round(elapsed_ms, 2),
                "hit": expected in top_docs,
                "precision_at_k": (sum(1 for doc in top_docs if doc == expected) / max(1, len(top_docs))),
                "n_chunks": len(evidence),
                "runtime_present": result.get("runtime") is not None,
                **validity,
            }
        # Flag-off equality: legacy must be deterministic and free of agentic runtime.
        legacy_again, _ = _run_path(corpus_dir, item["query"], top_k, agentic=False)
        first, second = runs["legacy"], legacy_again
        same = (
            [entry.get("chunk_id") for entry in first.get("evidence", [])]
            == [entry.get("chunk_id") for entry in second.get("evidence", [])]
            and first.get("answer") == second.get("answer")
            and first.get("claims") == second.get("claims")
            and first.get("runtime") is None
        )
        row["flag_off_equal"] = same
        if not same:
            equality_failures += 1
        per_query.append(row)

    for query in DECLINE_QUERIES:
        for path in ("legacy", "agentic"):
            result, _ = _run_path(corpus_dir, query, top_k, agentic=(path == "agentic"))
            decline_results[path].append(result.get("sufficient_evidence") is False)

    def _aggregate(path: str) -> dict[str, Any]:
        rows = [row[path] for row in per_query]
        latency = latencies[path]
        evidence_chunks = sum(row["evidence_chunks"] for row in rows)
        existing = sum(row["evidence_chunks_exist"] for row in rows)
        claims = sum(row["claims"] for row in rows)
        supported = sum(row["supported_claims"] for row in rows)
        map_entries = sum(row["citation_map_entries"] for row in rows)
        hash_ok = sum(row["citation_hash_matches"] for row in rows)
        return {
            "n_queries": len(rows),
            "doc_hit_at_k": sum(1 for row in rows if row["hit"]) / max(1, len(rows)),
            "precision_at_k": sum(row["precision_at_k"] for row in rows) / max(1, len(rows)),
            "evidence_chunk_existence": existing / max(1, evidence_chunks),
            "supported_claim_rate": supported / max(1, claims),
            "citation_hash_match_rate": (hash_ok / max(1, map_entries)) if map_entries else None,
            "decline_correct_rate": (
                sum(1 for ok in decline_results[path] if ok) / max(1, len(decline_results[path]))
            ),
            "latency_ms_p50": round(_percentile(latency, 0.50), 2),
            "latency_ms_p95": round(_percentile(latency, 0.95), 2),
            "latency_ms_mean": round(statistics.fmean(latency), 2) if latency else 0.0,
        }

    report = {
        "experiment": "EXP-003",
        "corpus": str(corpus_dir),
        "top_k": top_k,
        "n_ground_truth_queries": len(per_query),
        "n_decline_queries": len(DECLINE_QUERIES),
        "flag_off_equal": equality_failures == 0,
        "flag_off_failures": equality_failures,
        "legacy": _aggregate("legacy"),
        "agentic": _aggregate("agentic"),
        "per_query": per_query,
        "decline_queries": DECLINE_QUERIES,
    }
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, help="Reporter corpus directory (with tantivy/ and document_store.db)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-docs", type=int, default=15, help="Maximum documents used for ground truth")
    parser.add_argument("--output", default=None, help="Path for the JSON report")
    args = parser.parse_args()

    output = args.output or "outputs/experiments/E13-agentic/report.json"
    report = evaluate(
        args.corpus,
        top_k=args.top_k,
        max_docs=args.max_docs,
        output_path=output,
    )
    print(json.dumps({key: value for key, value in report.items() if key != "per_query"}, ensure_ascii=False, indent=2))
    print(f"report: {output}")
    return 0 if report["flag_off_equal"] else 1


__all__ = ["build_queries", "evaluate", "main"]
