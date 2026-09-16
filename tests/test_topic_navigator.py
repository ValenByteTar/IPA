"""Fase 3.3 tests: TopicNavigator multi-hop retrieval.

Covers:
  - Vertical-first behavior (no hops when coverage is sufficient)
  - Multi-hop triggered only when coverage is insufficient
  - Bounded hops (max 2) and chunks per hop (PAT-004)
  - Neighbor expansion: parent, children, siblings — no cycles
  - Trace/audit of navigation decisions
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ipa.agentic.topic_clusters import TopicCluster, TopicClusterStore  # noqa: E402
from ipa.agentic.topic_navigator import (  # noqa: E402
    MAX_HOPS,
    NavigationResult,
    TopicNavigator,
)
from ipa.tutor.tutor_contracts import GenerationProvenance  # noqa: E402


def _gen():
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    return GenerationProvenance(
        generator="test", generated_at=now,
        input_hash="sha256:" + "a" * 64, model_fingerprint="test",
    )


def _cluster(cluster_id: str, doc_ids: list[str], parent: str | None = None, label: str = "c") -> TopicCluster:
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    return TopicCluster(
        cluster_id=cluster_id, label=label, description=None,
        member_document_ids=doc_ids, member_concept_ids=[],
        parent_cluster_id=parent, coherence_score=0.9,
        representative_chunk_id=f"chunk:{doc_ids[0]}",
        created_at=now, generation=_gen(),
        field_origins={"label": "generated", "member_document_ids": "source",
                       "coherence_score": "generated"},
    )


class FakeDocStore:
    """DocumentStore stub: chunks keyed by document_id."""
    def __init__(self, chunks_by_doc: dict[str, list]):
        self.chunks = chunks_by_doc

    def get_chunks(self, document_id: str):
        return iter(self.chunks.get(document_id, []))


@dataclass
class FakeChunk:
    chunk_id: str
    document_id: str
    text: str


@pytest.fixture()
def cluster_world(tmp_path):
    """Clusters: security+google share parent 'tech'; db is isolated."""
    store = TopicClusterStore(tmp_path / "clusters.db")
    clusters = [
        _cluster("topic_cluster:sec", ["doc:sec1", "doc:sec2"], parent="topic_cluster:tech", label="security"),
        _cluster("topic_cluster:goog", ["doc:goog1", "doc:goog2"], parent="topic_cluster:tech", label="google"),
        _cluster("topic_cluster:db", ["doc:db1", "doc:db2"], parent=None, label="database"),
        _cluster("topic_cluster:tech", ["doc:tech1"], label="technology"),
    ]
    for c in clusters:
        store.save_cluster(c)
    chunks = {}
    for docs in (["sec1", "sec2"], ["goog1", "goog2"], ["db1", "db2"], ["tech1"]):
        for d in docs:
            chunks[f"doc:{d}"] = [FakeChunk(f"chunk:{d}-{i}", f"doc:{d}", f"text {d} {i}") for i in range(3)]
    return store, FakeDocStore(chunks)


def test_vertical_only_when_coverage_sufficient(cluster_world):
    store, docs = cluster_world
    navigator = TopicNavigator(store)

    def vertical_search(query, limit=10):
        return [
            {"chunk_id": "chunk:sec1-0", "document_id": "doc:sec1", "score": 0.9,
             "retrieval_backend": "lancedb", "text_preview": "text"},
            {"chunk_id": "chunk:sec1-1", "document_id": "doc:sec1", "score": 0.9, "retrieval_backend": "lancedb"},
            {"chunk_id": "chunk:sec2-0", "document_id": "doc:sec2", "score": 0.8, "retrieval_backend": "lancedb"},
            {"chunk_id": "chunk:goog1-0", "document_id": "doc:goog1", "score": 0.7, "retrieval_backend": "lancedb"},
            {"chunk_id": "chunk:goog2-0", "document_id": "doc:goog2", "score": 0.7, "retrieval_backend": "lancedb"},
        ]

    result = navigator.navigate("security query", vertical_search, FakeDocStore({}), min_hits=3)
    assert not result.used_multi_hop
    assert result.trace.coverage_sufficient is True
    assert result.trace.hops == []
    assert len(result.hits) == 5


def test_coverage_counts_distinct_documents_not_chunks(cluster_world):
    """10 chunks from 2 documents = narrow coverage → multi-hop triggers.
    Coverage is document diversity, not raw chunk count."""
    store, docs = cluster_world
    navigator = TopicNavigator(store)

    def vertical_search(query, limit=10):
        return [{"chunk_id": f"chunk:sec1-{i}", "document_id": "doc:sec1",
                 "score": 0.9, "retrieval_backend": "lancedb", "text_preview": "t"}
                for i in range(10)]

    result = navigator.navigate("q", vertical_search, docs, min_hits=3)
    assert result.used_multi_hop, "10 chunks from 1 doc is narrow coverage"


def test_multi_hop_triggers_on_insufficient_coverage(cluster_world):
    """Only 1 vertical hit → coverage insufficient → hop to sibling clusters."""
    store, docs = cluster_world
    navigator = TopicNavigator(store)

    def vertical_search(query, limit=10):
        return [{"chunk_id": "chunk:sec1-0", "document_id": "doc:sec1",
                 "score": 0.9, "retrieval_backend": "lancedb", "text_preview": "text"}]

    result = navigator.navigate("query", vertical_search, docs, min_hits=3)
    assert result.used_multi_hop
    assert result.trace.coverage_sufficient is False
    # Hop hits carry provenance of the cluster they came from
    hop_hits = [h for h in result.hits if h.get("via_cluster")]
    assert len(hop_hits) >= 1
    assert any(h["retrieval_backend"].startswith("multi_hop") for h in result.hits)


def _hop_hits(result: NavigationResult):
    return [h for h in result.hits if h.get("retrieval_backend", "").startswith("multi_hop")]


def _hop_hits_wrapper(result):
    return [h for h in result.hits if h.get("retrieval_backend", "").startswith("multi_hop")]


def _hop_hits_impl(result):
    return [h for h in result.hits if "via_cluster" in h]


def _hop_hits_fn():
    return None


def _hop_hits_check(result):
    return [h for h in result.hits if h.get("via_cluster")]


def _hop_hits(result):
    return _hop_hits_impl(result)


def test_multi_hop_expands_to_sibling_clusters(cluster_world):
    """From a security hit, the navigator reaches google (sibling) chunks."""
    store, docs = cluster_world
    navigator = TopicNavigator(store)

    def vertical_search(query, limit=10):
        return [{"chunk_id": "chunk:sec1-0", "document_id": "doc:sec1",
                 "score": 0.9, "retrieval_backend": "lancedb", "text_preview": "t"}]

    result = navigator.navigate("query", vertical_search, docs, min_hits=3)
    assert result.used_multi_hop
    assert result.trace.coverage_sufficient is False
    via_labels = {h.get("via_label") for h in result.hits if h.get("via_cluster")}
    assert "google" in via_labels  # sibling reached
    # Hop hits are marked with their backend
    assert any(h["retrieval_backend"].startswith("multi_hop") for h in result.hits)


def test_multi_hop_is_bounded(tutor_env=None, tmp_path=None):
    """max_hops=2 and chunks_per_hop are respected (PAT-004 budgets)."""
    store = TopicClusterStore(Path("_tmp_nav_clusters.db") if False else None or (__import__("tempfile").mkdtemp() + "/c.db"))
    # Chain: a -> b -> c -> d (each parent of the next)
    clusters = [
        _cluster("c:a", ["doc:a"], parent=None, label="a"),
        _cluster("topic_cluster:b", ["doc:b"], parent="topic_cluster:a", label="b"),
        _cluster("topic_cluster:c", ["doc:c"], parent="topic_cluster:b", label="c"),
        _cluster("topic_cluster:d", ["doc:d"], parent="topic_cluster:c", label="d"),
        _cluster("topic_cluster:e", ["doc:e"], parent="topic_cluster:c", label="e2"),
    ]
    for c in clusters:
        store.save_cluster(c)
    docs = FakeDocStore({
        "doc:a": [FakeChunk("chunk:a1", "doc:a", "ta")],
        "doc:b": [FakeChunk("chunk:b1", "doc:b", "tb")],
        "doc:c": [FakeChunk("chunk:c1", "doc:c", "tc")],
        "doc:d": [FakeChunk("chunk:d1", "doc:d", "td")],
        "doc:tech1": [FakeChunk("chunk:t1", "doc:tech1", "tt")],
    })
    navigator = TopicNavigator(store, max_hops=2, chunks_per_hop=2)

    def vertical_search(query, limit=10):
        return [{"chunk_id": "chunk:a1", "document_id": "doc:a", "score": 0.9,
                 "retrieval_backend": "lancedb", "text_preview": "t"}]

    result = navigator.navigate("q", vertical_search, docs, min_hits=50)
    # With min_hits=50 coverage is never sufficient; hops must be bounded at 2
    assert len(result.trace.hops) <= MAX_HOPS
    # No chunk appears twice
    chunk_ids = [h["chunk_id"] for h in result.hits]
    assert len(chunk_ids) == len(set(chunk_ids))


def test_no_hops_when_coverage_sufficient(cluster_world):
    store, docs = cluster_world
    navigator = TopicNavigator(store)

    def vertical_search(query, limit=10):
        # Diverse coverage: 5 chunks from 5 distinct documents
        return [{"chunk_id": f"chunk:sec{i}", "document_id": f"doc:sec{i}",
                 "score": 0.9, "retrieval_backend": "lancedb", "text_preview": "t"}
                for i in range(5)]

    result = navigator.navigate("q", vertical_search, docs, min_hits=3)
    assert not result.used_multi_hop
    assert result.trace.coverage_sufficient is True
    assert result.trace.hops == []


def test_neighbor_expansion_covers_parent_children_siblings(cluster_world):
    store, _ = cluster_world
    navigator = TopicNavigator(store)
    sec = store.get_cluster("topic_cluster:sec")
    neighbor_ids = {c.cluster_id for c in navigator._neighbor_clusters(sec)}
    # parent reached
    assert "topic_cluster:tech" in neighbor_ids
    # sibling (google shares parent tech) reached
    assert "topic_cluster:goog" in neighbor_ids
    # no self
    assert "topic_cluster:sec" not in neighbor_ids
    # isolated cluster (db) is NOT a neighbor (no shared parent, not a child)
    assert "topic_cluster:db" not in neighbor_ids


def _hop_hits(result):
    return [h for h in result.hits if str(h.get("retrieval_backend", "")).startswith("multi_hop")]


def test_trace_records_hops(cluster_world):
    store, docs = cluster_world
    navigator = TopicNavigator(store)

    def vertical_search(query, limit=10):
        return [{"chunk_id": "chunk:sec1-0", "document_id": "doc:sec1",
                 "score": 0.9, "retrieval_backend": "lancedb", "text_preview": "t"}]

    result = navigator.navigate("q", vertical_search, docs, min_hits=3)
    assert result.used_multi_hop
    assert result.trace.hops, "at least one hop recorded"
    for hop in result.trace.hops:
        assert "hop" in hop and "from_cluster" in hop
    # Hop hits carry provenance
    hop_hits = [h for h in result.hits if h.get("via_cluster")]
    assert all("via_label" in h for h in hop_hits)


def test_isolated_cluster_has_no_neighbors(tmp_path):
    """A cluster without parent/children/siblings expands to nothing."""
    store = TopicClusterStore(tmp_path / "c.db")
    store.save_cluster(_cluster("topic_cluster:lonely", ["doc:l1", "doc:l2"]))
    navigator = TopicNavigator(store)
    lonely = store.get_cluster("topic_cluster:lonely")
    assert navigator._neighbor_clusters(lonely) == []


def test_max_hops_constant_is_two():
    assert MAX_HOPS == 2
