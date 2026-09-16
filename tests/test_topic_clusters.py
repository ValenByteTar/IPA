"""Fase 3 contract tests: TopicCluster schema + store + deterministic clustering."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "validation"))

from ipa.agentic.topic_clusters import (  # noqa: E402
    MERGE_THRESHOLD,
    TopicCluster,
    TopicClusterStore,
    build_clusters,
    cosine,
)
from ipa.tutor.tutor_contracts import GenerationProvenance  # noqa: E402
from validate_agent_contract import validate  # noqa: E402


def _vec(base: list[float], noise: float = 0.0) -> list[float]:
    return [v + noise for v in base]


def _gen():
    from ipa.tutor.tutor_contracts import GenerationProvenance
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    return GenerationProvenance(
        generator="test", generated_at=now,
        input_hash="sha256:" + "a" * 64, model_fingerprint="test",
    )


def _cluster(**overrides):
    base = {
        "cluster_id": "topic_cluster:test001",
        "label": "asyncio / event loop",
        "description": None,
        "member_document_ids": ["doc:aaa", "doc:bbb"],
        "member_concept_ids": [],
        "parent_cluster_id": None,
        "coherence_score": 0.85,
        "representative_chunk_id": "chunk:rep1",
        "created_at": "2026-09-08T12:00:00.000000Z",
        "generation": {
            "generator": "topic-clusterer",
            "generated_at": "2026-09-08T12:00:00.000000Z",
            "input_hash": "sha256:" + "b" * 64,
            "model_fingerprint": "bge-m3-centroids",
        },
        "field_origins": {
            "label": "generated",
            "member_document_ids": "source",
            "coherence_score": "generated",
        },
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Schema validity
# ---------------------------------------------------------------------------

def test_topic_cluster_schema_is_valid_draft2020():
    from jsonschema import Draft202012Validator
    schema = _load_schema("topic_cluster.schema.json")
    Draft202012Validator.check_schema(schema)


def _load_schema(name: str) -> dict:
    import json
    path = Path(__file__).parents[1] / "contracts" / name
    return json.loads(path.read_text(encoding="utf-8"))


def test_topic_cluster_contract_validates():
    from validate_agent_contract import validate
    assert validate("TopicCluster", _cluster()) == []


def test_topic_cluster_requires_member_documents():
    from validate_agent_contract import validate
    errors = validate("TopicCluster", _cluster(member_document_ids=[]))
    assert errors, "empty member_document_ids must be rejected"
    assert any("member_document_ids" in e for e in errors)


def test_topic_cluster_cannot_be_own_parent():
    from validate_agent_contract import validate
    errors = validate("TopicCluster", _cluster(parent_cluster_id="topic_cluster:test001"))
    assert any("own parent" in e for e in errors)


def test_topic_cluster_requires_representative_chunk():
    from validate_agent_contract import validate
    errors = validate("TopicCluster", _cluster(representative_chunk_id=""))
    assert any("representative" in e for e in errors)


# ---------------------------------------------------------------------------
# Store roundtrip
# ---------------------------------------------------------------------------

def _make_cluster(doc_ids, coherence=0.85):
    from ipa.tutor.tutor_contracts import GenerationProvenance
    return TopicCluster(
        cluster_id="topic_cluster:t1",
        label="test cluster",
        description=None,
        member_document_ids=doc_ids,
        member_concept_ids=[],
        parent_cluster_id=None,
        coherence_score=coherence,
        representative_chunk_id="chunk:rep",
        created_at="2026-09-08T12:00:00.000000Z",
        generation=GenerationProvenance(
            generator="t", generated_at="2026-09-08T12:00:00.000000Z",
            input_hash="sha256:" + "a" * 64, model_fingerprint="m",
        ),
        field_origins={"label": "generated", "member_document_ids": "source",
                       "coherence_score": "generated"},
    )


def test_store_roundtrip(tmp_path):
    store = TopicClusterStore(tmp_path / "clusters.db")
    cluster = _make_cluster(["doc:1", "doc:2"])
    store.save_cluster(cluster)
    loaded = store.get_cluster(cluster.cluster_id)
    assert loaded is not None
    assert loaded.label == "test cluster"
    assert loaded.member_document_ids == ["doc:1", "doc:2"]
    assert loaded.coherence_score == 0.85
    store.close()


def test_store_migrates_legacy_four_column_schema(tmp_path):
    """Existing derived stores from before coherence_score remain readable."""
    import sqlite3
    legacy = tmp_path / "legacy.db"
    with sqlite3.connect(legacy) as conn:
        conn.execute("""
            CREATE TABLE topic_clusters (
                cluster_id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
    store = TopicClusterStore(legacy)
    cluster = _make_cluster(["doc:legacy1", "doc:legacy2"])
    store.save_cluster(cluster)
    assert store.get_cluster(cluster.cluster_id) is not None
    assert store.get_cluster(cluster.cluster_id).coherence_score == 0.85
    store.close()


def test_store_find_by_document(tmp_path):
    store = TopicClusterStore(tmp_path / "clusters.db")
    cluster = _make_cluster(["doc:x", "doc:y"])
    store.save_cluster(cluster)
    found = store.find_by_document("doc:y")
    assert found is not None
    assert found.cluster_id == cluster.cluster_id
    assert store.find_by_document("doc:missing") is None
    store.close()


# ---------------------------------------------------------------------------
# Clustering (deterministic)
# ---------------------------------------------------------------------------

def _vec(base_value: float, dim: int = 8) -> list[float]:
    import random
    rng = random.Random(int(base_value * 1000))
    vec = [base_value] * dim
    # Deterministic direction per base_value
    for i in range(dim):
        vec[i] = base_value + rng.uniform(-0.01, 0.01)
    return vec


def test_build_clusters_groups_similar_documents():
    # Three documents about topic A (similar vectors), two about topic B
    docs = {
        "doc:a1": "asyncio event loop coroutine",
        "doc:a2": "asyncio event loop scheduling",
        "doc:a3": "asyncio coroutine await loop",
        "doc:b1": "database index btree query",
        "doc:b2": "database query planner index",
    }
    embeddings = {
        "doc:a1": [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "doc:a2": [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "doc:a3": [0.85, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "doc:b1": [0.0, 0.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
        "doc:b2": [0.0, 0.0, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0],  # low sim to b1
    }
    # Make b1/b2 similar: orthogonal groups
    embeddings["doc:b2"] = [0.0, 0.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0]
    reps = {did: f"chunk:{did}" for did in docs}

    clusters = build_clusters(docs, embeddings, reps, merge_threshold=0.8)
    assert len(clusters) >= 1
    for cluster in clusters:
        assert len(cluster.member_document_ids) >= 2  # no singletons
        assert 0 <= cluster.coherence_score <= 1
        assert cluster.representative_chunk_id


def test_build_clusters_excludes_singletons():
    docs = {
        "doc:a": "asyncio event loop",
        "doc:b": "asyncio coroutine await",
        "doc:c": "totally unrelated quantum chemistry",
    }
    embeddings = {
        "doc:a": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "doc:b": [0.95, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "doc:c": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    }
    reps = {did: f"chunk:{did}" for did in docs}
    clusters = build_clusters(docs, embeddings, reps, merge_threshold=0.8)
    members = [m for c in clusters for m in c.member_document_ids]
    assert "doc:c" not in members  # singleton excluded
    assert {"doc:a", "doc:b"} <= set(members)


def test_build_clusters_is_deterministic():
    docs = {"doc:a": "alpha beta", "doc:b": "alpha gamma", "doc:c": "alpha delta"}
    embeddings = {
        "doc:a": [1.0, 0.1, 0.0], "doc:b": [0.98, 0.12, 0.0], "doc:c": [0.99, 0.11, 0.0],
    }
    reps = {did: f"chunk:{did}" for did in docs}
    c1 = build_clusters(docs, embeddings, reps)
    c2 = build_clusters(docs, embeddings, reps)
    assert [r.label for r in c1] == [r.label for r in c2]
    assert [r.member_document_ids for r in c1] == [r.member_document_ids for r in c2]


def test_build_clusters_parent_hierarchy_is_acyclic():
    # Any parent reference must point to an earlier cluster, so following
    # parents always terminates and cannot form a cycle.
    docs = {
        "doc:a1": "alpha one", "doc:a2": "alpha two",
        "doc:b1": "beta one", "doc:b2": "beta two",
        "doc:c1": "gamma one", "doc:c2": "gamma two",
    }
    embeddings = {
        "doc:a1": [1.0, 0.0, 0.0], "doc:a2": [0.99, 0.1, 0.0],
        "doc:b1": [0.85, 0.5, 0.0], "doc:b2": [0.86, 0.49, 0.0],
        "doc:c1": [0.0, 0.0, 1.0], "doc:c2": [0.0, 0.1, 0.99],
    }
    reps = {did: f"chunk:{did}" for did in docs}
    clusters = build_clusters(docs, embeddings, reps, merge_threshold=0.9, parent_threshold=0.5)
    by_id = {c.cluster_id: c for c in clusters}
    for cluster in clusters:
        seen = set()
        current = cluster
        while current.parent_cluster_id:
            assert current.cluster_id not in seen
            seen.add(current.cluster_id)
            assert current.parent_cluster_id in by_id
            current = by_id[current.parent_cluster_id]


def test_build_clusters_parent_hierarchy():
    # Two tight groups that are moderately similar to each other
    docs = {
        "doc:a1": "asyncio loop", "doc:a2": "asyncio await",
        "doc:b1": "async event scheduling", "doc:b2": "event loop scheduling",
    }
    embeddings = {
        "doc:a1": [1.0, 0.0, 0.0], "doc:a2": [0.97, 0.1, 0.0],
        "doc:b1": [0.85, 0.5, 0.0], "doc:b2": [0.86, 0.49, 0.0],
    }
    reps = {did: f"chunk:{did}" for did in docs}
    clusters = build_clusters(docs, embeddings, reps, merge_threshold=0.9, parent_threshold=0.5)
    # At least one cluster should exist; parents only link to other clusters
    for c in clusters:
        if c.parent_cluster_id is not None:
            assert c.parent_cluster_id != c.cluster_id
            assert any(o.cluster_id == c.parent_cluster_id for o in clusters)


def test_cosine_identical_and_orthogonal():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cluster_contract_validates():
    from validate_agent_contract import validate
    docs = {"doc:a": "alpha beta", "doc:b": "alpha gamma"}
    embeddings = {"doc:a": [1.0, 0.0, 0.0], "doc:b": [0.99, 0.1, 0.0]}
    reps = {did: f"chunk:{did}" for did in docs}
    clusters = build_clusters(docs, embeddings, reps, merge_threshold=0.5)
    assert clusters, "expected at least one cluster"
    for cluster in clusters:
        assert validate("TopicCluster", cluster.to_contract()) == []


# ---------------------------------------------------------------------------
# Vocabulary registration
# ---------------------------------------------------------------------------

def test_vocabulary_registers_topic_cluster():
    import json
    vocab = json.loads((Path(__file__).parents[1] / "contracts" / "contract_vocabulary.json").read_text(encoding="utf-8"))
    assert "TopicCluster" in vocab["records"]
    assert "topic_clusters_are_emergent" in vocab["invariants"]
    assert "topic_clusters_are_derived_not_authoritative" in vocab["invariants"]
    assert "memory_consolidation_requires_human_approval" in vocab["invariants"]
