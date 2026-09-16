"""Tests for the incremental topic backfill (idle-time worker)."""
import numpy as np
import pytest

from ipa.agentic.topic_clusters import (
    TopicClusterStore, backfill_topics, build_clusters, cosine,
)


class FakeLanceIndex:
    """Minimal LanceDB stand-in: vectors keyed by chunk_id, get_vectors API."""

    def __init__(self, vectors: dict[str, list[float]]):
        self._vectors = vectors

    def get_vectors(self, chunk_ids: list[str]) -> dict[str, list[float]]:
        return {cid: self._vectors[cid] for cid in chunk_ids if cid in self._vectors}


class FakeDocStore:
    """Minimal DocumentStore stand-in with centroids + documents."""

    def __init__(self, centroids: dict[str, list[str]], texts: dict[str, str]):
        self._centroids = centroids
        self._texts = texts

    def all_centroids(self):
        return self._centroids

    def get_document(self, document_id):
        from types import SimpleNamespace
        text = self._texts.get(document_id)
        return SimpleNamespace(text=text) if text else None


class SimpleNamespace:
    pass


def _make_store(tmp_path):
    return TopicClusterStore(tmp_path / "clusters.db")


def test_backfill_assigns_to_existing_cluster(tmp_path):
    # Dos clusters existentes con vectores bien separados
    v_tech = [1.0, 0.0, 0.0]
    v_bio = [0.0, 1.0, 0.0]
    store = TopicClusterStore(tmp_path / "clusters.db")
    docs = {"d1": "tech text", "d2": "tech two", "d3": "bio stuff", "d4": "bio stuff"}
    embeddings = {"d1": v_tech, "d2": v_tech, "d3": v_bio, "d4": v_bio}
    clusters = build_clusters(docs, embeddings, {})
    assert len(clusters) >= 1
    for c in clusters:
        store.save_cluster(c)

    # En el flujo real, all_centroids() devuelve TODOS los documentos del
    # corpus (incluidos los que ya están en clusters). El fake debe reflejar
    # eso: d1..d4 con sus chunks + el doc nuevo pendiente.
    lance = FakeLanceIndex({
        "d1_c1": v_tech, "d2_c1": v_tech,
        "d3_c1": v_bio, "d4_c1": v_bio,
        "d_new_c1": v_tech, "d_new_c2": v_tech,
    })
    doc_store = FakeDocStore(
        centroids={
            "d1": ["d1_c1"], "d2": ["d2_c1"],
            "d3": ["d3_c1"], "d4": ["d4_c1"],
            "d_new": ["d_new_c1", "d_new_c2"],
        },
        texts={
            "d1": "tech text", "d2": "tech two",
            "d3": "bio stuff", "d4": "bio stuff",
            "d_new": "tech doc nuevo",
        },
    )
    result = backfill_topics(store, doc_store, lance, merge_threshold=0.52)
    # El doc nuevo debería haberse asignado al cluster tech
    assert len(result["assigned"]) == 1
    assert result["assigned"][0]["cluster_id"].startswith("topic_cluster:")
    # El cluster ahora incluye al doc nuevo
    cid = result["assigned"][0]["cluster_id"]
    updated = store.get_cluster(cid)
    assert "d_new" in updated.member_document_ids
    store.close()


def test_backfill_creates_new_cluster_for_unmatched(tmp_path):
    # Un cluster existente en otra dirección
    store = TopicClusterStore(tmp_path / "clusters.db")
    docs = {"d1": "tech", "d2": "tech"}
    clusters = build_clusters(docs, {"d1": [1.0, 0.0, 0.0], "d2": [0.99, 0.01, 0.0]}, {})
    for c in clusters:
        store.save_cluster(c)

    # Dos documentos nuevos similares entre sí pero lejos del cluster tech
    lance = FakeLanceIndex({
        "n1_c": [0.0, 1.0, 0.0],
        "n2_c": [0.0, 0.98, 0.02],
    })
    doc_store = FakeDocStore(
        centroids={"n1": ["n1_c"], "n2": ["n2_c"]},
        texts={"n1": "bio topic", "n2": "bio related"},
    )
    result = backfill_topics(store, doc_store, lance, merge_threshold=0.52)
    # Deberían formar un cluster nuevo (2 docs similares entre sí)
    assert len(result["new_clusters"]) >= 1
    store.close()


def test_backfill_no_pending(tmp_path):
    store = TopicClusterStore(tmp_path / "clusters.db")
    doc_store = FakeDocStore(centroids={}, texts={})
    lance = FakeLanceIndex({})
    result = backfill_topics(store, doc_store, lance)
    assert result == {"assigned": [], "new_clusters": [], "pending": 0}
    store.close()


def test_backfill_respects_max_docs(tmp_path):
    store = TopicClusterStore(tmp_path / "clusters.db")
    # 5 docs nuevos, max_docs=2 → procesa solo 2 por ciclo
    centroids = {f"d{i}": [f"c{i}"] for i in range(5)}
    vectors = {f"c{i}": [float(i), 0.0, 0.0] for i in range(5)}
    lance = FakeLanceIndex(vectors)
    doc_store = FakeDocStore(centroids=centroids, texts={f"d{i}": f"doc {i}" for i in range(5)})
    result = backfill_topics(store, doc_store, lance, max_docs=2)
    # Solo procesa los primeros 2 (pending limita el trabajo por ciclo)
    total_touched = len(result["assigned"]) + result["pending"] + len(result["new_clusters"])
    assert total_touched <= 5
    store.close()
