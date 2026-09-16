"""Tests for the idle enrichment module (Level 1 deterministic enrichment)."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipa.agentic.idle_enrichment import (
    build_document_dicts,
    enrich_corpus_level1,
)
from ipa.agentic.topic_clusters import TopicClusterStore


class FakeLanceIndex:
    """Minimal LanceDB stand-in: document_embeddings + get_vectors."""

    def __init__(self, doc_embeddings: dict[str, list[float]] | None = None,
                 chunk_vectors: dict[str, list[float]] | None = None):
        self._doc_embeddings = doc_embeddings or {}
        self._chunk_vectors = chunk_vectors or {}

    def document_embeddings(self) -> dict[str, list[float]]:
        return self._doc_embeddings

    def get_vectors(self, chunk_ids):
        return {cid: self._chunk_vectors[cid] for cid in chunk_ids if cid in self._chunk_vectors}

    def close(self):
        pass


class FakeDocStore:
    """Minimal DocumentStore stand-in with centroids + documents + chunks."""

    def __init__(self, centroids: dict[str, list[str]],
                 texts: dict[str, str],
                 chunk_texts: dict[str, str] | None = None,
                 sources: dict[str, dict] | None = None):
        self._centroids = centroids
        self._texts = texts
        self._chunk_texts = chunk_texts or {}
        self._sources = sources or {}

    def all_centroids(self):
        return self._centroids

    def get_document(self, document_id):
        text = self._texts.get(document_id)
        if text is None:
            return None
        return SimpleNamespace(text=text, document_id=document_id)

    def get_chunk(self, chunk_id):
        text = self._chunk_texts.get(chunk_id)
        if text is None:
            return None
        return SimpleNamespace(chunk_id=chunk_id, text=text)

    def all_sources(self):
        return self._sources

    def close(self):
        pass


@pytest.fixture
def fake_corpus(tmp_path):
    """Create a fake corpus directory with store + lance paths."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "document_store.db").write_text("fake")
    (corpus / "vector" / "lancedb").mkdir(parents=True)
    return corpus


def test_build_document_dicts_basic():
    """build_document_dicts produces dicts with required fields."""
    store = FakeDocStore(
        centroids={"d1": ["d1_c1"], "d2": ["d2_c1"]},
        texts={"d1": "Title line\nBody text", "d2": "Another title\nMore body"},
        chunk_texts={"d1_c1": "chunk text 1", "d2_c1": "chunk text 2"},
    )
    lance = FakeLanceIndex(
        doc_embeddings={"d1": [1.0, 0.0], "d2": [0.0, 1.0]},
    )
    documents, embeddings = build_document_dicts(store, lance)
    assert len(documents) == 2
    assert documents[0]["document_id"] in ("d1", "d2")
    assert "title" in documents[0]
    assert "text" in documents[0]
    assert "representation_text" in documents[0]
    assert len(embeddings) == 2


def test_build_document_dicts_derives_title_from_first_line():
    """Title is derived from the first non-empty line of text."""
    store = FakeDocStore(
        centroids={"d1": ["d1_c1"]},
        texts={"d1": "\n\nMy Great Title\nBody content"},
    )
    lance = FakeLanceIndex(doc_embeddings={"d1": [1.0]})
    documents, _ = build_document_dicts(store, lance)
    assert documents[0]["title"] == "My Great Title"


def test_build_document_dicts_empty_text():
    """Documents with empty text get document_id as title."""
    store = FakeDocStore(
        centroids={"d1": ["d1_c1"]},
        texts={"d1": ""},
    )
    lance = FakeLanceIndex(doc_embeddings={"d1": [1.0]})
    documents, _ = build_document_dicts(store, lance)
    assert documents[0]["title"] == "d1"


def test_enrich_level1_no_corpus(tmp_path):
    """Level 1 returns skipped when corpus doesn't exist."""
    cluster_store = TopicClusterStore(tmp_path / "clusters.db")
    result = enrich_corpus_level1(tmp_path / "nonexistent", cluster_store)
    assert result["topics_new"] == 0
    assert "skipped" in result
    cluster_store.close()


def test_enrich_level1_creates_topics(fake_corpus, monkeypatch, tmp_path):
    """Level 1 creates topics from unclustered documents."""
    # We need to monkeypatch DocumentStore and LanceDBIndex since the
    # fake corpus has dummy files, not real SQLite/LanceDB.
    v_tech = [1.0, 0.0, 0.0]
    v_tech2 = [0.99, 0.01, 0.0]
    v_bio = [0.0, 1.0, 0.0]
    v_bio2 = [0.0, 0.98, 0.02]

    fake_store = FakeDocStore(
        centroids={
            "d1": ["d1_c1"], "d2": ["d2_c1"],
            "d3": ["d3_c1"], "d4": ["d4_c1"],
        },
        texts={
            "d1": "NVIDIA GPU architecture blog post",
            "d2": "CUDA programming guide update",
            "d3": "Biology research on cells",
            "d4": "Medical study on proteins",
        },
        chunk_texts={
            "d1_c1": "NVIDIA GPU architecture",
            "d2_c1": "CUDA programming",
            "d3_c1": "Biology cells",
            "d4_c1": "Medical proteins",
        },
    )
    fake_lance = FakeLanceIndex(
        doc_embeddings={"d1": v_tech, "d2": v_tech2, "d3": v_bio, "d4": v_bio2},
        chunk_vectors={
            "d1_c1": v_tech, "d2_c1": v_tech2,
            "d3_c1": v_bio, "d4_c1": v_bio2,
        },
    )

    import ipa.agentic.idle_enrichment as ie
    monkeypatch.setattr(ie, "DocumentStore", lambda path: fake_store, raising=False)
    # The import is inside the function, so we need to patch at the module level
    # that the function imports from. Let's patch the ipa module's DocumentStore.
    import ipa
    monkeypatch.setattr(ipa, "DocumentStore", lambda path: fake_store)
    monkeypatch.setattr(
        "ipa.indexes.lancedb_index.LanceDBIndex",
        lambda path: fake_lance,
    )

    cluster_store = TopicClusterStore(tmp_path / "test_clusters.db")
    try:
        result = enrich_corpus_level1(fake_corpus, cluster_store)
        assert result["topics_new"] >= 1
        # Verify clusters were saved
        clusters = cluster_store.list_clusters()
        assert len(clusters) >= 1
    finally:
        cluster_store.close()


def test_enrich_level1_all_clustered(fake_corpus, monkeypatch, tmp_path):
    """Level 1 still runs curation when all docs are already clustered."""
    fake_store = FakeDocStore(
        centroids={"d1": ["d1_c1"], "d2": ["d2_c1"]},
        texts={"d1": "doc one", "d2": "doc two"},
        chunk_texts={"d1_c1": "chunk 1", "d2_c1": "chunk 2"},
    )
    fake_lance = FakeLanceIndex(
        doc_embeddings={"d1": [1.0, 0.0], "d2": [0.0, 1.0]},
    )

    import ipa
    monkeypatch.setattr(ipa, "DocumentStore", lambda path: fake_store)
    monkeypatch.setattr(
        "ipa.indexes.lancedb_index.LanceDBIndex",
        lambda path: fake_lance,
    )

    # Pre-populate the cluster store with all docs
    cluster_store = TopicClusterStore(tmp_path / "test_clusters.db")
    try:
        from ipa.agentic.topic_clusters import build_clusters
        clusters = build_clusters(
            {"d1": "doc one", "d2": "doc two"},
            {"d1": [1.0, 0.0], "d2": [0.99, 0.01]},
            {},
        )
        for c in clusters:
            cluster_store.save_cluster(c)

        result = enrich_corpus_level1(fake_corpus, cluster_store)
        assert result["topics_new"] == 0
        # Curation still runs on uncurated docs
        assert result["curated"] >= 0
    finally:
        cluster_store.close()


def test_enrich_level1_checkpoint_skips_processed(fake_corpus, monkeypatch, tmp_path):
    """Level 1 skips docs already marked as clustered in the checkpoint."""
    fake_store = FakeDocStore(
        centroids={"d1": ["d1_c1"], "d2": ["d2_c1"], "d3": ["d3_c1"], "d4": ["d4_c1"]},
        texts={
            "d1": "NVIDIA GPU blog post",
            "d2": "CUDA programming guide",
            "d3": "Biology research cells",
            "d4": "Medical protein study",
        },
        chunk_texts={
            "d1_c1": "NVIDIA GPU", "d2_c1": "CUDA programming",
            "d3_c1": "Biology cells", "d4_c1": "Medical proteins",
        },
    )
    fake_lance = FakeLanceIndex(
        doc_embeddings={
            "d1": [1.0, 0.0, 0.0], "d2": [0.99, 0.01, 0.0],
            "d3": [0.0, 1.0, 0.0], "d4": [0.0, 0.98, 0.02],
        },
        chunk_vectors={
            "d1_c1": [1.0, 0.0, 0.0], "d2_c1": [0.99, 0.01, 0.0],
            "d3_c1": [0.0, 1.0, 0.0], "d4_c1": [0.0, 0.98, 0.02],
        },
    )

    import ipa
    monkeypatch.setattr(ipa, "DocumentStore", lambda path: fake_store)
    monkeypatch.setattr(
        "ipa.indexes.lancedb_index.LanceDBIndex",
        lambda path: fake_lance,
    )

    cluster_store = TopicClusterStore(tmp_path / "test_clusters.db")
    try:
        # Pre-mark d1 and d2 as already clustered
        cluster_store.mark_processed(["d1", "d2"], stage="clustered")

        # Run enrichment — should only process d3 and d4
        result = enrich_corpus_level1(fake_corpus, cluster_store)
        # d3 and d4 are similar → should form 1 topic
        assert result["topics_new"] >= 1
        assert result["unclustered_before"] == 2  # only d3 and d4

        # Verify d1 and d2 are NOT in any new cluster
        clusters = cluster_store.list_clusters()
        all_members = set()
        for c in clusters:
            all_members.update(c.member_document_ids)
        assert "d1" not in all_members
        assert "d2" not in all_members
        assert "d3" in all_members
        assert "d4" in all_members
    finally:
        cluster_store.close()


def test_enrich_level1_resumable_after_full_run(fake_corpus, monkeypatch, tmp_path):
    """Running Level 1 twice doesn't reprocess already-clustered docs."""
    fake_store = FakeDocStore(
        centroids={"d1": ["d1_c1"], "d2": ["d2_c1"]},
        texts={"d1": "NVIDIA GPU blog", "d2": "CUDA programming guide"},
        chunk_texts={"d1_c1": "NVIDIA GPU", "d2_c1": "CUDA programming"},
    )
    fake_lance = FakeLanceIndex(
        doc_embeddings={"d1": [1.0, 0.0], "d2": [0.99, 0.01]},
        chunk_vectors={"d1_c1": [1.0, 0.0], "d2_c1": [0.99, 0.01]},
    )

    import ipa
    monkeypatch.setattr(ipa, "DocumentStore", lambda path: fake_store)
    monkeypatch.setattr(
        "ipa.indexes.lancedb_index.LanceDBIndex",
        lambda path: fake_lance,
    )

    cluster_store = TopicClusterStore(tmp_path / "test_clusters.db")
    try:
        # First run
        result1 = enrich_corpus_level1(fake_corpus, cluster_store)
        assert result1["topics_new"] >= 1

        # Second run — should find nothing new to cluster
        result2 = enrich_corpus_level1(fake_corpus, cluster_store)
        assert result2["topics_new"] == 0
        assert result2["unclustered_before"] == 0
        # Curation should also skip already-curated docs
        assert result2["curated"] == 0
    finally:
        cluster_store.close()


def test_enrich_level1_persists_curation(fake_corpus, monkeypatch, tmp_path):
    """Level 1 persists curation decisions to the store."""
    fake_store = FakeDocStore(
        centroids={"d1": ["d1_c1"], "d2": ["d2_c1"]},
        texts={"d1": "NVIDIA GPU blog post about architecture",
                   "d2": "CUDA programming guide for developers"},
        chunk_texts={"d1_c1": "NVIDIA GPU", "d2_c1": "CUDA programming"},
    )
    fake_lance = FakeLanceIndex(
        doc_embeddings={"d1": [1.0, 0.0], "d2": [0.99, 0.01]},
        chunk_vectors={"d1_c1": [1.0, 0.0], "d2_c1": [0.99, 0.01]},
    )

    import ipa
    monkeypatch.setattr(ipa, "DocumentStore", lambda path: fake_store)
    monkeypatch.setattr(
        "ipa.indexes.lancedb_index.LanceDBIndex",
        lambda path: fake_lance,
    )

    cluster_store = TopicClusterStore(tmp_path / "test_clusters.db")
    try:
        enrich_corpus_level1(fake_corpus, cluster_store)

        # Verify curation decisions were persisted
        decision_d1 = cluster_store.get_curation_decision("d1")
        decision_d2 = cluster_store.get_curation_decision("d2")
        assert decision_d1 is not None
        assert decision_d2 is not None
        assert "scores" in decision_d1
        assert "relevance" in decision_d1["scores"]
    finally:
        cluster_store.close()


def test_save_clusters_batch_atomic(tmp_path):
    """save_clusters_batch saves all or nothing."""
    from ipa.agentic.topic_clusters import build_clusters
    store = TopicClusterStore(tmp_path / "test.db")
    try:
        docs = {"d1": "tech", "d2": "tech", "d3": "bio", "d4": "bio"}
        embeddings = {"d1": [1, 0], "d2": [0.99, 0.01], "d3": [0, 1], "d4": [0.01, 0.99]}
        clusters = build_clusters(docs, embeddings, {})
        assert len(clusters) >= 1

        store.save_clusters_batch(clusters)
        loaded = store.list_clusters()
        assert len(loaded) == len(clusters)
    finally:
        store.close()


def test_checkpoint_mark_and_check(tmp_path):
    """Checkpoint correctly tracks processed documents."""
    store = TopicClusterStore(tmp_path / "test.db")
    try:
        assert not store.is_processed("d1")
        store.mark_processed(["d1", "d2"], stage="clustered")
        assert store.is_processed("d1", "clustered")
        assert store.is_processed("d2", "clustered")
        assert not store.is_processed("d3")

        # Curated stage is "beyond" clustered
        store.update_stage(["d1"], "curated")
        assert store.is_processed("d1", "curated")
        assert store.is_processed("d1", "clustered")  # curated implies clustered
        assert not store.is_processed("d2", "curated")  # d2 is only clustered

        # processed_doc_ids filters by stage
        clustered = store.processed_doc_ids(stage="clustered")
        curated = store.processed_doc_ids(stage="curated")
        assert "d1" in clustered
        assert "d2" in clustered
        assert "d1" in curated
        assert "d2" not in curated
    finally:
        store.close()


def test_curation_decisions_persist_and_retrieve(tmp_path):
    """Curation decisions are saved and retrieved correctly."""
    store = TopicClusterStore(tmp_path / "test.db")
    try:
        decisions = [
            {
                "decision_id": "dec:1",
                "document_id": "d1",
                "report_id": "test",
                "scores": {"relevance": 0.8, "novelty": 0.6},
                "decision": "promote",
                "reason": "test reason",
            },
            {
                "decision_id": "dec:2",
                "document_id": "d2",
                "report_id": "test",
                "scores": {"relevance": 0.3, "novelty": 0.9},
                "decision": "reporter_only",
                "reason": "low relevance",
            },
        ]
        store.save_curation_decisions_batch(decisions)

        d1 = store.get_curation_decision("d1")
        assert d1 is not None
        assert d1["decision"] == "promote"
        assert d1["scores"]["relevance"] == 0.8

        all_decisions = store.list_curation_decisions()
        assert len(all_decisions) == 2
    finally:
        store.close()


# --- Provenance tests ---

def test_build_document_dicts_reads_provenance():
    """build_document_dicts reads source_domain and quality_score from document_sources."""
    store = FakeDocStore(
        centroids={"d1": ["d1_c1"]},
        texts={"d1": "NVIDIA GPU blog post"},
        chunk_texts={"d1_c1": "NVIDIA GPU"},
        sources={
            "d1": {
                "source_url": "https://developer.nvidia.com/blog/post",
                "source_domain": "developer.nvidia.com",
                "provenance": "configured_scrape",
                "quality_score": 0.85,
            }
        },
    )
    lance = FakeLanceIndex(doc_embeddings={"d1": [1.0, 0.0]})
    documents, _ = build_document_dicts(store, lance)
    assert len(documents) == 1
    assert documents[0]["source_domain"] == "developer.nvidia.com"
    assert documents[0]["quality_score"] == 0.85
    assert documents[0]["canonical_url"] == "https://developer.nvidia.com/blog/post"


def test_build_document_dicts_no_provenance_falls_back():
    """Without document_sources, fields fall back to empty defaults."""
    store = FakeDocStore(
        centroids={"d1": ["d1_c1"]},
        texts={"d1": "Some text"},
    )
    lance = FakeLanceIndex(doc_embeddings={"d1": [1.0]})
    documents, _ = build_document_dicts(store, lance)
    assert documents[0]["source_domain"] == ""
    assert documents[0]["quality_score"] == 0.0


# --- Promotion policy tests ---

def test_promotion_policy_configured_scrape_auto_promote():
    """configured_scrape provenance → auto-promote without score threshold."""
    from ipa.agentic.promotion_policy import evaluate_promotion
    result = evaluate_promotion("d1", "configured_scrape")
    assert result.should_promote is True
    assert "auto-promote" in result.reason


def test_promotion_policy_agent_research_above_threshold():
    """agent_research with score >= 0.70 → promote."""
    from ipa.agentic.promotion_policy import evaluate_promotion
    from ipa.reporter.reporter_contracts import ScoreBundle
    scores = ScoreBundle(
        relevance=0.90, novelty=0.80, source_quality=0.70,
        impact=0.60, depth=0.50, actionability=0.50,
    )
    result = evaluate_promotion("d1", "agent_research", scores=scores)
    assert result.should_promote is True
    assert result.score is not None
    assert result.score >= 0.70


def test_promotion_policy_agent_research_below_threshold():
    """agent_research with score < 0.70 → do not promote."""
    from ipa.agentic.promotion_policy import evaluate_promotion
    from ipa.reporter.reporter_contracts import ScoreBundle
    scores = ScoreBundle(
        relevance=0.50, novelty=0.50, source_quality=0.50,
        impact=0.50, depth=0.50, actionability=0.50,
    )
    result = evaluate_promotion("d1", "agent_research", scores=scores)
    assert result.should_promote is False
    assert result.score is not None
    assert result.score < 0.70


def test_promotion_policy_agent_research_no_scores():
    """agent_research without scores → do not promote."""
    from ipa.agentic.promotion_policy import evaluate_promotion
    result = evaluate_promotion("d1", "agent_research", scores=None)
    assert result.should_promote is False
    assert "no scores" in result.reason


def test_promotion_policy_unknown_provenance():
    """Unknown provenance → do not promote."""
    from ipa.agentic.promotion_policy import evaluate_promotion
    result = evaluate_promotion("d1", "unknown")
    assert result.should_promote is False


# --- Promotion queue tests ---

def test_promotion_queue_mark_and_list(tmp_path):
    """mark_promotion_pending + pending_promotions work correctly."""
    store = TopicClusterStore(tmp_path / "test.db")
    try:
        store.mark_promotion_pending("d1", "configured source: auto-promote", "configured_scrape")
        store.mark_promotion_pending("d2", "agent research: score 0.85 >= 0.70", "agent_research")

        pending = store.pending_promotions()
        assert len(pending) == 2
        assert pending[0]["document_id"] == "d1"
        assert pending[0]["provenance"] == "configured_scrape"
        assert pending[1]["document_id"] == "d2"
        assert pending[1]["provenance"] == "agent_research"
    finally:
        store.close()


def test_promotion_queue_mark_done(tmp_path):
    """mark_promotion_done removes from pending."""
    store = TopicClusterStore(tmp_path / "test.db")
    try:
        store.mark_promotion_pending("d1", "test", "configured_scrape")
        assert store.is_promotion_pending("d1") is True

        store.mark_promotion_done("d1")
        assert store.is_promotion_pending("d1") is False

        pending = store.pending_promotions()
        assert len(pending) == 0
    finally:
        store.close()


def test_promotion_queue_idempotent(tmp_path):
    """Re-queuing the same document updates the reason, doesn't duplicate."""
    store = TopicClusterStore(tmp_path / "test.db")
    try:
        store.mark_promotion_pending("d1", "reason 1", "configured_scrape")
        store.mark_promotion_pending("d1", "reason 2", "configured_scrape")

        pending = store.pending_promotions()
        assert len(pending) == 1
        assert pending[0]["reason"] == "reason 2"
    finally:
        store.close()


# --- DocumentStore provenance tests ---

def test_document_store_put_and_get_source(tmp_path):
    """DocumentStore.put_source + get_source work correctly."""
    from ipa import DocumentStore
    from ipa.contracts import CanonicalDocument
    store = DocumentStore(tmp_path / "test.db")
    try:
        # Create a document first (FK constraint)
        doc = CanonicalDocument(
            document_id="d1", parser_id="test", mime_type="text/plain",
            pages=1, text="test text", elements=[], source_spans=[],
        )
        store.put_document(doc, "artifact:1")

        store.put_source("d1", "https://example.com/page", "example.com", "configured_scrape", 0.75)
        src = store.get_source("d1")
        assert src is not None
        assert src["source_url"] == "https://example.com/page"
        assert src["source_domain"] == "example.com"
        assert src["provenance"] == "configured_scrape"
        assert src["quality_score"] == 0.75
    finally:
        store.close()


def test_document_store_all_sources(tmp_path):
    """DocumentStore.all_sources returns all provenance records."""
    from ipa import DocumentStore
    from ipa.contracts import CanonicalDocument
    store = DocumentStore(tmp_path / "test.db")
    try:
        for i in range(3):
            doc = CanonicalDocument(
                document_id=f"d{i}", parser_id="test", mime_type="text/plain",
                pages=1, text=f"text {i}", elements=[], source_spans=[],
            )
            store.put_document(doc, f"artifact:{i}")
            store.put_source(f"d{i}", f"https://site{i}.com", f"site{i}.com", "configured_scrape", 0.5 * i)

        all_src = store.all_sources()
        assert len(all_src) == 3
        assert "d0" in all_src
        assert "d2" in all_src
    finally:
        store.close()


def test_document_store_sources_by_provenance(tmp_path):
    """DocumentStore.sources_by_provenance filters by provenance type."""
    from ipa import DocumentStore
    from ipa.contracts import CanonicalDocument
    store = DocumentStore(tmp_path / "test.db")
    try:
        for i in range(4):
            doc = CanonicalDocument(
                document_id=f"d{i}", parser_id="test", mime_type="text/plain",
                pages=1, text=f"text {i}", elements=[], source_spans=[],
            )
            store.put_document(doc, f"artifact:{i}")
            prov = "configured_scrape" if i < 2 else "agent_research"
            store.put_source(f"d{i}", f"https://site{i}.com", f"site{i}.com", prov, 0.5)

        configured = store.sources_by_provenance("configured_scrape")
        agent = store.sources_by_provenance("agent_research")
        assert len(configured) == 2
        assert len(agent) == 2
        assert "d0" in configured
        assert "d2" in agent
    finally:
        store.close()
