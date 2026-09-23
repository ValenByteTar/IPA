from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "validation"))
from validate_reporter_contract import SCHEMAS, _registry, validate

from ipa.reporter.reporter_claims import validate_claims
from ipa.reporter.reporter_config import ReporterConfig
from ipa.reporter.reporter_contracts import ReporterDecision, ScoreBundle
from ipa.reporter.reporter_ai import ReporterLLM
from ipa.reporter.reporter_curation import curate_documents
from ipa.reporter.reporter_metadata import normalize_article
from ipa.reporter.reporter_pipeline import ReporterPipeline
from ipa.reporter.reporter_promotion import approve_promotion, pending_promotions, queue_promotion
from ipa.reporter.reporter_research import can_execute, create_research_request
from ipa.reporter.reporter_representation import build_representation
from ipa.reporter.reporter_store import ReporterStore
from ipa.reporter.reporter_topics import discover_topics, match_topic_continuity

HASH = "sha256:" + "a" * 64
PERIOD_START = "2026-08-01T00:00:00Z"
PERIOD_END = "2026-09-01T00:00:00Z"


def _doc(doc_id: str, text: str, domain: str = "example.com") -> dict:
    return {
        "document_id": doc_id,
        "artifact_id": HASH,
        "title": text[:40],
        "text": text,
        "source_domain": domain,
        "canonical_url": f"https://{domain}/{doc_id}",
        "source_url": f"https://{domain}/{doc_id}",
        "content_hash": HASH,
        "published_at": "2026-08-15T00:00:00Z",
        "quality_score": 0.9,
    }


def _config(tmp_path: Path) -> ReporterConfig:
    path = tmp_path / "reporter.yaml"
    path.write_text(
        """reporter:
  corpus_id: test
  period:
    start: '2026-08-01T00:00:00Z'
    end: '2026-09-01T00:00:00Z'
    label: '2026-08'
  category_generation:
    min_documents: 2
    allow_singleton_topics: true
    similarity_threshold: 0.25
  ranking:
    novelty: 0.25
    source_quality: 0.20
    source_diversity: 0.20
    potential_impact: 0.20
    user_interest: 0.15
""",
        encoding="utf-8",
    )
    return ReporterConfig.from_yaml(path)


@pytest.mark.parametrize("record_type", SCHEMAS)
def test_reporter_schemas_are_valid(record_type):
    schema_path = Path(__file__).parents[1] / "contracts" / SCHEMAS[record_type]
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)


def test_representation_extracts_title_and_abstract():
    representation = build_representation(
        "A Better Paper Title\nAuthors\n\nAbstract: This is the abstract.\n\nIntroduction\nNoise",
        "fallback",
    )
    assert representation.title == "A Better Paper Title"
    assert representation.abstract == "This is the abstract."
    assert representation.title_confidence == "medium"
    assert "A Better Paper Title" in representation.embedding_text


def test_metadata_normalizes_headers_and_hash(tmp_path):
    path = tmp_path / "article.txt"
    path.write_text("# New topic\n\nSource: https://example.com/a\nDate: 2026-08-10\n\nUseful content.", encoding="utf-8")
    metadata = normalize_article(path)
    assert metadata.title == "New topic"
    assert metadata.canonical_url == "https://example.com/a"
    assert metadata.published_at == "2026-08-10T00:00:00Z"
    assert metadata.content_hash.startswith("sha256:")


def test_llm_label_requires_structured_nonempty_output():
    class Result:
        ok = True
        text = '{"label":"Tema válido","description":"Descripción con evidencia."}'

    class Provider:
        def generate_chat(self, *_args, **_kwargs):
            return Result()

    assert ReporterLLM(Provider()).label([_doc("doc:one", "photonic processors")])["label"] == "Tema válido"


def test_llm_label_rejects_invalid_output():
    class Result:
        ok = True
        text = "No pude generar JSON"

    class Provider:
        def generate_chat(self, *_args, **_kwargs):
            return Result()

    assert ReporterLLM(Provider()).label([_doc("doc:one", "photonic processors")]) == {}


def test_curation_preserves_negative_decisions_without_deleting():
    docs = [_doc("doc:one", "same repeated content"), _doc("doc:two", "same repeated content")]
    decisions = curate_documents(docs, "report:2026-08:x", PERIOD_START, PERIOD_END)
    assert any(decision.decision == ReporterDecision.DUPLICATE for decision in decisions)
    assert len(docs) == 2


def test_curation_survives_non_numeric_quality_score():
    """Regression: a string quality_score ('significant' from an upstream
    classifier) crashed the whole batch with ValueError — now it degrades
    to the neutral fallback instead of aborting curation."""
    doc = _doc("doc:one", "relevant novel content about photonic processors")
    doc["quality_score"] = "significant"
    decisions = curate_documents([doc], "report:2026-08:x", PERIOD_START, PERIOD_END)
    assert len(decisions) == 1
    assert decisions[0].document_id == "doc:one"
    # String scores also survive inside the LLM judge payload.
    doc2 = _doc("doc:two", "different relevant content about optical computing")
    decisions = curate_documents(
        [doc2], "report:2026-08:x", PERIOD_START, PERIOD_END,
        classifier=lambda d: {"relevance": "high", "novelty": 0.9})
    assert len(decisions) == 1
    assert 0.0 <= decisions[0].scores.relevance <= 1.0
    assert decisions[0].scores.novelty == 0.9


def _emb(base: float, dims: int = 8) -> list[float]:
    return [base + i * 0.001 for i in range(dims)]


def test_novelty_gate_requires_lexical_confirmation_for_embedding_match():
    """Cosine >0.95 alone must not discard: site boilerplate makes distinct
    articles (weekly CVE alerts, series templates) look identical at the
    document-embedding level. Without high lexical overlap vs the matched
    document the doc stays REPORTER_ONLY (PM-004 false positives)."""
    template = "weekly advisory published by the same agency " + "shared boilerplate navigation footer disclaimer " * 8
    doc = _doc("doc:new", template + "CVE-2026-1111 affects Apache servers with remote code execution")
    hist_text = template + "CVE-2025-9999 affects OpenSSL certificate validation routines"
    decisions = curate_documents(
        [doc], "report:x", PERIOD_START, PERIOD_END,
        document_embeddings={"doc:new": _emb(1.0)},
        historical_embeddings=[_emb(1.0)],
        historical_documents=[{"document_id": "doc:old", "text": hist_text}],
    )
    assert decisions[0].decision == ReporterDecision.REPORTER_ONLY
    assert decisions[0].duplicate_of is None


def test_novelty_gate_confirms_duplicate_with_lexical_overlap():
    """Cosine >0.95 AND token overlap >=0.85 vs the matched document →
    true near-duplicate → DUPLICATE, with duplicate_of pointing at it."""
    text = "identical article body about photonic processors and memory bandwidth limits " * 3
    doc = _doc("doc:new", text)
    decisions = curate_documents(
        [doc], "report:x", PERIOD_START, PERIOD_END,
        document_embeddings={"doc:new": _emb(1.0)},
        historical_embeddings=[_emb(1.0)],
        historical_documents=[{"document_id": "doc:old", "text": text}],
    )
    assert decisions[0].decision == ReporterDecision.DUPLICATE
    assert decisions[0].duplicate_of == "doc:old"


def test_novelty_gate_keeps_document_when_match_unverifiable():
    """Cosine >0.95 with no aligned historical text to confirm → keep.
    An unverifiable fuzzy match must never cause deletion."""
    doc = _doc("doc:new", "distinct content about quantum networking and error correction")
    decisions = curate_documents(
        [doc], "report:x", PERIOD_START, PERIOD_END,
        document_embeddings={"doc:new": _emb(1.0)},
        historical_embeddings=[_emb(1.0)],
    )
    assert decisions[0].decision == ReporterDecision.REPORTER_ONLY


def test_novelty_gate_lexical_fallback_still_rejects_near_copies():
    """Without embeddings, novelty uses Jaccard directly; a >0.95 token
    overlap IS lexical confirmation → DUPLICATE still applies."""
    text = "identical body about memory hierarchy and cache coherence protocols " * 2
    doc = _doc("doc:new", text)
    decisions = curate_documents(
        [doc], "report:x", PERIOD_START, PERIOD_END,
        historical_documents=[{"document_id": "doc:old", "text": text}],
    )
    assert decisions[0].decision == ReporterDecision.DUPLICATE


def test_topic_discovery_is_not_domain_taxonomy():
    docs = [
        _doc("doc:a", "quantum photonic processors improve optical computation"),
        _doc("doc:b", "photonic quantum processors improve optical algorithms"),
        _doc("doc:c", "urban water recycling reduces municipal drought risk"),
    ]
    topics = discover_topics(docs, similarity_threshold=0.5, min_documents=2, allow_singletons=True)
    assert topics
    assert any("photonic" in (topic["label"] + " " + topic["description"]).lower() for topic in topics)
    assert all(topic["category_id"].startswith("topic:") for topic in topics)


def test_topic_labeler_output_is_normalized_before_schema_validation():
    topics = discover_topics(
        [_doc("doc:label", "photonic quantum processors")],
        similarity_threshold=0.5,
        min_documents=1,
        labeler=lambda group: {"label": "Photonics", "description": "A topic", "uncertainties": {"one": "uncertain"}},
    )
    assert topics[0]["uncertainties"] == ["uncertain"]


def test_topic_continuity_links_matching_periods():
    current = discover_topics([_doc("doc:new", "photonic quantum processors")], similarity_threshold=0.1, min_documents=1)
    previous = [{**current[0], "category_id": "topic:old", "document_ids": ["doc:old"]}]
    links = match_topic_continuity(current, previous)
    assert links[0].previous_category_id == "topic:old"


def test_reporter_pipeline_isolated_and_schema_valid(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "one.txt").write_text("Quantum photonic processors improve optical computation.", encoding="utf-8")
    (input_dir / "two.txt").write_text("Photonic quantum processors improve optical algorithms.", encoding="utf-8")
    output = tmp_path / "reporter"
    with ReporterPipeline(_config(tmp_path), output) as pipeline:
        report = pipeline.run(input_dir)
    assert (output / "corpus" / "document_store.db").exists()
    assert (output / "report.json").exists()
    assert (output / "reporter.db").exists()
    assert not (output / "main" / "document_store.db").exists()
    assert validate("ReporterReport", report) == []


def test_promotion_requires_explicit_approval(tmp_path):
    with ReporterStore(tmp_path / "reporter.db") as store:
        promotion_id = queue_promotion(store, "doc:one", "decision:one")
        assert pending_promotions(store)[0]["promotion_id"] == promotion_id
        approve_promotion(store, promotion_id, "user:test", "reviewed")
        assert pending_promotions(store) == []


def test_score_bundle_rejects_invalid_score():
    with pytest.raises(ValueError):
        ScoreBundle(1.1, 0.5, 0.5, 0.5)


def test_research_request_is_bounded_and_not_executable_before_approval():
    request = create_research_request(
        "goal:test", "concept:test", "Find the primary source for this topic.",
        [{"source_id": "doc:test", "source_type": "document"}],
        ["example.com"], {"max_urls": 5, "max_seconds": 60, "max_bytes": 100000, "max_depth": 1},
    )
    assert request["status"] == "pending_approval"
    assert not can_execute(request)


def test_claim_citation_validation_detects_supported_and_unsupported_claims():
    evidence = ["PagedAttention allocates KV cache in dynamic blocks and reduces fragmentation."]
    claims = validate_claims(
        "PagedAttention allocates KV cache in dynamic blocks [1]. It doubles every workload [1].",
        evidence,
    )
    assert claims[0]["support_level"] == "supported"
    assert claims[1]["support_level"] in {"partial", "unsupported"}


def test_claim_validation_rejects_unreferenced_claim():
    claims = validate_claims("This claim has no citation.", ["Unrelated source text."])
    assert claims[0]["support_level"] == "unsupported"


def test_claim_validation_checks_numbers():
    claims = validate_claims("The system reaches 4x throughput [1].", ["The system reaches 2x throughput."])
    assert claims[0]["number_support"] is False
    assert claims[0]["support_level"] != "supported"
