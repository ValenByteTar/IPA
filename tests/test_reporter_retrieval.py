from __future__ import annotations

from dataclasses import dataclass

from ipa.agentic.agentic_contracts import EvidenceHit, EvidenceSet, QueryIR
from ipa.contracts import DocumentChunk, SearchHit, SourceSpan
from ipa.agentic.reporter_context import ReporterContextBuilder, build_context
from ipa.agentic.reporter_retrieval import ReporterRetriever, retrieve_evidence


@dataclass
class FakeIndex:
    hits: list[SearchHit]

    def __post_init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, limit: int = 10) -> list[SearchHit]:
        self.calls.append((query, limit))
        return self.hits[:limit]


class FakeStore:
    def __init__(self, chunks: list[DocumentChunk]) -> None:
        self.chunks = {chunk.chunk_id: chunk for chunk in chunks}

    def get_chunk(self, chunk_id: str) -> DocumentChunk | None:
        return self.chunks.get(chunk_id)

    def get_chunks(self, document_id: str):
        return (chunk for chunk in self.chunks.values() if chunk.document_id == document_id)


def _chunk(chunk_id: str, document_id: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        content_hash="sha256:" + "a" * 64,
        text=text,
        metadata={"source_uri": f"file:///{document_id}.txt"},
        source_span=SourceSpan(
            artifact_id=f"artifact:{document_id}", page=1, offset_start=0, offset_end=len(text)
        ),
    )


def _search_hit(chunk_id: str, score: float) -> SearchHit:
    return SearchHit(chunk_id, score, None, "fake-tantivy")


def test_retrieval_is_bounded_and_filters_document_scope():
    chunks = [
        _chunk("chunk:outside", "doc:outside", "benefits and limitations"),
        _chunk("chunk:one", "doc:one", "The approach documents benefits."),
        _chunk("chunk:two", "doc:two", "A separate implementation detail."),
    ]
    index = FakeIndex(
        [_search_hit("chunk:outside", 99.0), _search_hit("chunk:one", 4.0)]
    )
    query = QueryIR(
        "Compare approach benefits and limitations",
        constraints={"document_ids": ["doc:one", "doc:two"], "top_k": 3},
        required_evidence=["benefits", "limitations"],
    )

    evidence = ReporterRetriever(index, FakeStore(chunks), max_candidates=4, max_chunks=3)(query)

    assert index.calls == [(query.raw_query, 4)]
    assert {hit.document_id for hit in evidence.hits} <= {"doc:one", "doc:two"}
    assert "doc:outside" not in {hit.document_id for hit in evidence.hits}
    assert all(hit.retrieval_stage == "scoped" for hit in evidence.hits)
    assert evidence.covered_requirements == ["benefits"]
    assert evidence.missing_requirements == ["limitations"]
    assert evidence.sufficiency == "partial"
    assert evidence.document_diversity == 2


def test_scoped_retrieval_can_select_store_chunk_missed_by_global_search():
    selected = _chunk("chunk:selected", "doc:selected", "the required recovery evidence")
    outside = _chunk("chunk:outside", "doc:outside", "recovery evidence")
    evidence = retrieve_evidence(
        QueryIR(
            "recovery",
            constraints={"document_ids": ["doc:selected"], "top_k": 1},
            required_evidence=["recovery"],
        ),
        FakeIndex([_search_hit("chunk:outside", 10.0)]),
        FakeStore([outside, selected]),
        max_candidates=2,
        max_chunks=1,
    )

    assert [hit.chunk_id for hit in evidence.hits] == ["chunk:selected"]
    assert evidence.sufficiency == "sufficient"


def test_global_retrieval_resolves_traceability_from_store():
    chunk = _chunk("chunk:one", "doc:one", "Relevant global evidence")
    evidence = retrieve_evidence(
        QueryIR("global evidence", constraints={"top_k": 1}),
        FakeIndex([_search_hit(chunk.chunk_id, 2.5)]),
        FakeStore([chunk]),
        max_candidates=5,
        max_chunks=2,
    )

    hit = evidence.hits[0]
    assert hit.document_id == "doc:one"
    assert hit.retrieval_stage == "initial"
    assert hit.retrieval_backend == "fake-tantivy"
    assert hit.source_ref["artifact_id"] == "artifact:doc:one"
    assert hit.source_span == {
        "artifact_id": "artifact:doc:one",
        "page": 1,
        "offset_start": 0,
        "offset_end": len(chunk.text),
    }
    assert hit.text_hash.startswith("sha256:")


def _evidence() -> EvidenceSet:
    query = QueryIR("bounded context")
    return EvidenceSet(
        query_ir=query,
        hits=[
            EvidenceHit("chunk:one", "doc:a", score=2.0, text_hash="sha256:" + "1" * 64),
            EvidenceHit("chunk:two", "doc:a", score=1.5, text_hash="sha256:" + "2" * 64),
            EvidenceHit("chunk:three", "doc:b", score=1.0, text_hash="sha256:" + "3" * 64),
        ],
        document_diversity=2,
        sufficiency="sufficient",
    )


def test_context_builder_obeys_chunk_and_token_budgets():
    evidence = _evidence()
    texts = {"chunk:one": "one two", "chunk:two": "three four", "chunk:three": "five"}
    builder = ReporterContextBuilder(
        texts,
        max_chunks=2,
        max_context_tokens=3,
        token_estimator=lambda text: len(text.split()),
    )

    package = builder(evidence)

    assert package.evidence is evidence
    assert package.token_count == 3
    assert list(package.citation_map) == ["[Doc 1, fragment 1]", "[Doc 2, fragment 1]"]
    assert package.citation_map["[Doc 1, fragment 1]"]["text"] == "one two"
    assert package.citation_map["[Doc 2, fragment 1]"]["chunk_id"] == "chunk:three"
    assert package.truncation_policy == "preserve-ranked-whole-chunks"
    assert package.input_hash.startswith("sha256:")


def test_context_builder_accepts_callable_and_hash_is_deterministic():
    evidence = _evidence()
    texts = {"chunk:one": "alpha", "chunk:two": "beta", "chunk:three": "gamma"}
    resolver = texts.get

    first = build_context(
        evidence, resolver, max_chunks=3, max_context_tokens=10, token_estimator=lambda _text: 1
    )
    second = build_context(
        evidence, resolver, max_chunks=3, max_context_tokens=10, token_estimator=lambda _text: 1
    )

    assert first.token_count == 3
    assert first.truncation_policy == "none"
    assert first.input_hash == second.input_hash
    assert list(first.citation_map) == [
        "[Doc 1, fragment 1]",
        "[Doc 1, fragment 2]",
        "[Doc 2, fragment 1]",
    ]
