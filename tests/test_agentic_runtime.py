from __future__ import annotations

import json
from dataclasses import replace

import pytest

from ipa.agentic.agentic_contracts import (
    ContextPackage,
    EvidenceHit,
    EvidenceSet,
    ExecutionBudget,
    QueryIR,
    TopicInvestigationState,
    TraceEvent,
)

HASH = "sha256:" + "a" * 64
NOW = "2026-09-01T12:00:00Z"
LATER = "2026-09-01T12:01:00Z"


def _query() -> QueryIR:
    return QueryIR(
        raw_query="Compare the documented approaches",
        intent="comparison",
        entities=["approach A", "approach B"],
        constraints={"document_ids": ["doc:1", "doc:2"]},
        topic_id="topic:1",
        report_id="report:1",
        required_evidence=["benefits", "limitations"],
        is_comparison=True,
        language="en",
    )


def _hit(chunk_id: str = "chunk:1", document_id: str = "doc:1") -> EvidenceHit:
    return EvidenceHit(
        chunk_id=chunk_id,
        document_id=document_id,
        score=0.91,
        retrieval_stage="scoped",
        retrieval_backend="test-index",
        source_ref={"artifact_id": "artifact:1", "uri": "file:///source.txt"},
        source_span={"offset_start": 10, "offset_end": 80},
        text_hash=HASH,
    )


def _evidence() -> EvidenceSet:
    return EvidenceSet(
        query_ir=_query(),
        hits=[_hit()],
        covered_requirements=["benefits"],
        missing_requirements=["limitations"],
        document_diversity=1,
        sufficiency="partial",
    )


def _context() -> ContextPackage:
    evidence = _evidence()
    return ContextPackage(
        query_ir=evidence.query_ir,
        evidence=evidence,
        citation_map={"Doc 1, fragment 1": {"document_id": "doc:1", "chunk_id": "chunk:1"}},
        token_count=120,
        truncation_policy="preserve-ranked-whole-chunks",
        input_hash=HASH,
    )


@pytest.mark.parametrize(
    ("record", "record_type"),
    [
        (_query(), QueryIR),
        (_hit(), EvidenceHit),
        (_evidence(), EvidenceSet),
        (_context(), ContextPackage),
        (ExecutionBudget(), ExecutionBudget),
        (TraceEvent(kind="planned", data={"intent": "comparison"}, timestamp=NOW), TraceEvent),
    ],
)
def test_contract_round_trip_is_json_serializable(record, record_type):
    payload = json.loads(json.dumps(record.to_dict()))
    assert record_type.from_dict(payload) == record


def test_first_increment_budget_defaults_are_bounded():
    budget = ExecutionBudget()
    assert budget.max_retrieval_rounds == 2
    assert budget.max_llm_calls == 1
    assert budget.max_repairs == 0
    assert budget.max_documents == 6
    assert budget.max_chunks == 12


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"max_iterations": 0}, "max_iterations"),
        ({"max_retrieval_rounds": 0}, "max_retrieval_rounds"),
        ({"max_llm_calls": 0}, "max_llm_calls"),
        ({"max_repairs": -1}, "max_repairs"),
        ({"max_documents": 0}, "max_documents"),
        ({"max_chunks": 0}, "max_chunks"),
        ({"max_context_tokens": 0}, "max_context_tokens"),
        ({"max_seconds": 0}, "max_seconds"),
    ],
)
def test_execution_budget_rejects_unbounded_values(changes, message):
    with pytest.raises(ValueError, match=message):
        ExecutionBudget(**changes)


def test_mutable_defaults_are_not_shared():
    first = QueryIR("first")
    second = QueryIR("second")
    first.entities.append("entity")
    assert second.entities == []

    first_state = TopicInvestigationState("first")
    second_state = TopicInvestigationState("second")
    first_state.counters["retrieval_rounds"] = 1
    first_state.phase_flags["planned"] = True
    assert second_state.counters["retrieval_rounds"] == 0
    assert second_state.phase_flags == {}


def test_nested_topic_state_round_trip_and_terminal_status():
    context = _context()
    state = TopicInvestigationState(
        question=context.query_ir.raw_query,
        run_id="run:1",
        report_id="report:1",
        category_id="topic:1",
        query_ir=context.query_ir,
        results=[{"document_id": "doc:1"}],
        evidence_set=context.evidence,
        context_package=context,
        answer="The approaches differ.",
        claims=[{"text": "The approaches differ.", "status": "supported"}],
        signals=[{"name": "evidence_sufficiency", "value": "partial"}],
        last_decision={"action": "finalize"},
        counters={
            "iterations": 1,
            "retrieval_rounds": 2,
            "llm_calls": 1,
            "repairs": 0,
            "documents": 1,
            "chunks": 1,
            "context_tokens": 120,
        },
        phase_flags={"planned": True, "retrieved": True},
        traces=[TraceEvent("planned", timestamp=NOW)],
        started_at=NOW,
        completed_at=LATER,
        status="completed",
    )
    restored = TopicInvestigationState.from_dict(json.loads(json.dumps(state.to_dict())))
    assert restored == state
    assert restored.terminal
    assert isinstance(restored.query_ir, QueryIR)
    assert isinstance(restored.evidence_set, EvidenceSet)
    assert isinstance(restored.context_package, ContextPackage)
    assert isinstance(restored.traces[0], TraceEvent)


def test_non_terminal_state_has_safe_lifecycle_defaults():
    state = TopicInvestigationState("What does the evidence show?")
    assert state.status == "pending"
    assert state.completed_at is None
    assert not state.terminal
    assert state.run_id
    assert state.counters["llm_calls"] == 0


def test_terminal_state_requires_completion_timestamp():
    with pytest.raises(ValueError, match="terminal states"):
        TopicInvestigationState("question", status="failed")


def test_state_rejects_budget_exhaustion():
    state = TopicInvestigationState("question")
    counters = dict(state.counters, retrieval_rounds=3)
    with pytest.raises(ValueError, match="retrieval_rounds"):
        replace(state, counters=counters)


def test_context_rejects_query_mismatch():
    evidence = _evidence()
    with pytest.raises(ValueError, match="same query_ir"):
        ContextPackage(query_ir=QueryIR("different query"), evidence=evidence)


def test_evidence_rejects_invalid_sufficiency_and_span():
    with pytest.raises(ValueError, match="sufficiency"):
        EvidenceSet(query_ir=_query(), sufficiency="unknown")
    with pytest.raises(ValueError, match="ordered"):
        EvidenceHit("chunk:1", "doc:1", source_span={"offset_start": 10, "offset_end": 5})


def test_contracts_reject_non_serializable_runtime_data():
    with pytest.raises(ValueError, match="JSON serializable"):
        QueryIR("question", constraints={"invalid": object()})
    with pytest.raises(ValueError, match="JSON serializable"):
        TraceEvent("event", data={"invalid": float("nan")})
