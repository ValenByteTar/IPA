"""Fase 2 contract tests: UserTopicRecord + UserEvidence.

Covers:
  - JSON Schema validity (Draft 2020-12) for both new contracts
  - Valid and invalid records against the schemas
  - Runtime dataclass invariants (mastery requires evidence, append-only
    evidence ordering, assessment evidence requires assessment_id)
  - Vocabulary registration (records + invariants)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "validation"))

from ipa.tutor.tutor_contracts import (  # noqa: E402
    EvidenceType,
    GenerationProvenance,
    MasteryStatus,
    SourceRef,
    UserEvidence,
    UserTopicRecord,
)
from validate_agent_contract import validate  # noqa: E402

NOW = "2026-09-07T12:00:00Z"
LATER = "2026-09-07T13:00:00Z"


def _generation() -> GenerationProvenance:
    return GenerationProvenance(
        generator="test",
        generated_at=NOW,
        input_hash="sha256:" + "a" * 64,
        model_fingerprint="test-model",
    )


def _source_ref() -> SourceRef:
    return SourceRef(
        source_id="chunk:test123",
        source_type="chunk",
        content_hash="sha256:" + "b" * 64,
    )


def _origins(fields: list[str]) -> dict[str, str]:
    return {field: "generated" for field in field_origins_keys(field)}


def field_origins_keys(fields):
    return {f: "generated" for f in fields}


# ---------------------------------------------------------------------------
# Schema validity
# ---------------------------------------------------------------------------

def test_user_topic_record_schema_is_valid_draft2020():
    from jsonschema import Draft202012Validator
    schema = _load_schema("user_topic_record.schema.json")
    Draft202012Validator.check_schema(schema)


def test_user_evidence_schema_is_valid_draft2020():
    from jsonschema import Draft202012Validator
    schema = _load_schema("user_evidence.schema.json")
    Draft202012Validator.check_schema(schema)


def _load_schema(name: str) -> dict:
    import json
    path = Path(__file__).parents[1] / "contracts" / name
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# UserTopicRecord — schema + invariants
# ---------------------------------------------------------------------------

def _topic_record(**overrides):
    base = {
        "record_id": "user_topic_record:asyncio",
        "topic_id": "concept:asyncio",
        "mastery_status": "understood",
        "mastery_score": 0.8,
        "attempts": 3,
        "last_assessment_id": "assessment:001",
        "evidence_ids": ["user_evidence:001"],
        "updated_at": LATER,
        "created_at": NOW,
        "generation": {
            "generator": "tutor",
            "generated_at": NOW,
            "input_hash": "sha256:" + "a" * 64,
            "model_fingerprint": "qwen3.5-9b",
        },
        "field_origins": {
            "mastery_status": "generated",
            "mastery_score": "generated",
            "attempts": "system",
            "last_assessment_id": "system",
            "evidence_ids": "system",
        },
    }
    base.update(overrides)
    return base


def test_user_topic_record_contract_validates():
    from validate_agent_contract import validate
    assert validate("UserTopicRecord", _topic_record()) == []


def test_user_topic_record_mastery_requires_assessment():
    from validate_agent_contract import validate
    record = _topic_record(last_assessment_id=None)
    errors = validate("UserTopicRecord", record)
    assert any("last_assessment_id" in e for e in errors)


def test_user_topic_record_unknown_must_not_reference_assessment():
    from validate_agent_contract import validate
    record = _topic_record(mastery_status="unknown", last_assessment_id="assessment:001")
    errors = validate("UserTopicRecord", record)
    assert errors, "unknown mastery with an assessment reference must be rejected"
    assert any("last_assessment_id" in e for e in errors)


def test_user_topic_record_rejects_bad_score():
    from validate_agent_contract import validate
    errors = validate("UserTopicRecord", _topic_record(mastery_score=1.5))
    assert any("mastery_score" in e for e in errors)


# ---------------------------------------------------------------------------
# UserEvidence — schema + invariants
# ---------------------------------------------------------------------------

def _evidence(**overrides):
    base = {
        "evidence_id": "user_evidence:001",
        "topic_id": "concept:asyncio",
        "evidence_type": "assessment",
        "assessment_id": "assessment:001",
        "session_id": "agent_session:001",
        "observation": "Learner explained event loops correctly.",
        "observed_at": NOW,
        "recorded_at": LATER,
        "source_refs": [{
            "source_id": "assessment:001",
            "source_type": "assessment",
            "content_hash": "sha256:" + "c" * 64,
        }],
        "generation": {
            "generator": "tutor",
            "generated_at": LATER,
            "input_hash": "sha256:" + "d" * 64,
            "model_fingerprint": "tutor-runtime",
        },
        "field_origins": {"observation": "generated", "source_refs": "source"},
    }
    base.update(overrides)
    return base


def test_user_evidence_contract_validates():
    from validate_agent_contract import validate
    assert validate("UserEvidence", _evidence()) == []


def test_user_evidence_assessment_requires_assessment_id():
    from validate_agent_contract import validate
    errors = validate("UserEvidence", _evidence(assessment_id=None))
    assert any("assessment_id" in e for e in errors)


def test_user_evidence_rejects_observed_after_recorded():
    from validate_agent_contract import validate
    errors = validate("UserEvidence", _evidence(observed_at=LATER, recorded_at=NOW))
    assert any("observed_at" in e for e in errors)


def test_user_evidence_requires_source_refs():
    from validate_agent_contract import validate
    errors = validate("UserEvidence", _evidence(source_refs=[]))
    assert any("source_refs" in e for e in errors)


# ---------------------------------------------------------------------------
# Runtime dataclasses
# ---------------------------------------------------------------------------

def _gen() -> GenerationProvenance:
    return GenerationProvenance(
        generator="tutor", generated_at=NOW,
        input_hash="sha256:" + "a" * 64, model_fingerprint="m",
    )


def _sref() -> SourceRef:
    return SourceRef(source_id="chunk:x1", source_type="chunk", content_hash=None)


def test_user_topic_record_dataclass_valid():
    rec = UserTopicRecord(
        record_id="user_topic_record:1", topic_id="concept:1",
        mastery_status=MasteryStatus.UNDERSTOOD, mastery_score=0.85,
        attempts=2, last_assessment_id="assessment:1",
        evidence_ids=["user_evidence:1"], updated_at=LATER, created_at=NOW,
        generation=_gen(),
        field_origins={"mastery_status": "generated", "mastery_score": "generated",
                       "attempts": "system", "last_assessment_id": "system",
                       "evidence_ids": "system"},
    )
    assert rec.mastery_status == MasteryStatus.UNDERSTOOD


def test_user_topic_record_dataclass_mastery_requires_assessment():
    with pytest.raises(ValueError, match="requires assessment evidence"):
        UserTopicRecord(
            record_id="user_topic_record:1", topic_id="concept:1",
            mastery_status=MasteryStatus.APPLIED, mastery_score=0.9,
            attempts=1, last_assessment_id=None,
            evidence_ids=[], updated_at=LATER, created_at=NOW,
            generation=_gen(),
            field_origins={"mastery_status": "generated", "mastery_score": "generated",
                           "attempts": "system", "last_assessment_id": "system",
                           "evidence_ids": "system"},
        )


def test_user_evidence_dataclass_assessment_requires_id():
    with pytest.raises(ValueError, match="assessment_id"):
        UserEvidence(
            evidence_id="user_evidence:1", topic_id="concept:1",
            evidence_type=EvidenceType.ASSESSMENT,
            observation="observed", observed_at=NOW, recorded_at=LATER,
            source_refs=[_sref()], generation=_gen(),
            field_origins={"observation": "generated", "source_refs": "source"},
            assessment_id=None,
        )


def test_user_evidence_dataclass_conversation_needs_no_assessment():
    ev = UserEvidence(
        evidence_id="user_evidence:2", topic_id="concept:1",
        evidence_type=EvidenceType.CONVERSATION,
        observation="asked about event loops", observed_at=NOW, recorded_at=LATER,
        source_refs=[_sref()], generation=_gen(),
        field_origins={"observation": "generated", "source_refs": "source"},
    )
    assert ev.assessment_id is None


# ---------------------------------------------------------------------------
# Vocabulary registration
# ---------------------------------------------------------------------------

def test_vocabulary_registers_fase2_records():
    import json
    vocab = json.loads((Path(__file__).parents[1] / "contracts" / "contract_vocabulary.json").read_text(encoding="utf-8"))
    assert "UserTopicRecord" in vocab["records"]
    assert "UserEvidence" in vocab["records"]
    assert "user_topic_records_require_evidence_for_mastery" in vocab["invariants"]
    assert "user_evidence_is_append_only" in vocab["invariants"]
