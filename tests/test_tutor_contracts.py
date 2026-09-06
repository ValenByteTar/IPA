from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "validation"))
from validate_tutor_contract import CONTRACTS, SCHEMAS, load_schema, validate

from ipa.tutor.tutor_contracts import (
    AssessmentResult,
    AssessmentType,
    GenerationProvenance,
    HumanApproval,
    LearningGoal,
    LearningGoalStatus,
    MasteryStatus,
    RecommendedAction,
    ResearchBudget,
    RoadmapUnit,
    SourceRef,
    SourceSpanRef,
    answer_hash,
    contract_dict,
)

HASH = "sha256:" + "a" * 64
NOW = "2026-08-30T12:00:00Z"
LATER = "2026-08-30T13:00:00Z"


def _source(source_id: str = "chunk:abc") -> dict:
    return {
        "source_id": source_id,
        "source_type": "chunk",
        "source_span": {"page": 1, "offset_start": 0, "offset_end": 42},
        "content_hash": HASH,
    }


def _generation() -> dict:
    return {
        "generator": "ipa.providers.exl3_provider",
        "generated_at": NOW,
        "input_hash": HASH,
        "model_fingerprint": "Qwen3.5-9B-EXL3-3.0bpw",
        "prompt_fingerprint": "tutor-contract-v1",
    }


def _approval() -> dict:
    return {
        "decision": "approved",
        "decided_at": NOW,
        "decided_by": "user:valen",
        "note": "Approved for the learning goal.",
    }


def _goal() -> dict:
    return {
        "goal_id": "goal:hybrid-rag",
        "title": "DiseÃ±ar sistemas Hybrid RAG",
        "description": "Aprender a diseÃ±ar y evaluar recuperaciÃ³n hÃ­brida.",
        "status": "confirmed",
        "success_criteria": ["DiseÃ±ar una arquitectura justificando lexical y vectorial"],
        "constraints": ["Usar fuentes primarias"],
        "created_at": NOW,
        "updated_at": LATER,
        "approval": _approval(),
        "generation": None,
        "field_origins": {
            "title": "user",
            "description": "user",
            "success_criteria": "user",
        },
    }


def _concept() -> dict:
    return {
        "concept_id": "concept:hybrid-retrieval",
        "title": "Hybrid retrieval",
        "definition": "CombinaciÃ³n explÃ­cita de recuperaciÃ³n lexical y vectorial.",
        "status": "validated",
        "difficulty": 0.6,
        "prerequisite_ids": ["concept:bm25", "concept:embeddings"],
        "learning_objectives": ["Explicar cuÃ¡ndo lexical y vectorial se complementan"],
        "mastery_criteria": ["DiseÃ±ar una estrategia de fusiÃ³n para un caso nuevo"],
        "common_misconceptions": ["Hybrid significa usar solo dos modelos densos"],
        "source_refs": [_source()],
        "created_at": NOW,
        "updated_at": LATER,
        "generation": _generation(),
        "field_origins": {
            "title": "source",
            "definition": "source",
            "difficulty": "generated",
            "prerequisite_ids": "generated",
            "learning_objectives": "generated",
            "mastery_criteria": "generated",
        },
    }


def _unit(index: int) -> dict:
    return {
        "unit_id": f"unit:{index}",
        "order": index,
        "concept_id": f"concept:topic-{index}",
        "reason": f"Unidad {index} necesaria para progresar.",
        "estimated_effort_minutes": 30,
        "source_refs": [_source(f"chunk:unit-{index}")],
        "assessment_types": ["explanation", "application"],
    }


def _roadmap() -> dict:
    return {
        "roadmap_id": "roadmap:hybrid-rag-v1",
        "goal_id": "goal:hybrid-rag",
        "version": 1,
        "status": "approved",
        "units": [_unit(1), _unit(2), _unit(3)],
        "assumptions": ["El usuario conoce Python"],
        "uncertainties": ["Dominio actual de evaluaciÃ³n IR"],
        "change_reason": None,
        "previous_roadmap_id": None,
        "created_at": NOW,
        "approval": _approval(),
        "generation": _generation(),
        "field_origins": {
            "goal_id": "user",
            "units": "generated",
            "assumptions": "generated",
            "uncertainties": "generated",
            "change_reason": "generated",
        },
    }


def _assessment() -> dict:
    return {
        "assessment_id": "assessment:001",
        "session_id": "session:001",
        "concept_id": "concept:hybrid-retrieval",
        "assessment_type": "application",
        "answer_hash": HASH,
        "rubric_id": "rubric:hybrid-application-v1",
        "score": 0.82,
        "status": "applied",
        "confidence": 0.78,
        "strengths": ["SeparÃ³ lexical y vectorial"],
        "gaps": ["No explicÃ³ reintentos"],
        "misconceptions": [],
        "evidence": [_source("assessment:attempt-001")],
        "recommended_action": "advance",
        "created_at": NOW,
        "generation": _generation(),
        "field_origins": {
            "answer_hash": "user",
            "score": "generated",
            "status": "generated",
            "strengths": "generated",
            "gaps": "generated",
            "misconceptions": "generated",
            "evidence": "source",
            "recommended_action": "generated",
        },
    }


def _research() -> dict:
    return {
        "request_id": "research:001",
        "goal_id": "goal:hybrid-rag",
        "concept_id": "concept:bm25",
        "question": "Â¿CuÃ¡l es la formulaciÃ³n original y vigente de BM25?",
        "trigger": "missing_primary_source",
        "gap_evidence": [_source("assessment:gap-001")],
        "allowed_domains": ["dl.acm.org", "microsoft.com"],
        "budget": {"max_urls": 20, "max_seconds": 600, "max_bytes": 50000000, "max_depth": 2},
        "status": "approved",
        "job_id": None,
        "result_source_refs": [],
        "created_at": NOW,
        "updated_at": LATER,
        "approval": _approval(),
        "generation": _generation(),
        "field_origins": {
            "question": "generated",
            "trigger": "system",
            "gap_evidence": "source",
            "allowed_domains": "user",
            "budget": "system",
        },
    }


VALID_RECORDS = {
    "LearningGoal": _goal,
    "Concept": _concept,
    "Roadmap": _roadmap,
    "AssessmentResult": _assessment,
    "ResearchRequest": _research,
}


@pytest.mark.parametrize("record_type", SCHEMAS)
def test_schema_is_valid_draft202012(record_type):
    Draft202012Validator.check_schema(load_schema(record_type))


def test_common_schema_is_valid_draft202012():
    schema = json.loads((CONTRACTS / "tutor_common.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)


@pytest.mark.parametrize("record_type", SCHEMAS)
def test_valid_record_passes(record_type):
    assert validate(record_type, VALID_RECORDS[record_type]()) == []


@pytest.mark.parametrize("record_type", SCHEMAS)
def test_unknown_top_level_field_fails(record_type):
    record = VALID_RECORDS[record_type]()
    record["unexpected"] = True
    assert any("unexpected" in error for error in validate(record_type, record))


def test_confirmed_goal_requires_human_approval():
    goal = _goal()
    goal["approval"] = None
    assert any("approval" in error for error in validate("LearningGoal", goal))


def test_generated_goal_field_requires_generation_provenance():
    goal = _goal()
    goal["field_origins"]["description"] = "generated"
    assert any("generation provenance" in error for error in validate("LearningGoal", goal))


def test_concept_cannot_be_own_prerequisite():
    concept = _concept()
    concept["prerequisite_ids"].append(concept["concept_id"])
    assert any("own prerequisite" in error for error in validate("Concept", concept))


def test_source_span_must_be_ordered():
    concept = _concept()
    concept["source_refs"][0]["source_span"] = {"page": 1, "offset_start": 50, "offset_end": 10}
    assert any("offset_start" in error for error in validate("Concept", concept))


def test_roadmap_requires_three_to_seven_units():
    roadmap = _roadmap()
    roadmap["units"] = roadmap["units"][:2]
    assert any("units" in error for error in validate("Roadmap", roadmap))


def test_roadmap_order_must_be_contiguous():
    roadmap = _roadmap()
    roadmap["units"][2]["order"] = 4
    assert any("contiguous" in error for error in validate("Roadmap", roadmap))


def test_roadmap_version_requires_predecessor_and_reason():
    roadmap = _roadmap()
    roadmap["version"] = 2
    errors = validate("Roadmap", roadmap)
    assert any("previous_roadmap_id" in error or "change_reason" in error for error in errors)


def test_low_score_cannot_advance():
    assessment = _assessment()
    assessment["score"] = 0.4
    assert any("score" in error for error in validate("AssessmentResult", assessment))


def test_misconception_requires_evidence_entry():
    assessment = _assessment()
    assessment["status"] = "misconception"
    assert any("misconceptions" in error for error in validate("AssessmentResult", assessment))


def test_research_requires_bounded_budget():
    request = _research()
    request["budget"]["max_urls"] = 0
    assert any("max_urls" in error for error in validate("ResearchRequest", request))


def test_running_research_requires_approval_and_job():
    request = _research()
    request["status"] = "running"
    request["approval"] = None
    errors = validate("ResearchRequest", request)
    assert any("approval" in error for error in errors)
    assert any("job_id" in error for error in errors)


def test_answer_hash_is_stable_and_content_sensitive():
    assert answer_hash("respuesta") == answer_hash("respuesta")
    assert answer_hash("respuesta") != answer_hash("otra")
    assert answer_hash("respuesta").startswith("sha256:")


def test_runtime_goal_enforces_human_approval():
    with pytest.raises(ValueError, match="human approval"):
        LearningGoal(
            goal_id="goal:test",
            title="Test",
            description="Test goal",
            status=LearningGoalStatus.CONFIRMED,
            success_criteria=["Complete test"],
            created_at=NOW,
            updated_at=NOW,
            approval=None,
            field_origins={"title": "user", "description": "user", "success_criteria": "user"},
        )


def test_runtime_assessment_serializes_to_schema_shape():
    generation = GenerationProvenance(
        generator="test", generated_at=NOW, input_hash=HASH,
        model_fingerprint="test-model", prompt_fingerprint=None,
    )
    result = AssessmentResult(
        assessment_id="assessment:runtime",
        session_id="session:runtime",
        concept_id="concept:runtime",
        assessment_type=AssessmentType.APPLICATION,
        answer_hash=HASH,
        rubric_id="rubric:runtime",
        score=0.8,
        status=MasteryStatus.APPLIED,
        strengths=["Correct"],
        gaps=[],
        misconceptions=[],
        evidence=[SourceRef("assessment:attempt", "assessment")],
        recommended_action=RecommendedAction.ADVANCE,
        created_at=NOW,
        generation=generation,
        field_origins={
            "answer_hash": "user", "score": "generated", "status": "generated",
            "strengths": "generated", "gaps": "generated",
            "misconceptions": "generated", "evidence": "source",
            "recommended_action": "generated",
        },
        confidence=0.8,
    )
    assert validate("AssessmentResult", contract_dict(result)) == []


def test_completed_research_schema_requires_job_and_results():
    request = _research()
    request["status"] = "completed"
    del request["job_id"]
    del request["result_source_refs"]
    errors = validate("ResearchRequest", request)
    assert any("job_id" in error for error in errors)
    assert any("result_source_refs" in error for error in errors)


def test_runtime_source_ref_rejects_invalid_type_and_hash():
    with pytest.raises(ValueError, match="source type"):
        SourceRef("source:001", "website")
    with pytest.raises(ValueError, match="content_hash"):
        SourceRef("chunk:001", "chunk", content_hash="md5:invalid")


def test_runtime_source_span_rejects_invalid_offsets():
    with pytest.raises(ValueError, match="offsets"):
        SourceSpanRef(offset_start=10, offset_end=5)


def test_runtime_roadmap_unit_enforces_bounds_and_evidence():
    with pytest.raises(ValueError, match="estimated effort"):
        RoadmapUnit(
            unit_id="unit:001",
            order=1,
            concept_id="concept:001",
            reason="Test",
            estimated_effort_minutes=2,
            source_refs=[SourceRef("chunk:001", "chunk")],
            assessment_types=[AssessmentType.APPLICATION],
        )


def test_runtime_research_budget_enforces_bounds():
    with pytest.raises(ValueError, match="max_urls"):
        ResearchBudget(max_urls=0, max_seconds=60, max_bytes=1000, max_depth=1)
