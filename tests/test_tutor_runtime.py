"""Fase 2 runtime tests: Tutor role with its own state scope.

Covers:
  - TutorStore: topic records upsert/get/list, append-only evidence
  - TutorSession: requires role='tutor', mastery-aware lesson payload
  - diagnose(): deterministic scaffold (no LLM)
  - assess(): LLM JSON classification + abstention + mastery update
  - The full loop: diagnose → lesson → assess → mastery update
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "validation"))

from ipa.agent import AgentCore, AgentMemory, ToolContext, load_identity  # noqa: E402
from ipa.tutor.tutor_contracts import (  # noqa: E402
    EvidenceType,
    GenerationProvenance,
    MasteryStatus,
    SourceRef,
    UserEvidence,
    UserTopicRecord,
)
from ipa.tutor.tutor_runtime import (  # noqa: E402
    ABSTENTION_THRESHOLD,
    TutorSession,
    TutorStore,
)


@dataclass
class FakeGenerationResult:
    text: str = ""
    error: str | None = None


class FakeProvider:
    """Scripted chat provider for deterministic tutor tests."""
    def __init__(self, responses=None, error=None):
        self.responses = list(responses or [])
        self.calls = []
        self.error = error

    def generate_chat(self, messages, *, max_new_tokens=None, temperature=None, **kw):
        self.calls.append({"messages": messages, "max_new_tokens": max_new_tokens})
        if self.error:
            return FakeGenerationResult(text="", error=self.error)
        if self.responses:
            return FakeGenerationResult(text=self.responses.pop(0))
        return FakeGenerationResult(text="{}")


@pytest.fixture()
def tutor_env(tmp_path):
    memory = AgentMemory(store_path=tmp_path / "agent.db")
    core = AgentCore(interface="cli", role="tutor", memory=memory)
    store = TutorStore(tmp_path / "tutor.db")
    return core, store


# ---------------------------------------------------------------------------
# TutorStore
# ---------------------------------------------------------------------------

def test_tutor_store_roundtrip_topic_record(tutor_env):
    _, store = tutor_env
    from ipa.tutor.tutor_contracts import GenerationProvenance
    record = _make_record("concept:asyncio", MasteryStatus.UNDERSTOOD, 0.8)
    store.upsert_topic_record(record)
    loaded = store.get_topic_record("concept:asyncio")
    assert loaded is not None
    assert loaded.mastery_status == MasteryStatus.UNDERSTOOD
    assert loaded.mastery_score == 0.8
    assert loaded.attempts == 3


def _make_record(topic_id, status, score, attempts=3):
    from ipa.tutor.tutor_contracts import GenerationProvenance
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    return UserTopicRecord(
        record_id=f"user_topic_record:{topic_id.replace(':', '_')}",
        topic_id=topic_id,
        mastery_status=status,
        mastery_score=score,
        attempts=attempts,
        last_assessment_id="assessment:test001",
        evidence_ids=["user_evidence:test001"],
        updated_at=now,
        created_at=now,
        generation=GenerationProvenance(
            generator="test", generated_at=now,
            input_hash="sha256:" + "a" * 64, model_fingerprint="test",
        ),
        field_origins={"mastery_status": "generated", "mastery_score": "generated",
                       "attempts": "system", "last_assessment_id": "system",
                       "evidence_ids": "system"},
    )


def test_tutor_store_evidence_is_append_only(tutor_env):
    _, store = tutor_env
    from ipa.tutor.tutor_contracts import GenerationProvenance, SourceRef
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    gen = GenerationProvenance(generator="t", generated_at=now,
                               input_hash="sha256:" + "a" * 64, model_fingerprint="m")
    ev = UserEvidence(
        evidence_id="user_evidence:dup", topic_id="concept:x",
        evidence_type=EvidenceType.CONVERSATION,
        observation="first", observed_at=now, recorded_at=now,
        source_refs=[SourceRef(source_id="s1", source_type="artifact", content_hash=None)],
        generation=gen, field_origins={"observation": "generated", "source_refs": "source"},
    )
    store.add_evidence(ev)
    # A second insert with the same id must raise (append-only, no REPLACE)
    with pytest.raises(Exception):
        store.add_evidence(ev)
    assert len(store.list_evidence("concept:x")) == 1


# ---------------------------------------------------------------------------
# TutorSession
# ---------------------------------------------------------------------------

def test_tutor_session_requires_tutor_role(tmp_path):
    memory = AgentMemory(store_path=tmp_path / "agent.db")
    core = AgentCore(interface="cli", role="general", memory=memory)
    store = TutorStore(tmp_path / "tutor.db")
    with pytest.raises(ValueError, match="role='tutor'"):
        TutorSession(core, store)


def test_diagnose_new_topic_is_unknown(tutor_env):
    core, store = tutor_env
    tutor = TutorSession(core, store)
    diagnosis = tutor.diagnose("concept:rust")
    assert diagnosis.mastery_status == MasteryStatus.UNKNOWN
    assert diagnosis.source == "new_topic"
    assert diagnosis.next_action.value == "human_review"


def test_diagnose_uses_store_state(tutor_env):
    core, store = tutor_env
    store.upsert_topic_record(_make_record("concept:asyncio", MasteryStatus.APPLIED, 0.9))
    tutor = TutorSession(core, store)
    diagnosis = tutor.diagnose("concept:asyncio")
    assert diagnosis.mastery_status == MasteryStatus.APPLIED
    assert diagnosis.next_action.value == "advance"
    assert diagnosis.source == "store"


def test_lesson_payload_includes_policy_and_mastery(tutor_env):
    core, store = tutor_env
    store.upsert_topic_record(_make_record("concept:asyncio", MasteryStatus.UNDERSTOOD, 0.8))
    tutor = TutorSession(core, store)
    core.start_session()
    messages = tutor.build_lesson_messages("concept:asyncio", "explícame los event loops")
    system = messages[0]["content"]
    assert "Política pedagógica" in system
    assert "understood" in system
    assert "event loops" in messages[-1]["content"]


def test_assess_requires_provider(tutor_env):
    core, store = tutor_env
    tutor = TutorSession(core, store)
    with pytest.raises(ValueError, match="provider"):
        tutor.assess("concept:x", "q", "a")


def test_assess_full_loop_updates_mastery(tutor_env):
    """diagnóstico → lección → assessment → mastery update (the Fase 2 loop)."""
    core, store = tutor_env
    provider = FakeProvider(responses=[
        # First call: the lesson turn
        "Lección sobre event loops: son el scheduler de coroutines...",
        # Second call: the assessment classification
        json.dumps({
            "score": 0.85,
            "status": "understood",
            "strengths": ["correct event loop explanation"],
            "gaps": [],
            "misconceptions": [],
            "recommended_action": "advance",
            "confidence": 0.9,
            "abstain": False,
        }),
    ])
    tutor = TutorSession(core, store, provider=provider)
    core.start_session()

    # 1. Diagnóstico: tema nuevo → unknown
    d1 = tutor.diagnose("concept:asyncio")
    assert d1.mastery_status == MasteryStatus.UNKNOWN

    # 2. Lección (con provider real sería el modelo; aquí el responder default)
    lesson = tutor.lesson("concept:asyncio", "explícame asyncio")
    assert lesson["reply"]

    # 3. Assessment: el LLM clasifica la respuesta del alumno
    result = tutor.assess("concept:asyncio", "¿qué es un event loop?", "un event loop programa coroutines")
    assert result["abstained"] is False
    assert result["assessment"]["score"] == 0.85
    assert result["assessment"]["status"] == "understood"
    from validate_tutor_contract import validate as validate_tutor
    assert validate_tutor("AssessmentResult", result["assessment"]) == []

    # 4. Mastery update: el assessment ES la evidencia
    record = result["record"]
    assert record.mastery_status == MasteryStatus.UNDERSTOOD
    assert record.mastery_score == 0.85
    assert record.last_assessment_id == result["assessment"]["assessment_id"]
    assert len(record.evidence_ids) == 1

    # 5. Re-diagnóstico: ahora el store conoce el estado
    d2 = tutor.diagnose("concept:asyncio")
    assert d2.mastery_status == MasteryStatus.UNDERSTOOD
    assert d2.evidence_count == 1


def test_assess_abstains_on_low_confidence(tutor_env):
    """Below the abstention threshold the Tutor refuses to score (BM-006)."""
    core, store = tutor_env
    provider = FakeProvider(responses=[json.dumps({
        "score": 0.5, "status": "needs_review", "confidence": 0.3, "abstain": False,
    })])
    tutor = TutorSession(core, store, provider=provider)
    core.start_session()

    result = tutor.assess("concept:x", "q", "a")
    assert result["abstained"] is True
    assert result["assessment"]["recommended_action"] == "human_review"
    assert result["record"] is None
    # The abstention is recorded as observation evidence (no score invented)
    evidence = store.list_evidence("concept:x")
    assert len(evidence) == 1
    assert "abstained" in evidence[0].observation


def test_assess_abstains_on_provider_error(tutor_env):
    core, store = tutor_env
    tutor = TutorSession(core, store, provider=FakeProvider(error="GPU OOM"))
    core.start_session()
    result = tutor.assess("concept:x", "q", "a")
    assert result["abstained"] is True
    assert "provider error" in result["assessment"]["reason"]


def test_assess_abstains_on_unparseable_output(tutor_env):
    core, store = tutor_env
    tutor = TutorSession(core, store, provider=FakeProvider(responses=["no json here"]))
    core.start_session()
    result = tutor.assess("concept:x", "q", "a")
    assert result["abstained"] is True
    assert tutor.last_fallback_reason is not None
    assert "unparseable" in tutor.last_fallback_reason


def test_mastery_persists_across_tutor_sessions(tutor_env):
    """The Tutor's state scope survives sessions (DEC-002 state scope)."""
    core, store = tutor_env
    provider = FakeProvider(responses=[json.dumps({
        "score": 0.9, "status": "applied", "confidence": 0.95, "abstain": False,
        "recommended_action": "advance",
    })])
    tutor1 = TutorSession(core, store, provider=provider)
    core.start_session(title="s1")
    result = tutor1.assess("concept:asyncio", "q", "a")

    # New session, same store: the mastery state is still there
    core2 = AgentCore(interface="cli", role="tutor", memory=core.memory)
    tutor2 = TutorSession(core2, store)
    d = tutor2.diagnose("concept:asyncio")
    assert d.mastery_status == MasteryStatus.APPLIED
    assert d.mastery_score == 0.9


# ---------------------------------------------------------------------------
# Roadmap pedagógico: LLM propone, humano aprueba (Fase 2)
# ---------------------------------------------------------------------------

CONCEPTS = [
    {"concept_id": "concept:bm25", "title": "BM25", "definition": "Lexical sparse retrieval"},
    {"concept_id": "concept:embeddings", "title": "Embeddings", "definition": "Dense vector representations"},
    {"concept_id": "concept:hybrid", "title": "Hybrid retrieval", "definition": "Fusion of lexical and dense"},
    {"concept_id": "concept:reranking", "title": "Reranking", "definition": "Re-ordering candidates"},
    {"concept_id": "concept:chunking", "title": "Chunking", "definition": "Splitting documents for indexing"},
]

ROADMAP_LLM_RESPONSE = json.dumps({
    "units": [
        {"concept_id": "concept:chunking", "reason": "base para indexar", "estimated_effort_minutes": 30,
         "assessment_types": ["explanation"]},
        {"concept_id": "concept:bm25", "reason": "lexical primero", "estimated_effort_minutes": 45,
         "assessment_types": ["explanation", "application"]},
        {"concept_id": "concept:embeddings", "reason": "denso después", "estimated_effort_minutes": 45,
         "assessment_types": ["explanation", "retrieval"]},
        {"concept_id": "concept:hybrid", "reason": "fusión de ambos", "estimated_effort_minutes": 60,
         "assessment_types": ["application", "critique"]},
    ],
    "assumptions": ["El alumno conoce Python básico"],
    "uncertainties": ["Nivel real de SQL por confirmar"],
})


def test_propose_roadmap_creates_proposed_status(tutor_env):
    """The LLM proposes; the roadmap lands as status='proposed' — never active."""
    core, store = tutor_env
    provider = FakeProvider(responses=[ROADMAP_LLM_RESPONSE])
    tutor = TutorSession(core, store, provider=provider)

    roadmap = tutor.propose_roadmap("goal:rag", CONCEPTS)
    assert roadmap.status.value == "proposed"
    assert roadmap.approval is None
    assert 3 <= len(roadmap.units) <= 7
    assert [u.order for u in roadmap.units] == [1, 2, 3, 4]
    # Every unit references a known concept
    known = {c["concept_id"] for c in CONCEPTS}
    assert all(u.concept_id in known for u in roadmap.units)
    # Persisted
    assert store.get_roadmap(roadmap.roadmap_id) is not None


def test_proposed_roadmap_cannot_be_activated(tutor_env):
    """The approval gate: proposed → active is forbidden without approval."""
    core, store = tutor_env
    tutor = TutorSession(core, store, provider=FakeProvider(responses=[ROADMAP_LLM_RESPONSE]))
    roadmap = tutor.propose_roadmap("goal:1", CONCEPTS)
    with pytest.raises(ValueError, match="only approved"):
        tutor.activate_roadmap(roadmap.roadmap_id)


def test_approve_then_activate_flow(tutor_env):
    """propose → approve (human) → activate: the full gate."""
    core, store = tutor_env
    tutor = TutorSession(core, store, provider=FakeProvider(responses=[ROADMAP_LLM_RESPONSE]))
    roadmap = tutor.propose_roadmap("goal:rag", CONCEPTS)

    approved = tutor.approve_roadmap(roadmap.roadmap_id, decided_by="Valen", note="ok")
    assert approved.status.value == "approved"
    assert approved.approval.decision.value == "approved"
    assert approved.approval.decided_by == "Valen"

    active = tutor.activate_roadmap(roadmap.roadmap_id)
    assert active.status.value == "active"
    assert active.approval is not None and active.approval.decided_by == "Valen"


def test_reject_roadmap(tutor_env):
    core, store = tutor_env
    tutor = TutorSession(core, store, provider=FakeProvider(responses=[ROADMAP_LLM_RESPONSE]))
    roadmap = tutor.propose_roadmap("goal:x", CONCEPTS)
    rejected = tutor.reject_roadmap(roadmap.roadmap_id, decided_by="Valen", note="muy largo")
    assert rejected.status.value == "rejected"
    # A rejected roadmap cannot be activated
    with pytest.raises(ValueError, match="only approved"):
        tutor.activate_roadmap(roadmap.roadmap_id)


def test_supersede_roadmap_preserves_record(tutor_env):
    """Debate: la propuesta anterior pasa a superseded, nunca se borra, y
    no puede activarse."""
    core, store = tutor_env
    tutor = TutorSession(core, store, provider=FakeProvider(responses=[ROADMAP_LLM_RESPONSE]))
    roadmap = tutor.propose_roadmap("goal:x", CONCEPTS)
    superseded = tutor.supersede_roadmap(roadmap.roadmap_id, decided_by="dashboard")
    assert superseded.status.value == "superseded"
    # El registro se preserva (unidades intactas)
    kept = store.get_roadmap(roadmap.roadmap_id)
    assert kept is not None
    assert kept.status.value == "superseded"
    assert len(kept.units) == len(roadmap.units)
    with pytest.raises(ValueError, match="only approved"):
        tutor.activate_roadmap(roadmap.roadmap_id)
    # Solo propuestas pueden supersederse
    with pytest.raises(ValueError, match="only proposed"):
        tutor.supersede_roadmap(roadmap.roadmap_id, decided_by="dashboard")


def test_propose_roadmap_with_feedback(tutor_env):
    """El feedback del debate entra al prompt del LLM."""
    core, store = tutor_env
    provider = FakeProvider(responses=[ROADMAP_LLM_RESPONSE])
    tutor = TutorSession(core, store, provider=provider)
    tutor.propose_roadmap(
        "goal:x", CONCEPTS, version=2, previous_roadmap_id="roadmap:old",
        change_reason="debate", feedback="quiero más práctica y menos teoría",
    )
    sent = json.dumps(provider.calls[0]["messages"], ensure_ascii=False)
    assert "más práctica y menos teoría" in sent


def test_active_roadmap_requires_human_approval_invariant(tutor_env):
    """The contract invariant: no code path can reach active without approval."""
    core, store = tutor_env
    tutor = TutorSession(core, store, provider=FakeProvider(responses=[ROADMAP_LLM_RESPONSE]))
    roadmap = tutor.propose_roadmap("goal:x", CONCEPTS)
    # Attempting to activate directly from proposed must fail
    with pytest.raises(ValueError, match="only approved"):
        tutor.activate_roadmap(roadmap.roadmap_id)
    # approve → activate works
    tutor.approve_roadmap(roadmap.roadmap_id, decided_by="Valen")
    active = tutor.activate_roadmap(roadmap.roadmap_id)
    assert active.status.value == "active"


def test_roadmap_contract_validates(tutor_env):
    """The proposed roadmap validates against the Roadmap JSON Schema."""
    core, store = tutor_env
    tutor = TutorSession(core, store, provider=FakeProvider(responses=[ROADMAP_LLM_RESPONSE]))
    roadmap = tutor.propose_roadmap("goal:x", CONCEPTS)
    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "validation"))
    from validate_tutor_contract import validate as validate_tutor
    from dataclasses import asdict
    payload = asdict(roadmap)
    payload["status"] = roadmap.status.value
    for unit in payload["units"]:
        unit["assessment_types"] = [at.value for at in unit["assessment_types"]]
    assert validate_tutor("Roadmap", payload) == []


def test_roadmap_versioning_requires_change_reason(tutor_env):
    """Version >1 requires previous_roadmap_id + change_reason (supersedes)."""
    core, store = tutor_env
    tutor = TutorSession(core, store, provider=FakeProvider(responses=[ROADMAP_LLM_RESPONSE]))
    with pytest.raises(ValueError, match="predecessor and change reason"):
        tutor.propose_roadmap("goal:x", CONCEPTS, version=2)  # no previous, no reason


def test_roadmap_persists_across_sessions(tutor_env):
    core, store = tutor_env
    tutor1 = TutorSession(core, store, provider=FakeProvider(responses=[ROADMAP_LLM_RESPONSE]))
    roadmap = tutor1.propose_roadmap("goal:persist", CONCEPTS)

    core2 = AgentCore(interface="cli", role="tutor", memory=core.memory)
    tutor2 = TutorSession(core2, store)
    loaded = tutor2.store.get_roadmap(roadmap.roadmap_id)
    assert loaded is not None
    assert loaded.status.value == "proposed"
    assert len(loaded.units) == 4


def test_propose_roadmap_absorbs_unknown_concepts(tutor_env):
    """El scaffold absorbe propuestas imperfectas: conceptos desconocidos se
    descartan y el top-up completa al mínimo del contrato — sin dead-end."""
    core, store = tutor_env
    bad_response = json.dumps({
        "units": [
            {"concept_id": "concept:nonexistent", "reason": "x", "estimated_effort_minutes": 30,
             "assessment_types": ["explanation"]},
            {"concept_id": "concept:bm25", "reason": "y", "estimated_effort_minutes": 30,
             "assessment_types": ["explanation"]},
            {"concept_id": "concept:embeddings", "reason": "z", "estimated_effort_minutes": 30,
             "assessment_types": ["explanation"]},
        ],
        "assumptions": [], "uncertainties": [],
    })
    tutor = TutorSession(core, store, provider=FakeProvider(responses=[bad_response]))
    roadmap = tutor.propose_roadmap("goal:x", CONCEPTS)
    ids = [u.concept_id for u in roadmap.units]
    assert "concept:nonexistent" not in ids
    assert len(ids) == 3 and len(set(ids)) == 3
    assert roadmap.status.value == "proposed"


def test_propose_roadmap_truncates_above_contract_maximum(tutor_env):
    """Un pedido explícito de 9 unidades se ajusta al rango del contrato
    (3-7): el prompt pide 7 y el scaffold trunca excedentes."""
    core, store = tutor_env
    many_concepts = [
        {**c, "concept_id": f"{c['concept_id']}:v{i // 5}"}
        for i, c in enumerate(CONCEPTS * 2, start=1)
    ]
    many = json.dumps({
        "units": [
            {"concept_id": c["concept_id"], "reason": f"u{i}", "estimated_effort_minutes": 30,
             "assessment_types": ["explanation"]}
            for i, c in enumerate(many_concepts, start=1)
        ],
        "assumptions": [], "uncertainties": [],
    })
    tutor = TutorSession(core, store, provider=FakeProvider(responses=[many]))
    roadmap = tutor.propose_roadmap("goal:big", many_concepts, n_units=9)
    assert len(roadmap.units) == 7
    assert [u.order for u in roadmap.units] == list(range(1, 8))
    ids = [u.concept_id for u in roadmap.units]
    assert len(set(ids)) == len(ids)


def test_propose_roadmap_retries_on_invalid_json(tutor_env):
    """First LLM output malformed → one corrective retry; valid JSON wins."""
    core, store = tutor_env
    provider = FakeProvider(responses=[
        "El roadmap es: {units: [roto",  # JSON inválido
        ROADMAP_LLM_RESPONSE,
    ])
    tutor = TutorSession(core, store, provider=provider)
    roadmap = tutor.propose_roadmap("goal:retry", CONCEPTS)
    assert roadmap.status.value == "proposed"
    assert len(roadmap.units) == 4
    assert len(provider.calls) == 2
    assert "no fue JSON válido" in provider.calls[1]["messages"][1]["content"]


def test_propose_roadmap_falls_back_deterministically(tutor_env):
    """Both LLM attempts malformed → deterministic scaffold roadmap; the
    human gate still applies (status stays 'proposed')."""
    core, store = tutor_env
    provider = FakeProvider(responses=["no json {", "tampoco { va"])
    tutor = TutorSession(core, store, provider=provider)
    roadmap = tutor.propose_roadmap("goal:fallback", CONCEPTS)
    assert roadmap.status.value == "proposed"
    assert roadmap.approval is None
    assert [u.concept_id for u in roadmap.units] == [
        c["concept_id"] for c in CONCEPTS[:5]
    ]
    assert tutor.last_fallback_reason is not None
    assert "roadmap" in tutor.last_fallback_reason


def test_propose_roadmap_dedups_repeated_concepts(tutor_env):
    """El LLM puede reutilizar un concept_id (el prompt lo invita cuando el
    alumno pide N unidades); el scaffold deduplica en vez de violar el
    contrato ('roadmap concept IDs must be unique')."""
    core, store = tutor_env
    dup_response = json.dumps({
        "units": [
            {"concept_id": "concept:chunking", "reason": "introducción",
             "estimated_effort_minutes": 30, "assessment_types": ["explanation"]},
            {"concept_id": "concept:bm25", "reason": "lexical primero",
             "estimated_effort_minutes": 30, "assessment_types": ["explanation"]},
            {"concept_id": "concept:chunking", "reason": "aplicación avanzada",
             "estimated_effort_minutes": 60, "assessment_types": ["application"]},
            {"concept_id": "concept:embeddings", "reason": "denso después",
             "estimated_effort_minutes": 45, "assessment_types": ["explanation"]},
        ],
        "assumptions": [], "uncertainties": [],
    })
    tutor = TutorSession(core, store, provider=FakeProvider(responses=[dup_response]))
    roadmap = tutor.propose_roadmap("goal:dedup", CONCEPTS)
    concept_ids = [u.concept_id for u in roadmap.units]
    assert len(concept_ids) == len(set(concept_ids))
    assert [u.order for u in roadmap.units] == list(range(1, len(concept_ids) + 1))
    assert roadmap.status.value == "proposed"
    assert store.get_roadmap(roadmap.roadmap_id) is not None


def test_propose_roadmap_tops_up_below_minimum_after_dedup(tutor_env):
    """Si tras deduplicar quedan menos de 3 unidades, el scaffold completa
    con conceptos no usados (orden del retrieval); el gate sigue aplicando."""
    core, store = tutor_env
    dup_response = json.dumps({
        "units": [
            {"concept_id": "concept:bm25", "reason": "a", "estimated_effort_minutes": 30,
             "assessment_types": ["explanation"]},
            {"concept_id": "concept:bm25", "reason": "b (repetido)", "estimated_effort_minutes": 30,
             "assessment_types": ["explanation"]},
            {"concept_id": "concept:embeddings", "reason": "c",
             "estimated_effort_minutes": 30, "assessment_types": ["explanation"]},
        ],
        "assumptions": [], "uncertainties": [],
    })
    tutor = TutorSession(core, store, provider=FakeProvider(responses=[dup_response]))
    roadmap = tutor.propose_roadmap("goal:topup", CONCEPTS)
    ids = [u.concept_id for u in roadmap.units]
    assert len(ids) == 3 and len(set(ids)) == 3
    assert ids[2] == "concept:hybrid"  # top-up: primer concepto no usado


def test_diagnose_summary_is_user_facing(tutor_env):
    """El summary del diagnóstico es dato factual para el alumno: sin la
    línea de policy (que es instrucción para el LLM de la lección) y sin
    punto final (el driver lo compone con '. ' → doble punto)."""
    core, store = tutor_env
    tutor = TutorSession(core, store)
    d = tutor.diagnose("concept:rust")
    assert "Comenzá" not in d.summary
    assert not d.summary.endswith(".")
    store.upsert_topic_record(_make_record("concept:asyncio", MasteryStatus.APPLIED, 0.9))
    d2 = tutor.diagnose("concept:asyncio")
    assert "Comenzá" not in d2.summary
    assert not d2.summary.endswith(".")


def test_roadmap_archive_hides_from_lists_but_preserves_record(tutor_env):
    """Archive is operational (not a contract status): hidden from lists and
    driver recovery, still fetchable by id, un-archivable."""
    core, store = tutor_env
    tutor = TutorSession(core, store, provider=FakeProvider(responses=[ROADMAP_LLM_RESPONSE]))
    roadmap = tutor.propose_roadmap("goal:arch", CONCEPTS)

    assert any(r.roadmap_id == roadmap.roadmap_id for r in store.list_roadmaps())
    store.set_roadmap_archived(roadmap.roadmap_id)
    assert all(r.roadmap_id != roadmap.roadmap_id for r in store.list_roadmaps())
    assert store.get_roadmap(roadmap.roadmap_id) is not None
    assert any(r.roadmap_id == roadmap.roadmap_id for r in store.list_roadmaps(include_archived=True))
    store.set_roadmap_archived(roadmap.roadmap_id, archived=False)
    assert any(r.roadmap_id == roadmap.roadmap_id for r in store.list_roadmaps())


# ---------------------------------------------------------------------------
# ResearchRequest: crear → aprobar (humano) → ejecutar (Fase 1 executor)
# ---------------------------------------------------------------------------

def test_create_research_request_starts_pending(tutor_env):
    core, store = tutor_env
    tutor = TutorSession(core, store)
    request = tutor.create_research_request(
        "concept:asyncio", "¿Cómo funciona el event loop de asyncio en profundidad?",
        allowed_domains=["docs.python.org", "realpython.com"],
    )
    assert request.status.value == "pending_approval"
    assert request.approval is None
    assert store.get_research_request(request.request_id) is not None


def test_pending_request_cannot_execute(tutor_env):
    """The contract invariant: research_execution_requires_human_approval."""
    core, store = tutor_env
    tutor = TutorSession(core, store)
    request = tutor.create_research_request(
        "concept:asyncio", "¿Cómo funciona el event loop de asyncio en profundidad?",
    )
    from ipa.tutor.tutor_contracts import ResearchStatus
    assert request.status == ResearchStatus.PENDING_APPROVAL
    # Execution must refuse
    ctx = ToolContext(memory=core.memory)
    with pytest.raises(ValueError, match="only approved"):
        tutor.execute_approved_research(request.request_id, ctx)


def test_approve_then_execute_flow(tutor_env, tmp_path, monkeypatch):
    """pending → approve (humano) → execute (Fase 1 executor) → completed."""
    core, store = tutor_env
    tutor = TutorSession(core, store)
    core.start_session()
    request = tutor.create_research_request(
        "concept:asyncio", "¿Cómo funciona el event loop de asyncio en profundidad?",
        allowed_domains=["docs.python.org"],
    )
    approved = tutor.approve_research_request(request.request_id, decided_by="Valen")
    assert approved.status.value == "approved"
    assert approved.approval.decided_by == "Valen"

    # Mock the web layer (search + scraper) — the gate under test is the
    # approval flow, not the network.
    from ipa.agent import research_executor as re_module
    from ipa.agent.web_search import SearchResult, SearchSummary

    def fake_search_web(query, **kwargs):
        return SearchSummary(query=query, results=[
            SearchResult("https://docs.python.org/asyncio", "Python asyncio event loop tutorial",
                         "python asyncio event loop tutorial coroutines", "docs.python.org"),
        ])

    monkeypatch.setattr(re_module, "search_web", fake_search_web)

    class FakeScrapeResult:
        success = True
        error = None
        date = None
        canonical_url = None
        image_paths = []
        document_paths = []
        def __init__(self):
            self.url = "https://docs.python.org/asyncio"
            self.text = "Python asyncio tutorial content. " * 60
            self.title = "Asyncio"
            self.content_hash = None
            self.quality_score = 0.9
            self.metadata = {"word_count": "500", "engine": "requests"}

    class FakeScraper:
        def __init__(self, **kw):
            pass
        def extract_article(self, url, days_back=0):
            return FakeScrapeResult()
        def save_article(self, result):
            return tmp_path / "landing" / "x.txt"

    monkeypatch.setattr("ipa.acquisition.web_scraper.WebScraper", FakeScraper, raising=True)

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    from ipa.storage.document_store import DocumentStore
    from ipa.indexes.bm25_index import BM25Index
    DocumentStore(corpus / "document_store.db").close()
    BM25Index(corpus / "bm25_index.db").close()
    ctx = ToolContext(memory=core.memory, corpus_dir=str(corpus))

    outcome = tutor.execute_approved_research(
        request.request_id, ctx, landing_dir=tmp_path / "landing",
    )
    completed = outcome["request"]
    if completed.status.value != "completed":
        # Surface the underlying research error for debugging
        pytest.fail(f"research failed: {outcome['tool_result'].error}")
    assert completed.status.value == "completed"
    assert completed.job_id is not None
    assert len(completed.result_source_refs) >= 1
    assert outcome["research"].success


def test_reject_research_request(tutor_env):
    core, store = tutor_env
    tutor = TutorSession(core, store)
    request = tutor.create_research_request(
        "concept:x", "¿Cómo funciona X en profundidad completa?",
    )
    cancelled = tutor.reject_research_request(request.request_id, decided_by="Valen", note="no hace falta")
    assert cancelled.status.value == "cancelled"
    ctx = ToolContext(memory=core.memory)
    with pytest.raises(ValueError, match="only approved"):
        tutor.execute_approved_research(request.request_id, ctx)


def test_research_request_contract_validates(tutor_env):
    core, store = tutor_env
    tutor = TutorSession(core, store)
    request = tutor.create_research_request(
        "concept:asyncio", "¿Cómo funciona el event loop de asyncio en profundidad?",
        allowed_domains=["docs.python.org"],
    )
    approved = tutor.approve_research_request(request.request_id, decided_by="Valen")
    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "validation"))
    from validate_tutor_contract import validate as validate_tutor
    from dataclasses import asdict
    payload = asdict(approved)
    payload["status"] = approved.status.value
    payload["trigger"] = approved.trigger.value
    payload["budget"] = asdict(approved.budget)
    payload["gap_evidence"] = [
        {**asdict(r), "source_type": r.source_type.value} for r in approved.gap_evidence
    ]
    payload["result_source_refs"] = [
        {**asdict(r), "source_type": r.source_type.value} for r in approved.result_source_refs
    ]
    assert validate_tutor("ResearchRequest", payload) == []


def test_lesson_links_topic_cluster_id(tutor_env, tmp_path):
    """Fase 0 gate: episodes link to topic_cluster_id when a cluster exists."""
    from datetime import datetime, timezone
    from ipa.agentic.topic_clusters import TopicCluster, TopicClusterStore
    from ipa.tutor.tutor_contracts import GenerationProvenance
    core, store = tutor_env
    cluster_store = TopicClusterStore(tmp_path / "clusters.db")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    cluster = TopicCluster(
        cluster_id="topic_cluster:asyncio", label="asyncio", description=None,
        member_document_ids=["concept:asyncio"], member_concept_ids=[],
        parent_cluster_id=None, coherence_score=0.9,
        representative_chunk_id="chunk:rep", created_at=now,
        generation=GenerationProvenance(
            generator="t", generated_at=now,
            input_hash="sha256:" + "a" * 64, model_fingerprint="m",
        ),
        field_origins={"label": "generated", "member_document_ids": "source",
                       "coherence_score": "generated"},
    )
    cluster_store.save_cluster(cluster)

    tutor = TutorSession(core, store, provider=FakeProvider(responses=["lección"]))
    core.start_session()
    result = tutor.lesson("concept:asyncio", "explícame", cluster_store=cluster_store)
    assert result["topic_cluster_id"] == "topic_cluster:asyncio"
    episodes = core.memory.get_episodes(core.session_id)
    assert all(e.topic_cluster_id == "topic_cluster:asyncio" for e in episodes)


def test_lesson_without_cluster_store_leaves_null(tutor_env):
    core, store = tutor_env
    tutor = TutorSession(core, store)
    core.start_session()
    result = tutor.lesson("concept:rust", "explícame rust")
    assert result["topic_cluster_id"] is None
    episodes = core.memory.get_episodes(core.session_id)
    assert all(e.topic_cluster_id is None for e in episodes)


# ---------------------------------------------------------------------------
# LearningGoal: el "proyecto" persistido que un roadmap sirve
# ---------------------------------------------------------------------------

def test_goal_store_roundtrip(tutor_env):
    core, store = tutor_env
    tutor = TutorSession(core, store)
    tutor.ensure_goal("goal:rag", title="RAG avanzado", description="quiero dominar rag")
    loaded = store.get_goal("goal:rag")
    assert loaded is not None
    assert loaded.status.value == "proposed"
    assert loaded.title == "RAG avanzado"
    assert loaded.success_criteria
    assert loaded.approval is None
    assert [g.goal_id for g in store.list_goals()] == ["goal:rag"]


def test_ensure_goal_is_idempotent_and_resurrects_cancelled(tutor_env):
    core, store = tutor_env
    tutor = TutorSession(core, store)
    g1 = tutor.ensure_goal("goal:x", title="X")
    g2 = tutor.ensure_goal("goal:x", title="Otro título")
    assert g2.title == "X"  # el goal vivo existente gana
    tutor.cancel_goal("goal:x")
    g3 = tutor.ensure_goal("goal:x", title="X")
    assert g3.status.value == "proposed"
    assert g3.created_at == g1.created_at  # resucitado, no recreado


def test_propose_roadmap_refines_goal_from_llm_block(tutor_env):
    """El JSON de propuesta puede traer un bloque 'goal'; el scaffold lo
    vuelca al LearningGoal persistido con origins 'generated'."""
    core, store = tutor_env
    response = json.dumps({
        "goal": {
            "title": "RAG híbrido",
            "description": "Dominar retrieval denso+sparse con fusión",
            "success_criteria": ["Explicar RRF", "Implementar un retriever híbrido"],
            "constraints": ["Nivel avanzado"],
        },
        "units": json.loads(ROADMAP_LLM_RESPONSE)["units"],
        "assumptions": [], "uncertainties": [],
    })
    tutor = TutorSession(core, store, provider=FakeProvider(responses=[response]))
    tutor.ensure_goal("goal:rag", title="rag")
    tutor.propose_roadmap("goal:rag", CONCEPTS)
    goal = store.get_goal("goal:rag")
    assert goal.title == "RAG híbrido"
    assert goal.success_criteria == ["Explicar RRF", "Implementar un retriever híbrido"]
    assert goal.constraints == ["Nivel avanzado"]
    assert goal.field_origins["title"] == "generated"
    assert goal.generation is not None


def test_propose_roadmap_prompt_requests_goal_block(tutor_env):
    core, store = tutor_env
    provider = FakeProvider(responses=[ROADMAP_LLM_RESPONSE])
    tutor = TutorSession(core, store, provider=provider)
    tutor.propose_roadmap("goal:rag", CONCEPTS)
    prompt = provider.calls[0]["messages"][-1]["content"]
    assert '"goal"' in prompt
    assert "success_criteria" in prompt


def test_goal_gate_mirrors_roadmap_approval(tutor_env):
    """Un solo gate: aprobar+activar el roadmap confirma+activa su goal
    con el mismo decided_by."""
    core, store = tutor_env
    tutor = TutorSession(core, store, provider=FakeProvider(responses=[ROADMAP_LLM_RESPONSE]))
    roadmap = tutor.propose_roadmap("goal:rag", CONCEPTS)
    tutor.approve_roadmap(roadmap.roadmap_id, decided_by="Valen")
    tutor.activate_roadmap(roadmap.roadmap_id)
    tutor.approve_goal(roadmap.goal_id, decided_by="Valen")
    tutor.activate_goal(roadmap.goal_id)
    goal = store.get_goal(roadmap.goal_id)
    assert goal.status.value == "active"
    assert goal.approval.approved and goal.approval.decided_by == "Valen"


def test_confirmed_goal_requires_human_approval(tutor_env):
    """Invariante del contrato: confirmed/active/completed sin approval → error."""
    from ipa.tutor.tutor_contracts import LearningGoal, LearningGoalStatus
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    with pytest.raises(ValueError, match="require human approval"):
        LearningGoal(
            goal_id="goal:x", title="x", description="d",
            status=LearningGoalStatus.ACTIVE,
            success_criteria=["c"], created_at=now, updated_at=now,
            approval=None,
            field_origins={"title": "user", "description": "user",
                           "success_criteria": "system"},
        )


def test_goal_validates_against_schema(tutor_env):
    """El payload persistido pasa el schema autoritativo LearningGoal."""
    from dataclasses import asdict
    from validate_tutor_contract import validate as validate_tutor
    core, store = tutor_env
    tutor = TutorSession(core, store)
    goal = tutor.ensure_goal("goal:rag", title="RAG", description="d")
    payload = asdict(goal)
    payload["status"] = goal.status.value
    assert validate_tutor("LearningGoal", payload) == []


def test_refine_never_rewrites_confirmed_goal(tutor_env):
    """Un goal aprobado es récord humano: el LLM no lo reescribe."""
    core, store = tutor_env
    tutor = TutorSession(core, store)
    tutor.ensure_goal("goal:rag", title="RAG")
    tutor.approve_goal("goal:rag", decided_by="Valen")
    tutor._refine_goal("goal:rag", {"title": "override"})
    assert store.get_goal("goal:rag").title == "RAG"


def test_goal_for_roadmap_backfills_legacy(tutor_env):
    """Roadmaps previos a learning_goals reciben un goal sintetizado al leerse,
    reusando el approval del propio roadmap para el estado active."""
    core, store = tutor_env
    tutor = TutorSession(core, store, provider=FakeProvider(responses=[ROADMAP_LLM_RESPONSE]))
    roadmap = tutor.propose_roadmap("goal:legacy", CONCEPTS)
    tutor.approve_roadmap(roadmap.roadmap_id, decided_by="Valen")
    tutor.activate_roadmap(roadmap.roadmap_id)
    # Simula un roadmap legacy: sin fila en learning_goals.
    store._conn.execute("DELETE FROM learning_goals WHERE goal_id = ?", (roadmap.goal_id,))
    store._conn.commit()
    goal = tutor.goal_for_roadmap(roadmap)
    assert goal.status.value == "active"
    assert goal.approval.decided_by == "Valen"
    assert store.get_goal("goal:legacy") is not None  # persistido para la próxima lectura
