"""Fase 3.4 tests: memory consolidation + user model inference (semi-automatic).

Covers:
  - MemoryConsolidator: proposals from episode groups (never deletes originals)
  - UserModelInference: mastery proposals from accumulated assessment evidence
  - Approval gate: proposals stay pending until a human decides
  - No regression proposals (inference never downgrades mastery)
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ipa.agentic.memory_consolidation import (  # noqa: E402
    ConsolidationStore,
    MemoryConsolidator,
    UserModelInference,
    apply_approved_mastery_inference,
    apply_approved_memory_consolidation,
    approve_proposal,
    reject_proposal,
)
from ipa.tutor.tutor_contracts import (  # noqa: E402
    EvidenceType,
    GenerationProvenance,
    MasteryStatus,
    SourceRef,
    SourceType,
    UserEvidence,
    UserTopicRecord,
)


def _gen():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    return GenerationProvenance(generator="t", generated_at=now,
                                input_hash="sha256:" + "a" * 64, model_fingerprint="m")


def _evidence(evidence_id: str, topic_id: str, obs: str, *, etype=EvidenceType.ASSESSMENT,
              assessment_id: str | None = None) -> UserEvidence:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    return UserEvidence(
        evidence_id=evidence_id, topic_id=topic_id, evidence_type=etype,
        observation=obs, observed_at=now, recorded_at=now,
        source_refs=[SourceRef(source_id=assessment_id or "s1",
                               source_type=SourceType.ASSESSMENT if assessment_id else SourceType.ARTIFACT,
                               content_hash=None)],
        generation=_gen(),
        field_origins={"observation": "generated", "source_refs": "source"},
        assessment_id=assessment_id,
    )


@pytest.fixture()
def store(tmp_path):
    s = ConsolidationStore(tmp_path / "consolidation.db")
    yield s
    s.close()


# ---------------------------------------------------------------------------
# Memory consolidation
# ---------------------------------------------------------------------------

def test_consolidation_proposal_created_from_episodes(store):
    consolidator = MemoryConsolidator(store)
    episodes = [
        {"episode_id": f"agent_episode:{i}", "turn_role": "user" if i % 2 == 0 else "assistant",
         "content": f"episodio {i} sobre asyncio"}
        for i in range(5)
    ]
    proposal = consolidator.propose_consolidation("concept:asyncio", episodes)
    assert proposal is not None
    assert proposal.status == "pending"
    assert proposal.kind == "memory_consolidation"
    assert len(proposal.source_episode_ids) == 5
    assert proposal.proposed_payload["originals_preserved"] is True


def test_consolidation_requires_min_episodes(store):
    consolidator = MemoryConsolidator(store)
    episodes = [{"episode_id": "e1", "turn_role": "user", "content": "x"}]
    assert consolidator.propose_consolidation("concept:x", episodes, min_episodes=3) is None


def test_consolidation_never_auto_applies(store):
    """The invariant: memory_consolidation_requires_human_approval."""
    consolidator = MemoryConsolidator(store)
    episodes = [{"episode_id": f"e{i}", "turn_role": "user", "content": "c"} for i in range(4)]
    proposal = consolidator.propose_consolidation("concept:x", episodes)
    assert proposal.status == "pending"
    # No apply method exists — approval is the only path forward
    assert not hasattr(consolidator, "apply")


def test_approve_and_reject_proposals(store):
    consolidator = MemoryConsolidator(store)
    episodes = [{"episode_id": f"e{i}", "turn_role": "user", "content": "c"} for i in range(3)]
    proposal = consolidator.propose_consolidation("concept:x", episodes)

    approved = approve_proposal(store, proposal.proposal_id, decided_by="Valen", note="ok")
    assert approved.status == "approved"
    assert approved.decided_by == "Valen"

    # A decided proposal cannot be re-decided
    with pytest.raises(ValueError, match="only pending"):
        approve_proposal(store, proposal.proposal_id, decided_by="Valen")


# ---------------------------------------------------------------------------
# User model inference
# ---------------------------------------------------------------------------

def _assessment_evidence(scores: list[float]) -> list[UserEvidence]:
    return [
        _evidence(f"user_evidence:a{i}", "concept:asyncio",
                  f"Assessment assessment:{i}: score={s:.2f}, status=understood.",
                  assessment_id=f"assessment:{i}")
        for i, s in enumerate(scores)
    ]


def test_inference_proposes_advance_on_high_scores(store):
    inference = UserModelInference(store)
    evidence = _assessment_evidence([0.9, 0.85, 0.88])
    proposal = inference.propose_mastery_update("concept:asyncio", evidence, current_record=None)
    assert proposal is not None
    assert proposal.kind == "mastery_inference"
    assert proposal.proposed_payload["proposed_mastery_status"] == "applied"
    assert proposal.status == "pending"


def test_inference_proposes_understood_on_medium_scores(store):
    inference = UserModelInference(store)
    evidence = _assessment_evidence([0.7, 0.65])
    proposal = inference.propose_mastery_update("concept:asyncio", evidence, current_record=None)
    assert proposal is not None
    assert proposal.proposed_payload["proposed_mastery_status"] == "understood"


def test_inference_needs_min_assessments(store):
    inference = UserModelInference(store)
    evidence = _assessment_evidence([0.9])
    assert inference.propose_mastery_update("concept:x", evidence, current_record=None) is None


def test_inference_never_proposes_regression(store):
    """Inference never proposes a mastery downgrade — only Valen can lower."""
    inference = UserModelInference(store)
    evidence = _assessment_evidence([0.5, 0.4])
    current = _make_record("concept:x", MasteryStatus.APPLIED, 0.9)
    assert inference.propose_mastery_update("concept:x", evidence, current_record=current) is None


def _make_record(topic_id, status, score, attempts=3):
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    return UserTopicRecord(
        record_id=f"user_topic_record:{topic_id.replace(':', '_')}",
        topic_id=topic_id,
        mastery_status=status,
        mastery_score=score,
        attempts=attempts,
        last_assessment_id="assessment:old",
        evidence_ids=["user_evidence:old"],
        updated_at=now,
        created_at=now,
        generation=GenerationProvenance(
            generator="t", generated_at=now,
            input_hash="sha256:" + "a" * 64, model_fingerprint="m",
        ),
        field_origins={"mastery_status": "generated", "mastery_score": "generated",
                       "attempts": "system", "last_assessment_id": "system",
                       "evidence_ids": "system"},
    )


def test_inference_no_regression(store):
    """Inference never proposes lowering an already-higher mastery state."""
    inference = UserModelInference(store)
    evidence = _assessment_evidence([0.5, 0.55])  # avg 0.525 → understood
    current = _make_record("concept:asyncio", MasteryStatus.APPLIED, 0.9)
    assert inference.propose_mastery_update("concept:asyncio", evidence, current_record=current) is None


def test_inference_requires_min_assessments(store):
    inference = UserModelInference(store)
    evidence = _assessment_evidence([0.9])  # only 1
    assert inference.propose_mastery_update("concept:x", evidence, current_record=None) is None


def test_apply_approved_memory_consolidation_materializes_without_deletion(store):
    consolidator = MemoryConsolidator(store)
    episodes = [{"episode_id": f"e{i}", "turn_role": "user", "content": "c"} for i in range(3)]
    proposal = consolidator.propose_consolidation("concept:x", episodes)
    approve_proposal(store, proposal.proposal_id, decided_by="Valen")
    applied = apply_approved_memory_consolidation(store, proposal.proposal_id)
    assert applied["originals_preserved"] is True
    row = store._conn.execute("SELECT summary FROM memory_consolidations WHERE proposal_id = ?", (proposal.proposal_id,)).fetchone()
    assert row is not None


def test_apply_pending_proposal_is_forbidden(store):
    consolidator = MemoryConsolidator(store)
    episodes = [{"episode_id": f"e{i}", "turn_role": "user", "content": "c"} for i in range(3)]
    proposal = consolidator.propose_consolidation("concept:x", episodes)
    with pytest.raises(ValueError, match="only approved"):
        apply_approved_memory_consolidation(store, proposal.proposal_id)


def test_pending_proposals_listing(store):
    consolidator = MemoryConsolidator(store)
    inference = UserModelInference(store)
    episodes = [{"episode_id": f"e{i}", "turn_role": "user", "content": "c"} for i in range(3)]
    consolidator.propose_consolidation("concept:a", episodes)
    evidence = _assessment_evidence([0.9, 0.9])
    inference.propose_mastery_update("concept:b", evidence, current_record=None)

    pending = store.list_proposals(status="pending")
    assert len(pending) == 2
    kinds = {p.kind for p in pending}
    assert kinds == {"memory_consolidation", "mastery_inference"}


def test_apply_sessfact_routes_to_user_model(store, tmp_path):
    """Las propuestas sessfact del SessionConsolidator materializan en
    user_facts (active) — es el único camino que llega al system prompt.
    Antes morían en un KeyError o en la tabla memory_consolidations."""
    from ipa.agent.user_model import UserModelStore, render_user_model_context
    from ipa.agentic.memory_consolidation import ConsolidationProposal
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    proposal = ConsolidationProposal(
        proposal_id="consolidation:sessfact:test00001",
        kind="memory_consolidation", topic_id="agent_session:s1",
        summary="Hecho del usuario detectado en sesión: trabaja en fotónica",
        source_episode_ids=["e1", "e2"],
        proposed_payload={"fact": "trabaja en fotónica",
                          "session_id": "agent_session:s1",
                          "origin": "session_consolidation",
                          "originals_preserved": True},
        status="pending", proposed_at=now,
    )
    store.save_proposal(proposal)
    approve_proposal(store, proposal.proposal_id, decided_by="Valen")

    um = UserModelStore(tmp_path / "user_model.db")
    try:
        applied = apply_approved_memory_consolidation(
            store, proposal.proposal_id, user_model_store=um)
        assert applied["fact"] == "trabaja en fotónica"
        active = um.list_facts(status="active")
        assert [f["fact"] for f in active] == ["trabaja en fotónica"]
        assert active[0]["decided_by"] == "Valen"
        # El hecho aprobado entra al contexto del system prompt.
        ctx = render_user_model_context(um)
        assert "trabaja en fotónica" in ctx
        # Re-aplicar no duplica.
        again = apply_approved_memory_consolidation(
            store, proposal.proposal_id, user_model_store=um)
        assert again["deduplicated"] is True
        assert len(um.list_facts(status="active")) == 1
    finally:
        um.close()
