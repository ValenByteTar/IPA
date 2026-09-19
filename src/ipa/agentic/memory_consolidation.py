"""Fase 3.4: consolidación de memoria con aprobación humana + inferencia user model.

Semi-automático (roadmap Fase 3, etapa 8-9):
  - MemoryConsolidator: propone consolidados de episodios antiguos (nunca borra)
  - UserModelInference: propone actualizaciones de mastery desde evidencia acumulada
  - Ambos quedan 'pending' hasta aprobación humana (invariante:
    memory_consolidation_requires_human_approval)
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ipa.tutor.tutor_contracts import (
    EvidenceType,
    MasteryStatus,
    UserEvidence,
    UserTopicRecord,
)

DEFAULT_CONSOLIDATION_STORE = Path("outputs/agent/consolidation.db")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


@dataclass(frozen=True)
class ConsolidationProposal:
    """A proposed consolidation awaiting human approval. Never auto-applied."""
    proposal_id: str
    kind: str  # "memory_consolidation" | "mastery_inference"
    topic_id: str | None
    summary: str
    source_episode_ids: list[str]
    proposed_payload: dict[str, Any]
    status: str  # "pending" | "approved" | "rejected"
    proposed_at: str
    decided_by: str | None = None
    decision_note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ConsolidationStore:
    """SQLite persistence for consolidation proposals (append-only proposals)."""

    def __init__(self, store_path: str | Path = DEFAULT_CONSOLIDATION_STORE) -> None:
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.store_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS consolidation_proposals (
                proposal_id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                topic_id TEXT,
                status TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                proposed_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memory_consolidations (
                proposal_id TEXT PRIMARY KEY,
                topic_id TEXT NOT NULL,
                summary TEXT NOT NULL,
                source_episode_ids_json TEXT NOT NULL,
                applied_at TEXT NOT NULL
            );
        """)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def save_proposal(self, proposal: ConsolidationProposal) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO consolidation_proposals VALUES (?, ?, ?, ?, ?, ?)",
            (proposal.proposal_id, proposal.kind, proposal.topic_id,
             proposal.status, json.dumps(proposal.to_dict(), ensure_ascii=False),
             proposal.proposed_at),
        )
        self._conn.commit()

    def get_proposal(self, proposal_id: str) -> ConsolidationProposal | None:
        row = self._conn.execute(
            "SELECT payload_json FROM consolidation_proposals WHERE proposal_id = ?",
            (proposal_id,),
        ).fetchone()
        return _deserialize_proposal(row[0]) if row else None

    def list_proposals(self, status: str | None = None) -> list[ConsolidationProposal]:
        if status:
            rows = self._conn.execute(
                "SELECT payload_json FROM consolidation_proposals WHERE status = ? ORDER BY proposed_at DESC",
                (status,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT payload_json FROM consolidation_proposals ORDER BY proposed_at DESC"
            ).fetchall()
        return [_deserialize_proposal(payload) for (payload,) in rows]


def _deserialize_proposal(payload: str) -> ConsolidationProposal:
    return ConsolidationProposal(**json.loads(payload))


class MemoryConsolidator:
    """Proposes memory consolidations from old episodes (semi-automatic).

    The proposal NEVER deletes or mutates episodes (append-only invariant);
    it creates a derived summary record that Valen can approve or reject.
    """

    def __init__(self, store: ConsolidationStore) -> None:
        self.store = store

    def propose_consolidation(
        self,
        topic_id: str,
        episodes: list[dict[str, Any]],
        *,
        min_episodes: int = 3,
    ) -> ConsolidationProposal | None:
        """Propose a consolidation when enough episodes share a topic.

        Deterministic scaffold: groups episodes, counts turns, extracts the
        most frequent terms. LLM refinement of the summary is optional and
        happens at apply-time, not proposal-time.
        """
        if len(episodes) < min_episodes:
            return None
        now = _now()
        episode_ids = [e.get("episode_id", "") for e in episodes]
        proposal_id = f"consolidation:{hashlib.sha256((''.join(sorted(episode_ids)) + now).encode()).hexdigest()[:16]}"

        # Deterministic scaffold summary: turn counts + first/last episode
        user_turns = sum(1 for e in episodes if e.get("turn_role") == "user")
        assistant_turns = sum(1 for e in episodes if e.get("turn_role") == "assistant")
        first = episodes[0].get("content", "")[:100]
        last = episodes[-1].get("content", "")[:100]
        summary = (
            f"{len(episodes)} episodios ({user_turns} user, {assistant_turns} assistant) "
            f"sobre {topic_id}. Primer turno: '{first}...'. Último: '{last}...'."
        )
        proposal = ConsolidationProposal(
            proposal_id=proposal_id,
            kind="memory_consolidation",
            topic_id=topic_id,
            summary=summary,
            source_episode_ids=episode_ids,
            proposed_payload={
                "consolidated_summary": summary,
                "episode_count": len(episodes),
                "original_episode_ids": episode_ids,
                "originals_preserved": True,
            },
            status="pending",
            proposed_at=now,
        )
        self.store.save_proposal(proposal)
        return proposal


class UserModelInference:
    """Proposes mastery updates from accumulated evidence (semi-automatic).

    Deterministic scaffold: counts evidence by type and infers a proposed
    mastery status. Valen approves or corrects — never auto-applied.
    """

    def __init__(self, store: ConsolidationStore) -> None:
        self.store = store

    def propose_mastery_update(
        self,
        topic_id: str,
        evidence: list[UserEvidence],
        current_record: UserTopicRecord | None,
        *,
        min_assessments: int = 2,
    ) -> ConsolidationProposal | None:
        """Propose a mastery update when enough assessment evidence accumulates.

        Deterministic rule: if the last N assessments average score >= 0.8 and
        status is consistently 'understood'/'applied', propose advancing.
        """
        assessments = [e for e in evidence if e.evidence_type == EvidenceType.ASSESSMENT]
        if len(assessments) < min_assessments:
            return None
        now = _now()
        proposal_id = f"inference:{hashlib.sha256((topic_id + now).encode()).hexdigest()[:16]}"

        # Deterministic scaffold: extract scores from observations
        import re
        scores = []
        for e in assessments:
            match = re.search(r"score=([\d.]+)", e.observation)
            if match:
                scores.append(float(match.group(1)))
        avg_score = sum(scores) / len(scores) if scores else None

        if avg_score is None:
            return None
        if avg_score >= 0.8:
            proposed_status = MasteryStatus.APPLIED.value
        elif avg_score >= 0.6:
            proposed_status = MasteryStatus.UNDERSTOOD.value
        else:
            proposed_status = MasteryStatus.NEEDS_REVIEW.value

        # Don't propose a downgrade of an already-higher state
        if current_record is not None:
            order = [
                MasteryStatus.UNKNOWN, MasteryStatus.EXPOSED,
                MasteryStatus.NEEDS_REVIEW, MasteryStatus.UNDERSTOOD,
                MasteryStatus.APPLIED,
            ]
            current_idx = order.index(current_record.mastery_status) if current_record.mastery_status in order else 0
            proposed_idx = order.index(MasteryStatus(proposed_status))
            if proposed_idx <= current_idx:
                return None  # no regression proposals from inference

        proposal = ConsolidationProposal(
            proposal_id=proposal_id,
            kind="mastery_inference",
            topic_id=topic_id,
            summary=(
                f"{len(assessments)} assessments, score promedio {avg_score:.2f} → "
                f"proponer mastery '{proposed_status}'"
            ),
            source_episode_ids=[e.evidence_id for e in assessments],
            proposed_payload={
                "proposed_mastery_status": proposed_status,
                "avg_score": round(avg_score, 3),
                "assessment_count": len(assessments),
                "evidence_ids": [e.evidence_id for e in assessments],
            },
            status="pending",
            proposed_at=now,
        )
        self.store.save_proposal(proposal)
        return proposal


def approve_proposal(store: ConsolidationStore, proposal_id: str, *, decided_by: str, note: str | None = None) -> ConsolidationProposal:
    proposal = store.get_proposal(proposal_id)
    if proposal is None:
        raise ValueError(f"unknown proposal: {proposal_id}")
    if proposal.status != "pending":
        raise ValueError(f"only pending proposals can be approved (status: {proposal.status})")
    approved = ConsolidationProposal(
        proposal_id=proposal.proposal_id, kind=proposal.kind, topic_id=proposal.topic_id,
        summary=proposal.summary, source_episode_ids=proposal.source_episode_ids,
        proposed_payload=proposal.proposed_payload, status="approved",
        proposed_at=proposal.proposed_at, decided_by=decided_by, decision_note=note,
    )
    store.save_proposal(approved)
    return approved


def reject_proposal(store: ConsolidationStore, proposal_id: str, *, decided_by: str, note: str | None = None) -> ConsolidationProposal:
    proposal = store.get_proposal(proposal_id)
    if proposal is None:
        raise ValueError(f"unknown proposal: {proposal_id}")
    rejected = ConsolidationProposal(
        proposal_id=proposal.proposal_id, kind=proposal.kind, topic_id=proposal.topic_id,
        summary=proposal.summary, source_episode_ids=proposal.source_episode_ids,
        proposed_payload=proposal.proposed_payload, status="rejected",
        proposed_at=proposal.proposed_at, decided_by=decided_by, decision_note=note,
    )
    store.save_proposal(rejected)
    return rejected


def apply_approved_memory_consolidation(
    store: ConsolidationStore, proposal_id: str,
    *, user_model_store: Any | None = None,
) -> dict[str, Any]:
    """Materialize an approved consolidation without deleting source episodes.

    Two payload shapes live under kind='memory_consolidation':
      - sessfact `{fact, session_id, origin}` (SessionConsolidator) → user_facts
        in the user model with status 'active' — the only path that reaches
        the system prompt (render_user_model_context) and the memory index.
      - legacy `consolidated_summary` (MemoryConsolidator) → memory_consolidations.
    """
    proposal = store.get_proposal(proposal_id)
    if proposal is None or proposal.status != "approved":
        raise ValueError("only approved consolidation proposals can be applied")
    if proposal.kind != "memory_consolidation":
        raise ValueError("proposal is not a memory consolidation")
    payload = proposal.proposed_payload
    if "fact" in payload:
        # sessfact: el gate ya pasó (proposal approved); materializar en el
        # user model con decided_by para provenance.
        from ipa.agent.user_model import UserModelStore
        um = user_model_store if user_model_store is not None else UserModelStore()
        try:
            fact = str(payload["fact"]).strip()
            existing = {f["fact"] for f in um.list_facts(status="active", limit=200)}
            if fact in existing:
                return {"proposal_id": proposal.proposal_id, "topic_id": proposal.topic_id,
                        "fact": fact, "deduplicated": True}
            fact_id = um.add_fact(
                fact, source=str(payload.get("origin", "session_consolidation")))
            um.decide_fact(
                fact_id, approved=True,
                decided_by=proposal.decided_by or "human")
        finally:
            if user_model_store is None:
                um.close()
        return {"proposal_id": proposal.proposal_id, "topic_id": proposal.topic_id,
                "fact": fact, "fact_id": fact_id, "originals_preserved": True}
    store._conn.execute(
        "INSERT OR REPLACE INTO memory_consolidations VALUES (?, ?, ?, ?, ?)",
        (proposal.proposal_id, proposal.topic_id, payload["consolidated_summary"],
         json.dumps(proposal.source_episode_ids), _now()),
    )
    store._conn.commit()
    return {"proposal_id": proposal.proposal_id, "topic_id": proposal.topic_id,
            "summary": payload["consolidated_summary"],
            "source_episode_ids": list(proposal.source_episode_ids),
            "originals_preserved": True}


def apply_approved_mastery_inference(store: ConsolidationStore, proposal_id: str, tutor_store: Any) -> Any:
    """Materialize approved mastery inference into the TutorStore.

    The update remains evidence-linked and uses the existing record when
    present; this function never applies pending/rejected proposals.
    """
    proposal = store.get_proposal(proposal_id)
    if proposal is None or proposal.status != "approved":
        raise ValueError("only approved mastery proposals can be applied")
    if proposal.kind != "mastery_inference":
        raise ValueError("proposal is not a mastery inference")
    current = tutor_store.get_topic_record(proposal.topic_id)
    if current is None:
        raise ValueError("mastery inference requires an existing topic record")
    payload = proposal.proposed_payload
    now = _now()
    from ipa.tutor.tutor_contracts import UserTopicRecord
    updated = UserTopicRecord(
        record_id=current.record_id, topic_id=current.topic_id,
        mastery_status=MasteryStatus(payload["proposed_mastery_status"]),
        mastery_score=float(payload["avg_score"]), attempts=current.attempts,
        last_assessment_id=current.last_assessment_id,
        evidence_ids=list(dict.fromkeys(current.evidence_ids + proposal.source_episode_ids)),
        updated_at=now, created_at=current.created_at, generation=current.generation,
        field_origins=current.field_origins, prerequisite_ids=current.prerequisite_ids,
        notes=f"Human-approved inference {proposal.proposal_id}",
    )
    tutor_store.upsert_topic_record(updated)
    return updated


__all__ = [
    "ConsolidationProposal",
    "ConsolidationStore",
    "MemoryConsolidator",
    "UserModelInference",
    "approve_proposal",
    "reject_proposal",
]
