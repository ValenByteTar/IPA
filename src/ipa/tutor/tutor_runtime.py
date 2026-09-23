"""Tutor runtime: the first agent role with its own state scope (Fase 2).

A role is not just a prompt — it is policy + persona + state scope (DEC-002,
roadmap Fase 2):

  - persona:  pedagogical extension of the base identity (system_prompt(role="tutor"))
  - policy:   diagnose before explain; evidence-based assessment; abstention
              when uncertain; scaffold deterministically, LLM only classifies
  - state:    TutorStore owns user_topic_records + user_evidence (append-only
              evidence log). The unified user model IS the mastery store.

Loop: diagnóstico → roadmap (LLM + human approval) → lección → assessment →
mastery update. If the corpus cannot support a lesson, the Tutor triggers a
ResearchRequest (Fase 1 agentic research).
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from ipa.agent.agent_core import AgentCore
from ipa.tutor.tutor_contracts import (
    AssessmentType,
    EvidenceType,
    GenerationProvenance,
    HumanApproval,
    HumanApprovalDecision,
    LearningGoal,
    LearningGoalStatus,
    MasteryStatus,
    RecommendedAction,
    Roadmap,
    RoadmapStatus,
    RoadmapUnit,
    SourceRef,
    SourceType,
    UserEvidence,
    UserTopicRecord,
    answer_hash,
)

DEFAULT_TUTOR_STORE = Path("outputs/agent/tutor.db")

# Abstention threshold: below this confidence the Tutor refuses to score and
# escalates to human review (BM-006: abstention 0.81-0.89 measured).
ABSTENTION_THRESHOLD = 0.5

# Pedagogical policy — the Tutor's operating rules (not just a persona).
TUTOR_POLICY = (
    "Política pedagógica:\n"
    "- Diagnosticá antes de explicar: revisá el estado de mastery del tema.\n"
    "- Explicá con andamiaje: partí de lo que el alumno ya sabe.\n"
    "- Explicaciones ricas: desarrollá la idea con definición, un ejemplo "
    "concreto y la conexión con lo que el alumno ya sabe (2-4 párrafos). "
    "Conciso en trámites y confirmaciones, generoso en la explicación.\n"
    "- Evaluá con evidencia: toda afirmación sobre el aprendizaje cita un assessment.\n"
    "- Abstenete si no tenés evidencia suficiente: mejor 'no sé' que inventar.\n"
    "- Si el corpus no alcanza para enseñar el tema, proponé investigar antes de improvisar."
)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _extract_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from LLM output (robust to prose/fences)."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in LLM output")
    # raw_decode parses the first complete object and ignores trailing
    # prose — the greedy-regex alternative breaks when the model appends
    # commentary after the JSON ("Extra data" error).
    try:
        obj, _end = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        match = _JSON_RE.search(text)
        if not match:
            raise ValueError("no JSON object in LLM output")
        obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise ValueError("LLM output is not a JSON object")
    return obj


@dataclass(frozen=True)
class DiagnosisResult:
    """Deterministic scaffold reading of learner state (no LLM needed)."""
    topic_id: str
    mastery_status: MasteryStatus
    mastery_score: float | None
    attempts: int
    evidence_count: int
    summary: str
    next_action: RecommendedAction
    source: str  # "store" | "new_topic"


class TutorStore:
    """SQLite store for pedagogical state: user_topic_records + user_evidence.

    The evidence log is append-only; topic records are upserted with mastery
    tracing back to evidence (DEC-002 unified user model).
    """

    def __init__(self, store_path: str | Path = DEFAULT_TUTOR_STORE) -> None:
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.store_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS user_topic_records (
                record_id TEXT PRIMARY KEY,
                topic_id TEXT NOT NULL UNIQUE,
                payload_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS user_evidence (
                evidence_id TEXT PRIMARY KEY,
                topic_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                recorded_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS roadmaps (
                roadmap_id TEXT PRIMARY KEY,
                goal_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                status TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS research_requests (
                request_id TEXT PRIMARY KEY,
                concept_id TEXT NOT NULL,
                status TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS unit_progress (
                roadmap_id TEXT NOT NULL,
                unit_order INTEGER NOT NULL,
                status TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (roadmap_id, unit_order)
            );
            CREATE TABLE IF NOT EXISTS unit_summaries (
                roadmap_id TEXT NOT NULL,
                unit_order INTEGER NOT NULL,
                summary TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (roadmap_id, unit_order)
            );
            CREATE TABLE IF NOT EXISTS tutor_focus (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                roadmap_id TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS session_roadmap (
                session_id TEXT PRIMARY KEY,
                roadmap_id TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS learning_goals (
                goal_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                title TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
        """)
        # Migración aditiva: flag operativo de archivado. No es un estado del
        # contrato (RoadmapStatus no cambia) — solo saca el roadmap de las
        # listas del dashboard y del recovery del driver.
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(roadmaps)")}
        if "archived" not in cols:
            self._conn.execute(
                "ALTER TABLE roadmaps ADD COLUMN archived INTEGER NOT NULL DEFAULT 0"
            )
        self._conn.commit()
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # -- topic records -----------------------------------------------------

    def get_topic_record(self, topic_id: str) -> UserTopicRecord | None:
        row = self._conn.execute(
            "SELECT payload_json FROM user_topic_records WHERE topic_id = ?", (topic_id,)
        ).fetchone()
        if row is None:
            return None
        data = json.loads(row[0])
        data["mastery_status"] = MasteryStatus(data["mastery_status"])
        return UserTopicRecord(**data)

    def upsert_topic_record(self, record: UserTopicRecord) -> None:
        payload = asdict(record)
        payload["mastery_status"] = record.mastery_status.value
        self._conn.execute(
            "INSERT OR REPLACE INTO user_topic_records VALUES (?, ?, ?, ?)",
            (record.record_id, record.topic_id,
             json.dumps(payload, ensure_ascii=False), record.updated_at),
        )
        self._conn.commit()

    def list_topic_records(self) -> list[UserTopicRecord]:
        rows = self._conn.execute(
            "SELECT payload_json FROM user_topic_records ORDER BY updated_at DESC"
        ).fetchall()
        records = []
        for (payload,) in rows:
            data = json.loads(payload)
            data["mastery_status"] = MasteryStatus(data["mastery_status"])
            records.append(UserTopicRecord(**data))
        return records

    # -- roadmaps (versioned, approval-gated) --------------------------------

    def save_roadmap(self, roadmap: Any) -> None:
        """Insert or update a roadmap record. Status transitions are enforced
        by TutorSession; the store is a dumb persistence layer."""
        from ipa.tutor.tutor_contracts import Roadmap
        if not isinstance(roadmap, Roadmap):
            raise TypeError("save_roadmap expects a Roadmap contract instance")
        payload = asdict(roadmap)
        payload["status"] = roadmap.status.value
        for unit in payload["units"]:
            unit["assessment_types"] = [at.value if hasattr(at, "value") else str(at) for at in unit["assessment_types"]]
        self._conn.execute(
            "INSERT OR REPLACE INTO roadmaps "
            "(roadmap_id, goal_id, version, status, payload_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (roadmap.roadmap_id, roadmap.goal_id, roadmap.version,
             roadmap.status.value, json.dumps(payload, ensure_ascii=False),
             roadmap.created_at),
        )
        self._conn.commit()

    def _deserialize_roadmap(self, payload: str) -> Any:
        """JSON → Roadmap contract (units, enums, approval reconstructed)."""
        from ipa.tutor.tutor_contracts import (
            AssessmentType, HumanApproval, HumanApprovalDecision,
            Roadmap, RoadmapStatus, RoadmapUnit, SourceRef,
        )
        data = json.loads(payload)
        data["status"] = RoadmapStatus(data["status"])
        units = []
        for unit in data["units"]:
            unit["assessment_types"] = [AssessmentType(at) for at in unit["assessment_types"]]
            span = unit.get("source_refs")
            refs = []
            for ref in (span or []):
                refs.append(SourceRef(
                    source_id=ref["source_id"],
                    source_type=SourceType(ref["source_type"]),
                    content_hash=ref.get("content_hash"),
                ))
            unit["source_refs"] = refs
            data_units = RoadmapUnit(**unit)
            # RoadmapUnit is frozen dataclass; rebuild list with typed units
            units.append(data_units)
        data["units"] = units
        approval = data.get("approval")
        if approval:
            approval["decision"] = HumanApprovalDecision(approval["decision"])
            data["approval"] = HumanApproval(**approval)
        return Roadmap(**data)

    def get_roadmap(self, roadmap_id: str) -> Any | None:
        row = self._conn.execute(
            "SELECT payload_json FROM roadmaps WHERE roadmap_id = ?", (roadmap_id,)
        ).fetchone()
        if row is None:
            return None
        return self._deserialize_roadmap(row[0])

    def set_roadmap_archived(self, roadmap_id: str, archived: bool = True) -> None:
        """Archive/unarchive a roadmap — operational flag, no contract status."""
        self._conn.execute(
            "UPDATE roadmaps SET archived = ? WHERE roadmap_id = ?",
            (1 if archived else 0, roadmap_id),
        )
        self._conn.commit()

    def list_roadmaps(self, goal_id: str | None = None, *, include_archived: bool = False) -> list[Any]:
        if goal_id:
            rows = self._conn.execute(
                "SELECT payload_json FROM roadmaps WHERE goal_id = ?"
                + ("" if include_archived else " AND archived = 0")
                + " ORDER BY version DESC",
                (goal_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT payload_json FROM roadmaps"
                + ("" if include_archived else " WHERE archived = 0")
                + " ORDER BY created_at DESC"
            ).fetchall()
        return [self._deserialize_roadmap(payload) for (payload,) in rows]

    # -- learning goals (the project a roadmap serves) ----------------------

    def save_goal(self, goal: Any) -> None:
        """Insert or update a learning goal record. The store is a dumb
        persistence layer — status/approval invariants live in the contract."""
        if not isinstance(goal, LearningGoal):
            raise TypeError("save_goal expects a LearningGoal contract instance")
        payload = asdict(goal)
        payload["status"] = goal.status.value
        if payload.get("approval"):
            payload["approval"]["decision"] = goal.approval.decision.value
        self._conn.execute(
            "INSERT OR REPLACE INTO learning_goals "
            "(goal_id, status, title, payload_json, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (goal.goal_id, goal.status.value, goal.title,
             json.dumps(payload, ensure_ascii=False),
             goal.created_at, goal.updated_at),
        )
        self._conn.commit()

    @staticmethod
    def _deserialize_goal(payload: str) -> LearningGoal:
        """JSON → LearningGoal contract (status, approval, provenance)."""
        data = json.loads(payload)
        data["status"] = LearningGoalStatus(data["status"])
        approval = data.get("approval")
        if approval:
            approval["decision"] = HumanApprovalDecision(approval["decision"])
            data["approval"] = HumanApproval(**approval)
        generation = data.get("generation")
        if generation:
            data["generation"] = GenerationProvenance(**generation)
        return LearningGoal(**data)

    def get_goal(self, goal_id: str) -> LearningGoal | None:
        row = self._conn.execute(
            "SELECT payload_json FROM learning_goals WHERE goal_id = ?", (goal_id,)
        ).fetchone()
        return self._deserialize_goal(row[0]) if row else None

    def list_goals(self) -> list[LearningGoal]:
        rows = self._conn.execute(
            "SELECT payload_json FROM learning_goals ORDER BY updated_at DESC"
        ).fetchall()
        return [self._deserialize_goal(payload) for (payload,) in rows]

    # -- unit progress (operational, additive — the Roadmap stays immutable) --

    def set_unit_status(self, roadmap_id: str, unit_order: int, status: str) -> None:
        """Track per-unit progress: 'pending' | 'current' | 'done'."""
        self._conn.execute(
            "INSERT INTO unit_progress (roadmap_id, unit_order, status, updated_at) "
            "VALUES (?, ?, ?, ?) ON CONFLICT (roadmap_id, unit_order) "
            "DO UPDATE SET status = excluded.status, updated_at = excluded.updated_at",
            (roadmap_id, unit_order, status, _now()),
        )
        self._conn.commit()

    def unit_statuses(self, roadmap_id: str) -> dict[int, str]:
        rows = self._conn.execute(
            "SELECT unit_order, status FROM unit_progress WHERE roadmap_id = ?",
            (roadmap_id,),
        ).fetchall()
        return {int(order): status for order, status in rows}

    def save_unit_summary(self, roadmap_id: str, unit_order: int, summary: str) -> None:
        """Per-unit lesson summary (pedagogical granularity for memory recall)."""
        self._conn.execute(
            "INSERT INTO unit_summaries (roadmap_id, unit_order, summary, created_at) "
            "VALUES (?, ?, ?, ?) ON CONFLICT (roadmap_id, unit_order) "
            "DO UPDATE SET summary = excluded.summary, created_at = excluded.created_at",
            (roadmap_id, unit_order, summary[:2000], _now()),
        )
        self._conn.commit()

    def list_unit_summaries(self, roadmap_id: str | None = None) -> list[dict[str, Any]]:
        if roadmap_id:
            rows = self._conn.execute(
                "SELECT roadmap_id, unit_order, summary FROM unit_summaries WHERE roadmap_id = ?",
                (roadmap_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT roadmap_id, unit_order, summary FROM unit_summaries"
            ).fetchall()
        return [
            {"roadmap_id": r, "unit_order": int(o), "summary": s}
            for r, o, s in rows
        ]

    # -- foco de roadmap (cross-sesión, una sola fila) ------------------------

    def set_focus(self, roadmap_id: str) -> None:
        """Foco global del usuario: el roadmap sobre el que estamos trabajando.
        Cualquier sesión (nueva o existente) lo adopta al continuar."""
        self._conn.execute(
            "INSERT INTO tutor_focus (id, roadmap_id, updated_at) VALUES (1, ?, ?) "
            "ON CONFLICT (id) DO UPDATE SET roadmap_id = excluded.roadmap_id, "
            "updated_at = excluded.updated_at",
            (roadmap_id, _now()),
        )
        self._conn.commit()

    def get_focus(self) -> str | None:
        row = self._conn.execute(
            "SELECT roadmap_id FROM tutor_focus WHERE id = 1").fetchone()
        return row[0] if row else None

    def clear_focus(self, roadmap_id: str) -> None:
        """Solamente si el foco apunta a ese roadmap (p.ej. al rechazarlo)."""
        self._conn.execute(
            "DELETE FROM tutor_focus WHERE id = 1 AND roadmap_id = ?",
            (roadmap_id,),
        )
        self._conn.commit()

    def set_session_roadmap(self, session_id: str, roadmap_id: str) -> None:
        """Asociación sesión→roadmap (para el tag de los resúmenes de sesión)."""
        self._conn.execute(
            "INSERT INTO session_roadmap (session_id, roadmap_id, updated_at) "
            "VALUES (?, ?, ?) ON CONFLICT (session_id) "
            "DO UPDATE SET roadmap_id = excluded.roadmap_id, updated_at = excluded.updated_at",
            (session_id, roadmap_id, _now()),
        )
        self._conn.commit()

    def get_session_roadmap(self, session_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT roadmap_id FROM session_roadmap WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return row[0] if row else None

    def clear_session_roadmap(self, session_id: str) -> None:
        """Desadopta el roadmap de la sesión (unpin desde el chip del chat)."""
        self._conn.execute(
            "DELETE FROM session_roadmap WHERE session_id = ?", (session_id,))
        self._conn.commit()

    # -- research requests (approval-gated) ---------------------------------

    def save_research_request(self, request: Any) -> None:
        from ipa.tutor.tutor_contracts import ResearchRequest
        if not isinstance(request, ResearchRequest):
            raise TypeError("save_research_request expects a ResearchRequest contract instance")
        payload = asdict(request)
        payload["status"] = request.status.value
        payload["trigger"] = request.trigger.value
        payload["budget"] = asdict(request.budget)
        payload["gap_evidence"] = [asdict(ref) for ref in request.gap_evidence]
        payload["result_source_refs"] = [asdict(ref) for ref in request.result_source_refs]
        self._conn.execute(
            "INSERT OR REPLACE INTO research_requests VALUES (?, ?, ?, ?, ?)",
            (request.request_id, request.concept_id, request.status.value,
             json.dumps(payload, ensure_ascii=False), request.updated_at),
        )
        self._conn.commit()

    def get_research_request(self, request_id: str) -> Any | None:
        from ipa.tutor.tutor_contracts import (
            HumanApproval, HumanApprovalDecision, ResearchBudget, ResearchRequest,
            ResearchStatus, ResearchTrigger, SourceRef, SourceType,
        )
        row = self._conn.execute(
            "SELECT payload_json FROM research_requests WHERE request_id = ?", (request_id,)
        ).fetchone()
        if row is None:
            return None
        data = json.loads(row[0])
        data["status"] = ResearchStatus(data["status"])
        data["trigger"] = ResearchTrigger(data["trigger"])
        data["budget"] = ResearchBudget(**data["budget"])
        data["gap_evidence"] = [
            SourceRef(source_id=r["source_id"], source_type=SourceType(r["source_type"]),
                      content_hash=r.get("content_hash"))
            for r in data["gap_evidence"]
        ]
        data["result_source_refs"] = [
            SourceRef(source_id=r["source_id"], source_type=SourceType(r["source_type"]),
                      content_hash=r.get("content_hash"))
            for r in data.get("result_source_refs", [])
        ]
        approval = data.get("approval")
        if approval:
            approval["decision"] = HumanApprovalDecision(approval["decision"])
            data["approval"] = HumanApproval(**approval)
        return ResearchRequest(**data)

    def list_research_requests(self, concept_id: str | None = None) -> list[Any]:
        if concept_id:
            rows = self._conn.execute(
                "SELECT payload_json FROM research_requests WHERE concept_id = ? ORDER BY created_at DESC",
                (concept_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT payload_json FROM research_requests ORDER BY created_at DESC"
            ).fetchall()
        return [self._deserialize_research_request(payload) for (payload,) in rows]

    def _deserialize_research_request(self, payload: str) -> Any:
        from ipa.tutor.tutor_contracts import (
            HumanApproval, HumanApprovalDecision, ResearchBudget, ResearchRequest,
            ResearchStatus, ResearchTrigger, SourceRef, SourceType,
        )
        data = json.loads(payload)
        data["status"] = ResearchStatus(data["status"])
        data["trigger"] = ResearchTrigger(data["trigger"])
        data["gap_evidence"] = [
            SourceRef(source_id=r["source_id"], source_type=SourceType(r["source_type"]),
                      content_hash=r.get("content_hash"))
            for r in data["gap_evidence"]
        ]
        data["result_source_refs"] = [
            SourceRef(source_id=r["source_id"], source_type=SourceType(r["source_type"]),
                      content_hash=r.get("content_hash"))
            for r in data.get("result_source_refs", [])
        ]
        data["budget"] = ResearchBudget(**data["budget"])
        approval = data.get("approval")
        if approval:
            approval["decision"] = HumanApprovalDecision(approval["decision"])
            data["approval"] = HumanApproval(**approval)
        return ResearchRequest(**data)

    # -- evidence (append-only) --------------------------------------------

    def add_evidence(self, evidence: UserEvidence) -> None:
        payload = asdict(evidence)
        payload["evidence_type"] = evidence.evidence_type.value
        # Append-only: plain INSERT — a duplicate evidence_id is a bug.
        self._conn.execute(
            "INSERT INTO user_evidence VALUES (?, ?, ?, ?)",
            (evidence.evidence_id, evidence.topic_id,
             json.dumps(payload, ensure_ascii=False), evidence.recorded_at),
        )
        self._conn.commit()

    def list_evidence(self, topic_id: str | None = None) -> list[UserEvidence]:
        if topic_id:
            rows = self._conn.execute(
                "SELECT payload_json FROM user_evidence WHERE topic_id = ? ORDER BY recorded_at",
                (topic_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT payload_json FROM user_evidence ORDER BY recorded_at"
            ).fetchall()
        evidence = []
        for (payload,) in rows:
            data = json.loads(payload)
            data["evidence_type"] = EvidenceType(data["evidence_type"])
            evidence.append(UserEvidence(**data))
        return evidence


class TutorSession:
    """The Tutor role: an AgentCore session with pedagogical policy + state."""

    def __init__(
        self,
        core: AgentCore,
        store: TutorStore,
        *,
        provider: Any | None = None,
        max_assessment_tokens: int = 400,
    ) -> None:
        if core.role != "tutor":
            raise ValueError("TutorSession requires an AgentCore with role='tutor'")
        self.core = core
        self.store = store
        self.provider = provider
        self.max_assessment_tokens = max_assessment_tokens
        self.last_fallback_reason: str | None = None

    # ------------------------------------------------------------------
    # Policy: mastery-aware lesson payload
    # ------------------------------------------------------------------

    def _mastery_context(self, topic_id: str) -> str:
        record = self.store.get_topic_record(topic_id)
        if record is None:
            return (
                f"Tema: {topic_id}. Estado del alumno: desconocido (sin evidencia). "
                "Comenzá con un diagnóstico antes de explicar."
            )
        evidence_count = len(self.store.list_evidence(topic_id))
        return (
            f"Tema: {topic_id}. Estado del alumno: {record.mastery_status.value} "
            f"(score {record.mastery_score if record.mastery_score is not None else 'n/a'}, "
            f"{record.attempts} intentos, {evidence_count} evidencias). "
            f"Último assessment: {record.last_assessment_id or 'ninguno'}."
        )

    def _diagnosis_summary(self, topic_id: str) -> str:
        """User-facing diagnosis summary — factual state only.

        _mastery_context() is LESSON POLICY (an instruction to the lesson LLM);
        it must not leak into student-facing text. No trailing period either:
        the driver composes f"...{summary}. " → double period.
        """
        record = self.store.get_topic_record(topic_id)
        if record is None:
            return f"Tema: {topic_id}. Estado del alumno: desconocido (sin evidencia)"
        evidence_count = len(self.store.list_evidence(topic_id))
        return (
            f"Tema: {topic_id}. Estado del alumno: {record.mastery_status.value} "
            f"(score {record.mastery_score if record.mastery_score is not None else 'n/a'}, "
            f"{record.attempts} intentos, {evidence_count} evidencias)"
        )

    def build_lesson_messages(self, topic_id: str, user_message: str, *, history_limit: int = 8,
                              progress_note: str | None = None) -> list[dict[str, str]]:
        """Identity + pedagogical policy + mastery context + bounded history."""
        messages = self.core.build_messages(user_message, history_limit=history_limit)
        system = (
            f"{self.core.identity.system_prompt(role='tutor')}\n\n"
            f"{TUTOR_POLICY}\n\n{self._mastery_context(topic_id)}"
        )
        if progress_note:
            system += f"\n\n{progress_note}"
        messages[0] = {"role": "system", "content": system}
        return messages

    def lesson(self, topic_id: str, user_message: str, *, responder: Any | None = None,
               cluster_store: Any | None = None,
               progress_note: str | None = None) -> dict[str, Any]:
        """A lesson turn: records into the agent session with tutor policy.

        When a ``cluster_store`` (Fase 3) is provided, episodes are linked to
        the topic cluster containing the lesson's documents (Fase 0 gate:
        topic_cluster_id wiring). Missing linkage is not an error — clusters
        are a derived index and may not cover every topic.
        """
        if responder is None and self.provider is not None:
            from ipa.agent.provider_wiring import build_responder
            responder = build_responder(self.provider)
        topic_cluster_id = None
        if cluster_store is not None:
            cluster = cluster_store.find_by_document(topic_id)
            if cluster is not None:
                topic_cluster_id = cluster.cluster_id
        # Submit through the core (records episodes) but with the policy payload
        session_id = self.core.ensure_session()
        messages = self.build_lesson_messages(
            topic_id, user_message, history_limit=8, progress_note=progress_note)
        self.core.memory.record_episode(
            session_id, turn_role="user", content=user_message,
            identity_hash=self.core.identity.identity_hash,
            topic_cluster_id=topic_cluster_id,
        )
        if responder is not None:
            reply = responder(messages)
        else:
            reply = "[sin provider configurado] Turno del tutor registrado."
        assistant = self.core.memory.record_episode(
            session_id, turn_role="assistant", content=reply,
            identity_hash=self.core.identity.identity_hash,
            topic_cluster_id=topic_cluster_id,
        )
        return {"session_id": session_id, "reply": reply,
                "assistant_episode_id": assistant.episode_id, "messages": messages,
                "topic_cluster_id": topic_cluster_id}

    # ------------------------------------------------------------------
    # Diagnóstico (deterministic scaffold)
    # ------------------------------------------------------------------

    def diagnose(self, topic_id: str) -> DiagnosisResult:
        """Read learner state from the store — deterministic, no LLM.

        The scaffold decides WHAT to ask; the LLM only classifies answers
        (BM-006: structured JSON classification validity 0.90).
        """
        record = self.store.get_topic_record(topic_id)
        if record is None:
            return DiagnosisResult(
                topic_id=topic_id, mastery_status=MasteryStatus.UNKNOWN,
                mastery_score=None, attempts=0, evidence_count=0,
                summary=self._diagnosis_summary(topic_id),
                next_action=RecommendedAction.HUMAN_REVIEW,
                source="new_topic",
            )
        if record.mastery_status in (MasteryStatus.UNDERSTOOD, MasteryStatus.APPLIED):
            next_action = RecommendedAction.ADVANCE
        elif record.mastery_status == MasteryStatus.MISCONCEPTION:
            next_action = RecommendedAction.CORRECT_MISCONCEPTION
        else:
            next_action = RecommendedAction.GUIDED_RETRY
        return DiagnosisResult(
            topic_id=topic_id,
            mastery_status=record.mastery_status,
            mastery_score=record.mastery_score,
            attempts=record.attempts,
            evidence_count=len(self.store.list_evidence(topic_id)),
            summary=self._diagnosis_summary(topic_id),
            next_action=next_action,
            source="store",
        )

    # ------------------------------------------------------------------
    # Roadmap pedagógico (LLM propone, humano aprueba)
    # ------------------------------------------------------------------

    def propose_roadmap(
        self,
        goal_id: str,
        concepts: list[dict[str, Any]],
        *,
        version: int = 1,
        previous_roadmap_id: str | None = None,
        change_reason: str | None = None,
        feedback: str | None = None,
        n_units: int | None = None,
    ) -> Roadmap:
        """LLM proposes a 3-7 unit learning roadmap; status stays 'proposed'.

        The proposal is contract-shaped (Roadmap schema) but INERT until a
        human approves it: status='proposed' cannot start lessons. The LLM
        proposes; the scaffold validates and records; the human decides
        (invariant: active_roadmaps_require_human_approval).

        Args:
            goal_id: learning goal this roadmap serves.
            concepts: available concepts, each {concept_id, title, definition,
                      difficulty?, prerequisite_ids?} — the LLM picks and orders
                      3-7 of them.
            version: 1 for a new roadmap; >1 requires previous_roadmap_id +
                     change_reason (supersedes, never mutates).
            n_units: explicit unit count the learner asked for ("roadmap de
                     6 fases"); clamped to the contract range 3-7.
        """
        if self.provider is None:
            raise ValueError("roadmap proposal requires a provider (LLM proposes)")
        # Dedup defensivo: el retrieval puede repetir concept_id entre chunks.
        _seen: set[str] = set()
        concepts = [
            c for c in concepts
            if c["concept_id"] not in _seen and not _seen.add(c["concept_id"])
        ]
        if len(concepts) < 3:
            raise ValueError("roadmap proposal requires at least 3 available concepts")

        concept_lines = "\n".join(
            f"- {c['concept_id']}: {c.get('title', '')} — {c.get('definition', '')[:150]}"
            for c in concepts
        )
        unit_clause = (
            f"Diseñá un roadmap de aprendizaje de {max(3, min(7, n_units))} unidades "
            "(pedido explícito del alumno, ajustado al rango válido 3-7). "
            if n_units else
            "Diseñá un roadmap de aprendizaje de 3 a 7 unidades para este alumno. "
        )
        prompt = (
            f"{self._mastery_context(concepts[0]['concept_id'])}\n\n"
            f"Conceptos disponibles:\n{concept_lines}\n\n"
            + (f"Feedback del alumno sobre la versión anterior:\n\"{feedback[:600]}\"\n\n" if feedback else "")
            + unit_clause +
            "Ordená las unidades de lo más básico a lo más avanzado, respetando "
            "prerrequisitos. Respondé SOLO JSON:\n"
            '{"goal": {"title": "<=60 chars", "description": "<=300 chars", '
            '"success_criteria": ["<criterio medible de éxito>"], '
            '"constraints": ["<restricción explícita del alumno>"]}, '
            '"units": [{"concept_id": "...", "reason": "<=30 palabras", '
            '"estimated_effort_minutes": 30, "assessment_types": ["explanation", "application"]}], '
            '"assumptions": ["..."], "uncertainties": ["..."]}\n'
            '"goal" describe el proyecto de aprendizaje: success_criteria = 1-5 criterios '
            "medibles para saber que el alumno lo logró; constraints = solo restricciones "
            "que el alumno haya pedido explícitamente (lista vacía si no hay).\n"
            "assessment_types válidos: retrieval, explanation, application, critique, transfer, misconception_correction."
        )
        parsed: dict[str, Any] | None = None
        last_err: Exception | None = None
        for attempt in range(2):
            corrective = (
                "\n\nTu respuesta anterior no fue JSON válido. "
                "Respondé ÚNICAMENTE el objeto JSON pedido, sin texto adicional."
            ) if attempt else ""
            messages = [
                {"role": "system", "content": "Sos un diseñador pedagógico. Respondé solo con el JSON pedido."},
                {"role": "user", "content": prompt + corrective},
            ]
            result = self.provider.generate_chat(
                messages, max_new_tokens=self.max_assessment_tokens * 2, temperature=0.0,
            )
            if getattr(result, "error", None):
                raise RuntimeError(f"roadmap proposal failed: {result.error}")
            try:
                candidate = _extract_json(result.text)
                if not isinstance(candidate.get("units"), list):
                    raise ValueError("LLM response has no units list")
                parsed = candidate
                break
            except Exception as exc:  # JSON malformado → retry
                last_err = exc
        if parsed is None:
            # Fallback determinístico: el scaffold ordena los conceptos como
            # vinieron del retrieval. El gate humano sigue aplicando — el LLM
            # falló, el andamiaje hace trabajo determinístico.
            self.last_fallback_reason = (
                f"roadmap: {type(last_err).__name__}: {str(last_err)[:120]}"
                if last_err else "roadmap: unknown"
            )
            parsed = {
                "units": [
                    {
                        "concept_id": c["concept_id"],
                        "reason": f"unidad {i}: {c.get('title', c['concept_id'])}",
                        "estimated_effort_minutes": 30,
                        "assessment_types": ["explanation"],
                    }
                    for i, c in enumerate(concepts[:7], start=1)
                ],
                "assumptions": [
                    "Roadmap generado por fallback determinístico: "
                    "la propuesta del LLM no fue JSON válido."
                ],
                "uncertainties": [],
            }

        # Scaffold: map the proposal onto contract-shaped units with provenance.
        # The LLM only picks/orders concepts; IDs, orders and source_refs are
        # deterministic scaffold work. _shape_units absorbs imperfect
        # proposals (unknown/duplicate concept_ids, out-of-range counts) —
        # an LLM slip must never dead-end the proposal with a contract error.
        now = _now()
        units = self._shape_units(parsed.get("units", []), concepts, goal_id)
        try:
            # El bloque "goal" de la propuesta refina el LearningGoal persistido
            # (no-op en fallback determinístico o con goal ya confirmado).
            self._refine_goal(goal_id, parsed.get("goal"))
        except Exception as exc:
            print(f"[tutor] goal refine failed: {exc}", flush=True)

        roadmap_id = f"roadmap:{hashlib.sha256(f'{goal_id}{now}'.encode()).hexdigest()[:16]}"
        roadmap = Roadmap(
            roadmap_id=roadmap_id,
            goal_id=goal_id,
            version=version,
            status=RoadmapStatus.PROPOSED,
            units=units,
            assumptions=[str(a) for a in parsed.get("assumptions", [])],
            uncertainties=[str(u) for u in parsed.get("uncertainties", [])],
            change_reason=change_reason,
            previous_roadmap_id=previous_roadmap_id,
            created_at=now,
            approval=None,
            generation=GenerationProvenance(
                generator="tutor-runtime",
                generated_at=now,
                input_hash="sha256:" + hashlib.sha256(
                    json.dumps({"goal_id": goal_id, "concepts": sorted(c["concept_id"] for c in concepts)},
                               sort_keys=True).encode()
                ).hexdigest(),
                model_fingerprint=getattr(self.provider, "model_id", "provider"),
            ),
            field_origins={
                "goal_id": "user", "units": "generated", "assumptions": "generated",
                "uncertainties": "generated", "change_reason": "user_or_generated",
            },
        )
        self.store.save_roadmap(roadmap)
        return roadmap

    def _shape_units(
        self,
        proposed_units: list[dict[str, Any]],
        concepts: list[dict[str, Any]],
        goal_id: str,
    ) -> list[RoadmapUnit]:
        """Map an LLM proposal onto contract-shaped units with provenance.

        The LLM only picks/orders concepts; IDs, orders and source_refs are
        deterministic scaffold work. Imperfect proposals are ABSORBED instead
        of dead-ending the gate: unknown concept_ids are dropped, repeated
        ones deduped, the count topped up to the contract minimum (3) with
        unused concepts (retrieval order) and truncated to the maximum (7).
        """
        concept_by_id = {c["concept_id"]: c for c in concepts}
        units: list[RoadmapUnit] = []
        seen: set[str] = set()

        def _append(concept_id: str, reason: str, effort: int,
                    atypes: list[AssessmentType]) -> None:
            i = len(units) + 1
            units.append(RoadmapUnit(
                unit_id=f"roadmap_unit:{hashlib.sha256(f'{goal_id}{concept_id}{i}'.encode()).hexdigest()[:12]}",
                order=i,
                concept_id=concept_id,
                reason=reason,
                estimated_effort_minutes=effort,
                source_refs=[SourceRef(
                    source_id=concept_id,
                    source_type=SourceType.CHUNK,
                    content_hash=None,
                )],
                assessment_types=atypes,
            ))

        for unit in proposed_units:
            concept_id = str(unit.get("concept_id", "")).strip()
            concept = concept_by_id.get(concept_id)
            if concept is None or concept_id in seen:
                continue  # desconocido o repetido → el scaffold lo absorbe
            seen.add(concept_id)
            assessment_types = [
                AssessmentType(at) for at in (unit.get("assessment_types") or ["explanation"])
                if str(at) in {t.value for t in AssessmentType}
            ] or [AssessmentType.EXPLANATION]
            effort = max(5, min(1440, int(unit.get("estimated_effort_minutes", 30))))
            _append(
                concept_id,
                str(unit.get("reason", ""))[:1000] or f"unidad {len(units) + 1}: {concept.get('title', concept_id)}",
                effort,
                assessment_types,
            )

        # Top-up hasta el mínimo del contrato (3) con conceptos no usados.
        for c in concepts:
            if len(units) >= 3:
                break
            if c["concept_id"] in seen:
                continue
            seen.add(c["concept_id"])
            _append(
                c["concept_id"],
                f"unidad {len(units) + 1}: {c.get('title', c['concept_id'])}",
                30,
                [AssessmentType.EXPLANATION],
            )
        # El contrato exige 3-7 unidades: truncar excedentes (orden intacto).
        del units[7:]
        return units

    def supersede_roadmap(self, roadmap_id: str, *, decided_by: str, note: str | None = None) -> Roadmap:
        """Mark a proposed roadmap superseded (replaced by a debated revision).

        The record is preserved — superseded is a contract status, not a
        deletion. Only proposed roadmaps can be superseded; approved/active
        roadmaps follow their own lifecycle.
        """
        roadmap = self.store.get_roadmap(roadmap_id)
        if roadmap is None:
            raise ValueError(f"unknown roadmap: {roadmap_id}")
        if roadmap.status != RoadmapStatus.PROPOSED:
            raise ValueError(f"only proposed roadmaps can be superseded (status: {roadmap.status.value})")
        # El contrato exige aprobación humana para los estados "aprobados"
        # (superseded incluido): la decisión de reemplazar la toma el humano
        # al debatir, así que queda registrada como aprobación con nota.
        from ipa.tutor.tutor_contracts import HumanApproval, HumanApprovalDecision
        superseded = Roadmap(
            roadmap_id=roadmap.roadmap_id,
            goal_id=roadmap.goal_id,
            version=roadmap.version,
            status=RoadmapStatus.SUPERSEDED,
            units=roadmap.units,
            assumptions=roadmap.assumptions,
            uncertainties=roadmap.uncertainties,
            change_reason=roadmap.change_reason,
            previous_roadmap_id=roadmap.previous_roadmap_id,
            created_at=roadmap.created_at,
            approval=HumanApproval(
                decision=HumanApprovalDecision.APPROVED,
                decided_at=_now(), decided_by=decided_by,
                note=note or "reemplazado por una revisión (debate)",
            ),
            generation=roadmap.generation,
            field_origins=roadmap.field_origins,
        )
        self.store.save_roadmap(superseded)
        return superseded

    def approve_roadmap(self, roadmap_id: str, *, decided_by: str, note: str | None = None) -> Roadmap:
        """Human approval gate: proposed → approved (or rejected).

        This is the ONLY path to an executable roadmap. The LLM can propose;
        only a human can approve (roadmap Fase 2, tutor_common contract).
        """
        roadmap = self.store.get_roadmap(roadmap_id)
        if roadmap is None:
            raise ValueError(f"unknown roadmap: {roadmap_id}")
        if roadmap.status != RoadmapStatus.PROPOSED:
            raise ValueError(f"only proposed roadmaps can be approved (status: {roadmap.status.value})")
        approved = Roadmap(
            roadmap_id=roadmap.roadmap_id,
            goal_id=roadmap.goal_id,
            version=roadmap.version,
            status=RoadmapStatus.APPROVED,
            units=roadmap.units,
            assumptions=roadmap.assumptions,
            uncertainties=roadmap.uncertainties,
            change_reason=roadmap.change_reason,
            previous_roadmap_id=roadmap.previous_roadmap_id,
            created_at=roadmap.created_at,
            approval=HumanApproval(
                decision=HumanApprovalDecision.APPROVED,
                decided_at=_now(),
                decided_by=decided_by,
                note=note,
            ),
            generation=roadmap.generation,
            field_origins=roadmap.field_origins,
        )
        self.store.save_roadmap(approved)
        return approved

    def reject_roadmap(self, roadmap_id: str, *, decided_by: str, note: str | None = None) -> Roadmap:
        """Human rejection: proposed → rejected."""
        roadmap = self.store.get_roadmap(roadmap_id)
        if roadmap is None:
            raise ValueError(f"unknown roadmap: {roadmap_id}")
        rejected = Roadmap(
            roadmap_id=roadmap.roadmap_id,
            goal_id=roadmap.goal_id,
            version=roadmap.version,
            status=RoadmapStatus.REJECTED,
            units=roadmap.units,
            assumptions=roadmap.assumptions,
            uncertainties=roadmap.uncertainties,
            change_reason=roadmap.change_reason,
            previous_roadmap_id=roadmap.previous_roadmap_id,
            created_at=roadmap.created_at,
            approval=HumanApproval(
                decision=HumanApprovalDecision.REJECTED,
                decided_at=_now(),
                decided_by=decided_by,
                note=note,
            ),
            generation=roadmap.generation,
            field_origins=roadmap.field_origins,
        )
        self.store.save_roadmap(rejected)
        return rejected

    def reopen_roadmap(self, roadmap_id: str, *, decided_by: str, note: str | None = None) -> Roadmap:
        """Human gate: return a decided roadmap to pending debate.

        approved/active/completed/rejected → proposed. The dashboard gate
        exposes three states (proposed | accepted | rejected); moving back
        to proposed records CHANGES_REQUESTED so the audit trail keeps the
        human decision. Superseded roadmaps are lineage history and cannot
        be reopened.
        """
        roadmap = self.store.get_roadmap(roadmap_id)
        if roadmap is None:
            raise ValueError(f"unknown roadmap: {roadmap_id}")
        if roadmap.status == RoadmapStatus.PROPOSED:
            return roadmap
        if roadmap.status == RoadmapStatus.SUPERSEDED:
            raise ValueError("superseded roadmaps cannot be reopened")
        reopened = Roadmap(
            roadmap_id=roadmap.roadmap_id,
            goal_id=roadmap.goal_id,
            version=roadmap.version,
            status=RoadmapStatus.PROPOSED,
            units=roadmap.units,
            assumptions=roadmap.assumptions,
            uncertainties=roadmap.uncertainties,
            change_reason=roadmap.change_reason,
            previous_roadmap_id=roadmap.previous_roadmap_id,
            created_at=roadmap.created_at,
            approval=HumanApproval(
                decision=HumanApprovalDecision.CHANGES_REQUESTED,
                decided_at=_now(),
                decided_by=decided_by,
                note=note,
            ),
            generation=roadmap.generation,
            field_origins=roadmap.field_origins,
        )
        self.store.save_roadmap(reopened)
        return reopened

    def activate_roadmap(self, roadmap_id: str) -> Roadmap:
        """approved → active. Only an approved roadmap can be activated."""
        roadmap = self.store.get_roadmap(roadmap_id)
        if roadmap is None:
            raise ValueError(f"unknown roadmap: {roadmap_id}")
        if roadmap.status != RoadmapStatus.APPROVED:
            raise ValueError(f"only approved roadmaps can be activated (status: {roadmap.status.value})")
        active = Roadmap(
            roadmap_id=roadmap.roadmap_id,
            goal_id=roadmap.goal_id,
            version=roadmap.version,
            status=RoadmapStatus.ACTIVE,
            units=roadmap.units,
            assumptions=roadmap.assumptions,
            uncertainties=roadmap.uncertainties,
            change_reason=roadmap.change_reason,
            previous_roadmap_id=roadmap.previous_roadmap_id,
            created_at=roadmap.created_at,
            approval=roadmap.approval,
            generation=roadmap.generation,
            field_origins=roadmap.field_origins,
        )
        self.store.save_roadmap(active)
        return active

    # ------------------------------------------------------------------
    # LearningGoal: el "proyecto" que un roadmap sirve (persistido, gateado)
    # ------------------------------------------------------------------

    def ensure_goal(
        self,
        goal_id: str,
        *,
        title: str,
        description: str = "",
        success_criteria: list[str] | None = None,
        constraints: list[str] | None = None,
    ) -> LearningGoal:
        """Create the goal if missing (status 'proposed'); resurrect cancelled.

        The deterministic scaffold fills the minimum contract fields so a
        project exists as soon as the learner names a topic — the LLM refines
        title/description/criteria at proposal time (_refine_goal) and the
        human gate approves it together with the roadmap (one decision).
        """
        existing = self.store.get_goal(goal_id)
        if existing is not None and existing.status not in (
            LearningGoalStatus.CANCELLED, LearningGoalStatus.COMPLETED
        ):
            return existing
        now = _now()
        title = (title or goal_id.removeprefix("goal:").replace("-", " ")).strip()[:200] or goal_id
        goal = LearningGoal(
            goal_id=goal_id,
            title=title,
            description=(description.strip() or f"Objetivo de aprendizaje: {title}")[:4000],
            status=LearningGoalStatus.PROPOSED,
            success_criteria=(
                [c.strip()[:500] for c in (success_criteria or []) if c.strip()][:20]
                or [f"Comprender {title} y poder aplicarlo en un caso concreto"]
            ),
            created_at=existing.created_at if existing else now,
            updated_at=now,
            approval=None,
            field_origins={
                "title": "user", "description": "user",
                "success_criteria": "system", "constraints": "user",
                "status": "system",
            },
            constraints=[c.strip()[:500] for c in (constraints or []) if c.strip()][:20],
        )
        self.store.save_goal(goal)
        return goal

    def _refine_goal(self, goal_id: str, patch: dict[str, Any] | None) -> None:
        """Fold the LLM's goal fields (from the roadmap proposal JSON) into
        the persisted LearningGoal — only while it is still 'proposed'.

        A confirmed/active goal is a human-approved record; the model never
        rewrites it. Fields the proposal actually changes are re-originated
        as 'generated' with the provider's provenance.
        """
        goal = self.store.get_goal(goal_id)
        if goal is None:
            goal = self.ensure_goal(goal_id, title="")
        if goal.status != LearningGoalStatus.PROPOSED or not isinstance(patch, dict):
            return
        title = str(patch.get("title") or "").strip()[:200] or goal.title
        description = str(patch.get("description") or "").strip()[:4000] or goal.description
        criteria = [
            str(c).strip()[:500] for c in (patch.get("success_criteria") or [])
            if str(c).strip()
        ][:20] or goal.success_criteria
        constraints = [
            str(c).strip()[:500] for c in (patch.get("constraints") or [])
            if str(c).strip()
        ][:20] or goal.constraints
        changed = {
            field: "generated"
            for field, new, old in (
                ("title", title, goal.title),
                ("description", description, goal.description),
                ("success_criteria", criteria, goal.success_criteria),
                ("constraints", constraints, goal.constraints),
            )
            if new != old
        }
        if not changed:
            return
        refined = replace(
            goal,
            title=title,
            description=description,
            success_criteria=criteria,
            constraints=constraints,
            field_origins={**goal.field_origins, **changed},
            generation=GenerationProvenance(
                generator="tutor-runtime",
                generated_at=_now(),
                input_hash="sha256:" + hashlib.sha256(
                    json.dumps({"goal_id": goal_id, "patch": patch},
                               sort_keys=True, default=str).encode()
                ).hexdigest(),
                model_fingerprint=(
                    getattr(self.provider, "model_id", "provider")
                    if self.provider else "tutor-runtime"
                ),
            ),
            updated_at=_now(),
        )
        self.store.save_goal(refined)

    def approve_goal(self, goal_id: str, *, decided_by: str,
                     note: str | None = None) -> LearningGoal | None:
        """proposed → confirmed. The same human gate that approves the
        roadmap confirms the goal it serves — one decision, one record."""
        goal = self.store.get_goal(goal_id)
        if goal is None or goal.status != LearningGoalStatus.PROPOSED:
            return goal
        goal = replace(
            goal,
            status=LearningGoalStatus.CONFIRMED,
            approval=HumanApproval(
                decision=HumanApprovalDecision.APPROVED,
                decided_at=_now(), decided_by=decided_by, note=note,
            ),
            updated_at=_now(),
        )
        self.store.save_goal(goal)
        return goal

    def activate_goal(self, goal_id: str) -> LearningGoal | None:
        """confirmed → active (the approval record carries over)."""
        goal = self.store.get_goal(goal_id)
        if goal is None or goal.status != LearningGoalStatus.CONFIRMED:
            return goal
        goal = replace(goal, status=LearningGoalStatus.ACTIVE, updated_at=_now())
        self.store.save_goal(goal)
        return goal

    def reopen_goal(self, goal_id: str, *,
                    decided_by: str) -> LearningGoal | None:
        """Decided goal → proposed again (its roadmap reopened for debate).

        Reopening a roadmap re-engages its goal: cancelled also comes back.
        A completed goal is a closed record and is never reopened.
        """
        goal = self.store.get_goal(goal_id)
        if goal is None or goal.status in (
            LearningGoalStatus.PROPOSED,
            LearningGoalStatus.COMPLETED,
        ):
            return goal
        goal = replace(
            goal,
            status=LearningGoalStatus.PROPOSED,
            approval=HumanApproval(
                decision=HumanApprovalDecision.CHANGES_REQUESTED,
                decided_at=_now(), decided_by=decided_by,
            ),
            updated_at=_now(),
        )
        self.store.save_goal(goal)
        return goal

    def cancel_goal(self, goal_id: str) -> LearningGoal | None:
        """→ cancelled. The caller checks no other live roadmap version still
        serves the goal before calling."""
        goal = self.store.get_goal(goal_id)
        if goal is None or goal.status == LearningGoalStatus.CANCELLED:
            return goal
        goal = replace(goal, status=LearningGoalStatus.CANCELLED, updated_at=_now())
        self.store.save_goal(goal)
        return goal

    def goal_for_roadmap(self, roadmap: Any) -> LearningGoal:
        """Get the persisted goal, synthesizing one for legacy roadmaps."""
        return goal_for_roadmap(self.store, roadmap)

    # ------------------------------------------------------------------
    # ResearchRequest: crear → aprobar (humano) → ejecutar (Fase 1 executor)
    # ------------------------------------------------------------------

    def create_research_request(
        self,
        concept_id: str,
        question: str,
        *,
        trigger: Any = None,
        allowed_domains: list[str] | None = None,
        budget: Any | None = None,
        goal_id: str = "goal:default",
    ) -> Any:
        """Create a ResearchRequest from a detected educational gap.

        The gap evidence is the deterministic coverage assessment (Fase 1
        scaffold). Status lands as 'pending_approval' — the LLM can detect a
        gap; only a human can authorize web research (contract invariant:
        research_execution_requires_human_approval).
        """
        from ipa.tutor.tutor_contracts import (
            GenerationProvenance, ResearchBudget, ResearchRequest,
            ResearchStatus, ResearchTrigger, SourceRef, SourceType,
        )
        now = _now()
        trigger = trigger or ResearchTrigger.INSUFFICIENT_SOURCES
        budget = budget or ResearchBudget(max_urls=15, max_seconds=300, max_bytes=20971520, max_depth=1)
        allowed_domains = allowed_domains or ["docs.python.org", "arxiv.org", "github.com"]
        # Gap evidence: the coverage assessment is the deterministic scaffold
        gap_evidence = [SourceRef(
            source_id=concept_id, source_type=SourceType.CHUNK, content_hash=None,
        )]
        request_id = f"research_request:{hashlib.sha256((concept_id + question + now).encode()).hexdigest()[:16]}"
        request = ResearchRequest(
            request_id=request_id,
            goal_id=goal_id,
            concept_id=concept_id,
            question=question,
            trigger=trigger,
            gap_evidence=gap_evidence,
            allowed_domains=allowed_domains,
            budget=budget,
            status=ResearchStatus.PENDING_APPROVAL,
            created_at=now,
            updated_at=now,
            approval=None,
            generation=GenerationProvenance(
                generator="tutor-runtime", generated_at=now,
                input_hash="sha256:" + hashlib.sha256(question.encode()).hexdigest(),
                model_fingerprint=getattr(self.provider, "model_id", "tutor-runtime") if self.provider else "tutor-runtime",
            ),
            field_origins={
                "question": "user_or_generated", "trigger": "system",
                "gap_evidence": "source", "allowed_domains": "user_or_system_policy",
                "budget": "system_policy",
            },
        )
        self.store.save_research_request(request)
        return request

    def approve_research_request(self, request_id: str, *, decided_by: str, note: str | None = None) -> Any:
        """Human approval gate: pending_approval → approved."""
        from ipa.tutor.tutor_contracts import ResearchStatus
        request = self.store.get_research_request(request_id)
        if request is None:
            raise ValueError(f"unknown research request: {request_id}")
        if request.status != ResearchStatus.PENDING_APPROVAL:
            raise ValueError(f"only pending_approval requests can be approved (status: {request.status.value})")
        from ipa.tutor.tutor_contracts import HumanApproval, HumanApprovalDecision, ResearchRequest
        approved = ResearchRequest(
            request_id=request.request_id, goal_id=request.goal_id,
            concept_id=request.concept_id, question=request.question,
            trigger=request.trigger, gap_evidence=request.gap_evidence,
            allowed_domains=request.allowed_domains, budget=request.budget,
            status=ResearchStatus.APPROVED,
            created_at=request.created_at, updated_at=_now(),
            approval=HumanApproval(
                decision=HumanApprovalDecision.APPROVED,
                decided_at=_now(), decided_by=decided_by, note=note,
            ),
            generation=request.generation, field_origins=request.field_origins,
            job_id=request.job_id, result_source_refs=request.result_source_refs,
        )
        self.store.save_research_request(approved)
        return approved

    def reject_research_request(self, request_id: str, *, decided_by: str, note: str | None = None) -> Any:
        """Human rejection: pending_approval → cancelled."""
        from ipa.tutor.tutor_contracts import HumanApproval, HumanApprovalDecision, ResearchRequest, ResearchStatus
        request = self.store.get_research_request(request_id)
        if request is None:
            raise ValueError(f"unknown research request: {request_id}")
        cancelled = ResearchRequest(
            request_id=request.request_id, goal_id=request.goal_id,
            concept_id=request.concept_id, question=request.question,
            trigger=request.trigger, gap_evidence=request.gap_evidence,
            allowed_domains=request.allowed_domains, budget=request.budget,
            status=ResearchStatus.CANCELLED,
            created_at=request.created_at, updated_at=_now(),
            approval=HumanApproval(
                decision=HumanApprovalDecision.REJECTED,
                decided_at=_now(), decided_by=decided_by, note=note,
            ),
            generation=request.generation, field_origins=request.field_origins,
            job_id=request.job_id, result_source_refs=request.result_source_refs,
        )
        self.store.save_research_request(cancelled)
        return cancelled

    def execute_approved_research(self, request_id: str, ctx: Any, *, judge: Any | None = None,
                                  landing_dir: Any = None) -> dict[str, Any]:
        """Execute an APPROVED ResearchRequest via the Fase 1 agentic executor.

        Only approved requests run; the executor receives the request_id for
        provenance tracing (web_source.research_request_id).

        ``landing_dir`` None → dir de trabajo privado por corrida (PM-004): la
        investigación no comparte ``Landing/web`` con el scraper del pipeline.
        """
        from ipa.tutor.tutor_contracts import ResearchRequest, ResearchStatus
        request = self.store.get_research_request(request_id)
        if request is None:
            raise ValueError(f"unknown research request: {request_id}")
        if request.status != ResearchStatus.APPROVED:
            raise ValueError(f"only approved research can execute (status: {request.status.value})")
        from ipa.agent.research_executor import execute_research
        session_id = self.core.ensure_session()
        episodes = self.core.memory.get_episodes(session_id, limit=1)
        ep_id = episodes[0].episode_id if episodes else self.core.memory.record_episode(
            session_id, turn_role="user", content=request.question,
            identity_hash=self.core.identity.identity_hash,
        ).episode_id
        call, result, research = execute_research(
            request.question, ctx,
            session_id=session_id, episode_id=ep_id,
            research_request_id=request.request_id,
            max_urls=request.budget.max_urls,
            max_seconds=request.budget.max_seconds,
            allowed_domains=list(request.allowed_domains),
            judge=judge,
            landing_dir=landing_dir,
        )
        # Record the outcome on the request (job_id + result refs)
        from ipa.tutor.tutor_contracts import SourceRef, SourceType
        result_refs = [
            SourceRef(source_id=ws.web_source_id, source_type=SourceType.ARTIFACT,
                      content_hash=ws.content_hash)
            for ws in research.web_sources
        ]
        final_status = ResearchStatus.COMPLETED if result.status == "completed" else ResearchStatus.FAILED
        completed = ResearchRequest(
            request_id=request.request_id, goal_id=request.goal_id,
            concept_id=request.concept_id, question=request.question,
            trigger=request.trigger, gap_evidence=request.gap_evidence,
            allowed_domains=request.allowed_domains, budget=request.budget,
            status=final_status,
            created_at=request.created_at, updated_at=_now(),
            approval=request.approval,
            generation=request.generation, field_origins=request.field_origins,
            job_id=call.tool_call_id, result_source_refs=result_refs,
        )
        self.store.save_research_request(completed)
        return {"request": completed, "tool_result": result, "research": research}

    # ------------------------------------------------------------------
    # Assessment (LLM classifies, scaffold records)
    # ------------------------------------------------------------------

    def assess(self, topic_id: str, question: str, answer: str, *, rubric_id: str = "default") -> dict[str, Any]:
        """Evaluate a learner answer: LLM classifies with structured JSON +
        abstention; the result is recorded as evidence and updates mastery.

        Returns {"assessment": AssessmentResult-like dict, "record": UserTopicRecord,
                 "abstained": bool}.
        """
        if self.provider is None:
            raise ValueError("assessment requires a provider (LLM classifies)")
        messages = [
            {"role": "system", "content": "Sos un evaluador pedagógico preciso. Respondé solo con el JSON pedido."},
            {"role": "user", "content": self._assessment_prompt(topic_id, question, answer)},
        ]
        result = self.provider.generate_chat(
            messages, max_new_tokens=self.max_assessment_tokens, temperature=0.0,
        )
        if getattr(result, "error", None):
            return self._abstain(topic_id, question, answer, f"provider error: {result.error}")
        try:
            parsed = _extract_json(result.text)
        except Exception as exc:
            self.last_fallback_reason = f"assess: {type(exc).__name__}: {str(exc)[:120]}"
            return self._abstain(topic_id, question, answer, "LLM output unparseable")

        if parsed.get("abstain") or float(parsed.get("confidence", 0.0)) < ABSTENTION_THRESHOLD:
            return self._abstain(
                topic_id, question, answer,
                f"low confidence: {parsed.get('confidence', 0)}",
            )

        assessment_id = f"assessment:{hashlib.sha256(f'{topic_id}{answer}'.encode()).hexdigest()[:16]}"
        now = _now()
        score = max(0.0, min(1.0, float(parsed.get("score", 0.0))))
        status = str(parsed.get("status", "needs_review"))
        recommended_action = str(parsed.get("recommended_action", "human_review"))
        # Normalize model-provided misconceptions into contract shape.
        misconceptions = []
        for item in parsed.get("misconceptions", []):
            if isinstance(item, dict):
                misconceptions.append({
                    "claim": str(item.get("claim", "unknown claim")),
                    "correction": str(item.get("correction", "review the concept")),
                    "evidence": [{
                        "source_id": assessment_id,
                        "source_type": "assessment",
                        "content_hash": None,
                    }],
                })
            else:
                misconceptions.append({
                    "claim": str(item),
                    "correction": "review the concept",
                    "evidence": [{
                        "source_id": assessment_id,
                        "source_type": "assessment",
                        "content_hash": None,
                    }],
                })
        assessment = {
            "assessment_id": assessment_id,
            "session_id": self.core.ensure_session(),
            "concept_id": topic_id,
            "assessment_type": "explanation",
            "answer_hash": answer_hash(answer),
            "rubric_id": rubric_id,
            "score": score,
            "status": status,
            "strengths": [str(s) for s in parsed.get("strengths", [])],
            "gaps": [str(g) for g in parsed.get("gaps", [])],
            "misconceptions": misconceptions,
            "recommended_action": recommended_action,
            "confidence": float(parsed.get("confidence", 0.0)),
            "created_at": now,
            "generation": {
                "generator": "tutor-runtime",
                "generated_at": now,
                "input_hash": answer_hash(answer),
                "model_fingerprint": getattr(self.provider, "model_id", "provider"),
            },
            "field_origins": {
                "answer_hash": "user", "score": "generated", "status": "generated",
                "strengths": "generated", "gaps": "generated",
                "misconceptions": "generated", "evidence": "source",
                "recommended_action": "generated",
            },
        }

        # The assessment IS the evidence (roadmap Fase 2)
        evidence = self.record_evidence(
            topic_id, EvidenceType.ASSESSMENT,
            observation=f"Assessment {assessment_id}: score={assessment['score']:.2f}, "
                        f"status={assessment['status']}. Q: {question[:100]} A: {answer[:200]}",
            assessment_id=assessment_id,
        )
        assessment["evidence"] = [{
            "source_id": assessment_id,
            "source_type": "assessment",
            "content_hash": answer_hash(answer),
        }]

        # Mastery update: the new record traces to this assessment
        record = self.update_mastery(topic_id, assessment, evidence.evidence_id)
        assessment["assessment_id"] = assessment_id
        return {"assessment": assessment, "record": record, "abstained": False}

    def _assessment_prompt(self, topic_id: str, question: str, answer: str) -> str:
        return (
            f"{self._mastery_context(topic_id)}\n\n"
            f"Consigna: {question}\n\n"
            f"Respuesta del alumno:\n{answer[:2000]}\n\n"
            "Evaluá la respuesta contra la consigna. Respondé SOLO JSON:\n"
            '{"score": 0.0-1.0, "status": "understood|applied|needs_review|misconception", '
            '"strengths": ["..."], "gaps": ["..."], "misconceptions": ["..."], '
            '"recommended_action": "advance|guided_retry|practical_retry|review_prerequisite|'
            'correct_misconception|research_gap|human_review", '
            '"confidence": 0.0-1.0, "abstain": false}'
        )

    def _abstain(self, topic_id: str, question: str, answer: str, reason: str) -> dict[str, Any]:
        """Abstention: no score invented; escalate to human review (BM-006)."""
        self.last_fallback_reason = f"abstain: {reason}" if (reason := reason) else ""
        assessment = {
            "assessment_id": f"assessment:abstained:{hashlib.sha256(answer.encode()).hexdigest()[:12]}",
            "topic_id": topic_id,
            "question": question,
            "answer_hash": answer_hash(answer),
            "score": None,
            "status": "needs_review",
            "recommended_action": "human_review",
            "abstained": True,
            "reason": reason,
            "created_at": _now(),
        }
        evidence = self.record_evidence(
            topic_id, EvidenceType.OBSERVATION,
            f"Assessment abstained: {reason}. Q: {question[:100]} A: {answer[:150]}",
            assessment_id=assessment["assessment_id"],
        )
        return {"assessment": assessment, "record": None, "abstained": True}

    def update_mastery(self, topic_id: str, assessment: dict[str, Any], evidence_id: str) -> UserTopicRecord:
        """Mastery update: the assessment IS the evidence (DEC-002)."""
        existing = self.store.get_topic_record(topic_id)
        now = _now()
        status = MasteryStatus(assessment["status"])
        record = UserTopicRecord(
            record_id=existing.record_id if existing else f"user_topic_record:{topic_id.replace(':', '_')}",
            topic_id=topic_id,
            mastery_status=status,
            mastery_score=float(assessment["score"]),
            attempts=(existing.attempts if existing else 0) + 1,
            last_assessment_id=assessment["assessment_id"],
            evidence_ids=(existing.evidence_ids if existing else []) + [evidence_id],
            updated_at=now,
            created_at=existing.created_at if existing else now,
            generation=GenerationProvenance(
                generator="tutor-runtime", generated_at=now,
                input_hash="sha256:" + hashlib.sha256(
                    json.dumps(assessment, sort_keys=True).encode()
                ).hexdigest(),
                model_fingerprint="tutor-runtime",
            ),
            field_origins={"mastery_status": "generated", "mastery_score": "generated",
                           "attempts": "system", "last_assessment_id": "system",
                           "evidence_ids": "system"},
        )
        self.store.upsert_topic_record(record)
        return record

    # ------------------------------------------------------------------
    # Evidence recording
    # ------------------------------------------------------------------

    def record_evidence(self, topic_id: str, evidence_type: EvidenceType, observation: str, *,
                        assessment_id: str | None = None,
                        source_refs: list[SourceRef] | None = None) -> UserEvidence:
        """Append evidence to the log (append-only)."""
        now = _now()
        evidence_id = "user_evidence:" + hashlib.sha256(
            f"{topic_id}{now}{observation}".encode()
        ).hexdigest()[:16]
        evidence = UserEvidence(
            evidence_id=evidence_id,
            topic_id=topic_id,
            evidence_type=evidence_type,
            observation=observation,
            observed_at=now,
            recorded_at=now,
            source_refs=source_refs or [SourceRef(
                source_id=assessment_id or topic_id,
                source_type="assessment" if assessment_id else "artifact",
                content_hash=None,
            )],
            generation=GenerationProvenance(
                generator="tutor-runtime", generated_at=now,
                input_hash="sha256:" + hashlib.sha256(observation.encode()).hexdigest(),
                model_fingerprint="tutor-runtime",
            ),
            field_origins={"observation": "generated", "source_refs": "source"},
            assessment_id=assessment_id,
            session_id=self.core.session_id,
        )
        self.store.add_evidence(evidence)
        return evidence


def goal_for_roadmap(store: TutorStore, roadmap: Any) -> LearningGoal:
    """Get the persisted goal for a roadmap, synthesizing one for legacy
    roadmaps written before learning_goals existed (bare goal:slug).

    Status mirrors the LATEST version of the goal's roadmap line, and a
    confirmed/active goal REUSES the roadmap's own HumanApproval — the human
    decision that approved the roadmap is literally the goal's approval
    record. Store-level (no session/provider) so read endpoints can use it.
    """
    goal = store.get_goal(roadmap.goal_id)
    if goal is not None:
        return goal
    latest = roadmap
    try:
        versions = store.list_roadmaps(roadmap.goal_id)  # version DESC
        if versions:
            latest = versions[0]
    except Exception:
        pass
    status = {
        RoadmapStatus.PROPOSED: LearningGoalStatus.PROPOSED,
        RoadmapStatus.APPROVED: LearningGoalStatus.CONFIRMED,
        RoadmapStatus.ACTIVE: LearningGoalStatus.ACTIVE,
        RoadmapStatus.COMPLETED: LearningGoalStatus.COMPLETED,
        RoadmapStatus.SUPERSEDED: LearningGoalStatus.PROPOSED,
        RoadmapStatus.REJECTED: LearningGoalStatus.CANCELLED,
    }.get(latest.status, LearningGoalStatus.PROPOSED)
    approval = latest.approval
    if status in (
        LearningGoalStatus.CONFIRMED,
        LearningGoalStatus.ACTIVE,
        LearningGoalStatus.COMPLETED,
    ) and not (approval and approval.approved):
        status, approval = LearningGoalStatus.PROPOSED, None
    now = _now()
    topic = roadmap.goal_id.removeprefix("goal:").replace("-", " ")
    goal = LearningGoal(
        goal_id=roadmap.goal_id,
        title=topic,
        description=(
            latest.change_reason
            or f"Objetivo de aprendizaje sobre {topic}"
        )[:4000],
        status=status,
        success_criteria=[f"Comprender {topic} y poder aplicarlo en un caso concreto"],
        created_at=latest.created_at,
        updated_at=now,
        approval=approval,
        field_origins={
            "title": "system", "description": "system",
            "success_criteria": "system", "constraints": "user",
            "status": "system",
        },
    )
    store.save_goal(goal)
    return goal


__all__ = [
    "ABSTENTION_THRESHOLD",
    "DEFAULT_TUTOR_STORE",
    "DiagnosisResult",
    "TutorSession",
    "TutorStore",
    "goal_for_roadmap",
]
