"""Punto 6: Calibración de incertidumbre + active research agenda.

El agente no sabe qué no sabe. No hay "mi confianza en X es baja → debería
buscar". No hay active learning: el agente no decide qué investigar por su
cuenta para servir mejor al usuario. Responde, no anticipa.

Esta capa trackea confianza por tópico y propone una agenda de investigación
activa: tópicos donde la confianza es baja y que el usuario podría preguntar.

Fuentes de señal de confianza (determinísticas, sin VRAM):
  - search_corpus: score promedio de hits por tópico (query)
    - score alto → confianza alta
    - 0 resultados → confianza 0
    - score bajo → confianza baja
  - compile_report: si el reporte tiene pocos documentos → confianza baja
  - research_topic: si la investigación ingirió pocos docs → confianza baja
  - Tutor assessments: mastery por tópico (ya existe en tutor.db)

Active research agenda:
  - Para cada tópico con confianza < THRESHOLD, proponer investigarlo
  - La propuesta es pending → aprobación humana (patrón ConsolidationStore)
  - Si se aprueba, se lanza research_topic automáticamente en idle

Invariante: las propuestas de research NUNCA se auto-ejecutan. Pending →
humano approve → el idle worker las ejecuta. El agente puede SUGERIR
("notá que sé poco sobre X, ¿investigo?") pero no decidir solo.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_UNCERTAINTY_STORE = Path("outputs/agent/uncertainty.db")

# Umbral de confianza: por debajo de esto, el tópico es candidato a research
CONFIDENCE_THRESHOLD = 0.4
# Mínimo de observaciones para considerar la confianza estable
MIN_OBSERVATIONS = 2


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TopicConfidence:
    """Confianza del agente en un tópico, trackeada over time."""
    topic: str  # query o tópico normalizado
    confidence: float  # 0.0-1.0, media de observaciones
    observation_count: int
    last_score: float
    last_observed_at: str
    source: str  # "search_corpus" | "compile_report" | "research_topic" | "tutor"
    needs_research: bool  # True si confidence < THRESHOLD y observations >= MIN

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResearchProposal:
    """Propuesta de investigación activa para un tópico de baja confianza."""
    proposal_id: str
    topic: str
    current_confidence: float
    rationale: str  # por qué se propone investigar
    suggested_query: str  # query sugerida para research_topic
    status: str  # "pending" | "approved" | "rejected" | "executed"
    proposed_at: str
    decided_by: str | None = None
    decided_at: str | None = None
    executed_task_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# UncertaintyStore
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS topic_confidence (
    topic           TEXT PRIMARY KEY,
    confidence      REAL NOT NULL,
    observation_count INTEGER NOT NULL DEFAULT 0,
    last_score      REAL NOT NULL,
    last_observed_at TEXT NOT NULL,
    source          TEXT NOT NULL,
    needs_research  INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS research_proposals (
    proposal_id     TEXT PRIMARY KEY,
    topic           TEXT NOT NULL,
    current_confidence REAL NOT NULL,
    rationale       TEXT NOT NULL,
    suggested_query TEXT NOT NULL,
    status          TEXT NOT NULL,
    proposed_at     TEXT NOT NULL,
    decided_by      TEXT,
    decided_at      TEXT,
    executed_task_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_proposals_status ON research_proposals(status);
CREATE INDEX IF NOT EXISTS idx_confidence_needs ON topic_confidence(needs_research);
"""


class UncertaintyStore:
    """SQLite persistence for confidence tracking and research proposals."""

    def __init__(self, store_path: str | Path | None = None) -> None:
        if store_path is None:
            store_path = DEFAULT_UNCERTAINTY_STORE
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.store_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def record_observation(self, topic: str, score: float, *, source: str = "search_corpus") -> TopicConfidence:
        """Registra una observación de score para un tópico y actualiza la confianza."""
        topic = topic.strip().lower()[:200]
        if not topic:
            raise ValueError("topic must be non-empty")
        score = max(0.0, min(1.0, score))
        now = _now()
        existing = self._conn.execute(
            "SELECT * FROM topic_confidence WHERE topic = ?", (topic,)
        ).fetchone()
        if existing:
            new_count = existing["observation_count"] + 1
            # Media móvil exponencial: peso 0.3 a la nueva observación
            new_confidence = existing["confidence"] * 0.7 + score * 0.3
            needs = 1 if (new_count >= MIN_OBSERVATIONS and new_confidence < CONFIDENCE_THRESHOLD) else 0
            self._conn.execute(
                "UPDATE topic_confidence SET confidence=?, observation_count=?, last_score=?, "
                "last_observed_at=?, source=?, needs_research=? WHERE topic=?",
                (new_confidence, new_count, score, now, source, needs, topic),
            )
        else:
            needs = 1 if (1 >= MIN_OBSERVATIONS and score < CONFIDENCE_THRESHOLD) else 0
            self._conn.execute(
                "INSERT INTO topic_confidence VALUES (?, ?, 1, ?, ?, ?, ?)",
                (topic, score, score, now, source, needs),
            )
            new_count = 1
            new_confidence = score
        self._conn.commit()
        return TopicConfidence(
            topic=topic, confidence=new_confidence, observation_count=new_count,
            last_score=score, last_observed_at=now, source=source,
            needs_research=bool(needs),
        )

    def get_confidence(self, topic: str) -> TopicConfidence | None:
        row = self._conn.execute(
            "SELECT * FROM topic_confidence WHERE topic = ?", (topic.strip().lower(),)
        ).fetchone()
        return self._row_to_confidence(row) if row else None

    def list_low_confidence(self, *, limit: int = 20) -> list[TopicConfidence]:
        """Tópicos con needs_research=True (baja confianza + suficientes observaciones)."""
        rows = self._conn.execute(
            "SELECT * FROM topic_confidence WHERE needs_research = 1 "
            "ORDER BY confidence ASC LIMIT ?", (limit,)
        ).fetchall()
        return [self._row_to_confidence(row) for row in rows]

    def list_all_confidence(self, *, limit: int = 50) -> list[TopicConfidence]:
        rows = self._conn.execute(
            "SELECT * FROM topic_confidence ORDER BY last_observed_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._row_to_confidence(row) for row in rows]

    def save_proposal(self, proposal: ResearchProposal) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO research_proposals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (proposal.proposal_id, proposal.topic, proposal.current_confidence,
             proposal.rationale, proposal.suggested_query, proposal.status,
             proposal.proposed_at, proposal.decided_by, proposal.decided_at,
             proposal.executed_task_id),
        )
        self._conn.commit()

    def list_proposals(self, *, status: str | None = None, limit: int = 20) -> list[ResearchProposal]:
        if status:
            rows = self._conn.execute(
                "SELECT * FROM research_proposals WHERE status = ? ORDER BY proposed_at DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM research_proposals ORDER BY proposed_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_proposal(row) for row in rows]

    def decide_proposal(self, proposal_id: str, *, approved: bool, decided_by: str = "human") -> None:
        p = next((p for p in self.list_proposals(limit=200) if p.proposal_id == proposal_id), None)
        if p is None:
            raise ValueError(f"unknown proposal: {proposal_id}")
        new_status = "approved" if approved else "rejected"
        updated = ResearchProposal(
            proposal_id=p.proposal_id, topic=p.topic, current_confidence=p.current_confidence,
            rationale=p.rationale, suggested_query=p.suggested_query,
            status=new_status, proposed_at=p.proposed_at, decided_by=decided_by,
            decided_at=_now(), executed_task_id=p.executed_task_id,
        )
        self.save_proposal(updated)

    def mark_executed(self, proposal_id: str, task_id: str) -> None:
        p = next((p for p in self.list_proposals(limit=200) if p.proposal_id == proposal_id), None)
        if p is None:
            raise ValueError(f"unknown proposal: {proposal_id}")
        updated = ResearchProposal(
            proposal_id=p.proposal_id, topic=p.topic, current_confidence=p.current_confidence,
            rationale=p.rationale, suggested_query=p.suggested_query,
            status="executed", proposed_at=p.proposed_at, decided_by=p.decided_by,
            decided_at=p.decided_at, executed_task_id=task_id,
        )
        self.save_proposal(updated)

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _row_to_confidence(row: sqlite3.Row) -> TopicConfidence:
        return TopicConfidence(
            topic=row["topic"], confidence=row["confidence"],
            observation_count=row["observation_count"], last_score=row["last_score"],
            last_observed_at=row["last_observed_at"], source=row["source"],
            needs_research=bool(row["needs_research"]),
        )

    @staticmethod
    def _row_to_proposal(row: sqlite3.Row) -> ResearchProposal:
        return ResearchProposal(
            proposal_id=row["proposal_id"], topic=row["topic"],
            current_confidence=row["current_confidence"], rationale=row["rationale"],
            suggested_query=row["suggested_query"], status=row["status"],
            proposed_at=row["proposed_at"], decided_by=row["decided_by"],
            decided_at=row["decided_at"], executed_task_id=row["executed_task_id"],
        )


# ---------------------------------------------------------------------------
# UncertaintyTracker — trackea confianza desde tool results
# ---------------------------------------------------------------------------

class UncertaintyTracker:
    """Registra observaciones de confianza desde resultados de tools.

    Hook: el dashboard llama record_observation después de cada
    search_corpus / compile_report / research_topic. Determinístico.
    """

    def __init__(self, store: UncertaintyStore) -> None:
        self.store = store

    def observe_search(self, query: str, hits_count: int, avg_score: float) -> TopicConfidence:
        """Registra confianza desde un resultado de search_corpus."""
        topic = query.strip().lower()[:200]
        # Score normalizado: 0 hits → 0, hits con score alto → alto
        if hits_count == 0:
            score = 0.0
        else:
            # avg_score suele estar en rango [0, 1] ya; si no, normalizar
            score = max(0.0, min(1.0, avg_score))
            # Penalizar pocos hits
            if hits_count < 3:
                score *= 0.5
        return self.store.record_observation(topic, score, source="search_corpus")

    def observe_compile(self, topic: str, doc_count: int) -> TopicConfidence:
        """Registra confianza desde un compile_report."""
        t = topic.strip().lower()[:200]
        # Pocos docs → baja confianza
        if doc_count == 0:
            score = 0.0
        elif doc_count < 5:
            score = 0.3
        elif doc_count < 15:
            score = 0.6
        else:
            score = 0.85
        return self.store.record_observation(t, score, source="compile_report")

    def observe_research(self, query: str, ingested_count: int) -> TopicConfidence:
        """Registra confianza desde un research_topic completado."""
        t = query.strip().lower()[:200]
        # Si research ingirió docs, la confianza sube (ahora sabemos más)
        if ingested_count == 0:
            score = 0.2  # research no encontró nada útil
        elif ingested_count < 3:
            score = 0.5
        else:
            score = 0.8
        return self.store.record_observation(t, score, source="research_topic")


# ---------------------------------------------------------------------------
# ActiveResearchAgenda — propone investigar tópicos de baja confianza
# ---------------------------------------------------------------------------

class ActiveResearchAgenda:
    """Propone research para tópicos de baja confianza. Pending → gate humano."""

    def __init__(self, store: UncertaintyStore) -> None:
        self.store = store

    def scan_and_propose(self) -> list[ResearchProposal]:
        """Escanea tópicos de baja confianza y propone research. Returns new proposals."""
        low_confidence = self.store.list_low_confidence(limit=10)
        existing_topics = {p.topic for p in self.store.list_proposals(limit=200)
                          if p.status in ("pending", "approved")}
        new_proposals: list[ResearchProposal] = []
        for tc in low_confidence:
            if tc.topic in existing_topics:
                continue  # ya propuesta
            proposal = ResearchProposal(
                proposal_id=f"research_prop:{hashlib.sha256((tc.topic + _now()).encode()).hexdigest()[:16]}",
                topic=tc.topic, current_confidence=tc.confidence,
                rationale=(
                    f"Confianza baja en '{tc.topic}' ({tc.confidence:.2f} tras "
                    f"{tc.observation_count} observaciones). Investigar para mejorar cobertura."
                ),
                suggested_query=tc.topic,
                status="pending", proposed_at=_now(),
            )
            self.store.save_proposal(proposal)
            new_proposals.append(proposal)
            existing_topics.add(tc.topic)
        return new_proposals

    def approved_proposals(self) -> list[ResearchProposal]:
        """Propuestas aprobadas listas para ejecutar en idle."""
        return [p for p in self.store.list_proposals(status="approved", limit=20)
                if p.status == "approved"]


# ---------------------------------------------------------------------------
# System prompt injection
# ---------------------------------------------------------------------------

def render_uncertainty_context(store: UncertaintyStore, *, max_topics: int = 5) -> str:
    """Render low-confidence topics for the system prompt. Empty if none."""
    low = store.list_low_confidence(limit=max_topics)
    if not low:
        return ""
    lines = [f"- '{t.topic}' (confianza {t.confidence:.2f}, {t.observation_count} obs.)" for t in low]
    return (
        "Tópicos con baja confianza (considerá sugerir investigación al usuario):\n"
        + "\n".join(lines)
    )


__all__ = [
    "TopicConfidence", "ResearchProposal", "UncertaintyStore",
    "UncertaintyTracker", "ActiveResearchAgenda",
    "render_uncertainty_context",
    "DEFAULT_UNCERTAINTY_STORE", "CONFIDENCE_THRESHOLD", "MIN_OBSERVATIONS",
]
