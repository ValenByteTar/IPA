"""Punto 8: User model transversal — modela al usuario como persona.

Hoy el user model vive en tutor.db (UserTopicRecord con mastery por tópico)
y está acoplado al rol Tutor. Solo modela mastery pedagógica, no modela al
usuario transversalmente: goals, intereses, preferencias de comunicación,
estilo de reporte, conocimiento existente, historial de tareas.

Este UserModelStore es transversal: todos los roles (chat general, tutor,
research, reporter) lo leen para adaptar respuestas. El rol Tutor sigue
escribiendo mastery pedagógica en tutor.db (su estado específico), pero
lee user_model.db para contexto.

Dimensiones modeladas:
  - user_goals: goals activos declarados o inferidos (ej: "paper sobre fotónica")
  - user_interests: intereses declarados + inferidos por frecuencia de queries
  - user_preferences: preferencias de comunicación (longitud, estilo, citas)
  - user_knowledge: conocimiento existente (migrado/derivado de tutor mastery)
  - user_task_history: historial de tareas (derivado de TaskStore)

Inferencia determinística (sin VRAM, idle Level 1):
  - Intereses por frecuencia: contar topics en search_corpus queries +
    compile_report topics + research_topic queries
  - Preferencia de longitud: comparar longitud de respuesta vs feedback
  - Goals inferidos: si hay N research_topic + compile_report sobre el
    mismo tema en un período, inferir goal (propuesta, gate humano)

Inferencia LLM (opcional, idle Level 2):
  - Una generación sobre episodios resumidos propone goals/estilo
  - Se guarda como pending → aprobación humana

Invariante: las inferencias son propuestas pending → humano approve →
active. Solo las active se inyectan en el system prompt. Las declaradas
por el usuario (via tool o config) son active inmediatamente.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_USER_MODEL_STORE = Path("outputs/agent/user_model.db")

# Mínimo de ocurrencias para inferir un interés
MIN_INTEREST_OCCURRENCES = 3
# Mínimo de tareas sobre el mismo tema para inferir un goal
MIN_TASKS_FOR_GOAL = 2


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UserGoal:
    """Un goal activo del usuario (ej: 'paper sobre fotónica')."""
    goal_id: str
    description: str
    status: str  # "active" | "completed" | "abandoned"
    source: str  # "declared" | "inferred"
    confidence: float
    proposed_at: str
    decided_by: str | None = None
    decided_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UserInterest:
    """Un interés del usuario con score de frecuencia."""
    topic: str
    score: float  # 0.0-1.0, normalizado
    occurrence_count: int
    source: str  # "declared" | "inferred_search" | "inferred_research" | "inferred_compile"
    last_seen: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UserPreference:
    """Una preferencia de comunicación del usuario."""
    key: str  # "response_length" | "citation_style" | "language" | "report_style"
    value: str  # "short" | "long" | "with_citations" | "sections" | etc.
    source: str  # "declared" | "inferred"
    confidence: float
    last_updated: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# UserModelStore
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS user_goals (
    goal_id         TEXT PRIMARY KEY,
    description     TEXT NOT NULL,
    status          TEXT NOT NULL,
    source          TEXT NOT NULL,
    confidence      REAL NOT NULL,
    proposed_at     TEXT NOT NULL,
    decided_by      TEXT,
    decided_at      TEXT
);
CREATE TABLE IF NOT EXISTS user_interests (
    topic           TEXT PRIMARY KEY,
    score           REAL NOT NULL,
    occurrence_count INTEGER NOT NULL DEFAULT 0,
    source          TEXT NOT NULL,
    last_seen       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS user_preferences (
    key             TEXT PRIMARY KEY,
    value           TEXT NOT NULL,
    source          TEXT NOT NULL,
    confidence      REAL NOT NULL,
    last_updated    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS user_facts (
    fact_id         TEXT PRIMARY KEY,
    fact            TEXT NOT NULL,
    source          TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    proposed_at     TEXT NOT NULL,
    decided_by      TEXT,
    decided_at      TEXT
);
CREATE INDEX IF NOT EXISTS idx_goals_status ON user_goals(status);
CREATE INDEX IF NOT EXISTS idx_interests_score ON user_interests(score);
CREATE INDEX IF NOT EXISTS idx_facts_status ON user_facts(status);
"""


class UserModelStore:
    """SQLite persistence for the transversal user model."""

    def __init__(self, store_path: str | Path | None = None) -> None:
        if store_path is None:
            store_path = DEFAULT_USER_MODEL_STORE
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.store_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # --- Goals ---

    def add_goal(self, description: str, *, source: str = "declared",
                 confidence: float = 1.0, status: str = "active") -> UserGoal:
        goal = UserGoal(
            goal_id=f"goal:{hashlib.sha256((description + _now()).encode()).hexdigest()[:16]}",
            description=description.strip()[:500], status=status, source=source,
            confidence=max(0.0, min(1.0, confidence)), proposed_at=_now(),
        )
        self._conn.execute(
            "INSERT OR REPLACE INTO user_goals VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (goal.goal_id, goal.description, goal.status, goal.source,
             goal.confidence, goal.proposed_at, goal.decided_by, goal.decided_at),
        )
        self._conn.commit()
        return goal

    def list_goals(self, *, status: str | None = None, limit: int = 20) -> list[UserGoal]:
        if status:
            rows = self._conn.execute(
                "SELECT * FROM user_goals WHERE status = ? ORDER BY proposed_at DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM user_goals ORDER BY proposed_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_goal(row) for row in rows]

    def active_goals(self) -> list[UserGoal]:
        return self.list_goals(status="active", limit=10)

    def decide_goal(self, goal_id: str, *, approved: bool, decided_by: str = "human") -> None:
        g = next((g for g in self.list_goals(limit=200) if g.goal_id == goal_id), None)
        if g is None:
            raise ValueError(f"unknown goal: {goal_id}")
        new_status = "active" if approved else "abandoned"
        updated = UserGoal(
            goal_id=g.goal_id, description=g.description, status=new_status,
            source=g.source, confidence=g.confidence, proposed_at=g.proposed_at,
            decided_by=decided_by, decided_at=_now(),
        )
        self._conn.execute(
            "INSERT OR REPLACE INTO user_goals VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (updated.goal_id, updated.description, updated.status, updated.source,
             updated.confidence, updated.proposed_at, updated.decided_by, updated.decided_at),
        )
        self._conn.commit()

    # --- Interests ---

    def record_interest_observation(self, topic: str, *, source: str = "inferred_search") -> UserInterest:
        """Incrementa el contador de un interés. Lo crea si no existe."""
        import math
        t = topic.strip().lower()[:200]
        if not t:
            raise ValueError("topic must be non-empty")
        now = _now()
        existing = self._conn.execute(
            "SELECT * FROM user_interests WHERE topic = ?", (t,)
        ).fetchone()
        if existing:
            new_count = existing["occurrence_count"] + 1
            # Score normalizado: log(1 + count) / log(1 + 20) ≈ saturación en ~20 obs
            new_score = min(1.0, math.log(1 + new_count) / math.log(1 + 20))
            self._conn.execute(
                "UPDATE user_interests SET score=?, occurrence_count=?, source=?, last_seen=? WHERE topic=?",
                (new_score, new_count, source, now, t),
            )
            self._conn.commit()
            return UserInterest(topic=t, score=new_score, occurrence_count=new_count,
                                source=source, last_seen=now)
        else:
            initial_score = 1.0 / math.log(1 + 20)
            self._conn.execute(
                "INSERT INTO user_interests VALUES (?, ?, 1, ?, ?)",
                (t, initial_score, source, now),
            )
            self._conn.commit()
            return UserInterest(topic=t, score=initial_score,
                                occurrence_count=1, source=source, last_seen=now)

    def declare_interest(self, topic: str) -> UserInterest:
        """Declara un interés explícito (score alto inmediatamente)."""
        t = topic.strip().lower()[:200]
        now = _now()
        self._conn.execute(
            "INSERT OR REPLACE INTO user_interests VALUES (?, 1.0, "
            "COALESCE((SELECT occurrence_count FROM user_interests WHERE topic=?), 0) + 5, 'declared', ?)",
            (t, t, now),
        )
        self._conn.commit()
        return UserInterest(topic=t, score=1.0, occurrence_count=5,
                            source="declared", last_seen=now)

    def list_interests(self, *, limit: int = 20) -> list[UserInterest]:
        rows = self._conn.execute(
            "SELECT * FROM user_interests ORDER BY score DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._row_to_interest(row) for row in rows]

    def top_interests(self, *, limit: int = 5) -> list[UserInterest]:
        return self.list_interests(limit=limit)

    # --- Preferences ---

    def set_preference(self, key: str, value: str, *, source: str = "declared", confidence: float = 1.0) -> UserPreference:
        now = _now()
        pref = UserPreference(key=key.strip()[:50], value=value.strip()[:200],
                              source=source, confidence=confidence, last_updated=now)
        self._conn.execute(
            "INSERT OR REPLACE INTO user_preferences VALUES (?, ?, ?, ?, ?)",
            (pref.key, pref.value, pref.source, pref.confidence, pref.last_updated),
        )
        self._conn.commit()
        return pref

    def get_preference(self, key: str) -> UserPreference | None:
        row = self._conn.execute(
            "SELECT * FROM user_preferences WHERE key = ?", (key.strip(),)
        ).fetchone()
        return self._row_to_preference(row) if row else None

    def list_preferences(self) -> list[UserPreference]:
        rows = self._conn.execute("SELECT * FROM user_preferences").fetchall()
        return [self._row_to_preference(row) for row in rows]

    # --- Facts (durable user facts from session consolidation) ---

    def add_fact(self, fact: str, *, source: str = "session_consolidation", status: str = "pending") -> str:
        fact_id = f"fact:{hashlib.sha256((fact + _now()).encode()).hexdigest()[:16]}"
        self._conn.execute(
            "INSERT INTO user_facts VALUES (?, ?, ?, ?, ?, ?, ?)",
            (fact_id, fact.strip()[:500], source, status, _now(), None, None),
        )
        self._conn.commit()
        return fact_id

    def list_facts(self, *, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        if status:
            rows = self._conn.execute(
                "SELECT * FROM user_facts WHERE status = ? ORDER BY proposed_at DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM user_facts ORDER BY proposed_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def decide_fact(self, fact_id: str, *, approved: bool, decided_by: str = "human") -> None:
        new_status = "active" if approved else "rejected"
        self._conn.execute(
            "UPDATE user_facts SET status=?, decided_by=?, decided_at=? WHERE fact_id=?",
            (new_status, decided_by, _now(), fact_id),
        )
        self._conn.commit()

    def active_facts(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT fact FROM user_facts WHERE status = 'active' ORDER BY proposed_at DESC LIMIT 20"
        ).fetchall()
        return [row[0] for row in rows]

    # --- Close ---

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _row_to_goal(row: sqlite3.Row) -> UserGoal:
        return UserGoal(
            goal_id=row["goal_id"], description=row["description"], status=row["status"],
            source=row["source"], confidence=row["confidence"], proposed_at=row["proposed_at"],
            decided_by=row["decided_by"], decided_at=row["decided_at"],
        )

    @staticmethod
    def _row_to_interest(row: sqlite3.Row) -> UserInterest:
        return UserInterest(
            topic=row["topic"], score=row["score"], occurrence_count=row["occurrence_count"],
            source=row["source"], last_seen=row["last_seen"],
        )

    @staticmethod
    def _row_to_preference(row: sqlite3.Row) -> UserPreference:
        return UserPreference(
            key=row["key"], value=row["value"], source=row["source"],
            confidence=row["confidence"], last_updated=row["last_updated"],
        )


# ---------------------------------------------------------------------------
# UserModelInferer — inferencia determinística (idle Level 1, sin VRAM)
# ---------------------------------------------------------------------------

class UserModelInferer:
    """Infiere intereses, preferencias y goals desde el uso acumulado.

    Determinístico (sin VRAM). Corre en idle Level 1.
    """

    def __init__(self, store: UserModelStore) -> None:
        self.store = store

    def infer_interests_from_episodes(self, episodes: list[dict[str, Any]]) -> list[UserInterest]:
        """Cuenta topics en tool_calls de search_corpus/research_topic/compile_report."""
        from ipa.agent.system_tools import SYSTEM_TOOL_NAMES
        topic_counter: Counter[str] = Counter()
        for ep in episodes:
            calls = ep.get("tool_calls", [])
            if isinstance(calls, str):
                try:
                    calls = json.loads(calls)
                except ValueError:
                    calls = []
            # Los tool_calls son strings (nombres); buscar queries en el contenido
            content = ep.get("content", "")
            # Heurística: extraer texto entre comillas en episodios assistant
            if ep.get("turn_role") == "assistant":
                matches = re.findall(r"'([^']{3,60})'", content)
                for m in matches:
                    topic_counter[m.lower()] += 1
            # Buscar "[TOOL:search_corpus]{...query...}" en el contenido
            tool_matches = re.findall(r"\[TOOL:\w+\]\s*(\{.*?\})", content)
            for tm in tool_matches:
                try:
                    args = json.loads(tm)
                    query = args.get("query", "") or args.get("topic", "")
                    if query:
                        topic_counter[query.lower().strip()[:200]] += 2  # peso mayor
                except ValueError:
                    pass

        updated: list[UserInterest] = []
        for topic, count in topic_counter.most_common(30):
            if count < MIN_INTEREST_OCCURRENCES:
                continue
            interest = self.store.record_interest_observation(topic, source="inferred_search")
            updated.append(interest)
        return updated

    def infer_goals_from_tasks(self, tasks: list[dict[str, Any]]) -> list[UserGoal]:
        """Si hay N+ tareas sobre el mismo tema, propone un goal (pending)."""
        import unicodedata
        # Track both normalized (for counting) and original (for display)
        goal_counter: Counter[str] = Counter()
        topic_display: dict[str, str] = {}
        for task in tasks:
            goal = task.get("goal", "")
            if not goal:
                continue
            original = goal.strip().lower()
            # Normalizar: quitar acentos para counting
            normalized = unicodedata.normalize("NFD", original)
            ascii_only = "".join(c for c in normalized if unicodedata.category(c) != "Mn")
            topic = ascii_only
            for prefix in ("investiga ", "hace un reporte sobre ",
                           "hace un reporte de ", "busca ",
                           "investiga sobre "):
                if topic.startswith(prefix):
                    topic = topic[len(prefix):]
                    break
            # Tomar solo la PRIMERA palabra como tema (evita que
            # "fotónica" y "fotónica aplicada" cuenten como distintos)
            words = topic.split()[:1]
            topic_key = words[0][:200] if words else topic[:200]
            if topic_key:
                goal_counter[topic_key] += 1
                # Guardar la versión original (con acentos) para display
                if topic_key not in topic_display:
                    # Extraer el tema original del goal sin normalizar
                    orig_topic = original
                    for prefix in ("investigá ", "investiga ", "hacé un reporte sobre ",
                                   "hace un reporte sobre ", "buscá ", "busca ",
                                   "investigá sobre ", "investiga sobre "):
                        if orig_topic.startswith(prefix):
                            orig_topic = orig_topic[len(prefix):]
                            break
                    orig_words = orig_topic.split()[:1]
                    topic_display[topic_key] = orig_words[0] if orig_words else topic_key

        proposals: list[UserGoal] = []
        existing_goals = {g.description.lower() for g in self.store.list_goals(limit=50)}
        for topic_key, count in goal_counter.most_common(10):
            if count < MIN_TASKS_FOR_GOAL:
                continue
            display = topic_display.get(topic_key, topic_key)
            desc = f"Trabajo recurrente sobre: {display}"
            if desc.lower() in existing_goals:
                continue
            goal = self.store.add_goal(
                desc, source="inferred", confidence=min(1.0, count / 5.0),
                status="pending",
            )
            proposals.append(goal)
            existing_goals.add(desc.lower())
        return proposals

    def infer_response_length_preference(self, episodes: list[dict[str, Any]]) -> UserPreference | None:
        """Detecta si el usuario prefiere respuestas cortas o largas."""
        user_turns = [e for e in episodes if e.get("turn_role") == "user"]
        more_detail = sum(1 for u in user_turns
                          if "más detalle" in u.get("content", "").lower()
                          or "mas detalle" in u.get("content", "").lower())
        shorter = sum(1 for u in user_turns
                      if "más corto" in u.get("content", "").lower()
                      or "mas corto" in u.get("content", "").lower()
                      or "breve" in u.get("content", "").lower())
        if more_detail >= 3 and more_detail > shorter:
            return self.store.set_preference("response_length", "long", source="inferred", confidence=min(1.0, more_detail / 10))
        if shorter >= 3 and shorter > more_detail:
            return self.store.set_preference("response_length", "short", source="inferred", confidence=min(1.0, shorter / 10))
        return None


# ---------------------------------------------------------------------------
# System prompt injection
# ---------------------------------------------------------------------------

def render_user_model_context(store: UserModelStore, *, max_goals: int = 3, max_interests: int = 5) -> str:
    """Render user model for the system prompt. Empty if no data."""
    parts: list[str] = []
    goals = store.active_goals()[:max_goals]
    if goals:
        parts.append("Goals activos del usuario: " + "; ".join(g.description for g in goals))
    interests = store.top_interests(limit=max_interests)
    if interests:
        top = ", ".join(f"{i.topic} ({i.score:.2f})" for i in interests if i.score > 0.1)
        if top:
            parts.append(f"Intereses: {top}")
    prefs = store.list_preferences()
    for p in prefs:
        if p.key == "response_length":
            parts.append(f"Prefiere respuestas {p.value}")
        elif p.key == "citation_style":
            parts.append(f"Citas: {p.value}")
    facts = store.active_facts()[:5]
    if facts:
        parts.append("Contexto del usuario: " + "; ".join(facts))
    if not parts:
        return ""
    return "Perfil del usuario (adaptá profundidad, estilo y relevancia):\n" + "\n".join(parts)


__all__ = [
    "UserGoal", "UserInterest", "UserPreference", "UserModelStore",
    "UserModelInferer", "render_user_model_context",
    "DEFAULT_USER_MODEL_STORE", "MIN_INTEREST_OCCURRENCES", "MIN_TASKS_FOR_GOAL",
]
