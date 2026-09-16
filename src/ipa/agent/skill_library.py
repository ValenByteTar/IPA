"""Punto 5: Skill library dinámica — adquisición y composición de skills.

Las skills del identity YAML son estáticas (texto en el system prompt).
Un AGI debería poder ADQUIRIR skills nuevas del uso: "hice search_corpus +
filter + summarize 5 veces, hagamos eso una skill". Hoy no puede.

Esta skill library detecta patrones repetidos de composición de tools y
los propone como skills nuevas (pending → aprobación humana → active).
Las skills active se inyectan en el system prompt junto con las estáticas
del YAML.

Detección determinística (sin VRAM, idle Level 1):
  - Contar secuencias de tool_calls en episodios
  - Si una secuencia de N tools aparece >= MIN_OCCURRENCES veces,
    proponerla como skill
  - La skill tiene: name, description, steps (tool calls), trigger
    (cuándo usarla)

Composición:
  - Una skill puede invocarse como un "macro" — el executor descompone
    la skill en sus tool calls individuales
  - Por ahora, las skills son SUGERENCIAS al LLM (texto en el prompt),
    no ejecución automática. El LLM decide cuándo componer.

Invariante: las skills NUNCA se auto-aplican. Propuesta pending → humano
approve → active. Solo las active aparecen en el system prompt.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_SKILL_STORE = Path("outputs/agent/skill_library.db")

# Mínimo de ocurrencias para proponer una skill (evita falsos positivos)
MIN_OCCURRENCES = 3
# Mínimo de tools en una secuencia para que sea una skill (no un solo tool)
MIN_TOOLS_IN_SKILL = 2


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Skill:
    """Una skill aprendida: secuencia de tools que se repite. Pending → active."""
    skill_id: str
    name: str  # nombre corto, ej: "research_and_report"
    description: str  # qué hace, cuándo usarla
    steps: list[dict[str, Any]]  # [{action, args_template}] — args con {placeholders}
    trigger: str  # cuándo el LLM debería usarla
    occurrences: int  # cuántas veces se observó el patrón
    confidence: float  # 0.0-1.0
    status: str  # "pending" | "approved" | "rejected" | "active"
    proposed_at: str
    source: str = "pattern_detection"  # "pattern_detection" | "llm_proposed" | "manual"
    decided_by: str | None = None
    decided_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# SkillLibraryStore
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS skills (
    skill_id        TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    description     TEXT NOT NULL,
    steps_json      TEXT NOT NULL,
    trigger         TEXT NOT NULL,
    occurrences     INTEGER NOT NULL,
    confidence      REAL NOT NULL,
    status          TEXT NOT NULL,
    proposed_at     TEXT NOT NULL,
    source          TEXT NOT NULL DEFAULT 'pattern_detection',
    decided_by      TEXT,
    decided_at      TEXT
);
CREATE INDEX IF NOT EXISTS idx_skills_status ON skills(status);
"""


class SkillLibraryStore:
    """SQLite persistence for learned skills (append-only proposals)."""

    def __init__(self, store_path: str | Path | None = None) -> None:
        if store_path is None:
            store_path = DEFAULT_SKILL_STORE
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.store_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def save_skill(self, skill: Skill) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO skills VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (skill.skill_id, skill.name, skill.description,
             json.dumps(skill.steps, ensure_ascii=False), skill.trigger,
             skill.occurrences, skill.confidence, skill.status, skill.proposed_at,
             skill.source, skill.decided_by, skill.decided_at),
        )
        self._conn.commit()

    def get_skill(self, skill_id: str) -> Skill | None:
        row = self._conn.execute("SELECT * FROM skills WHERE skill_id = ?", (skill_id,)).fetchone()
        return self._row_to_skill(row) if row else None

    def list_skills(self, *, status: str | None = None, limit: int = 50) -> list[Skill]:
        if status:
            rows = self._conn.execute(
                "SELECT * FROM skills WHERE status = ? ORDER BY proposed_at DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM skills ORDER BY proposed_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_skill(row) for row in rows]

    def active_skills(self) -> list[Skill]:
        """Skills approved que se inyectan en el system prompt."""
        return self.list_skills(status="approved", limit=20)

    def decide(self, skill_id: str, *, approved: bool, decided_by: str = "human") -> None:
        s = self.get_skill(skill_id)
        if s is None:
            raise ValueError(f"unknown skill: {skill_id}")
        new_status = "approved" if approved else "rejected"
        updated = Skill(
            skill_id=s.skill_id, name=s.name, description=s.description,
            steps=s.steps, trigger=s.trigger, occurrences=s.occurrences,
            confidence=s.confidence, status=new_status, proposed_at=s.proposed_at,
            source=s.source, decided_by=decided_by, decided_at=_now(),
        )
        self.save_skill(updated)

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _row_to_skill(row: sqlite3.Row) -> Skill:
        return Skill(
            skill_id=row["skill_id"], name=row["name"], description=row["description"],
            steps=json.loads(row["steps_json"]), trigger=row["trigger"],
            occurrences=row["occurrences"], confidence=row["confidence"],
            status=row["status"], proposed_at=row["proposed_at"],
            source=row["source"], decided_by=row["decided_by"],
            decided_at=row["decided_at"],
        )


# ---------------------------------------------------------------------------
# SkillDetector — detección determinística de patrones
# ---------------------------------------------------------------------------

class SkillDetector:
    """Detecta secuencias de tools repetidas y propone skills.

    Determinístico (sin VRAM). Corre en idle Level 1.
    """

    def __init__(self, store: SkillLibraryStore) -> None:
        self.store = store

    def detect(self, episodes: list[dict[str, Any]]) -> list[Skill]:
        """Analiza tool_calls en episodios y propone skills para patrones repetidos."""
        # Extraer secuencias de tool_calls por sesión
        session_sequences: dict[str, list[str]] = {}
        for ep in episodes:
            sid = ep.get("session_id", "")
            calls = ep.get("tool_calls", [])
            if isinstance(calls, str):
                try:
                    calls = json.loads(calls)
                except ValueError:
                    calls = []
            if calls:
                session_sequences.setdefault(sid, []).extend(calls)

        # Contar secuencias contiguas de 2-4 tools
        seq_counter: Counter[tuple[str, ...]] = Counter()
        for seq in session_sequences.values():
            for window_size in range(MIN_TOOLS_IN_SKILL, min(5, len(seq) + 1)):
                for i in range(len(seq) - window_size + 1):
                    seq_counter[tuple(seq[i:i + window_size])] += 1

        # Filtrar secuencias con suficientes ocurrencias
        existing_signatures = {self._signature(s.steps) for s in self.store.list_skills(limit=200)}
        proposals: list[Skill] = []
        for seq_tuple, count in seq_counter.most_common(20):
            if count < MIN_OCCURRENCES:
                continue
            seq_list = list(seq_tuple)
            sig = self._signature([{"action": t} for t in seq_list])
            if sig in existing_signatures:
                continue  # ya propuesta o decidida
            skill = self._build_skill(seq_list, count)
            if skill is not None:
                self.store.save_skill(skill)
                proposals.append(skill)
                existing_signatures.add(sig)
        return proposals

    @staticmethod
    def _signature(steps: list[dict[str, Any]]) -> str:
        actions = [s.get("action", s.get("action", "")) for s in steps]
        return hashlib.sha256("|".join(actions).encode()).hexdigest()[:16]

    @staticmethod
    def _build_skill(seq: list[str], occurrences: int) -> Skill | None:
        """Construye una Skill propuesta desde una secuencia de tool names."""
        if len(seq) < MIN_TOOLS_IN_SKILL:
            return None
        # Nombre: concatenar actions
        name = "_".join(seq[:3]) if len(seq) <= 3 else f"{seq[0]}_to_{seq[-1]}"
        # Descripción genérica
        steps_desc = " → ".join(seq)
        description = f"Flujo de {len(seq)} herramientas: {steps_desc}."
        trigger = f"Usar cuando el usuario pida algo que involucre: {', '.join(seq[:2])}."
        steps = [{"action": t, "args_template": {}} for t in seq]
        confidence = min(1.0, occurrences / 10.0)
        return Skill(
            skill_id=f"skill:{hashlib.sha256(("|".join(seq) + _now()).encode()).hexdigest()[:16]}",
            name=name, description=description, steps=steps, trigger=trigger,
            occurrences=occurrences, confidence=confidence,
            status="pending", proposed_at=_now(),
        )


# ---------------------------------------------------------------------------
# System prompt injection
# ---------------------------------------------------------------------------

def render_active_skills(store: SkillLibraryStore, *, max_skills: int = 6) -> str:
    """Render skills approved for the system prompt. Empty if none."""
    skills = store.active_skills()[:max_skills]
    if not skills:
        return ""
    lines = [f"- {s.name}: {s.description} Trigger: {s.trigger}" for s in skills]
    return "Skills aprendidas (flujos frecuentes, considerar cuando apliquen):\n" + "\n".join(lines)


__all__ = [
    "Skill", "SkillLibraryStore", "SkillDetector",
    "render_active_skills", "DEFAULT_SKILL_STORE",
    "MIN_OCCURRENCES", "MIN_TOOLS_IN_SKILL",
]
