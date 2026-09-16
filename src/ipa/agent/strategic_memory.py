"""Punto 4: Memoria estratégica — reflexión que extrae principios.

La memoria episódica (AgentMemory) graba qué pasó. La consolidación de
sesiones (SessionConsolidator) resume conversaciones. Pero falta una capa
que extraiga PRINCIPIOS durables del uso acumulado:

  - "cuando el usuario pregunta por X, suele querer Y"
  - "las búsquedas con query Z siempre dan ruido"
  - "el usuario prefiere respuestas cortas con citas"
  - "compile_report después de research_topic da mejores resultados
    que compile_report directo"

Estos principios son STATE del agente (no rebuildable desde el corpus),
viven en su propia store, y se proponen para aprobación humana (patrón
ConsolidationStore — nunca auto-aplicados).

Inferencia determinística (sin VRAM, corre en idle Level 1):
  - Contar patrones de tool_calls en episodios
  - Detectar queries que sistemáticamente dan pocos resultados
  - Detectar preferencias de longitud de respuesta

Inferencia LLM (opcional, idle Level 2):
  - Una generación sobre episodios resumidos propone principios abstractos
  - Se guarda como pending → aprobación humana

Invariante: los principios NUNCA se auto-aplican. El patrón es el mismo
que ConsolidationStore: propuesta pending → humano approve/reject → solo
los approved se inyectan en el system prompt.
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

DEFAULT_STRATEGIC_STORE = Path("outputs/agent/strategic_memory.db")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Principle:
    """Un principio estratégico extraído del uso. Pending → approve → active."""
    principle_id: str
    kind: str  # "tool_pattern" | "query_noise" | "response_style" | "workflow" | "llm_inferred"
    pattern: str  # descripción del patrón detectado
    principle: str  # el principio extraído (ej: "después de research_topic, siempre search_corpus antes de compile_report")
    evidence: dict[str, Any]  # datos que respaldan el principio (counts, examples)
    confidence: float  # 0.0-1.0, qué tan fuerte es la evidencia
    status: str  # "pending" | "approved" | "rejected" | "active"
    proposed_at: str
    decided_by: str | None = None
    decision_note: str | None = None
    decided_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# StrategicMemoryStore
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS strategic_principles (
    principle_id    TEXT PRIMARY KEY,
    kind            TEXT NOT NULL,
    pattern         TEXT NOT NULL,
    principle       TEXT NOT NULL,
    evidence_json   TEXT NOT NULL,
    confidence      REAL NOT NULL,
    status          TEXT NOT NULL,
    proposed_at     TEXT NOT NULL,
    decided_by      TEXT,
    decision_note   TEXT,
    decided_at      TEXT
);
CREATE INDEX IF NOT EXISTS idx_principles_status ON strategic_principles(status);
CREATE INDEX IF NOT EXISTS idx_principles_kind ON strategic_principles(kind);
"""


class StrategicMemoryStore:
    """SQLite persistence for strategic principles (append-only proposals)."""

    def __init__(self, store_path: str | Path | None = None) -> None:
        if store_path is None:
            store_path = DEFAULT_STRATEGIC_STORE
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.store_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def save_principle(self, principle: Principle) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO strategic_principles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (principle.principle_id, principle.kind, principle.pattern, principle.principle,
             json.dumps(principle.evidence, ensure_ascii=False), principle.confidence,
             principle.status, principle.proposed_at, principle.decided_by,
             principle.decision_note, principle.decided_at),
        )
        self._conn.commit()

    def get_principle(self, principle_id: str) -> Principle | None:
        row = self._conn.execute(
            "SELECT * FROM strategic_principles WHERE principle_id = ?", (principle_id,)
        ).fetchone()
        return self._row_to_principle(row) if row else None

    def list_principles(self, *, status: str | None = None, kind: str | None = None, limit: int = 50) -> list[Principle]:
        query = "SELECT * FROM strategic_principles"
        conditions: list[str] = []
        params: list[Any] = []
        if status:
            conditions.append("status = ?")
            params.append(status)
        if kind:
            conditions.append("kind = ?")
            params.append(kind)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY proposed_at DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_principle(row) for row in rows]

    def active_principles(self) -> list[Principle]:
        """Principios approved que se inyectan en el system prompt."""
        return self.list_principles(status="approved", limit=20)

    def decide(self, principle_id: str, *, approved: bool, decided_by: str = "human", note: str | None = None) -> None:
        p = self.get_principle(principle_id)
        if p is None:
            raise ValueError(f"unknown principle: {principle_id}")
        new_status = "approved" if approved else "rejected"
        updated = Principle(
            principle_id=p.principle_id, kind=p.kind, pattern=p.pattern,
            principle=p.principle, evidence=p.evidence, confidence=p.confidence,
            status=new_status, proposed_at=p.proposed_at, decided_by=decided_by,
            decision_note=note, decided_at=_now(),
        )
        self.save_principle(updated)

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _row_to_principle(row: sqlite3.Row) -> Principle:
        return Principle(
            principle_id=row["principle_id"], kind=row["kind"], pattern=row["pattern"],
            principle=row["principle"], evidence=json.loads(row["evidence_json"]),
            confidence=row["confidence"], status=row["status"],
            proposed_at=row["proposed_at"], decided_by=row["decided_by"],
            decision_note=row["decision_note"], decided_at=row["decided_at"],
        )


# ---------------------------------------------------------------------------
# StrategicReflector — extracción determinística + LLM opcional
# ---------------------------------------------------------------------------

class StrategicReflector:
    """Extrae principios del uso acumulado. Determinístico por defecto.

    Corre en idle (Level 1 — sin VRAM). Si hay provider, una generación
    LLM propone principios abstractos adicionales (pending, gate humano).
    """

    def __init__(self, store: StrategicMemoryStore) -> None:
        self.store = store

    def reflect(self, episodes: list[dict[str, Any]], *, provider: Any | None = None) -> list[Principle]:
        """Analiza episodios recientes y propone principios. Returns proposals."""
        proposals: list[Principle] = []
        # 1. Patrones de tool_calls (determinístico)
        proposals.extend(self._detect_tool_patterns(episodes))
        # 2. Queries ruidosas (determinístico)
        proposals.extend(self._detect_noisy_queries(episodes))
        # 3. Estilo de respuesta (determinístico)
        proposals.extend(self._detect_response_style(episodes))
        # 4. LLM opcional
        if provider is not None:
            proposals.extend(self._llm_reflect(episodes, provider))
        # Guardar solo proposals nuevas (dedupe por pattern)
        existing_patterns = {p.pattern for p in self.store.list_principles(limit=200)}
        new_proposals: list[Principle] = []
        for p in proposals:
            if p.pattern not in existing_patterns:
                self.store.save_principle(p)
                new_proposals.append(p)
                existing_patterns.add(p.pattern)
        return new_proposals

    def _detect_tool_patterns(self, episodes: list[dict[str, Any]]) -> list[Principle]:
        """Detecta secuencias de tools que se repiten."""
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

        # Contar secuencias de 2-3 tools
        seq_counter: Counter[tuple[str, ...]] = Counter()
        for seq in session_sequences.values():
            for i in range(len(seq) - 1):
                seq_counter[(seq[i], seq[i + 1])] += 1
            for i in range(len(seq) - 2):
                seq_counter[(seq[i], seq[i + 1], seq[i + 2])] += 1

        proposals: list[Principle] = []
        for seq, count in seq_counter.most_common(10):
            if count < 3:  # mínimo 3 ocurrencias
                continue
            pattern_str = " → ".join(seq)
            principle_text = f"Flujo frecuente ({count}x): {pattern_str}. Considerarlo como workflow por defecto."
            proposals.append(Principle(
                principle_id=f"principle:toolpat:{hashlib.sha256(pattern_str.encode()).hexdigest()[:16]}",
                kind="tool_pattern", pattern=pattern_str, principle=principle_text,
                evidence={"sequence": list(seq), "count": count},
                confidence=min(1.0, count / 10.0),
                status="pending", proposed_at=_now(),
            ))
        return proposals

    def _detect_noisy_queries(self, episodes: dict[str, Any] | list[dict[str, Any]]) -> list[Principle]:
        """Detecta queries de search_corpus que sistemáticamente dan pocos resultados."""
        # Buscar episodios assistant que contengan "0 resultados" o "sin resultados"
        proposals: list[Principle] = []
        query_failures: Counter[str] = Counter()
        for ep in episodes:
            if ep.get("turn_role") != "assistant":
                continue
            content = ep.get("content", "")
            # Heurística: si la respuesta menciona "0 resultados" o "sin resultados"
            if "0 resultados" in content.lower() or "sin resultados" in content.lower():
                # Buscar la query en el episodio user anterior (no tenemos acceso directo,
                # pero el contenido del assistant suele citar la query)
                # Heurística simple: extraer texto entre comillas
                matches = re.findall(r"'([^']{3,60})'", content)
                for m in matches:
                    query_failures[m] += 1
        for query, count in query_failures.most_common(5):
            if count < 2:
                continue
            proposals.append(Principle(
                principle_id=f"principle:noise:{hashlib.sha256(query.encode()).hexdigest()[:16]}",
                kind="query_noise", pattern=f"query '{query}' da 0 resultados",
                principle=f"La query '{query}' sistemáticamente no encuentra resultados. Considerar reformular o usar research_topic.",
                evidence={"query": query, "failure_count": count},
                confidence=min(1.0, count / 5.0),
                status="pending", proposed_at=_now(),
            ))
        return proposals

    def _detect_response_style(self, episodes: list[dict[str, Any]]) -> list[Principle]:
        """Detecta preferencias de longitud de respuesta."""
        # Contar si el usuario pide "más detalle" o "más corto" después de respuestas
        user_turns = [ep for ep in episodes if ep.get("turn_role") == "user"]
        assistant_turns = [ep for ep in episodes if ep.get("turn_role") == "assistant"]

        more_detail = sum(1 for u in user_turns if "más detalle" in u.get("content", "").lower() or "mas detalle" in u.get("content", "").lower())
        shorter = sum(1 for u in user_turns if "más corto" in u.get("content", "").lower() or "mas corto" in u.get("content", "").lower() or "breve" in u.get("content", "").lower())

        proposals: list[Principle] = []
        avg_len = sum(len(a.get("content", "")) for a in assistant_turns) / max(len(assistant_turns), 1)
        if more_detail >= 3 and more_detail > shorter:
            proposals.append(Principle(
                principle_id=f"principle:style:{hashlib.sha256(b'more_detail').hexdigest()[:16]}",
                kind="response_style", pattern="usuario pide más detalle frecuentemente",
                principle="El usuario prefiere respuestas más detalladas. Longitud promedio actual puede ser insuficiente.",
                evidence={"more_detail_requests": more_detail, "shorter_requests": shorter, "avg_response_len": avg_len},
                confidence=min(1.0, more_detail / 10.0),
                status="pending", proposed_at=_now(),
            ))
        elif shorter >= 3 and shorter > more_detail:
            proposals.append(Principle(
                principle_id=f"principle:style:{hashlib.sha256(b'shorter').hexdigest()[:16]}",
                kind="response_style", pattern="usuario pide respuestas más cortas frecuentemente",
                principle="El usuario prefiere respuestas breves. Longitud promedio actual puede ser excesiva.",
                evidence={"more_detail_requests": more_detail, "shorter_requests": shorter, "avg_response_len": avg_len},
                confidence=min(1.0, shorter / 10.0),
                status="pending", proposed_at=_now(),
            ))
        return proposals

    def _llm_reflect(self, episodes: list[dict[str, Any]], provider: Any) -> list[Principle]:
        """Una generación LLM propone principios abstractos. Pending → gate humano."""
        prompt = """Sos el módulo de reflexión estratégica de un agente personal. Analizá los episodios recientes y extraé PRINCIPIOS durables sobre el uso del agente. Respondé SOLO con JSON:

{"principles": [{"pattern": "patrón observado", "principle": "principio extraído", "confidence": 0.0-1.0}]}

Reglas:
- Solo principios ACCIONABLES (que cambien cómo responde el agente).
- Nada trivial ("el usuario habla español" no es principio).
- Si no hay principios claros, lista vacía.
- Sin markdown, sin texto fuera del JSON.

Episodios (resumen):
"""
        transcript = "\n".join(
            f"[{ep.get('turn_role', '?')}] {ep.get('content', '')[:300]}"
            for ep in episodes[-30:]
        )
        try:
            from .llm_text import generate_text
            text, error = generate_text(
                provider, [{"role": "user", "content": prompt + transcript[:8000]}],
                max_new_tokens=500,
            )
            if error or not text.strip():
                return []
            cleaned = re.sub(r"<\|im_start\|>|<\|im_end\|>", "", text).strip()
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if not match:
                return []
            data = json.loads(match.group(0))
            if not isinstance(data, dict):
                return []
            proposals: list[Principle] = []
            for p in data.get("principles", [])[:5]:
                if not isinstance(p, dict):
                    continue
                pattern = str(p.get("pattern", "")).strip()[:200]
                principle = str(p.get("principle", "")).strip()[:500]
                if not pattern or not principle:
                    continue
                confidence = float(p.get("confidence", 0.5))
                confidence = max(0.0, min(1.0, confidence))
                proposals.append(Principle(
                    principle_id=f"principle:llm:{hashlib.sha256((pattern + _now()).encode()).hexdigest()[:16]}",
                    kind="llm_inferred", pattern=pattern, principle=principle,
                    evidence={"source": "llm_reflection", "episode_count": len(episodes)},
                    confidence=confidence, status="pending", proposed_at=_now(),
                ))
            return proposals
        except Exception:
            return []


# ---------------------------------------------------------------------------
# System prompt injection
# ---------------------------------------------------------------------------

def render_active_principles(store: StrategicMemoryStore, *, max_principles: int = 8) -> str:
    """Render principles approved for the system prompt. Empty if none."""
    principles = store.active_principles()[:max_principles]
    if not principles:
        return ""
    lines = [f"- {p.principle}" for p in principles]
    return "Principios estratégicos aprendidos (aplicar en respuestas):\n" + "\n".join(lines)


__all__ = [
    "Principle", "StrategicMemoryStore", "StrategicReflector",
    "render_active_principles", "DEFAULT_STRATEGIC_STORE",
]
