"""Dynamic system-state layer for the agent system prompt.

Renders a compact summary of what the agent ACTUALLY has: corpus size,
user model contents, tutor state, tasks. The reactive 9B cannot infer
its own capabilities from tool names alone — it claimed to lack a
pedagogy engine while the Tutor runtime was fully operational. This
layer grounds self-knowledge in real state.

Each store read is fail-safe: a missing/corrupt store renders as no
data, never breaks the prompt.
"""
from __future__ import annotations

from pathlib import Path

_AGENT_DIR = Path("outputs") / "agent"
_MAIN_CORPUS = Path("outputs") / "experiments" / "E12-corpus"


def _count(db_path: Path, sql: str, params: tuple = ()) -> int | None:
    """Run a COUNT query; None if the DB/table is unavailable."""
    if not db_path.exists():
        return None
    try:
        import sqlite3
        conn = sqlite3.connect(str(db_path), timeout=5.0)
        try:
            row = conn.execute(sql, params).fetchone()
            return int(row[0]) if row else 0
        finally:
            conn.close()
    except Exception:
        return None


def render_system_state() -> str:
    """Render the dynamic system-state layer. Empty string if nothing to show."""
    lines: list[str] = []

    # Main corpus (the knowledge base the agent searches)
    docs = _count(_MAIN_CORPUS / "document_store.db",
                  "SELECT COUNT(*) FROM documents WHERE tombstoned = 0")
    if docs:
        lines.append(f"- Corpus principal: {docs} documentos indexados y consultables (search_corpus).")

    # User model (what the agent knows about Valen)
    goals = _count(_AGENT_DIR / "user_model.db",
                   "SELECT COUNT(*) FROM user_goals WHERE status = 'active'")
    interests = _count(_AGENT_DIR / "user_model.db",
                       "SELECT COUNT(*) FROM user_interests")
    if goals or interests:
        parts = []
        if goals:
            parts.append(f"{goals} metas activas")
        if interests:
            parts.append(f"{interests} intereses")
        lines.append(f"- Modelo de usuario: {', '.join(parts)} (get_user_profile).")

    # Tutor runtime (pedagogy engine — EXISTS, with real state)
    roadmaps = _count(_AGENT_DIR / "tutor.db",
                      "SELECT COUNT(*) FROM roadmaps")
    records = _count(_AGENT_DIR / "tutor.db",
                     "SELECT COUNT(*) FROM user_topic_records")
    if roadmaps or records:
        parts = []
        if roadmaps:
            parts.append(f"{roadmaps} roadmaps")
        if records:
            parts.append(f"{records} tópicos con seguimiento de mastery")
        lines.append(f"- Tutor pedagógico operativo: {', '.join(parts)} (rol tutor).")

    # Task planner (multi-step work)
    tasks = _count(_AGENT_DIR / "task_store.db",
                   "SELECT COUNT(*) FROM tasks")
    if tasks:
        lines.append(f"- Planificador de tareas: {tasks} tareas persistidas (plan_task, resume_task).")

    # Strategic memory / skills / uncertainty (cognitive layer state)
    principles = _count(_AGENT_DIR / "strategic_memory.db",
                        "SELECT COUNT(*) FROM strategic_principles WHERE status = 'active'")
    skills = _count(_AGENT_DIR / "skill_library.db",
                    "SELECT COUNT(*) FROM skills WHERE status = 'approved'")
    if principles:
        lines.append(f"- Memoria estratégica: {principles} principios activos.")
    if skills:
        lines.append(f"- Skills aprendidas: {skills} aprobadas.")

    if not lines:
        return ""
    return ("Estado real del sistema (usalo para saber qué sabés hacer — "
            "no inventes capacidades faltantes ni pidas permiso para usar lo que ya tenés):\n"
            + "\n".join(lines))


__all__ = ["render_system_state"]
