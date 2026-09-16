"""Capa cognitiva en idle — Tier 1 determinístico y Tier 2 con LLM.

Wirea los cuatro módulos de la capa cognitiva (Fase 4) al ciclo idle del
dashboard. Existían desde el diseño original pero ningún worker los
ejecutaba: intereses/goals del usuario, detección de skills repetidas,
principios estratégicos y agenda de investigación de tópicos de baja
confianza.

Reglas de la arquitectura:
  - Tier 1: determinístico, sin VRAM. Corre en cualquier idle.
  - Tier 2: usa el LLM (solo si ya está cargado, o en idle profundo).
    Nunca levanta el modelo por sí mismo.
  - Todo lo que sale de acá es PROPUESTA (`pending`): el gate humano del
    panel de Aprobaciones decide. Nada se aplica solo.
  - Los originales (episodios, tareas) nunca se mutan — es inferencia
    derivada y rebuildable.
"""
from __future__ import annotations

from typing import Any


def _episode_dicts(memory: Any, limit: int = 200) -> list[dict[str, Any]]:
    return [
        {
            "session_id": e.session_id,
            "turn_role": e.turn_role,
            "content": e.content,
            "tool_calls": list(e.tool_calls or []),
        }
        for e in memory.recent_episodes(limit=limit)
    ]


def _task_dicts(limit: int = 50) -> list[dict[str, Any]]:
    from ipa.agent.task_planner import TaskStore
    store = TaskStore()
    try:
        return [{"goal": t.goal} for t in store.list_tasks(limit=limit)]
    finally:
        store.close()


def load_episode_dicts(limit: int = 200) -> list[dict[str, Any]]:
    """Episodios recientes como dicts — se carga UNA vez por ciclo idle y se
    comparte entre las tareas cognitivas (así no compiten por agent_db)."""
    from ipa.agent.agent_memory import AgentMemory
    memory = AgentMemory()
    try:
        return _episode_dicts(memory, limit=limit)
    finally:
        memory.close()


def load_task_dicts(limit: int = 50) -> list[dict[str, Any]]:
    try:
        return _task_dicts(limit=limit)
    except Exception:
        return []


# ── Tareas cognitivas (una por store → paralelizables entre sí) ──────────

def infer_user_model(
    episodes: list[dict[str, Any]],
    tasks: list[dict[str, Any]] | None = None,
) -> dict[str, int]:
    """Intereses + goals recurrentes + preferencia de longitud (determinístico)."""
    from ipa.agent.user_model import UserModelInferer, UserModelStore

    um = UserModelStore()
    try:
        inferer = UserModelInferer(um)
        return {
            "interests": len(inferer.infer_interests_from_episodes(episodes)),
            "goals": len(inferer.infer_goals_from_tasks(tasks or [])),
            "preferences": 1 if inferer.infer_response_length_preference(episodes) else 0,
        }
    finally:
        um.close()


def detect_skills(episodes: list[dict[str, Any]]) -> dict[str, int]:
    """Secuencias de tools repetidas → propuestas de skill (pending)."""
    from ipa.agent.skill_library import SkillDetector, SkillLibraryStore

    store = SkillLibraryStore()
    try:
        return {"skills": len(SkillDetector(store).detect(episodes))}
    finally:
        store.close()


def reflect_principles(
    episodes: list[dict[str, Any]],
    *,
    provider: Any | None = None,
) -> dict[str, int]:
    """Principios estratégicos. Sin provider: determinístico (Tier 1).
    Con provider: agrega principios LLM (Tier 2)."""
    from ipa.agent.strategic_memory import StrategicMemoryStore, StrategicReflector

    store = StrategicMemoryStore()
    try:
        principles = StrategicReflector(store).reflect(episodes, provider=provider)
        llm = [p for p in principles if p.kind == "llm_inferred"]
        return {
            "principles": len(principles) - len(llm),
            "llm_principles": len(llm),
        }
    finally:
        store.close()


def scan_research_agenda() -> dict[str, int]:
    """Tópicos de baja confianza → propuestas de investigación (pending)."""
    from ipa.agent.uncertainty import ActiveResearchAgenda, UncertaintyStore

    store = UncertaintyStore()
    try:
        return {"research_proposals": len(ActiveResearchAgenda(store).scan_and_propose())}
    finally:
        store.close()


def run_cognitive_layer1(
    *,
    episodes: list[dict[str, Any]] | None = None,
    tasks: list[dict[str, Any]] | None = None,
) -> dict[str, int]:
    """Tier 1 completo en un solo llamado (conveniencia; el scheduler las
    registra por separado para paralelizarlas)."""
    if episodes is None:
        episodes = load_episode_dicts()
    if tasks is None:
        tasks = load_task_dicts()

    result = {
        "interests": 0, "goals": 0, "preferences": 0,
        "skills": 0, "principles": 0, "research_proposals": 0,
    }
    result.update(infer_user_model(episodes, tasks))
    result.update(detect_skills(episodes))
    result.update(reflect_principles(episodes))
    result.update(scan_research_agenda())
    return result


def run_cognitive_layer2(
    provider: Any,
    *,
    episodes: list[dict[str, Any]] | None = None,
) -> dict[str, int]:
    """Tier 2: reflexión con LLM (principios abstractos, pending → gate)."""
    if episodes is None:
        episodes = load_episode_dicts(limit=80)
    return reflect_principles(episodes, provider=provider)


__all__ = [
    "load_episode_dicts", "load_task_dicts",
    "infer_user_model", "detect_skills", "reflect_principles", "scan_research_agenda",
    "run_cognitive_layer1", "run_cognitive_layer2",
]
