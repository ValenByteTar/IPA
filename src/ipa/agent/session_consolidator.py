"""Automatic session consolidation (idle-time background worker).

When a session has been closed and idle for >= 5 minutes, the agent
summarizes it and routes the output by architectural role:

  - Session summary + title  → derived index on the session record
    (auto-applied: it is metadata ABOUT the conversation, rebuildable
    from the append-only episodes, never authority).
  - User facts ("Valen prefiere X", "trabaja en Y") → ConsolidationProposal
    (kind=memory_consolidation) → human approval gate per PAT-004.

Bounded: one session per cycle, max_new_tokens capped, skipped while a
chat stream is active (the generator is not thread-safe for concurrent
iterate() loops).
"""
from __future__ import annotations

import json
import re
from typing import Any

from .agent_memory import AgentMemory

_CONSOLIDATION_PROMPT = """Sos el módulo de consolidación de memoria de un agente personal.
Analizá la siguiente conversación y respondé SOLO con JSON válido:

{{"title": "título breve (max 6 palabras)", "summary": "resumen de la conversación (2-4 oraciones)", "key_topics": ["tema1", "tema2"], "user_facts": ["hecho durable sobre el usuario", "..."]}}

Reglas:
- "user_facts": SOLO hechos durables y accionables sobre el usuario (preferencias, proyectos, decisiones, contexto personal). Nada de trivialidades ni del contenido de la conversación en sí. Si no hay ninguno, lista vacía.
- "title": en el idioma de la conversación.
- Sin markdown, sin texto fuera del JSON.

Conversación:
"""


class SessionConsolidator:
    """Summarizes idle sessions; routes durable facts to the approval gate."""

    def __init__(self, memory: AgentMemory, provider: Any) -> None:
        self.memory = memory
        self.provider = provider

    def consolidate_session(self, session_id: str, *, max_episodes: int = 40) -> dict[str, Any] | None:
        """Summarize one session. Returns the parsed LLM payload or None."""
        session = self.memory.get_session(session_id)
        if session is None:
            return None
        episodes = self.memory.get_episodes(session_id, limit=max_episodes)
        if len(episodes) < 2:
            # Nada que consolidar: marcar para no reintentar eternamente.
            self.memory.update_session_summary(session_id, "(sesión trivial: menos de 2 turnos)")
            return None
        transcript = "\n".join(
            f"[{e.turn_role}] {e.content[:600]}" for e in episodes
        )
        prompt = _CONSOLIDATION_PROMPT + transcript[:12000]
        from .llm_text import generate_text
        # Contrato dual: providers que devuelven str (Ollama) u objetos con
        # .text/.error (ExL3 GenerationResult). Sin esto, un str hace que
        # getattr(result, "text", "") sea "" y toda consolidación muere acá.
        text, error = generate_text(self.provider, [{"role": "user", "content": prompt}], max_new_tokens=600)
        if error or not text.strip():
            return None
        payload = self._parse_json(text)
        if payload is None:
            return None
        # 1. Derived index: summary + title on the session (auto)
        summary = str(payload.get("summary", "")).strip()[:2000]
        title = str(payload.get("title", "")).strip()[:120]
        if summary:
            self.memory.update_session_summary(session_id, summary, title=title or None)
        # 2. Durable user facts → approval gate (never auto-applied)
        facts = [str(f).strip() for f in payload.get("user_facts", []) if str(f).strip()]
        proposals = []
        if facts:
            proposals = self._propose_facts(session_id, episodes, facts)
        return {"session_id": session_id, "title": title, "summary": summary,
                "key_topics": payload.get("key_topics", []), "fact_proposals": proposals}

    def _propose_facts(self, session_id: str, episodes: list, facts: list[str]) -> list[str]:
        from ipa.agentic.memory_consolidation import ConsolidationStore
        store = ConsolidationStore()
        try:
            proposal_ids = []
            episode_ids = [e.episode_id for e in episodes]
            for fact in facts[:5]:  # bounded: max 5 facts per session
                proposal_id = f"consolidation:sessfact:{abs(hash((session_id, fact))) % 10**16:016x}"
                from ipa.agentic.memory_consolidation import ConsolidationProposal
                from datetime import datetime, timezone
                proposal = ConsolidationProposal(
                    proposal_id=proposal_id,
                    kind="memory_consolidation",
                    topic_id=session_id,
                    summary=f"Hecho del usuario detectado en sesión: {fact}",
                    source_episode_ids=episode_ids[:10],
                    proposed_payload={
                        "fact": fact,
                        "session_id": session_id,
                        "origin": "session_consolidation",
                        "originals_preserved": True,
                    },
                    status="pending",
                    proposed_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                )
                store.save_proposal(proposal)
                proposal_ids.append(proposal_id)
            return proposal_ids
        finally:
            store.close()

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any] | None:
        cleaned = re.sub(r"</think>|<\|im_end\|>|<\|im_start\|>", "", text).strip()
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else None
        except ValueError:
            return None


def run_idle_consolidation(provider: Any, *, idle_minutes: float = 5, max_sessions: int = 2) -> list[dict[str, Any]]:
    """One consolidation cycle: summarize up to max_sessions idle sessions."""
    memory = AgentMemory()
    try:
        idle = memory.find_idle_sessions(idle_minutes=idle_minutes, limit=max_sessions)
        if not idle:
            return []
        consolidator = SessionConsolidator(memory, provider)
        results = []
        for session in idle:
            try:
                result = consolidator.consolidate_session(session.session_id)
                if result is not None:
                    results.append(result)
            except Exception:
                continue  # una sesión fallida no bloquea el ciclo
        return results
    finally:
        memory.close()
