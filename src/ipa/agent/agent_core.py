"""AgentCore: the omnipresent personal agent runtime (Fase 0 slice).

Owns identity, sessions and episodic memory. Surfaces (CLI, dashboard) are
thin clients that open sessions against this core; they never define
personality nor keep agent state (DEC-002).

Fase 0 scope: identity + sessions + episodic memory. Generation is injected
via an optional ``responder`` callable — when no provider is wired the core
returns a structured fallback, and the LLM-ready payload (``prepare_turn``)
still proves the identity behavior gate: changing the identity YAML changes
the built system prompt without touching code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .agent_identity import Identity, load_identity
from .agent_memory import AgentMemory, Episode

Responder = Callable[[list[dict[str, str]]], str]


@dataclass(frozen=True)
class TurnResult:
    session_id: str
    user_episode_id: str
    assistant_episode_id: str
    reply: str
    messages: list[dict[str, str]]


class AgentCore:
    """One agent, any surface. Opens durable sessions and records episodes."""

    def __init__(
        self,
        *,
        identity_path: str | Path | None = None,
        memory: AgentMemory | None = None,
        role: str = "general",
        interface: str = "cli",
        identity: Identity | None = None,
    ) -> None:
        self.identity = identity or load_identity(identity_path)
        self.memory = memory if memory is not None else AgentMemory()
        self.role = role
        self.interface = interface
        self.session_id: str | None = None

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def start_session(self, *, title: str | None = None, session_id: str | None = None) -> str:
        self.session_id = self.memory.open_session(
            interface=self.interface,
            role=self.role,
            identity_hash=self.identity.identity_hash,
            title=title,
            session_id=session_id,
        )
        return self.session_id

    def close_session(self) -> None:
        if self.session_id is not None:
            self.memory.close_session(self.session_id)
            self.session_id = None

    # ------------------------------------------------------------------
    # Turns
    # ------------------------------------------------------------------

    def build_messages(self, user_message: str, *, history_limit: int = 8) -> list[dict[str, str]]:
        """LLM-ready payload: identity system prompt + bounded recent history."""
        from datetime import datetime, timezone
        session = self.get_session()
        system = self.identity.system_prompt(role=self.role)
        # Inject current date so the model doesn't reject recent data as
        # "future" — it has no built-in sense of the real current date.
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        system += f"\n\nFecha actual del sistema: {today}. Los datos del corpus pueden ser recientes — no los rechaces por fecha."
        # Grounded responses: forbid answering knowledge questions from
        # parametric memory. Only corpus context or tool results count as
        # evidence; otherwise say there is no data and offer to research.
        system += (
            "\n\nREGLA DURA — grounding: para cualquier pregunta de conocimiento, "
            "los HECHOS que afirmes deben venir del contexto del corpus o de "
            "resultados de herramientas incluidos en este prompt, citados como [n]. "
            "PROHIBIDO presentar como hecho algo de tu conocimiento interno (puede "
            "estar desactualizado respecto del corpus). Permitido y bienvenido: "
            "sintetizar, comparar, razonar y especular sobre esos datos, siempre "
            "que la especulación esté marcada como tal (ej: 'esto es una "
            "inferencia', 'especulando'). Si el contexto no contiene la respuesta, "
            'decí explícitamente "no hay datos en el corpus sobre esto" y ofrecé '
            "lanzar una investigación web. Nunca rechaces datos del corpus por "
            "parecerte ficticios o futuros."
        )
        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        if session is not None:
            for episode in self.memory.get_episodes(session.session_id, limit=history_limit):
                messages.append({"role": episode.turn_role, "content": episode.content})
        messages.append({"role": "user", "content": user_message})
        return messages

    def submit(self, user_message: str, *, responder: Responder | None = None,
               topic_cluster_id: str | None = None) -> dict[str, Any]:
        """Build the identity-driven payload, record the user turn, generate
        (or fallback) and record the reply.

        ``topic_cluster_id`` links both episodes to a Fase 3 topic cluster when
        one exists (Fase 0 gate: nullable until Fase 3 wiring exists).
        """
        if not user_message.strip():
            raise ValueError("user_message must be a non-empty string")
        session_id = self.ensure_session()
        # Build messages BEFORE recording so history excludes the current turn.
        messages = self.build_messages(user_message, history_limit=8)
        self.memory.record_episode(
            session_id,
            turn_role="user",
            content=user_message,
            identity_hash=self.identity.identity_hash,
            topic_cluster_id=topic_cluster_id,
        )
        if responder is not None:
            reply = responder(messages)
        else:
            reply = (
                "[sin provider configurado] Turno registrado en la memoria del agente. "
                "La generación se conecta en fases posteriores (DEC-001 profile + Fase 1 tools)."
            )
        assistant = self.memory.record_episode(
            session_id,
            turn_role="assistant",
            content=reply,
            identity_hash=self.identity.identity_hash,
            topic_cluster_id=topic_cluster_id,
        )
        return {
            "session_id": session_id,
            "messages": messages,
            "reply": reply,
            "assistant_episode_id": assistant.episode_id,
        }

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------

    def ensure_session(self) -> str:
        if self.session_id is None:
            self.session_id = self.memory.open_session(
                interface=self.interface, role=self.role, identity_hash=self.identity.identity_hash,
            )
        elif self.get_session() is None:
            # Session ID provided but unknown in this store: start fresh.
            self.session_id = self.memory.open_session(
                interface=self.interface, role=self.role, identity_hash=self.identity.identity_hash,
            )
        elif self.get_session().status != "active":
            # Closed sessions are resumable: reopen instead of failing.
            self.memory.reopen_session(self.session_id)
        return self.session_id

    def get_session(self):
        return self.memory.get_session(self.session_id) if self.session_id else None

    def recall(self, limit: int = 10):
        """Recent episodes across sessions — the seed of cross-session memory."""
        return self.memory.recent_episodes(limit=limit)


__all__ = ["AgentCore"]
