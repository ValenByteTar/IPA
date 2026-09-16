"""Provider wiring for the agent core (Fase 2 bridge).

Connects a chat provider (ExL3Provider or compatible) to the agent:

  - ``build_responder(provider)``  → responder callable for AgentCore.submit()
  - ``build_llm_judge(provider)``  → LLMJudge for the agentic research flow

The provider is injected, never imported by the core: surfaces (CLI) decide
whether an LLM is attached. Without a provider the agent remains fully
functional with deterministic fallbacks (DEC-002).
"""
from __future__ import annotations

from typing import Any, Callable

from .judge import LLMJudge

Messages = list[dict[str, str]]
Responder = Callable[[Messages], str]


def build_responder(provider: Any, *, max_new_tokens: int = 512) -> Responder:
    """Build an AgentCore responder callable from a chat provider.

    The provider must expose ``generate_chat(messages, max_new_tokens=...,
    temperature=...)`` returning an object with ``.text`` and ``.error``
    (ExL3Provider satisfies this) or a plain string (OllamaProvider).
    The provider must already be loaded.
    """
    def responder(messages: Messages) -> str:
        result = provider.generate_chat(messages, max_new_tokens=max_new_tokens)
        if isinstance(result, str):
            return result
        if getattr(result, "error", None):
            return f"[provider error] {result.error}"
        return result.text

    return responder


def build_streaming_responder(
    provider: Any,
    *,
    max_new_tokens: int = 512,
    on_token: Callable[[str], None] | None = None,
) -> Responder:
    """Like build_responder but streams each chunk through ``on_token``.

    Uses ``provider.generate_chat_stream(messages, max_new_tokens=...)``
    (chunk shape ``{"text": str, "done": bool, "error"?: str}``) when the
    provider exposes it; otherwise falls back to ``generate_chat`` and
    emits the whole reply once. Either way returns the full text.
    """
    def responder(messages: Messages) -> str:
        stream = getattr(provider, "generate_chat_stream", None)
        if not callable(stream):
            text = build_responder(
                provider, max_new_tokens=max_new_tokens)(messages)
            if on_token is not None and text:
                on_token(text)
            return text
        parts: list[str] = []
        for chunk in stream(messages, max_new_tokens=max_new_tokens):
            if isinstance(chunk, str):
                text = chunk
            else:
                if chunk.get("error"):
                    return f"[provider error] {chunk['error']}"
                text = chunk.get("text", "")
            if text:
                parts.append(text)
                if on_token is not None:
                    on_token(text)
        return "".join(parts)

    return responder


def build_llm_judge(provider: Any, **kwargs: Any) -> LLMJudge:
    """Build an LLMJudge from a loaded chat provider."""
    return LLMJudge(provider, **kwargs)


__all__ = ["build_responder", "build_streaming_responder", "build_llm_judge"]
