"""Normalización del contrato de los providers de chat.

Dos familias conviven en el proyecto:

  - ExL3 (`GenerationResult`): objeto con `.text` / `.error`.
  - Ollama: devuelve el texto como `str` plano.

Todo módulo que consuma una generación fuera del chat debe pasar por acá —
si no, un provider `str` hace que `getattr(result, "text", "")` sea `""` y
la inferencia muere en silencio (bug real en la consolidación de sesiones).
"""
from __future__ import annotations

from typing import Any


def generate_text(
    provider: Any,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
) -> tuple[str, str | None]:
    """Run one chat generation. Returns (text, error)."""
    result = provider.generate_chat(
        messages, max_new_tokens=max_new_tokens, temperature=temperature,
    )
    if isinstance(result, str):
        return result, None
    text = getattr(result, "text", "") or ""
    error = getattr(result, "error", None)
    return text, (str(error) if error else None)


__all__ = ["generate_text"]
