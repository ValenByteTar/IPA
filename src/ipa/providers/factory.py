"""Fábrica del provider estrella con fallback CPU-only.

Regla del proyecto: si no hay GPU, el sistema funciona 100% por CPU.
ExLlamaV3 es un motor de inferencia GPU — no tiene modo CPU utilizable —
así que la fábrica cae automáticamente a Ollama (que corre en CPU) cuando
`has_gpu()` es False y el provider configurado era ExL3.

Selección:
  - IPA_LLM_PROVIDER=ollama (default) → Ollama (GGUF).
  - IPA_LLM_PROVIDER=exl3 → ExL3, PERO solo si hay GPU; si no, fallback
    a Ollama con warning (nunca falla el boot por falta de GPU).
"""
from __future__ import annotations

import os
from typing import Any

from .device import has_gpu

DEFAULT_OLLAMA_MODEL = "qwen3.5:9b-q4_K_M"


def create_star_provider(interactive: bool = False) -> Any:
    """Provider estrella activo según IPA_LLM_PROVIDER, con fallback CPU."""
    provider_type = os.environ.get("IPA_LLM_PROVIDER", "ollama").strip().lower()
    if provider_type != "ollama" and not has_gpu():
        print(
            f"[factory] IPA_LLM_PROVIDER={provider_type} requiere GPU y no hay "
            "GPU disponible — fallback a Ollama (CPU).",
            flush=True,
        )
        provider_type = "ollama"
    if provider_type == "ollama":
        from .ollama_provider import create_star_provider as _ollama

        model = os.environ.get("IPA_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        return _ollama(model=model)
    from .exl3_provider import create_star_provider as _exl3

    return _exl3(interactive=interactive)


__all__ = ["create_star_provider", "DEFAULT_OLLAMA_MODEL"]
