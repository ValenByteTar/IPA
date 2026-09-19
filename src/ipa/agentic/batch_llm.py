"""Ejecución de N generaciones en batch — explota el batching de ExL3.

Los tasks del idle (labels, clasificación de grises, veredictos de review)
llaman al LLM ítem por ítem: con ExL3 cargado eso es un batch de 1 y se pierde
el throughput agregado medido (81 tok/s a batch 3-4 vs 34 a batch 1, EXP-008).
Este helper agrupa los ítems y usa `generate_chat_batch` cuando el provider lo
soporta; con Ollama (que no lo tiene) cae a serial sin cambiar el contrato.

Contrato: mismas (text, error) por ítem, en orden, con errores por lote
aislados — un lote que falla no tumba a los demás.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence


def _result_pair(result: Any) -> tuple[str, str | None]:
    """Normaliza el retorno de un provider a (text, error).

    ExL3 devuelve GenerationResult (.text/.error); Ollama devuelve str.
    """
    if isinstance(result, str):
        return result, None
    text = getattr(result, "text", "") or ""
    error = getattr(result, "error", None)
    return text, (str(error) if error else None)


def supports_batch(provider: Any) -> bool:
    """True si el provider implementa generate_chat_batch (ExL3)."""
    return callable(getattr(provider, "generate_chat_batch", None))


def generate_many(
    provider: Any,
    conversations: Sequence[list[dict[str, str]]],
    *,
    max_new_tokens: int,
    batch_size: int | None = None,
    temperature: float | None = None,
    on_batch: Callable[[int, int], None] | None = None,
) -> list[tuple[str, str | None]]:
    """Ejecuta N conversaciones y devuelve [(text, error)] en el mismo orden.

    batch_size: tamaño de lote (default: el del provider, ej. 4 en ExL3).
    on_batch(done, total): callback de progreso (para logs/preempción).
    """
    items = list(conversations)
    if not items:
        return []
    out: list[tuple[str, str | None]] = [("", None)] * len(items)

    if not supports_batch(provider):
        # Serial (Ollama u otro provider sin batching).
        for i, msgs in enumerate(items):
            try:
                out[i] = _result_pair(provider.generate_chat(
                    msgs, max_new_tokens=max_new_tokens, temperature=temperature))
            except Exception as exc:
                out[i] = ("", str(exc)[:200])
            if on_batch:
                on_batch(i + 1, len(items))
        return out

    size = batch_size or int(getattr(provider, "batch_size", 1) or 1)
    size = max(1, size)
    for start in range(0, len(items), size):
        chunk = items[start:start + size]
        try:
            results = provider.generate_chat_batch(
                chunk, max_new_tokens=max_new_tokens, temperature=temperature)
            for j, res in enumerate(results):
                out[start + j] = _result_pair(res)
            # Si el provider devolvió menos resultados que prompts, el resto
            # queda con error explícito (no silencio).
            for j in range(len(results), len(chunk)):
                out[start + j] = ("", "batch devolvió menos resultados que prompts")
        except Exception as exc:
            err = str(exc)[:200]
            for j in range(len(chunk)):
                out[start + j] = ("", err)
        if on_batch:
            on_batch(min(start + size, len(items)), len(items))
    return out


__all__ = ["generate_many", "supports_batch"]
