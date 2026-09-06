"""Derived enrichment adapters (transitional facade)."""
from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "enrich_summary": ("enrichment", "enrich_summary"),
    "OllamaAdapter": ("ollama_adapter", "OllamaAdapter"),
}


def __getattr__(name: str):
    try:
        module_name, attr = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    return getattr(import_module(f"ipa.enrichment.{module_name}"), attr)


__all__ = sorted(_EXPORTS)

