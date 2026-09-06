"""Canonical storage bounded context (transitional facade)."""
from __future__ import annotations

from importlib import import_module

_EXPORTS = {"DocumentStore": ("document_store", "DocumentStore")}


def __getattr__(name: str):
    try:
        module_name, attr = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    return getattr(import_module(f"ipa.storage.{module_name}"), attr)


__all__ = sorted(_EXPORTS)

