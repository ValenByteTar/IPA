"""Ingestion bounded context (transitional facade)."""
from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "FastPathRunner": ("fast_path", "FastPathRunner"),
    "FastPathResult": ("fast_path", "FastPathResult"),
    "LandingZone": ("landing_zone", "LandingZone"),
    "detect_mime": ("mime_router", "detect_mime"),
    "route_to_parser": ("mime_router", "route_to_parser"),
    "parse": ("parsers", "parse"),
    "chunk_document": ("chunker", "chunk_document"),
}


def __getattr__(name: str):
    try:
        module_name, attr = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    return getattr(import_module(f"ipa.ingestion.{module_name}"), attr)


__all__ = sorted(_EXPORTS)

