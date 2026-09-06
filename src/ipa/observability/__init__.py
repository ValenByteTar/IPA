"""Cross-cutting observability (transitional facade)."""
from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "TraceLog": ("trace_log", "TraceLog"),
    "TraceEvent": ("trace_log", "TraceEvent"),
    "make_event": ("trace_log", "make_event"),
}


def __getattr__(name: str):
    try:
        module_name, attr = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    return getattr(import_module(f"ipa.observability.{module_name}"), attr)


__all__ = sorted(_EXPORTS)

