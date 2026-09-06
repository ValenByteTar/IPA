"""Model providers (transitional facade)."""
from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "ExL3Provider": ("exl3_provider", "ExL3Provider"),
    "GenerationResult": ("exl3_provider", "GenerationResult"),
    "create_star_provider": ("exl3_provider", "create_star_provider"),
}


def __getattr__(name: str):
    try:
        module_name, attr = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    return getattr(import_module(f"ipa.providers.{module_name}"), attr)


__all__ = sorted(_EXPORTS)

