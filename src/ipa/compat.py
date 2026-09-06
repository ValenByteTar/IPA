"""Compatibility helpers for the gradual res023_lab -> ipa migration."""
from __future__ import annotations

from importlib import import_module
from typing import Any


def legacy_module(name: str) -> Any:
    """Resolve a legacy module without copying its implementation."""
    if not name or name.startswith("."):
        raise ValueError("legacy module name must be absolute and non-empty")
    return import_module(f"res023_lab.{name}")

