"""Tutor contracts and capabilities (transitional facade)."""
from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "LearningGoal": ("tutor_contracts", "LearningGoal"),
    "Concept": ("tutor_contracts", "Concept"),
    "Roadmap": ("tutor_contracts", "Roadmap"),
    "AssessmentResult": ("tutor_contracts", "AssessmentResult"),
    "ResearchRequest": ("tutor_contracts", "ResearchRequest"),
}


def __getattr__(name: str):
    try:
        module_name, attr = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    return getattr(import_module(f"ipa.tutor.{module_name}"), attr)


__all__ = sorted(_EXPORTS)

