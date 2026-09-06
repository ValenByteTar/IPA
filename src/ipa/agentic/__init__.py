"""Bounded agentic context and evidence contracts (transitional facade)."""
from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "QueryIR": ("agentic_contracts", "QueryIR"),
    "EvidenceHit": ("agentic_contracts", "EvidenceHit"),
    "EvidenceSet": ("agentic_contracts", "EvidenceSet"),
    "ContextPackage": ("agentic_contracts", "ContextPackage"),
    "ExecutionBudget": ("agentic_contracts", "ExecutionBudget"),
    "TopicInvestigationState": ("agentic_contracts", "TopicInvestigationState"),
    "ReporterPlanner": ("reporter_planner", "ReporterPlanner"),
    "ReporterRetriever": ("reporter_retrieval", "ReporterRetriever"),
    "ReporterContextBuilder": ("reporter_context", "ReporterContextBuilder"),
}


def __getattr__(name: str):
    try:
        module_name, attr = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    return getattr(import_module(f"ipa.agentic.{module_name}"), attr)


__all__ = sorted(_EXPORTS)

