"""Reporter bounded context (transitional facade)."""
from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "CorpusService": ("corpus_service", "CorpusService"),
    "ReporterConfig": ("reporter_config", "ReporterConfig"),
    "ReporterPipeline": ("reporter_pipeline", "ReporterPipeline"),
    "ReporterStore": ("reporter_store", "ReporterStore"),
    "ReporterDecision": ("reporter_contracts", "ReporterDecision"),
    "TopicLink": ("reporter_contracts", "TopicLink"),
    "discover_topics": ("reporter_topics", "discover_topics"),
    "match_topic_continuity": ("reporter_topics", "match_topic_continuity"),
}


def __getattr__(name: str):
    try:
        module_name, attr = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    return getattr(import_module(f"ipa.reporter.{module_name}"), attr)


__all__ = sorted(_EXPORTS)

