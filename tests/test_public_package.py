from __future__ import annotations

import tomllib
from pathlib import Path


def test_ipa_public_package_facade_resolves_core_types():
    import ipa

    # La facade debe reportar la versión declarada en pyproject — comparar
    # contra el valor real y no un literal, para que el bump no rompa el test.
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    declared = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["version"]
    assert ipa.__version__ == declared
    assert ipa.FastPathRunner.__name__ == "FastPathRunner"
    assert ipa.DocumentStore.__name__ == "DocumentStore"


def test_ipa_bounded_contexts_resolve_migrated_implementations():
    from ipa.agentic import QueryIR
    from ipa.ingestion import FastPathRunner
    from ipa.indexes import TantivyIndex
    from ipa.observability import TraceLog
    from ipa.reporter import ReporterPipeline
    from ipa.storage import DocumentStore
    from ipa.tutor import LearningGoal

    assert QueryIR.__module__ == "ipa.agentic.agentic_contracts"
    assert FastPathRunner.__module__ == "ipa.ingestion.fast_path"
    assert TantivyIndex.__module__ == "ipa.indexes.tantivy_index"
    assert TraceLog.__module__ == "ipa.observability.trace_log"
    assert ReporterPipeline.__module__ == "ipa.reporter.reporter_pipeline"
    assert DocumentStore.__module__ == "ipa.storage.document_store"
    assert LearningGoal.__module__ == "ipa.tutor.tutor_contracts"
