from __future__ import annotations


def test_ipa_public_package_facade_resolves_core_types():
    import ipa

    assert ipa.__version__ == "0.1.0"
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
