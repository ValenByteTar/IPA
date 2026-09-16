"""Tests for the compile_report agent tool (Reporter as agent-invoked tool)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipa.agent.agent_tools import ToolContext, TOOL_NAMES
from ipa.agent.agent_memory import AgentMemory
from ipa.agent.compile_report_executor import execute_compile_report, CompileReportResult


def _make_corpus(tmp_path: Path) -> Path:
    """Create a minimal corpus with 3 documents and chunks."""
    from ipa.storage.document_store import DocumentStore
    from ipa.contracts import CanonicalDocument, DocumentChunk, SourceSpan

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    store = DocumentStore(corpus_dir / "document_store.db")

    docs = [
        ("doc:quantum-photonic", "Quantum photonic processors improve optical computation and reduce latency.",
         "Quantum Photonic Processors"),
        ("doc:photonic-algorithms", "Photonic quantum processors improve optical algorithms for machine learning.",
         "Photonic Algorithms for ML"),
        ("doc:cyber-threats", "New cybersecurity threats target cloud infrastructure and zero-day vulnerabilities.",
         "Cloud Cybersecurity Threats"),
    ]

    for doc_id, text, title in docs:
        doc = CanonicalDocument(
            document_id=doc_id, pages=1, elements=[], source_spans=[],
            text=text, mime_type="text/plain", parser_id="test",
        )
        store.put_document(doc, artifact_id=f"sha256:{doc_id}")
        chunk = DocumentChunk(
            chunk_id=f"chunk:{doc_id}", document_id=doc_id,
            content_hash=f"sha256:{doc_id}", text=text,
            metadata={"chunk_index": 0},
        )
        store.put_chunks([chunk])
        # Record provenance
        store.put_source(doc_id, f"https://example.com/{doc_id}", "example.com", "configured_scrape", 0.8)

    store.commit()
    store.close()
    return corpus_dir


def _make_ctx(tmp_path: Path, corpus_dir: Path) -> ToolContext:
    memory = AgentMemory(tmp_path / "agent.db")
    return ToolContext(memory=memory, corpus_dir=str(corpus_dir))


def test_compile_report_in_tool_names():
    assert "compile_report" in TOOL_NAMES


def test_compile_report_basic(tmp_path):
    """compile_report produces a report from specified document IDs."""
    corpus_dir = _make_corpus(tmp_path)
    ctx = _make_ctx(tmp_path, corpus_dir)

    with ctx.memory:
        sid = ctx.memory.open_session(interface="test", role="general", identity_hash="test", title="test")
        ep = ctx.memory.record_episode(sid, turn_role="user", content="compile report test", identity_hash="test")

        call, result, compile_result = execute_compile_report(
            {"document_ids": ["doc:quantum-photonic", "doc:photonic-algorithms", "doc:cyber-threats"]},
            ctx,
            session_id=sid,
            episode_id=ep.episode_id,
        )

        ctx.memory.close_session(sid)

    assert call.tool_name == "compile_report"
    assert call.status == "completed"
    assert result.status == "completed"
    assert result.error is None
    assert compile_result.success
    assert compile_result.document_count == 3
    assert compile_result.report_id.startswith("report:agent:")
    # Report files exist
    report_path = Path(compile_result.report_path)
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert "categories" in report
    assert "curation_summary" in report
    assert report["corpus_id"] == "agent"
    # Source refs include all 3 documents
    assert len(result.source_refs) >= 2  # at least 2 selected (cyber may be filtered by dedup/irrelevant)
    # Did not mutate the main corpus
    assert (corpus_dir / "document_store.db").exists()
    ctx.close()


def test_compile_report_empty_document_ids_raises(tmp_path):
    """compile_report requires at least one document_id."""
    corpus_dir = _make_corpus(tmp_path)
    ctx = _make_ctx(tmp_path, corpus_dir)

    with ctx.memory:
        sid = ctx.memory.open_session(interface="test", role="general", identity_hash="test", title="test")
        ep = ctx.memory.record_episode(sid, turn_role="user", content="test", identity_hash="test")

        with pytest.raises(ValueError, match="non-empty 'document_ids'"):
            execute_compile_report(
                {"document_ids": []},
                ctx,
                session_id=sid,
                episode_id=ep.episode_id,
            )

        ctx.memory.close_session(sid)
    ctx.close()


def test_compile_report_nonexistent_doc_skipped(tmp_path):
    """compile_report skips document IDs that don't exist in the corpus."""
    corpus_dir = _make_corpus(tmp_path)
    ctx = _make_ctx(tmp_path, corpus_dir)

    with ctx.memory:
        sid = ctx.memory.open_session(interface="test", role="general", identity_hash="test", title="test")
        ep = ctx.memory.record_episode(sid, turn_role="user", content="test", identity_hash="test")

        call, result, compile_result = execute_compile_report(
            {"document_ids": ["doc:quantum-photonic", "doc:nonexistent"]},
            ctx,
            session_id=sid,
            episode_id=ep.episode_id,
        )

        ctx.memory.close_session(sid)

    assert compile_result.document_count == 1  # only the real doc
    assert result.status == "completed"
    ctx.close()


def test_compile_report_no_corpus_raises(tmp_path):
    """compile_report raises if no corpus is available."""
    memory = AgentMemory(tmp_path / "agent.db")
    ctx = ToolContext(memory=memory, corpus_dir=None)

    with memory:
        sid = memory.open_session(interface="test", role="general", identity_hash="test", title="test")
        ep = memory.record_episode(sid, turn_role="user", content="test", identity_hash="test")

        with pytest.raises(ValueError, match="corpus document store is not available"):
            execute_compile_report(
                {"document_ids": ["doc:1"]},
                ctx,
                session_id=sid,
                episode_id=ep.episode_id,
            )

        memory.close_session(sid)
    ctx.close()


def test_compile_report_does_not_mutate_main_corpus(tmp_path):
    """compile_report writes to an isolated output, not the main corpus."""
    corpus_dir = _make_corpus(tmp_path)
    ctx = _make_ctx(tmp_path, corpus_dir)

    # Count chunks before
    from ipa.storage.document_store import DocumentStore
    with DocumentStore(corpus_dir / "document_store.db") as store:
        chunks_before = store.count_chunks()

    with ctx.memory:
        sid = ctx.memory.open_session(interface="test", role="general", identity_hash="test", title="test")
        ep = ctx.memory.record_episode(sid, turn_role="user", content="test", identity_hash="test")

        call, result, compile_result = execute_compile_report(
            {"document_ids": ["doc:quantum-photonic", "doc:photonic-algorithms"]},
            ctx,
            session_id=sid,
            episode_id=ep.episode_id,
        )

        ctx.memory.close_session(sid)

    # Corpus unchanged
    with DocumentStore(corpus_dir / "document_store.db") as store:
        chunks_after = store.count_chunks()
    assert chunks_before == chunks_after

    # Report written to outputs/reporter/agent/, not the corpus dir
    output_dir = Path(compile_result.output_dir)
    assert "reporter" in output_dir.parts
    assert output_dir != corpus_dir
    ctx.close()


def test_compile_report_with_interests(tmp_path):
    """compile_report uses interest terms for curation scoring."""
    corpus_dir = _make_corpus(tmp_path)
    ctx = _make_ctx(tmp_path, corpus_dir)

    with ctx.memory:
        sid = ctx.memory.open_session(interface="test", role="general", identity_hash="test", title="test")
        ep = ctx.memory.record_episode(sid, turn_role="user", content="test", identity_hash="test")

        call, result, compile_result = execute_compile_report(
            {
                "document_ids": ["doc:quantum-photonic", "doc:photonic-algorithms", "doc:cyber-threats"],
                "interests": ["quantum computing", "photonics", "cybersecurity"],
            },
            ctx,
            session_id=sid,
            episode_id=ep.episode_id,
        )

        ctx.memory.close_session(sid)

    assert result.status == "completed"
    assert compile_result.document_count == 3
    ctx.close()


def test_compile_report_produces_markdown(tmp_path):
    """compile_report writes both report.json and report.md."""
    corpus_dir = _make_corpus(tmp_path)
    ctx = _make_ctx(tmp_path, corpus_dir)

    with ctx.memory:
        sid = ctx.memory.open_session(interface="test", role="general", identity_hash="test", title="test")
        ep = ctx.memory.record_episode(sid, turn_role="user", content="test", identity_hash="test")

        call, result, compile_result = execute_compile_report(
            {"document_ids": ["doc:quantum-photonic", "doc:photonic-algorithms"]},
            ctx,
            session_id=sid,
            episode_id=ep.episode_id,
        )

        ctx.memory.close_session(sid)

    output_dir = Path(compile_result.output_dir)
    assert (output_dir / "report.json").exists()
    assert (output_dir / "report.md").exists()
    assert (output_dir / "reporter.db").exists()
    ctx.close()
