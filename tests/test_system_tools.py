"""Tests for the unified system-tool registry, catalog and marker parsing."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipa.agent import system_tools
from ipa.agent.system_tools import (
    SYSTEM_TOOL_NAMES,
    TOOL_CATALOG,
    SystemToolResult,
    execute_system_tool,
    parse_tool_marker,
)


# ---------------------------------------------------------------------------
# Registry consistency
# ---------------------------------------------------------------------------

def test_registry_names_match_specs():
    spec_names = {spec.name for spec in system_tools._SYSTEM_TOOLS}
    assert spec_names == SYSTEM_TOOL_NAMES


def test_every_spec_has_implementation():
    for spec in system_tools._SYSTEM_TOOLS:
        assert callable(spec.fn)
        assert system_tools._SYSTEM_IMPLEMENTATIONS[spec.name] is spec.fn


def test_catalog_lists_only_chat_visible_tools():
    # Progressive unlocking: TOOL_CATALOG (legacy) includes all unlockable tools.
    # Hidden tools (chat_visible=False) must not appear.
    for spec in system_tools._SYSTEM_TOOLS:
        line = f"- {spec.name}:"
        if spec.chat_visible:
            assert line in TOOL_CATALOG, f"{spec.name} missing from catalog"
        else:
            assert line not in TOOL_CATALOG, f"hidden alias {spec.name} leaked into catalog"


def test_catalog_contains_new_tools():
    for name in ("search_corpus", "list_topics", "list_promotions", "research_topic"):
        assert f"- {name}:" in TOOL_CATALOG


def test_catalog_does_not_list_removed_or_duplicate_tools():
    assert "- promote_to_main:" not in TOOL_CATALOG
    assert "- run_pipeline:" not in TOOL_CATALOG  # hidden alias
    assert "- run_report:" not in TOOL_CATALOG


def test_run_pipeline_alias_dispatchable_but_hidden():
    assert "run_pipeline" in SYSTEM_TOOL_NAMES
    assert system_tools._SYSTEM_IMPLEMENTATIONS["run_pipeline"].__name__ == "_run_pipeline_alias"


def test_unknown_tool_raises():
    with pytest.raises(ValueError, match="unknown system tool"):
        execute_system_tool("promote_to_main", {})  # removed from registry
    with pytest.raises(ValueError, match="unknown system tool"):
        execute_system_tool("nonexistent_tool", {})


# ---------------------------------------------------------------------------
# parse_tool_marker — canonical, tolerant and deterministic fuzzy
# ---------------------------------------------------------------------------

def test_parse_canonical_marker():
    assert parse_tool_marker('[TOOL:search_corpus]{"query": "quantum"}') == (
        "search_corpus", {"query": "quantum"}
    )


def test_parse_colonless_and_uppercase():
    assert parse_tool_marker('[TOOL list_topics]{"limit": 5}') == ("list_topics", {"limit": 5})
    assert parse_tool_marker('[TOOL:LIST_SOURCES]{}') == ("list_sources", {})


def test_parse_garbled_marker_full_name_containment():
    # Historical bug: model emits [TRUN_INGESTITION] for run_ingestion.
    assert parse_tool_marker('[TRUN_INGESTION]{"days_back": 7}') == ("run_ingestion", {"days_back": 7})
    assert parse_tool_marker('[TRUN_PIPELINE]{"days_back": 3}') == ("run_pipeline", {"days_back": 3})


def test_fuzzy_collision_is_deterministic():
    # "report" is a shared key_part (get_report / compile_report): the fuzzy
    # match must resolve deterministically, never by frozenset order.
    for _ in range(10):
        assert parse_tool_marker('[REPORT]{}')[0] == "get_report"
        assert parse_tool_marker('[COMPILE_REPORT]{"q": 1}')[0] == "compile_report"
        assert parse_tool_marker('[GET_REPORT]{}')[0] == "get_report"


def test_fuzzy_topic_collision():
    # "topic" vs "topics" key_parts: distinct tools, deterministic routing.
    assert parse_tool_marker('[TOPIC]{}')[0] == "research_topic"
    assert parse_tool_marker('[TOPICS]{}')[0] == "list_topics"


def test_fuzzy_no_match_returns_none():
    assert parse_tool_marker("just some prose, no marker") is None
    assert parse_tool_marker('[TODO]{"x": 1}') is None


def test_parse_marker_in_text():
    text = 'Voy a buscar eso. [TOOL:search_corpus]{"query": "q"}'
    assert parse_tool_marker(text) == ("search_corpus", {"query": "q"})


# ---------------------------------------------------------------------------
# Tool behaviors — validation paths that need no corpus/network
# ---------------------------------------------------------------------------

def test_search_corpus_requires_query():
    result = execute_system_tool("search_corpus", {})
    assert not result.ok
    assert "query" in result.error


def test_research_topic_requires_query():
    result = execute_system_tool("research_topic", {})
    assert not result.ok
    assert "query" in result.error


def test_research_topic_already_running(tmp_path, monkeypatch):
    monkeypatch.setattr(
        system_tools, "_research_progress",
        lambda: {"status": "running", "query": "quantum"},
    )
    result = execute_system_tool("research_topic", {"query": "otro tema"})
    assert result.ok
    assert result.data["already_running"] is True


def test_list_promotions_empty_when_no_db(tmp_path, monkeypatch):
    monkeypatch.setattr(system_tools, "TOPIC_CLUSTER_DB", tmp_path / "nope.db")
    result = execute_system_tool("list_promotions", {})
    assert result.ok
    assert result.data["pending"] == []


def test_list_topics_empty_when_no_db(tmp_path, monkeypatch):
    monkeypatch.setattr(system_tools, "TOPIC_CLUSTER_DB", tmp_path / "nope.db")
    result = execute_system_tool("list_topics", {})
    assert result.ok
    assert result.data["total"] == 0


def test_list_promotions_reads_queue(tmp_path, monkeypatch):
    from ipa.agentic.topic_clusters import TopicClusterStore

    db = tmp_path / "topic_clusters.db"
    store = TopicClusterStore(db)
    store.mark_promotion_pending("doc:a", "configured_scrape auto", "configured_scrape", "/tmp/corpus")
    store.mark_promotion_pending("doc:b", "agent_research score 0.85", "agent_research", "/tmp/corpus")
    store.mark_promotion_done("doc:a")
    store.close()
    monkeypatch.setattr(system_tools, "TOPIC_CLUSTER_DB", db)

    result = execute_system_tool("list_promotions", {})
    assert result.ok
    assert result.data["pending_total"] == 1
    assert result.data["pending"][0]["document_id"] == "doc:b"
    assert result.data["recent"][0]["document_id"] == "doc:a"


# ---------------------------------------------------------------------------
# execute_tool dispatch — executor-style tools reachable via the registry
# ---------------------------------------------------------------------------

def test_execute_tool_dispatches_compile_report(tmp_path):
    """execute_tool('compile_report') routes to the executor, not 'unknown tool'."""
    from ipa.agent.agent_tools import ToolContext, execute_tool
    from ipa.agent.agent_memory import AgentMemory
    from ipa.storage.document_store import DocumentStore
    from ipa.contracts import CanonicalDocument, DocumentChunk

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    store = DocumentStore(corpus_dir / "document_store.db")
    doc = CanonicalDocument(
        document_id="doc:1", pages=1, elements=[], source_spans=[],
        text="Quantum photonic processors.", mime_type="text/plain", parser_id="test",
    )
    store.put_document(doc, artifact_id="sha256:doc1")
    store.put_chunks([DocumentChunk(
        chunk_id="chunk:1", document_id="doc:1",
        content_hash="sha256:c1", text="Quantum photonic processors.",
        metadata={"chunk_index": 0},
    )])
    store.commit()
    store.close()

    memory = AgentMemory(tmp_path / "agent.db")
    ctx = ToolContext(memory=memory, corpus_dir=str(corpus_dir))
    sid = memory.open_session(interface="test", role="general", identity_hash="test", title="t")
    ep = memory.record_episode(sid, turn_role="user", content="x", identity_hash="test")

    call, result = execute_tool(
        "compile_report", {"document_ids": ["doc:1"]}, ctx,
        session_id=sid, episode_id=ep.episode_id,
    )
    assert call.tool_name == "compile_report"
    assert call.status == "completed"
    assert result.status == "completed"

    # Invalid args produce a failed contract, not an exception
    call2, result2 = execute_tool(
        "compile_report", {"document_ids": []}, ctx,
        session_id=sid, episode_id=ep.episode_id,
    )
    assert call2.status == "failed"
    assert "document_ids" in (result2.error or "")

    memory.close_session(sid)
    ctx.close()
    memory.close()


def test_execute_tool_dispatches_research_topic_validation(tmp_path):
    """execute_tool('research_topic') reaches the executor — bad args → failed contract."""
    from ipa.agent.agent_tools import ToolContext, execute_tool
    from ipa.agent.agent_memory import AgentMemory

    memory = AgentMemory(tmp_path / "agent.db")
    ctx = ToolContext(memory=memory, corpus_dir=None)
    sid = memory.open_session(interface="test", role="general", identity_hash="test", title="t")
    ep = memory.record_episode(sid, turn_role="user", content="x", identity_hash="test")

    call, result = execute_tool(
        "research_topic", {"query": "q", "freshness": "bogus"}, ctx,
        session_id=sid, episode_id=ep.episode_id,
    )
    assert call.status == "failed"
    assert "freshness" in (result.error or "")

    memory.close_session(sid)
    ctx.close()
    memory.close()


# ---------------------------------------------------------------------------
# Identity — capabilities and skills reach the system prompt
# ---------------------------------------------------------------------------

def test_identity_renders_skills(tmp_path):
    from ipa.agent.agent_identity import Identity

    identity = Identity(
        name="Test", user="U", language="español", persona="p",
        capabilities=["search_corpus"],
        skills=[{"name": "reporte_fino", "flow": "compile_report → get_report"}],
    )
    prompt = identity.system_prompt()
    assert "search_corpus" in prompt
    assert "reporte_fino" in prompt
    assert "compile_report" in prompt


def test_identity_file_has_updated_capabilities():
    from ipa.agent.agent_identity import load_identity

    identity = load_identity()
    for cap in ("search_corpus", "compile_report", "research_topic", "list_promotions"):
        assert cap in identity.capabilities
    assert identity.skills, "skills section missing from agent_identity.yaml"
