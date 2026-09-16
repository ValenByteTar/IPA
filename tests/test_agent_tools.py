"""Fase 1 tests: deterministic agent tools + contract validation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "validation"))

from ipa.agent import AgentMemory, ToolContext, ToolCall, ToolResult, execute_tool, TOOL_NAMES, load_identity  # noqa: E402
from validate_agent_contract import validate  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers and fixtures
# ---------------------------------------------------------------------------

HASH_PREFIX = "sha256:" + "a" * 64
NOW = "2026-09-06T12:00:00Z"
SESSION_ID = "agent_session:testsession001"
EPISODE_ID = "agent_episode:testepisode001"


@pytest.fixture()
def memory(tmp_path):
    with AgentMemory(store_path=tmp_path / "agent.db") as store:
        yield store


@pytest.fixture()
def ctx(memory):
    """ToolContext with memory but no corpus — for tools that only need memory."""
    return ToolContext(memory=memory)


def _make_episode(memory, session_id, content="hello world", turn_role="user"):
    """Create a session + episode for testing recall_conversation."""
    identity = load_identity()
    sid = memory.open_session(interface="cli", role="general", identity_hash=identity.identity_hash, session_id=session_id)
    ep = memory.record_episode(sid, turn_role=turn_role, content=content, identity_hash=identity.identity_hash)
    return sid, ep


# ---------------------------------------------------------------------------
# ToolCall / ToolResult contract validation
# ---------------------------------------------------------------------------

def test_tool_call_contract_validates():
    call = ToolCall(
        tool_call_id="tool_call:testcall001",
        session_id=SESSION_ID,
        episode_id=EPISODE_ID,
        tool_name="search_corpus",
        arguments={"query": "test", "limit": 5},
        called_at=NOW,
        status="completed",
    )
    errors = validate("ToolCall", call.to_contract())
    assert errors == [], errors


def test_tool_result_contract_validates():
    import hashlib, json as _json
    result_data = {"hits": [], "total": 0}
    correct_hash = "sha256:" + hashlib.sha256(
        _json.dumps(result_data, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    result = ToolResult(
        tool_result_id="tool_result:testresult001",
        tool_call_id="tool_call:testcall001",
        session_id=SESSION_ID,
        tool_name="search_corpus",
        result=result_data,
        result_hash=correct_hash,
        source_refs=[],
        started_at=NOW,
        completed_at=NOW,
        elapsed_ms=42,
        status="completed",
    )
    errors = validate("ToolResult", result.to_contract())
    assert errors == [], errors


def test_tool_call_failed_requires_error():
    call = ToolCall(
        tool_call_id="tool_call:testcall002",
        session_id=SESSION_ID,
        episode_id=EPISODE_ID,
        tool_name="search_corpus",
        arguments={},
        called_at=NOW,
        status="failed",
        error=None,  # should fail: failed status requires error
    )
    errors = validate("ToolCall", call.to_contract())
    assert any("error" in e for e in errors)


def test_tool_result_hash_detects_tampering():
    result = ToolResult(
        tool_result_id="tool_result:testresult002",
        tool_call_id="tool_call:testcall001",
        session_id=SESSION_ID,
        tool_name="search_corpus",
        result={"hits": [{"chunk_id": "c1"}], "total": 1},
        result_hash="sha256:" + "0" * 64,  # wrong hash
        source_refs=[],
        started_at=NOW,
        completed_at=NOW,
        elapsed_ms=10,
        status="completed",
    )
    errors = validate("ToolResult", result.to_contract())
    assert any("result_hash" in e for e in errors)


def test_tool_call_rejects_unknown_tool_name():
    call = ToolCall(
        tool_call_id="tool_call:testcall003",
        session_id=SESSION_ID,
        episode_id=EPISODE_ID,
        tool_name="hack_the_planet",
        arguments={},
        called_at=NOW,
        status="completed",
    )
    errors = validate("ToolCall", call.to_contract())
    assert any("tool_name" in e for e in errors)


# ---------------------------------------------------------------------------
# recall_conversation — only needs memory, no corpus
# ---------------------------------------------------------------------------

def test_recall_conversation_finds_episodes_by_session(memory, ctx):
    sid, ep = _make_episode(memory, "agent_session:recall001", content="hola desde el test")
    call, result = execute_tool(
        "recall_conversation",
        {"session_id": sid, "limit": 10},
        ctx,
        session_id=sid,
        episode_id=ep.episode_id,
    )
    assert call.status == "completed"
    assert result.status == "completed"
    assert result.result["total"] >= 1
    assert any("hola desde el test" in e["content_preview"] for e in result.result["episodes"])
    # Contract validation
    assert validate("ToolCall", call.to_contract()) == []
    assert validate("ToolResult", result.to_contract()) == []


def test_recall_conversation_filters_by_text(memory, ctx):
    sid, ep = _make_episode(memory, "agent_session:recall002", content="python machine learning topic")
    _make_episode(memory, "agent_session:recall003", content="completely different content about cooking")
    call, result = execute_tool(
        "recall_conversation",
        {"query": "python", "limit": 10},
        ctx,
        session_id=sid,
        episode_id=ep.episode_id,
    )
    assert result.status == "completed"
    assert all("python" in e["content_preview"].lower() for e in result.result["episodes"])


def test_recall_conversation_returns_empty_when_no_match(memory, ctx):
    sid, ep = _make_episode(memory, "agent_session:recall004", content="something")
    call, result = execute_tool(
        "recall_conversation",
        {"query": "nonexistent_xyz_term", "limit": 5},
        ctx,
        session_id=sid,
        episode_id=ep.episode_id,
    )
    assert result.status == "completed"
    assert result.result["total"] == 0


# ---------------------------------------------------------------------------
# search_corpus / list_topics / get_topic_info — need a real corpus
# ---------------------------------------------------------------------------

CORPUS_DIR = Path(__file__).parents[1] / "outputs" / "experiments" / "E12-corpus"


def _corpus_available():
    return (CORPUS_DIR / "document_store.db").exists() and (CORPUS_DIR / "vector" / "lancedb").exists()


pytestmark_corpus = pytest.mark.skipif(
    not _corpus_available(),
    reason="E12 corpus not available — run FastPath + LanceDB indexing first",
)


@pytestmark_corpus
def test_search_corpus_returns_hits_with_citations(tmp_path, memory):
    ctx = ToolContext(memory=memory, corpus_dir=str(CORPUS_DIR))
    try:
        sid, ep = _make_episode(memory, "agent_session:search001", content="search test")
        call, result = execute_tool(
            "search_corpus",
            {"query": "machine learning", "limit": 3},
            ctx,
            session_id=sid,
            episode_id=ep.episode_id,
        )
        assert call.status == "completed"
        assert result.status == "completed"
        assert result.result["total"] > 0
        assert len(result.source_refs) > 0
        # Every hit has a chunk_id and score
        for hit in result.result["hits"]:
            assert "chunk_id" in hit
            assert "score" in hit
            assert "document_id" in hit
        # Contract validation
        assert validate("ToolCall", call.to_contract()) == []
        assert validate("ToolResult", result.to_contract()) == []
    finally:
        ctx.close()


@pytestmark_corpus
def test_list_topics_returns_documents(tmp_path, memory):
    ctx = ToolContext(memory=memory, corpus_dir=str(CORPUS_DIR))
    try:
        sid, ep = _make_episode(memory, "agent_session:topics001", content="list topics")
        call, result = execute_tool(
            "list_topics",
            {"limit": 5},
            ctx,
            session_id=sid,
            episode_id=ep.episode_id,
        )
        assert call.status == "completed"
        assert result.status == "completed"
        assert result.result["total"] > 0
        for topic in result.result["topics"]:
            assert "document_id" in topic
        assert validate("ToolCall", call.to_contract()) == []
        assert validate("ToolResult", result.to_contract()) == []
    finally:
        ctx.close()


@pytestmark_corpus
def test_get_topic_info_returns_chunks(tmp_path, memory):
    ctx = ToolContext(memory=memory, corpus_dir=str(CORPUS_DIR))
    try:
        # First get a document_id from list_topics
        _, list_result = execute_tool(
            "list_topics",
            {"limit": 1},
            ctx,
            session_id="agent_session:topicinfo001",
            episode_id="agent_episode:topicinfo001",
        )
        assert list_result.status == "completed"
        assert list_result.result["total"] > 0
        doc_id = list_result.result["topics"][0]["document_id"]

        sid, ep = _make_episode(memory, "agent_session:topicinfo002", content="get topic info")
        call, result = execute_tool(
            "get_topic_info",
            {"document_id": doc_id, "max_chunks": 3},
            ctx,
            session_id=sid,
            episode_id=ep.episode_id,
        )
        assert call.status == "completed"
        assert result.status == "completed"
        assert result.result["document_id"] == doc_id
        assert len(result.result["chunk_previews"]) > 0
        assert len(result.source_refs) > 0
        assert validate("ToolCall", call.to_contract()) == []
        assert validate("ToolResult", result.to_contract()) == []
    finally:
        ctx.close()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_unknown_tool_fails_gracefully(memory, ctx):
    sid, ep = _make_episode(memory, "agent_session:err001", content="error test")
    with pytest.raises(ValueError, match="unknown tool"):
        execute_tool("nonexistent_tool", {}, ctx, session_id=sid, episode_id=ep.episode_id)


def test_search_corpus_without_corpus_dir_fails(memory):
    ctx = ToolContext(memory=memory)  # no corpus_dir
    sid, ep = _make_episode(memory, "agent_session:err002", content="no corpus")
    call, result = execute_tool(
        "search_corpus",
        {"query": "test"},
        ctx,
        session_id=sid,
        episode_id=ep.episode_id,
    )
    assert call.status == "failed"
    assert result.status == "failed"
    assert "corpus" in (result.error or "").lower()
    assert validate("ToolCall", call.to_contract()) == []
    assert validate("ToolResult", result.to_contract()) == []


def test_search_corpus_empty_query_fails(memory):
    ctx = ToolContext(memory=memory, corpus_dir=".")
    sid, ep = _make_episode(memory, "agent_session:err003", content="empty query")
    call, result = execute_tool(
        "search_corpus",
        {"query": ""},
        ctx,
        session_id=sid,
        episode_id=ep.episode_id,
    )
    assert call.status == "failed"
    assert result.status == "failed"
    assert "query" in (result.error or "").lower()


# ---------------------------------------------------------------------------
# Tool registry completeness
# ---------------------------------------------------------------------------

def test_tool_names_includes_all_fase1_tools():
    expected = {"search_corpus", "list_topics", "get_topic_info", "recall_conversation", "research_topic"}
    assert expected <= TOOL_NAMES
