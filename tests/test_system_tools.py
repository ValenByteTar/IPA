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


def _no_launch(monkeypatch, tmp_path):
    """Neutraliza el lanzamiento real: corpus fake + Popen capturado."""
    import ipa.agent.research_review as rr
    captured: dict = {}
    monkeypatch.setattr(system_tools, "_research_progress", lambda: {})
    monkeypatch.setattr(system_tools, "_main_corpus_dir", lambda: tmp_path)
    monkeypatch.setattr(system_tools, "_write_research_progress", lambda p: captured.update(p))
    monkeypatch.setattr(rr, "mark_researched", lambda q, **kw: None)
    import subprocess as sp
    monkeypatch.setattr(sp, "Popen", lambda argv, **kw: captured.setdefault("argv", argv))
    return rr, captured


def test_research_topic_dedup_blocks_explicit_call(tmp_path, monkeypatch):
    """El dedup aplica también a llamados explícitos del modelo (no solo al
    safety-net): reformular la query no debe relanzar la investigación."""
    rr, captured = _no_launch(monkeypatch, tmp_path)
    monkeypatch.setattr(rr, "find_recent_research", lambda q, **kw: {
        "query": "acuerdo ralentizacion avance ia big techs",
        "age_minutes": 4, "exact": False,
    })
    result = execute_system_tool(
        "research_topic",
        {"query": "acuerdo ralentizacion avance IA tres grandes tecnologicas"},
    )
    assert result.ok
    assert result.data["dedup"] is True
    assert result.data["matched_query"] == "acuerdo ralentizacion avance ia big techs"
    assert "force=true" in result.summary
    assert "argv" not in captured  # no se lanzó ningún subproceso


def test_research_topic_force_bypasses_dedup(tmp_path, monkeypatch):
    rr, captured = _no_launch(monkeypatch, tmp_path)
    monkeypatch.setattr(rr, "find_recent_research", lambda q, **kw: {
        "query": "lo mismo", "age_minutes": 1, "exact": True,
    })
    result = execute_system_tool(
        "research_topic", {"query": "lo mismo de nuevo", "force": True}
    )
    assert result.ok
    assert "dedup" not in result.data
    assert captured["status"] == "running"
    assert captured["argv"][1].endswith("run_research.py")


def test_research_topic_reinjects_urls_from_user_message(tmp_path, monkeypatch):
    """Las URLs del mensaje crudo del usuario se reinyectan en la query —
    el modelo las descarta al parafrasear."""
    rr, captured = _no_launch(monkeypatch, tmp_path)
    monkeypatch.setattr(rr, "find_recent_research", lambda q, **kw: None)
    url = "https://www.pagina12.com.ar/2026/09/12/las-big-tech-de-la-ia-dicen-estar-de-acuerdo-en-frenar-su-desarrollo/"
    result = execute_system_tool("research_topic", {
        "query": "acuerdo de las big tech para frenar la IA",
        "_user_message": f"Como que no, mira {url}",
    })
    assert result.ok
    assert url in result.data["query"]
    assert url in captured["argv"][2]  # la query lanzada lleva la URL


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


# ---------------------------------------------------------------------------
# get_document — apertura de un doc por id (identidad/provenance/lifecycle)
# ---------------------------------------------------------------------------

def _mk_corpus(corpus_dir: Path, doc_id: str, text: str,
               *, tombstone: bool = False) -> None:
    from ipa import DocumentStore
    from ipa.contracts import CanonicalDocument, DocumentChunk

    corpus_dir.mkdir(parents=True, exist_ok=True)
    store = DocumentStore(corpus_dir / "document_store.db")
    store.put_document(
        CanonicalDocument(document_id=doc_id, parser_id="test",
                          mime_type="text/plain", pages=1, text=text,
                          elements=[], source_spans=[]), "art:1")
    store.put_chunks([DocumentChunk(
        chunk_id=f"{doc_id}:c0", document_id=doc_id, content_hash="h",
        text=text[:200], metadata={}, source_span=None)])
    store.put_doc_meta(doc_id, normalized_hash="sha256:abc", title="Doc T",
                       published_at="2026-09-01", char_count=len(text),
                       extra={"duplicate_of_main": "m1"})
    store.put_source(doc_id, "https://x.com/a", "x.com", "agent_research",
                     quality_score=0.8, published_at="2026-09-01")
    if tombstone:
        store.tombstone_document(doc_id)
    store.commit()
    store.close()


def _mk_cluster_db(path: Path) -> None:
    import sqlite3
    conn = sqlite3.connect(str(path))
    conn.executescript("""
        CREATE TABLE promotion_queue (
            document_id TEXT PRIMARY KEY, reason TEXT, provenance TEXT,
            source_corpus TEXT DEFAULT '', status TEXT DEFAULT 'pending',
            queued_at TEXT, promoted_at TEXT);
        CREATE TABLE curation_decisions (
            decision_id TEXT PRIMARY KEY, document_id TEXT,
            report_id TEXT, payload_json TEXT, created_at TEXT);
        """)
    conn.execute(
        "INSERT INTO promotion_queue VALUES ('d1','score 0.8','agent_research',"
        "'research_staging','pending','2026-09-23T00:00:00Z',NULL)")
    conn.execute(
        "INSERT INTO curation_decisions VALUES ('dec1','d1','rep1',?, 'x')",
        (json.dumps({"decision": "reporter_only", "promotion_score": 0.62}),))
    conn.commit()
    conn.close()


def test_get_document_returns_full_record(tmp_path, monkeypatch):
    corpus = tmp_path / "main"
    _mk_corpus(corpus, "d1", "texto del documento de prueba" * 30)
    monkeypatch.setattr(system_tools, "_doc_corpora",
                        lambda: [("main", corpus)])
    cluster_db = tmp_path / "topic_clusters.db"
    _mk_cluster_db(cluster_db)
    monkeypatch.setattr(system_tools, "TOPIC_CLUSTER_DB", cluster_db)

    res = system_tools.execute_system_tool("get_document", {"doc_id": "d1"})
    assert res.ok, res.error
    d = res.data
    assert d["document_id"] == "d1" and d["corpus"] == "main"
    assert d["tombstoned"] is False and d["title"] == "Doc T"
    assert d["provenance"]["provenance"] == "agent_research"
    assert d["provenance"]["source_url"] == "https://x.com/a"
    assert d["extra"]["duplicate_of_main"] == "m1"
    assert d["promotion_queue"]["status"] == "pending"
    assert d["curation_decision"]["promotion_score"] == 0.62
    assert d["chunks_live"] == 1 and "texto del documento" in d["text_preview"]


def test_get_document_finds_staging_doc(tmp_path, monkeypatch):
    """Un doc pendiente de promoción vive en staging, no en main."""
    main = tmp_path / "main"
    _mk_corpus(main, "m1", "doc de main")
    staging = tmp_path / "staging"
    _mk_corpus(staging, "s1", "doc recién investigado")
    monkeypatch.setattr(system_tools, "_doc_corpora",
                        lambda: [("main", main), ("research_staging", staging)])
    monkeypatch.setattr(system_tools, "TOPIC_CLUSTER_DB",
                        tmp_path / "nope.db")

    res = system_tools.execute_system_tool("get_document", {"doc_id": "s1"})
    assert res.ok and res.data["corpus"] == "research_staging"


def test_get_document_reports_tombstoned(tmp_path, monkeypatch):
    corpus = tmp_path / "main"
    _mk_corpus(corpus, "d1", "doc muerto", tombstone=True)
    monkeypatch.setattr(system_tools, "_doc_corpora",
                        lambda: [("main", corpus)])
    monkeypatch.setattr(system_tools, "TOPIC_CLUSTER_DB",
                        tmp_path / "nope.db")
    res = system_tools.execute_system_tool("get_document", {"doc_id": "d1"})
    assert res.ok and res.data["tombstoned"] is True


def test_get_document_not_found(tmp_path, monkeypatch):
    corpus = tmp_path / "main"
    _mk_corpus(corpus, "d1", "doc")
    monkeypatch.setattr(system_tools, "_doc_corpora",
                        lambda: [("main", corpus)])
    res = system_tools.execute_system_tool("get_document", {"doc_id": "zzz"})
    assert not res.ok and "no encontrado" in res.error


def test_get_document_requires_doc_id():
    res = system_tools.execute_system_tool("get_document", {})
    assert not res.ok and "doc_id" in res.error
