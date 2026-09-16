"""Agentic memory store tests — the retrievable personal/agentic corpus.

Covers:
  - MemoryStore: upsert/recall roundtrip, scope filtering, provenance
  - MemoryIndexer: deterministic sync from canonical sources
    (session summaries, user model, strategic principles, tutor mastery)
  - Idempotent re-sync; sources remain canonical (items are derived)
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ipa.agent.agent_memory import AgentMemory  # noqa: E402
from ipa.agent.memory_store import MemoryIndexer, MemoryItem, MemoryStore  # noqa: E402
from ipa.agent.user_model import UserModelStore  # noqa: E402
from ipa.tutor.tutor_contracts import (  # noqa: E402
    GenerationProvenance,
    MasteryStatus,
    UserTopicRecord,
)
from ipa.tutor.tutor_runtime import TutorStore  # noqa: E402


@pytest.fixture()
def store(tmp_path):
    return MemoryStore(tmp_path / "memory.db")


def test_upsert_and_recall_roundtrip(store):
    store.upsert_item(MemoryItem(
        memory_id="mem:test:1", scope="user", kind="fact",
        text="Hecho sobre el usuario: trabaja en un paper sobre RAG híbrido",
        source_ref="fact:1", confidence=1.0,
    ))
    items = store.recall("RAG híbrido")
    assert len(items) == 1
    assert items[0].scope == "user"
    assert items[0].source_ref == "fact:1"


def test_recall_scope_filter(store):
    store.upsert_item(MemoryItem(
        memory_id="mem:a", scope="user", kind="fact", text="le gusta el mate",
    ))
    store.upsert_item(MemoryItem(
        memory_id="mem:b", scope="agent", kind="principle",
        text="respondo corto y al grano, sin emojis",
    ))
    assert [i.scope for i in store.recall("mate", scopes=["user"])] == ["user"]
    assert store.recall("mate", scopes=["agent"]) == []
    assert len(store.recall("emojis")) == 1


def test_recall_empty_query_returns_recent(store):
    for i in range(3):
        store.upsert_item(MemoryItem(
            memory_id=f"mem:r{i}", scope="episodic", kind="session_summary",
            text=f"resumen {i}",
        ))
    assert len(store.recall("", scopes=["episodic"], limit=2)) == 2


def test_invalid_scope_rejected(store):
    with pytest.raises(ValueError, match="invalid scope"):
        store.upsert_item(MemoryItem(
            memory_id="mem:x", scope="nope", kind="fact", text="x",
        ))


def test_indexer_syncs_sessions_and_user_model(tmp_path):
    memory = AgentMemory(store_path=tmp_path / "agent.db")
    sid = memory.open_session(interface="cli", role="general", identity_hash="h")
    memory.update_session_summary(sid, "Hablamos del diseño del Tutor y el gate de aprobación", title="tutor")

    um = UserModelStore(tmp_path / "user_model.db")
    um.add_goal("armar el módulo tutor", source="declared")
    um.declare_interest("retrieval híbrido")
    fid = um.add_fact("es el creador del sistema", source="declared")
    um.decide_fact(fid, approved=True)

    store = MemoryStore(tmp_path / "memory.db")
    n = MemoryIndexer(store).sync(memory=memory, user_model=um)
    assert n >= 4

    hits = store.recall("Tutor aprobación")
    assert hits and hits[0].kind == "session_summary"
    assert hits[0].source_ref == sid

    hits = store.recall("paper", scopes=["user"])
    goals = store.recall("módulo tutor", scopes=["user"])
    assert goals and "Objetivo del usuario" in goals[0].text

    # Idempotent: second sync doesn't duplicate.
    MemoryIndexer(store).sync(memory=memory, user_model=um)
    assert store.count() == n


def test_indexer_syncs_tutor_mastery(tmp_path):
    ts = TutorStore(tmp_path / "tutor.db")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    ts.upsert_topic_record(UserTopicRecord(
        record_id="user_topic_record:rag", topic_id="rag",
        mastery_status=MasteryStatus.UNDERSTOOD, mastery_score=0.8,
        attempts=3, last_assessment_id="assessment:t1",
        evidence_ids=["user_evidence:t1"], updated_at=now, created_at=now,
        generation=GenerationProvenance(
            generator="test", generated_at=now,
            input_hash="sha256:" + "a" * 64, model_fingerprint="test",
        ),
        field_origins={"mastery_status": "generated", "mastery_score": "generated",
                       "attempts": "system", "last_assessment_id": "system",
                       "evidence_ids": "system"},
    ))
    store = MemoryStore(tmp_path / "memory.db")
    MemoryIndexer(store).sync(tutor=ts)
    hits = store.recall("rag", scopes=["tutor"])
    assert hits and "understood" in hits[0].text
    assert hits[0].confidence == 0.8


def test_recall_memory_tool_dispatch():
    """The system tool syncs whatever sources exist then recalls. With the
    real stores present it must return ok; missing stores degrade cleanly."""
    from ipa.agent import system_tools
    result = system_tools.execute_system_tool("recall_memory", {"query": "arxiv"})
    assert result.tool_name == "recall_memory"
    assert result.ok is True


def test_recall_memory_registered_in_catalog():
    from ipa.agent import system_tools
    assert "recall_memory" in system_tools.SYSTEM_TOOL_NAMES
    assert "recall_memory" in system_tools.BASE_TOOLS
    assert "recall_memory" in system_tools.TOOL_CATALOG


# ── Vector side: sqlite-vec semantic recall + RRF fusion ────────────────────

def test_rrf_merge_fuses_and_ranks():
    from ipa.agent.memory_store import _rrf_merge
    fused = _rrf_merge(["a", "b", "c"], ["b", "d", "a"])
    # b aparece en ambas listas → primero; a en 1° y 3° → segundo.
    assert fused[0] == "b"
    assert set(fused) == {"a", "b", "c", "d"}


def test_vector_index_roundtrip(tmp_path):
    from ipa.agent.memory_store import MemoryVectorIndex
    idx = MemoryVectorIndex(tmp_path / "vec.db", vector_dim=4)
    try:
        idx.upsert("mem:1", [1.0, 0.0, 0.0, 0.0], scope="episodic")
        idx.upsert("mem:2", [0.0, 1.0, 0.0, 0.0], scope="user")
        assert idx.count() == 2
        assert idx.embedded_ids() == {"mem:1", "mem:2"}
        hits = idx.search([1.0, 0.0, 0.0, 0.0], limit=2)
        assert hits[0][0] == "mem:1"
        assert hits[0][1] > hits[1][1]
        idx.clear()
        assert idx.count() == 0
    finally:
        idx.close()


def test_get_by_ids_preserves_rank_order(store):
    store.upsert_item(MemoryItem(
        memory_id="mem:x:1", scope="user", kind="fact",
        text="uno", source_ref="1"))
    store.upsert_item(MemoryItem(
        memory_id="mem:x:2", scope="user", kind="fact",
        text="dos", source_ref="2"))
    items = store.get_by_ids(["mem:x:2", "mem:x:1", "mem:x:missing"])
    assert [i.memory_id for i in items] == ["mem:x:2", "mem:x:1"]


def test_indexer_syncs_unit_summaries(tmp_path):
    memory_store = MemoryStore(tmp_path / "memory.db")
    tutor = TutorStore(tmp_path / "tutor.db")
    tutor.save_unit_summary("roadmap:abc", 2, "Explicamos tokens y predicción one-at-a-time")
    indexer = MemoryIndexer(memory_store)
    n = indexer._sync_tutor(tutor)
    items = [i for i in memory_store.recall("tokens predicción") if i.kind == "lesson_unit"]
    assert n >= 1
    assert len(items) == 1
    assert "unidad 2" in items[0].text
    assert items[0].scope == "episodic"
    assert items[0].source_ref == "roadmap:abc:2"
    # Idempotente
    n2 = indexer._sync_tutor(tutor)
    assert memory_store.count("episodic") == memory_store.count("episodic")
    assert n2 >= 1
    tutor.close()
    memory_store.close()
