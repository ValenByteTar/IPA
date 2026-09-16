"""Tests for session management + idle consolidation (Fase 4 UX)."""
import os
import tempfile
import time
from datetime import datetime, timezone, timedelta

import pytest

os.environ.setdefault("IPA_AGENT_STORE", os.path.join(tempfile.gettempdir(), "test_sess_mgmt.db"))
STORE = os.environ["IPA_AGENT_STORE"]


@pytest.fixture()
def memory():
    if os.path.exists(STORE):
        os.remove(STORE)
    from ipa.agent.agent_memory import AgentMemory
    m = AgentMemory()
    yield m
    m.close()


def _make_session(memory, *, minutes_ago=10, episodes=4, status="closed"):
    sid = memory.open_session(interface="dashboard", role="general", identity_hash="test")
    for i in range(episodes):
        memory.record_episode(sid, turn_role="user" if i % 2 == 0 else "assistant",
                              content=f"turno {i}", identity_hash="test")
    if status != "active":
        memory.close_session(sid)
    # Backdate last_active_at
    old = (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat().replace("+00:00", "Z")
    memory._connection.execute(
        "UPDATE agent_sessions SET last_active_at = ? WHERE session_id = ?", (old, sid))
    memory._connection.commit()
    return sid


def test_rename_session(memory):
    sid = _make_session(memory)
    memory.rename_session(sid, "Mi chat de RoPE")
    assert memory.get_session(sid).title == "Mi chat de RoPE"


def test_rename_empty_title_raises(memory):
    sid = _make_session(memory)
    with pytest.raises(ValueError):
        memory.rename_session(sid, "   ")


def test_archive_session(memory):
    sid = _make_session(memory)
    memory.archive_session(sid)
    assert memory.get_session(sid).status == "archived"
    # Archived sessions are excluded from the default list
    ids = [s.session_id for s in memory.list_sessions()]
    assert sid not in ids
    # But visible with include_archived
    ids_all = [s.session_id for s in memory.list_sessions(include_archived=True)]
    assert sid in ids_all


def test_find_idle_sessions(memory):
    idle_old = _make_session(memory, minutes_ago=10)
    _make_session(memory, minutes_ago=1)  # too recent
    _make_session(memory, minutes_ago=10, status="active")  # not closed
    found = memory.find_idle_sessions(idle_minutes=5)
    assert [s.session_id for s in found] == [idle_old]


def test_find_idle_skips_consolidated(memory):
    sid = _make_session(memory, minutes_ago=10)
    memory.update_session_summary(sid, "resumen previo")
    assert memory.find_idle_sessions(idle_minutes=5) == []


def test_summary_update(memory):
    sid = _make_session(memory)
    memory.update_session_summary(sid, "Hablaban de RoPE", title="Rotaciones RoPE")
    s = memory.get_session(sid)
    assert s.summary == "Hablaban de RoPE"
    assert s.title == "Rotaciones RoPE"
    assert s.consolidated_at is not None


def test_find_idle_sessions_includes_archived(memory):
    """Archivar no saca la sesión del pipeline de consolidación: los
    episodios siguen siendo memoria y el resumen se genera igual."""
    sid = _make_session(memory, minutes_ago=10)
    memory.archive_session(sid)
    # archive_session refresca last_active_at — backdate de nuevo.
    old = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat().replace("+00:00", "Z")
    memory._connection.execute(
        "UPDATE agent_sessions SET last_active_at = ? WHERE session_id = ?", (old, sid))
    memory._connection.commit()
    found = memory.find_idle_sessions(idle_minutes=5)
    assert [s.session_id for s in found] == [sid]


def test_close_stale_active_sessions(memory):
    """Sesiones 'active' huérfanas (restart del dashboard) se cierran;
    las recientes no se tocan."""
    orphan = _make_session(memory, minutes_ago=45, status="active")
    fresh = _make_session(memory, minutes_ago=1, status="active")
    closed = _make_session(memory, minutes_ago=10)
    n = memory.close_stale_active_sessions(idle_minutes=30)
    assert n == 1
    assert memory.get_session(orphan).status == "closed"
    assert memory.get_session(fresh).status == "active"
    assert memory.get_session(closed).status == "closed"


def test_consolidator_accepts_plain_string_provider(memory):
    """OllamaProvider.generate_chat devuelve str plano — el consolidador
    debe aceptar ambos contratos (str y objeto .text)."""
    from ipa.agent.session_consolidator import SessionConsolidator

    class StringProvider:
        def generate_chat(self, messages, **kwargs):
            return '{"title": "Chat str", "summary": "Resumen desde string.", "key_topics": [], "user_facts": []}'

    sid = _make_session(memory)
    consolidator = SessionConsolidator(memory, StringProvider())
    result = consolidator.consolidate_session(sid)
    assert result is not None
    assert result["title"] == "Chat str"
    s = memory.get_session(sid)
    assert s.consolidated_at is not None


def test_consolidator_parses_and_routes(memory, tmp_path, monkeypatch):
    from ipa.agent.session_consolidator import SessionConsolidator
    from ipa.agentic.memory_consolidation import ConsolidationStore as _RealStore

    # El consolidador construye ConsolidationStore() sin argumentos: sin este
    # patch, el test escribe propuestas en el store REAL y contamina la cola
    # de aprobaciones del usuario.
    monkeypatch.setattr(
        "ipa.agentic.memory_consolidation.ConsolidationStore",
        lambda *a, **k: _RealStore(tmp_path / "consolidation.db"),
    )

    class FakeResult:
        def __init__(self, text):
            self.text = text
            self.error = None

    class FakeProvider:
        def generate_chat(self, messages, **kwargs):
            return FakeResult('```json {"title": "Rotaciones RoPE", "summary": "Explicación de matrices de rotación.", "key_topics": ["rope"], "user_facts": ["Valen estudia RoPE para su AGI local"]}```')

    sid = _make_session(memory, episodes=4)
    consolidator = SessionConsolidator(memory, FakeProvider())
    result = consolidator.consolidate_session(sid)
    assert result is not None
    assert result["title"] == "Rotaciones RoPE"
    assert "matrices de rotación" in result["summary"]
    assert len(result["fact_proposals"]) == 1
    # Session updated
    s = memory.get_session(sid)
    assert s.summary == "Explicación de matrices de rotación."
    assert s.consolidated_at is not None
    # Proposal created in the (temp) approval queue
    store = _RealStore(tmp_path / "consolidation.db")
    try:
        props = store.list_proposals(status="pending")
        assert any("Valen estudia RoPE" in p.summary for p in props)
    finally:
        store.close()


def test_consolidator_no_facts_no_proposals(memory):
    from ipa.agent.session_consolidator import SessionConsolidator

    class FakeResult:
        def __init__(self, text):
            self.text = text
            self.error = None

    class FakeProvider:
        def generate_chat(self, messages, **kwargs):
            return FakeResult('{"title": "Chat", "summary": "Charla casual.", "key_topics": [], "user_facts": []}')

    sid = _make_session(memory)
    consolidator = SessionConsolidator(memory, FakeProvider())
    result = consolidator.consolidate_session(sid)
    assert result is not None
    assert result["fact_proposals"] == []


def test_parse_json_with_think_junk():
    from ipa.agent.session_consolidator import SessionConsolidator
    text = '</think>{"title": "T", "summary": "S", "key_topics": [], "user_facts": []}'
    parsed = SessionConsolidator._parse_json(text)
    assert parsed is not None
    assert parsed["title"] == "T"
