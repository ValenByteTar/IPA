"""Tests del wireado de la capa cognitiva al ciclo idle (Tier 1 / Tier 2)."""
from __future__ import annotations

import json

import pytest


@pytest.fixture()
def stores(tmp_path, monkeypatch):
    """Apunta los 4 stores de la capa cognitiva a tmp."""
    paths = {
        "um": tmp_path / "user_model.db",
        "sk": tmp_path / "skill_library.db",
        "sm": tmp_path / "strategic_memory.db",
        "un": tmp_path / "uncertainty.db",
    }
    monkeypatch.setattr("ipa.agent.user_model.DEFAULT_USER_MODEL_STORE", paths["um"])
    monkeypatch.setattr("ipa.agent.skill_library.DEFAULT_SKILL_STORE", paths["sk"])
    monkeypatch.setattr("ipa.agent.strategic_memory.DEFAULT_STRATEGIC_STORE", paths["sm"])
    monkeypatch.setattr("ipa.agent.uncertainty.DEFAULT_UNCERTAINTY_STORE", paths["un"])
    return paths


def _tool_episodes(n: int = 5) -> list[dict]:
    return [
        {
            "session_id": f"s{i}", "turn_role": "assistant",
            "content": "[TOOL:search_corpus]{\"query\": \"retrieval\"}",
            "tool_calls": ["search_corpus", "compile_report"],
        }
        for i in range(n)
    ]


def test_layer1_wires_skills_principles_and_goals(stores):
    """Los cuatro módulos huérfanos ahora producen propuestas determinísticas."""
    from ipa.agentic.idle_cognition import run_cognitive_layer1

    result = run_cognitive_layer1(
        episodes=_tool_episodes(5),
        tasks=[{"goal": "investigá fotónica"}, {"goal": "investigá fotónica aplicada"}],
    )
    assert result["skills"] >= 1
    assert result["principles"] >= 1
    assert result["goals"] >= 1

    # Todo entra pending — el gate humano decide, nada se auto-aplica.
    from ipa.agent.skill_library import SkillLibraryStore
    from ipa.agent.strategic_memory import StrategicMemoryStore
    from ipa.agent.user_model import UserModelStore

    sk = SkillLibraryStore(stores["sk"])
    try:
        assert sk.active_skills() == []
        assert sk.list_skills(status="pending")
    finally:
        sk.close()
    sm = StrategicMemoryStore(stores["sm"])
    try:
        assert sm.active_principles() == []
        assert sm.list_principles(status="pending")
    finally:
        sm.close()
    um = UserModelStore(stores["um"])
    try:
        assert um.active_goals() == []
        assert um.list_goals(status="pending")
    finally:
        um.close()


def test_layer1_is_idempotent(stores):
    """Segunda corrida con los mismos datos no duplica propuestas."""
    from ipa.agentic.idle_cognition import run_cognitive_layer1

    eps = _tool_episodes(5)
    run_cognitive_layer1(episodes=eps, tasks=[])
    second = run_cognitive_layer1(episodes=eps, tasks=[])
    assert second["skills"] == 0
    assert second["principles"] == 0


def test_layer1_proposes_research_for_low_confidence(stores):
    """Tópicos con confianza baja acumulada → propuesta de investigación."""
    from ipa.agent.uncertainty import UncertaintyStore
    from ipa.agentic.idle_cognition import run_cognitive_layer1

    store = UncertaintyStore(stores["un"])
    try:
        for _ in range(3):
            store.record_observation("mystery_topic", 0.1)
    finally:
        store.close()

    result = run_cognitive_layer1(episodes=[], tasks=[])
    assert result["research_proposals"] >= 1
    store = UncertaintyStore(stores["un"])
    try:
        pending = store.list_proposals(status="pending")
        assert any(p.topic == "mystery_topic" for p in pending)
    finally:
        store.close()


def test_layer2_llm_principles_with_object_provider(stores):
    from ipa.agentic.idle_cognition import run_cognitive_layer2

    class ObjProvider:
        def generate_chat(self, messages, **kw):
            class R:
                text = json.dumps({"principles": [
                    {"pattern": "pide reportes seguido", "principle": "ofrecer compile_report antes de cerrar", "confidence": 0.7},
                ]})
                error = None
            return R()

    result = run_cognitive_layer2(ObjProvider(), episodes=_tool_episodes(3))
    assert result["llm_principles"] == 1


def test_layer2_accepts_plain_string_provider(stores):
    """Ollama devuelve str plano — Tier 2 debe funcionar igual (bug real
    que dejaba la reflexión muda)."""
    from ipa.agentic.idle_cognition import run_cognitive_layer2

    class StringProvider:
        def generate_chat(self, messages, **kw):
            return json.dumps({"principles": [
                {"pattern": "p", "principle": "principio desde string", "confidence": 0.6},
            ]})

    result = run_cognitive_layer2(StringProvider(), episodes=_tool_episodes(3))
    assert result["llm_principles"] == 1
