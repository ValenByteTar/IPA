"""Dashboard-facing Tutor driver tests.

Covers the chat state machine that wires the dashboard to TutorSession:
  - topic detection / "ask for a topic" fallback
  - diagnosis + LLM roadmap proposal → inert until human gate
  - approve/reject via typed message AND via button endpoint path
  - research proposal when the corpus can't support a roadmap
  - parallel sessions: independent state per session_id
  - lesson mode after activation
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ipa.agent import AgentCore, AgentMemory  # noqa: E402
from ipa.tutor.tutor_chat import TutorChatDriver  # noqa: E402
from ipa.tutor.tutor_runtime import TutorStore  # noqa: E402


@dataclass
class FakeGenerationResult:
    text: str = ""
    error: str | None = None


class FakeProvider:
    model_id = "fake-tutor-provider"

    def __init__(self, responses=None, error=None):
        self.responses = list(responses or [])
        self.calls = []
        self.error = error

    def generate_chat(self, messages, *, max_new_tokens=None, temperature=None, **kw):
        self.calls.append({"messages": messages})
        if self.error:
            return FakeGenerationResult(text="", error=self.error)
        if self.responses:
            return FakeGenerationResult(text=self.responses.pop(0))
        return FakeGenerationResult(text="respuesta de lección")


def _roadmap_json(*concept_ids: str) -> str:
    return json.dumps({
        "units": [
            {"concept_id": cid, "reason": f"base para {cid}",
             "estimated_effort_minutes": 30,
             "assessment_types": ["explanation"]}
            for cid in concept_ids
        ],
        "assumptions": [],
        "uncertainties": [],
    })


def _hits(*doc_ids: str):
    return [
        {"document_id": d, "text": f"contenido sobre {d}", "source_domain": "example.com"}
        for d in doc_ids
    ]


@pytest.fixture()
def env(tmp_path):
    memory = AgentMemory(store_path=tmp_path / "agent.db")
    core = AgentCore(interface="dashboard", role="tutor", memory=memory)
    store = TutorStore(tmp_path / "tutor.db")
    driver = TutorChatDriver(store=store)
    return core, store, driver


# ---------------------------------------------------------------------------

def test_no_topic_asks_for_one(env):
    core, store, driver = env
    out = driver.handle(core, "s1", "hola", FakeProvider())
    assert "aprender" in out["reply"].lower() or "tema" in out["reply"].lower()
    assert driver.state("s1").phase == "idle"


def test_topic_detection_strips_learner_context(env):
    """'punto de partida N' / 'desde cero' es nivel del alumno, no parte del tema."""
    core, store, driver = env
    assert driver._detect_topic("quiero aprender sobre el Reino Unido, punto de partida 0") == "el Reino Unido"
    assert driver._detect_topic("quiero aprender rust desde cero") == "rust"
    assert driver._detect_topic("quiero aprender transformers") == "transformers"


def test_holds_while_approved_research_in_flight(env):
    """Con una investigación aprobada corriendo, cualquier mensaje recibe una
    espera determinística — el LLM no improvisa pasos que dependen de datos
    que todavía no llegaron."""
    from ipa.tutor.tutor_runtime import TutorSession

    core, store, driver = env
    st = driver.state("s1")
    st.topic = "reino unido"
    st.topic_id = "reino-unido"
    st.goal_id = "goal:reino-unido"
    st.phase = "idle"

    sess = TutorSession(core, store)
    req = sess.create_research_request(
        "reino-unido", "Material de estudio sobre reino unido",
        goal_id="goal:reino-unido")
    sess.approve_research_request(req.request_id, decided_by="Valen")

    provider = FakeProvider()
    out = driver.handle(core, "s1", "dale, hacé todo", provider)
    assert "aguardamos" in out["reply"].lower()
    assert "fuente web" in out["reply"].lower()
    assert provider.calls == []  # determinístico: el LLM no se invoca


def test_resumes_topic_after_research_completed(env):
    """Tras una investigación aprobada+completada, un mensaje sin tema
    ("dale") retoma el topic guardado en vez de preguntar de nuevo."""
    import dataclasses
    from ipa.tutor.tutor_contracts import ResearchStatus
    from ipa.tutor.tutor_runtime import TutorSession

    core, store, driver = env
    st = driver.state("s1")
    st.topic = "reino unido"
    st.topic_id = "reino-unido"
    st.goal_id = "goal:reino-unido"

    sess = TutorSession(core, store)
    req = sess.create_research_request(
        "reino-unido", "Material de estudio sobre reino unido",
        goal_id="goal:reino-unido")

    # Sin completar: "dale" no retoma — pide el tema.
    out = driver.handle(core, "s1", "dale", FakeProvider())
    assert "aprender" in out["reply"].lower() or "tema" in out["reply"].lower()

    approved = sess.approve_research_request(req.request_id, decided_by="Valen")
    from ipa.tutor.tutor_contracts import SourceRef, SourceType
    store.save_research_request(
        dataclasses.replace(
            approved, status=ResearchStatus.COMPLETED,
            job_id="tool_call:test",
            result_source_refs=[SourceRef(
                source_id="web_source:x", source_type=SourceType.ARTIFACT,
                content_hash=None)],
        ))
    provider = FakeProvider(responses=[_roadmap_json("doc:a", "doc:b", "doc:c")])
    out = driver.handle(
        core, "s1", "dale", provider,
        retrieve=lambda q: _hits("doc:a", "doc:b", "doc:c"),
    )
    assert driver.state("s1").phase == "roadmap_proposed"


def test_topic_detection_triggers_diagnosis_and_roadmap(env):
    core, store, driver = env
    provider = FakeProvider(responses=[_roadmap_json("doc:a", "doc:b", "doc:c")])
    out = driver.handle(
        core, "s1", "quiero aprender transformers", provider,
        retrieve=lambda q: _hits("doc:a", "doc:b", "doc:c"),
    )
    st = driver.state("s1")
    assert st.phase == "roadmap_proposed"
    assert st.topic == "transformers"
    assert out["roadmap_proposal"] is not None
    assert len(out["roadmap_proposal"]["units"]) == 3
    # Roadmap persists inert in the store.
    roadmap = store.get_roadmap(out["roadmap_proposal"]["roadmap_id"])
    assert roadmap is not None
    assert roadmap.status.value == "proposed"


def test_debate_reproposes_and_supersedes(env):
    """Un mensaje que no es aprobar/rechazar con roadmap propuesto se trata
    como feedback: el LLM re-propone (v+1) y la versión anterior queda
    superseded — el gate humano sigue aplicando."""
    core, store, driver = env
    provider = FakeProvider(responses=[_roadmap_json("doc:a", "doc:b", "doc:c")])
    out = driver.handle(core, "s1", "quiero aprender RAG", provider,
                        retrieve=lambda q: _hits("doc:a", "doc:b", "doc:c"))
    st = driver.state("s1")
    first_id = st.roadmap_id
    assert st.phase == "roadmap_proposed"

    provider2 = FakeProvider(responses=[_roadmap_json("doc:c", "doc:b", "doc:a")])
    out2 = driver.handle(
        core, "s1", "quiero más profundidad práctica y menos teoría", provider2,
        retrieve=lambda q: _hits("doc:a", "doc:b", "doc:c"),
    )
    assert "revisado" in out2["reply"].lower()
    assert out2["roadmap_proposal"] is not None
    assert st.phase == "roadmap_proposed"
    assert st.roadmap_id != first_id
    # La propuesta anterior queda superseded, preservada
    old = store.get_roadmap(first_id)
    assert old is not None and old.status.value == "superseded"
    new = store.get_roadmap(st.roadmap_id)
    assert new.status.value == "proposed"
    assert new.version == 2
    assert new.previous_roadmap_id == first_id


def test_roadmap_gate_by_typed_approval(env):
    core, store, driver = env
    provider = FakeProvider(responses=[_roadmap_json("doc:a", "doc:b", "doc:c")])
    driver.handle(core, "s1", "quiero aprender RAG", provider,
                  retrieve=lambda q: _hits("doc:a", "doc:b", "doc:c"))
    out = driver.handle(core, "s1", "aprobado", FakeProvider())
    st = driver.state("s1")
    assert st.phase == "active"
    roadmap = store.get_roadmap(st.roadmap_id)
    assert roadmap.status.value == "active"


def test_roadmap_gate_by_button_endpoint(env):
    core, store, driver = env
    provider = FakeProvider(responses=[_roadmap_json("doc:a", "doc:b", "doc:c")])
    out = driver.handle(core, "s1", "quiero aprender RAG", provider,
                        retrieve=lambda q: _hits("doc:a", "doc:b", "doc:c"))
    rid = out["roadmap_proposal"]["roadmap_id"]
    res = driver.decide_roadmap("s1", rid, "approve")
    assert res["ok"] and res["status"] == "active"
    assert driver.state("s1").phase == "active"


def test_roadmap_rejection_returns_to_idle(env):
    core, store, driver = env
    provider = FakeProvider(responses=[_roadmap_json("doc:a", "doc:b", "doc:c")])
    out = driver.handle(core, "s1", "quiero aprender RAG", provider,
                        retrieve=lambda q: _hits("doc:a", "doc:b", "doc:c"))
    rid = out["roadmap_proposal"]["roadmap_id"]
    res = driver.decide_roadmap("s1", rid, "reject")
    assert res["ok"] and res["status"] == "rejected"
    st = driver.state("s1")
    assert st.phase == "idle" and st.roadmap_id is None


def test_insufficient_concepts_proposes_research(env):
    core, store, driver = env
    out = driver.handle(
        core, "s1", "quiero aprender cocina molecular", FakeProvider(),
        retrieve=lambda q: _hits("doc:a"),
    )
    st = driver.state("s1")
    assert st.phase == "research_pending"
    assert out["research_proposal"] is not None
    req = store.get_research_request(out["research_proposal"]["request_id"])
    assert req is not None
    assert req.status.value == "pending_approval"


def test_research_gate_typed_approval_and_reject(env):
    core, store, driver = env
    out = driver.handle(core, "s1", "quiero aprender X", FakeProvider(),
                        retrieve=lambda q: _hits("doc:a"))
    rid = out["research_proposal"]["request_id"]
    res = driver.decide_research("s1", rid, "approve")
    assert res["ok"] and res["status"] == "approved"
    assert store.get_research_request(rid).status.value == "approved"
    # Topic stays known; phase frees for a new turn once material arrives.
    assert driver.state("s1").phase == "idle"

    out2 = driver.handle(core, "s2", "quiero aprender Y", FakeProvider(),
                         retrieve=lambda q: _hits("doc:a"))
    rid2 = out2["research_proposal"]["request_id"]
    res2 = driver.decide_research("s2", rid2, "reject")
    assert res2["ok"] and res2["status"] == "cancelled"
    assert store.get_research_request(rid2).status.value == "cancelled"


def test_parallel_sessions_have_independent_state(env):
    core, store, driver = env
    p1 = FakeProvider(responses=[_roadmap_json("doc:a", "doc:b", "doc:c")])
    driver.handle(core, "s1", "quiero aprender RAG", p1,
                  retrieve=lambda q: _hits("doc:a", "doc:b", "doc:c"))
    out2 = driver.handle(core, "s2", "quiero aprender X", FakeProvider(),
                         retrieve=lambda q: _hits("doc:z"))
    assert driver.state("s1").phase == "roadmap_proposed"
    assert driver.state("s2").phase == "research_pending"
    assert driver.state("s1").roadmap_id != driver.state("s2").pending_request_id


def test_lesson_turn_after_activation(env):
    core, store, driver = env
    provider = FakeProvider(responses=[_roadmap_json("doc:a", "doc:b", "doc:c")])
    driver.handle(core, "s1", "quiero aprender RAG", provider,
                  retrieve=lambda q: _hits("doc:a", "doc:b", "doc:c"))
    driver.handle(core, "s1", "aprobado", FakeProvider())
    out = driver.handle(core, "s1", "explicame embeddings", FakeProvider())
    assert out["reply"] == "respuesta de lección"


class FakeStreamProvider(FakeProvider):
    """FakeProvider con generate_chat_stream: emite la respuesta en chunks."""

    def generate_chat_stream(self, messages, max_new_tokens=None,
                             temperature=None, **kw):
        result = self.generate_chat(messages)
        if result.error:
            yield {"text": "", "error": result.error, "done": True}
            return
        mid = len(result.text) // 2 or 1
        yield {"text": result.text[:mid], "done": False}
        yield {"text": result.text[mid:], "done": False}
        yield {"text": "", "done": True}


def test_lesson_streams_tokens_via_on_token(env):
    core, store, driver = env
    provider = FakeProvider(responses=[_roadmap_json("doc:a", "doc:b", "doc:c")])
    driver.handle(core, "s1", "quiero aprender RAG", provider,
                  retrieve=lambda q: _hits("doc:a", "doc:b", "doc:c"))
    driver.handle(core, "s1", "aprobado", FakeProvider())

    streamed: list[str] = []
    out = driver.handle(
        core, "s1", "explicame embeddings", FakeStreamProvider(),
        on_token=streamed.append,
    )
    assert out["reply"] == "respuesta de lección"
    assert "".join(streamed) == "respuesta de lección"
    assert len(streamed) == 2  # llegó en chunks, no de una


def test_lesson_on_token_fallback_without_stream(env):
    """Provider sin generate_chat_stream: on_token recibe el reply completo."""
    core, store, driver = env
    provider = FakeProvider(responses=[_roadmap_json("doc:a", "doc:b", "doc:c")])
    driver.handle(core, "s1", "quiero aprender RAG", provider,
                  retrieve=lambda q: _hits("doc:a", "doc:b", "doc:c"))
    driver.handle(core, "s1", "aprobado", FakeProvider())

    streamed: list[str] = []
    out = driver.handle(
        core, "s1", "explicame embeddings", FakeProvider(),
        on_token=streamed.append,
    )
    assert out["reply"] == "respuesta de lección"
    assert streamed == ["respuesta de lección"]


def test_roadmap_proposal_requires_provider(env):
    core, store, driver = env
    out = driver.handle(core, "s1", "quiero aprender RAG", None,
                        retrieve=lambda q: _hits("doc:a", "doc:b", "doc:c"))
    assert "roadmap" in out["reply"].lower()
    assert driver.state("s1").phase == "idle"


def test_pending_roadmap_reminder_records_episode(env):
    core, store, driver = env
    provider = FakeProvider(responses=[_roadmap_json("doc:a", "doc:b", "doc:c")])
    driver.handle(core, "s1", "quiero aprender RAG", provider,
                  retrieve=lambda q: _hits("doc:a", "doc:b", "doc:c"))
    out = driver.handle(core, "s1", "y ahora qué?", FakeProvider())
    assert "roadmap" in out["reply"].lower()
    assert driver.state("s1").phase == "roadmap_proposed"


# ── unit progress (visual stepper) ──────────────────────────────────────────

def _activate_roadmap(env, session="s1"):
    """Helper: propose + approve a 3-unit roadmap; returns (driver, store, roadmap_id)."""
    core, store, driver = env
    provider = FakeProvider(responses=[_roadmap_json("doc:a", "doc:b", "doc:c")])
    out = driver.handle(core, session, "quiero aprender RAG", provider,
                        retrieve=lambda q: _hits("doc:a", "doc:b", "doc:c"))
    rid = out["roadmap_proposal"]["roadmap_id"]
    driver.handle(core, session, "aprobado", FakeProvider())
    return core, store, driver, rid


def test_activation_seeds_first_unit_current(env):
    core, store, driver, rid = _activate_roadmap(env)
    assert store.unit_statuses(rid) == {1: "current"}


def test_advance_intent_moves_current_unit(env):
    core, store, driver, rid = _activate_roadmap(env)
    out = driver.handle(core, "s1", "ya entendí, siguiente unidad", FakeProvider())
    statuses = store.unit_statuses(rid)
    assert statuses[1] == "done" and statuses[2] == "current"
    assert "unidad 2" in out["reply"].lower()


def test_normal_lesson_does_not_advance(env):
    core, store, driver, rid = _activate_roadmap(env)
    driver.handle(core, "s1", "explicame embeddings", FakeProvider())
    assert store.unit_statuses(rid) == {1: "current"}


def test_advancing_last_unit_completes_roadmap_progress(env):
    core, store, driver, rid = _activate_roadmap(env)
    driver.handle(core, "s1", "siguiente unidad", FakeProvider())
    out = driver.handle(core, "s1", "siguiente unidad", FakeProvider())
    statuses = store.unit_statuses(rid)
    assert statuses[2] == "done" and statuses[3] == "current"
    out = driver.handle(core, "s1", "avancemos", FakeProvider())
    statuses = store.unit_statuses(rid)
    assert all(s == "done" for s in statuses.values())
    assert "final del roadmap" in out["reply"].lower()


def test_lesson_prompt_includes_unit_progress(env):
    core, store, driver, rid = _activate_roadmap(env)
    driver.handle(core, "s1", "siguiente unidad", FakeProvider())
    provider = FakeProvider()
    driver.handle(core, "s1", "explicame embeddings", provider)
    system = provider.calls[0]["messages"][0]["content"]
    assert "unidad 2 de 3" in system
    assert "ya enseñadas: 1" in system
    assert "no repitas" in system.lower()


def test_advance_generates_unit_summary(env):
    core, store, driver, rid = _activate_roadmap(env)
    # 1ª respuesta → la lección; 2ª → el resumen de la unidad completada.
    provider = FakeProvider(responses=[
        "La lección arranca por tokens.",
        "Resumen: explicamos qué es un token y la predicción uno-a-uno.",
    ])
    driver.handle(core, "s1", "ya entendí, siguiente unidad", provider)
    summaries = store.list_unit_summaries(rid)
    assert len(summaries) == 1
    assert summaries[0]["unit_order"] == 1
    assert "token" in summaries[0]["summary"]
    # El resumen llega al índice de memoria (indexer sync).
    from ipa.agent.memory_store import MemoryIndexer, MemoryStore
    mem = MemoryStore()
    try:
        MemoryIndexer(mem)._sync_tutor(store)
        hits = [i for i in mem.recall("token predicción") if i.kind == "lesson_unit"]
        assert hits and "unidad 1" in hits[0].text
    finally:
        mem.close()
