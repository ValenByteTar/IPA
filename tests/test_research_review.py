"""Tests for the research review queue + auto-research dedup helpers."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from ipa.agent.research_review import (
    ResearchReviewStore,
    find_recent_research,
    ingest_reviewed_doc,
    mark_researched,
    recently_researched,
    review_doc_with_llm,
)


@pytest.fixture()
def store(tmp_path, monkeypatch):
    db = tmp_path / "review.db"
    monkeypatch.setenv("IPA_RESEARCH_REVIEW_DB", str(db))
    s = ResearchReviewStore()
    yield s
    s.close()


def _item(store, url="https://example.com/a", text="contenido sustancial sobre Navier-Stokes"):
    rid = store.enqueue(
        url=url, title="paper sobre navier", text=text,
        reason="quality bajo", query="ecuacion de navier",
        research_request_id="rr:1",
    )
    return rid


def test_enqueue_and_pending(store):
    rid = _item(store)
    items = store.pending(limit=10)
    assert len(items) == 1
    assert items[0]["review_id"] == rid
    assert items[0]["url"] == "https://example.com/a"
    assert items[0]["query"] == "ecuacion de navier"


def test_enqueue_idempotent_per_url_query(store):
    rid1 = _item(store)
    rid2 = _item(store)  # same url + query → same id, no duplicate row
    assert rid1 == rid2
    assert len(store.pending(limit=10)) == 1


def test_mark_promoted_and_discarded(store):
    rid = _item(store)
    store.mark(rid, "promoted", "vale la pena", document_id="doc:1")
    assert store.pending(limit=10) == []
    counts = store.counts()
    assert counts.get("promoted") == 1

    rid2 = _item(store, url="https://example.com/b")
    store.mark(rid2, "discarded", "thin content")
    assert store.counts().get("discarded") == 1


@dataclass
class _Res:
    text: str = ""
    error: str | None = None


class _Provider:
    def __init__(self, text):
        self._text = text
        self.calls = []

    def generate_chat(self, messages, *, max_new_tokens=None, temperature=None):
        self.calls.append(messages)
        return _Res(text=self._text)


def test_review_verdict_promote():
    p = _Provider('{"promote": true, "reason": "paper técnico denso y on-topic"}')
    out = review_doc_with_llm(p, {"query": "navier", "url": "u", "title": "t",
                                  "reason": "r", "text": "x" * 100})
    assert out["promote"] is True
    assert "técnico" in out["reason"] or "tecnico" in out["reason"]
    assert out["error"] is None


def test_review_verdict_reject_and_bad_json():
    p = _Provider('{"promote": false, "reason": "lista de enlaces"}')
    out = review_doc_with_llm(p, {"query": "q", "url": "u", "title": "t",
                                  "reason": "r", "text": "x"})
    assert out["promote"] is False

    bad = _Provider("no json at all")
    out = review_doc_with_llm(bad, {"query": "q", "url": "u", "title": "t",
                                    "reason": "r", "text": "x"})
    assert out["promote"] is False
    assert out["error"] is not None


def test_ingest_reviewed_doc(tmp_path):
    corpus = tmp_path / "corpus"
    landing = tmp_path / "landing"
    landing.mkdir()
    item = {
        "review_id": "review:abc123",
        "url": "https://example.com/navier",
        "title": "Navier-Stokes explicado",
        "text": "La ecuación de Navier-Stokes describe el movimiento de fluidos. " * 40,
    }
    doc_id = ingest_reviewed_doc(corpus, landing, item)
    assert doc_id is not None

    from ipa.storage.document_store import DocumentStore
    store = DocumentStore(corpus / "document_store.db")
    try:
        doc = store.get_document(doc_id)
        assert doc is not None
        assert "Navier-Stokes" in (doc.text or "")
        src = store.get_source(doc_id) or {}
        assert src.get("provenance") == "agent_research"
        assert src.get("source_url") == "https://example.com/navier"
    finally:
        store.close()


def test_dedup_helpers(tmp_path):
    path = tmp_path / "recent.json"
    assert not recently_researched("Ecuación de Navier", path=path)
    mark_researched("Ecuación de Navier", path=path)
    assert recently_researched("ecuacion de navier", path=path)  # normaliza
    assert not recently_researched("otro tema", path=path)


def test_dedup_expires(tmp_path, monkeypatch):
    path = tmp_path / "recent.json"
    mark_researched("tema viejo", path=path)
    # Forzar timestamp viejo
    data = json.loads(path.read_text(encoding="utf-8"))
    for k in data:
        data[k] = "2020-01-01T00:00:00.000000Z"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert not recently_researched("tema viejo", path=path)


# ---------------------------------------------------------------------------
# find_recent_research — matching tolerante a la reformulación del modelo
# ---------------------------------------------------------------------------

def test_find_recent_research_exact(tmp_path):
    path = tmp_path / "recent.json"
    mark_researched("Ecuación de Navier", path=path)
    hit = find_recent_research("ecuacion de navier", path=path)
    assert hit is not None
    assert hit["exact"] is True
    assert hit["age_minutes"] == 0


def test_find_recent_research_matches_reformulation(tmp_path):
    """El modelo reformula la query entre turnos: el match por tokens la agarra."""
    path = tmp_path / "recent.json"
    mark_researched("acuerdo ralentizacion avance IA big techs", path=path)
    hit = find_recent_research(
        "acuerdo ralentizacion avance IA tres grandes tecnologicas", path=path
    )
    assert hit is not None
    assert hit["exact"] is False
    assert hit["query"] == "acuerdo ralentizacion avance ia big techs"


def test_find_recent_research_shorter_subset_matches(tmp_path):
    path = tmp_path / "recent.json"
    mark_researched("acuerdo ralentizacion avance IA big techs", path=path)
    assert find_recent_research("acuerdo IA", path=path) is not None


def test_find_recent_research_rejects_unrelated(tmp_path):
    path = tmp_path / "recent.json"
    mark_researched("ecuacion de navier stokes fluidos", path=path)
    assert find_recent_research("historia de roma republica", path=path) is None


def test_find_recent_research_stopword_query_needs_exact(tmp_path):
    """Sin tokens de contenido no hay match difuso (evita falsos positivos)."""
    path = tmp_path / "recent.json"
    mark_researched("acuerdo ralentizacion avance ia", path=path)
    assert find_recent_research("dame toda la informacion", path=path) is None


def test_find_recent_research_respects_window(tmp_path):
    path = tmp_path / "recent.json"
    mark_researched("acuerdo ralentizacion avance IA big techs", path=path)
    data = json.loads(path.read_text(encoding="utf-8"))
    for k in data:
        data[k] = "2020-01-01T00:00:00.000000Z"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert find_recent_research("acuerdo ralentizacion avance IA big techs", path=path) is None


def test_find_recent_research_picks_most_recent(tmp_path):
    """Con varias coincidencias gana la corrida más reciente (la que el
    usuario tiene arriba en el chat)."""
    path = tmp_path / "recent.json"
    mark_researched("tema comun", path=path)
    mark_researched("tema comun variante", path=path)
    hit = find_recent_research("tema comun", path=path)
    assert hit is not None
    assert hit["query"] == "tema comun variante"
