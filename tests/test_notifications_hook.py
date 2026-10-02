"""Tests del hook notify_research de IPA Push (DEC-011).

El contenido del aviso es SOLO metadatos del evento (DEC-002): query, estado,
conteo — nunca contenido de sesiones/episodios.
"""

from __future__ import annotations

import pytest

from ipa.notifications.config import PushConfig
from ipa.notifications.service import PushHub, notify_research
from ipa.notifications.store import Store


@pytest.fixture()
def hub(tmp_path):
    cfg = PushConfig(events={"research": True})
    return PushHub(config=cfg, store=Store(db_path=tmp_path / "push.db"))


def test_done_arma_metadatos(hub):
    progress = {
        "status": "done",
        "query": "rag híbrido",
        "result": {"ingested": 12},
        "finished_at": "2026-09-25T10:00:00Z",
    }
    assert notify_research(progress, hub=hub) is True
    batch = hub.store.next_batch(0)
    assert len(batch) == 1
    event = batch[0]
    assert event["event_type"] == "research.done"
    assert event["title"] == "Investigación lista: rag híbrido"
    assert "12" in event["body"]
    assert event["payload"] == {
        "query": "rag híbrido",
        "status": "done",
        "urls_ingested": 12,
        "finished_at": "2026-09-25T10:00:00Z",
    }
    # Sin contenido de sesión/episodio en el payload
    assert "content" not in event["payload"]
    assert "text" not in event["payload"]


def test_failed_incluye_error_truncado(hub):
    progress = {
        "status": "failed",
        "query": "x",
        "error": "e" * 500,
    }
    assert notify_research(progress, hub=hub) is True
    event = hub.store.next_batch(0)[0]
    assert event["event_type"] == "research.failed"
    assert event["title"] == "Investigación falló: x"
    assert len(event["body"]) == 200


def test_status_intermedio_no_encola(hub):
    assert notify_research({"status": "running", "query": "q"}, hub=hub) is True
    assert notify_research({"status": "idle"}, hub=hub) is True
    assert hub.store.next_batch(0) == []


def test_evento_deshabilitado_por_config_se_consumo(tmp_path):
    cfg = PushConfig(events={"research": False})
    hub = PushHub(config=cfg, store=Store(db_path=tmp_path / "push.db"))
    assert notify_research({"status": "done", "query": "q"}, hub=hub) is True
    assert hub.store.next_batch(0) == []


def test_colapso_por_query_no_spamea(hub):
    for i in range(3):
        notify_research(
            {"status": "done", "query": "misma", "result": {"ingested": i}},
            hub=hub,
        )
    batch = hub.store.next_batch(0)
    assert len(batch) == 1  # solo la última; las previas quedaron superseded
    assert batch[0]["payload"]["urls_ingested"] == 2
