"""Tests del store de IPA Push (DEC-011): cola, slot único, colapso, retención."""

from __future__ import annotations

import pytest

from ipa.notifications.store import Store


@pytest.fixture()
def store(tmp_path):
    return Store(db_path=tmp_path / "push.db")


def test_slot_unico_bind_y_revoke(store):
    assert store.slot() is None
    store.bind_slot("fp-1", "Pixel")
    slot = store.slot()
    assert slot["cert_fingerprint"] == "fp-1"
    assert slot["name"] == "Pixel"
    # Re-bind reemplaza (pair --replace) y resetea last_ack_id
    store.touch(last_ack_id=7)
    store.bind_slot("fp-2", "Pixel2")
    assert store.slot()["cert_fingerprint"] == "fp-2"
    assert store.slot()["last_ack_id"] == 0
    # Revoke devuelve el fingerprint y libera
    assert store.revoke_slot() == "fp-2"
    assert store.slot() is None
    assert store.revoke_slot() is None


def test_fanout_sin_slot_luego_paired(store):
    nid = store.enqueue("research.done", "t", "b", {"query": "q"})
    # Sin slot: el aviso queda pending en la cola (nadie puede poller sin
    # cert emparejado) y se entrega al emparejar — backlog intacto.
    assert [e["id"] for e in store.next_batch(0)] == [nid]
    store.bind_slot("fp", "Pixel")
    assert [e["id"] for e in store.next_batch(0)] == [nid]


def test_batch_marca_sent_y_ack_avanza_cursor(store):
    store.bind_slot("fp", "Pixel")
    a = store.enqueue("research.done", "a")
    b = store.enqueue("research.done", "b")
    batch = store.next_batch(0)
    assert {e["id"] for e in batch} == {a, b}
    store.mark_sent([a, b])
    # Sin ack: sigue entregable (reintento)
    assert {e["id"] for e in store.next_batch(0)} == {a, b}
    store.ack([a, b])
    assert store.next_batch(0) == []
    # Cursor: solo eventos nuevos
    c = store.enqueue("research.done", "c")
    assert [e["id"] for e in store.next_batch(0)] == [c]
    # El cursor avanza con ACKS, no con encolados: sigue en b (id 2)
    assert store.slot()["last_ack_id"] == b


def test_poll_con_since_no_reentrega_ackeados(store):
    store.bind_slot("fp", "Pixel")
    a = store.enqueue("research.done", "a")
    store.next_batch(0)
    store.mark_sent([a])
    store.ack([a])
    assert store.next_batch(a) == []
    assert store.next_batch(a - 1) == []  # ya ackeado, no vuelve


def test_colapso_supersede_pendientes_misma_clave(store):
    store.bind_slot("fp", "Pixel")
    first = store.enqueue("research.done", "v1", collapse_key="query-x")
    second = store.enqueue("research.done", "v2", collapse_key="query-x")
    batch = store.next_batch(0)
    assert [e["id"] for e in batch] == [second]
    # La primera quedó superseded, no entregable
    assert all(e["id"] != first for e in batch)
    # Claves distintas no colapsan entre sí
    other = store.enqueue("research.done", "y", collapse_key="query-y")
    assert {e["id"] for e in store.next_batch(0)} == {second, other}


def test_mark_failed_agota_intentos(store):
    store.bind_slot("fp", "Pixel")
    nid = store.enqueue("research.done", "a")
    for _ in range(5):
        store.mark_failed([nid], "boom")
    assert store.next_batch(0) == []  # state=failed, no entregable
    stats = store.stats()
    assert stats["deliveries"].get("failed") == 1


def test_prune_borra_ackeados_viejos_y_huerfanos(store):
    store.bind_slot("fp", "Pixel")
    old = store.enqueue("research.done", "old")
    store.next_batch(0)
    store.mark_sent([old])
    store.ack([old])
    # Viaja 8 días atrás
    import sqlite3

    with sqlite3.connect(store.db_path) as c:
        c.execute(
            "UPDATE notifications SET created_at='2026-09-01T00:00:00Z' WHERE id=?",
            (old,),
        )
        c.execute(
            "UPDATE deliveries SET acked_at='2026-09-01T00:00:00Z'"
            " WHERE notification_id=?",
            (old,),
        )
    fresh = store.enqueue("research.done", "fresh")
    removed = store.prune(retention_days=7)
    assert removed >= 1
    ids = {e["id"] for e in store.next_batch(0)}
    assert old not in ids
    assert fresh in ids


def test_stats_refleja_estado(store):
    store.bind_slot("fp", "Pixel")
    store.enqueue("research.done", "a")
    store.enqueue("research.failed", "b")
    stats = store.stats()
    assert stats["total"] == 2
    assert stats["deliveries"].get("pending") == 2
    assert stats["slot"]["name"] == "Pixel"
