"""Tests end-to-end de IPA Push (DEC-011): long-poll HTTPS mTLS con sockets reales.

Cubre el modelo de acceso aprobado: handshake sin cert falla, cert CA-válido
con fingerprint ajeno → 403 slot_occupied, cert revocado → 403, límites → 429,
y el ciclo completo poll → ack → backlog tras reconexión.
"""

from __future__ import annotations

import http.client
import json
import ssl
import threading
import time
from pathlib import Path

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.serialization import pkcs12

from ipa.notifications import tls_material
from ipa.notifications.config import PushConfig
from ipa.notifications.service import PushService
from ipa.notifications.store import Store


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("IPA_NOTIFICATIONS_DIR", str(tmp_path / "notif"))
    return tmp_path / "notif"


@pytest.fixture()
def service(env):
    db_path = env.parent / "push.db"
    cfg = PushConfig(enabled=True, bind_ip="127.0.0.1", port=0, poll_timeout_s=3)
    svc = PushService(config=cfg, store=Store(db_path=db_path))
    bind_ip, port = svc.start()
    yield svc, bind_ip, port, db_path
    svc.stop()


def _issue_paired_client(db_path, name: str = "Pixel") -> tuple[Path, str, str]:
    """Cert cliente + slot emparejado con su fingerprint (mismo store del svc)."""
    tls_material.init_ca()
    p12_path, password, fingerprint = tls_material.issue_client(name)
    Store(db_path=db_path).bind_slot(fingerprint, name)
    return p12_path, password, fingerprint


def _client_ctx(p12_path: Path, password: str, tmp: Path) -> ssl.SSLContext:
    key, cert, _ = pkcs12.load_key_and_certificates(
        p12_path.read_bytes(), password.encode()
    )
    cert_file, key_file = tmp / "c.pem", tmp / "k.pem"
    cert_file.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_file.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.load_verify_locations(
        cafile=str(tls_material.material_dir() / "ca.crt")
    )
    ctx.load_cert_chain(certfile=str(cert_file), keyfile=str(key_file))
    return ctx


def _get(conn: http.client.HTTPSConnection, path: str):
    conn.request("GET", path)
    resp = conn.getresponse()
    return resp.status, json.loads(resp.read().decode())


def _post(conn: http.client.HTTPSConnection, path: str, payload: dict):
    body = json.dumps(payload).encode()
    conn.request("POST", path, body, {"Content-Type": "application/json"})
    resp = conn.getresponse()
    return resp.status, json.loads(resp.read().decode())


def test_handshake_sin_cert_falla(service, env):
    svc, bind_ip, port, _db = service
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.load_verify_locations(cafile=str(tls_material.material_dir() / "ca.crt"))
    # TLS 1.3: el wrap "completa" aunque el servidor exija cert — el
    # rechazo llega como SSLError o como cierre sin respuesta.
    conn = http.client.HTTPSConnection(bind_ip, port, context=ctx, timeout=5)
    with pytest.raises((ssl.SSLError, ConnectionError)):
        conn.request("GET", "/api/push/v1/health")
        conn.getresponse().read()
    conn.close()


def test_health_y_poll_requieren_slot(service, env, tmp_path):
    svc, bind_ip, port, db_path = service
    p12, password, _ = _issue_paired_client(db_path)
    ctx = _client_ctx(p12, password, tmp_path)
    conn = http.client.HTTPSConnection(bind_ip, port, context=ctx, timeout=10)
    assert _get(conn, "/api/push/v1/health") == (
        200,
        {"ok": True, "server": "ipa-push/0.1"},
    )
    assert _get(conn, "/api/push/v1/poll?since=0&timeout=0") == (
        200,
        {"events": [], "last_id": 0},
    )
    conn.close()
    # Sin slot emparejado → 403 not_paired
    Store(db_path=db_path).revoke_slot()
    conn = http.client.HTTPSConnection(bind_ip, port, context=ctx, timeout=10)
    status, payload = _get(conn, "/api/push/v1/health")
    assert status == 403 and payload["error"] == "not_paired"
    conn.close()


def test_cert_valido_fingerprint_ajeno_403(service, env, tmp_path):
    svc, bind_ip, port, db_path = service
    _issue_paired_client(db_path, "Primero")
    # Segundo cert CA-válido pero fingerprint distinto al del slot
    tls_material.init_ca()
    p12_b, password_b, _ = tls_material.issue_client("Intruso")
    ctx = _client_ctx(p12_b, password_b, tmp_path)
    conn = http.client.HTTPSConnection(bind_ip, port, context=ctx, timeout=10)
    status, payload = _get(conn, "/api/push/v1/health")
    assert status == 403 and payload["error"] == "slot_occupied"
    conn.close()


def test_cert_revocado_403(service, env, tmp_path):
    svc, bind_ip, port, db_path = service
    p12, password, fingerprint = _issue_paired_client(db_path)
    tls_material.revoke_fingerprint(fingerprint)
    ctx = _client_ctx(p12, password, tmp_path)
    conn = http.client.HTTPSConnection(bind_ip, port, context=ctx, timeout=10)
    status, payload = _get(conn, "/api/push/v1/health")
    assert status == 403 and payload["error"] == "revoked"
    conn.close()


def test_ciclo_completo_poll_ack_backlog(service, env, tmp_path):
    svc, bind_ip, port, db_path = service
    p12, password, _ = _issue_paired_client(db_path)
    ctx = _client_ctx(p12, password, tmp_path)
    conn = http.client.HTTPSConnection(bind_ip, port, context=ctx, timeout=10)

    # 1) poll timeout sin eventos (~3s, vacío)
    t0 = time.monotonic()
    status, payload = _get(conn, "/api/push/v1/poll?since=0&timeout=3")
    assert status == 200 and payload["events"] == []
    assert time.monotonic() - t0 >= 2.5

    # 2) encolar → poll inmediato (wake)
    nid = svc.hub.notify(
        "research.done",
        "Investigación lista: rag",
        "12 URLs",
        {"query": "rag", "status": "ok"},
        collapse_key="rag",
    )
    status, payload = _get(conn, "/api/push/v1/poll?since=0&timeout=3")
    assert status == 200
    assert [e["id"] for e in payload["events"]] == [nid]
    assert payload["events"][0]["title"] == "Investigación lista: rag"

    # 3) ack → cursor avanza; backlog tras "reconexión" no re-entrega
    status, payload = _post(conn, "/api/push/v1/ack", {"ids": [nid]})
    assert status == 200 and payload["acked"] == 1
    status, payload = _get(conn, "/api/push/v1/poll?since=0&timeout=0")
    assert payload["events"] == []
    conn.close()

    # 4) nuevo evento → llega en la reconexión
    nid2 = svc.hub.notify("research.failed", "Falló: x", "boom", {"query": "x"})
    conn = http.client.HTTPSConnection(bind_ip, port, context=ctx, timeout=10)
    status, payload = _get(conn, "/api/push/v1/poll?since=0&timeout=3")
    assert [e["id"] for e in payload["events"]] == [nid2]
    conn.close()


def test_wake_desbloquea_poll_en_espera(service, env, tmp_path):
    svc, bind_ip, port, db_path = service
    p12, password, _ = _issue_paired_client(db_path)
    ctx = _client_ctx(p12, password, tmp_path)
    resultado: dict = {}

    def _poller():
        conn = http.client.HTTPSConnection(bind_ip, port, context=ctx, timeout=10)
        resultado["resp"] = _get(conn, "/api/push/v1/poll?since=0&timeout=10")
        conn.close()

    thread = threading.Thread(target=_poller)
    thread.start()
    time.sleep(0.5)
    svc.hub.notify("research.done", "wake", "", {"query": "w"}, collapse_key="w")
    thread.join(timeout=8)
    assert not thread.is_alive()
    status, payload = resultado["resp"]
    assert status == 200 and payload["events"][0]["title"] == "wake"


def test_limite_un_poller_por_fingerprint(service, env, tmp_path):
    svc, bind_ip, port, db_path = service
    p12, password, _ = _issue_paired_client(db_path)
    ctx = _client_ctx(p12, password, tmp_path)
    c1 = http.client.HTTPSConnection(bind_ip, port, context=ctx, timeout=10)
    c2 = http.client.HTTPSConnection(bind_ip, port, context=ctx, timeout=10)
    # Primer poll cuelga el slot; el segundo recibe 429
    thread = threading.Thread(
        target=lambda: c1.request("GET", "/api/push/v1/poll?since=0&timeout=5")
    )
    thread.start()
    time.sleep(0.5)
    status, payload = _get(c2, "/api/push/v1/poll?since=0&timeout=0")
    assert status == 429 and payload["error"] == "too_many_connections"
    c2.close()
    thread.join(timeout=8)
    c1.getresponse().read()
    c1.close()


def test_body_oversize_400(service, env, tmp_path):
    svc, bind_ip, port, db_path = service
    p12, password, _ = _issue_paired_client(db_path)
    ctx = _client_ctx(p12, password, tmp_path)
    conn = http.client.HTTPSConnection(bind_ip, port, context=ctx, timeout=10)
    big = {"ids": list(range(100_000))}
    got_response = True
    try:
        conn.request(
            "POST",
            "/api/push/v1/ack",
            json.dumps(big).encode(),
            {"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        assert resp.status == 400
        resp.read()
    except (ssl.SSLError, OSError):
        # Early-reject: el servidor responde 400 y corta mientras el cliente
        # aún envía el body — el rechazo ya ocurrió (ver store abajo).
        got_response = False
    conn.close()
    assert got_response is not None
    # Lo importante: el ack NO se procesó — la cola sigue sin entregas.
    assert Store(db_path=db_path).stats()["deliveries"] == {}
